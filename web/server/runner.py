"""Server-side training run control: spawn and stop the engine as a subprocess.

Backs the browser's one-click Train / Stop buttons. Keeps a single active run
(enough for a local single-GPU/CPU box); starting while one is live is refused.
Reuses ``local_train.build_local_config`` so the spawned trainer streams telemetry
to this same server and inherits the machine-tuned engine config. Every run is
recorded in the SQLite ``Registry`` for history.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .local_train import ENGINE_DIR, PYTHON, build_local_config
from .registry import Registry

RUNS_DIR = Path(__file__).resolve().parent / ".runs"
ENGINE_RUNS_DIR = ENGINE_DIR / "runs"
WS_ADDRESS = "ws://localhost:8000/broadcast"
STOP_GRACE_SECONDS = 20.0

# Envs per worker, from autotune (peak was 120 envs / 10 workers = 12/worker).
# Deriving num_envs = instances * this keeps the engine's divisibility rules
# satisfied for any worker count: num_envs % workers == 0, and env_batch_size
# (= num_envs) % (num_envs/workers) == 0.
ENVS_PER_WORKER = 12

# Health thresholds. Healthy runs go quiet for long stretches: on CPU the envs
# take minutes to boot (8 instances measured ~8 min to the first batch) and
# pause entirely during each PPO train phase (~5 min observed at 4 instances),
# so anything tighter than this false-alarms on runs that are actually fine.
WARMUP_GRACE_SECONDS = 15 * 60
STALL_GRACE_SECONDS = 10 * 60


def _zombie_children(pid: int) -> int:
    """Count defunct children of the trainer — dead spawn workers linger as
    zombies while pufferlib's main process keeps polling as if all is well."""
    try:
        import psutil

        return sum(
            1 for c in psutil.Process(pid).children() if c.status() == psutil.STATUS_ZOMBIE
        )
    except Exception:
        return 0


def derive_env_layout(instances: int) -> dict[str, int]:
    """Map an 'instances' (worker) count to a valid num_envs/workers/batch trio."""
    num_workers = max(1, instances)
    num_envs = num_workers * ENVS_PER_WORKER
    return {"num_workers": num_workers, "num_envs": num_envs, "env_batch_size": num_envs}


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class RunManager:
    def __init__(self, registry: Optional[Registry] = None) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._run_id: Optional[str] = None
        self._started_at: Optional[float] = None
        self._log_path: Optional[Path] = None
        self._log_file = None
        self._command: Optional[str] = None
        self._params: dict[str, Any] = {}
        self._finished = True  # no active run yet
        self._exp_id: Optional[str] = None
        self._pre_run_dirs: set[str] = set()
        self.registry = registry or Registry()

    # -- state ---------------------------------------------------------------

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _reconcile(self) -> None:
        """Detect a process that exited on its own and record it once."""
        if self._proc is None or self._finished:
            return
        self._discover_exp_id()
        if self._proc.poll() is not None:
            rc = self._proc.returncode
            status = "exited" if rc == 0 else "failed"
            self._finalize(status, rc)

    def _discover_exp_id(self) -> None:
        """Best-effort: find the engine checkpoint dir created for this run."""
        if self._exp_id is not None or not ENGINE_RUNS_DIR.exists():
            return
        new_dirs = [
            p.name
            for p in ENGINE_RUNS_DIR.glob("pokemon-red-*")
            if p.is_dir() and p.name not in self._pre_run_dirs
        ]
        if new_dirs:
            self._exp_id = sorted(new_dirs)[-1]
            if self._run_id:
                self.registry.set_exp_id(self._run_id, self._exp_id)

    def _finalize(self, status: str, returncode: Optional[int]) -> None:
        if self._finished:
            return
        self._finished = True
        if self._run_id:
            self.registry.finish_run(self._run_id, status, _now_iso(), returncode)
        if self._log_file:
            self._log_file.close()
            self._log_file = None

    def status(self) -> dict:
        self._reconcile()
        running = self.is_running()
        returncode = None
        if self._proc is not None and not running:
            returncode = self._proc.returncode
        return {
            "running": running,
            "run_id": self._run_id,
            "command": self._command,
            "params": self._params,
            "exp_id": self._exp_id,
            "pid": self._proc.pid if self._proc else None,
            "started_at": (
                datetime.fromtimestamp(self._started_at, tz=timezone.utc).isoformat()
                if self._started_at
                else None
            ),
            "uptime_seconds": (
                round(time.time() - self._started_at, 1)
                if (running and self._started_at)
                else None
            ),
            "returncode": returncode,
            "log": self.tail_log(40),
        }

    def health(self, last_batch_at: Optional[float]) -> dict:
        """Classify the active run as ok / warming / degraded.

        The failure this catches: an env worker dies (env exception, fork
        accident) and the trainer spin-polls at "running" forever with no
        data — hit twice on 2026-07-05. Signals: zombie children of the
        trainer, or telemetry silence beyond the grace windows.
        ``last_batch_at`` is the server's wall-clock time of the newest
        ingested coordinate batch (from any run — compared against this
        run's start so a previous run's data doesn't count).
        """
        if not self.is_running():
            return {"health": None, "health_reason": None}
        assert self._proc is not None and self._started_at is not None
        zombies = _zombie_children(self._proc.pid)
        if zombies:
            return {
                "health": "degraded",
                "health_reason": (
                    f"{zombies} worker process(es) died — the run won't produce data; "
                    "stop and restart it"
                ),
            }
        now = time.time()
        if last_batch_at is None or last_batch_at < self._started_at:
            if now - self._started_at > WARMUP_GRACE_SECONDS:
                return {
                    "health": "degraded",
                    "health_reason": (
                        f"no telemetry {int((now - self._started_at) / 60)} min after start — "
                        "check the log; the run is likely hung"
                    ),
                }
            return {"health": "warming", "health_reason": None}
        age = now - last_batch_at
        if age > STALL_GRACE_SECONDS:
            return {
                "health": "degraded",
                "health_reason": (
                    f"no data for {int(age / 60)} min — longer than a normal train phase; "
                    "the trainer may be hung"
                ),
            }
        return {"health": "ok", "health_reason": None}

    def tail_log(self, lines: int = 100) -> list[str]:
        if not self._log_path or not self._log_path.exists():
            return []
        with self._log_path.open("r", errors="replace") as f:
            return [ln.rstrip("\n") for ln in deque(f, maxlen=lines)]

    # -- lifecycle -----------------------------------------------------------

    def start(
        self,
        command: str = "train",
        device: Optional[str] = None,
        instances: Optional[int] = None,
        num_envs: Optional[int] = None,
        num_workers: Optional[int] = None,
        env_batch_size: Optional[int] = None,
        total_timesteps: Optional[int] = None,
        wrappers_name: Optional[str] = None,
        reward_name: Optional[str] = None,
        debug: bool = False,
    ) -> dict:
        if self.is_running():
            raise RuntimeError("a training run is already active; stop it first")
        if not PYTHON.exists():
            raise RuntimeError(f"{PYTHON} not found — create the venv and install the engine")

        # The "instances" slider derives a valid env layout; explicit
        # num_envs/num_workers still win if provided directly.
        if instances is not None:
            layout = derive_env_layout(instances)
            num_workers = num_workers or layout["num_workers"]
            num_envs = num_envs or layout["num_envs"]
            env_batch_size = env_batch_size or layout["env_batch_size"]

        train_overrides = {
            "device": device,
            "num_envs": num_envs,
            "num_workers": num_workers,
            "env_batch_size": env_batch_size,
            "total_timesteps": total_timesteps,
        }
        run_config = build_local_config(WS_ADDRESS, train_overrides)

        self._run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = RUNS_DIR / self._run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = run_dir / "train.log"
        self._log_file = self._log_path.open("w")

        # Snapshot existing engine checkpoint dirs so we can spot this run's.
        self._exp_id = None
        self._pre_run_dirs = (
            {p.name for p in ENGINE_RUNS_DIR.glob("pokemon-red-*")}
            if ENGINE_RUNS_DIR.exists()
            else set()
        )

        env = dict(os.environ)
        # macOS: the engine forces the fork start method, unsafe once torch/PyBoy/SDL
        # have initialized the ObjC runtime. See train_macos.sh / README macOS section.
        env["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

        cmd = [str(PYTHON), "-m", "pokemonred_puffer.train", command, "--config", str(run_config)]
        if wrappers_name:
            cmd += ["--wrappers-name", wrappers_name]
        if reward_name:
            cmd += ["--reward-name", reward_name]
        if debug:
            cmd.append("--debug")

        # start_new_session so the trainer + its forked workers share a process
        # group we can signal as a unit on stop().
        self._proc = subprocess.Popen(
            cmd,
            cwd=ENGINE_DIR,
            env=env,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self._started_at = time.time()
        self._command = command
        self._params = {
            k: v
            for k, v in {
                "device": device,
                "instances": instances,
                "num_envs": num_envs,
                "num_workers": num_workers,
                "env_batch_size": env_batch_size,
                "total_timesteps": total_timesteps,
                "wrappers_name": wrappers_name,
                "reward_name": reward_name,
                "debug": debug,
            }.items()
            if v is not None
        }
        self._finished = False
        self.registry.add_run(
            self._run_id, command, self._params, _now_iso(), str(self._log_path)
        )
        return self.status()

    def stop(self) -> dict:
        if not self.is_running():
            self._reconcile()
            return self.status()

        proc = self._proc
        assert proc is not None
        # Graceful: the engine checkpoints on SIGTERM. Signal the whole group.
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass

        deadline = time.time() + STOP_GRACE_SECONDS
        while time.time() < deadline and proc.poll() is None:
            time.sleep(0.3)

        if proc.poll() is None:
            # Force-kill the group if it didn't exit in time.
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=5)

        self._discover_exp_id()
        self._finalize("stopped", proc.returncode)
        return self.status()
