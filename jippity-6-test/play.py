import argparse
import base64
import csv
import hashlib
import io
import json
import zlib
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

from pyboy import PyBoy


ROM = Path("../PokemonRed.gb")
HISTORY = Path("history")
SAVES = Path("saves")
STATE_SAVE_INTERVAL = 100
FIELDS = [
    "session_id", "event", "frame", "frames", "x", "y", "map_id", "buttons",
    "state_sha256", "initial_state", "rom_sha256", "pyboy_version",
]


def position(game):
    return (
        game.memory[0xD362],  # Pokemon Red: player X coordinate
        game.memory[0xD361],  # Pokemon Red: player Y coordinate
        game.memory[0xD35E],  # Pokemon Red: current map ID
    )


def save_state(game):
    state = io.BytesIO()
    game.save_state(state)
    return state.getvalue()


def screenshot(game, number, directory=HISTORY, latest=Path("latest.png")):
    directory.mkdir(parents=True, exist_ok=True)
    screen = game.screen.image
    screen.save(latest)
    screen.save(directory / f"{number:05d}.png")
    print(f"Screenshot {number}: {latest}", flush=True)


def prepare_log(path):
    if path.exists() and path.stat().st_size:
        with path.open(newline="") as csv_file:
            header = next(csv.reader(csv_file))
        if header == ["x", "y", "map_id", "buttons"]:
            backup = path.with_name(f"steps.legacy-{uuid4().hex}.csv")
            path.rename(backup)
            print(f"Preserved older, non-replayable log at {backup}", flush=True)
        elif header != FIELDS:
            raise ValueError(f"Unrecognized CSV format in {path}")


def record(stream=True, stream_metadata=None, load_state=None):
    HISTORY.mkdir(exist_ok=True)
    path = Path("steps.csv")
    prepare_log(path)
    session_id = uuid4().hex
    rom_sha256 = hashlib.sha256(ROM.read_bytes()).hexdigest()
    pyboy_version = version("pyboy")
    game = PyBoy(str(ROM), window="null")
    frame = 0
    completed_steps = 0
    screenshots = 0
    streamer = None

    try:
        if load_state is not None:
            with load_state.open("rb") as state_file:
                game.load_state(state_file)
            print(f"Loaded state: {load_state}", flush=True)
        if stream:
            from stream_agent_wrapper import CoordinateStreamer

            streamer = CoordinateStreamer(stream_metadata)
        with path.open("a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
            if csv_file.tell() == 0:
                writer.writeheader()

            def log_event(event, button="", frames=0, **metadata):
                state = save_state(game)
                x, y, map_id = position(game)
                writer.writerow({
                    "session_id": session_id,
                    "event": event,
                    "frame": frame,
                    "frames": frames,
                    "x": x,
                    "y": y,
                    "map_id": map_id,
                    "buttons": button,
                    "state_sha256": hashlib.sha256(state).hexdigest(),
                    **metadata,
                })
                csv_file.flush()

            def advance(frames, button=""):
                nonlocal frame
                previous = position(game)
                elapsed = 0
                # Sample every frame while holding the button continuously.
                # A tick row ends at each tile/map change or command boundary.
                for index in range(frames):
                    game.tick(1)
                    frame += 1
                    elapsed += 1
                    current = position(game)
                    if streamer is not None:
                        streamer.record(current)
                    if current != previous or index == frames - 1:
                        log_event("tick", button, elapsed)
                        elapsed = 0
                    previous = current

            def take_screenshot():
                nonlocal screenshots
                screenshots += 1
                screenshot(game, screenshots)
                log_event("screenshot")
                if streamer is not None:
                    streamer.flush()

            initial_state = base64.b64encode(zlib.compress(save_state(game))).decode("ascii")
            log_event(
                "start", initial_state=initial_state, rom_sha256=rom_sha256,
                pyboy_version=pyboy_version,
            )
            if load_state is None:
                advance(60)
            else:
                # Save states retain held buttons, but not queued releases.
                # Log the releases so a resumed session remains replayable.
                for button in ("a", "b", "start", "select", "left", "right", "up", "down"):
                    game.button_release(button)
                    log_event("release", button)
            take_screenshot()
            while True:
                try:
                    steps = json.loads(input())
                except EOFError:
                    break
                for button, frames in steps:
                    button = button or ""
                    if not isinstance(button, str) or type(frames) is not int or frames < 1:
                        raise ValueError("Each step must be [button string, positive integer frames]")
                    if button:
                        game.button_press(button)
                        log_event("press", button)
                    advance(frames, button)
                    if button:
                        game.button_release(button)
                        log_event("release", button)
                    completed_steps += 1
                    if completed_steps % STATE_SAVE_INTERVAL == 0:
                        state_path = SAVES / session_id / f"step_{completed_steps:05d}.state"
                        state_path.parent.mkdir(parents=True, exist_ok=True)
                        state_path.write_bytes(save_state(game))
                        print(f"Saved state at step {completed_steps}: {state_path}", flush=True)
                take_screenshot()
            log_event("end")
    finally:
        try:
            if streamer is not None:
                streamer.close()
        finally:
            game.stop(save=False)


def replay(path, selected_session=None, save_final_state=False):
    # The CSV embeds a compressed initial save state, which can be a large field.
    csv.field_size_limit(16 * 1024 * 1024)
    rom_sha256 = hashlib.sha256(ROM.read_bytes()).hexdigest()
    pyboy_version = version("pyboy")
    game = None
    session_id = None
    found = False
    frame = 0
    screenshots = 0
    directory = None

    def finish_session():
        nonlocal game
        if game is None:
            return
        if save_final_state:
            state_path = SAVES / session_id / "final.state"
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_bytes(save_state(game))
            print(f"Saved final state at frame {frame}: {state_path}", flush=True)
        game.stop(save=False)
        game = None

    try:
        with path.open(newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames != FIELDS:
                raise ValueError("This CSV cannot be replayed: its replay data is missing")
            for row in reader:
                if selected_session is not None and row["session_id"] != selected_session:
                    continue
                event = row["event"]
                if event == "start":
                    if row["rom_sha256"] != rom_sha256:
                        raise ValueError("Replay requires the same PokemonRed.gb ROM")
                    if row["pyboy_version"] != pyboy_version:
                        raise ValueError(f"Replay requires PyBoy {row['pyboy_version']}")
                    finish_session()
                    game = PyBoy(str(ROM), window="null")
                    state = zlib.decompress(base64.b64decode(row["initial_state"], validate=True))
                    game.load_state(io.BytesIO(state))
                    game.set_emulation_speed(0)
                    session_id = row["session_id"]
                    if len(session_id) != 32 or any(c not in "0123456789abcdef" for c in session_id):
                        raise ValueError("Invalid session ID")
                    directory = HISTORY / "replay" / session_id
                    frame = 0
                    screenshots = 0
                    found = True
                elif game is None or row["session_id"] != session_id:
                    raise ValueError(f"Missing session start at CSV line {reader.line_num}")
                elif event == "press":
                    game.button_press(row["buttons"])
                elif event == "release":
                    game.button_release(row["buttons"])
                elif event == "tick":
                    frames = int(row["frames"])
                    if frames < 1:
                        raise ValueError(f"Invalid frame count at CSV line {reader.line_num}")
                    for _ in range(frames):
                        game.tick(1)
                        frame += 1
                elif event == "screenshot":
                    screenshots += 1
                    screenshot(game, screenshots, directory, directory / "latest.png")
                elif event != "end":
                    raise ValueError(f"Unknown event {event!r} at CSV line {reader.line_num}")

                expected_position = tuple(int(row[key]) for key in ("x", "y", "map_id"))
                if (
                    frame != int(row["frame"])
                    or position(game) != expected_position
                    or hashlib.sha256(save_state(game)).hexdigest() != row["state_sha256"]
                ):
                    raise ValueError(f"Replay diverged at CSV line {reader.line_num}, frame {frame}")
                if event == "end":
                    finish_session()
        if not found:
            raise ValueError("No matching recorded session found")
        # Logs captured during play may end without an explicit "end" event.
        finish_session()
    finally:
        if game is not None:
            game.stop(save=False)


def main():
    parser = argparse.ArgumentParser(description="Record or replay frame-exact Pokemon Red sessions")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--replay", type=Path, help="Replay sessions from this CSV without modifying it")
    mode.add_argument("--load-state", type=Path, help="Resume recording from a PyBoy .state file")
    parser.add_argument("--session", help="Replay only this session ID (default: replay all sessions)")
    parser.add_argument(
        "--save-final-state", action="store_true",
        help="Save each replayed session's final state to saves/<session_id>/final.state",
    )
    parser.add_argument("--no-stream", action="store_true", help="Disable live coordinate broadcasting")
    parser.add_argument("--stream-metadata", type=json.loads, default={}, help="Stream metadata as a JSON object")
    args = parser.parse_args()
    if args.session and args.replay is None:
        parser.error("--session requires --replay")
    if args.save_final_state and args.replay is None:
        parser.error("--save-final-state requires --replay")
    if not isinstance(args.stream_metadata, dict):
        parser.error("--stream-metadata must be a JSON object")
    if args.replay is None:
        record(stream=not args.no_stream, stream_metadata=args.stream_metadata, load_state=args.load_state)
    else:
        replay(args.replay, args.session, save_final_state=args.save_final_state)


if __name__ == "__main__":
    main()
