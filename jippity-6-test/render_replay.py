"""Render a play.py CSV as a silent, 160x144, 60 fps MP4 at 4x game speed.

Usage (in the science conda environment):
    python render_reply.py steps_backup.csv
    python render_reply.py steps_backup.csv -o replay.mp4
"""

import argparse
import base64
import csv
import hashlib
import io
import shutil
import subprocess
import tempfile
import time
import zlib
from importlib.metadata import version
from pathlib import Path

from pyboy import PyBoy

from play import FIELDS, ROM, position, save_state


VIDEO_FPS = 60
SPEED = 4
DEFAULT_ROM = (Path(__file__).resolve().parent / ROM).resolve()


def read_events(path, selected_session):
    # A session's compressed initial save state can exceed CSV's default limit.
    csv.field_size_limit(16 * 1024 * 1024)
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError("This CSV cannot be replayed: its replay data is missing")
        events = [
            (reader.line_num, row)
            for row in reader
            if selected_session is None or row["session_id"] == selected_session
        ]
    if not events:
        raise ValueError("No matching recorded session found")
    return events


def render_replay(path, output, rom=DEFAULT_ROM, selected_session=None, scale=1):
    if scale < 1:
        raise ValueError("Scale must be a positive integer")
    if output.suffix.lower() != ".mp4":
        raise ValueError("Output must have an .mp4 extension")
    if output.resolve() in (path.resolve(), rom.resolve()):
        raise ValueError("Output must not overwrite the CSV or ROM")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}; choose another -o path")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required; activate the science conda environment")

    # Read a snapshot so an actively appended recording cannot change our input.
    events = read_events(path, selected_session)
    total_frames = 0
    rom_sha256 = hashlib.sha256(rom.read_bytes()).hexdigest()
    pyboy_version = version("pyboy")
    session_id = None
    expected_frame = 0
    for line, row in events:
        event = row["event"]
        if event == "start":
            session_id = row["session_id"]
            if len(session_id) != 32 or any(c not in "0123456789abcdef" for c in session_id):
                raise ValueError(f"Invalid session ID at CSV line {line}")
            if row["rom_sha256"] != rom_sha256:
                raise ValueError("Replay requires the same ROM used by the recording")
            if row["pyboy_version"] != pyboy_version:
                raise ValueError(f"Replay requires PyBoy {row['pyboy_version']}")
            expected_frame = 0
        elif session_id is None or row["session_id"] != session_id:
            raise ValueError(f"Missing session start at CSV line {line}")
        elif event == "tick":
            frames = int(row["frames"])
            if frames < 1:
                raise ValueError(f"Invalid frame count at CSV line {line}")
            expected_frame += frames
            total_frames += frames
        elif event not in ("press", "release", "screenshot", "end"):
            raise ValueError(f"Unknown event {event!r} at CSV line {line}")
        if int(row["frame"]) != expected_frame:
            raise ValueError(f"Inconsistent frame count at CSV line {line}")
        if event == "end":
            session_id = None
    if total_frames == 0:
        raise ValueError("The selected recording contains no game frames")

    expected_video_frames = (total_frames + SPEED - 1) // SPEED
    print(
        f"Rendering {total_frames:,} game frames to {expected_video_frames:,} video frames "
        f"at {VIDEO_FPS} fps ({160 * scale}x{144 * scale}, {SPEED}x speed, "
        f"{expected_video_frames / VIDEO_FPS / 60:.2f} minutes).",
        flush=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output.stem}-", suffix=".mp4", dir=output.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-f", "rawvideo", "-pixel_format", "rgba", "-video_size", "160x144",
        "-framerate", str(VIDEO_FPS), "-i", "pipe:0", "-an",
        "-filter_threads", "1", "-vf", f"scale=iw*{scale}:ih*{scale}:flags=neighbor",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-threads", "4", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(temporary_path),
    ]
    game = None
    game_frames = 0
    video_frames = 0
    pending_frame = None
    started = last_report = time.monotonic()
    try:
        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
            with encoder.stdin:
                for line, row in events:
                    event = row["event"]
                    if event == "start":
                        if game is not None:
                            game.stop(save=False)
                        game = PyBoy(str(rom), window="null")
                        state = zlib.decompress(base64.b64decode(row["initial_state"], validate=True))
                        game.load_state(io.BytesIO(state))
                        game.set_emulation_speed(0)
                        frame = 0
                    elif event == "press":
                        game.button_press(row["buttons"])
                    elif event == "release":
                        game.button_release(row["buttons"])
                    elif event == "tick":
                        # Match play.py's per-frame rendering for exact state hashes.
                        # Sampling uses the global frame count, not CSV row boundaries.
                        for _ in range(int(row["frames"])):
                            if not game.tick(1):
                                raise RuntimeError(f"Emulation stopped at CSV line {line}")
                            frame += 1
                            game_frames += 1
                            if game_frames % SPEED == 0:
                                encoder.stdin.write(game.screen.raw_buffer)
                                video_frames += 1
                                if video_frames % 600 == 0:
                                    now = time.monotonic()
                                    if now - last_report >= 10:
                                        elapsed = now - started
                                        remaining = elapsed * (total_frames - game_frames) / game_frames
                                        print(
                                            f"{100 * game_frames / total_frames:5.1f}%: "
                                            f"{game_frames:,}/{total_frames:,} game frames, "
                                            f"{video_frames:,} video frames, "
                                            f"about {remaining / 60:.1f} min remaining",
                                            flush=True,
                                        )
                                        last_report = now
                        # Preserve the last partial group even if the session ends.
                        pending_frame = (
                            bytes(game.screen.raw_buffer) if game_frames % SPEED else None
                        )

                    expected_position = tuple(int(row[key]) for key in ("x", "y", "map_id"))
                    if (
                        frame != int(row["frame"])
                        or position(game) != expected_position
                        or hashlib.sha256(save_state(game)).hexdigest() != row["state_sha256"]
                    ):
                        raise ValueError(f"Replay diverged at CSV line {line}, frame {frame}")
                    if event == "end":
                        game.stop(save=False)
                        game = None

                # Backups taken during play need not contain a final "end" event.
                # If fewer than four frames remain, show them for one output frame.
                if pending_frame is not None:
                    encoder.stdin.write(pending_frame)
                    video_frames += 1
            if encoder.wait() != 0:
                raise RuntimeError("ffmpeg failed to encode the replay")
        if video_frames != expected_video_frames:
            raise RuntimeError("Encoded frame count does not match the recording")
        temporary_path.replace(output)
    finally:
        try:
            if game is not None:
                game.stop(save=False)
        finally:
            temporary_path.unlink(missing_ok=True)

    print(
        f"Saved {output} ({video_frames:,} frames, {VIDEO_FPS} fps, {160 * scale}x{144 * scale}); "
        f"verified all {len(events):,} recorded state checkpoints.",
        flush=True,
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path, help="Replayable CSV recorded by play.py")
    parser.add_argument("-o", "--output", type=Path, help="Output MP4 (default: <csv stem>_4x.mp4)")
    parser.add_argument("--rom", type=Path, default=DEFAULT_ROM, help="ROM path (default: play.py's ROM)")
    parser.add_argument("--session", help="Render only this session ID (default: all sessions)")
    parser.add_argument("--scale", type=int, default=1, help="Integer pixel enlargement (default: 1, native 160x144)")
    args = parser.parse_args()
    output = args.output or args.csv.with_name(f"{args.csv.stem}_4x.mp4")
    try:
        render_replay(args.csv, output, args.rom, args.session, args.scale)
    except (OSError, ValueError, RuntimeError, zlib.error) as error:
        parser.exit(1, f"Error: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Render interrupted.\n")


if __name__ == "__main__":
    main()
