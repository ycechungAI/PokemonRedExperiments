"""Animate a play.py coordinate log at one pixel per global map tile.

Usage (requires ffmpeg, available in the science conda environment):
    python render_coordinates.py
    python render_coordinates.py old/steps3_patch.csv -o coordinates.mp4
    python render_coordinates.py --speed 10 --fps 30 --no-alpha
    python render_coordinates.py --max-game-frames 6000 -o preview.mp4

Defaults to old/steps3_patch.csv and ../v2/map_data.json, relative to this
script. Writes <csv stem>_coordinates_10x.mp4 and a matching *_alpha.mov.
The MP4 uses 8-bit H.264 Main / yuv420p with an opaque black background for
player compatibility. The ProRes 4444 MOV preserves an exact alpha mask for
compositing and uses full-resolution color; H.264 does not encode alpha.
ProRes files are substantially larger than the MP4. Use --no-alpha when only
the MP4 is needed. Both outputs retain the original canvas size and timing.

Blue pixels persist at every visited tile; the white pixel is the player.
Global (x, y) = region.coordinates + local (x, y), with y increasing downward.
The canvas covers the JSON's complete bounds, including its Kanto region:
436 x 444 with origin (0, 0) for the supplied map, without global_map.py's
20-tile padding. No scaling, interpolation, or lines across warps are added.
Samples outside their region's tileSize are skipped (map transitions can
briefly pair a new map ID with the previous map's local coordinates).
During a warp, the new map ID can arrive 33-34 emulator frames before its
destination coordinates. Keep the last stable player position during that
interval, instead of painting the mismatched map/coordinate pair. Ordinary
walking and connected route crossings keep their original frame timestamps.
Use --raw-transitions to include those transient pairs for comparison.

play.py logs every coordinate change at its exact emulator frame. Reading
those positions preserves the complete recorded tile path without PyBoy or
a ROM. Only tick events advance time; menus and battles keep their duration.
The default 60 fps video samples every 10 emulator frames, for 10x playback
assuming a 60 fps emulator. Every visited tile is painted even between video
samples, and a final partial interval gets one frame. Multiple sessions are
concatenated in CSV order. Existing output files are never overwritten.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import tempfile
import time
from contextlib import ExitStack, suppress
from dataclasses import dataclass
from fractions import Fraction
from math import ceil
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR / "old" / "steps3_patch.csv"
DEFAULT_MAP = SCRIPT_DIR.parent / "v2" / "map_data.json"
GAME_FPS = 60
TRAIL = bytes((64, 192, 255, 255))
PLAYER = bytes((255, 255, 255, 255))
EVENTS = {"start", "tick", "press", "release", "screenshot", "end"}
REQUIRED_FIELDS = {"session_id", "event", "frame", "frames", "x", "y", "map_id"}


@dataclass(frozen=True)
class Position:
    frame: int
    x: int
    y: int
    map_id: int


@dataclass
class Recording:
    positions: list[Position]
    total_frames: int
    rows: int


@dataclass
class Layout:
    regions: dict
    origin_x: int
    origin_y: int
    width: int
    height: int

    def pixel(self, position):
        region = self.regions.get(position.map_id)
        if region is None or position.map_id < 0:
            return None
        width, height = region["tileSize"]
        if not (0 <= position.x < width and 0 <= position.y < height):
            return None
        x, y = region["coordinates"]
        return x + position.x - self.origin_x, y + position.y - self.origin_y


def read_layout(path):
    with Path(path).open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict) or not isinstance(data.get("regions"), list):
        raise ValueError("Map JSON must contain a regions array")
    regions = {}
    for region in data["regions"]:
        try:
            map_id = int(region["id"])
            for key in ("coordinates", "tileSize"):
                pair = region[key]
                if not isinstance(pair, list) or len(pair) != 2 or any(type(v) is not int for v in pair):
                    raise ValueError(f"{key} must contain two integers")
            if min(region["tileSize"]) <= 0:
                raise ValueError("tileSize must be positive")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid map region: {region!r}: {error}") from error
        if map_id in regions:
            raise ValueError(f"Duplicate map ID: {map_id}")
        regions[map_id] = region
    if not regions:
        raise ValueError("Map JSON contains no regions")
    x0 = min(r["coordinates"][0] for r in regions.values())
    y0 = min(r["coordinates"][1] for r in regions.values())
    x1 = max(r["coordinates"][0] + r["tileSize"][0] for r in regions.values())
    y1 = max(r["coordinates"][1] + r["tileSize"][1] for r in regions.values())
    return Layout(regions, x0, y0, x1 - x0, y1 - y0)


def read_recording(path, selected_session=None):
    # Embedded initial save states can exceed the csv module's default limit.
    csv.field_size_limit(16 * 1024 * 1024)
    positions = []
    total_frames = rows = expected_frame = 0
    session = None
    seen_sessions = set()
    previous = None
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, strict=True)
        if not REQUIRED_FIELDS.issubset(reader.fieldnames or []):
            raise ValueError("CSV must include session_id, event, frame, frames, x, y, map_id")
        for row in reader:
            line = reader.line_num
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"CSV line {line}: wrong number of columns")
            if selected_session is not None and row["session_id"] != selected_session:
                continue
            rows += 1
            try:
                frame, frames, x, y, map_id = (int(row[k]) for k in ("frame", "frames", "x", "y", "map_id"))
            except ValueError as error:
                raise ValueError(f"CSV line {line}: frame counts and coordinates must be integers") from error
            event = row["event"]
            if event not in EVENTS:
                raise ValueError(f"CSV line {line}: unknown event {event!r}")
            if (event == "tick" and frames < 1) or (event != "tick" and frames != 0):
                raise ValueError(f"CSV line {line}: only ticks may advance time, by a positive frame count")
            if event == "start":
                session = row["session_id"]
                if not session or session in seen_sessions:
                    raise ValueError(f"CSV line {line}: missing or duplicate session ID")
                seen_sessions.add(session)
                expected_frame = 0
            elif session is None or row["session_id"] != session:
                raise ValueError(f"CSV line {line}: missing session start or interleaved sessions")
            if event == "tick":
                expected_frame += frames
                total_frames += frames
            if frame != expected_frame:
                raise ValueError(f"CSV line {line}: expected frame {expected_frame}, got {frame}")
            current = (x, y, map_id)
            if current != previous:
                positions.append(Position(total_frames, x, y, map_id))
                previous = current
            if event == "end":
                session = None
    if not rows:
        raise ValueError("No matching recorded session found")
    if not total_frames:
        raise ValueError("The selected recording contains no game frames")
    return Recording(positions, total_frames, rows)


def stabilize_map_transitions(recording):
    """Remove provisional map/XY pairs, keeping all stable sample timestamps.

    A provisional pair changes maps but retains the previous local tile (or
    its adjacent doorway tile). The next coordinate change corrects XY by
    more than one tile on the same map within 40 emulator frames. In this
    recording all such corrections occur after exactly 33 or 34 frames.
    Route connections wrap local coordinates immediately, so they do not
    match this pattern. Holding the preceding position avoids both a false
    trail pixel and a spurious detour while the map is loading.
    """
    stable = []
    suppressed = []
    positions = recording.positions
    for index, current in enumerate(positions):
        if 0 < index < len(positions) - 1:
            before, after = positions[index - 1], positions[index + 1]
            old_distance = abs(current.x - before.x) + abs(current.y - before.y)
            correction = abs(after.x - current.x) + abs(after.y - current.y)
            if (
                current.map_id != before.map_id
                and after.map_id == current.map_id
                and old_distance <= 1
                and correction > 1
                and 0 < after.frame - current.frame <= 40
            ):
                suppressed.append(current)
                continue
        stable.append(current)
    return Recording(stable, recording.total_frames, recording.rows), suppressed


class CoordinateMap:
    def __init__(self, layout):
        self.layout = layout
        self.rgba = bytearray(layout.width * layout.height * 4)
        self.current = None
        self.visited_tiles = 0
        self.skipped_positions = 0

    def visit(self, position):
        if self.current is not None:
            self.rgba[self.current:self.current + 4] = TRAIL
            self.current = None
        point = self.layout.pixel(position)
        if point is None:
            self.skipped_positions += 1
            return
        x, y = point
        offset = (y * self.layout.width + x) * 4
        if self.rgba[offset + 3] == 0:
            self.visited_tiles += 1
        self.rgba[offset:offset + 4] = PLAYER
        self.current = offset


def video_frames(recording, canvas, speed=Fraction(10), fps=60, max_game_frames=None):
    """Yield a reusable RGBA buffer, sampling at the end of each video interval."""
    if speed <= 0 or fps <= 0:
        raise ValueError("Speed and FPS must be positive")
    if max_game_frames is not None and max_game_frames < 1:
        raise ValueError("max_game_frames must be positive")
    total = min(recording.total_frames, max_game_frames or recording.total_frames)
    step = GAME_FPS * Fraction(str(speed)) / fps
    next_position = 0
    pixels = memoryview(canvas.rgba)
    for index in range(ceil(total / step)):
        cutoff = min((index + 1) * step, total)
        while next_position < len(recording.positions) and recording.positions[next_position].frame <= cutoff:
            canvas.visit(recording.positions[next_position])
            next_position += 1
        yield pixels


def render_coordinates(csv_path, map_path, output, *, alpha_output=None,
                       speed=Fraction(10), fps=60, selected_session=None, max_game_frames=None,
                       raw_transitions=False):
    csv_path, map_path, output = Path(csv_path), Path(map_path), Path(output)
    speed = Fraction(str(speed))
    if speed <= 0 or fps <= 0:
        raise ValueError("Speed and FPS must be positive")
    if max_game_frames is not None and max_game_frames < 1:
        raise ValueError("max_game_frames must be positive")
    targets = [output]
    if output.suffix.lower() != ".mp4":
        raise ValueError("Output must have an .mp4 extension")
    if alpha_output is not None:
        alpha_output = Path(alpha_output)
        if alpha_output.suffix.lower() != ".mov":
            raise ValueError("Transparent output must have a .mov extension")
        targets.append(alpha_output)
    for target in targets:
        if target.resolve() in (csv_path.resolve(), map_path.resolve()):
            raise ValueError("Output must not overwrite the CSV or map JSON")
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Output already exists: {target}; choose another output path")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required; activate the science conda environment")
    layout = read_layout(map_path)
    if layout.width % 2 or layout.height % 2:
        raise ValueError("The map canvas must have even dimensions for compatible H.264 4:2:0 output")
    recording = read_recording(csv_path, selected_session)
    total = min(recording.total_frames, max_game_frames or recording.total_frames)
    suppressed = []
    if not raw_transitions:
        recording, suppressed = stabilize_map_transitions(recording)
    suppressed_count = sum(position.frame <= total for position in suppressed)
    expected_frames = ceil(Fraction(total * fps, GAME_FPS) / speed)
    canvas = CoordinateMap(layout)
    print(
        f"Rendering {total:,} game frames as {expected_frames:,} video frames at {fps} fps, "
        f"{float(speed):g}x speed ({expected_frames / fps / 60:.2f} minutes).\n"
        f"Canvas: {layout.width}x{layout.height}; 1 pixel = 1 tile; "
        f"global origin ({layout.origin_x}, {layout.origin_y}).",
        flush=True,
    )
    with ExitStack() as stack:
        temporary_paths = []
        for target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            directory = stack.enter_context(tempfile.TemporaryDirectory(prefix=f".{target.stem}-", dir=target.parent))
            temporary_paths.append(Path(directory) / target.name)
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-f", "rawvideo", "-pixel_format", "rgba",
            "-video_size", f"{layout.width}x{layout.height}", "-framerate", str(fps),
            "-i", "pipe:0", "-filter_threads", "1",
            "-map", "0:v:0", "-an", "-c:v", "libx264", "-preset", "veryfast",
            # Drop alpha explicitly and use broadly supported 8-bit 4:2:0.
            # scale changes the color matrix/range only, not the dimensions.
            "-vf", "format=rgb24,scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,setsar=1",
            "-profile:v", "main", "-crf", "12", "-g", str(fps * 2), "-threads", "4",
            "-pix_fmt", "yuv420p", "-color_range", "tv", "-color_primaries", "bt709",
            "-color_trc", "bt709", "-colorspace", "bt709", "-movflags", "+faststart+write_colr",
            str(temporary_paths[0]),
        ]
        if alpha_output is not None:
            command += [
                "-filter_threads", "1", "-map", "0:v:0", "-an",
                "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuva444p10le,setsar=1",
                "-c:v", "prores_ks", "-profile:v", "4", "-tag:v", "ap4h",
                "-qscale:v", "4", "-alpha_bits", "16", "-pix_fmt", "yuva444p10le", "-threads", "4",
                "-color_range", "tv", "-color_primaries", "bt709", "-color_trc", "bt709",
                "-colorspace", "bt709", "-movflags", "+faststart+write_colr",
                str(temporary_paths[1]),
            ]
        encoder_log = stack.enter_context(tempfile.TemporaryFile())
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=encoder_log)
        started = last_report = time.monotonic()
        written = 0
        try:
            try:
                for pixels in video_frames(recording, canvas, speed, fps, max_game_frames):
                    encoder.stdin.write(pixels)
                    written += 1
                    if written % 600 == 0:
                        now = time.monotonic()
                        if now - last_report >= 10:
                            remaining = (now - started) * (expected_frames - written) / written
                            print(
                                f"{100 * written / expected_frames:5.1f}%: {written:,}/{expected_frames:,} frames; "
                                f"{canvas.visited_tiles:,} visited tiles; about {remaining / 60:.1f} min remaining",
                                flush=True,
                            )
                            last_report = now
            except BrokenPipeError:
                pass  # Report ffmpeg's actual diagnostic below.
            with suppress(BrokenPipeError):
                encoder.stdin.close()
            returncode = encoder.wait()
            if returncode != 0 or written != expected_frames:
                encoder_log.seek(0)
                detail = encoder_log.read().decode(errors="replace")[-8000:].strip()
                raise RuntimeError(f"ffmpeg failed ({written}/{expected_frames} frames): {detail}")
        finally:
            if encoder.poll() is None:
                encoder.terminate()
                try:
                    encoder.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    encoder.kill()
                    encoder.wait()
            with suppress(BrokenPipeError):
                encoder.stdin.close()
        for temporary, target in zip(temporary_paths, targets):
            # Atomic publication without overwriting an existing output.
            os.link(temporary, target)
    print(f"Saved {output}", flush=True)
    if alpha_output is not None:
        print(f"Saved {alpha_output} (ProRes 4444 with alpha transparency)", flush=True)
    print(
        f"{canvas.visited_tiles:,} visited global tiles from {recording.rows:,} CSV rows; "
        f"suppressed {suppressed_count:,} transient map/coordinate pairs; "
        f"skipped {canvas.skipped_positions:,} unmapped/out-of-region coordinate changes.",
        flush=True,
    )
    return output


def positive_speed(text):
    try:
        value = Fraction(text)
    except (ValueError, ZeroDivisionError) as error:
        raise argparse.ArgumentTypeError("Speed must be a positive number") from error
    if value <= 0:
        raise argparse.ArgumentTypeError("Speed must be positive")
    return value


def positive_int(text):
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Must be a positive integer") from error
    if value < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="?", type=Path, default=DEFAULT_CSV, help="Recorded CSV (default: old/steps3_patch.csv)")
    parser.add_argument("--map-data", type=Path, default=DEFAULT_MAP, help="Map JSON (default: ../v2/map_data.json)")
    parser.add_argument("-o", "--output", type=Path, help="MP4 path (default: <csv stem>_coordinates_<speed>x.mp4)")
    alpha = parser.add_mutually_exclusive_group()
    alpha.add_argument("--alpha-output", type=Path, help="ProRes 4444 MOV path (default: <output stem>_alpha.mov; larger than MP4)")
    #alpha.add_argument("--no-alpha", action="store_true", help="Only write the MP4")
    parser.add_argument("--speed", type=positive_speed, default=Fraction(10), help="Playback speed relative to a 60 fps emulator (default: 10)")
    parser.add_argument("--fps", type=positive_int, default=60, help="Output frames per second (default: 60)")
    parser.add_argument("--session", help="Only render this session ID")
    parser.add_argument("--raw-transitions", action="store_true", help="Include transient map/coordinate pairs during warps (for comparison)")
    parser.add_argument("--max-game-frames", type=positive_int, help="Render only the first N emulator frames, for a preview")
    args = parser.parse_args()
    speed_label = f"{float(args.speed):g}"
    output = args.output or args.csv.with_name(f"{args.csv.stem}_coordinates_{speed_label}x.mp4")
    alpha_output = None # (args.alpha_output or output.with_name(f"{output.stem}_alpha.mov"))
    try:
        render_coordinates(
            args.csv, args.map_data, output, alpha_output=alpha_output,
            speed=args.speed, fps=args.fps, selected_session=args.session, max_game_frames=args.max_game_frames,
            raw_transitions=args.raw_transitions,
        )
    except (OSError, ValueError, RuntimeError, csv.Error) as error:
        parser.exit(1, f"Error: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Render interrupted.\n")


if __name__ == "__main__":
    main()
