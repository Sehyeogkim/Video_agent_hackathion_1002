#!/usr/bin/env python3
"""Turn one downloaded clip into short frame-pair clips.

Reads the first ``--window-sec`` seconds (default 1.0). Samples one frame every
``--frame-gap-sec`` seconds (default 0.1). Joins those frames into non-overlapping
pairs, so 1.0s at 0.1s spacing is 10 frames and 5 clips. Each pair is an mp4.

Single clip:

    python process_clip.py --video clips/yes_001.mp4 --label true --out-dir processed_clips/train

Whole download, split into train, eval, and test:

    python process_clip.py --sources clips/sources.jsonl --processed-dir processed_clips
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import tempfile
from decimal import Decimal
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = DATA_DIR / "processed_clips"
SPLITS = ("train", "eval", "test")


def sample_times(window_sec: float, frame_gap_sec: float) -> list[float]:
    if window_sec <= 0 or frame_gap_sec <= 0:
        sys.exit("--window-sec and --frame-gap-sec must be positive")
    window = Decimal(str(window_sec))
    gap = Decimal(str(frame_gap_sec))
    times = []
    cursor = Decimal("0")
    while cursor < window:
        times.append(float(cursor))
        cursor += gap
    return times


def pair_times(times: list[float]) -> list[tuple[float, float]]:
    """Non-overlapping pairs. A leftover final frame is dropped."""
    return [(times[index], times[index + 1]) for index in range(0, len(times) - 1, 2)]


def extract_frame(video: Path, timestamp: float, dest: Path) -> None:
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-ss",
            f"{timestamp:.6f}",
            "-frames:v",
            "1",
            str(dest),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not dest.exists():
        tail = (result.stderr or "")[-400:]
        sys.exit(f"Could not read frame at {timestamp:.3f}s from {video}\n{tail}")


def join_frames(frames: list[Path], frame_gap_sec: float, dest: Path) -> None:
    """Play each frame for frame_gap_sec seconds."""
    fps = 1.0 / frame_gap_sec
    with tempfile.TemporaryDirectory() as temp_name:
        temp = Path(temp_name)
        for index, frame in enumerate(frames):
            shutil.copy(frame, temp / f"frame_{index}.png")
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                f"{fps:.8f}",
                "-i",
                str(temp / "frame_%d.png"),
                "-frames:v",
                str(len(frames)),
                "-vf",
                "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(dest),
            ],
            capture_output=True,
            text=True,
        )
    if result.returncode != 0:
        tail = (result.stderr or "")[-400:]
        sys.exit(f"Could not join frames into {dest}\n{tail}")


def load_yolo(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {"source": None, "missing": True, "frames": []}
    return json.loads(path.read_text())


def nearest_detection_frames(document: dict, times: list[float]) -> list[dict]:
    frames = document.get("frames") or []
    picked = []
    for timestamp in times:
        if not frames:
            picked.append({"time_sec": timestamp, "detections": []})
            continue
        nearest = min(frames, key=lambda frame: abs(float(frame.get("time_sec") or 0) - timestamp))
        picked.append(
            {
                "time_sec": timestamp,
                "matched_time_sec": nearest.get("time_sec"),
                "frame_index": nearest.get("frame_index"),
                "detections": nearest.get("detections") or [],
            }
        )
    return picked


def write_pair_yolo(document: dict, times: list[float], dest: Path) -> None:
    payload = {
        "source": document.get("source"),
        "video_shape": document.get("video_shape"),
        "fps": document.get("fps"),
        "missing": bool(document.get("missing")),
        "frames": nearest_detection_frames(document, times),
    }
    dest.write_text(json.dumps(payload))


def process_video(
    video: Path,
    out_dir: Path,
    label: bool,
    window_sec: float,
    frame_gap_sec: float,
    yolo_path: Path | None = None,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = pair_times(sample_times(window_sec, frame_gap_sec))
    if not pairs:
        sys.exit(f"Window {window_sec}s with gap {frame_gap_sec}s produced no frame pairs")
    if yolo_path is None:
        yolo_path = video.with_suffix(".yolo.json")
    document = load_yolo(yolo_path)
    written = []
    with tempfile.TemporaryDirectory() as temp_name:
        temp = Path(temp_name)
        for index, (start, end) in enumerate(pairs):
            first = temp / f"{index}_a.png"
            second = temp / f"{index}_b.png"
            extract_frame(video, start, first)
            extract_frame(video, end, second)
            name = f"{video.stem}_p{index:02d}.mp4"
            yolo_name = f"{video.stem}_p{index:02d}.yolo.json"
            dest = out_dir / name
            join_frames([first, second], frame_gap_sec, dest)
            write_pair_yolo(document, [start, end], out_dir / yolo_name)
            written.append({"clip": name, "label": label, "yolo": yolo_name, "times": [start, end]})
    return written


def load_sources(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if "clip" not in row or not isinstance(row.get("label"), bool):
            sys.exit(f"{path}:{line_number} needs clip and a boolean label")
        clip = Path(row["clip"])
        if not clip.is_absolute():
            clip = path.parent / clip
        rows.append({**row, "path": clip})
    if not rows:
        sys.exit(f"{path} has no clips")
    return rows


def assign_splits(rows: list[dict], train_ratio: float, eval_ratio: float, test_ratio: float, seed: int) -> dict[str, str]:
    """Keep every pair from one download in the same split when each class has 3 or more videos."""
    total = train_ratio + eval_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        sys.exit("--train-ratio, --eval-ratio, and --test-ratio must sum to 1")
    by_label: dict[bool, list[dict]] = {True: [], False: []}
    for row in rows:
        by_label[row["label"]].append(row)
    plan: dict[str, str] = {}
    rng = random.Random(seed)
    clip_level = any(len(group) < 3 for group in by_label.values())
    if clip_level:
        return {}
    for group in by_label.values():
        rng.shuffle(group)
        count = len(group)
        n_test = min(max(1, round(count * test_ratio)), count - 2)
        n_eval = min(max(1, round(count * eval_ratio)), count - 1 - n_test)
        n_train = count - n_eval - n_test
        buckets = ["train"] * n_train + ["eval"] * n_eval + ["test"] * n_test
        for row, bucket in zip(group, buckets):
            plan[str(row["path"])] = bucket
    return plan


def manifest_row(row: dict, clip_prefix: str = "") -> dict:
    clip = f"{clip_prefix}{row['clip']}" if clip_prefix else row["clip"]
    record = {"clip": clip, "label": row["label"]}
    if row.get("yolo"):
        record["yolo"] = f"{clip_prefix}{row['yolo']}" if clip_prefix else row["yolo"]
    return record


def write_manifest(path: Path, rows: list[dict], clip_prefix: str = "") -> None:
    lines = [json.dumps(manifest_row(row, clip_prefix)) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def process_sources(args: argparse.Namespace) -> None:
    sources = load_sources(args.sources)
    missing = [str(row["path"]) for row in sources if not row["path"].exists()]
    if missing:
        sys.exit("Missing downloaded clips:\n" + "\n".join(missing))
    plan = assign_splits(sources, args.train_ratio, args.eval_ratio, args.test_ratio, args.seed)
    processed = args.processed_dir
    if processed.exists():
        shutil.rmtree(processed)
    processed.mkdir(parents=True)
    grouped: dict[str, list[dict]] = {split: [] for split in SPLITS}

    if plan:
        for row in sources:
            split = plan[str(row["path"])]
            yolo_path = None
            if row.get("yolo"):
                yolo_path = Path(row["yolo"])
                if not yolo_path.is_absolute():
                    yolo_path = args.sources.parent / yolo_path
            written = process_video(
                row["path"],
                processed / split,
                row["label"],
                args.window_sec,
                args.frame_gap_sec,
                yolo_path,
            )
            grouped[split].extend(written)
    else:
        pending = []
        hold = processed / "_unsplit"
        for row in sources:
            yolo_path = None
            if row.get("yolo"):
                yolo_path = Path(row["yolo"])
                if not yolo_path.is_absolute():
                    yolo_path = args.sources.parent / yolo_path
            written = process_video(
                row["path"], hold, row["label"], args.window_sec, args.frame_gap_sec, yolo_path
            )
            for item in written:
                pending.append({**item, "label": row["label"], "src": hold / item["clip"]})
        rng = random.Random(args.seed)
        for label in (True, False):
            group = [item for item in pending if item["label"] is label]
            rng.shuffle(group)
            count = len(group)
            n_test = min(max(1, round(count * args.test_ratio)), max(0, count - 2)) if count >= 3 else 0
            n_eval = min(max(1, round(count * args.eval_ratio)), max(0, count - 1 - n_test)) if count >= 3 else 0
            if count < 3:
                n_train, n_eval, n_test = count, 0, 0
            else:
                n_train = count - n_eval - n_test
            buckets = ["train"] * n_train + ["eval"] * n_eval + ["test"] * n_test
            for item, split in zip(group, buckets):
                dest_dir = processed / split
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = dest_dir / item["src"].name
                shutil.move(str(item["src"]), dest)
                yolo_src = item["src"].with_name(item["yolo"])
                if yolo_src.exists():
                    shutil.move(str(yolo_src), dest_dir / item["yolo"])
                grouped[split].append({"clip": dest.name, "label": label, "yolo": item.get("yolo")})
        shutil.rmtree(hold, ignore_errors=True)

    combined = []
    for split in SPLITS:
        write_manifest(processed / split / "manifest.jsonl", grouped[split])
        for row in grouped[split]:
            combined.append(
                {
                    "clip": f"{split}/{row['clip']}",
                    "label": row["label"],
                    "yolo": f"{split}/{row['yolo']}" if row.get("yolo") else None,
                }
            )
        yes = sum(1 for row in grouped[split] if row["label"])
        print(f"{split}: {len(grouped[split])} clips ({yes} yes, {len(grouped[split]) - yes} no)")
    write_manifest(processed / "manifest.jsonl", combined)
    print(f"wrote {processed}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, help="One downloaded mp4")
    parser.add_argument("--label", choices=("true", "false"), help="Required with --video. true means an accident follows.")
    parser.add_argument("--out-dir", type=Path, help="Where to write pair clips for --video")
    parser.add_argument("--sources", type=Path, help="sources.jsonl from download_clips.py")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--window-sec", type=float, default=1.0, help="How much of the start of each download to use")
    parser.add_argument("--frame-gap-sec", type=float, default=0.1, help="Time between sampled frames")
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.sources:
        process_sources(args)
        return
    if not args.video or not args.label or not args.out_dir:
        sys.exit("Pass --video, --label, and --out-dir, or pass --sources")
    written = process_video(args.video, args.out_dir, args.label == "true", args.window_sec, args.frame_gap_sec)
    for row in written:
        print(json.dumps({"clip": row["clip"], "label": row["label"], "times": row["times"]}))


if __name__ == "__main__":
    main()
