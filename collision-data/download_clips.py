#!/usr/bin/env python3
"""Download 5-second warehouse clips into collision-data/clips/.

Positives are segments whose caption describes a person contacting a forklift,
or the closest person-versus-forklift conflict in that recording. Negatives are
an equal number of other warehouse segments with a person and a forklift and no
such event. This script only downloads. Frame clips are built by process_clip.py.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
CLIPS_DIR = DATA_DIR / "clips"
LOCATIONS = ("warehouse3",)


def team_config() -> Path:
    configs = sorted(Path("/config").glob("*.config"))
    if len(configs) != 1:
        sys.exit(f"Expected exactly one /config/*.config; found {len(configs)}")
    return configs[0]


def load_backend() -> tuple[str, str]:
    username = password = backend = None
    for line in team_config().read_text().splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key == "USERNAME":
            username = value
        elif key == "PASSWORD":
            password = value
        elif key == "INGRESS_URL":
            backend = value
    if not (username and password and backend):
        sys.exit("Team config is missing USERNAME, PASSWORD, or INGRESS_URL")
    body = json.dumps({"username": username, "password": password}).encode()
    request = urllib.request.Request(
        f"{backend}/api/v1/auth/login",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        token = json.load(response)["access_token"]
    return backend, token


def api_get(backend: str, token: str, path: str) -> dict:
    request = urllib.request.Request(
        f"{backend}{path}",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def explore(backend: str, token: str, location: str) -> list[dict]:
    chunks: list[dict] = []
    offset = 0
    while True:
        payload = api_get(
            backend,
            token,
            f"/api/v1/videos/explore?scope=all&limit=100&offset={offset}&location={urllib.parse.quote(location)}",
        )
        page = payload.get("chunks") or []
        chunks.extend(page)
        if len(page) < 100:
            return chunks
        offset += 100


def event_score(text: str) -> int:
    """3 = stated contact, 2 = person grabs or climbs on the forklift, 1 = approach or flight."""
    lowered = text.lower()
    if re.search(r"collid|makes contact|crash|struck|bumped into", lowered):
        return 3
    if re.search(r"places (their|his|her) hands|grabs the side|climb", lowered):
        return 2
    if re.search(r"approaching the person|runs away|run away|runs toward", lowered):
        return 1
    return 0


def seed_of(filename: str) -> str:
    match = re.search(r"run_\d+_seed_\d+", filename)
    return match.group(0) if match else filename


def segments_of(chunks: list[dict]) -> list[dict]:
    rows = []
    for chunk in chunks:
        for segment in chunk.get("timeline") or []:
            text = segment.get("reasoning_content") or ""
            source = segment.get("source")
            if not source:
                continue
            rows.append(
                {
                    "filename": chunk.get("filename") or "",
                    "seed": seed_of(chunk.get("filename") or ""),
                    "camera_id": chunk.get("camera_id"),
                    "location": chunk.get("location"),
                    "source": source,
                    "start": segment.get("segment_start_sec"),
                    "end": segment.get("segment_end_sec"),
                    "text": text,
                    "score": event_score(text),
                }
            )
    return rows


def choose(rows: list[dict], max_videos: int) -> tuple[list[dict], list[dict]]:
    """One positive and one negative segment per recording, up to max_videos each."""
    by_seed: dict[str, list[dict]] = {}
    for row in rows:
        by_seed.setdefault(row["seed"], []).append(row)

    positives = []
    for seed, group in sorted(by_seed.items()):
        ranked = sorted(group, key=lambda row: (-row["score"], row["filename"], row["start"] or 0))
        best = ranked[0]
        # Only a stated contact or collision is a positive. Climbs and near-misses stay out.
        if best["score"] < 3:
            continue
        positives.append(best)
    positives = positives[:max_videos]
    used_sources = {row["source"] for row in positives}

    negatives = []
    for positive in positives:
        candidates = [
            row
            for row in by_seed.get(positive["seed"], [])
            if row["source"] not in used_sources
            and row["score"] == 0
            and "person" in row["text"].lower()
            and "forklift" in row["text"].lower()
            and not re.search(r"toward|walk|approach|climb|runs|contact|grab", row["text"], re.I)
        ]
        candidates.sort(
            key=lambda row: (
                0 if re.search(r"stationary|stands still|remains still", row["text"], re.I) else 1,
                row["filename"],
                row["start"] or 0,
            )
        )
        if not candidates:
            continue
        picked = candidates[0]
        used_sources.add(picked["source"])
        negatives.append(picked)
    return positives, negatives


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def download_segment(backend: str, token: str, source: str, dest: Path, seconds: float) -> None:
    query = urllib.parse.urlencode({"source": source, "token": token})
    request = urllib.request.Request(f"{backend}/api/v1/videos/stream?{query}")
    temp = dest.with_suffix(".part.mp4")
    try:
        with urllib.request.urlopen(request, timeout=180) as response, temp.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        sys.exit(f"Download failed for {source}: HTTP {exc.code} {detail}")
    if temp.stat().st_size < 1000:
        sys.exit(f"Download for {source} was too small to be a video")
    duration = probe_duration(temp)
    if duration > seconds + 0.05:
        trimmed = dest.with_suffix(".trim.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(temp), "-t", str(seconds), "-c", "copy", str(trimmed)],
            check=True,
            capture_output=True,
        )
        trimmed.replace(dest)
        temp.unlink(missing_ok=True)
    else:
        temp.replace(dest)


def write_sources(path: Path, rows: list[dict]) -> None:
    lines = []
    for row in rows:
        lines.append(
            json.dumps(
                {
                    "clip": row["clip"],
                    "label": row["label"],
                    "source": row["source"],
                    "seed": row["seed"],
                    "reason": row["text"][:240],
                },
                ensure_ascii=False,
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips-dir", type=Path, default=CLIPS_DIR)
    parser.add_argument("--download-seconds", type=float, default=5.0)
    parser.add_argument("--max-videos", type=int, default=3, help="Max positive videos. Negatives match that count.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    backend, token = load_backend()
    rows = []
    for location in LOCATIONS:
        rows.extend(segments_of(explore(backend, token, location)))
    positives, negatives = choose(rows, args.max_videos)
    if not positives or not negatives:
        sys.exit("No positive/negative warehouse pair was found")
    count = min(len(positives), len(negatives))
    positives, negatives = positives[:count], negatives[:count]

    selected = []
    for index, row in enumerate(positives, start=1):
        selected.append({**row, "label": True, "clip": f"yes_{index:03d}.mp4"})
    for index, row in enumerate(negatives, start=1):
        selected.append({**row, "label": False, "clip": f"no_{index:03d}.mp4"})

    for row in selected:
        print(f"{'yes' if row['label'] else 'no '} {row['clip']} score={row['score']} {row['filename']} {row['start']}-{row['end']}s")

    args.clips_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        return

    for row in selected:
        dest = args.clips_dir / row["clip"]
        print(f"downloading {row['clip']}")
        download_segment(backend, token, row["source"], dest, args.download_seconds)
        print(f"  {probe_duration(dest):.2f}s")
    write_sources(args.clips_dir / "sources.jsonl", selected)
    print(f"wrote {count} yes and {count} no clips to {args.clips_dir}")


if __name__ == "__main__":
    main()
