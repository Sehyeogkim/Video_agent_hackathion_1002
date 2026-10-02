#!/usr/bin/env python3
"""Download collision and non-collision clips. Rules live in fetch-collisions/."""

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

ROOT = Path(__file__).resolve().parent
KEYWORDS_PATH = ROOT / "fetch-collisions" / "keywords.json"
CLIPS_DIR = ROOT / "clips"


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


def api_json(backend: str, token: str, path: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Authorization": f"Bearer {token}"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(f"{backend}{path}", data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def load_keywords() -> dict:
    return json.loads(KEYWORDS_PATH.read_text())


def phrases(words: list[str]) -> re.Pattern:
    parts = [re.escape(word) for word in words]
    return re.compile("|".join(parts), re.I)


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part for part in parts if part]


def negated(sentence: str, start: int) -> bool:
    """A 'no' / 'not' anywhere before the keyword in the sentence rejects the match."""
    window = sentence[:start]
    return bool(re.search(r"\bno\b|\bnot\b|\bwithout\b|\bnever\b", window, re.I))


def rejected_phrase(sentence: str, keywords: dict) -> bool:
    lowered = sentence.lower()
    return any(phrase in lowered for phrase in keywords["reject_phrases"])


def machine_interaction(text: str, keywords: dict) -> bool:
    """Interaction phrases count only when a person and a machine are both in the caption."""
    if not keywords.get("interaction"):
        return False
    has_person = bool(re.search(r"\b(person|worker|human|pedestrian)\b", text, re.I))
    has_machine = bool(re.search(r"forklift|\brobot\b|\bmachine\b", text, re.I))
    return has_person and has_machine


def has_actor(text: str, keywords: dict) -> bool:
    lowered = text.lower()
    return any(re.search(rf"\b{re.escape(actor)}\b", lowered) for actor in keywords["actors"])


def match_sentence(sentence: str, pattern: re.Pattern, keywords: dict) -> re.Match | None:
    if rejected_phrase(sentence, keywords):
        return None
    found = pattern.search(sentence)
    if not found or negated(sentence, found.start()):
        return None
    return found


def classify(text: str, keywords: dict) -> tuple[int, str]:
    """Return (rank, keyword). Rank 3 is a collision word, 2 is physical contact, 1 is a try-phrase."""
    positive = phrases(keywords["positive"])
    contact = phrases(keywords["contact"])
    trial = phrases(keywords["try"])
    trial_reject = phrases(keywords["try_reject"])
    best = (0, "")
    for sentence in sentences(text):
        found = match_sentence(sentence, positive, keywords)
        if found and has_actor(text, keywords):
            return 3, found.group(0).lower()
        found = match_sentence(sentence, contact, keywords)
        if found and has_actor(text, keywords) and best[0] < 2:
            best = (2, found.group(0).lower())
        if keywords.get("interaction") and best[0] < 2 and machine_interaction(text, keywords):
            found = match_sentence(sentence, phrases(keywords["interaction"]), keywords)
            if found and best[0] < 2:
                best = (2, found.group(0).lower())
        found = match_sentence(sentence, trial, keywords)
        if (
            found
            and has_actor(text, keywords)
            and not trial_reject.search(text)
            and best[0] < 1
        ):
            best = (1, found.group(0).lower())
    return best


def explore_all(backend: str, token: str) -> list[dict]:
    chunks = []
    offset = 0
    while True:
        payload = api_json(
            backend,
            token,
            f"/api/v1/videos/explore?scope=all&limit=100&offset={offset}",
        )
        page = payload.get("chunks") or []
        chunks.extend(page)
        if len(page) < 100:
            return chunks
        offset += 100


def rows_from_chunks(chunks: list[dict], keywords: dict) -> list[dict]:
    rows = []
    for chunk in chunks:
        for segment in chunk.get("timeline") or []:
            source = segment.get("source")
            text = segment.get("reasoning_content") or ""
            if not source or not text:
                continue
            rank, keyword = classify(text, keywords)
            rows.append(
                {
                    "filename": chunk.get("filename") or "",
                    "location": chunk.get("location"),
                    "camera_id": chunk.get("camera_id"),
                    "source": source,
                    "start": segment.get("segment_start_sec"),
                    "end": segment.get("segment_end_sec"),
                    "text": text,
                    "rank": rank,
                    "keyword": keyword,
                }
            )
    return rows


def choose(
    rows: list[dict],
    max_videos: int,
    locations: set[str] | None = None,
    all_segments: bool = False,
) -> tuple[list[dict], list[dict]]:
    if locations:
        rows = [row for row in rows if row["location"] in locations]
    if all_segments:
        positives = [row for row in rows if row["rank"] > 0]
    else:
        by_file: dict[str, list[dict]] = {}
        for row in rows:
            by_file.setdefault(row["filename"], []).append(row)
        positives = []
        for group in by_file.values():
            group.sort(key=lambda row: (-row["rank"], row["start"] or 0))
            if group[0]["rank"] > 0:
                positives.append(group[0])
    positives.sort(key=lambda row: (-row["rank"], row["filename"], row["start"] or 0))
    if max_videos > 0:
        positives = positives[:max_videos]
    used = {row["source"] for row in positives}
    negatives = [
        row for row in rows if row["rank"] == 0 and row["source"] not in used
    ]
    negatives.sort(key=lambda row: (row["filename"], row["start"] or 0))
    negatives = negatives[: len(positives)]
    if len(negatives) < len(positives):
        positives = positives[: len(negatives)]
    return positives, negatives


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def fetch_detections(backend: str, token: str, source: str) -> dict:
    """YOLO sidecar for one segment. A missing sidecar is an empty document, not a failure."""
    query = urllib.parse.urlencode({"source": source})
    request = urllib.request.Request(
        f"{backend}/api/v1/videos/detections?{query}",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return {"source": None, "missing": True, "frames": []}
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        sys.exit(f"Detections failed for {source}: HTTP {exc.code} {detail}")


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
    if probe_duration(temp) > seconds + 0.05:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips-dir", type=Path, default=CLIPS_DIR)
    parser.add_argument("--download-seconds", type=float, default=5.0)
    parser.add_argument("--max-videos", type=int, default=30, help="Cap on positives. 0 means no cap.")
    parser.add_argument("--locations", default="", help="Comma-separated locations. Empty searches the whole archive.")
    parser.add_argument("--all-segments", action="store_true", help="Keep every matching segment, not one per file.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    keywords = load_keywords()
    backend, token = load_backend()
    locations = {part.strip() for part in args.locations.split(",") if part.strip()} or None
    rows = rows_from_chunks(explore_all(backend, token), keywords)
    positives, negatives = choose(rows, args.max_videos, locations, args.all_segments)
    if not positives:
        sys.exit("No segment matched the collision keywords")
    count = min(len(positives), len(negatives))
    if count == 0:
        sys.exit("Positives were found but no negative segments remain")
    positives, negatives = positives[:count], negatives[:count]

    selected = []
    for index, row in enumerate(positives, start=1):
        selected.append({**row, "label": True, "clip": f"yes_{index:03d}.mp4"})
    for index, row in enumerate(negatives, start=1):
        selected.append({**row, "label": False, "clip": f"no_{index:03d}.mp4"})

    for row in selected:
        mark = "yes" if row["label"] else "no "
        print(
            f"{mark} {row['clip']} rank={row['rank']} keyword={row['keyword'] or '-'} "
            f"{row['location']} {row['start']}-{row['end']}s {row['filename'][:70]}"
        )

    if args.dry_run:
        return

    args.clips_dir.mkdir(parents=True, exist_ok=True)
    for child in args.clips_dir.iterdir():
        if child.is_file():
            child.unlink()
    for row in selected:
        dest = args.clips_dir / row["clip"]
        print(f"downloading {row['clip']}")
        download_segment(backend, token, row["source"], dest, args.download_seconds)
        yolo_name = f"{dest.stem}.yolo.json"
        yolo = fetch_detections(backend, token, row["source"])
        (args.clips_dir / yolo_name).write_text(json.dumps(yolo))
        row["yolo"] = yolo_name
        frames = yolo.get("frames") or []
        print(f"  yolo frames={len(frames)} detections={yolo.get('detection_count')}")
    lines = []
    for row in selected:
        lines.append(
            json.dumps(
                {
                    "clip": row["clip"],
                    "label": row["label"],
                    "yolo": row.get("yolo") or f"{Path(row['clip']).stem}.yolo.json",
                    "source": row["source"],
                    "keyword": row["keyword"],
                    "reason": row["text"][:240],
                },
                ensure_ascii=False,
            )
        )
    (args.clips_dir / "sources.jsonl").write_text("\n".join(lines) + "\n")
    print(f"wrote {count} yes and {count} no clips to {args.clips_dir}")


if __name__ == "__main__":
    main()
