"""Score the untouched Qwen3.6-27B on eval and test, then trace a few clips.

The fine-tuned LoRA is not on the inference host yet. This scores the same
base model, with thinking disabled, so a later fine-tuned run can be compared
on the same pairs. Probability is the softmax of the true and false token
logprobs. A timeline walks the original 5-second download.
"""

import base64
import json
import math
import os
import subprocess
import tempfile
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("/home/gp3singh/Video_agent_hackathion_1002")
DATA = ROOT / "data_collection_agent"
OUT = ROOT / "finetune"
QUESTION = "Will a person and a robot or forklift collide within the next few seconds?"
MODEL = "Qwen/Qwen3.6-27B"


def api(payload):
    req = urllib.request.Request(
        "https://api.inference.wandb.ai/v1/chat/completions",
        data=json.dumps(payload).encode(),
        method="POST",
        headers={
            "Authorization": f"Bearer {os.environ['WANDB_API_KEY']}",
            "User-Agent": "safeai-collision/1.0",
            "Content-Type": "application/json",
            "OpenAI-Project": f"{os.environ['WANDB_TEAM']}/{os.environ['WANDB_PROJECT']}",
        },
    )
    last = None
    for _ in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            last = exc.read().decode()[:300]
            if exc.code not in (429, 500, 502, 503):
                raise RuntimeError(f"{exc.code} {last}") from exc
        except Exception as exc:
            last = str(exc)
    raise RuntimeError(last or "request failed")


def jpegs_at(video: Path, times: list[float]) -> list[bytes]:
    blobs = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, t in enumerate(times):
            dest = Path(tmp) / f"f_{i:02d}.jpg"
            subprocess.run(
                [
                    "ffmpeg", "-v", "error", "-y",
                    "-i", str(video),
                    "-ss", f"{t:.3f}",
                    "-frames:v", "1",
                    "-vf", "scale=640:-2",
                    "-q:v", "5",
                    str(dest),
                ],
                check=True,
            )
            blobs.append(dest.read_bytes())
    return blobs


def probability(body: dict) -> tuple[float, str]:
    choice = body["choices"][0]
    text = (choice["message"].get("content") or "").strip().lower()
    steps = (choice.get("logprobs") or {}).get("content") or []
    masses = {"true": 0.0, "false": 0.0}
    if steps:
        for item in steps[0].get("top_logprobs") or []:
            key = item["token"].strip().lower()
            if key in masses:
                masses[key] += math.exp(item["logprob"])
    if masses["true"] + masses["false"] > 0:
        p = masses["true"] / (masses["true"] + masses["false"])
    elif text.startswith("true"):
        p = 1.0
    elif text.startswith("false"):
        p = 0.0
    else:
        p = 0.5
    pred = "true" if p >= 0.5 else "false"
    return p, pred


def score_images(blobs: list[bytes]) -> tuple[float, str]:
    parts = [{"type": "text", "text": QUESTION}]
    for blob in blobs:
        parts.append({
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(blob).decode()},
        })
    body = api({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Warehouse camera. Answer with only true or false."},
            {"role": "user", "content": parts},
        ],
        "max_tokens": 4,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 10,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    return probability(body)


def pair_times(clip_name: str) -> list[float]:
    # yes_005_p02.mp4 is the third pair: frames at 0.4s and 0.5s.
    index = int(Path(clip_name).stem.rsplit("_p", 1)[1])
    start = index * 0.2
    return [start, start + 0.1]


def score_pair(split: str, row: dict) -> dict:
    video = DATA / "processed_clips" / split / row["clip"]
    p, pred = score_images(jpegs_at(video, [0.0, 0.1]))
    label = "true" if row["label"] else "false"
    group = Path(row["clip"]).stem.rsplit("_p", 1)[0]
    return {
        "split": split,
        "clip": row["clip"],
        "group": group,
        "label": label,
        "pred": pred,
        "p_collision": round(p, 4),
        "correct": pred == label,
    }


def metrics(rows: list[dict]) -> dict:
    tp = sum(r["label"] == "true" and r["pred"] == "true" for r in rows)
    tn = sum(r["label"] == "false" and r["pred"] == "false" for r in rows)
    fp = sum(r["label"] == "false" and r["pred"] == "true" for r in rows)
    fn = sum(r["label"] == "true" and r["pred"] == "false" for r in rows)
    n = len(rows)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "n": n,
        "accuracy": round((tp + tn) / n, 4) if n else 0.0,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "mean_p_collision_on_true": round(sum(r["p_collision"] for r in rows if r["label"] == "true") / max(1, sum(r["label"] == "true" for r in rows)), 4),
        "mean_p_collision_on_false": round(sum(r["p_collision"] for r in rows if r["label"] == "false") / max(1, sum(r["label"] == "false" for r in rows)), 4),
    }


def main():
    jobs = []
    for split in ("eval", "test"):
        manifest = DATA / "processed_clips" / split / "manifest.jsonl"
        for line in manifest.read_text().splitlines():
            if line.strip():
                jobs.append((split, json.loads(line)))
    rows = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(score_pair, split, row) for split, row in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            rows.append(fut.result())
            if i % 20 == 0 or i == len(futures):
                print(f"scored {i}/{len(futures)}", flush=True)
    rows.sort(key=lambda r: (r["split"], r["clip"]))
    (OUT / "baseline_predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = {split: metrics([r for r in rows if r["split"] == split]) for split in ("eval", "test")}
    both = metrics(rows)
    summary["eval_and_test"] = both
    (OUT / "baseline_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
