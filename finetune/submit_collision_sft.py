"""Submit the warehouse collision clips as a W&B Serverless SFT job.

Valen-Preview-0923 is not a Serverless Training base model, and this machine
has no GPU. The job fine-tunes Qwen/Qwen3.6-27B, the catalog vision model, to
answer whether a person and a robot or forklift collide within the next few
seconds. Each example is the two frames of one processed pair.
"""

import base64
import json
import os
import subprocess
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path("/home/gp3singh/Video_agent_hackathion_1002")
TRAIN = ROOT / "data_collection_agent/processed_clips/train"
QUESTION = (
    "Will a person and a robot or forklift collide within the next few seconds?"
)
ENTITY = os.environ["WANDB_TEAM"]
PROJECT = os.environ["WANDB_PROJECT"]
MODEL_NAME = "safeai-collision"
BASE_MODEL = "Qwen/Qwen3.6-27B"
API = "https://api.training.wandb.ai"


def api(method, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        API + path,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {os.environ['WANDB_API_KEY']}",
            "User-Agent": "safeai-collision/1.0",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode()
            return resp.status, json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode()
        raise SystemExit(f"{method} {path} failed: {exc.code} {detail[:500]}") from exc


def frames(mp4: Path) -> list[bytes]:
    with tempfile.TemporaryDirectory() as tmp:
        pattern = str(Path(tmp) / "f_%02d.jpg")
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(mp4),
                "-vf",
                "scale=640:-2",
                "-q:v",
                "5",
                pattern,
            ],
            check=True,
        )
        paths = sorted(Path(tmp).glob("f_*.jpg"))
        if not paths:
            raise RuntimeError(f"no frames extracted from {mp4.name}")
        return [p.read_bytes() for p in paths]


def build_jsonl(dest: Path) -> dict:
    rows = [json.loads(line) for line in TRAIN.joinpath("manifest.jsonl").read_text().splitlines() if line.strip()]
    true_n = false_n = 0
    with dest.open("w") as out:
        for row in rows:
            label = "true" if row["label"] else "false"
            true_n += label == "true"
            false_n += label == "false"
            parts = [{"type": "text", "text": QUESTION}]
            for blob in frames(TRAIN / row["clip"]):
                encoded = base64.b64encode(blob).decode()
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                    }
                )
            record = {
                "messages": [
                    {
                        "role": "system",
                        "content": "Warehouse camera. Answer with only true or false.",
                    },
                    {"role": "user", "content": parts},
                    {"role": "assistant", "content": label},
                ]
            }
            out.write(json.dumps(record) + "\n")
    return {"examples": len(rows), "true": true_n, "false": false_n, "bytes": dest.stat().st_size}


def main():
    dest = Path("/tmp/safeai_collision_train.jsonl")
    stats = build_jsonl(dest)
    print(
        f"jsonl examples={stats['examples']} true={stats['true']} "
        f"false={stats['false']} bytes={stats['bytes']}"
    )

    status, model = api(
        "POST",
        "/v1/preview/models",
        {
            "project": PROJECT,
            "name": MODEL_NAME,
            "base_model": BASE_MODEL,
            "entity": ENTITY,
            "return_existing": True,
        },
    )
    print(f"model status={status} id={model['id']} name={model['name']} base={model['base_model']}")

    import wandb

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        id=model["run_id"],
        name=MODEL_NAME,
        resume="allow",
        job_type="sft-data",
        settings=wandb.Settings(silent=True),
    )
    artifact_name = f"{MODEL_NAME}-sft-data"
    artifact = wandb.Artifact(
        artifact_name,
        type="dataset",
        metadata={"format": "jsonl", "num_trajectories": stats["examples"]},
    )
    artifact.add_file(str(dest), name="train.jsonl")
    logged = run.log_artifact(artifact)
    logged.wait()
    run.finish()
    training_data_url = f"wandb-artifact:///{ENTITY}/{PROJECT}/{artifact_name}:v0"
    print(f"artifact {training_data_url}")

    status, job = api(
        "POST",
        "/v1/preview/sft-training-jobs",
        {
            "model_id": model["id"],
            "training_data_url": training_data_url,
            "config": {
                "batch_size": 1,
                "learning_rate": 5e-5,
                "assistant_turns": "last",
                "metric_logging": {"enabled": True, "target_training_step": 1},
            },
        },
    )
    print(f"job status={status} id={job.get('id')} state={job.get('status')}")
    print(json.dumps({k: job.get(k) for k in ("id", "status", "model_id", "created_at")}))


if __name__ == "__main__":
    main()
