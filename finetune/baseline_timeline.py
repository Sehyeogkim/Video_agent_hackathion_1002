"""Walk three promising downloads and plot collision probability over time."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baseline_eval import DATA, OUT, jpegs_at, score_images

CLIPS = [
    ("eval", "yes_007", "approaches a forklift"),
    ("eval", "yes_012", "approaches a forklift"),
    ("test", "yes_020", "toward a forklift"),
]
TIMES = [round(i * 0.4, 1) for i in range(12)]  # 0.0 .. 4.4s inside the 5s download


def trace(group: str) -> list[dict]:
    video = DATA / "clips" / f"{group}.mp4"
    points = []
    for t in TIMES:
        blobs = jpegs_at(video, [t, round(t + 0.1, 1)])
        p, pred = score_images(blobs)
        points.append({"t": t, "p_collision": round(p, 4), "pred": pred, "frame": blobs[0]})
        print(f"{group} t={t:.1f} p={p:.3f} {pred}", flush=True)
    return points


def draw(traces: list[tuple[str, str, str, list[dict]]]) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })
    fig = plt.figure(figsize=(12.2, 11.4))
    outer = fig.add_gridspec(len(traces), 1, hspace=0.42, top=0.93, bottom=0.05)
    for row, (split, group, phrase, points) in enumerate(traces):
        inner = outer[row].subgridspec(2, 1, height_ratios=[1.15, 1.0], hspace=0.28)
        frames = inner[0].subgridspec(1, 6, wspace=0.05)
        show = points[::2]
        for i, point in enumerate(show):
            ax = fig.add_subplot(frames[0, i])
            ax.imshow(plt.imread(_write_frame(group, point)))
            ax.set_title(f"{point['t']:.1f}s", fontsize=9, color="#333333", pad=2)
            ax.axis("off")
        ax = fig.add_subplot(inner[1])
        ts = [p["t"] for p in points]
        ps = [p["p_collision"] for p in points]
        ax.plot(ts, ps, color="#1f4e79", linewidth=2, marker="o", markersize=4)
        ax.axhline(0.5, color="#b85c38", linewidth=1, linestyle="--")
        ax.set_xlim(-0.05, 4.6)
        ax.set_ylim(0, 1)
        ax.set_ylabel("P(collision)")
        if row == len(traces) - 1:
            ax.set_xlabel("Seconds into the clip")
        ax.grid(True, alpha=0.25)
        ax.set_title(
            f"{split}  {group}  ·  {phrase}  ·  label true",
            loc="left", fontsize=11, pad=8,
        )
    fig.suptitle(
        "Base Qwen3.6-27B  ·  collision probability as the clip plays",
        fontsize=15,
    )
    dest = OUT / "baseline-timelines.png"
    fig.savefig(dest, dpi=140, bbox_inches="tight")
    print(dest, dest.stat().st_size)


def _write_frame(group: str, point: dict) -> Path:
    folder = OUT / "demo_frames" / group
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"t{point['t']:.1f}.jpg"
    path.write_bytes(point["frame"])
    return path


def main():
    traces = []
    serial = []
    for split, group, phrase in CLIPS:
        points = trace(group)
        traces.append((split, group, phrase, points))
        for point in points:
            serial.append({
                "split": split,
                "group": group,
                "phrase": phrase,
                "t": point["t"],
                "p_collision": point["p_collision"],
                "pred": point["pred"],
            })
    (OUT / "baseline_timelines.jsonl").write_text("".join(json.dumps(r) + "\n" for r in serial))
    draw(traces)


if __name__ == "__main__":
    main()
