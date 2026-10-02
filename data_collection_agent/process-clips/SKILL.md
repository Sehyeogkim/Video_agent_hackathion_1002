---
name: process-collision-clips
description: >-
  Turn downloaded collision clips into short frame-pair examples with a
  boolean label, split into train, eval, and test. Use after fetch-collisions,
  or when rebuilding processed_clips from clips already on disk.
---

# Process collision clips

`fetch-collisions` writes 5-second mp4 files. This step keeps only the opening window of each file and turns it into frame pairs. The model should see the lead-up, then a label for whether a collision follows. It does not predict the second the collision happens.

## Window

Defaults:

- `--window-sec 1.0` — use the first second of each download
- `--frame-gap-sec 0.1` — take a frame every 0.1 seconds

That is 10 frames. Join them into non-overlapping pairs: (0.0, 0.1), (0.2, 0.3), (0.4, 0.5), (0.6, 0.7), (0.8, 0.9). Five clips, two frames each. A leftover final frame is dropped.

Both values are flags. Changing them rebuilds `processed_clips/` from the downloads. It does not re-download.

## Labels and splits

`label: true` means a collision or contact follows this clip. `label: false` means it does not. Copy the boolean from `sources.jsonl`. Every pair from one download shares that label.

Each download has a YOLO sidecar, `yes_XXX.yolo.json`. For a pair at times `t0` and `t1`, copy the nearest detection frames into `yes_XXX_pNN.yolo.json` next to that pair. Keep `source`, `video_shape`, and `fps` from the sidecar. Boxes stay in pixel coordinates: `label`, `confidence`, `bbox` as `[x1, y1, x2, y2]`.

When a class has at least 3 source videos, keep all pairs from one download in the same split. Otherwise split the pair clips. Defaults: `--train-ratio 0.6 --eval-ratio 0.2 --test-ratio 0.2`.

## Run

```bash
python3 data_collection_agent/process_clip.py \
  --sources data_collection_agent/clips/sources.jsonl \
  --window-sec 1.0 \
  --frame-gap-sec 0.1
```

One file:

```bash
python3 data_collection_agent/process_clip.py \
  --video data_collection_agent/clips/yes_001.mp4 \
  --label true \
  --out-dir data_collection_agent/processed_clips/train
```

## Output

```text
data_collection_agent/processed_clips/
  train/manifest.jsonl
  eval/manifest.jsonl
  test/manifest.jsonl
  manifest.jsonl
```

Each line has `clip`, `label`, and `yolo`. Paths in a split manifest are filenames in that folder. The top manifest uses `train/...` paths.
