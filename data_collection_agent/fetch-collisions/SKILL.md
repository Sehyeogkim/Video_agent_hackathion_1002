---
name: fetch-collisions
description: >-
  Find archive segments where a person, robot, or machine collides with or
  makes contact with anything, and download those 5-second clips. Use when
  collecting collision clips, searching for crashes, impacts, or contact
  between humans, robots, forklifts, and vehicles.
---

# Fetch collision clips

Scan every indexed segment. Download a 5-second clip when a person, robot, or machine collides with or physically contacts something. Write negatives from the remaining segments, one for each positive.

Read [keywords.json](keywords.json) before searching. That file is the keyword list.

## What counts

A segment is positive when one sentence matches `positive` or `contact` in `keywords.json`, the sentence also mentions an `actors` word, and the match is not negated.

Also try the `try` phrases. Keep a `try` match only when the caption does not contain a `try_reject` phrase. "The vehicle yields and the cyclist passes safely" is not a collision.

Reject a match when `no`, `not`, `without`, or `never` appears earlier in the same sentence, or when the sentence contains a `reject_phrases` entry. "No accidents" and "contact information" and "speed bump" are not collisions. A `try` phrase is also rejected when `try_reject` appears anywhere in the caption.

By default, one segment per parent file. If several segments match, keep the strongest one (`positive` before `contact` before `interaction` before `try`). `--all-segments` keeps every match.

`interaction` phrases count only when the caption has a person and a forklift, robot, or machine. Use them for the warehouse set, where a collision is usually described as someone approaching or fleeing a machine.

## Negatives

From segments that did not match, take the same number of clips. These are `label: false`. `--locations warehouse3,indoor` keeps both classes inside the warehouse footage.

## Run

From the repo root:

```bash
python3 data_collection_agent/fetch_collisions.py
```

`--max-videos` caps positives (default 30). `--download-seconds` defaults to 5. `--dry-run` prints the choice and does not download.

Output:

- `data_collection_agent/clips/yes_XXX.mp4` and `no_XXX.mp4`
- `data_collection_agent/clips/yes_XXX.yolo.json` and `no_XXX.yolo.json`, the YOLO sidecar from `GET /api/v1/videos/detections` (`frames[].detections[]` with `label`, `confidence`, `bbox`). A 404 becomes `{"missing": true, "frames": []}`.
- `data_collection_agent/clips/sources.jsonl` with `clip`, `label`, `yolo`, `source`, `keyword`, `reason`

Do not put the token or password in a file or in the log. Login through `/config/*.config` the same way as `retrieval/login`.

Then run the `process-collision-clips` skill on `clips/sources.jsonl`.
