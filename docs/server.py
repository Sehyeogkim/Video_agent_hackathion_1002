"""Serve the VastCAM page and score frames through W&B on the same origin.

GitHub Pages cannot call api.inference.wandb.ai from the browser: the POST
response does not include Access-Control-Allow-Origin. This server keeps the
key here and returns only the probability.
"""

import json
import math
import os
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
QUESTION = "Will a person and a robot or forklift collide within the next few seconds?"


def probability(body):
    choice = body["choices"][0]
    text = (choice.get("message") or {}).get("content") or ""
    text = text.strip().lower()
    steps = ((choice.get("logprobs") or {}).get("content")) or []
    mass = {"true": 0.0, "false": 0.0}
    if steps:
        for item in steps[0].get("top_logprobs") or []:
            key = item.get("token", "").strip().lower()
            if key in mass:
                mass[key] += math.exp(item["logprob"])
    if mass["true"] + mass["false"] > 0:
        return mass["true"] / (mass["true"] + mass["false"])
    if text.startswith("true"):
        return 1.0
    if text.startswith("false"):
        return 0.0
    return 0.5


def score(model, images):
    parts = [{"type": "text", "text": QUESTION}]
    for image in images:
        parts.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image}})
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": "Warehouse camera. Answer with only true or false."},
            {"role": "user", "content": parts},
        ],
        "max_tokens": 4,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 10,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        "https://api.inference.wandb.ai/v1/chat/completions",
        data=payload,
        method="POST",
        headers={
            "Authorization": "Bearer " + os.environ["WANDB_API_KEY"],
            "Content-Type": "application/json",
            "OpenAI-Project": os.environ.get("WANDB_TEAM", "vastdata") + "/" + os.environ.get("WANDB_PROJECT", "team-43"),
            "User-Agent": "vastcam-demo/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return probability(json.loads(resp.read().decode()))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode()[:300]
        raise RuntimeError(detail or str(exc.code)) from exc


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/api/score":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            p = score(body.get("model") or "Qwen/Qwen3.6-27B", body.get("images") or [])
        except Exception as exc:
            data = json.dumps({"error": str(exc)[:300]}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        data = json.dumps({"p": round(p, 4)}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8765"))
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
