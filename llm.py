import hashlib
import json
import os
import time
from collections import defaultdict, deque

import requests
from dotenv import load_dotenv

from config import MODELS

load_dotenv()

API_URL = "https://api.groq.com/openai/v1/chat/completions"
CACHE_DIR = os.path.join("cache", "llm")
LOG_PATH = os.path.join("cache", "calls.jsonl")
EST_OUT = 150      # guessed output tokens per call; replace with the real average after the smoke test
SAFETY = 0.85      # stay at 85% of each per-minute cap
RATE_HEADERS = ["x-ratelimit-remaining-requests", "x-ratelimit-remaining-tokens",
                "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"]

_windows = defaultdict(deque)   # per model: (timestamp, tokens) of calls in the last 60 s


def _key(model_id, messages, params):
    blob = json.dumps({"model": model_id, "messages": messages, "params": params},
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def _wait_for_budget(model_key, est_tokens):
    """Sleep until the last 60 s of usage leaves room for one more call."""
    cfg, win = MODELS[model_key], _windows[model_key]
    while win:
        now = time.time()
        while win and now - win[0][0] >= 60:
            win.popleft()
        if not win:
            break
        used = sum(t for _, t in win)
        if used + est_tokens <= cfg["tpm"] * SAFETY and len(win) < cfg["rpm"] * SAFETY:
            break
        time.sleep(max(win[0][0] + 60 - now, 0.2))


def chat(model_key, messages, **params):
    """Call 'small' or 'large'. Returns a dict with text, latency, token counts, cached flag."""
    cfg = MODELS[model_key]
    model_id = cfg["id"]
    key = _key(model_id, messages, params)
    path = os.path.join(CACHE_DIR, key + ".json")

    # 1. cache
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return {**json.load(f), "cached": True}

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not found. Check your .env file.")

    est = len(json.dumps(messages)) / 3.5 + EST_OUT
    body = {"model": model_id, "messages": messages, **params}

    for attempt in range(6):
        _wait_for_budget(model_key, est)                  # 2. pacing (outside the timer)
        t0 = time.perf_counter()
        r = requests.post(API_URL, headers={"Authorization": f"Bearer {api_key}"},
                          json=body, timeout=60)
        latency = time.perf_counter() - t0                # timer wraps the API call only
        if r.status_code == 429:                          # 3. retry
            wait = float(r.headers.get("retry-after", 2 ** attempt)) + 1
            if wait > 120:
                raise RuntimeError(f"429 with retry-after={wait:.0f}s: likely the daily cap. "
                                   "Stop and check the console.")
            time.sleep(wait)
            continue
        if r.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        if not r.ok:
            raise RuntimeError(f"{r.status_code}: {r.text[:300]}")
        break
    else:
        raise RuntimeError("Gave up after 6 attempts")

    data = r.json()
    usage = data.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    choice = data["choices"][0]
    rec = {
        "model": model_id,
        "messages": messages,
        "params": params,
        "text": choice["message"].get("content") or "",
        "finish_reason": choice.get("finish_reason"),
        "latency_s": latency,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": details.get("reasoning_tokens"),
        "headers": {h: r.headers.get(h) for h in RATE_HEADERS},
        "ts": time.time(),
        "raw": data,
    }
    _windows[model_key].append((time.time(), usage.get("total_tokens") or est))

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f)
    with open(LOG_PATH, "a", encoding="utf-8") as f:       # 4. logging
        log = {k: rec[k] for k in ("ts", "model", "latency_s", "prompt_tokens",
                                   "completion_tokens", "reasoning_tokens", "headers")}
        f.write(json.dumps(log) + "\n")
    return {**rec, "cached": False}