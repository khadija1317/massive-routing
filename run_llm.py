"""run_llm.py - run the small and large model on the frozen val/test samples.

Usage :
    python run_llm.py        # full run: 150 + 150 rows, both models
    python run_llm.py 3      # dry run: first 3 rows per split
"""
import os
import sys
import time

import pandas as pd

from config import LLM_PARAMS
from data import load_massive, get_samples
from llm import chat
from prompt import build_messages, parse

MODEL_KEYS = ["small", "large"]
SPLITS = ["validation", "test"]


def main(limit=None):
    out_path = "cache/llm_results_dryrun.csv" if limit else "cache/llm_results.csv"

    dfs, names = load_massive()
    samples = get_samples(dfs)

    rows = []
    tokens = {k: 0 for k in MODEL_KEYS}   # running prompt+completion totals
    n_fresh = 0                           # real API calls (not cache hits)
    t0 = time.time()

    try:
        for split in SPLITS:
            df = samples[split]
            if limit:
                df = df.head(limit)
            for i, r in enumerate(df.itertuples(index=False), start=1):
                messages = build_messages(r.utt, names)   # same prompt for both models
                for key in MODEL_KEYS:
                    res = chat(key, messages, **LLM_PARAMS)
                    p = parse(res["text"], names)
                    rows.append(dict(
                        split=split, id=r.id, model=key, gold=r.label,
                        pred=p["intent"], confidence=p["confidence"],
                        json_ok=p["json_ok"], label_ok=p["label_ok"],
                        finish_reason=res["finish_reason"],
                        latency_s=res["latency_s"],
                        prompt_tokens=res["prompt_tokens"],
                        completion_tokens=res["completion_tokens"],
                        reasoning_tokens=res["reasoning_tokens"],
                        text=res["text"],
                    ))
                    tokens[key] += res["prompt_tokens"] + res["completion_tokens"]
                    n_fresh += 0 if res["cached"] else 1
                if i % 10 == 0 or i == len(df):
                    print(f"{split:10s} {i:3d}/{len(df)} | fresh calls {n_fresh} | "
                          f"tokens small {tokens['small']:,} large {tokens['large']:,} | "
                          f"{(time.time() - t0) / 60:.1f} min", flush=True)
    finally:
        os.makedirs("cache", exist_ok=True)
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"saved {len(rows)} rows -> {out_path}", flush=True)


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit)