SEED = 42
N_VAL = 150
N_TEST = 150


MODELS = {
    "small": {"id": "openai/gpt-oss-20b", "price_in_per_M": 0.075, "price_out_per_M": 0.30,
              "rpm": 30, "rpd": 1000, "tpm": 8000, "tpd": 200_000},
    "large": {"id": "openai/gpt-oss-120b", "price_in_per_M": 0.15, "price_out_per_M": 0.60,
              "rpm": 30, "rpd": 1000, "tpm": 8000, "tpd": 200_000},
}

LLM_PARAMS = dict(temperature=0, reasoning_effort="low", max_completion_tokens=400)