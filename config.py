SEED = 42
N_VAL = 150
N_TEST = 150


MODELS = {
    "small": {"id": "openai/gpt-oss-20b", "price_in_per_M": None, "price_out_per_M": None,
              "rpm": 30, "rpd": 1000, "tpm": 8000, "tpd": 200_000},
    "large": {"id": "openai/gpt-oss-120b", "price_in_per_M": None, "price_out_per_M": None,
              "rpm": 30, "rpd": 1000, "tpm": 8000, "tpd": 200_000},
}