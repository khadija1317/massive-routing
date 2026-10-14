from huggingface_hub import hf_hub_download
from datasets import Dataset


import json, os
from config import SEED, N_VAL, N_TEST

SAMPLES_PATH = "samples.json"

def get_samples(dfs, path=SAMPLES_PATH):
    """Return {'validation': df, 'test': df}. Draws once, then always reloads the same ids."""
    if os.path.exists(path):
        with open(path) as f:
            ids = json.load(f)
    else:
        ids = {
            "validation": dfs["validation"].sample(N_VAL, random_state=SEED)["id"].tolist(),
            "test": dfs["test"].sample(N_TEST, random_state=SEED)["id"].tolist(),
        }
        with open(path, "w") as f:
            json.dump(ids, f)
    out = {}
    for split, split_ids in ids.items():
        df = dfs[split]
        assert df["id"].is_unique, f"ids not unique in {split}"
        out[split] = df.set_index("id").loc[split_ids].reset_index()
    return out

REPO = "AmazonScience/massive"

def load_massive():
    """Return ({split: DataFrame[id, utt, label]}, list_of_60_label_names)."""
    out = {}
    for split in ["train", "validation", "test"]:
        path = hf_hub_download(REPO, f"en-US/{split}/0000.parquet",
                               repo_type="dataset", revision="refs/convert/parquet")
        ds = Dataset.from_parquet(path)
        names = ds.features["intent"].names
        df = ds.to_pandas()[["id", "utt", "intent"]]
        df["label"] = df["intent"].map(lambda i: names[i])   # int -> string like "alarm_set"
        out[split] = df[["id", "utt", "label"]]
    return out, names