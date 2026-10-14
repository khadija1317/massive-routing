import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score
from data import load_massive, get_samples


def make_model(C):
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
        LogisticRegression(C=C, max_iter=1000),
    )


def metrics(y_true, y_pred):
    present = sorted(set(y_true))  # macro-F1 over classes present in the sample
    return {
        "acc": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=present,
                             average="macro", zero_division=0),
    }


def run_s0():
    dfs, _ = load_massive()
    samples = get_samples(dfs)
    train, full_val = dfs["train"], dfs["validation"]

    # Pick C on the FULL validation split (free, no API calls).
    best_C, best_acc = None, -1
    for C in [1, 10, 100]:
        m = make_model(C).fit(train["utt"], train["label"])
        acc = accuracy_score(full_val["label"], m.predict(full_val["utt"]))
        print(f"C={C}: full-val acc {acc:.3f}")
        if acc > best_acc:
            best_C, best_acc = C, acc
    print("chosen C =", best_C)

    model = make_model(best_C).fit(train["utt"], train["label"])

    rows = []
    for split in ["validation", "test"]:
        df = samples[split]
        proba = model.predict_proba(df["utt"])
        pred = model.classes_[proba.argmax(axis=1)]
        conf = proba.max(axis=1)
        print(split, metrics(df["label"], pred))
        ok = pred == df["label"].values
        print(f"  mean conf when right {conf[ok].mean():.2f} | when wrong {conf[~ok].mean():.2f}")
        rows.append(pd.DataFrame({"split": split, "id": df["id"].values,
                                  "s0_pred": pred, "s0_conf": conf}))

    os.makedirs("cache", exist_ok=True)
    pd.concat(rows).to_csv("cache/s0_preds.csv", index=False)


if __name__ == "__main__":
    run_s0()