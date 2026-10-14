import json
import re


def build_messages(utt, names):
    system = (
        "You classify requests sent to a voice assistant into exactly one intent.\n"
        "Allowed intents: " + ", ".join(sorted(names)) + "\n"
        'Reply with JSON only, no other text: {"intent": "<one allowed intent>", '
        '"confidence": <number between 0 and 1>}'
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": utt}]


def parse(text, names):
    """Return dict: intent, confidence, json_ok, label_ok."""
    out = {"intent": None, "confidence": None, "json_ok": False, "label_ok": False}
    m = re.search(r"\{.*\}", text, re.DOTALL)      # tolerate stray text around the JSON
    if not m:
        return out
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return out
    out["json_ok"] = True
    out["intent"] = str(obj.get("intent", "")).strip()
    try:
        out["confidence"] = float(obj.get("confidence"))
    except (TypeError, ValueError):
        pass
    out["label_ok"] = out["intent"] in set(names)
    return out