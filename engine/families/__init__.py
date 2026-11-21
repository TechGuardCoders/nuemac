"""Model families: each one tells the engine how to load a model type, build its
cache, snapshot and restore it, which token ids stop generation, and how to
parse its output. Adding one: docs/recipes/adding-a-family.md."""
import json
import os


class ContextFull(Exception):
    pass


def _families():
    from . import inkling2

    return [inkling2]


def for_model(model_path):
    cfg = json.load(open(os.path.join(model_path, "config.json")))
    mtype = cfg.get("model_type") or (cfg.get("text_config") or {}).get("model_type")
    for fam in _families():
        if mtype in fam.MODEL_TYPES:
            return fam
    names = sorted(t for f in _families() for t in f.MODEL_TYPES)
    raise SystemExit(f"model_type {mtype!r} has no Nuemac family yet (supported: {', '.join(names)})")
