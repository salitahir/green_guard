import os, json, shutil, sys

# --- CONFIG -------------------------------------------------------------------
# Change this list if your model folders live elsewhere.
MODEL_DIRS = [
    "models/roberta_esg_relevance_v1",
    "models/roberta_esg_category_v1",
]
LABELS = {
    "models/roberta_esg_relevance_v1": ["No", "Yes"],
    "models/roberta_esg_category_v1": ["E", "S", "G"],
}

# --- HELPERS ------------------------------------------------------------------
def dbg_ls(d):
    try:
        print(f"\n[ls] {d}")
        for p in os.listdir(d):
            full = os.path.join(d, p)
            print("  ", p, "(dir)" if os.path.isdir(full) else "(file)")
    except Exception as e:
        print(f"[warn] cannot list {d}: {e}")

def ensure_file_renamed(d, old_name, new_name):
    oldp = os.path.join(d, old_name)
    newp = os.path.join(d, new_name)
    if os.path.isdir(newp):
        # bad state: a directory named like the file; remove it
        print(f"[fix] '{newp}' is a directory — removing it")
        shutil.rmtree(newp)
    if os.path.isfile(oldp) and not os.path.isfile(newp):
        os.rename(oldp, newp)
        print(f"[mv] {old_name} -> {new_name}")
    return os.path.isfile(newp)

def normalize_model_dir(d):
    print(f"\n=== Normalizing: {d} ===")
    if not os.path.isdir(d):
        raise FileNotFoundError(f"Model folder not found: {d}")

    dbg_ls(d)

    # Normalize expected HF filenames
    ok_cfg  = ensure_file_renamed(d, "config", "config.json") or os.path.isfile(os.path.join(d,"config.json"))
    ok_tok  = ensure_file_renamed(d, "tokenizer_config", "tokenizer_config.json") or os.path.isfile(os.path.join(d,"tokenizer_config.json"))
    ok_stm  = ensure_file_renamed(d, "special_tokens_map", "special_tokens_map.json") or os.path.isfile(os.path.join(d,"special_tokens_map.json"))
    ok_merg = ensure_file_renamed(d, "merges", "merges.txt") or os.path.isfile(os.path.join(d,"merges.txt"))
    ok_vocab= ensure_file_renamed(d, "vocab", "vocab.json") or os.path.isfile(os.path.join(d,"vocab.json"))

    # Confirm we have a model weight file (safetensors or bin)
    has_weights = any(os.path.isfile(os.path.join(d, f)) for f in ("model.safetensors","pytorch_model.bin"))
    if not has_weights:
        raise FileNotFoundError(f"No model weights found in {d} (expected 'model.safetensors' or 'pytorch_model.bin')")

    if not ok_cfg:
        raise FileNotFoundError(f"No config.json found in {d} after normalization")

    # Inject id2label / label2id
    cfg_path = os.path.join(d, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    id2label = LABELS.get(d)
    if not id2label:
        # Try to match by suffix if the key differs only by prefix
        for key, val in LABELS.items():
            if d.endswith(os.path.basename(key)):
                id2label = val
                break
    if not id2label:
        raise KeyError(f"No labels configured for: {d}")

    cfg["id2label"] = {str(i): lab for i, lab in enumerate(id2label)}
    cfg["label2id"] = {lab: i for i, lab in enumerate(id2label)}

    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print(f"[ok] Injected labels into {cfg_path}")
    return True

def main():
    root = os.getcwd()
    print(f"[cwd] {root}")
    for d in MODEL_DIRS:
        normalize_model_dir(d)
        dbg_ls(d)
    print("\nAll done ✅")

if __name__ == "__main__":
    sys.exit(main())
