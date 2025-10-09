import os, sys, json, shutil, subprocess, textwrap, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

PRINT_PREFIX = "[green-guard]"

def sh(cmd, check=True):
    print(f"{PRINT_PREFIX} $ {cmd}")
    result = subprocess.run(cmd, shell=True)
    if check and result.returncode != 0:
        sys.exit(result.returncode)

def warn(msg): print(f"{PRINT_PREFIX} WARN: {msg}")
def info(msg): print(f"{PRINT_PREFIX} {msg}")

# ---------- 1) Basic repo sanity ----------
info(f"Repo root = {ROOT}")
needed_dirs = ["source/greenguard", "data", "models", "notebooks", "scripts"]
for d in needed_dirs:
    p = ROOT / d
    if not p.exists():
        warn(f"Missing expected directory: {d}")

pkg_dir = ROOT / "source" / "greenguard"
(pkg_dir / "__init__.py").touch(exist_ok=True)

# ---------- 2) Python environment & deps ----------
# Ensure venv is active
if not sys.prefix or ".venv" not in sys.prefix:
    warn("Virtualenv not detected. You should 'python -m venv .venv && source .venv/bin/activate' before running.")
# Install minimal deps (CPU-safe)
sh("pip install --upgrade pip", check=True)
# Install torch CPU wheels if not present; ignore errors if already installed
sh("python -c \"import torch,sys;print(torch.__version__)\" || pip install --index-url https://download.pytorch.org/whl/cpu torch", check=False)
sh("pip install 'transformers>=4.43' 'huggingface_hub>=0.24' accelerate pandas scikit-learn", check=True)

# ---------- 3) Make package installable (pyproject + editable) ----------
pyproject = ROOT / "pyproject.toml"
if not pyproject.exists():
    pyproject.write_text(textwrap.dedent("""
        [build-system]
        requires = ["setuptools", "wheel"]
        build-backend = "setuptools.build_meta"

        [project]
        name = "greenguard"
        version = "0.1.0"
        requires-python = ">=3.9"
        dependencies = [
          "torch",
          "transformers>=4.43",
          "huggingface_hub>=0.24",
          "accelerate>=0.33",
          "pandas",
          "scikit-learn"
        ]

        [tool.setuptools]
        package-dir = {"" = "source"}
        packages = ["greenguard"]
    """).strip()+"\n")
    info("Created pyproject.toml")

# install editable
sh("pip install -e .", check=True)

# ---------- 4) Normalize local model folders & inject labels ----------
MODEL_DIRS = {
    "models/roberta_esg_relevance_v1": ["No","Yes"],
    "models/roberta_esg_category_v1":  ["E","S","G"],
}

def normalize_model_dir(model_dir: Path, id2label):
    if not model_dir.exists():
        warn(f"Model dir missing: {model_dir}")
        return False

    def ensure_rename(old, new):
        op_old = model_dir/old
        op_new = model_dir/new
        if op_new.is_dir():
            shutil.rmtree(op_new)
        if op_old.exists() and not op_new.exists():
            op_old.rename(op_new)
            info(f"Renamed {old} -> {new}")
        return op_new.exists()

    ok = True
    ok &= ensure_rename("config", "config.json") or (model_dir/"config.json").exists()
    ok &= ensure_rename("tokenizer_config", "tokenizer_config.json") or (model_dir/"tokenizer_config.json").exists()
    ok &= ensure_rename("special_tokens_map", "special_tokens_map.json") or (model_dir/"special_tokens_map.json").exists()
    ok &= ensure_rename("merges", "merges.txt") or (model_dir/"merges.txt").exists()
    ok &= ensure_rename("vocab", "vocab.json") or (model_dir/"vocab.json").exists()

    has_weights = (model_dir/"model.safetensors").exists() or (model_dir/"pytorch_model.bin").exists()
    if not has_weights:
        warn(f"No weights (model.safetensors/pytorch_model.bin) in {model_dir}")
        ok = False

    cfg_path = model_dir/"config.json"
    if not cfg_path.exists():
        warn(f"Missing config.json in {model_dir}")
        ok = False
    else:
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception as e:
            warn(f"config.json unreadable in {model_dir}: {e}")
            ok = False
        else:
            cfg["id2label"] = {str(i): lab for i, lab in enumerate(id2label)}
            cfg["label2id"] = {lab: i for i, lab in enumerate(id2label)}
            cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
            info(f"Injected id2label/label2id in {cfg_path}")

    return ok

all_ok = True
for relpath, labels in MODEL_DIRS.items():
    ok = normalize_model_dir(ROOT/relpath, labels)
    all_ok &= ok

# ---------- 5) Push to Hugging Face (if token present) ----------
from huggingface_hub import HfApi, create_repo, hf_hub_url
api = HfApi()

HF_REPOS = {
    "models/roberta_esg_relevance_v1": "salitahir/roberta-esg-relevance-green-guard-v1",
    "models/roberta_esg_category_v1":  "salitahir/roberta-esg-category-green-guard-v1",
}

def upload_six_files(local_dir: Path, repo_id: str):
    # Create repo if missing
    create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)

    # Collect files to push
    files = []
    for name in ["model.safetensors", "pytorch_model.bin", "config.json",
                 "tokenizer_config.json", "special_tokens_map.json", "merges.txt", "vocab.json"]:
        p = local_dir/name
        if p.exists():
            files.append(name)

    if not files:
        warn(f"No files found to upload in {local_dir}.")
        return

    operations = []
    for fname in files:
        operations.append({"op":"addOrUpdate",
                           "path_in_repo": fname,
                           "path_or_fileobj": str(local_dir/fname)})

    api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload weights and tokenizer files",
        operations=operations,
    )
    info(f"Uploaded to https://huggingface.co/{repo_id}")

# Only attempt upload if token is available
try:
    me = api.whoami()
    info(f"HF auth OK as: {me.get('name') or me.get('email')}")
    for relpath, repo in HF_REPOS.items():
        upload_six_files(ROOT/relpath, repo)
except Exception as e:
    warn(f"Skipping HF upload (auth or network issue): {e}")

# ---------- 6) Ensure clean inference that loads from HF ----------
INF = pkg_dir/"inference.py"
HF_REL = HF_REPOS["models/roberta_esg_relevance_v1"]
HF_CAT = HF_REPOS["models/roberta_esg_category_v1"]

desired_inference = textwrap.dedent(f"""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    HF_REL = "{HF_REL}"
    HF_CAT = "{HF_CAT}"

    class ESGClassifier:
        def __init__(self, rel_path: str = HF_REL, cat_path: str = HF_CAT):
            self.tok_rel = AutoTokenizer.from_pretrained(rel_path)
            self.m_rel = AutoModelForSequenceClassification.from_pretrained(rel_path).eval()
            self.tok_cat = AutoTokenizer.from_pretrained(cat_path)
            self.m_cat = AutoModelForSequenceClassification.from_pretrained(cat_path).eval()

        @torch.inference_mode()
        def predict_one(self, text: str):
            r = self.tok_rel(text, return_tensors="pt", truncation=True)
            rel_idx = self.m_rel(**r).logits.argmax(-1).item()
            rel = self.m_rel.config.id2label[str(rel_idx)]
            if rel == "No":
                return {{"relevance": "No", "esg": "Non-ESG"}}
            c = self.tok_cat(text, return_tensors="pt", truncation=True)
            cat_idx = self.m_cat(**c).logits.argmax(-1).item()
            esg = self.m_cat.config.id2label[str(cat_idx)]
            return {{"relevance": "Yes", "esg": esg}}
""").strip()+"\n"

# Replace file if missing or different
write_inf = True
if INF.exists():
    try:
        current = INF.read_text(encoding="utf-8").strip()
        if current == desired_inference.strip():
            write_inf = False
    except Exception:
        pass
if write_inf:
    INF.write_text(desired_inference, encoding="utf-8")
    info("Wrote clean src/greenguard/inference.py")

# ---------- 7) .gitignore cleanup ----------
gi = ROOT/".gitignore"
gi_lines = set()
if gi.exists(): gi_lines = set(gi.read_text().splitlines())
for line in ["models/", "artifacts/models/", "*.bin", "*.safetensors", "__pycache__/", ".venv/"]:
    gi_lines.add(line)
gi.write_text("\n".join(sorted([l for l in gi_lines if l.strip()!='']))+"\n")
info("Updated .gitignore")

# ---------- 8) Move legacy artifacts that are no longer needed for inference ----------
legacy_dir = ROOT/"artifacts"/"legacy"
legacy_dir.mkdir(parents=True, exist_ok=True)
for name in ["label_encoder_esg.pkl", "claim_label_encoder.pkl", "label_mapping_relevance.json"]:
    p = ROOT/"artifacts"/name
    if p.exists():
        shutil.move(str(p), str(legacy_dir/p.name))
        info(f"Moved {p.name} to artifacts/legacy/ (kept for provenance)")

# ---------- 9) Sanity test: import & inference on a sample ----------
ok_import = False
try:
    import greenguard  # noqa
    from greenguard.inference import ESGClassifier
    ok_import = True
    info("Import greenguard OK")
except Exception as e:
    warn(f"Import error: {e}")

if ok_import:
    try:
        clf = ESGClassifier()
        out = clf.predict_one("We reduced Scope 2 emissions by 24% in 2024.")
        info(f"Sample prediction: {out}")
    except Exception as e:
        warn(f"Inference test failed (models may not be uploaded yet): {e}")

# ---------- 10) Git commit (idempotent) ----------
sh("git add -A", check=False)
sh("git commit -m 'Audit/fix: normalize models, inject labels, upload to HF, clean inference, package installable' || true", check=False)
info("All done ✅  Review the log above for any WARN lines.")
