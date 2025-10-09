from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


HF_REL = "salitahir/roberta-esg-relevance-green-guard-v1"
HF_CAT = "salitahir/roberta-esg-category-green-guard-v1"


def _safe_label(config, idx: int, default: str | None = None) -> str:
    """
    Return a human-friendly label for class index `idx`, robust to various
    id2label formats: dict with int or str keys, list, or missing.
    """
    id2label = getattr(config, "id2label", None)
    # dict case
    if isinstance(id2label, dict):
        if str(idx) in id2label:
            return id2label[str(idx)]
        if idx in id2label:
            return id2label[idx]
    # list/tuple case
    if isinstance(id2label, (list, tuple)) and 0 <= idx < len(id2label):
        return id2label[idx]
    # fallback via label2id reverse
    label2id = getattr(config, "label2id", None)
    if isinstance(label2id, dict):
        rev = {v: k for k, v in label2id.items()}
        if idx in rev:
            return rev[idx]
        if str(idx) in rev:
            return rev[str(idx)]
    # absolute last resort
    return default if default is not None else str(idx)


def _is_relevant_label(name: str) -> bool:
    """
    Decide if a label string indicates ESG relevance (binary model).
    Accepts common variants to be robust against configs like LABEL_1, yes, etc.
    """
    s = (name or "").strip().lower()
    # common positives
    return s in {"yes", "relevant", "esg", "positive", "1", "label_1"}


class ESGClassifier:
    def __init__(self, rel_path: str = HF_REL, cat_path: str = HF_CAT):
        self.tok_rel = AutoTokenizer.from_pretrained(rel_path)
        self.m_rel = AutoModelForSequenceClassification.from_pretrained(rel_path).eval()

        self.tok_cat = AutoTokenizer.from_pretrained(cat_path)
        self.m_cat = AutoModelForSequenceClassification.from_pretrained(cat_path).eval()

    @torch.inference_mode()
    def predict_one(self, text: str):
        # Stage 1: relevance
        r = self.tok_rel(text, return_tensors="pt", truncation=True)
        r_logits = self.m_rel(**r).logits
        r_idx = int(r_logits.argmax(-1).item())
        r_label = _safe_label(self.m_rel.config, r_idx, default=str(r_idx))

        if not _is_relevant_label(r_label):
            # Treat anything not recognized as "relevant" as Non-ESG
            return {"relevance": "No", "esg": "Non-ESG"}

        # Stage 2: ESG category
        c = self.tok_cat(text, return_tensors="pt", truncation=True)
        c_logits = self.m_cat(**c).logits
        c_idx = int(c_logits.argmax(-1).item())
        c_label = _safe_label(self.m_cat.config, c_idx, default=str(c_idx))

        # Normalize common variants (E/S/G, label_0/1/2, etc.)
        norm = c_label.strip().upper()
        if norm in {"E", "S", "G"}:
            esg = norm
        elif norm in {"LABEL_0", "0"}:
            esg = "E"
        elif norm in {"LABEL_1", "1"}:
            esg = "S"
        elif norm in {"LABEL_2", "2"}:
            esg = "G"
        else:
            # fallback if label names are unexpected
            esg = c_label

        return {"relevance": "Yes", "esg": esg}
