from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

HF_REL = "salitahir/roberta-esg-relevance-green-guard-v1"
HF_CAT = "salitahir/roberta-esg-category-green-guard-v1"

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
            return {"relevance": "No", "esg": "Non-ESG"}
        c = self.tok_cat(text, return_tensors="pt", truncation=True)
        cat_idx = self.m_cat(**c).logits.argmax(-1).item()
        esg = self.m_cat.config.id2label[str(cat_idx)]
        return {"relevance": "Yes", "esg": esg}
