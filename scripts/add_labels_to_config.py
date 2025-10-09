import json, os

def inject_labels(model_dir, id2label):
    cfg_path = os.path.join(model_dir, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["id2label"] = {str(i): lbl for i, lbl in enumerate(id2label)}
    cfg["label2id"] = {lbl: i for i, lbl in enumerate(id2label)}
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"✅ Injected {len(id2label)} labels into {cfg_path}")

inject_labels("models/roberta_esg_relevance_v1", ["No", "Yes"])
inject_labels("models/roberta_esg_category_v1", ["E", "S", "G"])