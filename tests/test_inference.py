import os
import pytest

from greenguard.inference import ESGClassifier


def test_import_and_class_exists():
    # Just verify the package is importable and the class is defined
    assert ESGClassifier is not None


@pytest.mark.skipif(os.environ.get("SKIP_HF_DOWNLOAD") == "1", reason="Skipping network-heavy model load")
def test_predict_one_smoke():
    clf = ESGClassifier()
    out = clf.predict_one("We reduced Scope 2 emissions by 24% in 2024.")
    assert set(out.keys()) == {"relevance", "esg"}
