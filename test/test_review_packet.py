"""The SwiGLU review packet is complete and points at real files."""
import json
from pathlib import Path

PACKET = Path("review/swiglu_split")


def test_packet_manifest_files_exist():
    manifest = json.loads((PACKET / "manifest.json").read_text())
    for key in ("source", "reference", "generated", "handwritten", "inputs"):
        filename = manifest[key].split(":")[0]
        assert (PACKET / filename).exists(), filename
    assert manifest["kernel_evidence"]["source_map"] == {"split": 16, "gate": 16, "up": 16}


def test_packet_has_two_implementations():
    handwritten = (PACKET / "handwritten.py").read_text()
    generated = (PACKET / "generated_module.py").read_text()
    assert "split_copy_kernel" in handwritten
    assert "run_swiglu_oai_split" in generated
