from pathlib import Path


def test_layer_packages_and_canonical_grammar_exist():
    root = Path(__file__).resolve().parents[1] / "compiler"
    for layer in ("frontend", "ir", "analysis", "optimization", "backend", "runtime"):
        assert (root / layer / "__init__.py").exists()
    assert (root / "grammar.lark").exists()
    assert not (root / "frontend" / "grammar.lark").exists()


def test_legacy_imports_resolve_to_layer_implementations():
    from compiler.ast_node import TensorType as OldTensorType
    from compiler.frontend.ast_node import TensorType as NewTensorType
    from compiler.triton_backend import TritonProgram as OldProgram
    from compiler.backend.triton_backend import TritonProgram as NewProgram
    assert OldTensorType is NewTensorType
    assert OldProgram is NewProgram
