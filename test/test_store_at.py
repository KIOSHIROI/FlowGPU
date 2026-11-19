"""`store_at`: the first consumer of the program-uniform scalar capability.

The hand-written Task153 kernel reads four program-uniform scalars per program
and then writes contiguous runs at `base + arange(lane)`. `store_at` is exactly
that write shape: `base = offsets[p]` is one scalar load per program, and the
run `values[p, :]` goes to `dst_flat[base + arange(lane_count)]`. The address is
`dst_ptr + scalar_base + affine(arange)` -- which `analysis/write_address`
classifies as affine, and which the measured targets sustain (unlike
`scatter_by_index`'s arbitrary per-lane destination, which stays on its own
gate).

What this file pins, in the order of `docs/STORE_AT_DESIGN_20261022.md`:

* the staged type rule (both spellings; integer offsets; same dtype; row count);
* the emission: one scalar load of the base per program, never re-derived per
  lane; write address affine (checked by the emitter itself, contract 3);
* coverage: one kernel/one pass when the runs provably tile `dst`, otherwise a
  copy kernel whose launch is guarded by a **metadata-only shape comparison**
  (no device read);
* the fail-closed host contract: bounds and pairwise-disjoint bases checked
  before any allocation or launch (one D2H + sync), and the
  `store_at_trusted` author-level declaration that skips it and says so in the
  emitted source;
* numerics: the emitted source executed on the CPU Triton double against an
  independent, hand-written oracle (`expected[base:base+L] = values[p]`), plus
  the generated reference backend. NumPy is not installed in this venv, so the
  oracle is the same element assignment in plain Python;
* the access-shape classification (`affine_write`, sustained) and the fact that
  an affine-write target does not refuse the plan.

CPU evidence only. The board/device is the device experiment; `store_at` is not
registered in ``frontend/operator.py`` yet -- the tests register the staged
spellings and remove them again, so every default build is untouched.
"""
import ast
import math
import types

import pytest

from flowgpu.compiler.analysis import access_shape, write_address
from flowgpu.compiler.backend.platform_configs import resolve_platform
from flowgpu.compiler.backend.triton_backend import module
from flowgpu.compiler.compile import compile_source
from flowgpu.compiler.frontend.diagnostic import CompileError
from flowgpu.compiler.frontend.operator import OPERATORS
from flowgpu.compiler.frontend.store_at import (
    STAGED_OPERATOR,
    STAGED_OPERATORS,
    TRUSTED_OPERATOR,
    register as register_store_at,
    unregister as unregister_store_at,
)
from test import segmented_support
from flowgpu.compiler.frontend.diagnostic import RuleError

LITERAL = (
    "pipeline lit { input dst:bf16[256] input starts:int32[4] "
    "input values:bf16[4,64] out=store_at(dst,starts,values) output out }"
)
SYMBOLIC = (
    "pipeline sym { input dst:bf16[T] input starts:int32[C] "
    "input values:bf16[C,L] out=store_at(dst,starts,values) output out }"
)
TRUSTED = (
    "pipeline tr { input dst:bf16[256] input starts:int32[4] "
    "input values:bf16[4,64] out=store_at_trusted(dst,starts,values) output out }"
)
EXAMPLE = """
pipeline kda_prologue_store {
    input dst    : bf16[T]
    input starts : int32[C]
    input values : bf16[C,BT,K]
    out = store_at(dst, starts, values)
    output out
}
"""


@pytest.fixture
def staged():
    register_store_at()
    try:
        yield
    finally:
        unregister_store_at()


def pipeline(source, **options):
    return compile_source(source, **options).pipelines[0]


def kernels(source, **options):
    return tuple(pipeline(source, **options).triton.kernels)


# ---------------------------------------------------------------------------
# staged type rule
# ---------------------------------------------------------------------------


def test_store_at_is_staged_until_the_coordinated_registry_commit():
    """Before the merge window the registry is untouched; after it, canonical.

    `operator.py` is owned by the in-flight `segmented_cumsum` work; the two
    registrations are staged through `store_at.register()`. The same assertion
    passes on both sides of that commit, and the fixture's `unregister()` only
    removes operators this module added.
    """
    present = [name in OPERATORS for name in STAGED_OPERATORS]
    assert all(present) or not any(present)
    if all(present):
        from flowgpu.compiler.frontend.store_at_rule import infer_store_at
        for name in STAGED_OPERATORS:
            assert OPERATORS[name].infer is infer_store_at
            assert OPERATORS[name].pointwise is False
            assert OPERATORS[name].arity == 3


def test_a_program_uniform_base_requires_a_rank_one_integer_table(staged):
    from flowgpu.compiler.frontend.store_at import infer_store_at
    from flowgpu.compiler.frontend.ast_node import TensorType

    dst = TensorType("bf16", ["T"])
    values = TensorType("bf16", ["C", "L"])
    with pytest.raises(RuleError, match='one program-uniform base per program'):
        infer_store_at([dst, TensorType("fp32", ["C"]), values])
    with pytest.raises(RuleError, match='one program-uniform base per program'):
        infer_store_at([dst, TensorType("int32", ["C", "2"]), values])
    with pytest.raises(RuleError, match='values have C2 rows and offsets have C'):
        infer_store_at([dst, TensorType("int32", ["C"]),
                        TensorType("bf16", ["C2", "L"])])
    with pytest.raises(RuleError, match='store_at requires matching dtypes'):
        infer_store_at([dst, TensorType("int32", ["C"]),
                        TensorType("fp16", ["C", "L"])])
    result = infer_store_at([dst, TensorType("int32", ["C"]), values])
    assert result == dst


def test_both_spellings_have_the_same_rule(staged):
    program = pipeline(TRUSTED)
    assert program.triton.complete
    assert program.triton.kernels[0].kind == "store_at"


# ---------------------------------------------------------------------------
# emission: one scalar base load, affine write, no per-lane index
# ---------------------------------------------------------------------------


def test_a_tiled_destination_emits_one_kernel_and_one_pass(staged):
    result = kernels(LITERAL)
    assert len(result) == 1
    (kernel,) = result
    assert "copy" not in kernel.name

    scalar_line = next(line for line in kernel.source.splitlines()
                       if "= tl.load(starts_ptr" in line)
    # One program-uniform base load: no lane index in it, no mask, pinned to a
    # 64-bit address term.
    assert "tl.arange" not in scalar_line
    assert "mask=" not in scalar_line
    assert ".to(tl.int64)" in scalar_line
    assert kernel.source.count("tl.load(") == 2      # base + values tile

    # The write is `dst_ptr + scalar_base + lane`: affine by construction.
    addresses = write_address.store_addresses(kernel.source)
    assert addresses and all(address.affine for address in addresses)
    assert "tl.store(out_ptr + base_scalar + lane, value, mask=mask)" in kernel.source

    # The base is loaded exactly once, at kernel top level.
    assert kernel.source.count("tl.load(starts_ptr") == 1


def test_an_untiled_destination_emits_copy_then_write(staged):
    result = kernels(SYMBOLIC)
    assert [kernel.name for kernel in result] == ["sym_k0_copy", "sym_k0"]
    copy, write = result

    # The copy is a plain contiguous pass; its store address is affine too.
    assert copy.launch_guard == "(T) != (C) * (L)"
    assert all(address.affine
               for address in write_address.store_addresses(copy.source))
    assert all(address.affine
               for address in write_address.store_addresses(write.source))

    # The launch decision is a metadata-only shape comparison: no tensor data,
    # no `.numel()` call, no device read.
    assert ".numel()" not in copy.launch_guard
    assert ".detach" not in copy.launch_guard
    assert "dst" not in copy.launch_guard


def test_the_copy_launch_is_guarded_and_the_write_is_not(staged):
    host = pipeline(SYMBOLIC).triton.host
    lines = [line.strip() for line in host.splitlines()]
    guard = lines.index("if (T) != (C) * (L):")
    assert "sym_k0_copy[" in lines[guard + 1]
    assert not lines[guard + 2].startswith("if ")
    assert any("sym_k0[" in line for line in lines[guard + 2:])


def test_one_program_per_value_row(staged):
    (kernel,) = kernels(LITERAL)
    assert kernel.grid == "(4,)"
    assert "p = tl.program_id(0)" in kernel.source
    # The base is indexed by the program, never by a lane.
    assert "tl.load(starts_ptr + p)" in kernel.source


def test_store_at_is_an_opaque_anchor_so_upstream_pointwise_stays_its_own_kernel(staged):
    source = (
        "pipeline p { input dst:bf16[256] input starts:int32[4] "
        "input values:bf16[4,64] input scale:fp32 "
        "scaled=mul(values,scale) out=store_at(dst,starts,scaled) output out }"
    )
    result = kernels(source)
    kinds = sorted(kernel.kind for kernel in result)
    assert kinds == ["pointwise", "store_at"]


def test_the_emitter_refuses_a_non_affine_store_address():
    """Contract 3, unconditionally: the classifier runs on the emitter's output."""
    from flowgpu.compiler.backend import store_at_backend

    bad = """def k(out_ptr):
    idx = tl.program_id(0) * 16 + tl.arange(0, 16)
    lane = tl.arange(0, 16)
    tl.store(out_ptr + idx // 8 + lane, 0)"""
    with pytest.raises(CompileError, match="applied to lane term 'idx'; lane dependency: idx, lane\\); decode the offset once per program \\(a program-uniform scalar base\\), not per lane\\. This target sustains only scalar_base \\+ affine\\(arange\\) writes") as caught:
        store_at_backend._require_affine(bad, "k",
                                         types.SimpleNamespace(pipeline="p"))
    assert "non-affine write address" in str(caught.value)


# ---------------------------------------------------------------------------
# fail-closed host contract
# ---------------------------------------------------------------------------


def test_the_host_reads_the_offsets_once_before_allocating(staged):
    host = pipeline(LITERAL).triton.host
    check = host.index("store_at caller contract")
    assert host.index("detach().cpu().tolist()") > check
    assert host.index("store_at caller contract") < host.index("torch.empty")
    assert "out of bounds" in host
    assert "overlap" in host


def test_the_trusted_spelling_skips_the_d2h_check_and_says_so(staged):
    program = pipeline(TRUSTED)
    host = program.triton.host
    assert "detach().cpu()" not in host
    assert "trusted_offsets author declaration" in host
    assert "skipped" in host
    # It is still a store_at: same kernel shape, same affine write.
    kernel = program.triton.kernels[0]
    assert all(address.affine
               for address in write_address.store_addresses(kernel.source))


def test_a_generated_host_parses_for_every_spelling_and_coverage(staged):
    for source in (LITERAL, SYMBOLIC, TRUSTED):
        host = pipeline(source).triton.host
        compile(host, "<host>", "exec")       # valid Python, no NameError at parse


def test_a_non_contiguous_destination_is_refused_before_launch(staged):
    program = pipeline(LITERAL).triton
    namespace, allocations, triton = load_emitted(module([program]))
    dst = _Buffer([0.0] * 256, (256,), "bf16", (2, 2))   # non-compact strides
    starts = _Buffer([0, 64, 128, 192], (4,), "int32")
    values = _Buffer([1.0] * 256, (4, 64), "bf16")
    with pytest.raises(ValueError, match='input dst must be contiguous for its consuming kernels \\(store_at do not carry layout metadata\\)') as caught:
        namespace["run_lit"](dst, starts, values)
    assert "contiguous" in str(caught.value)
    assert triton.launches == []


# ---------------------------------------------------------------------------
# numeric oracle: emitted source vs an independent assignment formula
# ---------------------------------------------------------------------------


def _oracle(dst, starts, values, lane_count):
    out = list(dst)
    for p, base in enumerate(starts):
        out[base:base + lane_count] = values[p * lane_count:(p + 1) * lane_count]
    return out


def _values(count, seed=7):
    return [((index * seed) % 17) - 8.0 for index in range(count)]


def test_the_emitted_source_matches_the_oracle_on_a_tiled_destination(staged):
    program = pipeline(LITERAL).triton
    dst = _values(256)
    starts = [192, 0, 128, 64]           # order must not matter
    values = _values(4 * 64, seed=3)
    namespace, allocations, triton = load_emitted(module([program]))
    result = namespace["run_lit"](
        _Buffer(dst, (256,), "bf16"), _Buffer(starts, (4,), "int32"),
        _Buffer(values, (4, 64), "bf16"),
    )
    assert triton.launches == [("lit_k0", (4,))]
    assert result["out"].data == _oracle(dst, starts, values, 64)


def test_the_copy_pass_preserves_untouched_regions(staged):
    source = (
        "pipeline part { input dst:fp32[T] input starts:int32[C] "
        "input values:fp32[C,L] out=store_at(dst,starts,values) output out }"
    )
    program = pipeline(source).triton
    dst = _values(320)
    starts = [256, 0]
    values = _values(2 * 64, seed=5)
    namespace, allocations, triton = load_emitted(module([program]))
    result = namespace["run_part"](
        _Buffer(dst, (320,), "fp32"), _Buffer(starts, (2,), "int32"),
        _Buffer(values, (2, 64), "fp32"),
    )
    assert [name for name, _grid in triton.launches] == ["part_k0_copy", "part_k0"]
    assert result["out"].data == _oracle(dst, starts, values, 64)


def test_a_fully_tiled_symbolic_destination_skips_the_copy_launch(staged):
    source = (
        "pipeline fit { input dst:fp32[T] input starts:int32[C] "
        "input values:fp32[C,L] out=store_at(dst,starts,values) output out }"
    )
    program = pipeline(source).triton
    dst = _values(128)
    starts = [64, 0]
    values = _values(2 * 64, seed=9)
    namespace, allocations, triton = load_emitted(module([program]))
    result = namespace["run_fit"](
        _Buffer(dst, (128,), "fp32"), _Buffer(starts, (2,), "int32"),
        _Buffer(values, (2, 64), "fp32"),
    )
    # T == C * L: the metadata-only guard is false, so only the write runs.
    assert [name for name, _grid in triton.launches] == ["fit_k0"]
    assert result["out"].data == _oracle(dst, starts, values, 64)


def test_the_generated_reference_matches_the_oracle_and_the_emitted_source(staged):
    from flowgpu.compiler.backend.reference import module as reference_module

    program = pipeline(LITERAL).triton
    dst = _values(256)
    starts = [0, 64, 128, 192]
    values = _values(4 * 64, seed=3)
    namespace = {}
    exec(reference_module([compile_source(LITERAL).pipelines[0].reference]),
         namespace)
    expected = namespace["run_lit"](list(dst), list(starts), list(values))["out"]
    assert expected == _oracle(dst, starts, values, 64)


@pytest.mark.parametrize("starts,message", [
    ([0, 64, 128, -1], "out of bounds"),
    ([0, 64, 128, 200], "out of bounds"),
    ([0, 32, 128, 192], "overlap"),
    ([0, 64, 64, 192], "overlap"),
])
def test_invalid_bases_are_refused_before_any_launch(staged, starts, message):
    program = pipeline(LITERAL).triton
    namespace, allocations, triton = load_emitted(module([program]))
    with pytest.raises(ValueError, match='store_at offsets o') as caught:
        namespace["run_lit"](
            _Buffer([0.0] * 256, (256,), "bf16"),
            _Buffer(starts, (4,), "int32"),
            _Buffer([1.0] * 256, (4, 64), "bf16"),
        )
    assert message in str(caught.value)
    assert triton.launches == []


def test_the_trusted_spelling_skips_the_bounds_check(staged):
    program = pipeline(TRUSTED).triton
    namespace, allocations, triton = load_emitted(module([program]))
    # No host ValueError: the author declared the bases valid. (The CPU double
    # would flag an out-of-bounds store, which is what a real device would do
    # when the declaration is a lie; the contract is the author's, auditable in
    # the emitted comment.)
    namespace["run_tr"](
        _Buffer([0.0] * 256, (256,), "bf16"),
        _Buffer([0, 64, 128, 192], (4,), "int32"),
        _Buffer([1.0] * 256, (4, 64), "bf16"),
    )
    assert [name for name, _grid in triton.launches] == ["tr_k0"]


# ---------------------------------------------------------------------------
# access shape, platform gate, and the 153-shaped example
# ---------------------------------------------------------------------------


def test_the_plan_is_classified_affine_write_and_not_refused_on_an_affine_target(staged):
    config = resolve_platform("kunlunxin",
                              allowed_kernel_kinds=("pointwise", "store_at"))
    program = pipeline(SYMBOLIC, platform_config=config)
    assert program.triton.complete
    shape = access_shape.group_shape(program.fusion)
    assert shape.kind == access_shape.AFFINE_WRITE
    assert shape.data_dependent
    # The gather/scatter gate speaks about GATHER/SCATTER/PERMUTED; an affine
    # write at a program-uniform base is the shape the target requires.
    assert access_shape.group_conflicts(program.fusion,
                                        resolve_platform("kunlunxin")) == []


def test_a_kunlun_build_with_the_store_at_kind_emits_the_affine_write_path(staged):
    config = resolve_platform("kunlunxin",
                              allowed_kernel_kinds=("pointwise", "store_at"))
    program = pipeline(SYMBOLIC, platform_config=config)
    assert program.triton.complete
    kernel = program.triton.kernels[-1]
    assert all(address.affine
               for address in write_address.store_addresses(kernel.source))


def test_the_153_shaped_example_compiles_with_one_base_load_per_program(staged):
    program = pipeline(EXAMPLE)
    assert program.triton.complete
    write = program.triton.kernels[-1]
    assert write.name == "kda_prologue_store_k0"
    assert write.source.count("tl.load(starts_ptr") == 1
    assert "tl.load(starts_ptr + p)" in write.source
    # The program covers chunks of BT*K contiguous elements; the coverage
    # decision is `T != C * (BT * K)` and stays host-metadata-only.
    copy = program.triton.kernels[0]
    assert copy.launch_guard == "(T) != (C) * (BT * K)"


# ---------------------------------------------------------------------------
# CPU Triton double utilities
# ---------------------------------------------------------------------------


class _Buffer(segmented_support.Buffer):
    """The emitted host's caller contract calls `.detach().cpu().tolist()`."""

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self.data)


def _allow_rewrite_store(address, value, mask=True, **hints):
    """`tl.store` that permits a second write, for the copy-then-write path."""
    buffer = address.buffer

    def put(index, item, keep):
        if keep:
            assert 0 <= index < len(buffer.data), ("out-of-bounds CPU store", index)
            buffer.writes.add(index)
            buffer.data[index] = segmented_support.cast(item, buffer.dtype)
        return 0

    segmented_support.apply(put, address.offset, value, mask)


def load_emitted(code):
    """Exec an emitted module against the CPU Triton double."""
    language = segmented_support.TL()
    language.store = _allow_rewrite_store
    triton = segmented_support.Triton(language)
    allocations = []

    def empty(shape, *, dtype, device):
        if isinstance(shape, int):
            shape = (shape,)
        buffer = _Buffer([None] * math.prod(shape), shape, dtype)
        allocations.append(buffer)
        return buffer

    torch = types.SimpleNamespace(
        empty=empty,
        empty_like=lambda x: empty(x.shape, dtype=x.dtype, device=x.device),
        strided="strided", float16="fp16", bfloat16="bf16", float32="fp32",
        int32="int32", int64="int64", bool="bool",
    )
    tree = ast.parse(code)
    tree.body = [
        node for node in tree.body
        if not isinstance(node, (ast.Import, ast.ImportFrom))
        and not (isinstance(node, ast.FunctionDef)
                 and node.name == "_flowgpu_device_key")
    ]
    namespace = {"tl": language, "triton": triton, "torch": torch}
    exec(compile(ast.fix_missing_locations(tree), "<cpu-double>", "exec"),
         namespace)
    return namespace, allocations, triton
