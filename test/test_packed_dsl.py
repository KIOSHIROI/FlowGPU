"""The NVFP4 DSL program is the promoted replacement for task 172's extern kernel.

The program is compiled and executed through the reference backend; its packed
data bytes and packed scale word must match ``flowgpu.compiler.packed_layout`` exactly
(the task grades both outputs exact).  The oracle itself is pinned against the
statement in ``test_packed_layout_v2.py``.
"""
from __future__ import annotations

import pathlib
import random

from flowgpu.compiler import compile_source
from flowgpu.compiler.backend import reference
from flowgpu.compiler.packed_layout import quantize_nvfp4_indexer

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/packed/fp4_indexer.fgpu"
CONSTANTS = (6.0, 6e-4, -1.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0,
             0, 1, -1, 4, 8, 16, 23, 24, 31, 254, 255, 0x7FFFFF)


def _pipeline():
    return compile_source(EXAMPLE.read_text(), source_path=str(EXAMPLE)).pipelines[0]


def _run(values):
    pipeline = _pipeline()
    namespace = {}
    exec(reference.module([pipeline.reference]), namespace)
    produced = namespace["run_quantize_fp4_indexer"](values, *CONSTANTS)
    return produced


def test_the_program_compiles_and_emits_the_expected_tensor_shapes():
    pipeline = _pipeline()
    assert pipeline.triton.complete, [str(m) for m in pipeline.triton.messages()]
    assert pipeline.reference.complete, [str(m) for m in pipeline.reference.messages()]
    assert pipeline.triton.kernels
    text = reference.module([pipeline.reference])
    assert "run_quantize_fp4_indexer" in text
    produced = _run([0.0] * 512)
    assert len(produced["q"]) == 256          # [4,4,16] uint8 bytes
    assert len(produced["sf_word"]) == 4      # [4,1] one packed word per token


def test_the_program_reproduces_the_packed_layout_oracle_for_random_inputs():
    for seed in (1, 11, 47):
        generator = random.Random(seed)
        values = [generator.uniform(-7.0, 7.0) for _ in range(4 * 4 * 32)]
        produced = _run(values)
        packed, words = quantize_nvfp4_indexer(values, [4, 128])
        got_packed = [int(float(value)) for value in produced["q"]]
        got_words = [int(word) & 0xFFFFFFFF for word in produced["sf_word"]]
        assert got_packed == packed, seed
        assert got_words == [word & 0xFFFFFFFF for word in words], seed


def test_boundary_values_stay_exact():
    values = [0.0, -0.0, 1e-9, 1e9, 6.0, -6.0, 0.25, -0.25, 0.250001,
              5.1, -5.1, 448.0, -448.0] + [0.75] * (512 - 13)
    produced = _run(values)
    packed, words = quantize_nvfp4_indexer(values, [4, 128])
    assert [int(float(value)) for value in produced["q"]] == packed
    assert [int(word) & 0xFFFFFFFF for word in produced["sf_word"]] == [
        word & 0xFFFFFFFF for word in words]


# --- AWQ int4 (task 161) ----------------------------------------------------

AWQ_EXAMPLE = ROOT / "examples/packed/awq_dequant.fgpu"
AWQ_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)


def _awq_expected(qweight, zeros, scales, rows=4, columns=16, group=2):
    """Independent statement translation with the task's fp16 output rounding."""
    import struct

    words_per_row = columns // 8
    expected = []
    for row in range(rows):
        scale_row = row // group
        for word in range(words_per_row):
            raw = [(qweight[row * words_per_row + word] >> shift) & 0xF
                   for shift in range(0, 32, 4)]
            raw_zero = [(zeros[scale_row * words_per_row + word] >> shift) & 0xF
                        for shift in range(0, 32, 4)]
            for column in range(8):
                value = raw[AWQ_ORDER[column]] - raw_zero[AWQ_ORDER[column]]
                product = value * scales[scale_row * columns + word * 8 + column]
                expected.append(struct.unpack("e", struct.pack("e", product))[0])
    return expected


def test_awq_dequant_program_compiles_and_matches_the_statement():
    import random
    import struct

    pipeline = compile_source(AWQ_EXAMPLE.read_text(),
                              source_path=str(AWQ_EXAMPLE)).pipelines[0]
    assert pipeline.triton.complete, [str(m) for m in pipeline.triton.messages()]
    assert pipeline.reference.complete, [str(m) for m in pipeline.reference.messages()]
    namespace = {}
    exec(reference.module([pipeline.reference]), namespace)
    generator = random.Random(21)
    qweight = [generator.getrandbits(32) for _ in range(8)]
    zeros = [generator.getrandbits(32) for _ in range(4)]
    scales = [struct.unpack("e", struct.pack("e", generator.uniform(0.01, 2.0)))[0]
              for _ in range(32)]
    group_index = [0, 0, 1, 1]
    produced = namespace["run_awq_dequantize"](
        qweight, zeros, scales, group_index, 0, 4, 8, 12, 16, 20, 24, 28, 15)["out"]
    expected = _awq_expected(qweight, zeros, scales)
    assert len(produced) == len(expected) == 64
    for got, want in zip(produced, expected):
        assert abs(got - want) < 1e-9, (got, want)


def test_awq_dequant_program_covers_all_nibble_positions_and_signs():
    import struct

    # every nibble of the words is a distinct value; scales 1.0 makes the
    # result exactly (nibble - zero nibble)
    qweight = [0x76543210, 0xFEDCBA98] * 4      # K*W = 4 rows * 2 words
    zeros = [0x00000000, 0x11111111] * 2        # (K/G)*W = 2 groups * 2 words
    ones = [1.0] * 32
    pipeline = compile_source(AWQ_EXAMPLE.read_text(),
                              source_path=str(AWQ_EXAMPLE)).pipelines[0]
    namespace = {}
    exec(reference.module([pipeline.reference]), namespace)
    produced = namespace["run_awq_dequantize"](
        qweight, zeros, ones, [0, 0, 1, 1], 0, 4, 8, 12, 16, 20, 24, 28, 15)["out"]
    expected = _awq_expected(qweight, zeros, ones)
    assert [round(value) for value in produced] == [round(value) for value in expected]
