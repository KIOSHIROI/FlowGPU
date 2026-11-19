"""The carried-tuple scan family: `cummin_last` / `cummax_last` (P1).

`cumsum` carries one fp32 total; a prefix min/max must also carry *which index*
attained it.  That tuple is the structural step towards the `iterate` region
(docs/ITERATION_REGION.md), and the reference lowering here is the numeric
oracle the device lowering will be checked against.

Contract pinned by these tests (device probe against `torch.cummin`/`torch.cummax`
is the remaining check before the emitted form is enabled):

* values are the prefix min/max;
* on ties the **earliest** index is kept (this is what makes the merge function
  associative: `a if a.value <= b.value else b`);
* once NaN appears, the value stays NaN and the index stays at the first NaN;
* `indices` has the input's shape and `int64` dtype.

The emitted Triton form is deliberately *refused* for now -- with a refusal, not
an exception -- so nothing silently produces the wrong answer before the device
probe exists.
"""
import pytest

from compiler.backend import reference
from compiler.compile import compile_source
from compiler.frontend.operator import OPERATORS

PROGRAM = '''
pipeline p {{
    input x : fp32[{rows},{cols}]
    values, indices = {operator}(x)
    output values
    output indices
}}'''


def _run(operator, rows, cols, values):
    source = PROGRAM.format(operator=operator, rows=rows, cols=cols)
    pipeline = compile_source(source).pipelines[0]
    namespace = {}
    exec(reference.module([pipeline.reference]), namespace)
    return pipeline, namespace['run_p'](values)


def _oracle(values, rows, cols, choose):
    out_values, out_indices = [], []
    for row in range(rows):
        running = None
        running_index = 0
        for column in range(cols):
            position = row * cols + column
            value = values[position]
            if running is None or running == running:
                if running is None:
                    running, running_index = value, column
                elif value != value:
                    running, running_index = float('nan'), column
                elif choose(value, running):
                    running, running_index = value, column
            out_values.append(running)
            out_indices.append(running_index)
    return out_values, out_indices


@pytest.mark.parametrize('operator,choose', [
    ('cummin_last', lambda value, running: value < running),
    ('cummax_last', lambda value, running: value > running),
])
def test_the_reference_prefix_extreme_matches_a_sequential_oracle(operator, choose):
    rows, cols = 3, 5
    values = [3.0, 1.0, 2.0, 1.0, 5.0,
              -1.0, -1.0, 0.0, 4.0, -2.0,
              2.0, 2.0, 2.0, 1.0, 9.0]
    pipeline, produced = _run(operator, rows, cols, values)
    wanted_values, wanted_indices = _oracle(values, rows, cols, choose)
    assert produced['values'] == pytest.approx(wanted_values, nan_ok=True)
    assert produced['indices'] == wanted_indices


def test_ties_keep_the_earliest_index():
    pipeline, produced = _run('cummin_last', 1, 4, [2.0, 2.0, 2.0, 1.0])
    assert produced['indices'] == [0, 0, 0, 3]


def test_nan_propagates_and_keeps_the_first_nan_index():
    pipeline, produced = _run('cummin_last', 1, 4, [1.0, float('nan'), 0.0, -5.0])
    # the prefix stays finite up to the NaN, then it stays NaN: nothing can beat
    # a NaN, so the index stops at the first NaN
    assert produced['values'][0] == 1.0
    assert all(value != value for value in produced['values'][1:])
    assert produced['indices'] == [0, 1, 1, 1]


def test_infinities_participate():
    pipeline, produced = _run('cummax_last', 1, 4, [1.0, float('inf'), -float('inf'), 2.0])
    assert produced['values'] == [1.0, float('inf'), float('inf'), float('inf')]
    assert produced['indices'] == [0, 1, 1, 1]


def test_the_indices_result_is_int64_and_same_shape():
    values = OPERATORS['cummin_last'].infer(
        [type('T', (), {'dtype': 'fp32', 'shape': ['4', '8']})()], 2)
    assert values[1].dtype == 'int64'
    assert values[1].shape == ['4', '8']


def test_the_device_form_is_refused_rather_than_guessed():
    source = PROGRAM.format(operator='cummin_last', rows=3, cols=5)
    pipeline = compile_source(source).pipelines[0]
    assert len(pipeline.triton.kernels) == 0
    refusals = [refusal for refusal in pipeline.triton.refusals
                if refusal.operator == 'cummin_last']
    assert refusals and 'carried-tuple' in refusals[0].reason
