import math
import pytest
from compiler.compile import compile_source
from compiler import reference
from compiler.cli import main
from compiler.diagnostic import CompileError


def run(source,args,**opts):
    p=compile_source(source).pipelines[0];ns={}
    exec(reference.module([p.reference],**opts),ns)
    return ns['run_p'](*args),p


@pytest.mark.parametrize('axis',[0,1,2,-1,-2,-3])
@pytest.mark.parametrize('operation',['sum_axis','max_axis','softmax_axis','rms_norm_axis'])
def test_arbitrary_axis_matches_independent_coordinate_oracle(operation,axis):
    shape=(2,3,4);a=axis%3;x=[(i-12)/8 for i in range(24)]
    src=f'pipeline p {{ input x : fp32[2,3,4] '
    if operation=='rms_norm_axis':src+=f'input w : fp32[{shape[a]}] y={operation}(x,w,{axis}) output y }}';args=[x,[1.]*shape[a]]
    else:src+=f'y={operation}(x,{axis}) output y }}';args=[x]
    got,p=run(src,args)
    output_shape=list(shape)
    if operation in {'sum_axis','max_axis'}:output_shape[a]=1
    expected=[]
    for i in range(output_shape[0]):
        for j in range(output_shape[1]):
            for k in range(output_shape[2]):
                point=[i,j,k];values=[]
                for v in range(shape[a]):
                    q=point.copy();q[a]=v;values.append(x[(q[0]*3+q[1])*4+q[2]])
                if operation=='sum_axis':value=sum(values)
                elif operation=='max_axis':value=max(values)
                elif operation=='softmax_axis':
                    m=max(values);exps=[math.exp(v-m) for v in values];value=exps[point[a]]/sum(exps)
                else:value=values[point[a]]/math.sqrt(sum(v*v for v in values)/len(values)+1e-6)
                expected.append(value)
    assert got['y']==pytest.approx(expected,abs=1e-6)
    assert p.dataflow.values['y'].shape==list(map(str,output_shape))
    assert not any(n.op.endswith('_axis') for n in p.dataflow.nodes)


@pytest.mark.parametrize('expression',['sum_axis(x,3)','sum_axis(x,-4)','sum_axis(x,x)','sum_axis(x,[1])','add(x,1)','sum_axis(x,1,2)'])
def test_bad_axis_arguments_are_structured_diagnostics(expression):
    with pytest.raises(CompileError, match='error in pipeline'):compile_source(f'pipeline p {{ input x : fp32[2,3,4] y={expression} output y }}')


def test_rank_one_symbolic_and_weight_contract():
    got,_=run('pipeline p { input x : bf16[D] y=sum_axis(x,-1) output y }',[[1.,2.,3.],3])
    assert got['y']==[6.]
    with pytest.raises(CompileError, match='olumn 56: rms_norm weight shape \\[4\\] does not match the last dimension of \\[4,2,3\\]'):compile_source('pipeline p { input x : fp32[2,3,4] input w : fp32[4] y=rms_norm_axis(x,w,1) output y }')
    with pytest.raises(CompileError, match='olumn 37: reduce_last requires floating-point tensors'):compile_source('pipeline p { input x : int32[2,3] y=sum_axis(x,0) output y }')


def test_fp32_emulation_is_separate_from_fp64_spec_and_int_wrap():
    src='pipeline p { input x : fp32[1,3] y=sum_last(x) output y }'
    assert run(src,[[16777216.,1.,-16777216.]])[0]['y']==[1.]
    assert run(src,[[16777216.,1.,-16777216.]],emulate_fp32=True)[0]['y']==[0.]
    src='pipeline p { input x : int32[2] input y : int32[2] z=add(x,y) output z }'
    assert run(src,[[2**31-1,-2**31],[1,-1]],emulate_fp32=True)[0]['z']==[-2**31,2**31-1]


def test_fp32_ieee_nonfinite_and_math_do_not_recurse():
    src='pipeline p { input x : fp32[4] input b : fp32[4] y=div(x,b) e=exp(x) output y output e }'
    got,_=run(src,[[0.,1.,-1.,float('nan')],[0.,0.,0.,1.]],emulate_fp32=True)
    assert math.isnan(got['y'][0]) and got['y'][1]==math.inf and got['y'][2]==-math.inf
    assert math.isnan(got['e'][3])


def test_cli_emulation_is_explicit(tmp_path,capsys):
    path=tmp_path/'p.fgpu';path.write_text('pipeline p { input x : fp32[3] y=sum_axis(x,-1) output y }')
    assert main([str(path),'--emit','reference','--emulate-fp32'])==0
    assert 'class _F32' in capsys.readouterr().out
    assert main([str(path),'--emit','triton','--emulate-fp32'])==2


# ---------------------------------------------------------------------------
# 13.6 store rotation: a trailing rotation chain on a row-wise anchor's
# output is folded into the sweep's store address, not materialized.
# ---------------------------------------------------------------------------


def test_axis_form_costs_one_rowwise_kernel_and_one_slot():
    """The *_axis form must not pay more than the canonical hand-written form.

    Leading rotations fold into loads, trailing rotations fold into the
    store address, so softmax_axis(x,0) lowers to exactly what
    move_first_last -> softmax -> move_first_last lowers to.
    """
    for ranks in ([(2, 3, 4), 0], [(2, 3, 4), 1], [(4, 8), 0]):
        shape, axis = ranks
        dims = ','.join(str(d) for d in shape)
        p = compile_source(
            f'pipeline p {{ input x : fp32[{dims}] y=softmax_axis(x,{axis}) output y }}'
        ).pipelines[0]
        rowwise = [k for k in p.triton.kernels if k.kind == 'rowwise']
        assert len(p.triton.kernels) == 1 and rowwise, \
            f'{shape} axis {axis}: {[k.kind for k in p.triton.kernels]}'
        assert len(p.memory.slots) == 1
        canonical = compile_source(
            f'pipeline p {{ input x : fp32[{dims}] '
            f'r=move_first_last(x) s=softmax(r) y=move_first_last(s) output y }}'
        ).pipelines[0]
        assert len(canonical.triton.kernels) == len(p.triton.kernels)
        assert len(canonical.memory.slots) == len(p.memory.slots)


@pytest.mark.parametrize('shape,axis', [
    ((2, 3, 4), 0), ((2, 3, 4), 1), ((2, 3, 4), 2),
    ((4, 2, 3), 0), ((2, 3, 5, 7), 0), ((2, 3, 5, 7), 1), ((5, 7), 1),
])
def test_store_rotation_matches_an_independent_inverse_map_oracle(shape, axis):
    """End-to-end: the folded store permutation equals a brute-force oracle.

    The oracle never uses the compiler's index algebra: it applies
    softmax over the axis' coordinates directly, which is the definition
    the DSL must preserve no matter how the sweep addresses memory.
    """
    flat = [(i * 7 % 13 - 6) / 4 for i in range(math.prod(shape))]
    src = (f'pipeline p {{ input x : fp32[{",".join(map(str, shape))}] '
           f'y=softmax_axis(x,{axis}) output y }}')
    got, p = run(src, [flat])
    assert len(p.triton.kernels) == 1

    def coordinates(linear):
        out, rest = [], math.prod(shape)
        for dim in shape:
            rest //= dim
            out.append(linear // rest % dim)
        return out

    expected = []
    for point in range(math.prod(shape)):
        coordinate = coordinates(point)
        values = []
        for v in range(shape[axis]):
            probe = list(coordinate)
            probe[axis] = v
            linear = 0
            for index, dim in zip(probe, shape):
                linear = linear * dim + index
            values.append(flat[linear])
        largest = max(values)
        exponentials = [math.exp(v - largest) for v in values]
        expected.append(exponentials[coordinate[axis]] / sum(exponentials))
    assert got['y'] == pytest.approx(expected, abs=1e-6)


def test_double_output_rotation_chain_composes_in_one_kernel():
    """A two-link rotation chain (rank-3 axis-1 form) stays one sweep.

    Each link's inverse map composes in chain order; three kernels plus two
    buffers would mean the fold silently stopped at the first link.
    """
    p = compile_source(
        'pipeline p { input x : fp32[2,3,4] y=softmax_axis(x,1) output y }'
    ).pipelines[0]
    assert [k.kind for k in p.triton.kernels] == ['rowwise']
    text = reference.module([p.reference])
    body = text[text.find('def p_k0'):text.find('def run_')]
    # The store address applies two rotation links, so the generated index
    # nests two map applications.
    assert body.count('_moved') >= 2 or body.count('store rotation') >= 2 \
        or body.count('%') >= 4


TASK_SOURCE = '''pipeline pool_fp32 {
 input kv : fp32[B,K,D]
 input scores : fp32[B,K,D]
 v = to_fp32(kv)
 s = to_fp32(scores)
 w = softmax_axis(s,1)
 p = mul(v,w)
 acc = sum_axis(p,1)
 out = cast_like(acc,kv)
 output out
}'''


def test_cast_inside_a_rotation_chain_recomputes_in_the_sweep():
    """The fused_softmax_pool form: casts and the multiply join the sweeps.

    A cast or same-shape elementwise op between rotations is per-element
    math at one mapped address, so both sweeps recompute it (13.7's
    prologue contract) instead of materializing it: the whole pipeline is
    two sweeps plus the one genuinely-shared cast of kv.
    """
    p = compile_source(
        TASK_SOURCE, shape_bindings={'B': 2, 'K': 8, 'D': 64}
    ).pipelines[0]
    assert len(p.triton.kernels) == 3
    assert len(p.memory.slots) == 3
    assert p.triton.refusals == ()
    assert p.reference.refusals == ()
    # Both sweeps carry their mixed chains: the softmax sweep folds the
    # cast of scores, the sum sweep folds the multiply of v and w.
    sweep_domains = [g for g in p.fusion.groups if g.reduction_domain]
    assert len(sweep_domains) == 2
    assert any(n.op == 'to_fp32' for n in sweep_domains[0].members)
    assert any(n.op == 'mul' for n in sweep_domains[1].members)


def test_pool_task_matches_an_independent_elementwise_oracle():
    """End-to-end numeric oracle for the folded pool form, reference side."""
    import random
    random.seed(11)
    B, K, D = 2, 4, 3
    kv = [round(random.uniform(-2, 2), 3) for _ in range(B * K * D)]
    sc = [round(random.uniform(-2, 2), 3) for _ in range(B * K * D)]
    p = compile_source(TASK_SOURCE).pipelines[0]
    ns = {}
    exec(reference.module([p.reference]), ns)
    got = ns['run_pool_fp32'](kv, sc, B, K, D)
    expected = []
    for b in range(B):
        for d in range(D):
            numerator = denominator = 0.0
            for k in range(K):
                row = [sc[(b * K + j) * D + d] for j in range(K)]
                largest = max(row)
                exponentials = [math.exp(v - largest) for v in row]
                weight = exponentials[k] / sum(exponentials)
                numerator += kv[(b * K + k) * D + d] * weight
                denominator += weight
            expected.append(numerator / denominator)
    assert got['out'] == pytest.approx(expected, abs=1e-6)


def test_task_chunked_rowwise_declares_every_chain_dimension():
    """A chunked sweep's rotation maps name dims its indexing never does.

    On a platform with a measured reduction ceiling the symbolic-row form
    lowers to a two-sweep kernel; the folded chain's `_moved(first, rest)`
    maps reference the links' source dimensions (B, K, D), which must be
    kernel parameters. The A800 caught the free-name NameError that a
    bound-shape local run cannot.
    """
    import re
    from compiler.platform_configs import resolve_platform
    config = resolve_platform('cuda')
    p = compile_source(TASK_SOURCE, platform_config=config).pipelines[0]
    text = '\n\n'.join(k.source for k in p.triton.kernels)
    for match in re.finditer(r'def (\w+)\(([^)]*)\):', text):
        name, parameters = match.group(1), [
            part.strip().split(':')[0].strip() for part in match.group(2).split(',')
        ]
        start = match.end()
        following = text.find('\ndef ', start)
        body = text[start: following if following != -1 else len(text)]
        for dimension in ('B', 'K', 'D'):
            if re.search(rf'\b{dimension}\b', body):
                assert dimension in parameters, \
                    f'{name} uses {dimension} but does not declare it'


def test_external_consumer_of_a_chain_interior_forces_materialization():
    """13.7's register contract: a value read outside the sweep gets a slot.

    When `p` is also consumed by an external add, the sum sweep may fold
    only the rotations; the multiply keeps its kernel and buffer.
    """
    p = compile_source(
        '''pipeline external_p {
 input kv : fp32[B,K,D]
 input scores : fp32[B,K,D]
 v = to_fp32(kv)
 s = to_fp32(scores)
 w = softmax_axis(s,1)
 p = mul(v,w)
 acc = sum_axis(p,1)
 out = cast_like(acc,kv)
 dumped = add(p,kv)
 output out
 output dumped
}''', shape_bindings={'B': 2, 'K': 8, 'D': 64}
    ).pipelines[0]
    assert p.triton.refusals == ()
    assert p.reference.refusals == ()
    multiply = [n for n in p.dataflow.nodes if n.op == 'mul'][0]
    # The multiply's result is materialized: it owns a slot.
    assert any(multiply.output in slot.sources
               for slot in p.memory.slots if hasattr(slot, 'sources')) or \
        len(p.memory.slots) > 3
    # And the sum sweep folded only the rotation links, not the multiply.
    sum_group = next(g for g in p.fusion.groups
                     if any(n.op == 'sum_last' for n in g.members))
    assert not any(n.op == 'mul' for n in sum_group.members)
    assert sum(1 for n in sum_group.members
               if n.op == 'move_first_last') == 2


def test_indexed_reduction_store_tail_uses_inverse_map():
    """A800-caught Triton codegen bug the reference suite cannot catch.

    The indexed (FG_ROWS/FG_COLUMNS) reduction kernel folds the store
    rotation tail into the sweep: the accumulator is per-row, so the
    result must be written at the inverse-mapped ROW index, not at the
    domain's row*width position. A previous version stored the tail at
    the domain linear index, producing correct-looking small shapes by
    luck and a constant-output failure at (1, 257, 33).

    Structurally: the `out` store in the sum_last kernel must use a
    `store_rotation` address (the inverse of the move_first_last link),
    masked by `row_valid`, and must NOT live inside the column loop
    (a per-row result written once, not once per column chunk).
    """
    import re
    from compiler.platform_configs import resolve_platform
    config = resolve_platform('cuda')
    p = compile_source(TASK_SOURCE, platform_config=config).pipelines[0]
    text = '\n\n'.join(k.source for k in p.triton.kernels)
    # The sum_last kernel stores `out` (the only result that is not a row).
    sum_kernel = next(
        k for k in text.split('\n\n')
        if 'tl.sum(' in k and 'out_ptr' in k
    )
    # Extract every tl.store line that writes out_ptr.
    stores = re.findall(
        r'tl\.store\(out_ptr \+ ([^,]+), ([^,]+), mask=(\w+)\)',
        sum_kernel,
    )
    assert stores, 'sum_last kernel must store out_ptr at least once'
    for address, value, mask in stores:
        # The store address must be the composed inverse map, not the raw
        # full_index / domain linear position.
        assert 'store_rotation' in address or '%' in address, \
            f'out store writes to raw index {address!r}; the store-rotation ' \
            f'tail must be inverse-mapped'
        # A per-row result is masked by row validity, not column validity.
        assert mask == 'row_valid', \
            f'out store masked by {mask!r}; a per-row tail store must use ' \
            f'row_valid so it writes once per row, not once per column chunk'
    # The tail store must be OUTSIDE the column loop (indent 4, not 8):
    # verify no tail store line begins with 8 spaces.
    for match in re.finditer(
        r'tl\.store\(out_ptr \+ store_rotation', sum_kernel
    ):
        prefix = sum_kernel[:match.start()].rsplit('\n', 1)[-1]
        assert not prefix.startswith('        '), \
            'the tail store must be emitted after the column loop, not inside it'
