"""FlowGPU lowering for lightning-attention decode."""
from flowgpu.compiler.backend.backends import Argument, Names, identifier, group_operands, group_results


def triton_kernel(graph, group, memory, name, config):
    from flowgpu.compiler.backend.triton_backend import TritonKernel
    assert len(group.members) == 2 and all(n.op == 'linear_decode_forward' for n in group.members)
    operands = group_operands(graph, group); results = group_results(graph, group)
    q, k, v, cache, slope, slots = operands; out, cache_out = results
    nms = Names(graph); ptr = {x: nms.fresh(identifier(x) + '_ptr') for x in operands + results}
    pairs = [(ptr[x], Argument('tensor', x)) for x in operands + results]
    def c(h, a):
        z = nms.fresh(h); pairs.append((f'{z}: tl.constexpr', a)); return z
    B = c('B', Argument('shape', f'{q}:0')); H = c('H', Argument('shape', f'{q}:1')); D = c('D', Argument('shape', f'{q}:3'))
    S = c('S', Argument('shape', f'{cache}:0'))
    # BLOCK_SIZE is the *declaration's* literal, not a caller-supplied host
    # parameter: the host used to take it as an argument and a caller passing a
    # different number silently changed the kernel the DSL described
    # (review P1-2/P1-9, 2026-10-26).
    block_value = int(group.anchor.attributes.block_size)
    BLOCK = c('BLOCK', Argument('value', str(block_value)))
    # tl.arange needs a power-of-two length while the declared D is a runtime
    # constexpr; widen the tile and keep `kd < D` as the mask.
    DW = c('DW', Argument('value', f'(1 << ({q}.shape[3] - 1).bit_length())'))
    qb = c('q_b', Argument('stride', f'{q}:0')); qh = c('q_h', Argument('stride', f'{q}:1')); qd = c('q_d', Argument('stride', f'{q}:3')); vd_stride = c('v_d', Argument('stride', f'{v}:3'))
    cb = c('c_b', Argument('stride', f'{cache}:0')); ch = c('c_h', Argument('stride', f'{cache}:1'))
    cd0 = c('c_d0', Argument('stride', f'{cache}:2')); cd1 = c('c_d1', Argument('stride', f'{cache}:3'))
    ob = c('o_b', Argument('stride', f'{out}:0'))
    from flowgpu.compiler.backend.triton_backend import DTYPE_TO_TRITON
    zero_dtype = DTYPE_TO_TRITON[graph.values[out].dtype]
    source = f'''# linear_decode_forward
@triton.jit
def {name}({", ".join(x for x,_ in pairs)}):
    b = tl.program_id(0)
    h = tl.program_id(1)
    db = tl.program_id(2)
    slot = tl.load({ptr[slots]} + b)
    kd = tl.arange(0, {DW})
    vd = db * {BLOCK} + tl.arange(0, {BLOCK})
    qm = kd < {D}
    vm = vd < {D}
    # Padding batches and illegal slots have no destination.  The reference
    # zeroes the whole output and continues for either; the kernel writes the
    # same zero rows instead of returning with uninitialized `out` (padding) or
    # addressing cache with an out-of-range slot (review 2026-10-26).
    if (slot < 0) | (slot >= {S}):
        tl.store({ptr[out]} + b * {ob} + h * {D} + vd,
                 tl.zeros(({BLOCK},), {zero_dtype}), mask=vm)
        return
    qoff = b * {qb} + h * {qh}
    coff = slot * {cb} + h * {ch}
    qv = tl.load({ptr[q]} + qoff + kd * {qd}, mask=qm, other=0.0).to(tl.float32)
    kv = tl.load({ptr[k]} + qoff + kd * {qd}, mask=qm, other=0.0).to(tl.float32)
    vv = tl.load({ptr[v]} + qoff + vd * {vd_stride}, mask=vm, other=0.0).to(tl.float32)
    off = kd[:, None] * {cd0} + vd[None, :] * {cd1}
    mask = qm[:, None] & vm[None, :]
    old_ptr = {ptr[cache]} + coff + off
    new_ptr = {ptr[cache_out]} + coff + off
    state = kv[:, None] * vv[None, :] + tl.exp(-tl.load({ptr[slope]} + h).to(tl.float32)) * tl.load(old_ptr, mask=mask, other=0.0).to(tl.float32)
    result = tl.sum(qv[:, None] * state, axis=0)
    # `cache_out` is a result, so the kernel writes IT and leaves the input
    # tensor untouched.
    tl.store(new_ptr, state, mask=mask)
    tl.store({ptr[out]} + b * {ob} + h * {D} + vd, result, mask=vm)
'''
    return TritonKernel(name, 'linear_decode', tuple(n.op for n in group.members), operands, results,
                        tuple(x for x,_ in pairs), tuple(a for _,a in pairs),
                        f'({q}.shape[0], {q}.shape[1], triton.cdiv({q}.shape[3], {block_value}))', source,
                        launch_options=(('num_warps', config.num_warps),))
