"""Prove a shift-equivariant pair of FP32 exponential reductions.

The proof concerns finite real arithmetic, not IEEE associativity. Emission
must guard exceptional local statistics and replay the original expression.
No program or variable names participate in recognition.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ShiftedExpMerge:
    score: str
    maximum: str
    weights: str
    scalar_sum: str
    wide_sum: str
    exp_op: str


def prove_shifted_exp_merge(graph, group):
    nodes={n.output:n for n in group.members}
    reductions=[n for n in group.members if n.op in {'sum_last','max_last'}]
    maximum,scalar=[n for n in reductions if n is not group.anchor]
    wide=group.anchor

    def node(value,op=None):
        result=nodes.get(value)
        if result is None or (op is not None and result.op!=op):
            raise ValueError('two-stage merge requires exp(score - max(score)) and linear weighted sum')
        if graph.values[value].dtype!=('bool' if result.op=='isfinite' else 'fp32'):
            raise ValueError('two-stage merge cannot cross a low-precision boundary')
        return result

    weights=node(scalar.inputs[0])
    if weights.op not in {'exp','exp2'}:
        raise ValueError('two-stage merge scalar sum must reduce exponential weights')
    shifted=node(weights.inputs[0],'sub')
    if shifted.inputs[0]!=maximum.inputs[0]:
        raise ValueError('two-stage merge score must be the actual max input')
    broad=node(shifted.inputs[1],'broadcast_like')
    shift=broad.inputs[0]
    if shift!=maximum.output:
        safe=node(shift,'where');condition=node(safe.inputs[0],'isfinite')
        zero=node(safe.inputs[2],'zeros_like')
        if condition.inputs[0]!=maximum.output or safe.inputs[1]!=maximum.output or zero.inputs[0]!=maximum.output:
            raise ValueError('two-stage merge requires a shift equal to max on finite scores')
    product=node(wide.inputs[0],'mul')
    shape=tuple(group.cross_domain)
    expanded=tuple(group.reduction_domain)

    def expanded_weights(value):
        b=nodes.get(value)
        if b is None or b.op!='broadcast_like' or graph.values[value].dtype!='fp32':return False
        view=nodes.get(b.inputs[0])
        return (view is not None and view.op=='reshape' and view.inputs[0]==weights.output
                and tuple(graph.values[view.output].shape)==shape[:-1]+('1',shape[-1])
                and tuple(graph.values[value].shape)==expanded)

    weight_args=[i for i,v in enumerate(product.inputs) if expanded_weights(v)]
    if len(weight_args)!=1:
        raise ValueError('two-stage merge wide sum must be linear in the same weights')
    payload=product.inputs[1-weight_args[0]]
    forbidden={n.output for n in reductions}|{weights.output}

    def depends(value):
        if value in forbidden:return True
        n=nodes.get(value)
        if n is None or n.op in {'ones_like','zeros_like','neg_inf_like'}:return False
        args=n.inputs[:1] if n.op in {'cast_like','cast_to','reshape','broadcast_like','to_fp32','move_first_last'} else n.inputs
        return any(depends(v) for v in args)

    if depends(payload):
        raise ValueError('two-stage merge payload cannot depend on global reduction statistics')
    return ShiftedExpMerge(maximum.inputs[0],maximum.output,weights.output,
                           scalar.output,wide.output,weights.op)
