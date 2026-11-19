"""Bounded execution IR for a closed max -> independent sums -> epilogue.

This plan owns stage order, scratch and guards before target emission. It
does not recognize task names or choose a latency winner. FP32 sum regrouping
requires numerical validation; unsupported graphs fail the explicit opt-in.
"""
from dataclasses import dataclass

from flowgpu.compiler.shared import Names, group_operands, group_results, product


@dataclass(frozen=True)
class ScratchValue:
    name: str
    elements: str
    writer: int
    readers: tuple
    dtype: str = 'fp32'

    @property
    def size(self):
        return f'4 * ({self.elements})'


@dataclass(frozen=True)
class ReductionStage:
    kind: str
    inputs: tuple
    outputs: tuple
    depends_on: tuple
    grid: str


@dataclass(frozen=True)
class ReductionStagePlan:
    stages: tuple
    scratch: tuple
    base_domain: tuple
    wide_domain: tuple
    r_tile: str
    d_tile: str
    segments: str
    d_tiles: str
    base_sources: tuple
    guards: tuple
    merge: object = None

    def validate(self, graph):
        if self.merge is not None:
            if (self.merge.exp_op not in {'exp','exp2'}
                    or any(v not in graph.values for v in (self.merge.score,self.merge.maximum,self.merge.weights,self.merge.scalar_sum,self.merge.wide_sum))
                    or graph.producer_of(self.merge.weights).op!=self.merge.exp_op):
                raise ValueError('invalid shifted-exponential merge proof')
        expected=('local','merge') if self.merge is not None else ('max','partials','finish')
        if tuple(s.kind for s in self.stages)!=expected:
            raise ValueError('invalid segmented stage order')
        names={s.name for s in self.scratch}
        if len(names)!=(4 if self.merge is not None else 3) or names & set(graph.values):
            raise ValueError('stage scratch names must be unique and private')
        ready={v for v in graph.values if graph.producer_of(v).op=='input'}
        for index,stage in enumerate(self.stages):
            if not set(stage.inputs)<=ready or any(type(d) is not int or not 0<=d<index for d in stage.depends_on):
                raise ValueError('stage reads an unavailable value or forward dependency')
            if set(stage.outputs)&ready:
                raise ValueError('stage output has more than one writer')
            ready.update(stage.outputs)
        for scratch in self.scratch:
            if (type(scratch.writer) is not int or not 0<=scratch.writer<len(self.stages)
                    or scratch.dtype!='fp32' or scratch.name not in self.stages[scratch.writer].outputs
                    or not scratch.readers or any(scratch.name not in self.stages[i].inputs
                                                 or i<=scratch.writer for i in scratch.readers
                                                 if type(i) is int and 0<=i<len(self.stages))
                    or any(type(i) is not int or not 0<=i<len(self.stages) for i in scratch.readers)):
                raise ValueError('scratch lifetime disagrees with stage dataflow')
        if any(graph.resolve(n.inputs[0]) not in ready for n in graph.nodes if n.op=='output'):
            raise ValueError('stage plan does not produce every declared output')


def plan_reduction_stages(graph, fusion, *, merge=False):
    if len(fusion.groups)!=1 or not fusion.groups[0].cross_domain:
        raise ValueError('segmented strategy requires one closed cross-domain reduction region')
    group=fusion.groups[0]
    reductions=[n for n in group.members if n.op in {'sum_last','max_last'}]
    base=[n for n in reductions if n is not group.anchor]
    if (len(base)!=2 or base[0].op!='max_last' or base[1].op!='sum_last'
            or group.anchor.op!='sum_last'
            or not any(base[1].output in batch and group.anchor.output in batch
                       for batch in group.reduction_batches)):
        raise ValueError('segmented strategy needs max then independent scalar/wide FP32 sums')
    if any(graph.values[n.output].dtype!='fp32' or graph.values[n.inputs[0]].dtype!='fp32'
           for n in reductions):
        raise ValueError('segmented strategy requires FP32 reduction boundaries')
    produced={n.output:n for n in group.members}
    def sources(value, stops=()):
        if value in stops:return set()
        node=produced.get(value)
        if node is None:return {value}
        if node.op in {'ones_like','zeros_like','neg_inf_like'}:return set()
        args=node.inputs[:1] if node.op in {'reshape','cast_like','cast_to','to_fp32','broadcast_like','move_first_last'} else node.inputs
        return set().union(*(sources(v,stops) for v in args))
    results=group_results(graph,group)
    if any(sources(v,{n.output for n in reductions}) for v in results):
        raise ValueError('segmented epilogue must not read original payloads')
    operands=group_operands(graph,group)
    max_sources=sources(base[0].inputs[0])
    scalar_sources=sources(base[1].inputs[0],{base[0].output})
    wide_sources=sources(group.anchor.inputs[0],{base[0].output})
    proof=None
    if merge:
        from flowgpu.compiler.ir.reduction_algebra import prove_shifted_exp_merge
        proof=prove_shifted_exp_merge(graph,group)
    names=Names(graph)
    mx,den,num=(names.fresh('fg_segment_'+s) for s in ('max','den','num'))
    r,d=str(group.cross_domain[-1]),str(group.reduction_domain[-2])
    rows,columns=group.cross_tile
    rt=f'min({rows}, triton.next_power_of_2({r}))'
    dt=f'min({columns}, triton.next_power_of_2({d}))'
    # ceil(R/rows) equals ceil(R/min(rows,next_pow2(R))) for positive R.
    # Polynomial integer expressions keep scratch accounting target-independent.
    segments=f'(({r} + {rows-1}) // {rows})'
    tiles=f'(({d} + {columns-1}) // {columns})'
    outer=product(group.cross_domain[:-1]);small=f'({outer}) * ({segments})'
    scratch=(ScratchValue(mx,small,0,(1,2)),ScratchValue(den,small,1,(2,)),
             ScratchValue(num,f'({small}) * ({d})',1,(2,)))
    staged=(ReductionStage('max',tuple(v for v in operands if v in max_sources),(mx,),(),f'({outer}, {segments})'),
            ReductionStage('partials',(mx,)+tuple(v for v in operands if v in scalar_sources|wide_sources),(den,num),(0,),f'({outer}, {tiles}, {segments})'),
            ReductionStage('finish',(mx,den,num),results,(0,1),f'({outer}, {tiles})'))
    if proof is not None:
        risk=names.fresh('fg_segment_risk')
        scratch=(ScratchValue(mx,small,0,(1,)),ScratchValue(den,small,0,(1,)),
                 ScratchValue(num,f'({small}) * ({d})',0,(1,)),
                 ScratchValue(risk,f'({small}) * ({d})',0,(1,)))
        staged=(ReductionStage('local',tuple(operands),tuple(s.name for s in scratch),(),f'({outer}, {tiles}, {segments})'),
                ReductionStage('merge',tuple(s.name for s in scratch)+tuple(operands),results,(0,),f'({outer}, {tiles})'))
    guards=((f'triton.next_power_of_2({segments}) > 256','segmented reduction supports at most256 partial rows'),
            (f'{tiles} > 65535','segmented expanded-axis grid exceeds65535'),
            (f'triton.next_power_of_2({segments}) * ({dt}) > 32768','segmented finalize tile exceeds32768 lanes'),
            (' + '.join(s.size for s in scratch)+' > 64*1024*1024','segmented workspace exceeds64MiB'))
    plan=ReductionStagePlan(staged,scratch,tuple(group.cross_domain),tuple(group.reduction_domain),
                            rt,dt,segments,tiles,tuple(v for v in operands if v in max_sources|scalar_sources),guards,proof)
    plan.validate(graph)
    return plan
