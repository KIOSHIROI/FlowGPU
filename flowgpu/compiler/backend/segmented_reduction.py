"""Render the staged execution IR using the shared cross-domain emitter."""
import ast
from dataclasses import replace


def lower_stages(graph, group, memory, name, config):
    from flowgpu.compiler.backend.cross_domain_backend import triton_kernel
    plan=memory.stage_plan
    plan.validate(graph)
    kernels=[]
    original_inputs={v for v in graph.values if graph.producer_of(v).op=='input'}
    for stage in plan.stages:
        kernel=triton_kernel(graph,group,memory,name+'_'+stage.kind,config,
                             segmentation=plan,stage=stage.kind)
        # Finish's IR contract forbids payload rereads; verify the actual body
        # too, rather than trusting broad graph-level operand metadata.
        if stage.kind=='finish':
            pointers={formal.split(':')[0] for formal,arg in zip(kernel.parameters,kernel.launch)
                      if arg.kind=='tensor' and arg.text in original_inputs}
            for node in ast.walk(ast.parse(kernel.source)):
                if isinstance(node,ast.Call) and ast.unparse(node.func)=='tl.load':
                    if pointers & {v.id for v in ast.walk(node.args[0]) if isinstance(v,ast.Name)}:
                        raise ValueError('segmented finish cannot read original payloads')
        kernels.append(replace(kernel,operands=stage.inputs,results=stage.outputs,grid=stage.grid))
    return tuple(kernels)
