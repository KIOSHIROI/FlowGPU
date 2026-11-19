"""Explicit reusable buffers for exact-context, native sequential deployments.

No tensor contents or addresses are cached. Callers own buffers and must use
separate sets for overlapping streams/calls; a returned output aliases its set.
The ordinary allocating API retains its existing ownership semantics.
"""
import ast
import hashlib
import math
from dataclasses import dataclass, asdict


WIDTHS = {'torch.float16':2, 'torch.bfloat16':2, 'torch.float32':4,
          'torch.int32':4, 'torch.int64':8, 'torch.bool':1}


@dataclass(frozen=True)
class BufferRequirement:
    name: str
    shape: tuple
    dtype: str
    outputs: tuple

    @property
    def nbytes(self):
        return math.prod(self.shape)*WIDTHS[self.dtype]


@dataclass(frozen=True)
class ExecutionPlan:
    pipeline: str
    inputs: tuple
    buffers: tuple
    outputs: tuple
    launches: tuple
    original_sha256: str
    source_sha256: str
    code: str

    def as_dict(self):
        return dict(pipeline=self.pipeline, inputs=list(self.inputs),
            buffers=[dict(**asdict(b),nbytes=b.nbytes) for b in self.buffers],
            outputs=dict(self.outputs), launches=list(self.launches),
            buffer_bytes=sum(b.nbytes for b in self.buffers),
            original_sha256=self.original_sha256, source_sha256=self.source_sha256,
            ownership='caller-owned; outputs alias buffers; distinct sets for concurrent streams')


def _check_buffers(buffers, specs, inputs):
    # Kept for previously emitted plans. New plans specialize these checks.
    import torch
    if type(buffers) is not dict or set(buffers) != {s[0] for s in specs}:
        raise ValueError('execution plan requires exactly its named buffers')
    intervals = []
    def interval(t):
        # Conservative envelope also handles padded inputs and zero strides.
        span=1+sum((n-1)*s for n,s in zip(t.shape,t.stride()))
        return t.data_ptr(),t.data_ptr()+span*t.element_size()
    for t in inputs:
        intervals.append(interval(t))
    for name,shape,dtype in specs:
        t=buffers[name]
        if (not isinstance(t,torch.Tensor) or t.layout!=torch.strided
                or tuple(t.shape)!=shape or str(t.dtype)!=dtype
                or t.device!=inputs[0].device or not t.is_contiguous()
                or t.data_ptr()%128):
            raise ValueError('execution plan buffer metadata/alignment mismatch: '+name)
        low,high=interval(t)
        if any(low < end and start < high for start,end in intervals):
            raise ValueError('execution plan writable buffers must not overlap inputs or other buffers')
        intervals.append((low,high))


def _guarded_input_spans(host, inputs, fresh):
    """Use only the canonical exact guards, never unverified caller metadata.

    A pointer local is assigned inside the existing short-circuit guard. It is
    read only after that guard succeeds and is refreshed on every invocation.
    """
    pos=int(bool(host.body and isinstance(host.body[0],ast.Expr)
                 and isinstance(host.body[0].value,ast.Constant)
                 and isinstance(host.body[0].value.value,str)))
    intervals=[]
    for name,guard in zip(inputs,host.body[pos:pos+len(inputs)]):
        try:
            terms=guard.test.values
            shape,stride,dtype,device,alignment=(ast.literal_eval(terms[i].comparators[0]) for i in range(1,6))
            if (type(shape) is not tuple or type(stride) is not tuple or len(shape)!=len(stride)
                    or any(type(n) is not int or n<=0 for n in shape)
                    or any(type(s) is not int or s<0 for s in stride)
                    or dtype not in WIDTHS or type(device) is not str
                    or type(alignment) is not int or not 0<=alignment<128):
                raise ValueError('unsupported input span')
            expected=ast.parse(
                f"if ({name}.layout != torch.strided or tuple({name}.shape) != {shape!r} or "
                f"tuple({name}.stride()) != {stride!r} or str({name}.dtype) != {dtype!r} or "
                f"str({name}.device) != {device!r} or {name}.data_ptr() % 128 != {alignment!r}):\n"
                "    raise ValueError('prepared input context changed; prepare again')").body[0]
            if ast.dump(guard)!=ast.dump(expected):raise ValueError('noncanonical input guard')
        except (AttributeError,IndexError,TypeError,ValueError):
            raise ValueError('execution plan requires canonical exact input guards') from None
        pointer=fresh('_flowgpu_input_ptr')
        call=terms[-1].left.left
        terms[-1].left.left=ast.NamedExpr(ast.Name(pointer,ast.Store()),call)
        span=(1+sum((n-1)*s for n,s in zip(shape,stride)))*WIDTHS[dtype]
        intervals.append((pointer,span))
    if len(intervals)!=len(inputs) or not inputs:
        raise ValueError('execution plan requires guarded tensor inputs')
    return intervals


def _buffer_checks(parameter, allocations, inputs, intervals, fresh):
    """Emit fixed-layout checks; no cached pointers, tensors or validation result."""
    # Deterministic across Python hash seeds as well as repeated construction.
    names='{'+', '.join(repr(b.name) for _,b in allocations)+'}' if allocations else 'set()'
    statements=ast.parse(
        f"if type({parameter}) is not dict or {parameter}.keys() != {names}:\n"
        "    raise ValueError('execution plan requires exactly its named buffers')").body
    device=fresh('_flowgpu_buffer_device')
    statements+=ast.parse(f'{device} = {inputs[0]}.device').body
    for _,b in allocations:
        pointer=fresh('_flowgpu_buffer_ptr')
        statements+=ast.parse(
            f"{b.name} = {parameter}[{b.name!r}]\n"
            f"if (not isinstance({b.name}, torch.Tensor) or {b.name}.layout != torch.strided or "
            f"tuple({b.name}.shape) != {b.shape!r} or str({b.name}.dtype) != {b.dtype!r} or "
            f"{b.name}.device != {device} or not {b.name}.is_contiguous() or "
            f"({pointer} := {b.name}.data_ptr()) % 128):\n"
            f"    raise ValueError('execution plan buffer metadata/alignment mismatch: {b.name}')").body
        # Input envelopes include padding/zero strides. Output guards prove a
        # contiguous exact-size tensor, so its span is the static byte count.
        overlap=' or '.join(f'({pointer} < {p}+{span} and {p} < {pointer}+{b.nbytes})'
                            for p,span in intervals)
        statements+=ast.parse(f"if {overlap}:\n"
            "    raise ValueError('execution plan writable buffers must not overlap inputs or other buffers')").body
        intervals.append((pointer,b.nbytes))
    return statements


def build_execution_plan(code, *, pipeline):
    """Derive bindings from a prepared host, never reconstruct its launch math."""
    tree=ast.parse(code)
    from compiler.optimization.strategy import launch_value
    constants={}
    for n in tree.body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
            try:value=launch_value(ast.unparse(n.value),constants)
            except (ValueError,TypeError,ZeroDivisionError):continue
            if type(value) is int:constants[n.targets[0].id]=value
    hosts=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name.startswith('run_')]
    if len(hosts)!=1 or hosts[0].name!='run_'+pipeline:
        raise ValueError('one prepared named pipeline required')
    host=hosts[0]
    if 'prepared input context changed' not in code or 'prepared device/software context changed' not in code:
        raise ValueError('execution plan requires exact-context prepared guards')
    if any(isinstance(n,ast.With) for n in ast.walk(host)):
        raise ValueError('execution plan requires sequential native kernels')
    inputs=tuple(a.arg for a in host.args.args)
    returns=[n for n in host.body if isinstance(n,ast.Return)]
    if len(returns)!=1 or not isinstance(returns[0].value,ast.Dict):
        raise ValueError('execution plan requires named tensor outputs')
    result=returns[0].value
    if any(not isinstance(k,ast.Constant) or not isinstance(k.value,str) for k in result.keys) or any(
            not isinstance(v,ast.Name) for v in result.values):
        raise ValueError('execution plan requires direct tensor outputs')
    outputs=tuple((k.value,v.id) for k,v in zip(result.keys,result.values))
    allocations=[]
    for statement in host.body:
        if (isinstance(statement,ast.Assign) and isinstance(statement.value,ast.Call)
                and ast.unparse(statement.value.func)=='torch.empty'):
            if len(statement.targets)!=1 or not isinstance(statement.targets[0],ast.Name):
                raise ValueError('unsupported allocation binding')
            call=statement.value;kw={k.arg:k.value for k in call.keywords}
            if len(call.args)!=1 or set(kw)!={'dtype','device'}:
                raise ValueError('unsupported allocation contract')
            try:
                literal=ast.literal_eval(call.args[0])
                shape=(literal,) if type(literal) is int else tuple(literal)
            except (ValueError,TypeError):raise ValueError('buffer dimensions must be prepared constants')
            dtype=ast.unparse(kw['dtype'])
            if dtype not in WIDTHS or any(type(n) is not int or n<=0 for n in shape):
                raise ValueError('unsupported prepared buffer type')
            name=statement.targets[0].id
            allocations.append((statement,BufferRequirement(name,shape,dtype,tuple(k for k,v in outputs if v==name))))
    jit_names={n.name for n in tree.body if isinstance(n,ast.FunctionDef)
               and any(ast.unparse(d)=='triton.jit' for d in n.decorator_list)}
    launches=[]
    metadata_methods={'data_ptr','stride','is_contiguous'}
    for n in ast.walk(host):
        if not isinstance(n,ast.Call):continue
        if isinstance(n.func,ast.Subscript):
            if not isinstance(n.func.value,ast.Name) or n.func.value.id not in jit_names:
                raise ValueError('execution plan requires fixed native JIT launches')
            try:grid=launch_value(ast.unparse(n.func.slice),constants)
            except (ValueError,TypeError):raise ValueError('launch grid must be prepared constants')
            n.func.slice=ast.parse(repr(grid),mode='eval').body
            launches.append(dict(kernel=n.func.value.id,grid=grid))
        else:
            name=ast.unparse(n.func)
            # `abs` is the 32-bit address-range guard's runtime stride term.
            if name in {'torch.empty','tuple','str','ValueError','_flowgpu_device_key','abs'}:continue
            if isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.attr in metadata_methods:continue
            raise ValueError('unsupported host operation in execution plan: '+name)
    if sum(isinstance(n,ast.Call) and ast.unparse(n.func)=='torch.empty' for n in ast.walk(host))!=len(allocations):
        raise ValueError('nested allocations are not supported by execution plans')
    all_names={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}|set(inputs)
    def fresh(stem):
        while stem in all_names:
            stem = stem + '_'
        all_names.add(stem)
        return stem
    parameter=fresh('_flowgpu_buffers')
    intervals=_guarded_input_spans(host,inputs,fresh)
    host.name='run_into_'+pipeline
    host.args.args.append(ast.arg(arg=parameter))
    checks=_buffer_checks(parameter,allocations,inputs,intervals,fresh)
    # Every semantic/input guard remains before the first buffer binding/launch.
    position=next((i for i,n in enumerate(host.body) if any(n is a for a,_ in allocations)
                   or isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
                   and isinstance(n.value.func,ast.Subscript)),len(host.body)-1)
    host.body[position:position]=checks
    host.body=[n for n in host.body if not any(n is a for a,_ in allocations)]
    fixed=ast.unparse(ast.fix_missing_locations(tree))+'\n'
    return ExecutionPlan(pipeline,inputs,tuple(b for _,b in allocations),outputs,tuple(launches),
        hashlib.sha256(code.encode()).hexdigest(),hashlib.sha256(fixed.encode()).hexdigest(),fixed)


@dataclass
class PreparedExecutionPlan:
    descriptor: ExecutionPlan
    record: dict
    _run: object
    device: str

    def allocate_buffers(self):
        import torch
        if torch.cuda.is_current_stream_capturing():
            raise ValueError('allocate execution plan buffers before capture')
        return {b.name:torch.empty(b.shape,dtype=getattr(torch,b.dtype.split('.')[-1]),device=self.device)
                for b in self.descriptor.buffers}

    def run_into(self,*args,buffers):
        if len(args)!=len(self.descriptor.inputs):raise ValueError('execution plan input count mismatch')
        return self._run(*args,buffers)


def prepare_execution_plan(source, *, pipeline, args, cache_dir, validator, **options):
    import torch
    from pathlib import Path
    from compiler.runtime.preparation import prepare_pipeline
    from compiler.runtime.runtime import _load_generated_module
    _,record=prepare_pipeline(source,pipeline=pipeline,args=args,cache_dir=cache_dir,validator=validator,**options)
    code=Path(record['source_path']).read_text()
    if hashlib.sha256(code.encode()).hexdigest()!=record['source_sha256']:
        raise ValueError('prepared source changed before execution plan construction')
    descriptor=build_execution_plan(code,pipeline=pipeline)
    loaded=_load_generated_module(descriptor.code,cache_dir=Path(cache_dir)/'execution-plans',key=descriptor.source_sha256)
    plan=PreparedExecutionPlan(descriptor,dict(preparation=record,execution_plan=descriptor.as_dict()),
        getattr(loaded,'run_into_'+pipeline),str(args[0].device))
    buffers=plan.allocate_buffers()
    result=plan.run_into(*args,buffers=buffers)
    torch.cuda.synchronize(args[0].device)
    if validator(result) is not True:raise ValueError('independent execution plan validation failed')
    from compiler.runtime.cache import compiler_fingerprint
    if compiler_fingerprint()!=record['identity']['compiler_fingerprint']:
        raise ValueError('compiler changed during execution plan preparation')
    return plan
