"""Small standard-library execution double for the emitted stage programs.

Executes actual emitted expressions, indices, masks and stores on CPU. It is
not Triton's interpreter, a device compiler, or device correctness evidence.
Uninitialized reads, live out-of-bounds accesses and duplicate stores fail.
"""
import ast
import itertools
import math
import struct
import types

#: Floating dtypes the double models with real rounding. bf16 uses round-to-
#: nearest-even on the fp32 bit pattern truncated to 16 bits (8 explicit
#: mantissa bits), because the Task160 precision probe emits actual bf16
#: arithmetic and the validation margin depends on that rounding being modelled
#: rather than approximated by fp32.
FLOAT_DTYPES = frozenset({'fp16', 'bf16', 'fp32'})


def _round_bf16(value):
    if not isinstance(value, float):
        value = float(value)
    if math.isnan(value) or math.isinf(value) or value == 0.0:
        return value
    try:
        bits = struct.unpack('>I', struct.pack('>f', value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)
    # Round the low 16 bits to nearest-even before dropping them.
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack('>f', struct.pack('>I', bits))[0]


def cast(x, dtype):
    if dtype == 'bool':
        return bool(x)
    if dtype in ('int32', 'int64'):
        return int(x)
    if dtype == 'bf16':
        return _round_bf16(x)
    try:
        return struct.unpack('e' if dtype=='fp16' else 'f',
                             struct.pack('e' if dtype=='fp16' else 'f', x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def promoted_dtype(values):
    """IEEE-style promotion between the double's vector dtypes.

    Identical float dtypes keep their dtype (hardware fp16/bf16 arithmetic
    rounds to the narrow format); a mix promotes to fp32; anything without a
    float operand is integer arithmetic.
    """
    floats = [value.dtype for value in values if value.dtype in FLOAT_DTYPES]
    if not floats:
        return 'int64'
    if len(set(floats)) == 1:
        return floats[0]
    return 'fp32'


def indices(shape):
    return itertools.product(*(range(n) for n in shape))


def flat(index, shape):
    result = 0
    for i, n in zip(index, shape):
        result = result*n+i
    return result


class V:
    def __init__(self, data, shape=(), dtype=None):
        self.shape = tuple(shape)
        self.data = list(data) if isinstance(data, (list, tuple)) else [data]
        self.dtype = dtype or ('bool' if all(type(x) is bool for x in self.data)
                               else 'fp32' if any(isinstance(x, float) for x in self.data) else 'int64')
        assert len(self.data)==math.prod(self.shape)
        self.data = [cast(x,self.dtype) for x in self.data]

    def to(self, dtype):
        return V(self.data,self.shape,dtype)

    def at(self, index):
        tail = index[len(index)-len(self.shape):] if self.shape else ()
        return self.data[flat(tuple(0 if n==1 else i for i,n in zip(tail,self.shape)),self.shape)]

    def __neg__(self):
        return V([-value for value in self.data], self.shape, self.dtype)

    def __bool__(self):
        if self.shape:
            raise ValueError('ambiguous CPU vector truth')
        return bool(self.data[0])

    def __index__(self):
        # A program id / program count is a scalar in the emitted source, and an
        # emitted grid-stride loop hands one to Python's `range`. Modelling it
        # as an indexable scalar lets the double execute the loop the device
        # compiler accepts, instead of only the unlooped form.
        if self.shape:
            raise ValueError('only a scalar CPU vector can be an index')
        return int(self.data[0])

    def __getitem__(self, key):
        key = key if isinstance(key,tuple) else (key,)
        assert all(k is None or (isinstance(k,slice) and k==slice(None)) for k in key)
        assert sum(k is not None for k in key)==len(self.shape)
        axis=iter(self.shape)
        return V(self.data,tuple(1 if k is None else next(axis) for k in key),self.dtype)


def vector(x):
    return x if isinstance(x,V) else V(x)


def apply(fn, *values, dtype=None):
    values=[vector(v) for v in values]
    rank=max(len(v.shape) for v in values)
    padded=[(1,)*(rank-len(v.shape))+v.shape for v in values]
    shape=tuple(max(ns) for ns in zip(*padded))
    assert all(all(a==b or a==1 for a,b in zip(p,shape)) for p in padded)
    dtype=dtype or promoted_dtype(values)
    return V([fn(*(v.at(index) for v in values)) for index in indices(shape)],shape,dtype)


def divide(a,b):
    if b==0:
        return math.nan if a==0 else math.copysign(math.inf,a*b if b else a)
    return a/b


for name,fn in dict(__add__=lambda a,b:a+b,__sub__=lambda a,b:a-b,
                    __mul__=lambda a,b:a*b,__floordiv__=lambda a,b:a//b,
                    __mod__=lambda a,b:a%b,__truediv__=divide,
                    __and__=lambda a,b:bool(a)&bool(b),__or__=lambda a,b:bool(a)|bool(b),
                    __eq__=lambda a,b:a==b,__ne__=lambda a,b:a!=b,
                    __lt__=lambda a,b:a<b,__le__=lambda a,b:a<=b,
                    __gt__=lambda a,b:a>b,__ge__=lambda a,b:a>=b).items():
    boolean=name in {'__and__','__or__','__eq__','__ne__','__lt__','__le__','__gt__','__ge__'}
    setattr(V,name,lambda self,other,fn=fn,boolean=boolean:apply(fn,self,other,dtype='bool' if boolean else None))
    if name in ('__add__','__sub__','__mul__','__floordiv__','__mod__','__truediv__','__and__','__or__'):
        setattr(V,'__r'+name[2:],lambda self,other,fn=fn,boolean=boolean:apply(fn,other,self,dtype='bool' if boolean else None))


class DType(str):
    """A dtype name that also answers the emitters' `ptr.dtype.element_ty`."""
    @property
    def element_ty(self):
        return str(self)


class Buffer:
    def __init__(self, data, shape, dtype, strides=None):
        self.data=list(data);self.shape=tuple(shape);self.ndim=len(shape)
        self.dtype=DType(dtype)
        self.strides=tuple(strides or [math.prod(shape[i+1:]) for i in range(len(shape))])
        self.device='cpu-emulator';self.layout='strided';self.is_cuda=True
        self.writes=set()

    def stride(self,axis=None):
        return self.strides if axis is None else self.strides[axis]

    def is_contiguous(self):
        return self.strides==tuple(math.prod(self.shape[i+1:]) for i in range(self.ndim))

    def __add__(self, offset):
        return Address(self,vector(offset))


class Address:
    def __init__(self,buffer,offset):
        self.buffer=buffer;self.offset=offset

    @property
    def dtype(self):
        return self.buffer.dtype

    def __add__(self,offset):
        return Address(self.buffer,self.offset+offset)


#: Hints the emitted source may pass to `tl.load`/`tl.store`. They are policy
#: hints to the device cache hierarchy and cannot change values, so the double
#: accepts and ignores them (a typo in a spelling is still rejected).
LOAD_HINTS = frozenset({'eviction_policy'})
STORE_HINTS = frozenset({'eviction_policy', 'cache_modifier'})


def load(address,mask=True,other=0,**hints):
    assert set(hints) <= LOAD_HINTS, ('unsupported tl.load hint', sorted(hints))
    b=address.buffer
    def get(i,m):
        if not m:return other
        assert 0<=i<len(b.data),('out-of-bounds CPU load',i,len(b.data))
        assert b.data[i] is not None,('uninitialized CPU read',i)
        return b.data[i]
    return apply(get,address.offset,mask,dtype=b.dtype)


def store(address,value,mask=True,**hints):
    assert set(hints) <= STORE_HINTS, ('unsupported tl.store hint', sorted(hints))
    b=address.buffer
    def put(i,x,m):
        if m:
            assert 0<=i<len(b.data),('out-of-bounds CPU store',i,len(b.data))
            assert i not in b.writes,('duplicate CPU store',i)
            b.writes.add(i);b.data[i]=cast(x,b.dtype)
        return 0
    apply(put,address.offset,value,mask)


def reduce(value,axis,maximum=False):
    value=vector(value);shape=value.shape[:axis]+value.shape[axis+1:];out=[]
    for index in indices(shape):
        xs=[value.at(index[:axis]+(i,)+index[axis:]) for i in range(value.shape[axis])]
        if maximum:
            out.append(math.nan if any(math.isnan(x) for x in xs) else max(xs))
        else:
            acc=cast(0,value.dtype)
            for x in xs:acc=cast(acc+x,value.dtype)
            out.append(acc)
    return V(out,shape,value.dtype)


class TL:
    constexpr=None;float16='fp16';bfloat16='bf16';float32='fp32';int64='int64';int32='int32';int1='bool'
    def __init__(self):self.pid=(0,0,0);self.grid=(0,0,0)
    def program_id(self,axis):return V(self.pid[axis])
    def num_programs(self,axis):return V(self.grid[axis])
    def arange(self,a,b):return V(list(range(a,b)),(b-a,))
    def full(self,shape,value,dtype):return V([value]*math.prod(shape),shape,dtype)
    def zeros(self,shape,dtype):return V([0]*math.prod(shape),shape,dtype)
    def sum(self,value,axis):return reduce(value,axis)
    def max(self,value,axis):return reduce(value,axis,True)
    def abs(self,value):return apply(abs,value)
    def maximum(self,a,b):return apply(max,a,b)
    def where(self,c,a,b):
        return apply(lambda c,a,b:a if c else b,c,a,b,dtype='fp32' if vector(a).dtype.startswith('fp') or vector(b).dtype.startswith('fp') else None)
    def exp(self,x):return apply(lambda a:math.exp(a),x)
    def exp2(self,x):
        # The device saturates to inf instead of raising, so a lane where the
        # fp32 base-2 exponential overflows must be executable here too.
        def evaluate(a):
            try:
                return 2.**a
            except OverflowError:
                return math.inf
        return apply(evaluate,x)
    def log(self,x):return apply(lambda a:math.log(a) if a>0 else -math.inf if a==0 else math.nan,x)
    def log2(self,x):return apply(lambda a:math.log2(a) if a>0 else -math.inf if a==0 else math.nan,x)
    load=staticmethod(load);store=staticmethod(store)
    cdiv=staticmethod(lambda a,b:-(-a//b))


class Triton:
    def __init__(self,tl):
        self.tl=tl
        self.launches=[]
        #: Successful launch records `(kernel, grid, positional arguments)`.
        #: `launches` keeps its historical two-tuple shape; the arguments are a
        #: separate record so a test can assert on a constexpr dispatch value.
        self.arguments=[]
    def jit(self,fn):
        runtime=self
        class Kernel:
            def __getitem__(self,grid):
                grid=grid if isinstance(grid,tuple) else (grid,)
                def call(*args,**kwargs):
                    runtime.launches.append((fn.__name__,grid))
                    runtime.arguments.append((fn.__name__,grid,args))
                    runtime.tl.grid=grid+(0,)*(3-len(grid))
                    for index in indices(grid):
                        runtime.tl.pid=index+(0,)*(3-len(index));fn(*args)
                return call
        return Kernel()
    cdiv=staticmethod(lambda a,b:-(-a//b))
    next_power_of_2=staticmethod(lambda a:1<<(a-1).bit_length())


def execute(code, inputs, pipeline='lse'):
    tl=TL();triton=Triton(tl);allocations=[]
    def empty(shape,*,dtype,device):
        if isinstance(shape,int):shape=(shape,)
        buffer=Buffer([None]*math.prod(shape),shape,dtype);allocations.append(buffer);return buffer
    # The emitted host validates `dtype is torch.<dtype>` and allocates outputs
    # with `torch.empty(..., dtype=torch.<dtype>)`, so every dtype the emitters
    # name has to exist on the double's torch namespace.
    torch=types.SimpleNamespace(empty=empty,empty_like=lambda x:empty(x.shape,dtype=x.dtype,device=x.device),
                               strided='strided',
                               float16='fp16',float32='fp32',bfloat16='bf16',
                               int32='int32',int64='int64',bool='bool')
    tree=ast.parse(code)
    tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef)
               and n.name!='_flowgpu_device_key']
    ns=dict(tl=tl,triton=triton,torch=torch)
    exec(compile(ast.fix_missing_locations(tree),'<emitted-cpu-double>','exec'),ns)
    result=ns['run_'+pipeline](*inputs)
    for buffer in allocations:
        assert len(buffer.writes)==len(buffer.data),('unwritten CPU allocation',buffer.shape)
    return {key:buffer.data for key,buffer in result.items()},triton.launches


import random

def inputs(shape,distribution):
    r,b,h,d=shape;randomizer=random.Random(7421+r+d)
    values=[cast(randomizer.uniform(-2,2),'fp16') for _ in range(r*b*h*d)]
    lse=[cast(randomizer.uniform(-8,8)*(20 if distribution=='peaky' else 1),'fp32') for _ in range(r*b*h)]
    if distribution=='mixed-invalid':
        for bi in range(b):
            for hi in range(h):
                lse[bi*h+hi]=math.nan;lse[((r-1)*b+bi)*h+hi]=math.inf
        for ri in range(r):lse[ri*b*h]=-math.inf
    xpad=[math.nan]*(r*b*h*(d+2));lpad=[math.nan]*(r*b*(h+2))
    for ri in range(r):
        for bi in range(b):
            for hi in range(h):
                lpad[(ri*b+bi)*(h+2)+hi+1]=lse[(ri*b+bi)*h+hi]
                for di in range(d):xpad[((ri*b+bi)*h+hi)*(d+2)+di+1]=values[((ri*b+bi)*h+hi)*d+di]
    # Shift the view's origin, retaining poison padding between logical rows.
    x=Buffer(xpad[1:],shape,'fp16',(b*h*(d+2),h*(d+2),d+2,1))
    l=Buffer(lpad[1:],(r,b,h),'fp32',(b*(h+2),h+2,1))
    return (x,l),values,lse


def oracle(shape,values,lse,base2=False):
    r,b,h,d=shape;out=[];stat=[]
    for row in range(b*h):
        scores=[lse[ri*b*h+row] if math.isfinite(lse[ri*b*h+row]) else -math.inf for ri in range(r)]
        maximum=max(scores);maximum=maximum if math.isfinite(maximum) else 0.
        weights=[2.**(a-maximum) if base2 else math.exp(a-maximum) for a in scores]
        den=sum(weights)
        stat.append((math.log2(den) if base2 else math.log(den))+maximum if den else -math.inf)
        for di in range(d):
            numerator=sum(values[(ri*b*h+row)*d+di]*weights[ri] for ri in range(r))
            out.append(cast(numerator/den,'fp16') if den else math.nan)
    return dict(out=out,out_lse=stat)


def check(got,want):
    assert set(got)==set(want)
    for name in got:
        assert len(got[name])==len(want[name])
        for g,w in zip(got[name],want[name]):
            if math.isnan(w):assert math.isnan(g)
            elif math.isinf(w):assert g==w
            else:assert abs(g-w)<=.003+(.003 if name=='out' else 1e-5)*abs(w),(name,g,w)

