"""Generated scalar FP32 arithmetic emulation; not a GPU instruction simulator.

Every scalar floating operation rounds to nearest-even FP32. Stored fp16/bf16
rounding is unchanged. Reductions remain sequential and transcendentals use
Python libm rounded to FP32; no promise of Triton reduction trees, FMA, TF32,
tensor cores, or hardware exp approximations. This complements the FP64 oracle.
"""

PREAMBLE = '''
# Explicit scalar FP32 emulation, independent of storage precision.
_base_fp32, _base_fp16, _base_bf16 = fp32, fp16, bf16
_base_math = math

class _F32(float):
    def __new__(cls, value):
        return float.__new__(cls, _base_fp32(float(value)))
    def __add__(self, other): return _F32(float(self) + float(other))
    def __radd__(self, other): return _F32(float(other) + float(self))
    def __sub__(self, other): return _F32(float(self) - float(other))
    def __rsub__(self, other): return _F32(float(other) - float(self))
    def __mul__(self, other): return _F32(float(self) * float(other))
    def __rmul__(self, other): return _F32(float(other) * float(self))
    def __truediv__(self, other): return _F32(ieee_div(float(self), float(other)))
    def __rtruediv__(self, other): return _F32(ieee_div(float(other), float(self)))
    def __pow__(self, other): return _F32(float(self) ** float(other))
    def __rpow__(self, other): return _F32(float(other) ** float(self))
    def __neg__(self): return _F32(-float(self))
    def __abs__(self): return _F32(abs(float(self)))

class _FP32Math:
    def __getattr__(self, name):
        value = getattr(_base_math, name)
        if not callable(value): return _F32(value) if isinstance(value,float) else value
        def call(*args, **kwargs):
            result = value(*args, **kwargs)
            return _F32(result) if isinstance(result, float) else result
        return call

math = _FP32Math()
def fp32(value): return _F32(value)
def fp16(value): return _F32(_base_fp16(float(value)))
def bf16(value): return _F32(_base_bf16(float(value)))
'''
