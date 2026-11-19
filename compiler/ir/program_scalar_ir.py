"""Program-uniform scalar: one device load per program, folded into a base.

Task153's hand-written kernel makes the shape this IR names:

    seq  = tl.load(chunk_indices_ptr + c * 2)
    bos  = tl.load(cu_seqlens_ptr + seq)
    base = bos + chunk_id * BT
    tl.store(out_ptr + base * HK + t * HK + arange(BLOCK), ...)

Every value above is **one number per program**, not one per lane: the load
address is uniform, so the device performs a single scalar load and every lane
sees the same value. The subsequent address is `scalar_base + affine(lane)`,
which is the only write shape 昆仑芯/华为 were measured to sustain (Task152
0.03x and Task154 0.02-0.05x were the shapes that could not be written that
way; the shipped Task147/150 kernels, all affine, read 6.28x/2.01x there).

This module is the IR contract, and it decides the four semantics questions in
one place instead of leaving them to be discovered per lowering:

1. **Bounds are fail-closed.** The default ``bounds="reject"`` admits only an
   index the *host* can check before launch: a literal, a program-id expression
   (the grid bounds it), or a host scalar argument (the value is known at the
   call). The emitted host raises ``ValueError`` when an index reaches past the
   table, before any allocation or launch. An index that is itself a device
   scalar (`UniformScalarRef`, e.g. `cu_seqlens[seq]`) cannot be checked by the
   host; it requires the caller to say ``bounds="trusted"``, and that is a
   documented contract, not a safety claim.
2. **The index is a scalar, never a block.** A rank-0 value only. There is no
   per-lane form: that is the whole distinction this category exists for, and a
   per-lane value here would be a gather, not a uniform load.
3. **Integer dtypes only.** A table of `int32`/`int64`, read as the declared
   integer dtype and used as an address term. A floating index is a cast the
   language does not make implicitly.
4. **The program domain is defined by the lowering, not by this module.** The
   value is meaningless outside a kernel whose launch mapping defines "one
   program"; the staged consumer (the operator registered after the compiler
   side lands) owns that definition. That is why this is an IR contract plus an
   emission helper rather than a user-visible operator yet.

The address contract that consumes the value is `AffineAddress`: `base` may be
a program-uniform scalar (data as a base is the point), and each term is a
lane index or a program-uniform scalar with a *constant* coefficient. Division,
modulo and data-derived coefficients cannot be constructed here -- a lowering
that needs those is a lowering the measured targets refuse, and
`analysis/write_address` refuses it a second time on the emitted text.
"""
import keyword
from dataclasses import dataclass

PROGRAM_SCALAR_DTYPES = ("int32", "int64")
BOUNDS_POLICIES = ("reject", "trusted")

def _identifier(name, what):
    # `isidentifier()` is true for Python keywords, which are not usable as
    # generated variable names (`class`, `return`, ...); the emitters write the
    # name verbatim, so the check has to reject them here (review P2,
    # 2026-10-26).
    if (type(name) is not str or not name or not name.isidentifier()
            or keyword.iskeyword(name)):
        raise ValueError(f"{what} must be a non-empty, non-keyword identifier")


@dataclass(frozen=True)
class ProgramIndex:
    """A program-id-derived integer index: ``pid * scale + offset``.

    Uniform across the lanes of one program, so an address term built from it
    is a scalar. ``scale`` and ``offset`` are static so the host can bound the
    largest value the expression can take from the launch geometry.
    """

    axis: int = 0
    scale: int = 1
    offset: int = 0

    def __post_init__(self):
        if type(self.axis) is not int or self.axis < 0:
            raise ValueError("program index axis must be a non-negative integer")
        if type(self.scale) is not int or self.scale <= 0:
            raise ValueError("program index scale must be a positive integer")
        if type(self.offset) is not int or self.offset < 0:
            raise ValueError("program index offset must be a non-negative integer")

    @property
    def expression(self):
        """The emitted scalar expression inside the kernel."""
        term = f"tl.program_id({self.axis})"
        if self.scale != 1:
            term = f"({term}) * {self.scale}"
        if self.offset:
            term = f"({term}) + {self.offset}"
        return term

    def maximum(self, grid_count):
        """The largest index this expression can take for a launch of that grid."""
        return f"(({grid_count}) - 1) * {self.scale} + {self.offset}"

    def as_dict(self):
        return dict(kind="program", axis=self.axis, scale=self.scale,
                    offset=self.offset)


@dataclass(frozen=True)
class ScalarRef:
    """A rank-0 value the host passes (or another program scalar produced)."""

    name: str
    host_visible: bool = True

    def __post_init__(self):
        _identifier(self.name, "scalar index value")

    def as_dict(self):
        return dict(kind="scalar", name=self.name, host_visible=self.host_visible)


@dataclass(frozen=True)
class ProgramScalar:
    """One integer scalar read per program from a 1-D device table.

    ``table`` is a graph value name of a rank-1 integer tensor. ``index`` is a
    `ProgramIndex`, a literal int, or a `ScalarRef`. The result is rank 0 and
    program-uniform; every lane of the program reads the same number.
    """

    table: str
    index: ProgramIndex | int | ScalarRef
    dtype: str = "int64"
    bounds: str = "reject"

    def __post_init__(self):
        _identifier(self.table, "program scalar table")
        if self.dtype not in PROGRAM_SCALAR_DTYPES:
            raise ValueError(
                f"program scalar dtype must be int32 or int64, got {self.dtype!r}"
            )
        if self.bounds not in BOUNDS_POLICIES:
            raise ValueError(
                f"program scalar bounds must be reject or trusted, got "
                f"{self.bounds!r}"
            )
        if type(self.index) is int:
            if self.index < 0:
                raise ValueError("a literal program scalar index must be non-negative")
        elif isinstance(self.index, ScalarRef):
            # A host scalar can be checked at launch; a device scalar cannot.
            if not self.index.host_visible and self.bounds != "trusted":
                raise ValueError(
                    f"program scalar index '{self.index.name}' is itself a "
                    f"device value, so the host cannot prove it is in bounds; "
                    f"pass bounds='trusted' to accept the caller's contract"
                )
        elif not isinstance(self.index, ProgramIndex):
            raise ValueError(
                "program scalar index must be ProgramIndex, a literal int, or "
                "ScalarRef"
            )

    @property
    def program_uniform(self):
        return True

    @property
    def host_checkable(self):
        """Whether the host can prove the index is inside the table.

        `bounds="trusted"` is the *opposite* case: the contract says the caller
        guarantees the bound, so no proof exists and no host check is emitted.
        The old implementation returned True for it, which read as "the host can
        check this" while `backend/program_scalar.host_guards` deliberately
        emitted nothing (review P2, 2026-10-26).
        """
        if isinstance(self.index, (int, ProgramIndex)):
            return True
        return self.index.host_visible

    def as_dict(self):
        index = self.index if isinstance(self.index, int) else self.index.as_dict()
        return dict(table=self.table, index=index, dtype=self.dtype,
                    bounds=self.bounds)


@dataclass(frozen=True)
class LaneIndex:
    """One per-lane coordinate of the kernel's index domain."""

    name: str = "lane"

    def __post_init__(self):
        _identifier(self.name, "lane index name")


@dataclass(frozen=True)
class AddressTerm:
    """``coefficient * index`` with a static or constexpr coefficient."""

    coefficient: int | str
    index: LaneIndex | ProgramIndex

    def __post_init__(self):
        if type(self.coefficient) is int:
            pass
        elif type(self.coefficient) is str and self.coefficient.isidentifier():
            # A named constexpr/stride symbol; a *value* name would be a
            # data-derived coefficient, which this contract does not admit.
            pass
        else:
            raise ValueError(
                "address term coefficient must be an int or an identifier"
            )
        if not isinstance(self.index, (LaneIndex, ProgramIndex)):
            raise ValueError(
                "address term index must be a LaneIndex or ProgramIndex; a "
                "data-derived coefficient or a scalar scale factor is not an "
                "affine address term"
            )

    @property
    def uniform(self):
        return isinstance(self.index, ProgramIndex)


@dataclass(frozen=True)
class AffineAddress:
    """``base + sum(coefficient * index)``: the only write shape this contract admits.

    ``base`` may be a `ProgramScalar` (data folded into the base -- the reason
    the category exists) or a literal. Terms are per-lane coordinates or
    program-uniform indices with constant coefficients. Constructing a division,
    a modulo or a data-scaled lane term is not possible: such a lowering uses
    `analysis/write_address` to see its verdict on the emitted text instead.
    """

    label: str
    base: ProgramScalar | int | None = None
    terms: tuple = ()

    def __post_init__(self):
        _identifier(self.label, "address label")
        if self.base is not None and type(self.base) is not int \
                and not isinstance(self.base, ProgramScalar):
            raise ValueError(
                "affine address base must be a ProgramScalar or an int"
            )
        object.__setattr__(self, "terms", tuple(self.terms))
        for term in self.terms:
            if not isinstance(term, AddressTerm):
                raise ValueError("affine address terms must be AddressTerm")
        if self.base is None and not self.terms:
            raise ValueError("affine address needs a base or at least one term")

    def expression(self):
        """The emitted offset expression (without the pointer)."""
        parts = []
        if isinstance(self.base, ProgramScalar):
            parts.append(f"({self.base.table}_scalar)")
        elif isinstance(self.base, int):
            parts.append(str(self.base))
        for term in self.terms:
            index = term.index.expression if isinstance(term.index, ProgramIndex) \
                else term.index.name
            if str(term.coefficient) == "1":
                parts.append(f"({index})")
            else:
                parts.append(f"({term.coefficient}) * ({index})")
        return " + ".join(parts)

    def as_dict(self):
        return dict(
            label=self.label,
            base=self.base.as_dict() if isinstance(self.base, ProgramScalar)
            else self.base,
            terms=[dict(coefficient=term.coefficient,
                        index=(term.index.expression
                               if isinstance(term.index, ProgramIndex)
                               else term.index.name),
                        uniform=term.uniform) for term in self.terms],
        )


#: The staged internal operator name and the program-index expression its
#: lowering binds the table with. This is the integration point the user-visible
#: operator connects to: when the frontend registers it (the step after this
#: compiler-side work), the rule and the lowering read the same mapping, so they
#: cannot disagree about what `index` means. The name is intentionally not in
#: `compiler/frontend/operator.py`: the user-visible spelling is that step's
#: decision, and keeping it out leaves every default registry unchanged.
STAGED_PROGRAM_SCALAR_OP = "program_scalar"
PROGRAM_SCALAR_INDEX = {
    STAGED_PROGRAM_SCALAR_OP: ProgramIndex(axis=0),
}
