"""Is this kernel's write address `scalar_base + affine(arange)`?

The two slow backends in the corpus (昆仑芯, 华为) sustain an affine write: a
store whose per-lane address is the scalar base plus a linear function of the
lane index. Task152 read 0.03x there and Task154 0.02-0.05x; both were found on
the official evaluator, not here. This module is the check that turns that
measurement into a compile-time verdict: it reads the **emitted kernel source**
and classifies every `tl.store` address *before the kernel reaches a caller*.

Two dimensions, and they are deliberately separate (the same distinction
``backend/contracts.py`` makes for reads):

* **lane division/modulo** -- an address whose per-lane part is reached through
  ``//`` or ``%`` of a value that depends on ``tl.arange``. A program-id
  division is not this: a program id is uniform across the lanes of one
  program, so ``(pid // TILES) * W`` is a scalar coefficient. Renaming a
  scoped statement *is* this class, and it is the 152 shape.
* **data-dependent lane term** -- an address built from a value that came out
  of ``tl.load`` **per lane** (a gather/scatter address). A program-uniform
  scalar load is not this: its result is one number per program, so
  ``base + idx`` is affine. That distinction is what the program-uniform
  scalar primitive exists to give the language.

The check is a static judgement on generated text, not a device model. It says
nothing about tile shapes, masks or speed; it only refuses the *shape* of an
address that a measured target collapses on. It is gated by
``PlatformConfig.affine_write_addresses`` (True only where that behaviour was
measured), so the bytes of every platform that does not carry the record are
untouched.
"""
import ast
import re

from compiler.core.diagnostic import CompileError, DiagnosticKind

AFFINE = "affine"
LANE_DIVMOD = "lane_divmod"
LANE_LOAD = "lane_load"
LANE_COEFFICIENT = "lane_coefficient"
NON_AFFINE = "non_affine"


def definitions(source):
    """Single-target ``name = expression`` lines, last assignment winning.

    Emitted kernels define every index term on its own line before the address
    that uses it, so an address reads as a small system of definitions. Same
    convention as ``backend/contracts.py``; kept local because this module is an
    analysis of the address *contract*, not a second consumer of that file's
    access-shape judgements.
    """
    table = {}
    for raw in source.splitlines():
        line = raw.strip()
        if "=" not in line or "==" in line:
            continue
        if line.startswith(("#", "if ", "for ", "while ", "return ", "with ")):
            continue
        lhs, _, rhs = line.partition("=")
        lhs = lhs.strip()
        if lhs.isidentifier():
            parsed = _parse(rhs.strip())
            if parsed is not None:
                table[lhs] = parsed
    return table


def _parse(expression):
    try:
        return ast.parse(expression, mode="eval").body
    except SyntaxError:
        return None


def _call_name(node):
    """The dotted/attribute name of a call, e.g. ``tl.load`` or ``x.to``."""
    function = node.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        base = function.value
        while isinstance(base, ast.Attribute):
            base = base.value
        if isinstance(base, ast.Name):
            return f"{base.id}.{function.attr}"
    return ""


def _tokens(node):
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}


class _Verdict:
    """What one subexpression is, for the address contract.

    ``lane`` means the value varies across the lanes of one program.
    ``loaded`` means the value came out of memory (data, not index arithmetic).
    ``detail`` is the first reason this subexpression is not affine, when it is
    not; it is built from facts (operator, operand text) so a message can name
    the offending part.
    """

    __slots__ = ("lane", "loaded", "affine", "kind", "lane_names", "detail",
                 "reduced")

    def __init__(self, lane=False, loaded=False, affine=True, kind=AFFINE,
                 lane_names=frozenset(), detail="", reduced=False):
        self.lane = lane
        self.loaded = loaded
        self.affine = affine
        self.kind = kind
        self.lane_names = frozenset(lane_names)
        self.detail = detail
        self.reduced = reduced

    @property
    def bad(self):
        return self.kind != AFFINE


def _uniform(loaded=False):
    return _Verdict(lane=False, loaded=loaded, affine=True)


def _classification(node, table, cache):
    key = id(node)
    if key in cache:
        return cache[key]
    verdict = _classify(node, table, cache)
    cache[key] = verdict
    return verdict


def _classify(node, table, cache):
    if isinstance(node, ast.Constant):
        return _uniform()

    if isinstance(node, ast.Name):
        if node.id in table:
            verdict = _classification(table[node.id], table, cache)
            if verdict.lane:
                # Name the variable the source actually wrote; that is what a
                # message can point a reader at, not the `tl.arange` inside its
                # definition.
                return _Verdict(lane=True, loaded=verdict.loaded,
                                affine=verdict.affine, kind=verdict.kind,
                                lane_names=frozenset({node.id}),
                                detail=verdict.detail, reduced=verdict.reduced)
            return verdict
        # A kernel formal that is not defined in the body: a pointer base, a
        # scalar input, or a constexpr. None of them varies with the lane, and
        # none of them is data. Pointer names are the address base and are
        # stripped before classification by `_address_expression`; a pointer
        # that appears inside the offset is still uniform arithmetic here.
        return _uniform()

    if isinstance(node, ast.Attribute):
        return _classification(node.value, table, cache)

    if isinstance(node, ast.Subscript):
        # ``row[:, None]`` / ``x[i]``: a slice or a scalar index preserves the
        # dependence of the value being indexed. (An index *into* a lane vector
        # by a lane value would be a gather, but the emitters never write that
        # in an address.)
        value = _classification(node.value, table, cache)
        if node.slice is not None:
            index = _classification(node.slice, table, cache)
            if index.lane and not value.lane:
                return _Verdict(lane=True, loaded=value.loaded, affine=False,
                                kind=NON_AFFINE,
                                lane_names=index.lane_names,
                                detail="lane value used as an index")
        return value

    if isinstance(node, ast.Call):
        name = _call_name(node)
        # The receiver of a method call is an operand: `tl.load(...).to(tl.int32)`
        # is a load, and `value.to(...)` inherits `value`'s dependence. Keywords
        # are operands too -- a per-lane `mask=` makes even a uniform-address
        # load a per-lane value.
        operands = list(node.args) + [keyword.value for keyword in node.keywords]
        if (isinstance(node.func, ast.Attribute)
                and not (isinstance(node.func.value, ast.Name)
                         and node.func.value.id in {"tl", "triton"})):
            operands.append(node.func.value)
        arguments = [_classification(argument, table, cache)
                     for argument in operands]
        lane = any(argument.lane for argument in arguments)
        loaded = any(argument.loaded for argument in arguments)
        names = frozenset().union(*(argument.lane_names for argument in arguments)) \
            if arguments else frozenset()
        if name.endswith("tl.arange"):
            return _Verdict(lane=True, loaded=False, affine=True,
                            lane_names=frozenset(), detail="")
        if name.endswith("tl.load"):
            # A load with a uniform address/mask is one value per program (the
            # program-uniform scalar this contract permits as a base); a load
            # with any lane-dependent argument is a per-lane value from memory.
            if lane:
                return _Verdict(lane=True, loaded=True, affine=False,
                                kind=LANE_LOAD, lane_names=names,
                                detail="per-lane `tl.load`")
            return _Verdict(lane=False, loaded=True, affine=True,
                            kind=AFFINE, lane_names=frozenset(),
                            detail="uniform `tl.load`")
        if name.endswith("tl.program_id") or name.endswith("tl.num_programs"):
            return _uniform()
        if any(name.endswith(reducer) for reducer in ("tl.sum", "tl.max",
                                                      "tl.min", "tl.reduce")):
            # A reduction over a lane value is one value per program again.
            return _Verdict(lane=False, loaded=loaded, affine=True,
                            reduced=True)
        # ``.to(...)``, ``tl.cdiv``, ``int(...)``, ``min``/``max`` -- the
        # dependence of the result is the dependence of its arguments, and its
        # verdict is the first bad argument's verdict.
        bad = next((argument for argument in arguments if argument.bad), None)
        return _Verdict(lane=lane, loaded=loaded,
                        affine=all(a.affine for a in arguments),
                        kind=bad.kind if bad is not None else AFFINE,
                        lane_names=names,
                        detail=bad.detail if bad is not None else "")

    if isinstance(node, ast.BinOp):
        left = _classification(node.left, table, cache)
        right = _classification(node.right, table, cache)
        lane = left.lane or right.lane
        loaded = left.loaded or right.loaded
        names = left.lane_names | right.lane_names
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
            operator = {ast.Div: "/", ast.FloorDiv: "//", ast.Mod: "%"}[type(node.op)]
            if lane:
                lane_side = left if left.lane else right
                return _Verdict(lane=True, loaded=loaded, affine=False,
                                kind=LANE_DIVMOD,
                                lane_names=lane_side.lane_names,
                                detail=f"`{operator}` applied to lane term "
                                       f"'{_render(node.left if left.lane else node.right)}'")
            return _Verdict(lane=False, loaded=loaded,
                            affine=left.affine and right.affine,
                            lane_names=frozenset(),
                            detail=left.detail or right.detail)
        if isinstance(node.op, (ast.Add, ast.Sub)):
            return _Verdict(lane=lane, loaded=loaded,
                            affine=left.affine and right.affine,
                            kind=left.kind if left.bad else right.kind,
                            lane_names=names,
                            detail=left.detail or right.detail)
        if isinstance(node.op, ast.Mult):
            if left.lane and right.lane:
                return _Verdict(lane=True, loaded=loaded, affine=False,
                                kind=NON_AFFINE, lane_names=names,
                                detail="product of two lane-dependent terms")
            if left.lane and right.loaded and not right.lane:
                return _Verdict(lane=True, loaded=True, affine=False,
                                kind=LANE_COEFFICIENT, lane_names=left.lane_names,
                                detail=f"uniform load '{_render(node.right)}' "
                                       f"scales a lane term, not a scalar base")
            if right.lane and left.loaded and not left.lane:
                return _Verdict(lane=True, loaded=True, affine=False,
                                kind=LANE_COEFFICIENT, lane_names=right.lane_names,
                                detail=f"uniform load '{_render(node.left)}' "
                                       f"scales a lane term, not a scalar base")
            return _Verdict(lane=lane, loaded=loaded,
                            affine=left.affine and right.affine,
                            kind=left.kind if left.bad else right.kind,
                            lane_names=names,
                            detail=left.detail or right.detail)
        return _Verdict(lane=lane, loaded=loaded, affine=False,
                        kind=left.kind if left.bad else right.kind,
                        lane_names=names,
                        detail=left.detail or f"operator "
                                              f"`{type(node.op).__name__}` in a "
                                              f"lane-dependent address")

    if isinstance(node, ast.UnaryOp):
        operand = _classification(node.operand, table, cache)
        return operand

    if isinstance(node, ast.IfExp):
        body = _classification(node.body, table, cache)
        orelse = _classification(node.orelse, table, cache)
        return _Verdict(lane=body.lane or orelse.lane,
                        loaded=body.loaded or orelse.loaded,
                        affine=body.affine and orelse.affine,
                        kind=body.kind if body.bad else orelse.kind,
                        lane_names=body.lane_names | orelse.lane_names,
                        detail=body.detail or orelse.detail)

    if isinstance(node, (ast.Compare, ast.BoolOp)):
        operands = ([node.left] + node.comparators if isinstance(node, ast.Compare)
                    else node.values)
        verdicts = [_classification(operand, table, cache)
                    for operand in operands]
        lane = any(verdict.lane for verdict in verdicts)
        loaded = any(verdict.loaded for verdict in verdicts)
        return _Verdict(lane=lane, loaded=loaded, affine=lane,
                        lane_names=frozenset().union(
                            *(verdict.lane_names for verdict in verdicts)),
                        detail=next((v.detail for v in verdicts if v.detail), ""))

    return _Verdict(lane=True, loaded=True, affine=False, kind=NON_AFFINE,
                    detail=f"unsupported address node {type(node).__name__}")


def _render(node):
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover - ast.unparse is always available on 3.9+
        return "<expression>"


def _address_expression(source, call):
    """The first argument of a ``tl.store(...)`` call, paren-balanced."""
    inner = source[call[1] + 1:call[2] - 1]
    depth = 0
    for index, char in enumerate(inner):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            return inner[:index].strip()
    return inner.strip()


def _calls(source, name):
    """Yield ``(start, open, close)`` spans of every ``name(...)`` call."""
    prefix = name + "("
    start = 0
    while True:
        at = source.find(prefix, start)
        if at < 0:
            return
        open_at = at + len(name)
        depth = 0
        index = open_at
        while index < len(source):
            char = source[index]
            if char in "([{":
                depth += 1
            elif char in ")]}":
                depth -= 1
                if depth == 0:
                    break
            index += 1
        yield at, open_at, index
        start = index + 1


class _Binding:
    __slots__ = ("name", "dependency", "root")

    def __init__(self, name, dependency, root):
        self.name = name
        self.dependency = dependency
        self.root = root


def _address_badness(expression, table, cache):
    """Walk the address for the first *contract* violation."""
    node = _parse(expression)
    if node is None:
        return None
    verdict = _classification(node, table, cache)
    if not verdict.bad:
        return None
    return verdict


def store_addresses(source):
    """Classify every ``tl.store`` address in one emitted kernel.

    Returns `StoreAddress` records in source order. An empty tuple means every
    write is ``scalar_base + affine(arange)`` -- runs, blocked tiles and
    program-id divisions included, because those are affine in the lane and the
    measured targets accept them.
    """
    table = definitions(source)
    cache = {}
    found = []
    for call in _calls(source, "tl.store"):
        expression = _address_expression(source, call)
        line = source.count("\n", 0, call[0]) + 1
        node = _parse(expression)
        verdict = _classification(node, table, cache) if node is not None \
            else _Verdict(affine=False, kind=NON_AFFINE, detail="unparseable address")
        target = _render(node.left) if isinstance(node, ast.BinOp) and \
            isinstance(node.op, ast.Add) else expression
        root = _address_root(expression, table, cache)
        found.append(StoreAddress(
            target=target,
            expression=expression,
            line=line,
            verdict=verdict.kind if verdict.bad else AFFINE,
            lane=", ".join(sorted(verdict.lane_names)) or root,
            detail=verdict.detail,
            uniform_load=_has_uniform_load(node, table, cache),
        ))
    return tuple(found)


def _has_uniform_load(node, table, cache):
    """Whether the address folds in a program-uniform (scalar) device load.

    That is the *permitted* data dependence: one number per program used as a
    base. It is reported so a test can tell "data folded into the base" apart
    from "no data at all" without re-parsing the address.
    """
    if node is None:
        return False
    for child in ast.walk(node):
        if not isinstance(child, ast.Name) or child.id not in table:
            continue
        verdict = _classification(table[child.id], table, cache)
        if verdict.loaded and not verdict.lane:
            return True
    return False


def _address_root(expression, table, cache):
    """The lane root(s) the address depends on, named as the source names them."""
    node = _parse(expression)
    if node is None:
        return ""
    names = _tokens(node)
    roots = []
    for name in names:
        if name not in table:
            continue
        definition = table[name]
        if any(isinstance(child, ast.Call)
               and _call_name(child).endswith("tl.arange")
               for child in ast.walk(definition)):
            roots.append(name)
    return ", ".join(sorted(roots))


class StoreAddress:
    """One emitted store address and the verdict the contract gives it."""

    __slots__ = ("target", "expression", "line", "verdict", "lane", "detail",
                 "uniform_load")

    def __init__(self, target, expression, line, verdict, lane, detail,
                 uniform_load=False):
        self.target = target
        self.expression = expression
        self.line = line
        self.verdict = verdict
        self.lane = lane
        self.detail = detail
        self.uniform_load = uniform_load

    @property
    def affine(self):
        return self.verdict == AFFINE

    def __repr__(self):  # pragma: no cover - debugging aid
        return (f"StoreAddress({self.target!r}, {self.verdict}, "
                f"lane={self.lane!r}, line={self.line})")


def violations(source, kernel_name="kernel"):
    """Human-readable contract violations in one kernel's write addresses.

    Each message names the address, the lane dependency it was decoded from and
    the line, because "this kernel is not affine" is not actionable: the point
    of the check is to tell a lowering *which* store and *which* lane term it
    has to turn into a scalar base.
    """
    found = []
    for address in store_addresses(source):
        if address.affine:
            continue
        lane = address.lane or "per-lane value"
        if address.verdict == LANE_DIVMOD:
            detail = (f"the per-lane part is reached through "
                      f"{address.detail}")
            hint = "decode the offset once per program (a program-uniform "                    "scalar base), not per lane"
        elif address.verdict == LANE_LOAD:
            detail = (f"the address is data-dependent per lane "
                      f"({address.detail})")
            hint = ("a program-uniform scalar (one load per program) must be "
                    "folded into the base; a per-lane load makes the store a "
                    "scatter")
        elif address.verdict == LANE_COEFFICIENT:
            detail = address.detail
            hint = "a data-derived scalar is an additive base, not a per-lane "                    "scale factor"
        else:
            detail = address.detail or "the lane terms do not compose additively"
            hint = "the write must be scalar_base + affine(arange)"
        found.append(
            f"{kernel_name}: store address '{address.expression}' at line "
            f"{address.line} is not affine in the lane ({detail}; lane "
            f"dependency: {lane}); {hint}. This target sustains only "
            f"scalar_base + affine(arange) writes"
        )
    return found


def require_affine_writes(kernel, config, pipeline=None):
    """Refuse a kernel whose write address this target cannot sustain.

    Runs over the final emitted source, i.e. before the kernel is handed to a
    caller (after ABI pruning, so the text is exactly what would be launched).
    Enabled only where the platform record says affine writes are required, so
    a target that does not carry that measurement is byte-for-byte unaffected.
    """
    if not config.affine_write_addresses:
        return kernel
    problems = violations(kernel.source, kernel.name)
    if not problems:
        return kernel
    raise CompileError(
        DiagnosticKind.OPERATOR,
        "affine write address required: " + "; ".join(problems),
        pipeline=pipeline,
    )


def audit(programs):
    """Report-only: every non-affine store address in a lowered program.

    Used to state the evidence in the operator-loop document and to keep the
    check honest: a corpus scan runs it over every emitted kernel of the
    examples and reports what it finds, without refusing anything.
    """
    report = []
    for program in programs:
        for kernel in program.kernels:
            for address in store_addresses(kernel.source):
                if not address.affine:
                    report.append(dict(kernel=kernel.name, kind=kernel.kind,
                                       verdict=address.verdict,
                                       lane=address.lane, line=address.line,
                                       expression=address.expression,
                                       detail=address.detail))
    return report
