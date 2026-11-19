"""Executable backend code-generation contracts.

`platform_configs` records what each target's compiler is known to honour. This
module turns those records into a check that can be run against *emitted* source,
so a violated contract fails in a test instead of on the vendor's evaluator.

Everything here was established by single-variable experiments on official
evaluations (2026-10-19, Task144), not inferred from documentation:

  昆仑芯   a masked load does not honour ``other``.
          Attempts 6 vs 7 of the same kernel differ only in the load mask and
          flip correctness from 7/7 to Failed; a *store* mask is fine (attempt 6
          shipped one and passed). So on this target a zero border must be owned
          by an allocation, never by ``other=``.

  华为     ``tl.make_block_ptr`` with ``boundary_check`` / ``padding_option``
          does not produce correct values.
          The same 2-D tiled kernel is 55/55 exact on H100 and 2.7%-14.7% wrong
          on 华为 with ordinary-magnitude differences. The substitute that works
          is a 2-D ``tl.arange`` tile with explicit masks.

  昆仑芯   a *gather* load and a *loop-carried store* are both unaffordable.
          Task154's `normalize_and_patchify` (a patch permutation) reads 1.81x on
          国际通用 A and 0.02x on 昆仑芯, and its source-order rewrite only reaches
          0.05x; Task152's `kda_gate_chunk_cumsum` walks a chunk with a running
          accumulator and one store per step, and reads 0.03x there with both a
          runtime loop and a `static_range` unroll.  Row-wise kernels that
          accumulate in registers and store once are fine: 6.28x (Task147).

Each entry exists because the alternative -- discovering it again by burning a
scarce submission -- is what this module is meant to prevent.

Two checks below are adjacent and must not be confused; conflating them is a
mistake this file made once and then paid for by missing two official
observations (`research/kunlun_feasibility/RESULTS.md`):

* ``store_inside_loop_ok`` -- a **temporal** property: a stored value that
  depends on a register the same loop body updates.  There is no address
  computation in it at all; Task152 is this shape.  It is decided structurally
  over the emitted kernel's loop tree (`loop_carried_stores`), because "is the
  store inside a `for`" is the wrong question: Task147/149 stream a wide row
  with a store per chunk and measure 6.28x / 2.01x there.
* ``gather_load_ok`` -- a **spatial** property: an address that is a
  permutation of, or data-dependent in, the lane index.  Task154 is this shape,
  and it has no loop in it at all.  It belongs to ``analysis/access_shape``,
  which classifies addressing from the IR, and this module only reads that
  classification back out of the emitted text.

So the loop-structure judgement below cannot speak about Task154-style permuted
writes, and does not try: `loop_carried_stores` would return nothing for those
kernels even if the addressing were wrong, which is exactly what the second
`RESULTS.md` correction says.  A test pins that silence so nobody re-derives the
wrong conclusion from a clean report.
"""

import ast
import re
import textwrap
from dataclasses import dataclass

from flowgpu.compiler.analysis.dependences import MAY, MUST


def call_spans(source, name):
    """Yield the full text of every ``name(...)`` call, with parens balanced.

    Spans rather than lines: an emitted ``tl.load`` routinely wraps across lines,
    and a line-oriented scan would miss the ``other=`` on a continuation line.
    """
    start = 0
    prefix = name + "("
    while True:
        at = source.find(prefix, start)
        if at < 0:
            return
        depth = 0
        index = at + len(name)
        while index < len(source):
            char = source[index]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    break
            index += 1
        yield source[at:index + 1]
        start = index + 1


def violations(source, platform):
    """Constructs in ``source`` that this platform is known not to honour.

    Returns a list of human-readable strings; empty means the emitted source
    stays inside the measured contract. This is a static check on generated
    text, not a proof: it flags exactly the constructs that were measured to
    break, and nothing else.
    """
    found = []

    if not platform.masked_load_other_honoured:
        # Refined on 2026-10-20: 昆仑芯 does not honour `other=` on the lanes the
        # mask excludes, but that only matters when the loaded value can reach a
        # lane the load did not cover.  A 1-D tail mask whose result is stored
        # under the *same* mask is safe there -- Task147's rmsnorm family does
        # exactly that and measures 6.28x, Task149 2.01x -- while every kernel
        # that failed had a store mask broader than the load mask (Task144
        # attempts 6/7: load `keep`, store `in_range`; Task154 v2: load
        # `in_image`, store `in_patch`).  So the contract is mask identity, not
        # the presence of `other=`.
        store_masks = {_normalise_mask(_keyword_argument(span, "mask"))
                       for span in call_spans(source, "tl.store")}
        store_masks.discard("")
        for call in call_spans(source, "tl.load"):
            if "mask=" not in call or "other=" not in call:
                continue
            load_mask = _normalise_mask(_keyword_argument(call, "mask"))
            if load_mask in store_masks:
                continue
            head = " ".join(call.split())[:96]
            found.append(
                f"{platform.name}: masked load passes `other=` and no store uses "
                f"the same mask, so a value loaded under one predicate can be "
                f"stored under another ({head}...); store under exactly the load "
                f"mask, or own the zero region with an allocation"
            )

    if not platform.block_ptr_reliable:
        for token in ("make_block_ptr", "boundary_check", "padding_option"):
            if token in source:
                found.append(
                    f"{platform.name}: emitted `{token}`, which is not reliable on "
                    f"this target; use a 2-D tl.arange tile with explicit masks"
                )

    if not platform.gather_load_ok:
        # 2026-10-20, second pass: the criterion is about *addressing*, so it
        # applies to writes too. 昆仑芯 collapses a permuted store exactly as it
        # collapses a gather read -- Task154 v3/v4 write through a patch
        # permutation and measure 0.05x, and checking only loads missed them.
        for op, noun, advice in (
            ("tl.load", "load is a gather (a scattered read)",
             "iterate the source densely and move the permutation into the store address"),
            ("tl.store", "store is a scatter (a permuted write)",
             "iterate the destination densely and move the permutation into the load address"),
        ):
            for call in call_spans(source, op):
                offset = _first_argument(call)
                if gather_like(source, offset):
                    head = " ".join(call.split())[:96]
                    found.append(
                        f"{platform.name}: this {noun} -- its address depends on a "
                        f"loaded value, or is a permutation rather than an affine or "
                        f"stride-linear function of the lane index ({head}...); {advice}"
                    )

    if not platform.store_inside_loop_ok:
        for carried in loop_carried_stores(source):
            found.append(
                f"{platform.name}: a `tl.store` whose value carries the register "
                f"`{'`, `'.join(carried.registers)}` across iterations of "
                f"`{carried.loop}` is a "
                f"loop-carried dependence this target cannot sustain "
                f"({carried.certainty}: {carried.store}); accumulate in registers "
                f"and store once"
            )

    return found


def _definitions(source):
    """Single-target `name = expression` lines, last assignment winning.

    Emitted code defines each index term on its own line before the address that
    uses it, so the address reads as a small system of definitions.
    """
    table = {}
    for raw in source.splitlines():
        line = raw.strip()
        if "=" not in line or "==" in line:
            continue
        if line.startswith(("#", "if ", "for ", "while ", "return ")):
            continue
        lhs, _, rhs = line.partition("=")
        lhs = lhs.strip()
        if lhs.isidentifier():
            table[lhs] = rhs.strip()
    return table


def _strip_noops(expression):
    """Drop the `// 1` and `* 1` an emitter's stride generalisation leaves behind."""
    text = re.sub(r"//\s*1\b", " ", expression)
    return re.sub(r"\*\s*1\b", " ", text)


# Two rules were tried and retired on the way to the access-shape criterion, and
# their reasons are kept here because they are the reasons the current rule looks
# the way it does:
#
#   * "any address whose names came from `//` or `%` is a gather" refused Task155's
#     legal 0.64x kernel and every 2-D tile on 昆仑芯 -- a stride-linear tile is a
#     blocked access, not a scatter;
#   * "pair the `//` and `%` divisors at parenthesis depth zero" then failed on the
#     opposite side: an unpaired `%` is how a row-broadcast index is written
#     (`(idx // N) % M`), and it is not a scatter either.
#
# What survived is the subtract-remainder test in `_remainder_definitions` plus
# data-dependence: a permutation is written as `base - quotient * divisor`, and a
# gather is an address built from a loaded value. The retired helpers were deleted
# rather than left uncalled -- an uncalled function that encodes a refuted rule is
# a trap for the next reader.


def _closure(offset, definitions):
    """Every definition the address depends on, transitively."""
    seen, frontier = {}, [token for token in re.findall(r"[A-Za-z_]\w*", offset)
                          if token in definitions]
    while frontier:
        name = frontier.pop()
        if name in seen:
            continue
        rhs = definitions[name]
        seen[name] = rhs
        frontier.extend(token for token in re.findall(r"[A-Za-z_]\w*", rhs)
                        if token in definitions and token not in seen)
    return seen


def _remainder_definitions(definitions):
    """Coefficients that recompute a remainder as `base - quotient * divisor`.

    That pair (`r = e - t * PLANE` with `t = e // PLANE`) is how a permutation is
    written: the lane index is split and only partly recombined, so the address
    walks the source in a different order than the lanes do. It is the scatter
    shape, as opposed to a tile, which pairs `// D` with `% D`.
    """
    out = set()
    shape = re.compile(
        r"^\(?\s*([A-Za-z_]\w*)\s*\)?\s*-\s*\(?\s*([A-Za-z_]\w*)\s*\)?"
        r"\s*\*\s*\(?\s*([A-Za-z_]\w*|[0-9]+)\s*\)?$")
    for name, rhs in definitions.items():
        match = shape.match(rhs.strip())
        if not match:
            continue
        base, quotient, divisor = match.groups()
        if re.search(rf"\b{re.escape(base)}\s*//\s*{re.escape(divisor)}\b",
                     definitions.get(quotient, "")):
            out.add(name)
    return out


def tainted_by_load(source):
    """Names whose value can come from memory, transitively.

    A load addressed by one of these is a real gather: the address is data, not
    index arithmetic.
    """
    table = _definitions(source)
    tainted = {name for name, value in table.items() if "tl.load" in value}
    changed = True
    while changed:
        changed = False
        for name, value in table.items():
            if name in tainted:
                continue
            if any(re.search(rf"\b{re.escape(other)}\b", value) for other in tainted):
                tainted.add(name)
                changed = True
    return tainted


def gather_like(source, offset):
    """Whether a load address is a scatter rather than a blocked lane address.

    An access-*shape* judgement (2026-10-20, measured on ten official 昆仑芯
    outcomes -- research/kunlun_feasibility/RESULTS.md), decided on the
    coefficients the address is built from rather than on its text:

    * data-dependent (built from a loaded value) -> gather;
    * a coefficient that recomputes a remainder as `base - quotient * divisor`
      -> permutation, the scatter that collapses this target;
    * a coefficient whose top-level `%` divisor differs from its top-level `//`
      divisor -> the index is partly recombined, i.e. permuted;
    * anything else -- affine, a broadcast `idx // D`, or a pair of `// D` and
      `% D` that merely re-splits one index into rows and columns -- is blocked.

    Task155's legal 0.64x kernel addresses
    `idx // dim1 * stride0 + idx // 1 % dim1 * stride1`; the textual rule this
    replaces refused it, and would have refused every 2-D tile on this target.
    """
    if mentions(offset, tainted_by_load(source)):
        return True
    definitions = _definitions(source)
    closure = _closure(offset, definitions)
    remainders = _remainder_definitions(definitions)
    if any(name in remainders for name in closure):
        return True
    # A bare `%` is NOT evidence of a scatter: `broadcast_index = (idx // N) % M`
    # wraps a row index to pick a per-row weight, and 昆仑芯 handles the
    # broadcast (Task155's legal 0.64x kernel does exactly this). The measured
    # discriminator is the subtract-remainder above, which is how a permutation
    # is actually written. Dropping the modulus test removed three false
    # positives and cost nothing on the four illegal observations.
    return False


def _keyword_argument(call, keyword):
    """Text of ``keyword=...`` inside a call span, paren/bracket aware."""
    marker = keyword + "="
    at = call.find(marker)
    if at < 0:
        return ""
    start = at + len(marker)
    depth = 0
    for index in range(start, len(call)):
        char = call[index]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            if depth == 0:
                return call[start:index]
            depth -= 1
        elif char == "," and depth == 0:
            return call[start:index]
    return call[start:]


def _normalise_mask(text):
    return " ".join(text.split())


def mentions(text, names):
    """Whether ``text`` references any of ``names`` as a whole identifier."""
    if not names:
        return False
    out = set()
    token = ""
    for char in text:
        if char.isalnum() or char == "_":
            token += char
        else:
            if token:
                out.add(token)
            token = ""
    if token:
        out.add(token)
    return bool(out & names)


def _first_argument(call):
    """The first argument of a ``name(...)`` span, paren/bracket aware."""
    inner = call[call.index("(") + 1:]
    depth = 0
    for index, char in enumerate(inner):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            return inner[:index]
    return inner


@dataclass(frozen=True)
class CarriedStore:
    """One ``tl.store`` whose stored value crosses an iteration boundary.

    The record is what the message is built from, and it is deliberately made of
    facts rather than a verdict: which loop, which register, and how certain the
    dependence is in ``dependences``' three-valued sense.
    """

    loop: str
    store: str
    registers: tuple
    certainty: str


#: A statement that may or may not execute its body. Its *header* is still
#: evaluated every iteration, which is why the two are separated below.
_CONDITIONAL = (ast.If, ast.While, ast.For, ast.Try, ast.With)


def loop_carried_stores(source):
    """Every ``tl.store`` in ``source`` whose value depends on a carried register.

    This is a *temporal* judgement about one emitted kernel, and it is made on
    the kernel's loop tree rather than on its text. The distinction that matters
    on 昆仑芯 is not "is there a store inside a loop" -- Task147/149 stream a wide
    row with a store per chunk and measure 6.28x / 2.01x there -- but whether the
    stored value depends on a register the same loop body updates. Task152's
    chunk cumsum is that shape (``acc += gate`` then a store of ``acc``) and
    measures 0.03x with both a runtime loop and a ``static_range`` unroll, so the
    loop form is not what the backend rejects; the dependence chain is.

    Structural, in three steps:

    1. for each loop, find the **carried** registers -- names the body both binds
       and reads *before binding them in that iteration*, which is what makes the
       read see the previous iteration's value;
    2. resolve each store's stored value through the body's assignments, so a
       value reached via two hops is still attributed to the register it came
       from;
    3. attribute the store to the **innermost** loop that carries it, so a nested
       accumulation is reported once, where the chain actually binds.

    Certainty reuses ``analysis.dependences``' vocabulary rather than inventing a
    second one, because the two modules answer the same question at different
    granularities -- between kernels there, inside one kernel here -- and a
    reader who has learned ``must-overlap``/``may-overlap`` should not have to
    learn a third word:

    * ``must-overlap`` -- the carried register is read and updated on the body's
      straight-line path, so every iteration depends on the one before it;
    * ``may-overlap`` -- every update of that register sits behind a condition in
      the loop body, so an iteration that takes no branch carries nothing.

    What this cannot say anything about: a kernel whose store *address* is a
    permutation or is data-dependent but whose value carries nothing. Such a
    kernel has no loop-carried dependence at all, so this function returns
    nothing for it -- Task154 is exactly that kernel, and its failure is an
    addressing fact belonging to ``analysis/access_shape`` (``PERMUTED`` /
    ``SCATTER``), which the ``gather_load_ok`` check reads back from the source.
    Two different questions, two different checks; conflating them is how the
    first version of this module missed v3/v4.
    """
    # Emitted kernels are whole modules, but `violations` is also called on
    # snippets from the tests, so an indented fragment is dedented before it is
    # parsed. A fragment with no loop has nothing for this judgement to say and
    # is skipped; a fragment that *does* declare a loop and still will not parse
    # is a real error, not a clean result -- a checker that returns nothing when
    # it cannot read its input is the failure mode this file exists to avoid.
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError as error:
        if re.search(r"^\s*(for|while)\s", source, re.MULTILINE):
            raise ValueError(
                "a loop-bearing source must parse before its loop structure can "
                "be judged, and emitted Triton is Python by construction "
                f"(Design.md 13.8); this does not: {error}"
            ) from error
        return ()

    claimed = set()
    found = []
    # innermost first, so a store belongs to the loop whose iterations it binds
    for _, loop in sorted(_loops(tree), key=lambda item: -item[0]):
        for store, carried in _carried_in(loop):
            if id(store) in claimed:
                continue
            claimed.add(id(store))
            found.append(carried)
    return tuple(found)


def _loops(tree):
    """Every loop in the source, with its nesting depth."""
    found = []

    def walk(node, depth):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.For, ast.While)):
                found.append((depth, child))
            walk(child, depth + 1 if isinstance(child, (ast.For, ast.While)) else depth)

    walk(tree, 0)
    return found


def _bound(node):
    """Names a statement binds, on any path inside it."""
    return {child.id for child in ast.walk(node)
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)}


def _reads(stmt):
    """(reads on every execution, reads only inside a branch) of one statement.

    An augmented assignment's target is a read as well as a write -- ``acc += x``
    reads the carry it then updates -- and the AST puts it in a ``Store`` context,
    so it is added by hand. Getting this wrong is what made the previous textual
    version miss ``acc = acc + x``: it only looked for the ``+=`` token.
    """
    always, conditional = set(), set()

    def visit(node, inside):
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            (conditional if inside else always).add(node.target.id)
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                (conditional if inside else always).add(node.id)
            return
        if isinstance(node, _CONDITIONAL):
            for field in ("test", "iter", "target"):
                child = getattr(node, field, None)
                if child is not None:
                    visit(child, inside)
            for field in ("body", "orelse", "finalbody"):
                for child in getattr(node, field, []) or []:
                    visit(child, True)
            for handler in getattr(node, "handlers", []) or []:
                visit(handler, True)
            return
        for child in ast.iter_child_nodes(node):
            visit(child, inside)

    visit(stmt, False)
    return always, conditional


def _straight_line_binding(stmt):
    """Names a statement binds on every execution of it.

    Only a bare assignment qualifies: a binding inside an ``if`` happens on some
    iterations, and a binding inside a nested loop happens after this iteration's
    reads of interest.
    """
    if isinstance(stmt, ast.Assign):
        return {name.id for target in stmt.targets
                for name in ast.walk(target) if isinstance(name, ast.Name)}
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        return {stmt.target.id}
    return set()


def _carried_of(loop):
    """``{register: certainty}`` for the values that cross this loop's iterations.

    A name is carried when the body reads it before binding it *in that
    iteration*: that read must see the previous iteration's value. A register
    rebound unconditionally at the top of the body (``acc = tl.zeros(...)``) is
    the opposite case and is correctly not carried.

    The certainty attached here is a property of the *register*, not of the
    store that exposes it, because the cost is the loop's serialisation: a
    register updated on every iteration makes every iteration wait for the
    previous one, whichever store happens to read it and whether or not that
    store is itself guarded. So:

    * ``must-overlap`` -- the update is on the body's straight-line path;
    * ``may-overlap`` -- every update of it sits behind a condition in the body,
      so an iteration that takes no branch carries nothing.
    """
    if isinstance(loop, ast.For):
        induction = {name.id for name in ast.walk(loop.target)
                     if isinstance(name, ast.Name)}
    else:
        induction = set()

    bound, conditional_bindings = set(), set()
    for stmt in loop.body:
        bound |= _bound(stmt)
        conditional_bindings |= _conditional_bindings(stmt)
    bound -= induction

    defined = set()
    read_before = {}
    for stmt in loop.body:
        always, conditional = _reads(stmt)
        for name in (always | conditional) & bound:
            if name in defined:
                continue
            # an unconditional read settles it as `must`; a conditional one only
            # proposes `may`, and a later unconditional read can still promote it
            read_before.setdefault(name, name in always)
            read_before[name] = read_before[name] or (name in always)
        defined |= _straight_line_binding(stmt)

    return {
        name: (MUST if unconditional and name not in conditional_bindings else MAY)
        for name, unconditional in read_before.items()
    }


def _conditional_bindings(stmt):
    """Names a statement binds only when a branch inside it is taken."""
    if not isinstance(stmt, _CONDITIONAL):
        return set()
    names = set()
    for field in ("body", "orelse", "finalbody"):
        for child in getattr(stmt, field, []) or []:
            names |= _bound(child)
    for handler in getattr(stmt, "handlers", []) or []:
        names |= _bound(handler)
    return names


def _carried_in(loop):
    """(store node, CarriedStore) for each store this loop's iterations bind."""
    carried = _carried_of(loop)
    if not carried:
        return []

    assignments = {}
    for stmt in loop.body:
        for child in ast.walk(stmt):
            if isinstance(child, ast.Assign) and len(child.targets) == 1 \
                    and isinstance(child.targets[0], ast.Name):
                assignments[child.targets[0].id] = child.value
            elif isinstance(child, ast.AugAssign) and isinstance(child.target, ast.Name):
                assignments.setdefault(child.target.id, child.value)

    header = _header(loop)
    found = []
    for store in _stores(loop):
        value = store.args[1] if len(store.args) > 1 else None
        if value is None:
            continue
        reached = _registers_reached(value, assignments) & set(carried)
        if not reached:
            continue
        # the strongest claim available: `must` if any reached register is must
        certainty = MUST if MUST in {carried[name] for name in reached} else MAY
        found.append((store, CarriedStore(
            loop=header,
            store=_one_line(ast.unparse(store)),
            registers=tuple(sorted(reached)),
            certainty=certainty,
        )))
    return found


def _registers_reached(expression, assignments):
    """Every name ``expression`` depends on, transitively through the body."""
    seen, frontier = set(), [name.id for name in ast.walk(expression)
                             if isinstance(name, ast.Name)
                             and isinstance(name.ctx, ast.Load)]
    while frontier:
        name = frontier.pop()
        if name in seen:
            continue
        seen.add(name)
        source = assignments.get(name)
        if source is not None:
            frontier.extend(child.id for child in ast.walk(source)
                            if isinstance(child, ast.Name)
                            and isinstance(child.ctx, ast.Load))
    return seen


def _stores(loop):
    for child in ast.walk(loop):
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)                 and child.func.attr == "store"                 and isinstance(child.func.value, ast.Name)                 and child.func.value.id == "tl":
            yield child


def _header(loop):
    if isinstance(loop, ast.For):
        return f"for {ast.unparse(loop.target)} in {ast.unparse(loop.iter)}"
    return f"while {ast.unparse(loop.test)}"


def _one_line(text, width=96):
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= width else collapsed[:width] + "..."


def assert_contracts(source, platform):
    """Raise ``ValueError`` when emitted source violates a measured contract."""
    problems = violations(source, platform)
    if problems:
        raise ValueError("backend contract violated: " + "; ".join(problems))
    return source


def audit_pipeline(pipeline, platform):
    """Audit every kernel of a lowered pipeline, keyed by kernel name."""
    report = {}
    for kernel in pipeline.triton.kernels:
        problems = violations(kernel.source, platform)
        if problems:
            report[kernel.name] = problems
    return report
