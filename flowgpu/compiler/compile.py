"""The single entry point into the FlowGPU flowgpu.compiler.

`compile_source` is the only function a caller (the CLI today, an agent
repair loop later) needs. Every stage stays private to this module, so the
later stages of Design.md 13 -- Dataflow IR, fusion, liveness, backend --
can be inserted here without changing any caller.

Every failure leaves as a CompileError carrying the category Design.md 17.3
promises, a position, and -- when the file has more than one pipeline -- the
pipeline it came from. Nothing is re-classified here: each category is
decided by the layer that detected the failure, because that is the only
layer that knows it.
"""

import re
from dataclasses import dataclass, fields, is_dataclass, replace

from flowgpu.compiler.core.ast import InputDecl, PipelineDecl
from flowgpu.compiler.core.types import TensorType
from flowgpu.compiler.core.literals import ShapeLiteral
from flowgpu.compiler.ir.dataflow import DataflowGraph
from flowgpu.compiler.ir.composite_ops import expand as expand_composed_operators
from flowgpu.compiler.ir.dataflow_builder import DataflowBuilder
from flowgpu.compiler.core.diagnostic import CompileError, DiagnosticKind
from flowgpu.compiler.optimization.fusion import FusionReport, fuse, coalesce_store_cache_groups
from flowgpu.compiler.optimization.optimizer import optimize_graph
from flowgpu.compiler.analysis.liveness import MemoryPlan, plan
from flowgpu.compiler.analysis.planning import summarize_plan
from flowgpu.compiler.ir.verification import verify_graph
from flowgpu.compiler.core.position import Position
from flowgpu.compiler.analysis.access_shape import group_conflicts
from flowgpu.compiler.backend.platform_configs import PlatformConfig, resolve_platform
from flowgpu.compiler.backend.backends import LAYOUT_OPS, library_dispatch
from flowgpu.compiler.backend.backend_policy import make_backend_policy
from flowgpu.compiler.core.shape_constraints import ShapeConstraints
from flowgpu.compiler.analysis.kernel_schedule import analyze_schedule
from flowgpu.compiler.backend.primitives import PRIMITIVES
from flowgpu.compiler.core.symbol_table import SymbolTable
from flowgpu.compiler.core.symbol_table_builder import SymbolTableBuilder
from flowgpu.compiler.backend.reference import ReferenceProgram
from flowgpu.compiler.backend.reference import lower as lower_reference
from flowgpu.compiler.backend.triton_backend import TritonProgram
from flowgpu.compiler.backend.triton_backend import lower as lower_triton
from flowgpu.compiler.ir.lowering_ir import analyze as analyze_lowering_ir
from flowgpu.compiler.ir.semantic_ir import SemanticProgram, normalize as normalize_semantics


# Lark writes the location into its message in two shapes -- "at line 2,
# column 13." for a bad token and "at line 1 col 34" for a bad character --
# and the diagnostic carries it in a field, so strip it either way. The comma
# before it goes too: a lexical message is worded "..., at line 1 col 34", and
# stripping only the location would leave the comma stranded.
LOCATION_SUFFIX = re.compile(r",?\s+at line \d+(?:,? col(?:umn)? \d+)?\.?$")


@dataclass
class PipelineResult:
    """One compiled pipeline: its tree, its type environment, its graph.

    The analyses keep their order of 13: the graph is built, fusion reads the
    graph to decide which values never need a buffer, liveness reads both to
    decide when the buffers that remain are alive and which of them can share
    storage (11 lists `materialization requirement` and `reuse opportunity` as
    separate things to derive, and this is why), and lowering emits one kernel
    per fusion group against that buffer plan.
    """

    pipeline: PipelineDecl
    symbols: SymbolTable
    semantic: SemanticProgram
    normalization: tuple
    dataflow: DataflowGraph
    fusion: FusionReport
    memory: MemoryPlan
    lowering_ir: tuple
    #: None when the caller asked for analyses only (``emit=False``).
    triton: TritonProgram | None

    # Two targets, one set of decisions. The Triton module is the artifact
    # 13.8 asks for; the reference program is the same program in Python that
    # this machine can run, which is what makes the semantics of every
    # operator and every index checkable. Both read 13.6's partition and
    # 13.7's buffer plan, so they share optimization decisions. Independent numerical
    # tests are still needed to check each implementation and the plan itself.
    reference: ReferenceProgram | None
    backend_policy: object
    kernel_schedule: object
    #: What the target is measured not to sustain about this plan's *addressing*,
    #: judged from the fusion plan before any source exists (21.9's access
    #: dimension, not the byte model). Report-only: it is computed on every
    #: compile so that `--emit contracts` and the JSON view can show it, and it
    #: only becomes a refusal when ``platform_gate=True`` is asked for.
    platform_conflicts: tuple = ()
    #: IR-native scalar placement plan when ``scalar_placement='sps'``; the
    #: decisions are also applied to the Triton host/kernel ABI (S1).
    scalar_placement_plan: object = None
    #: Records from the graph-optimisation pipeline (identity/CSE/DCE), empty
    #: when ``optimize=False``. Kept on the result so a caller can explain why
    #: the dataflow differs from a conservative build.
    optimization: tuple = ()
    #: Structural plan cost (launches, buffers, slots, modeled traffic A).
    planning: object = None


# The measured minimax arm for a cross-domain reduction is never worse than 1.16x
# the per-shape best, where the untuned recompute caps reach 6.67x and materialize
# 4.47x (docs/PARALLELISM_GAP_20261019.md sections 40, 44.3). 'auto' applies it
# only when the program actually has a cross-domain reduction, so every other
# program keeps the materialize partition it was validated with.
DEFAULT_CROSS_DOMAIN_STRATEGY = "auto"


@dataclass
class CompileResult:
    """A successfully compiled file -- one pipeline per entry.

    A file is `pipeline+` (EBNF `program: pipeline+`), so this is a list
    even when the usual file holds a single pipeline.
    """

    pipelines: list[PipelineResult]
    source_path: str | None = None


def syntax_error_from(exc: "object") -> CompileError:
    """Turn a Lark failure into a syntax diagnostic.

    Shared by the parser stage below and by the CLI's `--emit tokens`, which
    runs the lexical stage on its own: one rendering of a syntax error, not
    two.
    """
    first_line = str(exc).splitlines()[0] if str(exc) else "invalid syntax"
    # Lark already embeds "at line N, column M" in its message, and the
    # position is carried by the diagnostic, so strip the duplicate.
    message = LOCATION_SUFFIX.sub("", first_line).strip()
    line = getattr(exc, "line", None)
    column = getattr(exc, "column", None)

    return CompileError(
        DiagnosticKind.SYNTAX,
        message or "invalid syntax",
        position=(
            Position(line, column)
            if line is not None and column is not None
            else None
        ),
    )


def auto_cross_domain_config():
    """The caps the measured minimax arm uses for a cross-domain reduction.

    Kept next to the resolution that applies them so the number and its reason
    cannot drift apart. The evidence is in
    docs/PARALLELISM_GAP_20261019.md sections 40 and 44.3.
    """
    return resolve_platform(indexed_columns=4096, indexed_rows=16,
                            reduction_block_limit=65536, num_warps=8)


def compile_source(source: str, source_path: str | None = None, *, gpu_backend="triton", attention_impl="math", platform_config: PlatformConfig | None = None, custom_ops=False, autotune=False, autotune_fold=False, fusion_policy="traffic", library_matmul="unfused", shape_bindings=None, shape_constraints=None, index_fusion=True, reduction_fusion=True, cross_domain_strategy=DEFAULT_CROSS_DOMAIN_STRATEGY, gather_fusion=False, gather_plan=None, numeric_mode="fp64", concurrency="sequential", launch_cache=False, semantic_view_folding=True, prune_kernel_abi=False, emit=True, platform_gate=False,
    dependence_guard=False, split_views=False, layout_views=None, scan_epilogue=False, _pipelines=None,
    scalar_placement="off", scalar_specialization=None,
    scalar_calls=1, optimize=True, identity_folding=True) -> CompileResult:
    """Parse and type-check `source`, or raise CompileError.

    A builtin TypeError or ValueError escaping from here is not a bad
    program: it is a compiler defect, and it should surface as a traceback
    rather than be dressed up as a diagnostic the user is asked to fix. The
    old code decided the diagnostic category from exactly those two classes.
    """
    if type(semantic_view_folding) is not bool:
        raise ValueError('semantic_view_folding must be a bool')
    if type(prune_kernel_abi) is not bool:
        raise ValueError('prune_kernel_abi must be a bool')
    if type(launch_cache) is not bool:
        raise ValueError('launch_cache must be a bool')
    if type(optimize) is not bool:
        raise ValueError('optimize must be a bool')
    if type(identity_folding) is not bool:
        raise ValueError('identity_folding must be a bool')
    if type(split_views) is not bool:
        raise ValueError('split_views must be a bool')
    if layout_views not in (None, True, False, "cost", "experimental"):
        raise ValueError("layout_views must be None, True, False, 'cost' or 'experimental'")
    if type(scan_epilogue) is not bool:
        raise ValueError('scan_epilogue must be a bool')
    if type(emit) is not bool:
        raise ValueError('emit must be a bool')
    if not isinstance(platform_gate, bool):
        raise ValueError('platform_gate must be a bool')
    if gather_plan is not None and gather_fusion:
        raise ValueError("gather_plan and gather_fusion are mutually exclusive")
    if scalar_placement not in {"off", "sps"}:
        raise ValueError("scalar_placement must be 'off' or 'sps'")
    if scalar_specialization is not None and not isinstance(scalar_specialization, dict):
        raise TypeError("scalar_specialization must be a dict of scalar name -> value")
    if type(scalar_calls) is not int or scalar_calls <= 0:
        raise ValueError("scalar_calls must be a positive int")
    dispatch = library_dispatch(gpu_backend, library_matmul)
    if attention_impl not in {"math", "sdpa"}:
        raise ValueError("attention_impl must be 'math' or 'sdpa'")
    # 'auto' is allowed to pick the platform caps as well as the strategy, but
    # only when the caller did not choose them. Remembered before the default is
    # materialised so the two cases stay distinguishable.
    caller_platform = platform_config
    platform_config = resolve_platform() if platform_config is None else platform_config
    if not isinstance(platform_config, PlatformConfig):
        raise TypeError("platform_config must be a PlatformConfig")
    if type(autotune_fold) is not bool:
        raise ValueError('autotune_fold must be a bool')
    if autotune_fold and (platform_config.name != 'cuda' or gpu_backend not in {'triton','hybrid'}):
        raise ValueError('fold autotune currently requires CUDA triton/hybrid')
    backend_policy = make_backend_policy(
        gpu_backend=gpu_backend, library_matmul=library_matmul,
        platform_config=platform_config, numeric_mode=numeric_mode,
        concurrency=concurrency,
    )
    if cross_domain_strategy not in {"materialize", "recompute", "segmented", "segmented-merge", "auto"}:
        raise ValueError("cross_domain_strategy must be materialize, recompute, segmented, segmented-merge or auto")
    # 'auto' resolves to recompute only if this program actually has a
    # cross-domain reduction; otherwise it stays materialize and the emitted
    # source is unchanged.
    resolved_cross_domain = "materialize" if cross_domain_strategy == 'auto' else cross_domain_strategy
    # `cross_domain_config` is assigned after every policy rewrite below: the
    # segmented strategy replaces `platform_config` (coalesce_reduction_sweeps),
    # and the old early assignment left fusion/lowering reading the pre-rewrite
    # object -- hidden only because segmented reductions do not read that flag
    # (review P2, 2026-10-26).
    # Checked against the resolved strategy, not the requested one: 'auto' on a
    # platform or configuration without indexed and reduction fusion is not an
    # error, it simply resolves to materialize, which is what those targets
    # already emit.
    if resolved_cross_domain != "materialize" and (platform_config.name != "cuda" or not index_fusion or not reduction_fusion):
        raise ValueError("cross-domain recompute requires CUDA indexed and reduction fusion")
    if cross_domain_strategy in {'segmented','segmented-merge'}:
        if gpu_backend!='triton' or concurrency!='sequential' or autotune or autotune_fold or custom_ops:
            raise ValueError('segmented strategy requires fixed native sequential lowering')
        platform_config=replace(platform_config,coalesce_reduction_sweeps=True)
    if platform_config.name != "cuda" and gpu_backend != "triton":
        raise ValueError("non-CUDA platform profiles require gpu_backend='triton'")
    cross_domain_config = platform_config
    if autotune:
        from flowgpu.compiler.backend.autotune import configs_for
        if gpu_backend not in {"triton", "hybrid"}:
            raise ValueError("autotune requires a CUDA triton or hybrid gpu_backend")
        if not configs_for(platform_config.name):
            raise ValueError(f"autotune unavailable for {platform_config.name}")
        if "matmul" not in platform_config.allowed_kernel_kinds:
            raise ValueError(f"autotune requires explicitly enabling experimental matmul on {platform_config.name}")
    if fusion_policy not in {"rule", "traffic"}:
        raise ValueError("fusion_policy must be 'rule' or 'traffic'")
    if _pipelines is not None:
        if source is not None:
            raise ValueError("pass either source or _pipelines, not both")
        pipelines = list(_pipelines)
        if not pipelines:
            raise ValueError("_pipelines must contain at least one pipeline")
    else:
        if source is None:
            raise ValueError("compile_source needs source text")
        # The parser is imported lazily so an AST/eDSL caller does not load
        # Lark or the DSL frontend just to compile an already-built program.
        from lark.exceptions import LarkError
        from flowgpu.compiler.frontend.ast_builder import ASTBuilder
        from flowgpu.compiler.frontend.parser import parser

        try:
            tree = parser.parse(source)
        except LarkError as exc:
            raise syntax_error_from(exc) from exc
        except RecursionError as exc:
            raise CompileError(
                DiagnosticKind.SYNTAX,
                "the program is nested too deeply to compile; flatten the expression "
                "into named intermediate values",
            ) from exc

        try:
            pipelines = ASTBuilder().transform(tree)
        except RecursionError as exc:
            raise CompileError(
                DiagnosticKind.SYNTAX,
                "the program is nested too deeply to compile; flatten the expression "
                "into named intermediate values",
            ) from exc
        except LarkError as exc:
            # lark wraps a RecursionError raised inside a transformer callback in a
            # VisitError; a depth limit is a diagnosable program property, not a
            # crash (review P2, 2026-10-26).
            if isinstance(getattr(exc, "orig_exc", None), RecursionError):
                raise CompileError(
                    DiagnosticKind.SYNTAX,
                    "the program is nested too deeply to compile; flatten the "
                    "expression into named intermediate values",
                ) from exc
            raise

    # CUDA defaults layout_views on (device-validated on three real tasks);
    # other backends keep it off until their layout/indexed contracts are
    # validated.  Explicit True/False/"cost" overrides the platform default.
    if layout_views is None:
        layout_views_enabled = platform_config.layout_views
    elif layout_views == "experimental":
        # Build/diagnostic escape hatch only: permits a non-CUDA target to emit
        # an indexed layout-view region so a real device submission can gather
        # the evidence needed to promote the capability.  Never a default.
        layout_views_enabled = True
    else:
        layout_views_enabled = layout_views is True or layout_views == "cost"
        if layout_views_enabled and not platform_config.layout_views:
            raise CompileError(
                DiagnosticKind.OPERATOR,
                f"platform '{platform_config.name}' has no validated layout-view "
                "fusion contract; compile with layout_views=False (the "
                "platform default) or use a CUDA target",
                pipeline=pipelines[0].name if pipelines else None,
            )
    layout_views_cost = layout_views == "cost"
    if scan_epilogue:
        from flowgpu.compiler.optimization.scan_epilogue import rewrite as rewrite_scan
        pipelines = rewrite_scan(pipelines)

    # Two pipelines with one name would generate two `def run_<name>` in the
    # same module and the later would quietly win; a module is addressed by
    # pipeline name everywhere downstream (review P1-12, 2026-10-26).
    seen_names = {}
    for pipeline in pipelines:
        if pipeline.name in seen_names:
            raise CompileError(
                DiagnosticKind.DEPENDENCY,
                f"duplicate pipeline name '{pipeline.name}'; every pipeline in "
                "a source file needs a unique name",
                position=pipeline.position,
                pipeline=pipeline.name,
            )
        seen_names[pipeline.name] = pipeline
    if scalar_specialization:
        scalar_names = {
            statement.name
            for pipeline in pipelines
            for statement in pipeline.statements
            if isinstance(statement, InputDecl) and statement.type.is_scalar
        }
        unknown = set(scalar_specialization) - scalar_names
        if unknown:
            raise ValueError(f"scalar_specialization names not scalar inputs: {sorted(unknown)}")
    constraints = ShapeConstraints(shape_constraints or ())
    shape_nodes = []
    def collect(value):
        if isinstance(value, (TensorType, ShapeLiteral)):
            shape_nodes.append(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)
        elif is_dataclass(value):
            for field in fields(value):
                collect(getattr(value, field.name))
    collect(pipelines)
    if shape_bindings is not None:
        if not isinstance(shape_bindings, dict) or any(
            not isinstance(key, str) or type(value) is not int or value <= 0
            for key, value in shape_bindings.items()
        ):
            raise ValueError("shape_bindings must map symbol names to positive integers")
        known = {d for value in shape_nodes for d in value.shape if not str(d).isdigit()}
        unknown = set(shape_bindings) - known
        if unknown:
            raise ValueError(f"unknown shape symbols: {', '.join(sorted(unknown))}")
        for value in shape_nodes:
            value.shape = [str(shape_bindings.get(d, d)) for d in value.shape]

    constraints.validate(
        known_symbols={
            str(dimension)
            for value in shape_nodes
            for dimension in value.shape
            if not str(dimension).isdigit()
        } | set(shape_bindings or ()),
        bindings=shape_bindings,
    )

    compiled = []
    failed = []

    for pipeline in pipelines:
        # One bad pipeline does not hide the others: 17.3's repair loop should
        # hear about every pipeline that failed, not just the first one.
        try:
            symbols = SymbolTableBuilder(
                allow_symbolic_split=(split_views or layout_views_enabled)).build(pipeline)
        except CompileError as error:
            failed.extend(error.all_diagnostics())
            continue

        
        # 13.4 runs on every compile, not on request: lowering is a stage of
        # the compiler, and a caller that asks for the graph is asking for
        # something that already exists. 13.6 and 13.7 run for the same
        # reason, and in that order, because 13.7's reuse is about the
        # buffers that fusion does not remove.
        semantic = DataflowBuilder(
            allow_symbolic_split=(split_views or layout_views_enabled)).build_semantic(
            pipeline, symbols, shape_constraints=constraints.constraints)
        dataflow, normalization = normalize_semantics(semantic, fold_views=semantic_view_folding)
        # Composed operators (segmented_cumsum) are identities over operators
        # the compiler already lowers; expanding them before fusion and the
        # buffer plan is what gives their components the ordinary treatment.
        dataflow, expansion = expand_composed_operators(dataflow)
        normalization = normalization + expansion
        optimized = optimize_graph(dataflow, enabled=optimize, identity_folding=identity_folding)
        dataflow = optimized.graph
        optimization = optimized.records
        # Even with optimization disabled, the pass layer's invariants must hold
        # before fusion/liveness consume the graph.
        verify_graph(dataflow)
        # Layout ops host no pointwise regions: their lowering is a host
        # copy with no registers to fuse into, so a relu epilogue on a split
        # would be a category error rather than an optimization.
        # Selective hybrid: with library_matmul='unfused' (the default),
        # matmul stays fusible in hybrid mode. A fused group keeps its
        # Triton kernel -- the fused single-kernel form is where generated
        # code beats the eager two-kernel sequence (A800 autotune PoC) --
        # while a bare matmul group has nothing to fuse and lowers to the
        # cuBLAS library call, where cuBLAS beats our fixed-tile GEMM at
        # every measured M. library_matmul='all' restores the pre-selective
        # hybrid: every matmul opaque and every matmul a library call.
        # attention and rms_norm stay opaque in hybrid either way: their
        # library sequences are correctness-driven choices (the FP32 library
        # reduction the decoder comparison validated), not performance
        # choices.
        opaque = dispatch.opaque_anchors + LAYOUT_OPS + PRIMITIVES
        from flowgpu.compiler.analysis.traffic import placement_cost
        from flowgpu.compiler.backend.autotune import configs_for
        candidates = configs_for(platform_config.name) if autotune else ()
        bm = min(c[0] for c in candidates) if candidates else 64
        bn = min(c[1] for c in candidates) if candidates else 64
        fusion = fuse(dataflow, opaque_anchors=opaque + ('store_cache_4d', 'linear_decode_forward', 'patchify',
                              'scatter_by_index', 'store_at', 'store_at_trusted',
                              'topk_last', 'gather_last', 'segment_rank',
                              'argmax_last', 'argmin_last',
                              # A scan's pointwise prologue/epilogue is emitted
                              # inside the scan kernel (each lane is independent
                              # once the scan tile exists), so `cumsum` and
                              # `cumsum_totals` are no longer opaque: a pointwise
                              # consumer such as `add_broadcast(scan, row)` joins
                              # the group and folds into the same kernel.  The
                              # carried-tuple scans stay opaque until their tuple
                              # merge is emitted.
                              'cummin_last', 'cummax_last',
                              'gated_delta_recurrence'),
                      placement_cost=placement_cost(bm, bn) if fusion_policy == "traffic" else None)
        fusion = coalesce_store_cache_groups(dataflow, fusion)
        if gather_fusion:
            if platform_config.name != 'cuda':
                raise ValueError('gather_fusion is currently a CUDA-only experiment')
            from flowgpu.compiler.optimization.row_gather_fusion import fuse_row_gathers
            fusion = fuse_row_gathers(dataflow, fusion)
        if gather_plan is not None:
            if platform_config.name != 'cuda':
                raise ValueError('gather_plan is currently a CUDA-only experiment')
            from flowgpu.compiler.optimization.row_gather_fusion import apply_gather_plan
            fusion = apply_gather_plan(dataflow, fusion, gather_plan, platform_config)
        # Shape-aware regions currently have real CUDA evidence only. Keep
        # other platforms on their established partition and whitelist.
        if index_fusion and (platform_config.name == "cuda" or platform_config.experimental_region_fusion) and {
            "primitive", "pointwise", "layout"
        }.issubset(platform_config.allowed_kernel_kinds):
            from flowgpu.compiler.optimization.indexed_fusion import fuse_indexed
            # Opt-in: a last-axis split lowers to a view pair and fuses into
            # its consumer, instead of materialising two copies. Off by
            # default so every existing emission stays byte-identical.
            fusion = fuse_indexed(dataflow, fusion, split_views=split_views, layout_views=layout_views_enabled, layout_view_cost=layout_views_cost)
            if reduction_fusion:
                from flowgpu.compiler.optimization.reduction_fusion import fuse_reductions
                fusion = fuse_reductions(dataflow, fusion,
                                         max_reductions=platform_config.reduction_group_limit,
                                         coalesce=platform_config.coalesce_reduction_sweeps)
                if resolved_cross_domain in {"recompute", "segmented", "segmented-merge"} or cross_domain_strategy == 'auto':
                    from flowgpu.compiler.optimization.cross_domain_fusion import fuse_cross_domain
                    if cross_domain_strategy == 'auto':
                        # The measured minimax arm: on five shapes its worst-case
                        # regret against the per-shape best was 1.16x, where the
                        # untuned recompute caps reach 6.67x and materialize 4.47x.
                        config = (auto_cross_domain_config()
                                  if caller_platform is None else platform_config)
                        fused = fuse_cross_domain(dataflow, fusion, config)
                        if fused != fusion:
                            fusion = fused
                            resolved_cross_domain = 'recompute'
                            cross_domain_config = config
                    else:
                        fusion = fuse_cross_domain(dataflow, fusion, platform_config)
        memory = plan(dataflow, fusion, reuse=(concurrency == "sequential"),
                 dependence_guard=dependence_guard)
        if resolved_cross_domain in {'segmented','segmented-merge'}:
            from flowgpu.compiler.ir.reduction_stages import plan_reduction_stages
            memory=replace(memory,stage_plan=plan_reduction_stages(dataflow,fusion,merge=resolved_cross_domain=='segmented-merge'))
        planning = summarize_plan(dataflow, fusion, memory)

        # The pre-emission gate. `contracts.violations` reads emitted text, so by
        # the time it can answer, the target has already been handed a plan it
        # was measured not to sustain; the fusion report is the last point where
        # the plan is decided and no source has been written, so the access-shape
        # verdict is taken here. Report-only unless the caller asks for the gate:
        # `platform_gate` is off by default because turning a checker into a
        # refusal changes what compiles, and `docs/COMPILER_ACCEPTANCE.md`
        # requires parent/candidate evidence for that, which needs a device.
        planned_conflicts = group_conflicts(fusion, platform_config)
        if platform_gate and planned_conflicts:
            raise CompileError(
                DiagnosticKind.OPERATOR,
                "this target is measured not to sustain the plan's addressing: "
                + "; ".join(planned_conflicts),
                pipeline=pipeline.name,
            )

        # 13.8 runs here too, and a kernel it cannot emit is data rather than
        # an error: the program is valid FlowGPU either way, so --emit summary
        # has to keep reporting valid source even if a target lacks a lowering.
        # What the backend cannot emit is reported by --emit triton and by the
        # JSON view, both of which read TritonProgram.refusals.
        # ``emit=False`` stops after the analyses. It exists so a caller can ask
        # structural questions -- how many kernels, how wide a tile, how many
        # programs -- without paying for code generation. Everything before this
        # point is unchanged, so the analyses are identical either way.
        placement_plan = None
        constexpr_literals = {}
        if scalar_placement == "sps":
            from flowgpu.compiler.optimization.scalar_placement import plan_placements
            local_specialization = {
                name: value for name, value in (scalar_specialization or {}).items()
                if any(isinstance(statement, InputDecl)
                       and statement.type.is_scalar
                       and statement.name == name
                       for statement in pipeline.statements)
            }
            placement_plan = plan_placements(
                pipeline, dataflow, specialization=local_specialization,
                calls=scalar_calls, fusion=fusion)
            constexpr_literals = placement_plan.constexpr_scalars
        triton_program = lower_triton(
            pipeline, dataflow, fusion, memory,
            gpu_backend=gpu_backend, library_matmul=library_matmul,
            attention_impl=attention_impl, platform_config=cross_domain_config,
            custom_ops=custom_ops, autotune=autotune, autotune_fold=autotune_fold, concurrency=concurrency,
            prune_kernel_abi=prune_kernel_abi, launch_cache=launch_cache,
            constexpr_scalars=constexpr_literals,
        ) if emit else None
        # ``emit=False`` stops after the analyses: the reference host is also a
        # lowering, and running it paid for a pass the caller declined (review
        # P2, 2026-10-26).
        reference_program = lower_reference(
            pipeline, dataflow, fusion, memory, numeric_mode=numeric_mode,
        ) if emit else None
        compiled.append(
            PipelineResult(
                pipeline=pipeline,
                symbols=symbols,
                semantic=semantic,
                normalization=normalization,
                dataflow=dataflow,
                fusion=fusion,
                memory=memory,
                lowering_ir=analyze_lowering_ir(dataflow, fusion, platform_config),
                triton=triton_program,
                reference=reference_program,
                backend_policy=backend_policy,
                kernel_schedule=(analyze_schedule(
                    triton_program.kernels, memory, mode=concurrency
                ) if triton_program is not None else None),
                platform_conflicts=planned_conflicts,
                scalar_placement_plan=placement_plan,
                optimization=optimization,
                planning=planning,
            )
        )

    if failed:
        # The first diagnostic is still the one that stopped the compile; the
        # rest are what the same walk found. Order is source order across
        # pipelines, which is the order a person would fix them in.
        raise CompileError(
            failed[0].kind,
            failed[0].message,
            position=failed[0].position,
            pipeline=failed[0].pipeline,
            others=tuple(failed[1:]),
        )

    return CompileResult(pipelines=compiled, source_path=source_path)


def compile_pipelines(pipelines, source_path=None, **options):
    """Compile already-built core AST pipelines (the Python eDSL entry point).

    This is the same compiler as ``compile_source`` after the parse stage: it
    runs symbol construction, type inference, dataflow lowering, fusion,
    liveness and emission on ``flowgpu.compiler.core.ast`` objects.  It exists so a
    frontend other than the text DSL can use the full pipeline without
    serialising to DSL source or importing the lexer/parser.
    """
    return compile_source(None, source_path, _pipelines=pipelines, **options)
