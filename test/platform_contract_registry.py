"""Registry of PlatformConfig fields and the evidence behind each one.

Every field is classified so a new policy knob cannot land without an owner:

* ``measured``  -- a backend fact established by a device evaluation (or a
  measured launch curve).  These must name an executable fixture.
* ``policy``    -- a search/emission policy whose consumer has tests but no
  single device truth (e.g. autotune candidate tables).
* ``plumbing``  -- identity/metadata fields with no behaviour of their own.

``covered_by`` is either ``("test/test_x.py", "test_name")`` or a fixture
function defined in ``test/test_platform_contract_fixtures.py``.  The
completeness test enforces one entry per dataclass field and that every
``measured`` entry resolves to a real test.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Contract:
    field: str
    kind: str            # measured | policy | plumbing | identity
    covered_by: tuple    # (path, test_function) or ("fixture", function_name)
    note: str


def _t(path, name, note=""):
    return Contract(path.rsplit("/", 1)[-1].split(".")[0], "measured", (path, name), note)


CONTRACTS = {
    # --- identity / plumbing --------------------------------------------------
    "name": Contract("name", "identity", ("fixture", "test_every_field_is_registered"),
                     "platform identifier; validated against PLATFORMS"),
    "device_type": Contract("device_type", "plumbing", ("test/test_custom_ops.py", None),
                            "cuda vs npu device namespace used by host emission"),
    "min_align_bytes": Contract("min_align_bytes", "plumbing", ("docs/BACKEND_CONTRACTS_20261019.md", None),
                                "documented alignment hint; no pointer guarantee is claimed"),
    # --- measured backend contracts ------------------------------------------
    "masked_load_other_honoured": Contract("masked_load_other_honoured", "measured",
        ("fixture", "test_masked_load_other_contract_is_audited_and_executed"), "kunlunxin: false"),
    "block_ptr_reliable": Contract("block_ptr_reliable", "measured",
        ("fixture", "test_block_ptr_contract_is_enforced_per_target"), "ascend: false"),
    "vector_tile_2d": Contract("vector_tile_2d", "measured",
        ("fixture", "test_2d_tile_policy_changes_emission"), "ascend: true"),
    "tile_rows": Contract("tile_rows", "measured",
        ("fixture", "test_2d_tile_policy_changes_emission"), ""),
    "tile_cols": Contract("tile_cols", "measured",
        ("fixture", "test_2d_tile_policy_changes_emission"), ""),
    "gather_load_ok": Contract("gather_load_ok", "measured",
        ("fixture", "test_gather_and_loop_store_contracts_are_audited"), "kunlunxin: false"),
    "store_inside_loop_ok": Contract("store_inside_loop_ok", "measured",
        ("fixture", "test_gather_and_loop_store_contracts_are_audited"), "kunlunxin: false"),
    "affine_write_addresses": Contract("affine_write_addresses", "measured",
        ("test/test_program_scalar.py", "test_only_the_measured_targets_require_affine_write_addresses"), ""),
    "measured_block_curve": Contract("measured_block_curve", "measured",
        ("test/test_measured_block_curve.py", None), "kunlunxin launch-parameter curve"),
    "max_grid_dim": Contract("max_grid_dim", "measured",
        ("fixture", "test_max_grid_dim_caps_the_launch_and_writes_every_element"), "ascend coreDim cap"),
    "reduction_block_limit": Contract("reduction_block_limit", "measured",
        ("fixture", "test_reduction_block_limit_chunks_and_stays_correct"), ""),
    "group_m": Contract("group_m", "measured",
        ("fixture", "test_group_m_swizzle_is_emitted"), "L2 swizzle"),
    "index_dtype": Contract("index_dtype", "measured",
        ("test/test_index_width_policy.py", "test_the_default_table_keeps_cuda_class_targets_on_int64"), ""),
    "sigmoid_impl": Contract("sigmoid_impl", "measured",
        ("fixture", "test_sigmoid_impl_switches_emission_and_executes"), ""),
    "matmul_block_m": Contract("matmul_block_m", "measured",
        ("fixture", "test_matmul_tile_policy_threads_header_and_launch"), "GPU MODE measured tile"),
    "matmul_block_n": Contract("matmul_block_n", "measured",
        ("fixture", "test_matmul_tile_policy_threads_header_and_launch"), ""),
    "matmul_block_k": Contract("matmul_block_k", "measured",
        ("fixture", "test_matmul_tile_policy_threads_header_and_launch"), ""),
    "matmul_num_warps": Contract("matmul_num_warps", "measured",
        ("fixture", "test_matmul_tile_policy_threads_header_and_launch"), ""),
    "matmul_num_stages": Contract("matmul_num_stages", "measured",
        ("fixture", "test_matmul_tile_policy_threads_header_and_launch"), ""),
    # --- policy fields -------------------------------------------------------
    "mapping_strategy": Contract("mapping_strategy", "policy",
        ("test/test_grid_stride_execution.py", "test_the_emitted_loop_stride_is_the_launched_program_count"), ""),
    "default_block_size": Contract("default_block_size", "policy",
        ("test/test_backend_contracts.py", "test_a_measured_curve_decides_the_platform_default"), ""),
    "preferred_grid_multiplier": Contract("preferred_grid_multiplier", "policy",
        ("test/test_grid_stride_execution.py", "test_the_emitted_loop_stride_is_the_launched_program_count"), ""),
    "num_warps": Contract("num_warps", "policy", ("test/test_grid_stride_execution.py", None), ""),
    "num_stages": Contract("num_stages", "policy", ("test/test_grid_stride_execution.py", None), ""),
    "device_cores": Contract("device_cores", "policy",
        ("test/test_grid_stride_execution.py", "test_the_emitted_loop_stride_is_the_launched_program_count"), ""),
    "tuning_status": Contract("tuning_status", "policy", ("test/test_platform_configs.py", None), ""),
    "allowed_kernel_kinds": Contract("allowed_kernel_kinds", "policy",
        ("test/test_cann_bench_generator.py", "test_the_ascend_profile_chunks_wide_reduction_rows"), ""),
    "transpose_mapping": Contract("transpose_mapping", "policy", ("test/test_transpose_mapping.py", None), ""),
    "autotune_status": Contract("autotune_status", "policy", ("test/test_compiler_feedback.py", None), ""),
    "indexed_columns": Contract("indexed_columns", "policy", ("test/test_auto_cross_domain.py", None), ""),
    "indexed_rows": Contract("indexed_rows", "policy", ("test/test_auto_cross_domain.py", None), ""),
    "gather_mapping": Contract("gather_mapping", "policy", ("test/test_gather_tuning.py", None), ""),
    "reduction_group_limit": Contract("reduction_group_limit", "policy",
        ("test/test_reduction_sweep_batches.py", None), ""),
    "coalesce_reduction_sweeps": Contract("coalesce_reduction_sweeps", "policy",
        ("test/test_reduction_sweep_batches.py", None), ""),
    "experimental_region_fusion": Contract("experimental_region_fusion", "policy",
        ("test/test_cross_domain.py", None), ""),
    "strided_reduction_outputs": Contract("strided_reduction_outputs", "policy",
        ("test/test_scalar_row_reductions.py", None), ""),
    "xpu_buffer_size_limit": Contract("xpu_buffer_size_limit", "policy",
        ("test/test_scalar_row_reductions.py", None), ""),
    "eager_row_result_stores": Contract("eager_row_result_stores", "policy",
        ("test/test_scalar_row_reductions.py", None), ""),
    "linear_view_indexing": Contract("linear_view_indexing", "policy",
        ("test/test_linear_view_indexing.py", None), ""),
    "paired_view_loads": Contract("paired_view_loads", "policy",
        ("test/test_broadcast_row_tiling.py", None), ""),
    "split_half_inputs": Contract("split_half_inputs", "policy",
        ("test/test_mask_elision_and_hints.py", None), ""),
    "runtime_validation": Contract("runtime_validation", "policy",
        ("test/test_mask_elision_and_hints.py", None), ""),
    "index_guard": Contract("index_guard", "policy",
        ("test/test_index_width_policy.py", None), "consumed with runtime_validation"),
    "mask_elision": Contract("mask_elision", "policy", ("test/test_mask_elision_and_hints.py", None), ""),
    "layout_on_device": Contract("layout_on_device", "policy",
                                  ("test/test_layout_device.py", "test_the_flag_is_off_by_default"),
                                  "host vs device lowering of the four layout ops; default keeps bytes"),
    "layout_views": Contract("layout_views", "policy",
                             ("test/test_layout_views_contract.py",
                              "test_only_cuda_declares_a_validated_layout_view_contract"),
                             "CUDA default layout-view fusion; non-CUDA stays off "
                             "until its indexed layout contract is device-validated"),
    "streaming_hints": Contract("streaming_hints", "policy",
        ("test/test_mask_elision_and_hints.py", None), ""),
    "broadcast_row_tiling": Contract("broadcast_row_tiling", "policy",
        ("test/test_broadcast_row_tiling.py", None), ""),
}
