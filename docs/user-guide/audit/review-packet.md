# Review packet: SwiGLU split

The first review packet narrows the entry point to four independent pieces:

| piece | file |
|---|---|
| operator statement | `reference.py` (PyTorch) |
| generated Triton | `generated_module.py` (FlowGPU output) |
| handwritten Triton | `handwritten.py` (two kernels) |
| inputs | `inputs.json` |
| evidence | `manifest.json` |

Run:

```bash
python review/swiglu_split/run_packet.py
```

It checks both implementations against the PyTorch reference and reports the
two layouts:

- **generated**: one primitive kernel; `split` stays as index loads, so there
  are no `gate`/`up` buffers;
- **handwritten**: `split_copy_kernel` materializes `gate`/`up`, then a
  pointwise kernel consumes them.

## Review question

The packet asks:

> If `split` changes from contiguous halves to even/odd columns, which load
> address changes, and which input rows expose the error?

The generated source uses `row * 16 + column` and `row * 16 + 8 + column`; the
handwritten source makes the same two addresses explicit in
`split_copy_kernel`. The DSL line for both `gate` and `up` is line 16
(`gate, up = split(x)`), which the source map records.

## Status

Packet structure, reference and source map are complete. GPU execution is
pending: the remote H100 was unreachable when the packet was assembled. This is
an internal review artifact, not a user-study result.
