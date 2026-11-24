# Trained-model parity

`examples/train_tiny_closed_loop.py` trains one tiny block (D=64, H=128,
vocabulary 128) on a fixed 8-token pattern, then generates tokens twice:

- PyTorch on the full sequence with a causal mask;
- FlowGPU through cached prefill -> masked decode -> LM head -> greedy sampling.

## Result (bf16)

| item | value |
|---|---|
| prompt | `[0..7]` |
| PyTorch generated | `[0, 0, 0, 0, 7]` |
| FlowGPU generated | `[0, 0, 0, 0, 7]` |
| token match | **true** |
| prefill hidden max diff | 0.09375 |
| prefill K max diff | 0.0234375 |
| prefill V max diff | 0.015625 |

A float32 control gives the same tokens and prefill hidden max diff
`2.9e-6`, so the bf16 differences are rounding, not a semantic divergence.

## What this proves

- RoPE, RMSNorm, attention scale, cached prefix, causal mask and cache layout
  agree with an independent PyTorch implementation;
- prefill and masked decode produce the same next-token decisions;
- the full chain can be checked token-for-token, not only by shape/finite
  smoke tests.

## What this does not prove

- language quality (the model is trained on a synthetic pattern);
- performance;
- generality across models, tokenizers or serving policies.

Raw data:
[`tiny-trained-parity-20261125.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/tiny-trained-parity-20261125.json).
