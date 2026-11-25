# Lesson 4: matmul

Entry point: [`lessons/04-matmul/`](https://github.com/KIOSHIROI/FlowGPU/tree/main/lessons/04-matmul).

```fgpu
pipeline matmul_intro {
    input a : fp32[M,K]
    input b : fp32[K,N]
    out = matmul(a, b)
    output out
}
```

Shape flow:

```text
a   [M,K]
b   [K,N]
out [M,N]
```

`K` is the reduction axis; `M` and `N` survive. Addresses:

```text
a[i,k]   = a[i*K + k]
b[k,j]   = b[k*N + j]
out[i,j] = out[i*N + j]
```

The compiler selects a `BLOCK_M/BLOCK_N/BLOCK_K` tile. With the default CPU
policy it prints:

```text
kernels: 1
tile: BLOCK_M=128 BLOCK_N=128 BLOCK_K=64
```

For the tiny lesson shape `[4,8] @ [8,16]`, one tile covers the whole output
and the masks protect the unused rows/columns. That is the bridge to Lesson 7
(address and masks).

Run without a GPU:

```bash
python lessons/04-matmul/run_lesson.py
```
