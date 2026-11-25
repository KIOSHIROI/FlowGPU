# FlowGPU lessons

The ladder follows dataflow complexity, not operator difficulty:

1. values / shapes / broadcast — `01-values/`
2. layout: split, concat, interleave
3. simple reductions
4. matmul
5. fusion
6. materialized vs fused
7. address and masks
8. handwritten Triton comparison
9. a small operator review packet

Each lesson answers one question before showing code:

> Where does this value come from, is it materialized, how is its address
> computed, and how is the tail masked?

Start with `01-values/README.md`. No GPU is needed for Lesson 1.
