"""Torch reference/oracle functions for the device runner.

The oracle runs in float64 so a device result can be checked against one
tolerance definition.  The eager implementation is the dtype-faithful version
that a user would write; it is also what ``torch.compile`` traces.
"""
from __future__ import annotations


def eager(case_name: str, args):
    import torch
    if case_name in {"add_literal", "vectoradd_symbolic"}:
        return {"out": args[0] + args[1]}
    if case_name == "matmul_symbolic":
        return {"out": args[0] @ args[1]}
    if case_name == "softmax_wide":
        return {"q": torch.softmax(args[0], dim=-1)}
    if case_name == "mapping_pointwise":
        return {"z": (args[0] + args[1]) * args[1]}
    if case_name == "neg_literal":
        return {"out": -args[0]}
    if case_name == "pow_literal":
        return {"out": torch.pow(args[0], args[1])}
    if case_name == "mean_last":
        return {"out": torch.mean(args[0], dim=-1, keepdim=True)}
    if case_name == "sort_last":
        return {"out": torch.sort(args[0], dim=-1).values}
    if case_name == "residual_norm":
        x, residual, weight = args
        y = x + residual
        mean = (y.float() * y.float()).mean(dim=-1, keepdim=True)
        z = (y.float() * torch.rsqrt(mean + 1e-6) * weight.float()).to(x.dtype)
        return {"z": z}
    raise ValueError("no torch eager function for " + case_name)


def oracle(case_name: str, args):
    import torch
    if case_name in {"add_literal", "vectoradd_symbolic"}:
        return {"out": args[0].double() + args[1].double()}
    if case_name == "matmul_symbolic":
        return {"out": args[0].double() @ args[1].double()}
    if case_name == "softmax_wide":
        return {"q": torch.softmax(args[0].double(), dim=-1)}
    if case_name == "mapping_pointwise":
        return {"z": (args[0] + args[1]) * args[1]}
    if case_name == "neg_literal":
        return {"out": -args[0].double()}
    if case_name == "pow_literal":
        return {"out": torch.pow(args[0].double(), args[1].double())}
    if case_name == "mean_last":
        return {"out": torch.mean(args[0].double(), dim=-1, keepdim=True)}
    if case_name == "sort_last":
        return {"out": torch.sort(args[0].double(), dim=-1).values}
    if case_name == "residual_norm":
        x, residual, weight = args
        y = x.double() + residual.double()
        mean = (y * y).mean(dim=-1, keepdim=True)
        return {"z": y * torch.rsqrt(mean + 1e-6) * weight.double()}
    raise ValueError("no torch oracle for " + case_name)
