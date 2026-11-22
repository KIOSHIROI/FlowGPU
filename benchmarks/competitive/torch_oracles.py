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
    if case_name == "batch_norm":
        x, mean, var, weight, bias, eps = args
        xf = x.float()
        return {"out": ((xf - mean.float()) / torch.sqrt(var.float() + eps) * weight.float()
                        + bias.float()).to(x.dtype)}
    if case_name == "conv2d_same":
        import torch.nn.functional as F
        x, weight, bias = args
        return {"out": F.conv2d(x.permute(0, 3, 1, 2).float(), weight.float(),
                                bias.float(), padding=1).permute(0, 2, 3, 1).to(x.dtype)}
    if case_name == "conv2d_3x3":
        import torch.nn.functional as F
        x, weight, bias = args
        return {"out": F.conv2d(x.permute(0, 3, 1, 2).float(), weight.float(),
                                bias.float()).permute(0, 2, 3, 1).to(x.dtype)}
    if case_name == "conv2d_1x1":
        x, weight, bias = args
        n, h, w, c = x.shape
        out = x.reshape(n * h * w, c).float() @ weight.float().t() + bias.float()
        return {"out": out.reshape(n, h, w, weight.shape[0]).to(x.dtype)}
    if case_name == "layer_norm":
        x, weight, bias, eps = args
        xf = x.float()
        mean = xf.mean(dim=-1, keepdim=True)
        var = ((xf - mean) ** 2).mean(dim=-1, keepdim=True)
        return {"out": ((xf - mean) / torch.sqrt(var + eps) * weight.float()
                        + bias.float()).to(x.dtype)}
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
    if case_name == "batch_norm":
        x, mean, var, weight, bias, eps = args
        xf = x.double()
        return {"out": ((xf - mean.double()) / torch.sqrt(var.double() + eps)
                        * weight.double() + bias.double())}
    if case_name == "conv2d_same":
        import torch.nn.functional as F
        x, weight, bias = args
        return {"out": F.conv2d(x.double().permute(0, 3, 1, 2), weight.double(),
                                bias.double(), padding=1).permute(0, 2, 3, 1)}
    if case_name == "conv2d_3x3":
        import torch.nn.functional as F
        x, weight, bias = args
        return {"out": F.conv2d(x.double().permute(0, 3, 1, 2), weight.double(),
                                bias.double()).permute(0, 2, 3, 1)}
    if case_name == "conv2d_1x1":
        x, weight, bias = args
        n, h, w, c = x.shape
        out = x.double().reshape(n * h * w, c) @ weight.double().t() + bias.double()
        return {"out": out.reshape(n, h, w, weight.shape[0])}
    if case_name == "layer_norm":
        x, weight, bias, eps = args
        xf = x.double()
        mean = xf.mean(dim=-1, keepdim=True)
        var = ((xf - mean) ** 2).mean(dim=-1, keepdim=True)
        return {"out": ((xf - mean) / torch.sqrt(var + eps) * weight.double()
                        + bias.double())}
    if case_name == "sort_last":
        return {"out": torch.sort(args[0].double(), dim=-1).values}
    if case_name == "residual_norm":
        x, residual, weight = args
        y = x.double() + residual.double()
        mean = (y * y).mean(dim=-1, keepdim=True)
        return {"z": y * torch.rsqrt(mean + 1e-6) * weight.double()}
    raise ValueError("no torch oracle for " + case_name)
