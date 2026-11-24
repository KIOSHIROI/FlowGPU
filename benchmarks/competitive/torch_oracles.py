"""Torch reference/oracle functions for the device runner.

The oracle runs in float64 so a device result can be checked against one
tolerance definition.  The eager implementation is the dtype-faithful version
that a user would write; it is also what ``torch.compile`` traces.
"""
from __future__ import annotations


def eager(case_name: str, args):
    import torch
    if case_name == "decode_gqa_nocopy":
        q, k_cache, v_cache, k_new, v_new, loc = args
        kc = k_cache.float().clone(); vc = v_cache.float().clone()
        kc[:, loc[0]] = k_new.float(); vc[:, loc[0]] = v_new.float()
        kk = kc.repeat_interleave(2, dim=0); vv = vc.repeat_interleave(2, dim=0)
        scores = q.float() @ kk.transpose(-1, -2) / (q.shape[-1] ** 0.5)
        return {"out": (torch.softmax(scores, dim=-1) @ vv).to(q.dtype)}
    if case_name == "decode_batch_gqa":
        q, k_cache, v_cache, k_new, v_new, loc = args
        kc = k_cache.float().clone(); vc = v_cache.float().clone()
        for b in range(kc.shape[0]):
            kc[b, :, loc[b]] = k_new[b].float()
            vc[b, :, loc[b]] = v_new[b].float()
        kk = kc.repeat_interleave(2, dim=1); vv = vc.repeat_interleave(2, dim=1)
        scores = q.float() @ kk.transpose(-1, -2) / (q.shape[-1] ** 0.5)
        return {"out": (torch.softmax(scores, dim=-1) @ vv).to(q.dtype)}
    if case_name == "decode_gqa_cache":
        q, k_cache, v_cache, k_new, v_new, loc = args
        kc = k_cache.float().clone(); vc = v_cache.float().clone()
        kc[:, loc[0]] = k_new.float(); vc[:, loc[0]] = v_new.float()
        kk = kc.unsqueeze(0).repeat_interleave(2, dim=1)
        vv = vc.unsqueeze(0).repeat_interleave(2, dim=1)
        scores = q.float() @ kk.transpose(-1, -2) / (q.shape[-1] ** 0.5)
        return {"out": (torch.softmax(scores, dim=-1) @ vv).to(q.dtype)}
    if case_name == "decode_with_cache":
        q, k_cache, v_cache, k_new, v_new, loc = args
        kc = k_cache.float().clone()
        vc = v_cache.float().clone()
        kc[loc[0], 0] = k_new[0].float()
        vc[loc[0], 0] = v_new[0].float()
        scores = q.float() @ kc[:, 0].t() / (q.shape[-1] ** 0.5)
        return {"out": (torch.softmax(scores, dim=-1) @ vc[:, 0]).to(q.dtype)}
    if case_name == "causal_attention_gqa":
        q, k, v = args
        h, g = q.shape[1], k.shape[1]
        kk = k.repeat_interleave(h // g, dim=1)
        vv = v.repeat_interleave(h // g, dim=1)
        scores = q.float() @ kk.float().transpose(-1, -2) / (q.shape[-1] ** 0.5)
        mask = torch.ones(q.shape[-2], k.shape[-2], device=q.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
        return {"out": (torch.softmax(scores, dim=-1) @ vv.float()).to(q.dtype)}
    if case_name == "decode_attention":
        q, k_cache, v_cache = args
        return {"out": torch.softmax(q @ k_cache.t() / (q.shape[-1] ** 0.5), dim=-1) @ v_cache}
    if case_name == "llama3_decoder_layer":
        x, norm1_w, w_q, w_k, w_v, w_o, norm2_w, w_gate, w_up, w_down, cos, sin = args
        def rms(v, w):
            vf = v.float()
            return vf * torch.rsqrt((vf * vf).mean(dim=-1, keepdim=True) + 1e-6) * w.float()
        def rope(v, c, s):
            d = v.shape[-1]
            first = torch.arange(d, device=v.device) < d // 2
            partner = torch.where(first, torch.arange(d, device=v.device) + d // 2,
                                  torch.arange(d, device=v.device) - d // 2)
            vp = v.index_select(-1, partner)
            sign = torch.where(first, -1.0, 1.0)
            return v * c.float() + sign * vp * s.float()
        n1 = rms(x, norm1_w)
        q, k, v = n1 @ w_q.float(), n1 @ w_k.float(), n1 @ w_v.float()
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        attn = torch.softmax(q @ k.t(), dim=-1) @ v
        x_mid = x.float() + attn @ w_o.float()
        n2 = rms(x_mid, norm2_w)
        hidden = torch.nn.functional.silu(n2 @ w_gate.float()) * (n2 @ w_up.float())
        return {"out": (x_mid + hidden @ w_down.float()).to(x.dtype)}
    if case_name == "qk_transform":
        q, k, qw, kw, cos, sin = args
        def rms(x, w):
            xf = x.float()
            return (xf * torch.rsqrt((xf * xf).mean(dim=-1, keepdim=True) + 1e-6) * w.float())
        def rope(x, c, s):
            d = x.shape[-1]
            first = torch.arange(d, device=x.device) < d // 2
            partner = torch.where(first, torch.arange(d, device=x.device) + d // 2,
                                  torch.arange(d, device=x.device) - d // 2)
            xp = x.index_select(-1, partner)
            sign = torch.where(first, -1.0, 1.0)
            return (x * c.float() + sign * xp * s.float()).to(x.dtype)
        return {"qr": rope(rms(q, qw), cos, sin)}
    if case_name == "topk_last":
        return {"values": torch.topk(args[0], 8, dim=-1).values}
    if case_name == "moe_router_topk_large":
        x, router_weight, cap = args
        logits = x @ router_weight.t()
        capped = torch.tanh(logits / cap) * cap
        probs = torch.softmax(capped, dim=-1)
        values, ids = torch.topk(capped, 8, dim=-1)
        return {"weights": torch.gather(probs, -1, ids)}
    if case_name == "moe_router_topk":
        x, router_weight, cap = args
        logits = x @ router_weight.t()
        capped = torch.tanh(logits / cap) * cap
        probs = torch.softmax(capped, dim=-1)
        values, ids = torch.topk(capped, 2, dim=-1)
        return {"weights": torch.gather(probs, -1, ids)}
    if case_name == "conv2d_stride2_same":
        import torch.nn.functional as F
        x, weight, bias = args
        return {"out": F.conv2d(x.permute(0, 3, 1, 2).float(), weight.float(),
                                bias.float(), stride=2, padding=1).permute(0, 2, 3, 1).to(x.dtype)}
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
    if case_name == "conv2d_stride2":
        import torch.nn.functional as F
        x, weight, bias = args
        return {"out": F.conv2d(x.permute(0, 3, 1, 2).float(), weight.float(),
                                bias.float(), stride=2).permute(0, 2, 3, 1).to(x.dtype)}
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
    if case_name == "decode_gqa_nocopy":
        q, k_cache, v_cache, k_new, v_new, loc = args
        kc = k_cache.double().clone(); vc = v_cache.double().clone()
        kc[:, loc[0]] = k_new.double(); vc[:, loc[0]] = v_new.double()
        kk = kc.repeat_interleave(2, dim=0); vv = vc.repeat_interleave(2, dim=0)
        scores = q.double() @ kk.transpose(-1, -2) / (q.shape[-1] ** 0.5)
        return {"out": torch.softmax(scores, dim=-1) @ vv}
    if case_name == "decode_batch_gqa":
        q, k_cache, v_cache, k_new, v_new, loc = args
        kc = k_cache.double().clone(); vc = v_cache.double().clone()
        for b in range(kc.shape[0]):
            kc[b, :, loc[b]] = k_new[b].double()
            vc[b, :, loc[b]] = v_new[b].double()
        kk = kc.repeat_interleave(2, dim=1); vv = vc.repeat_interleave(2, dim=1)
        scores = q.double() @ kk.transpose(-1, -2) / (q.shape[-1] ** 0.5)
        return {"out": torch.softmax(scores, dim=-1) @ vv}
    if case_name == "decode_gqa_cache":
        q, k_cache, v_cache, k_new, v_new, loc = args
        kc = k_cache.double().clone(); vc = v_cache.double().clone()
        kc[:, loc[0]] = k_new.double(); vc[:, loc[0]] = v_new.double()
        kk = kc.unsqueeze(0).repeat_interleave(2, dim=1)
        vv = vc.unsqueeze(0).repeat_interleave(2, dim=1)
        scores = q.double() @ kk.transpose(-1, -2) / (q.shape[-1] ** 0.5)
        return {"out": torch.softmax(scores, dim=-1) @ vv}
    if case_name == "decode_with_cache":
        q, k_cache, v_cache, k_new, v_new, loc = args
        kc = k_cache.double().clone()
        vc = v_cache.double().clone()
        kc[loc[0], 0] = k_new[0].double()
        vc[loc[0], 0] = v_new[0].double()
        scores = q.double() @ kc[:, 0].t() / (q.shape[-1] ** 0.5)
        return {"out": torch.softmax(scores, dim=-1) @ vc[:, 0]}
    if case_name == "causal_attention_gqa":
        q, k, v = args
        h, g = q.shape[1], k.shape[1]
        kk = k.repeat_interleave(h // g, dim=1)
        vv = v.repeat_interleave(h // g, dim=1)
        scores = q.double() @ kk.double().transpose(-1, -2) / (q.shape[-1] ** 0.5)
        mask = torch.ones(q.shape[-2], k.shape[-2], device=q.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
        return {"out": torch.softmax(scores, dim=-1) @ vv.double()}
    if case_name == "decode_attention":
        q, k_cache, v_cache = args
        return {"out": torch.softmax(q.double() @ k_cache.double().t() / (q.shape[-1] ** 0.5), dim=-1) @ v_cache.double()}
    if case_name == "llama3_decoder_layer":
        x, norm1_w, w_q, w_k, w_v, w_o, norm2_w, w_gate, w_up, w_down, cos, sin = args
        def rms(v, w):
            vf = v.double()
            return vf * torch.rsqrt((vf * vf).mean(dim=-1, keepdim=True) + 1e-6) * w.double()
        def rope_fn(v, c, s):
            d = v.shape[-1]
            first = torch.arange(d, device=v.device) < d // 2
            partner = torch.where(first, torch.arange(d, device=v.device) + d // 2,
                                  torch.arange(d, device=v.device) - d // 2)
            vp = v.index_select(-1, partner)
            sign = torch.where(first, -1.0, 1.0)
            return v * c.double() + sign * vp * s.double()
        n1 = rms(x, norm1_w)
        q, k, v = n1 @ w_q.double(), n1 @ w_k.double(), n1 @ w_v.double()
        q, k = rope_fn(q, cos, sin), rope_fn(k, cos, sin)
        attn = torch.softmax(q @ k.t(), dim=-1) @ v
        x_mid = x.double() + attn @ w_o.double()
        n2 = rms(x_mid, norm2_w)
        hidden = torch.nn.functional.silu(n2 @ w_gate.double()) * (n2 @ w_up.double())
        return {"out": x_mid + hidden @ w_down.double()}
    if case_name == "qk_transform":
        q, k, qw, kw, cos, sin = args
        def rms(x, w):
            xf = x.double()
            return xf * torch.rsqrt((xf * xf).mean(dim=-1, keepdim=True) + 1e-6) * w.double()
        def rope(x, c, s):
            d = x.shape[-1]
            first = torch.arange(d, device=x.device) < d // 2
            partner = torch.where(first, torch.arange(d, device=x.device) + d // 2,
                                  torch.arange(d, device=x.device) - d // 2)
            xp = x.index_select(-1, partner)
            sign = torch.where(first, -1.0, 1.0)
            return x * c.double() + sign * xp * s.double()
        return {"qr": rope(rms(q, qw), cos, sin)}
    if case_name == "topk_last":
        return {"values": torch.topk(args[0].double(), 8, dim=-1).values}
    if case_name == "moe_router_topk_large":
        x, router_weight, cap = args
        logits = x.double() @ router_weight.double().t()
        capped = torch.tanh(logits / cap.double()) * cap.double()
        probs = torch.softmax(capped, dim=-1)
        values, ids = torch.topk(capped, 8, dim=-1)
        return {"weights": torch.gather(probs, -1, ids)}
    if case_name == "moe_router_topk":
        x, router_weight, cap = args
        logits = x.double() @ router_weight.double().t()
        capped = torch.tanh(logits / cap.double()) * cap.double()
        probs = torch.softmax(capped, dim=-1)
        values, ids = torch.topk(capped, 2, dim=-1)
        return {"weights": torch.gather(probs, -1, ids)}
    if case_name == "conv2d_stride2_same":
        import torch.nn.functional as F
        x, weight, bias = args
        return {"out": F.conv2d(x.double().permute(0, 3, 1, 2), weight.double(),
                                bias.double(), stride=2, padding=1).permute(0, 2, 3, 1)}
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
    if case_name == "conv2d_stride2":
        import torch.nn.functional as F
        x, weight, bias = args
        return {"out": F.conv2d(x.double().permute(0, 3, 1, 2), weight.double(),
                                bias.double(), stride=2).permute(0, 2, 3, 1)}
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
