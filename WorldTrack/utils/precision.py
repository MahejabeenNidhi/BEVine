# WorldTrack/utils/precision.py
"""
Thin wrappers for numerically sensitive ops that do not support
half-precision natively. Each wrapper promotes to fp32 only when
the input is already in a reduced-precision dtype, so they are
transparent for fp32 inputs (mmCows) and safe for bf16 (JerCCows).
"""
import torch


def fp32_inverse(x: torch.Tensor) -> torch.Tensor:
    """
    Matrix inverse computed in fp32 for numerical stability.
    Works transparently for fp32, bf16, and fp16 inputs.
    """
    orig = x.dtype
    if orig in (torch.float16, torch.bfloat16):
        return torch.inverse(x.float()).to(orig)
    return torch.inverse(x)


def fp32_so3_exp(v: torch.Tensor) -> torch.Tensor:
    """
    Rodrigues exponential map in fp32.
    Delegates to geom.so3_exp after an fp32 upcast.
    """
    from utils.geom import so3_exp
    orig = v.dtype
    if orig in (torch.float16, torch.bfloat16):
        return so3_exp(v.float()).to(orig)
    return so3_exp(v)
