"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

SM90 (Hopper) CuTe-DSL routed-experts MoE with block-FP8 (W8A8) weights.

Inputs follow the usual fused-MoE contract: bf16 hidden states, the router's
top-k ids/weights, FP8 E4M3 ``w13`` ``[E, 2I, H]`` (gate rows first, then up)
and ``w2`` ``[E, H, I]`` with fp32 128x128 block scales. Activations are
quantized inside the kernels (dynamic 1x128, fp32 scales), the activation is
SiLU-GLU, and the top-k reduction is done in fp32.

Two kernel families, chosen by batch size:
  * M < 2048: route (+ fused 1x128 quant) and one fused GEMM1 + SiLU-GLU +
    requant + GEMM2 kernel (swap-AB WGMMA, weights on the MMA M side).
  * M >= 2048: multi-CTA route + token-order quant, a persistent non-swapped
    GEMM1/GEMM2 kernel, and an fp32 -> bf16 finalize.

The kernels are currently specialized to E=256, top_k=8, hidden=2048,
intermediate=512 (Qwen3.5/3.6-35B-A3B routed experts at TP1).
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import triton
import triton.language as tl
import tvm_ffi

from .hopper import fp8_block_moe_decode as _dec
from .hopper import fp8_block_moe_prefill as _pre

NUM_EXPERTS = _dec.E
TOP_K = _dec.TOPK
HIDDEN_SIZE = _dec.H
INTERMEDIATE_SIZE = _dec.I
BLOCK_SIZE = 128

_PREFILL_MIN_M = 2048
_PREFILL_BT = 128
_PREFILL_STAGES = 3
_ROUTE_MC_MIN_PAIRS = 8192
_FUSED_ROUTE_MAX_PAIRS = 4096
_ALIGN = 256


def is_sm90_fp8_block_moe_supported(
    num_experts: int, top_k: int, hidden_size: int, intermediate_size: int
) -> bool:
    """Whether the kernels support this routed-experts shape."""
    return (num_experts, top_k, hidden_size, intermediate_size) == (
        NUM_EXPERTS,
        TOP_K,
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
    )


@dataclass
class Sm90Fp8BlockMoEWeights:
    """Weights re-laid out for the kernels by ``prepare_sm90_fp8_block_moe_weights``.

    ``w13_decode``: gate/up rows interleaved in 8-row groups, tile-major
    256x128 boxes (small-batch kernel). ``w13_prefill``: 256x128 panels ordered
    [expert, 128-h tile, 128-k block, gate/up, row, k] (large-batch kernel).
    ``w2``: tile-major 256x128 boxes (both kernels).
    """

    w13_decode: torch.Tensor
    w13_prefill: torch.Tensor
    w2: torch.Tensor


def _check_weights(w13: torch.Tensor, w2: torch.Tensor) -> None:
    E, I, H = NUM_EXPERTS, INTERMEDIATE_SIZE, HIDDEN_SIZE
    if w13.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
        raise ValueError("w13 and w2 must be float8_e4m3fn")
    if tuple(w13.shape) != (E, 2 * I, H) or tuple(w2.shape) != (E, H, I):
        raise ValueError(
            f"expected w13 {(E, 2 * I, H)} and w2 {(E, H, I)}, got "
            f"{tuple(w13.shape)} and {tuple(w2.shape)}"
        )


@torch.no_grad()
def prepare_sm90_fp8_block_moe_weights(
    w13: torch.Tensor, w2: torch.Tensor
) -> Sm90Fp8BlockMoEWeights:
    """Re-lay out ``w13`` ``[E, 2I, H]`` and ``w2`` ``[E, H, I]`` once at load time.

    The 128x128 block scales keep their checkpoint layout.
    """
    _check_weights(w13, w2)
    E, I, H = NUM_EXPERTS, INTERMEDIATE_SIZE, HIDDEN_SIZE
    w13 = w13.contiguous()
    w2 = w2.contiguous()
    # gate/up interleave: row (mt, slab, g, r) <- w13 row g*I + mt*128 + slab*8 + r
    w13i = (
        w13.view(E, 2, 4, 16, 8, H)
        .permute(0, 2, 3, 1, 4, 5)
        .reshape(E * 4, 256, H // 128, 128)
    )
    w13_decode = w13i.permute(0, 2, 1, 3).contiguous().view(-1, 128)
    del w13i
    w13_prefill = (
        w13.view(E, 2, 4, 128, H // 128, 128)
        .permute(0, 2, 4, 1, 3, 5)
        .contiguous()
        .view(-1, 128)
    )
    w2_packed = (
        w2.view(E * (H // 256), 256, I // 128, 128)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(-1, 128)
    )
    return Sm90Fp8BlockMoEWeights(w13_decode, w13_prefill, w2_packed)


_Layout = Tuple[Tuple[str, Tuple[int, ...], torch.dtype], ...]


def _decode_layout(M: int, BN: int) -> _Layout:
    E, H, I = NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE
    P = M * TOP_K
    PS = (P + 3) // 4 * 4
    max_items = (P + BN - 1) // BN + min(E, P)
    i32, f32, fp8 = torch.int32, torch.float32, torch.float8_e4m3fn
    return (
        ("expert_off", (E + 1,), i32),
        ("work", (max_items, 4), i32),
        ("dep", (max_items,), i32),
        ("pos_of_pair", (P,), i32),
        ("sorted_tok", (P,), i32),
        ("sorted_wt", (P,), f32),
        ("xq", (P, H), fp8),
        ("xsT", (H // 128, PS), f32),
        ("hq", (P, I), fp8),
        ("hsT", (I // 128, PS), f32),
        ("accbuf", (M, H), f32),
        ("hist", (_dec.RMC_CTAS, E), i32),
    )


def _prefill_layout(M: int) -> _Layout:
    E, H, I = NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE
    BN = _PREFILL_BT
    P = M * TOP_K
    PS = (P + 3) // 4 * 4
    max_items = (P + BN - 1) // BN + min(E, P)
    i32, f32, fp8 = torch.int32, torch.float32, torch.float8_e4m3fn
    return (
        ("expert_off", (E + 1,), i32),
        ("work", (max_items + 16, 4), i32),
        ("dep", (max_items + 16,), i32),
        ("pos_of_pair", (P,), i32),
        ("sorted_tok", (P,), i32),
        ("sorted_wt", (P,), f32),
        ("hq", (P, I), fp8),
        ("hsT", (I // 128, PS), f32),
        ("accbuf", (M, H), f32),
        ("hist", (_pre.RMC_CTAS, E), i32),
        ("xqc", (M, H), fp8),
        ("xsc", (M, H // 128), f32),
    )


def _layout_bytes(layout: _Layout) -> int:
    off = 0
    for _, shape, dtype in layout:
        n = dtype.itemsize
        for d in shape:
            n *= d
        off = (off + n + _ALIGN - 1) // _ALIGN * _ALIGN
    return off


def _carve(pool: torch.Tensor, layout: _Layout) -> Dict[str, torch.Tensor]:
    views: Dict[str, torch.Tensor] = {}
    off = 0
    for name, shape, dtype in layout:
        n = dtype.itemsize
        for d in shape:
            n *= d
        views[name] = pool[off : off + n].view(dtype).view(shape)
        off = (off + n + _ALIGN - 1) // _ALIGN * _ALIGN
    return views


class Sm90Fp8BlockMoEWorkspace:
    """Scratch for up to ``max_num_tokens`` tokens per call.

    One workspace can serve every MoE layer that runs on the same stream (the
    calls are serialized). It also holds the small persistent counters the
    kernels re-arm at the end of each call, so it must outlive any CUDA graph
    that captured a call using it.
    """

    def __init__(self, max_num_tokens: int, device: Union[torch.device, str, int]):
        if max_num_tokens <= 0:
            raise ValueError("max_num_tokens must be positive")
        self.max_num_tokens = max_num_tokens
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        candidates = []
        for m in (64, _PREFILL_MIN_M - 1):
            m = min(m, max_num_tokens)
            candidates.append(_layout_bytes(_decode_layout(m, _dec._pick_bn(m))))
        if max_num_tokens >= _PREFILL_MIN_M:
            candidates.append(_layout_bytes(_prefill_layout(max_num_tokens)))
        self.pool = torch.empty(max(candidates), dtype=torch.uint8, device=self.device)
        # Per-kernel-family meta/grid-barrier counters: zero at first use, re-armed
        # by every call.
        self._counters = {
            key: (
                torch.zeros(4, dtype=torch.int32, device=self.device),
                torch.zeros(2, dtype=torch.int32, device=self.device),
            )
            for key in (("decode", 16), ("decode", 32), ("decode", 64), ("prefill",))
        }
        self.num_sms = torch.cuda.get_device_properties(
            self.device
        ).multi_processor_count

    @property
    def nbytes(self) -> int:
        return self.pool.numel()


_default_workspaces: Dict[torch.device, Sm90Fp8BlockMoEWorkspace] = {}


def _default_workspace(device: torch.device, M: int) -> Sm90Fp8BlockMoEWorkspace:
    ws = _default_workspaces.get(device)
    if ws is None or ws.max_num_tokens < M:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "cute_dsl_fused_moe_fp8_block_scale_sm90 needs a workspace for "
                f"{M} tokens during CUDA graph capture; pass one sized for the "
                "largest batch"
            )
        ws = Sm90Fp8BlockMoEWorkspace(max(M, 1), device)
        _default_workspaces[device] = ws
    return ws


def _check_inputs(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    weights: Sm90Fp8BlockMoEWeights,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    E, H, I = NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE
    M = hidden_states.shape[0]
    if hidden_states.dtype != torch.bfloat16 or tuple(hidden_states.shape) != (M, H):
        raise ValueError(f"hidden_states must be bf16 [M, {H}]")
    if not hidden_states.is_contiguous() or not out.is_contiguous():
        raise ValueError("hidden_states and out must be contiguous")
    if out.dtype != torch.bfloat16 or tuple(out.shape) != (M, H):
        raise ValueError(f"out must be bf16 [{M}, {H}]")
    if topk_ids.dtype != torch.int32 or tuple(topk_ids.shape) != (M, TOP_K):
        raise ValueError(f"topk_ids must be int32 [{M}, {TOP_K}]")
    if topk_weights.dtype != torch.float32 or tuple(topk_weights.shape) != (M, TOP_K):
        raise ValueError(f"topk_weights must be float32 [{M}, {TOP_K}]")
    if not topk_ids.is_contiguous() or not topk_weights.is_contiguous():
        raise ValueError("topk_ids and topk_weights must be contiguous")
    if tuple(w13_scale.shape) != (E, 2 * I // BLOCK_SIZE, H // BLOCK_SIZE) or tuple(
        w2_scale.shape
    ) != (E, H // BLOCK_SIZE, I // BLOCK_SIZE):
        raise ValueError("w13_scale/w2_scale must be the 128x128 block scales")
    if w13_scale.dtype != torch.float32 or w2_scale.dtype != torch.float32:
        raise ValueError("w13_scale and w2_scale must be float32")
    if not w13_scale.is_contiguous() or not w2_scale.is_contiguous():
        raise ValueError("w13_scale and w2_scale must be contiguous")
    if hidden_states.device.type != "cuda" or torch.cuda.get_device_capability(
        hidden_states.device
    ) != (9, 0):
        raise RuntimeError("cute_dsl_fused_moe_fp8_block_scale_sm90 requires SM90")
    n13 = E * 2 * I * (H // 128)
    n2 = E * H * (I // 128)
    if (
        tuple(weights.w13_decode.shape) != (n13, 128)
        or tuple(weights.w13_prefill.shape) != (n13, 128)
        or tuple(weights.w2.shape) != (n2, 128)
    ):
        raise ValueError("weights must come from prepare_sm90_fp8_block_moe_weights")


@triton.jit
def _mask_unrouted_kernel(
    ids_ptr, wts_ptr, ids_out_ptr, wts_out_ptr, n, num_experts, BLOCK: tl.constexpr
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    e = tl.load(ids_ptr + offs, mask=m, other=0)
    w = tl.load(wts_ptr + offs, mask=m, other=0.0)
    bad = (e < 0) | (e >= num_experts)
    tl.store(ids_out_ptr + offs, tl.where(bad, 0, e), mask=m)
    tl.store(wts_out_ptr + offs, tl.where(bad, 0.0, w), mask=m)


def _mask_unrouted(
    topk_ids: torch.Tensor, topk_weights: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Send unrouted slots (ids outside [0, E), e.g. vLLM's -1 padding) to expert 0
    with zero weight, which equals skipping them. The routing kernels index
    per-expert tables with the raw id."""
    ids = torch.empty_like(topk_ids)
    wts = torch.empty_like(topk_weights)
    n = topk_ids.numel()
    BLOCK = 1024
    _mask_unrouted_kernel[(triton.cdiv(n, BLOCK),)](
        topk_ids, topk_weights, ids, wts, n, NUM_EXPERTS, BLOCK=BLOCK
    )
    return ids, wts


@torch.no_grad()
def cute_dsl_fused_moe_fp8_block_scale_sm90(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    weights: Sm90Fp8BlockMoEWeights,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    workspace: Optional[Sm90Fp8BlockMoEWorkspace] = None,
) -> torch.Tensor:
    """Routed-experts MoE: ``out[m] = sum_k w[m,k] * W2_e(silu(G_e x) * U_e x)``.

    Args:
        hidden_states: bf16 ``[M, H]``.
        topk_ids: int32 ``[M, top_k]`` expert ids; slots with an id outside
            ``[0, E)`` (e.g. -1 padding) are unrouted and contribute nothing.
        topk_weights: float32 ``[M, top_k]`` router weights.
        weights: output of ``prepare_sm90_fp8_block_moe_weights``.
        w13_scale: float32 ``[E, 2I/128, H/128]`` block scales of ``w13``.
        w2_scale: float32 ``[E, H/128, I/128]`` block scales of ``w2``.
        out: optional bf16 ``[M, H]`` output.
        workspace: scratch sized for at least ``M`` tokens; required when the
            call is captured into a CUDA graph.

    Runs on the current CUDA stream.
    """
    M = hidden_states.shape[0]
    if out is None:
        out = torch.empty_like(hidden_states)
    _check_inputs(
        hidden_states, topk_ids, topk_weights, weights, w13_scale, w2_scale, out
    )
    if M == 0:
        return out
    ws = (
        workspace
        if workspace is not None
        else _default_workspace(hidden_states.device, M)
    )
    if ws.max_num_tokens < M:
        raise ValueError(
            f"workspace holds {ws.max_num_tokens} tokens, the call has {M}"
        )
    # The kernels launch on TVM-FFI's environment stream; pin it to the torch
    # stream so they stay ordered with (and are captured alongside) the caller.
    topk_ids, topk_weights = _mask_unrouted(topk_ids, topk_weights)
    with tvm_ffi.use_torch_stream():
        _dispatch(hidden_states, topk_ids, topk_weights, weights, w13_scale,
                  w2_scale, out, ws)  # fmt: skip
    return out


def _dispatch(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    weights: Sm90Fp8BlockMoEWeights,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    out: torch.Tensor,
    ws: Sm90Fp8BlockMoEWorkspace,
) -> None:
    M = hidden_states.shape[0]
    P = M * TOP_K
    ids = topk_ids.view(P)
    wts = topk_weights.view(P)
    nsm = ws.num_sms

    if M >= _PREFILL_MIN_M:
        meta, gbar = ws._counters[("prefill",)]
        s = _carve(ws.pool, _prefill_layout(M))
        _pre._get_compiled(("route_mc_quantc", _PREFILL_BT))(
            ids, wts, s["expert_off"], s["work"], meta, s["sorted_tok"],
            s["sorted_wt"], s["dep"], s["hist"], gbar,
            hidden_states, s["xqc"], s["xsc"], P, M, nsm,
        )  # fmt: skip
        # Keep the fp32 accumulator slice of one work item within ~12 MiB of L2.
        g2 = 2 if M * 2 * 256 * 4 <= (12 << 20) else 1
        producer_heavy = M * 256 * 4 > (12 << 20)
        _pre._get_compiled(("prefill", _PREFILL_STAGES, g2, producer_heavy))(
            s["xqc"], s["xsc"], weights.w13_prefill, w13_scale,
            s["hq"], s["hsT"].t(), weights.w2, w2_scale,
            s["work"], meta, s["dep"], s["accbuf"], s["sorted_tok"], s["sorted_wt"],
            out, nsm,
        )  # fmt: skip
        _pre._get_compiled(("finalize",))(s["accbuf"], out, M)
        return

    BN = _dec._pick_bn(M)
    meta, gbar = ws._counters[("decode", BN)]
    s = _carve(ws.pool, _decode_layout(M, BN))
    if P <= _FUSED_ROUTE_MAX_PAIRS:
        # One token per CTA; the kernel also supports several tokens per CTA pass.
        nqc = M
        nq = (M + nqc - 1) // nqc
        nch = (P // 4 + _dec.QUANT_THREADS - 1) // _dec.QUANT_THREADS
        _dec._get_compiled(("quant_route", BN, nq, nch))(
            hidden_states, ids, wts, s["expert_off"], s["work"], meta, s["sorted_tok"],
            s["sorted_wt"], s["dep"], s["xq"], s["xsT"], s["accbuf"], nqc,
        )  # fmt: skip
    elif P > _ROUTE_MC_MIN_PAIRS:
        _dec._get_compiled(("route_mc", BN))(
            ids, wts, s["expert_off"], s["work"], meta, s["pos_of_pair"],
            s["sorted_tok"], s["sorted_wt"], s["dep"], s["hist"], gbar, P,
        )  # fmt: skip
        _dec._get_compiled(("quant",))(
            hidden_states, s["pos_of_pair"], s["xq"], s["xsT"], s["accbuf"], M
        )
    else:
        _dec._get_compiled(("route", BN))(
            ids, wts, s["expert_off"], s["work"], meta, s["pos_of_pair"],
            s["sorted_tok"], s["sorted_wt"], s["dep"], P,
        )  # fmt: skip
        _dec._get_compiled(("quant",))(
            hidden_states, s["pos_of_pair"], s["xq"], s["xsT"], s["accbuf"], M
        )
    _dec._get_compiled(("fused", BN, _dec._pick_stages(BN)))(
        weights.w13_decode, s["xq"], s["xsT"].t(), w13_scale,
        weights.w2, s["hq"], s["hsT"].t(), w2_scale,
        s["expert_off"], s["work"], meta, s["dep"],
        s["accbuf"], out, s["sorted_tok"], s["sorted_wt"], ids,
        1 if P <= _FUSED_ROUTE_MAX_PAIRS else 0, nsm,
    )  # fmt: skip


__all__ = [
    "Sm90Fp8BlockMoEWeights",
    "Sm90Fp8BlockMoEWorkspace",
    "cute_dsl_fused_moe_fp8_block_scale_sm90",
    "is_sm90_fp8_block_moe_supported",
    "prepare_sm90_fp8_block_moe_weights",
]
