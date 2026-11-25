# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ruff: noqa
# Kernels for flashinfer.fused_moe.cute_dsl.sm90_fp8_block_moe; scratch, weight packing
# and dispatch live there.
"""Block-FP8 (W8A8) routed-experts MoE for Qwen3.6-35B-A3B on H200, written in CuTe DSL.

Pipeline (3 launches for small M, 4 for large M, all CuTe DSL kernels):
  1. route    : one CTA - per-expert histogram, prefix sums, token->sorted-slot map, GEMM work list.
  2. quant    : per token - dynamic 1x128 FP8 quant of x, scattered to its 8 sorted slots
                (fp8 rows + transposed per-group scales); zero the fp32 accumulation buffer.
  3. fused GEMM1+GEMM2: swap-AB grouped FP8 WGMMA (weights = MMA-M, tokens = MMA-N), TMA pipeline,
                per-128-K promotion with weight x activation scales in fp32, fused SiLU-GLU and
                dynamic 1x128 FP8 re-quant of h (gate/up rows interleaved by setup so one CTA owns
                a full 128-column quant group).
     GEMM2 uses the same mainloop on w2; its epilogue multiplies router weight and atomically adds
     into the per-token fp32 accumulator (top-k reduction in fp32).
  4. finalize : fp32 accumulator -> bf16 out.
"""

import math

import torch

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass import Float32, Int32, const_expr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
from cutlass._mlir.dialects import llvm

E, TOPK, H, I = 256, 8, 2048, 512
GROUP = 128
FP8_MAX = 448.0
QUANT_EPS = 1e-10

ROUTE_THREADS = 1024
QUANT_THREADS = 256  # 2048 / 8 elements per thread


@dsl_user_op
def _red_add_v4_f32(ptr, a, b, c, d, *, loc=None, ip=None):
    """Fire-and-forget 16-byte fp32 global reduction at a 16-byte aligned address."""
    llvm.inline_asm(
        None,
        [
            ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
            Float32(c).ir_value(loc=loc, ip=ip),
            Float32(d).ir_value(loc=loc, ip=ip),
        ],
        "red.global.v4.f32.add [$0], {$1, $2, $3, $4};",
        "l,f,f,f,f",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_acquire_i32(ptr: cute.Pointer, *, loc=None, ip=None) -> Int32:
    ptr_i64 = ptr.toint(loc=loc, ip=ip).ir_value()
    value = llvm.inline_asm(
        T.i32(),
        [ptr_i64],
        "ld.global.acquire.gpu.b32 $0, [$1];",
        "=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(value)


@dsl_user_op
def _red_release_i32(ptr: cute.Pointer, value: Int32, *, loc=None, ip=None) -> None:
    ptr_i64 = ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [ptr_i64, Int32(value).ir_value(loc=loc, ip=ip)],
        "red.release.gpu.global.add.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


# --------------------------------------------------------------------------------------------
# 1. routing
# --------------------------------------------------------------------------------------------
@cute.kernel
def _route_kernel(
    ids: cute.Tensor,  # (P,) int32 flattened topk_ids
    tw: cute.Tensor,  # (P,) fp32 flattened topk_weights
    expert_off: cute.Tensor,  # (E+1,) int32
    wdesc: cute.Tensor,  # (Wmax, 4) int32 per token tile: expert, first sorted row, valid rows
    meta: cute.Tensor,  # (4,) int32      [0] = number of token tiles, [1] = work counter
    pos_of_pair: cute.Tensor,  # (P,) int32
    sorted_tok: cute.Tensor,  # (P,) int32 token index of each sorted row
    sorted_wt: cute.Tensor,  # (P,) fp32 router weight of each sorted row
    dep: cute.Tensor,  # (Wmax,) int32 GEMM1->GEMM2 dependency counters
    P: Int32,
    BN: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    cute.arch.griddepcontrol_launch_dependents()
    for i in cutlass.range(tidx, cute.size(dep), ROUTE_THREADS, unroll=1):
        dep[i] = Int32(0)
    if tidx == 0:
        meta[1] = Int32(0)
        meta[3] = Int32(0)
    smem = utils.SmemAllocator()
    s_cnt = smem.allocate_tensor(Int32, cute.make_layout(E), byte_alignment=16)
    s_cur = smem.allocate_tensor(Int32, cute.make_layout(E), byte_alignment=16)
    s_wsum = smem.allocate_tensor(Int32, cute.make_layout(16), byte_alignment=16)

    if tidx < E:
        s_cnt[tidx] = Int32(0)
    cute.arch.sync_threads()
    nblk = P // TOPK
    v = cute.make_rmem_tensor((TOPK,), Int32)
    for blk in cutlass.range(tidx, nblk, ROUTE_THREADS, unroll=1):
        cute.autovec_copy(cute.local_tile(ids, (TOPK,), (blk,)), v)
        for i in cutlass.range_constexpr(TOPK):
            cute.arch.atomic_add(s_cnt.iterator + v[i], Int32(1), scope="cta")
    cute.arch.sync_threads()

    warp = tidx // 32
    lane = tidx % 32
    c = Int32(0)
    nt = Int32(0)
    if tidx < E:
        c = s_cnt[tidx]
        nt = (c + (BN - 1)) // BN
    ic = c
    it = nt
    for i in cutlass.range_constexpr(5):
        off = 1 << i
        oc = cute.arch.shuffle_sync_up(ic, off)
        ot = cute.arch.shuffle_sync_up(it, off)
        if lane >= off:
            ic = ic + oc
            it = it + ot
    if tidx < E:
        if lane == 31:
            s_wsum[warp] = ic
            s_wsum[8 + warp] = it
    cute.arch.sync_threads()
    if tidx < E:
        base_c = Int32(0)
        base_t = Int32(0)
        for w in cutlass.range_constexpr(8):
            if w < warp:
                base_c = base_c + s_wsum[w]
                base_t = base_t + s_wsum[8 + w]
        exc = base_c + ic - c
        ext = base_t + it - nt
        s_cur[tidx] = exc
        expert_off[tidx] = exc
        for k in cutlass.range(nt, unroll=1):
            wdesc[ext + k, 0] = tidx
            wdesc[ext + k, 1] = exc + k * BN
            wdesc[ext + k, 2] = cutlass.min(Int32(BN), c - k * BN)
        if tidx == E - 1:
            expert_off[E] = exc + c
            meta[0] = ext + nt
    cute.arch.sync_threads()
    wv = cute.make_rmem_tensor((TOPK,), Float32)
    pv = cute.make_rmem_tensor((TOPK,), Int32)
    for blk in cutlass.range(tidx, nblk, ROUTE_THREADS, unroll=1):
        cute.autovec_copy(cute.local_tile(ids, (TOPK,), (blk,)), v)
        cute.autovec_copy(cute.local_tile(tw, (TOPK,), (blk,)), wv)
        for i in cutlass.range_constexpr(TOPK):
            pos = cute.arch.atomic_add(s_cur.iterator + v[i], Int32(1), scope="cta")
            pv[i] = pos
            sorted_tok[pos] = blk
            sorted_wt[pos] = wv[i]
        cute.autovec_copy(pv, cute.local_tile(pos_of_pair, (TOPK,), (blk,)))


@cute.jit
def _route_host(
    ids,
    tw,
    expert_off,
    wdesc,
    meta,
    pos_of_pair,
    sorted_tok,
    sorted_wt,
    dep,
    P: Int32,
    BN: cutlass.Constexpr,
    stream,
):
    _route_kernel(
        ids, tw, expert_off, wdesc, meta, pos_of_pair, sorted_tok, sorted_wt, dep, P, BN
    ).launch(grid=[1, 1, 1], block=[ROUTE_THREADS, 1, 1], stream=stream)


RMC_CTAS = 64
RMC_THREADS = 256


@cute.kernel
def _route_mc_kernel(
    ids: cute.Tensor,  # (P,) int32 flattened topk_ids
    tw: cute.Tensor,  # (P,) fp32 flattened topk_weights
    expert_off: cute.Tensor,  # (E+1,) int32
    wdesc: cute.Tensor,  # (Wmax, 4) int32
    meta: cute.Tensor,  # (4,) int32
    pos_of_pair: cute.Tensor,  # (P,) int32
    sorted_tok: cute.Tensor,  # (P,) int32
    sorted_wt: cute.Tensor,  # (P,) fp32
    dep: cute.Tensor,  # (Wmax,) int32
    hist: cute.Tensor,  # (RMC_CTAS, E) int32 per-CTA counts, then per-CTA base offsets
    gbar: cute.Tensor,  # (2,) int32 grid barrier: arrivals, generation (monotonic, never reset)
    P: Int32,
    BN: cutlass.Constexpr,
):
    """Multi-CTA routing for large batches: per-CTA expert histograms, a grid barrier whose last
    arriving CTA computes the per-(CTA, expert) base offsets and the token-tile descriptors, then a
    parallel scatter of every (token, k) pair to its expert-sorted slot."""
    tidx, _, _ = cute.arch.thread_idx()
    g, _, _ = cute.arch.block_idx()
    cute.arch.griddepcontrol_launch_dependents()
    smem = utils.SmemAllocator()
    s_cnt = smem.allocate_tensor(Int32, cute.make_layout(E), byte_alignment=16)
    s_wsum = smem.allocate_tensor(Int32, cute.make_layout(16), byte_alignment=16)
    s_last = smem.allocate_tensor(Int32, cute.make_layout(2), byte_alignment=16)

    nd = cute.size(dep)
    for i in cutlass.range(
        g * RMC_THREADS + tidx, nd, RMC_CTAS * RMC_THREADS, unroll=1
    ):
        dep[i] = Int32(0)
    if g == 0:
        if tidx == 0:
            meta[1] = Int32(0)
            meta[3] = Int32(0)
    s_cnt[tidx] = Int32(0)
    cute.arch.sync_threads()
    nblk = P // TOPK
    per = (nblk + RMC_CTAS - 1) // RMC_CTAS
    b0 = g * per
    b1 = cutlass.min(b0 + per, nblk)
    v = cute.make_rmem_tensor((TOPK,), Int32)
    for blk in cutlass.range(b0 + tidx, b1, RMC_THREADS, unroll=1):
        cute.autovec_copy(cute.local_tile(ids, (TOPK,), (blk,)), v)
        for i in cutlass.range_constexpr(TOPK):
            cute.arch.atomic_add(s_cnt.iterator + v[i], Int32(1), scope="cta")
    cute.arch.sync_threads()
    hist[g, tidx] = s_cnt[tidx]

    # ---- grid barrier; the last CTA to arrive builds the offsets ----
    cute.arch.sync_threads()
    if tidx == 0:
        gen0 = cute.arch.atomic_add(
            gbar.iterator + 1, Int32(0), sem="acquire", scope="gpu"
        )
        cute.arch.fence_acq_rel_gpu()
        old = cute.arch.atomic_add(gbar.iterator, Int32(1), sem="acq_rel", scope="gpu")
        s_last[0] = Int32(1) if (old % RMC_CTAS) == RMC_CTAS - 1 else Int32(0)
        s_last[1] = gen0
    cute.arch.sync_threads()
    if s_last[0] != 0:
        cute.arch.fence_acq_rel_gpu()
        # per-(CTA, expert) exclusive prefix over CTAs for expert tidx
        hv = cute.make_rmem_tensor((16,), Int32)
        run = Int32(0)
        for gb in cutlass.range(RMC_CTAS // 16, unroll=1):
            for q in cutlass.range_constexpr(16):
                hv[q] = hist[gb * 16 + q, tidx]
            for q in cutlass.range_constexpr(16):
                run = run + hv[q]
        c = run
        nt = (c + (BN - 1)) // BN
        warp = tidx // 32
        lane = tidx % 32
        ic = c
        it = nt
        for i in cutlass.range_constexpr(5):
            off = 1 << i
            oc = cute.arch.shuffle_sync_up(ic, off)
            ot = cute.arch.shuffle_sync_up(it, off)
            if lane >= off:
                ic = ic + oc
                it = it + ot
        if lane == 31:
            s_wsum[warp] = ic
            s_wsum[8 + warp] = it
        cute.arch.sync_threads()
        base_c = Int32(0)
        base_t = Int32(0)
        for w in cutlass.range_constexpr(8):
            if w < warp:
                base_c = base_c + s_wsum[w]
                base_t = base_t + s_wsum[8 + w]
        exc = base_c + ic - c
        ext = base_t + it - nt
        expert_off[tidx] = exc
        for k in cutlass.range(nt, unroll=1):
            wdesc[ext + k, 0] = tidx
            wdesc[ext + k, 1] = exc + k * BN
            wdesc[ext + k, 2] = cutlass.min(Int32(BN), c - k * BN)
        run = exc
        for gb in cutlass.range(RMC_CTAS // 16, unroll=1):
            for q in cutlass.range_constexpr(16):
                hv[q] = hist[gb * 16 + q, tidx]
            for q in cutlass.range_constexpr(16):
                hist[gb * 16 + q, tidx] = run
                run = run + hv[q]
        if tidx == E - 1:
            expert_off[E] = exc + c
            meta[0] = ext + nt
        cute.arch.sync_threads()
        if tidx == 0:
            # Every CTA has arrived in this generation. Re-arm the reusable arrival counter
            # before publishing the next epoch so long-running serving cannot overflow int32.
            gbar[0] = Int32(0)
            cute.arch.fence_acq_rel_gpu()
            cute.arch.atomic_add(
                gbar.iterator + 1, Int32(1), sem="release", scope="gpu"
            )
    else:
        if tidx == 0:
            gcur = cute.arch.atomic_add(
                gbar.iterator + 1, Int32(0), sem="acquire", scope="gpu"
            )
            while gcur == s_last[1]:
                gcur = cute.arch.atomic_add(
                    gbar.iterator + 1, Int32(0), sem="acquire", scope="gpu"
                )
            cute.arch.fence_acq_rel_gpu()
        cute.arch.sync_threads()

    # ---- scatter ----
    s_cnt[tidx] = hist[g, tidx]
    cute.arch.sync_threads()
    wv = cute.make_rmem_tensor((TOPK,), Float32)
    pv = cute.make_rmem_tensor((TOPK,), Int32)
    for blk in cutlass.range(b0 + tidx, b1, RMC_THREADS, unroll=1):
        cute.autovec_copy(cute.local_tile(ids, (TOPK,), (blk,)), v)
        cute.autovec_copy(cute.local_tile(tw, (TOPK,), (blk,)), wv)
        for i in cutlass.range_constexpr(TOPK):
            pos = cute.arch.atomic_add(s_cnt.iterator + v[i], Int32(1), scope="cta")
            pv[i] = pos
            sorted_tok[pos] = blk
            sorted_wt[pos] = wv[i]
        cute.autovec_copy(pv, cute.local_tile(pos_of_pair, (TOPK,), (blk,)))


@cute.jit
def _route_mc_host(
    ids,
    tw,
    expert_off,
    wdesc,
    meta,
    pos_of_pair,
    sorted_tok,
    sorted_wt,
    dep,
    hist,
    gbar,
    P: Int32,
    BN: cutlass.Constexpr,
    stream,
):
    _route_mc_kernel(
        ids,
        tw,
        expert_off,
        wdesc,
        meta,
        pos_of_pair,
        sorted_tok,
        sorted_wt,
        dep,
        hist,
        gbar,
        P,
        BN,
    ).launch(grid=[RMC_CTAS, 1, 1], block=[RMC_THREADS, 1, 1], stream=stream)


# --------------------------------------------------------------------------------------------
# 2. activation quant + scatter to sorted slots, zero the fp32 accumulator
# --------------------------------------------------------------------------------------------
@cute.kernel
def _quant_kernel(
    x: cute.Tensor,  # (M, H) bf16
    pos_of_pair: cute.Tensor,  # (P,) int32
    xq: cute.Tensor,  # (P, H) fp8
    xsT: cute.Tensor,  # (H/128, PS) fp32
    accbuf: cute.Tensor,  # (M, H) fp32
):
    tidx, _, _ = cute.arch.thread_idx()
    m, _, _ = cute.arch.block_idx()
    cute.arch.griddepcontrol_launch_dependents()

    gx = cute.local_tile(x[m, None], (8,), (tidx,))
    fx_bf = cute.make_rmem_tensor((8,), cutlass.BFloat16)
    cute.autovec_copy(gx, fx_bf)
    fx = cute.make_rmem_tensor((8,), Float32)
    fx.store(fx_bf.load().to(Float32))
    amax = Float32(0.0)
    for i in cutlass.range_constexpr(8):
        amax = cute.arch.fmax(amax, cute.arch.fmax(fx[i], -fx[i]))
    for i in cutlass.range_constexpr(4):
        amax = cute.arch.fmax(amax, cute.arch.shuffle_sync_bfly(amax, 1 << i))
    amax = cute.arch.fmax(amax, Float32(QUANT_EPS))
    scale = amax / Float32(FP8_MAX)
    inv = Float32(FP8_MAX) / amax
    for i in cutlass.range_constexpr(8):
        q = fx[i] * inv
        q = cute.arch.fmin(cute.arch.fmax(q, Float32(-FP8_MAX)), Float32(FP8_MAX))
        fx[i] = q
    q8 = cute.make_rmem_tensor((8,), cutlass.Float8E4M3FN)
    q8.store(fx.load().to(cutlass.Float8E4M3FN))
    grp = tidx // 16
    # pos_of_pair comes from the routing kernel
    cute.arch.griddepcontrol_wait()
    for k in cutlass.range_constexpr(TOPK):
        pos = pos_of_pair[m * TOPK + k]
        cute.autovec_copy(q8, cute.local_tile(xq[pos, None], (8,), (tidx,)))
        if tidx % 16 == 0:
            xsT[grp, pos] = scale
    z = cute.make_rmem_tensor((8,), Float32)
    z.fill(0.0)
    cute.autovec_copy(z, cute.local_tile(accbuf[m, None], (8,), (tidx,)))


@cute.kernel
def _quant_route_linear_kernel(
    x: cute.Tensor,
    ids: cute.Tensor,
    tw: cute.Tensor,
    expert_off: cute.Tensor,
    wdesc: cute.Tensor,
    meta: cute.Tensor,
    pos_of_pair: cute.Tensor,
    sorted_tok: cute.Tensor,
    sorted_wt: cute.Tensor,
    dep: cute.Tensor,
    xq: cute.Tensor,
    xsT: cute.Tensor,
    accbuf: cute.Tensor,
    gbar: cute.Tensor,
    BN: cutlass.Constexpr,
):
    """One-launch linear routing plus quantization for small batches.

    CTA 0 performs the O(P) expert counting-sort. At most 128 resident CTAs synchronize once,
    then cooperatively quantize and scatter all rows without a second launch.
    """
    tidx, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    ncta, _, _ = cute.arch.grid_dim()
    cute.arch.griddepcontrol_launch_dependents()

    smem = utils.SmemAllocator()
    s_cnt = smem.allocate_tensor(Int32, cute.make_layout(E), byte_alignment=16)
    s_cur = smem.allocate_tensor(Int32, cute.make_layout(E), byte_alignment=16)
    s_wsum = smem.allocate_tensor(Int32, cute.make_layout(16), byte_alignment=16)
    s_last = smem.allocate_tensor(Int32, cute.make_layout(2), byte_alignment=16)

    if bid == 0:
        if tidx < E:
            s_cnt[tidx] = Int32(0)
        cute.arch.sync_threads()
        P = cute.size(ids)
        nblk = P // TOPK
        v = cute.make_rmem_tensor((TOPK,), Int32)
        for blk in cutlass.range(tidx, nblk, QUANT_THREADS, unroll=1):
            cute.autovec_copy(cute.local_tile(ids, (TOPK,), (blk,)), v)
            for i in cutlass.range_constexpr(TOPK):
                cute.arch.atomic_add(s_cnt.iterator + v[i], Int32(1), scope="cta")
        cute.arch.sync_threads()

        warp = tidx // 32
        lane = tidx % 32
        c = s_cnt[tidx]
        nt = (c + (BN - 1)) // BN
        ic = c
        it = nt
        for i in cutlass.range_constexpr(5):
            off = 1 << i
            oc = cute.arch.shuffle_sync_up(ic, off)
            ot = cute.arch.shuffle_sync_up(it, off)
            if lane >= off:
                ic = ic + oc
                it = it + ot
        if lane == 31:
            s_wsum[warp] = ic
            s_wsum[8 + warp] = it
        cute.arch.sync_threads()
        base_c = Int32(0)
        base_t = Int32(0)
        for w in cutlass.range_constexpr(8):
            if w < warp:
                base_c = base_c + s_wsum[w]
                base_t = base_t + s_wsum[8 + w]
        exc = base_c + ic - c
        ext = base_t + it - nt
        s_cur[tidx] = exc
        expert_off[tidx] = exc
        for k in cutlass.range(nt, unroll=1):
            wdesc[ext + k, 0] = tidx
            wdesc[ext + k, 1] = exc + k * BN
            wdesc[ext + k, 2] = cutlass.min(Int32(BN), c - k * BN)
        for i in cutlass.range(tidx, cute.size(dep), QUANT_THREADS, unroll=1):
            dep[i] = Int32(0)
        if tidx == 0:
            meta[1] = Int32(0)
            meta[3] = Int32(0)
        if tidx == E - 1:
            expert_off[E] = exc + c
            meta[0] = ext + nt
        cute.arch.sync_threads()

        wv = cute.make_rmem_tensor((TOPK,), Float32)
        pv = cute.make_rmem_tensor((TOPK,), Int32)
        for blk in cutlass.range(tidx, nblk, QUANT_THREADS, unroll=1):
            cute.autovec_copy(cute.local_tile(ids, (TOPK,), (blk,)), v)
            cute.autovec_copy(cute.local_tile(tw, (TOPK,), (blk,)), wv)
            for i in cutlass.range_constexpr(TOPK):
                pos = cute.arch.atomic_add(s_cur.iterator + v[i], Int32(1), scope="cta")
                pv[i] = pos
                sorted_tok[pos] = blk
                sorted_wt[pos] = wv[i]
            cute.autovec_copy(pv, cute.local_tile(pos_of_pair, (TOPK,), (blk,)))

    # Grid barrier: CTA 0 publishes the complete route through a release epoch. The reusable
    # arrival counter and epoch are monotonic, so no stale zero/reset race exists across calls.
    cute.arch.sync_threads()
    if tidx == 0:
        gen0 = cute.arch.atomic_add(
            gbar.iterator + 1, Int32(0), sem="acquire", scope="gpu"
        )
        old = cute.arch.atomic_add(gbar.iterator, Int32(1), sem="acq_rel", scope="gpu")
        s_last[0] = Int32(1) if (old % ncta) == ncta - 1 else Int32(0)
        s_last[1] = gen0
    cute.arch.sync_threads()
    if s_last[0] != 0:
        if tidx == 0:
            cute.arch.fence_acq_rel_gpu()
            cute.arch.atomic_add(
                gbar.iterator + 1, Int32(1), sem="release", scope="gpu"
            )
    else:
        if tidx == 0:
            gcur = cute.arch.atomic_add(
                gbar.iterator + 1, Int32(0), sem="acquire", scope="gpu"
            )
            while gcur == s_last[1]:
                gcur = cute.arch.atomic_add(
                    gbar.iterator + 1, Int32(0), sem="acquire", scope="gpu"
                )
            cute.arch.fence_acq_rel_gpu()
    cute.arch.sync_threads()

    M = cute.size(x, mode=[0])
    for m in cutlass.range(bid, M, ncta, unroll=1):
        gx = cute.local_tile(x[m, None], (8,), (tidx,))
        fx_bf = cute.make_rmem_tensor((8,), cutlass.BFloat16)
        cute.autovec_copy(gx, fx_bf)
        fx = cute.make_rmem_tensor((8,), Float32)
        fx.store(fx_bf.load().to(Float32))
        amax = Float32(0.0)
        for i in cutlass.range_constexpr(8):
            amax = cute.arch.fmax(amax, cute.arch.fmax(fx[i], -fx[i]))
        for i in cutlass.range_constexpr(4):
            amax = cute.arch.fmax(amax, cute.arch.shuffle_sync_bfly(amax, 1 << i))
        amax = cute.arch.fmax(amax, Float32(QUANT_EPS))
        scale = amax / Float32(FP8_MAX)
        inv = Float32(FP8_MAX) / amax
        for i in cutlass.range_constexpr(8):
            q = fx[i] * inv
            fx[i] = cute.arch.fmin(
                cute.arch.fmax(q, Float32(-FP8_MAX)), Float32(FP8_MAX)
            )
        q8 = cute.make_rmem_tensor((8,), cutlass.Float8E4M3FN)
        q8.store(fx.load().to(cutlass.Float8E4M3FN))
        grp = tidx // 16
        for k in cutlass.range_constexpr(TOPK):
            pos = pos_of_pair[m * TOPK + k]
            cute.autovec_copy(q8, cute.local_tile(xq[pos, None], (8,), (tidx,)))
            if tidx % 16 == 0:
                xsT[grp, pos] = scale
        z = cute.make_rmem_tensor((8,), Float32)
        z.fill(0.0)
        cute.autovec_copy(z, cute.local_tile(accbuf[m, None], (8,), (tidx,)))


@cute.kernel
def _quant_route_kernel(
    x: cute.Tensor,
    ids: cute.Tensor,
    tw: cute.Tensor,
    expert_off: cute.Tensor,
    wdesc: cute.Tensor,
    meta: cute.Tensor,
    sorted_tok: cute.Tensor,
    sorted_wt: cute.Tensor,
    dep: cute.Tensor,
    xq: cute.Tensor,
    xsT: cute.Tensor,
    accbuf: cute.Tensor,
    BN: cutlass.Constexpr,
    NQ: cutlass.Constexpr,
    NCH: cutlass.Constexpr,
):
    """Small-M fused quantization and stable expert counting-sort.

    Token CTAs recompute stable ranks while one independent CTA builds expert descriptors. This
    removes descriptor construction from token 0's quant/scatter critical path without a grid wait.
    """
    tidx, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    lane = tidx % 32
    warp = tidx // 32

    smem = utils.SmemAllocator()
    s_cnt = smem.allocate_tensor(Int32, cute.make_layout(E), byte_alignment=16)
    s_wsum = smem.allocate_tensor(Int32, cute.make_layout(16), byte_alignment=16)
    s_slot = smem.allocate_tensor(Int32, cute.make_layout(TOPK * NQ), byte_alignment=16)
    s_red = smem.allocate_tensor(
        Int32,
        cute.make_layout((8, TOPK * NQ), stride=(TOPK * NQ, 1)),
        byte_alignment=16,
    )
    P = cute.size(ids)
    if bid == 0:
        # meta[2] starts at zero and the preceding call's finalize re-arms it after all persistent
        # readers finish, so this route can launch its dependent grid without a front-end GPU fence.
        cute.arch.griddepcontrol_launch_dependents()
        s_cnt[tidx] = Int32(0)
        cute.arch.sync_threads()
        nblk = P // TOPK
        v = cute.make_rmem_tensor((TOPK,), Int32)
        for blk in cutlass.range(tidx, nblk, QUANT_THREADS, unroll=1):
            cute.autovec_copy(cute.local_tile(ids, (TOPK,), (blk,)), v)
            for k in cutlass.range_constexpr(TOPK):
                cute.arch.atomic_add(s_cnt.iterator + v[k], Int32(1), scope="cta")
        cute.arch.sync_threads()
        c = s_cnt[tidx]
        nt = (c + (BN - 1)) // BN
        ic = c
        it = nt
        for i in cutlass.range_constexpr(5):
            off = 1 << i
            oc = cute.arch.shuffle_sync_up(ic, off)
            ot = cute.arch.shuffle_sync_up(it, off)
            if lane >= off:
                ic = ic + oc
                it = it + ot
        if lane == 31:
            s_wsum[warp] = ic
            s_wsum[8 + warp] = it
        cute.arch.sync_threads()
        base_c = Int32(0)
        base_t = Int32(0)
        for w in cutlass.range_constexpr(8):
            if w < warp:
                base_c = base_c + s_wsum[w]
                base_t = base_t + s_wsum[8 + w]
        exc = base_c + ic - c
        ext = base_t + it - nt
        expert_off[tidx] = exc
        for k in cutlass.range(nt, unroll=1):
            wdesc[ext + k, 0] = tidx
            wdesc[ext + k, 1] = exc + k * BN
            wdesc[ext + k, 2] = cutlass.min(Int32(BN), c - k * BN)
        for i in cutlass.range(tidx, cute.size(dep), QUANT_THREADS, unroll=1):
            dep[i] = Int32(0)
        if tidx == 0:
            meta[1] = Int32(0)
            meta[3] = Int32(0)
        if tidx == E - 1:
            expert_off[E] = P
            meta[0] = ext + nt
        cute.arch.sync_threads()
        if tidx == 0:
            cute.arch.fence_acq_rel_gpu()
            cute.arch.atomic_add(
                meta.iterator + 2, Int32(1), sem="release", scope="gpu"
            )
    else:
        # Token CTA b quantizes tokens b - 1 + i * (grid - 1), i < NQ per pass. Every global load of a
        # pass is issued before any dependent work, and each pair's stable expert-sorted slot comes from
        # a single key comparison, so a token CTA retires after about one memory round trip.
        cute.arch.griddepcontrol_launch_dependents()
        Mt = cute.size(x, mode=[0])
        nqc = cute.arch.grid_dim()[0] - 1
        for base in cutlass.range(bid - 1, Mt, nqc * NQ, unroll=1):
            # every global load of the pass is issued before any dependent work
            xb = cute.make_rmem_tensor((8 * NQ,), cutlass.BFloat16)
            ek = cute.make_rmem_tensor((TOPK * NQ,), Int32)
            for i in cutlass.range_constexpr(NQ):
                mi = cutlass.min(base + i * nqc, Mt - 1)
                cute.autovec_copy(
                    cute.local_tile(x[mi, None], (8,), (tidx,)),
                    cute.local_tile(xb, (8,), (i,)),
                )
                cute.autovec_copy(
                    cute.local_tile(ids, (TOPK,), (mi,)),
                    cute.local_tile(ek, (TOPK,), (i,)),
                )
            twv = Float32(0.0)
            if tidx < TOPK * NQ:
                mw = cutlass.min(base + (tidx // TOPK) * nqc, Mt - 1)
                twv = tw[mw * TOPK + tidx % TOPK]
            evs = cute.make_rmem_tensor((4 * NCH,), Int32)
            for c in cutlass.range_constexpr(NCH):
                if tidx + c * QUANT_THREADS < P // 4:
                    cute.autovec_copy(
                        cute.local_tile(ids, (4,), (tidx + c * QUANT_THREADS,)),
                        cute.local_tile(evs, (4,), (c,)),
                    )
            # stable expert-sorted slot of pair (i, k) = #pairs q with (e_q, q) < (e_ik, ik): one compare
            # of the keys e * 65536 + q (q < P <= 4096 in this regime)
            kk = cute.make_rmem_tensor((TOPK * NQ,), Int32)
            for i in cutlass.range_constexpr(NQ):
                for k in cutlass.range_constexpr(TOPK):
                    kk[i * TOPK + k] = (
                        ek[i * TOPK + k] * Int32(65536) + (base + i * nqc) * TOPK + k
                    )
            cnt = cute.make_rmem_tensor((TOPK * NQ,), Int32)
            cnt.fill(0)
            for c in cutlass.range_constexpr(NCH):
                if tidx + c * QUANT_THREADS < P // 4:
                    for qq in cutlass.range_constexpr(4):
                        kq = (
                            evs[c * 4 + qq] * Int32(65536)
                            + (tidx + c * QUANT_THREADS) * 4
                            + qq
                        )
                        for jq in cutlass.range_constexpr(TOPK * NQ):
                            cnt[jq] = cnt[jq] + (Int32(1) if kq < kk[jq] else Int32(0))
            for jq in cutlass.range_constexpr(TOPK * NQ):
                v = cnt[jq]
                for i in cutlass.range_constexpr(5):
                    v = v + cute.arch.shuffle_sync_bfly(v, 1 << i)
                if lane == 0:
                    s_red[warp, jq] = v
            cute.arch.sync_threads()
            if tidx < TOPK * NQ:
                mq = base + (tidx // TOPK) * nqc
                if mq < Mt:
                    pos = Int32(0)
                    for w in cutlass.range_constexpr(8):
                        pos = pos + s_red[w, tidx]
                    s_slot[tidx] = pos
                    sorted_tok[pos] = mq
                    sorted_wt[pos] = twv
            cute.arch.sync_threads()
            grp = tidx // 16
            for i in cutlass.range_constexpr(NQ):
                m = base + i * nqc
                if m < Mt:
                    fx = cute.make_rmem_tensor((8,), Float32)
                    fx.store(cute.local_tile(xb, (8,), (i,)).load().to(Float32))
                    amax = Float32(0.0)
                    for u in cutlass.range_constexpr(8):
                        amax = cute.arch.fmax(amax, cute.arch.fmax(fx[u], -fx[u]))
                    for u in cutlass.range_constexpr(4):
                        amax = cute.arch.fmax(
                            amax, cute.arch.shuffle_sync_bfly(amax, 1 << u)
                        )
                    amax = cute.arch.fmax(amax, Float32(QUANT_EPS))
                    scale = amax / Float32(FP8_MAX)
                    inv = Float32(FP8_MAX) / amax
                    for u in cutlass.range_constexpr(8):
                        q = fx[u] * inv
                        fx[u] = cute.arch.fmin(
                            cute.arch.fmax(q, Float32(-FP8_MAX)), Float32(FP8_MAX)
                        )
                    q8 = cute.make_rmem_tensor((8,), cutlass.Float8E4M3FN)
                    q8.store(fx.load().to(cutlass.Float8E4M3FN))
                    for k in cutlass.range_constexpr(TOPK):
                        pos = s_slot[i * TOPK + k]
                        cute.autovec_copy(
                            q8, cute.local_tile(xq[pos, None], (8,), (tidx,))
                        )
                        if tidx % 16 == 0:
                            xsT[grp, pos] = scale
                    z = cute.make_rmem_tensor((8,), Float32)
                    z.fill(0.0)
                    cute.autovec_copy(
                        z, cute.local_tile(accbuf[m, None], (8,), (tidx,))
                    )
            # s_red / s_slot are rewritten by the next pass
            cute.arch.sync_threads()


@cute.jit
def _quant_host(x, pos_of_pair, xq, xsT, accbuf, M: Int32, stream):
    _quant_kernel(x, pos_of_pair, xq, xsT, accbuf).launch(
        grid=[M, 1, 1], block=[QUANT_THREADS, 1, 1], stream=stream, use_pdl=True
    )


@cute.jit
def _quant_route_host(
    x,
    ids,
    tw,
    expert_off,
    wdesc,
    meta,
    sorted_tok,
    sorted_wt,
    dep,
    xq,
    xsT,
    accbuf,
    NQC: Int32,
    BN: cutlass.Constexpr,
    NQ: cutlass.Constexpr,
    NCH: cutlass.Constexpr,
    stream,
):
    _quant_route_kernel(
        x,
        ids,
        tw,
        expert_off,
        wdesc,
        meta,
        sorted_tok,
        sorted_wt,
        dep,
        xq,
        xsT,
        accbuf,
        BN,
        NQ,
        NCH,
    ).launch(grid=[NQC + 1, 1, 1], block=[QUANT_THREADS, 1, 1], stream=stream)


# --------------------------------------------------------------------------------------------
# 5. finalize fp32 -> bf16
# --------------------------------------------------------------------------------------------
@cute.kernel
def _finalize_kernel(
    accbuf: cute.Tensor, out: cute.Tensor, meta: cute.Tensor, PARTS: cutlass.Constexpr
):
    tidx, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    m = bid // PARTS
    part = bid - m * PARTS
    cute.arch.griddepcontrol_wait()
    if bid == 0 and tidx == 0:
        meta[2] = Int32(0)
    width = 8 // PARTS
    tile = tidx + part * QUANT_THREADS
    f = cute.make_rmem_tensor((width,), Float32)
    cute.autovec_copy(cute.local_tile(accbuf[m, None], (width,), (tile,)), f)
    o = cute.make_rmem_tensor((width,), cutlass.BFloat16)
    o.store(f.load().to(cutlass.BFloat16))
    cute.autovec_copy(o, cute.local_tile(out[m, None], (width,), (tile,)))


@cute.jit
def _finalize_host(accbuf, out, meta, M: Int32, PARTS: cutlass.Constexpr, stream):
    _finalize_kernel(accbuf, out, meta, PARTS).launch(
        grid=[M * PARTS, 1, 1], block=[QUANT_THREADS, 1, 1], stream=stream, use_pdl=True
    )


# --------------------------------------------------------------------------------------------
# 3/4. grouped swap-AB block-FP8 GEMM
# --------------------------------------------------------------------------------------------
class MoEFused:
    """Persistent kernel running every GEMM1 and GEMM2 tile of the layer.

    Work item = (phase, expert e, token tile of BN sorted rows, 256 weight rows); items are claimed
    in order from a global atomic counter (all GEMM1 items before all GEMM2 items), so a GEMM2 item
    only waits on GEMM1 items already claimed by running CTAs (no deadlock). GEMM2 waits on a
    per-(e, token tile) counter that the 4 GEMM1 tiles of that token tile bump after storing h.

    Warps 0-7: two consumer warpgroups (WGMMA m64nBNk32, 2 x m64 each -> 256 rows).
    Warp 8   : scheduler + TMA producer (weights tile 256x128, token tile BNx128).
    """

    def __init__(self, BN: int, stages: int):
        self.BN = BN
        self.BM = 256
        self.BK = 128
        self.stages = stages
        self.qstages = _pick_qstages(BN)
        # GEMM2 work item = G2 consecutive 256-row output tiles (longer items amortize scheduling)
        self.G2 = 2 if BN >= 64 else 1
        self.grp_major = BN >= 64
        self.evict_first_weights = BN <= 32
        self.out_tok = min(BN, 32)
        self.epi_bytes = max(BN * 128, 2 * self.out_tok * (128 + 4) * 4)
        self.ng2 = (H // 256) // self.G2
        assert BN <= 64
        self.BNS = BN
        self.NSUB = 1
        self.num_cons_threads = 256
        self.threads = 384
        self.cons_regs = 232
        self.prod_regs = 40
        self.prod_warp = 8
        self.sched_warp = 9
        self.sig_warp = 10
        self.sched_warp2 = 11
        self.sched_threads = 64 if BN == 32 else 32
        self.nkb1 = H // 128
        self.nkb2 = I // 128
        self.mt1 = (2 * I) // 256
        self.mt2 = H // 256

    @cute.jit
    def __call__(
        self,
        w13t: cute.Tensor,  # (E*4*16*256, 128) fp8: gate/up-interleaved, tile-major [e][mt][kb][256][128]
        xq: cute.Tensor,  # (P, H) fp8 sorted activations
        xs: cute.Tensor,  # (PS, H/128) fp32 view, stride (1, PS)
        w13s: cute.Tensor,  # (E, 8, 16) fp32
        w2t: cute.Tensor,  # (E*8*4*256, 128) fp8: tile-major [e][mt][kb][256][128]
        hq: cute.Tensor,  # (P, I) fp8 sorted intermediate
        hs: cute.Tensor,  # (PS, I/128) fp32 view, stride (1, PS)
        w2s: cute.Tensor,  # (E, 16, 4) fp32
        expert_off: cute.Tensor,
        work: cute.Tensor,  # (Wmax, 4) int32 token-tile descriptors
        meta: cute.Tensor,
        dep: cute.Tensor,
        accbuf: cute.Tensor,  # (M, H) fp32
        out: cute.Tensor,  # (M, H) bf16
        sorted_pair: cute.Tensor,  # (P,) int32 token index per sorted row
        topk_w: cute.Tensor,  # (P,) fp32 router weight per sorted row
        ids: cute.Tensor,  # (P,) flattened topk ids; retained to preserve the compiled-call ABI
        early_desc: Int32,
        grid_n: Int32,
        stream,
    ):
        fp8 = cutlass.Float8E4M3FN
        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            fp8,
            fp8,
            cute.nvgpu.warpgroup.OperandMajorMode.K,
            cute.nvgpu.warpgroup.OperandMajorMode.K,
            Float32,
            (2, 1, 1),
            tiler_mn=(64, self.BNS),
        )
        ab_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(utils.LayoutEnum.ROW_MAJOR, fp8, self.BK),
            fp8,
        )
        self.a_smem_staged = cute.tile_to_shape(
            ab_atom, (self.BM, self.BK, self.stages), order=(0, 1, 2)
        )
        self.b_smem_staged = cute.tile_to_shape(
            ab_atom, (self.BN, self.BK, self.stages), order=(0, 1, 2)
        )
        a_one = cute.slice_(self.a_smem_staged, (None, None, 0))
        b_one = cute.slice_(self.b_smem_staged, (None, None, 0))
        g2s = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        tma_a1, tA1 = cute.nvgpu.cpasync.make_tiled_tma_atom(
            g2s, w13t, a_one, (self.BM, self.BK)
        )
        tma_b1, tB1 = cute.nvgpu.cpasync.make_tiled_tma_atom(
            g2s, xq, b_one, (self.BN, self.BK)
        )
        tma_a2, tA2 = cute.nvgpu.cpasync.make_tiled_tma_atom(
            g2s, w2t, a_one, (self.BM, self.BK)
        )
        tma_b2, tB2 = cute.nvgpu.cpasync.make_tiled_tma_atom(
            g2s, hq, b_one, (self.BN, self.BK)
        )
        self.tx_bytes = self.BM * self.BK + self.BN * self.BK

        @cute.struct
        class SharedStorage:
            mbar: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
            qbar: cute.struct.MemRange[cutlass.Int64, self.qstages * 2]
            sgbar: cute.struct.MemRange[cutlass.Int64, 4 * 2]
            sSig: cute.struct.Align[cute.struct.MemRange[Int32, 4], 16]
            sInv: cute.struct.Align[cute.struct.MemRange[Float32, self.BN], 16]
            sItem: cute.struct.Align[cute.struct.MemRange[Int32, self.qstages], 16]
            sClaim: cute.struct.Align[cute.struct.MemRange[Int32, self.qstages], 16]
            sDesc: cute.struct.Align[cute.struct.MemRange[Int32, 8 * self.qstages], 16]
            sW: cute.struct.Align[cute.struct.MemRange[Float32, 64 * self.qstages], 16]
            sSc: cute.struct.Align[
                cute.struct.MemRange[Float32, 16 * self.BN * self.qstages], 16
            ]
            sTok: cute.struct.Align[
                cute.struct.MemRange[Int32, self.BN * self.qstages], 16
            ]
            sWt: cute.struct.Align[
                cute.struct.MemRange[Float32, self.BN * self.qstages], 16
            ]
            sRed: cute.struct.Align[cute.struct.MemRange[Float32, 8 * self.BN], 16]
            # GEMM1's fp8 SiLU-GLU staging and GEMM2's fp32 transpose staging have disjoint
            # lifetimes, so one byte buffer funds the wider BN32 epilogue without another SMEM ring.
            sEpi: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, self.epi_bytes], 16
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[fp8, cute.cosize(self.a_smem_staged)], 1024
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[fp8, cute.cosize(self.b_smem_staged)], 1024
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            tma_a1,
            tA1,
            tma_b1,
            tB1,
            tma_a2,
            tA2,
            tma_b2,
            tB2,
            xs,
            hs,
            w13s,
            w2s,
            expert_off,
            work,
            meta,
            dep,
            hq,
            accbuf,
            out,
            sorted_pair,
            topk_w,
            early_desc,
            self.tiled_mma,
            self.a_smem_staged,
            self.b_smem_staged,
        ).launch(
            grid=[grid_n, 1, 1],
            block=[self.threads, 1, 1],
            min_blocks_per_mp=1,
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        tma_a1: cute.CopyAtom,
        mA1: cute.Tensor,
        tma_b1: cute.CopyAtom,
        mB1: cute.Tensor,
        tma_a2: cute.CopyAtom,
        mA2: cute.Tensor,
        tma_b2: cute.CopyAtom,
        mB2: cute.Tensor,
        mS1: cute.Tensor,
        mS2: cute.Tensor,
        w13s: cute.Tensor,
        w2s: cute.Tensor,
        expert_off: cute.Tensor,
        work: cute.Tensor,
        meta: cute.Tensor,
        dep: cute.Tensor,
        hq: cute.Tensor,
        accbuf: cute.Tensor,
        out: cute.Tensor,
        sorted_pair: cute.Tensor,
        topk_w: cute.Tensor,
        early_desc: Int32,
        tiled_mma: cute.TiledMma,
        a_smem_staged: cute.ComposedLayout,
        b_smem_staged: cute.ComposedLayout,
    ):
        BN = self.BN
        BM = self.BM
        BK = self.BK
        BNS = self.BNS
        NSUB = self.NSUB
        NJS = BNS // 8
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()

        if warp_idx == self.prod_warp:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_a1)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_b1)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_a2)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_b2)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        mainloop = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar.data_ptr(),
            num_stages=self.stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 8),
            tx_count=self.tx_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        sched = pipeline.PipelineAsync.create(
            barrier_storage=storage.qbar.data_ptr(),
            num_stages=self.qstages,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.sched_threads
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_cons_threads + 32
            ),
        )
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        if early_desc != 0:
            if tidx == 0:
                ready = cute.arch.atomic_add(
                    meta.iterator + 2, Int32(0), sem="acquire", scope="gpu"
                )
                while ready == 0:
                    ready = cute.arch.atomic_add(
                        meta.iterator + 2, Int32(0), sem="acquire", scope="gpu"
                    )
            cute.arch.sync_threads()
        else:
            cute.arch.griddepcontrol_wait()
        P_rows = expert_off[E]
        sA = storage.sA.get_tensor(a_smem_staged.outer, swizzle=a_smem_staged.inner)
        sB = storage.sB.get_tensor(b_smem_staged.outer, swizzle=b_smem_staged.inner)
        sItem = storage.sItem.get_tensor(cute.make_layout(self.qstages))
        sClaim = storage.sClaim.get_tensor(cute.make_layout(self.qstages))
        sSig = storage.sSig.get_tensor(cute.make_layout(4))
        sInv = storage.sInv.get_tensor(cute.make_layout(BN))
        sigq = pipeline.PipelineAsync.create(
            barrier_storage=storage.sgbar.data_ptr(),
            num_stages=4,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_cons_threads
            ),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32),
        )
        sDesc = storage.sDesc.get_tensor(
            cute.make_layout((8, self.qstages), stride=(1, 8))
        )
        sW = storage.sW.get_tensor(cute.make_layout((64, self.qstages), stride=(1, 64)))
        sSc = storage.sSc.get_tensor(
            cute.make_layout((BN, 16, self.qstages), stride=(1, BN, 16 * BN))
        )
        sTok = storage.sTok.get_tensor(
            cute.make_layout((BN, self.qstages), stride=(1, BN))
        )
        sWt = storage.sWt.get_tensor(
            cute.make_layout((BN, self.qstages), stride=(1, BN))
        )
        sRed = storage.sRed.get_tensor(cute.make_layout((8, BN), stride=(BN, 1)))
        sEpi = storage.sEpi.get_tensor(cute.make_layout(self.epi_bytes))
        sH = cute.make_tensor(
            sEpi.iterator, cute.make_layout((BN, 128), stride=(128, 1))
        )
        sOut = cute.make_tensor(
            cute.recast_ptr(sEpi.iterator, dtype=Float32),
            cute.make_layout(
                (2, self.out_tok, 128 + 4),
                stride=(self.out_tok * (128 + 4), 128 + 4, 1),
            ),
        )

        T = meta[0]
        n1 = T * self.mt1
        # Split only the final GEMM2 wave into two half-K items, shortening the decode tail.
        nsplit = Int32(0)
        if const_expr(not self.grp_major):
            nsplit = cutlass.min(gdim, T * self.ng2)
        total = n1 + T * self.ng2 + nsplit

        if warp_idx >= self.prod_warp:
            cute.arch.setmaxregister_decrease(self.prod_regs)
        is_sched = warp_idx == self.sched_warp
        if const_expr(self.sched_threads == 64):
            is_sched = is_sched or warp_idx == self.sched_warp2
        if is_sched:
            # ---------------- scheduler: claim + prepare items ahead of the TMA warp ----------------
            stid = lane
            if const_expr(self.sched_threads == 64):
                if warp_idx == self.sched_warp2:
                    stid = stid + Int32(32)
            qstate = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.qstages
            )
            idx = Int32(0)
            if early_desc != 0 and bidx < n1:
                idx = self._claim_static(
                    work,
                    mS1,
                    w13s,
                    sItem,
                    sDesc,
                    sW,
                    sSc,
                    sched,
                    qstate,
                    stid,
                    P_rows,
                    bidx,
                )
            else:
                cute.arch.griddepcontrol_wait()
                idx = self._claim_and_publish(
                    meta,
                    dep,
                    work,
                    expert_off,
                    mS1,
                    mS2,
                    w13s,
                    w2s,
                    sorted_pair,
                    topk_w,
                    sItem,
                    sClaim,
                    sDesc,
                    sW,
                    sSc,
                    sTok,
                    sWt,
                    sched,
                    qstate,
                    stid,
                    n1,
                    total,
                    P_rows,
                    bidx,
                    gdim,
                    nsplit,
                )
            qstate.advance()
            while idx < total:
                idx = self._claim_and_publish(
                    meta,
                    dep,
                    work,
                    expert_off,
                    mS1,
                    mS2,
                    w13s,
                    w2s,
                    sorted_pair,
                    topk_w,
                    sItem,
                    sClaim,
                    sDesc,
                    sW,
                    sSc,
                    sTok,
                    sWt,
                    sched,
                    qstate,
                    stid,
                    n1,
                    total,
                    P_rows,
                    Int32(-1),
                    gdim,
                    nsplit,
                )
                qstate.advance()
        elif warp_idx == self.sig_warp:
            # ---------------- dependency signaller ----------------
            cute.arch.griddepcontrol_wait()
            sgs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 4)
            sigq.consumer_wait(sgs)
            jv = sSig[sgs.index]
            while jv >= 0:
                if lane == 0:
                    cute.arch.fence_acq_rel_gpu()
                    cute.arch.atomic_add(
                        dep.iterator + jv, Int32(1), sem="release", scope="gpu"
                    )
                sigq.consumer_release(sgs)
                sgs.advance()
                sigq.consumer_wait(sgs)
                jv = sSig[sgs.index]
        elif warp_idx == self.prod_warp:
            # ---------------- TMA producer ----------------
            # Decode streams each expert's weights once; keep the reusable activation tiles in L2.
            wpol = {}
            bpol = {}
            if const_expr(self.evict_first_weights):
                wpol = {"cache_policy": cutlass.Int64(0x12F0000000000000)}
                bpol = {"cache_policy": cutlass.Int64(0x14F0000000000000)}
            pstate = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.stages
            )
            qs = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.qstages
            )
            gA1 = cute.local_tile(mA1, (BM, BK), (None, 0))
            gA2 = cute.local_tile(mA2, (BM, BK), (None, 0))
            tA1s, tA1g = cute.nvgpu.cpasync.tma_partition(
                tma_a1,
                0,
                cute.make_layout(1),
                cute.group_modes(sA, 0, 2),
                cute.group_modes(gA1, 0, 2),
            )
            tA2s, tA2g = cute.nvgpu.cpasync.tma_partition(
                tma_a2,
                0,
                cute.make_layout(1),
                cute.group_modes(sA, 0, 2),
                cute.group_modes(gA2, 0, 2),
            )
            npre = Int32(0)
            j0 = bidx // self.mt1
            mt0 = bidx - j0 * self.mt1
            row00 = Int32(0)
            if bidx < n1:
                e0 = work[j0, 0]
                row00 = work[j0, 1]
                abase0 = (e0 * self.mt1 + mt0) * self.nkb1
                npre = Int32(min(self.stages, self.nkb1))
                for kb in cutlass.range(npre, unroll=1):
                    mainloop.producer_acquire(pstate)
                    bar = mainloop.producer_get_barrier(pstate)
                    cute.copy(
                        tma_a1,
                        tA1g[(None, abase0 + kb)],
                        tA1s[(None, pstate.index)],
                        tma_bar_ptr=bar,
                        **wpol,
                    )
                    pstate.advance()
            cute.arch.griddepcontrol_wait()
            gB10 = cute.local_tile(
                cute.domain_offset((row00, 0), mB1), (BN, BK), (0, None)
            )
            tB10s, tB10g = cute.nvgpu.cpasync.tma_partition(
                tma_b1,
                0,
                cute.make_layout(1),
                cute.group_modes(sB, 0, 2),
                cute.group_modes(gB10, 0, 2),
            )
            bstate = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.stages
            )
            for kb in cutlass.range(npre, unroll=1):
                bar = mainloop.producer_get_barrier(bstate)
                cute.copy(
                    tma_b1,
                    tB10g[(None, kb)],
                    tB10s[(None, bstate.index)],
                    tma_bar_ptr=bar,
                    **bpol,
                )
                bstate.advance()
            sched.consumer_wait(qs)
            idx = sItem[qs.index]
            while idx >= 0:
                slot = qs.index
                abase = sDesc[0, slot]
                row0 = sDesc[1, slot]
                is1 = sDesc[2, slot] != 0
                nkb = sDesc[3, slot]
                kb0 = sDesc[7, slot]
                if not is1:
                    cute.arch.fence_proxy("async.global")
                gB1 = cute.local_tile(
                    cute.domain_offset((row0 if is1 else Int32(0), 0), mB1),
                    (BN, BK),
                    (0, None),
                )
                gB2 = cute.local_tile(
                    cute.domain_offset((Int32(0) if is1 else row0, 0), mB2),
                    (BN, BK),
                    (0, None),
                )
                tB1s, tB1g = cute.nvgpu.cpasync.tma_partition(
                    tma_b1,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sB, 0, 2),
                    cute.group_modes(gB1, 0, 2),
                )
                tB2s, tB2g = cute.nvgpu.cpasync.tma_partition(
                    tma_b2,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sB, 0, 2),
                    cute.group_modes(gB2, 0, 2),
                )
                kstart = Int32(0)
                if idx == bidx:
                    kstart = npre
                for kb in cutlass.range(kstart, nkb, unroll=1):
                    mainloop.producer_acquire(pstate)
                    bar = mainloop.producer_get_barrier(pstate)
                    if is1:
                        cute.copy(
                            tma_a1,
                            tA1g[(None, abase + kb)],
                            tA1s[(None, pstate.index)],
                            tma_bar_ptr=bar,
                            **wpol,
                        )
                        cute.copy(
                            tma_b1,
                            tB1g[(None, kb)],
                            tB1s[(None, pstate.index)],
                            tma_bar_ptr=bar,
                            **bpol,
                        )
                    else:
                        cute.copy(
                            tma_a2,
                            tA2g[(None, abase + kb)],
                            tA2s[(None, pstate.index)],
                            tma_bar_ptr=bar,
                            **wpol,
                        )
                        cute.copy(
                            tma_b2,
                            tB2g[(None, (kb0 + kb) & 3)],
                            tB2s[(None, pstate.index)],
                            tma_bar_ptr=bar,
                            **bpol,
                        )
                    mainloop.producer_commit(pstate)
                    pstate.advance()
                sched.consumer_release(qs)
                qs.advance()
                sched.consumer_wait(qs)
                idx = sItem[qs.index]
            mainloop.producer_tail(pstate)
        elif warp_idx < self.prod_warp:
            # ---------------- consumers ----------------
            cute.arch.setmaxregister_increase(self.cons_regs)
            cute.arch.griddepcontrol_wait()
            ctid = tidx
            cwarp = warp_idx
            cons_bar = pipeline.NamedBarrier(
                barrier_id=1, num_threads=self.num_cons_threads
            )
            thr_mma = tiled_mma.get_slice(ctid)
            tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sA))
            tCrB0 = tiled_mma.make_fragment_B(
                thr_mma.partition_B(
                    cute.local_tile(sB, (BNS, BK, self.stages), (0, 0, 0))
                )
            )
            tCcC = thr_mma.partition_C(cute.make_identity_tensor((BM, BN)))
            tCcT = thr_mma.partition_C(cute.make_identity_tensor((BM, BNS)))
            acc = cute.make_rmem_tensor(tCcC.shape, Float32)
            tmp = cute.make_rmem_tensor(tCcT.shape, Float32)
            tmp2 = cute.make_rmem_tensor(tCcT.shape, Float32)
            tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
            num_kblk = cute.size(tCrA, mode=[2])
            sx = cute.make_rmem_tensor((NJS * 2,), Float32)
            qv = cute.make_rmem_tensor((2,), Float32)
            q8 = cute.make_rmem_tensor((2,), cutlass.Float8E4M3FN)
            sHb = cute.make_tensor(
                cute.recast_ptr(sH.iterator, dtype=cutlass.Float8E4M3FN), sH.layout
            )
            # row (weight row in tile) of register (cc, hh, jj, mi) = rbase + mi*128 + hh*8
            rbase = tCcC[0][0]

            cstate = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.stages
            )
            rstate = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.stages
            )
            qs = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.qstages
            )
            sgstate = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 4
            )
            sched.consumer_wait(qs)
            idx = sItem[qs.index]
            while idx >= 0:
                slot = qs.index
                row0 = sDesc[1, slot]
                is1 = sDesc[2, slot] != 0
                nkb = sDesc[3, slot]
                kb0 = sDesc[7, slot]
                nvalid = sDesc[4, slot]
                j = sDesc[5, slot]
                mt = sDesc[6, slot]

                acc.fill(0.0)
                # software-pipelined k loop: the WGMMAs of k-block kb+1 run while k-block kb is
                # promoted into the fp32 accumulator (two partial accumulators tmp / tmp2)
                mainloop.consumer_wait(cstate)
                self._issue(tiled_mma, tmp, tCrA, tCrB0, cstate.index)
                cstate.advance()
                for kb2 in cutlass.range(nkb // 2 - 1, unroll=1):
                    kb = kb2 * 2
                    mainloop.consumer_wait(cstate)
                    self._issue(tiled_mma, tmp2, tCrA, tCrB0, cstate.index)
                    cstate.advance()
                    cute.nvgpu.warpgroup.wait_group(1)
                    mainloop.consumer_release(rstate)
                    rstate.advance()
                    self._promote(acc, tmp, sx, sSc, sW, tCcC, kb0 + kb, is1, slot)
                    mainloop.consumer_wait(cstate)
                    self._issue(tiled_mma, tmp, tCrA, tCrB0, cstate.index)
                    cstate.advance()
                    cute.nvgpu.warpgroup.wait_group(1)
                    mainloop.consumer_release(rstate)
                    rstate.advance()
                    self._promote(acc, tmp2, sx, sSc, sW, tCcC, kb0 + kb + 1, is1, slot)
                    if not is1:
                        self._epi2(
                            acc,
                            sWt,
                            sTok,
                            sOut,
                            accbuf,
                            tCcC,
                            kb0 + kb + 1,
                            mt,
                            nvalid,
                            slot,
                            ctid,
                            kb + 1 == nkb - 1,
                        )
                kbl = nkb - 2
                mainloop.consumer_wait(cstate)
                self._issue(tiled_mma, tmp2, tCrA, tCrB0, cstate.index)
                cstate.advance()
                cute.nvgpu.warpgroup.wait_group(1)
                mainloop.consumer_release(rstate)
                rstate.advance()
                self._promote(acc, tmp, sx, sSc, sW, tCcC, kb0 + kbl, is1, slot)
                cute.nvgpu.warpgroup.wait_group(0)
                mainloop.consumer_release(rstate)
                rstate.advance()
                self._promote(acc, tmp2, sx, sSc, sW, tCcC, kb0 + kbl + 1, is1, slot)
                if not is1:
                    self._epi2(
                        acc,
                        sWt,
                        sTok,
                        sOut,
                        accbuf,
                        tCcC,
                        kb0 + kbl + 1,
                        mt,
                        nvalid,
                        slot,
                        ctid,
                        True,
                    )

                if is1:
                    # ---- SiLU-GLU (in place, into the gate slots) + 1x128 FP8 requant of h ----
                    for ni in cutlass.range_constexpr(NSUB):
                        for mi in cutlass.range_constexpr(2):
                            for jj in cutlass.range_constexpr(NJS):
                                for cc in cutlass.range_constexpr(2):
                                    ig = cc + 4 * jj + 4 * NJS * mi + 8 * NJS * ni
                                    g = acc[ig]
                                    # silu(g) * u = g * u / (1 + 2^(-g * log2(e)))
                                    acc[ig] = (
                                        g
                                        * cute.arch.rcp_approx(
                                            Float32(1.0)
                                            + cute.math.exp2(
                                                g * Float32(-1.4426950408889634),
                                                fastmath=True,
                                            )
                                        )
                                        * acc[ig + 2]
                                    )
                    for ni in cutlass.range_constexpr(NSUB):
                        for jj in cutlass.range_constexpr(NJS):
                            for cc in cutlass.range_constexpr(2):
                                ig = cc + 4 * jj + 8 * NJS * ni
                                a0 = acc[ig]
                                a1 = acc[ig + 4 * NJS]
                                am = cute.arch.fmax(
                                    cute.arch.fmax(a0, -a0), cute.arch.fmax(a1, -a1)
                                )
                                am = cute.arch.fmax(
                                    am, cute.arch.shuffle_sync_bfly(am, 4)
                                )
                                am = cute.arch.fmax(
                                    am, cute.arch.shuffle_sync_bfly(am, 8)
                                )
                                am = cute.arch.fmax(
                                    am, cute.arch.shuffle_sync_bfly(am, 16)
                                )
                                if lane < 4:
                                    sRed[cwarp, tCcC[ig][1]] = am
                    cons_bar.arrive_and_wait()
                    # one warp finishes the per-token amax over all consumer warps and stores scale / 1/scale
                    if cwarp == 0:
                        for c0 in cutlass.range_constexpr((BN + 31) // 32):
                            col = lane + 32 * c0
                            if col < BN:
                                am = sRed[0, col]
                                for w in cutlass.range_constexpr(1, 8):
                                    am = cute.arch.fmax(am, sRed[w, col])
                                am = cute.arch.fmax(am, Float32(QUANT_EPS))
                                scale = am / Float32(FP8_MAX)
                                sInv[col] = Float32(FP8_MAX) / am
                                if col < nvalid:
                                    mS2[row0 + col, mt] = scale
                    cons_bar.arrive_and_wait()
                    for ni in cutlass.range_constexpr(NSUB):
                        for jj in cutlass.range_constexpr(NJS):
                            for cc in cutlass.range_constexpr(2):
                                ig = cc + 4 * jj + 8 * NJS * ni
                                inv = sInv[tCcC[ig][1]]
                                for mi in cutlass.range_constexpr(2):
                                    q = acc[ig + 4 * NJS * mi] * inv
                                    acc[ig + 4 * NJS * mi] = cute.arch.fmin(
                                        cute.arch.fmax(q, Float32(-FP8_MAX)),
                                        Float32(FP8_MAX),
                                    )
                    for ni in cutlass.range_constexpr(NSUB):
                        for jj in cutlass.range_constexpr(NJS):
                            for cc in cutlass.range_constexpr(2):
                                ig = cc + 4 * jj + 8 * NJS * ni
                                col = tCcC[ig][1]
                                for mi in cutlass.range_constexpr(2):
                                    qv[mi] = acc[ig + 4 * NJS * mi]
                                q8.store(qv.load().to(cutlass.Float8E4M3FN))
                                for mi in cutlass.range_constexpr(2):
                                    rg = rbase + mi * 128
                                    sHb[col, (rg // 16) * 8 + rg % 16] = q8[mi]
                    cons_bar.arrive_and_wait()
                    for it in cutlass.range(
                        ctid, BN * 8, self.num_cons_threads, unroll=1
                    ):
                        r = it // 8
                        ch = it % 8
                        if r < nvalid:
                            cute.autovec_copy(
                                cute.local_tile(sHb[r, None], (16,), (ch,)),
                                cute.local_tile(
                                    hq[row0 + r, None], (16,), (mt * 8 + ch,)
                                ),
                            )
                    cons_bar.arrive_and_wait()
                    # hand the GEMM1->GEMM2 release to the signal warp (keeps the gpu-scope fence off
                    # the consumer critical path)
                    sigq.producer_acquire(sgstate)
                    if ctid == 0:
                        sSig[sgstate.index] = j
                    sigq.producer_commit(sgstate)
                    sgstate.advance()
                sched.consumer_release(qs)
                qs.advance()
                sched.consumer_wait(qs)
                idx = sItem[qs.index]
            # stop the signal warp
            sigq.producer_acquire(sgstate)
            if ctid == 0:
                sSig[sgstate.index] = Int32(-1)
            sigq.producer_commit(sgstate)
            # The persistent grid is exactly one resident CTA per SM.  Once every consumer has
            # deposited its final red.global contribution, use one consumer thread per CTA for a
            # device-wide completion barrier, then reuse the resident grid for fp32 -> bf16.
            cons_bar.arrive_and_wait()
            if ctid == 0:
                _red_release_i32(meta.iterator + 3, Int32(1))
                spins = Int32(0)
                while _ld_acquire_i32(meta.iterator + 3) < gdim:
                    spins = spins + Int32(1)
            cons_bar.arrive_and_wait()
            if bidx == 0 and ctid == 0:
                # Re-arm the fused-router descriptor flag for the next same-stream invocation.
                meta[2] = Int32(0)
            f4 = cute.make_rmem_tensor((4,), Float32)
            o4 = cute.make_rmem_tensor((4,), cutlass.BFloat16)
            for chunk in cutlass.range(
                bidx, cute.size(accbuf, mode=[0]) * 2, gdim, unroll=1
            ):
                m = chunk // 2
                part = chunk - m * 2
                tile = ctid + part * QUANT_THREADS
                cute.autovec_copy(cute.local_tile(accbuf[m, None], (4,), (tile,)), f4)
                o4.store(f4.load().to(cutlass.BFloat16))
                cute.autovec_copy(o4, cute.local_tile(out[m, None], (4,), (tile,)))

    @cute.jit
    def _issue(self, tiled_mma, tmpX, tCrA, tCrB, sidx):
        # fresh 4-instruction chain for one 128-K block (accumulate onto a zeroed partial)
        tmpX.fill(0.0)
        cute.nvgpu.warpgroup.fence()
        for kk in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
            cute.gemm(
                tiled_mma,
                tmpX,
                tCrA[(None, None, kk, sidx)],
                tCrB[(None, None, kk, sidx)],
                tmpX,
            )
        cute.nvgpu.warpgroup.commit_group()

    @cute.jit
    def _promote(self, acc, tmpX, sx, sSc, sW, tCcC, kb, is1, slot):
        """acc += partial(kb) * weight_block_scale(kb, row) * activation_scale(kb, token), in fp32."""
        NJS = self.BNS // 8
        kbx = kb if is1 else (kb & 3)
        for jj in cutlass.range_constexpr(NJS):
            for cc in cutlass.range_constexpr(2):
                sx[jj * 2 + cc] = sSc[tCcC[cc + 4 * jj][1], kbx, slot]
        for mi in cutlass.range_constexpr(2):
            for hh in cutlass.range_constexpr(2):
                w_ = sW[kb * 4 + mi * 2 + hh, slot]
                for jj in cutlass.range_constexpr(NJS):
                    for cc in cutlass.range_constexpr(2):
                        ii = cc + 2 * hh + 4 * jj + 4 * NJS * mi
                        acc[ii] = acc[ii] + tmpX[ii] * (w_ * sx[jj * 2 + cc])

    @cute.jit
    def _epi2(
        self, acc, sWt, sTok, sOut, accbuf, tCcC, kb, mt, nvalid, slot, ctid, last
    ):
        """After every 4 k-blocks, transpose each warpgroup's accumulator through shared memory
        so four adjacent outputs use one 16-byte fp32 global reduction instruction."""
        fire = Int32(1) if (kb & 3) == 3 else Int32(0)
        if last:
            fire = Int32(1)
        if fire != 0:
            OT = self.out_tok
            wg = ctid // 128
            wtid = ctid % 128
            out_mt = mt + (kb >> 2)
            v4 = cute.make_rmem_tensor((4,), Float32)
            for p in cutlass.range_constexpr(self.BN // OT):
                if const_expr(p > 0):
                    cute.arch.barrier(barrier_id=2 + wg, number_of_threads=128)
                for i in cutlass.range_constexpr(cute.size(acc)):
                    n8 = (i // 4) % (self.BN // 8)
                    if const_expr((8 * n8) // OT == p):
                        row = tCcC[i][0]
                        sOut[wg, tCcC[i][1] - p * OT, (row // 128) * 64 + row % 64] = (
                            acc[i]
                        )
                cute.arch.barrier(barrier_id=2 + wg, number_of_threads=128)
                for u in cutlass.range_constexpr(OT // 4):
                    x = wtid + 128 * u
                    n = x // 32
                    q = x % 32
                    if n + p * OT < nvalid:
                        pos = n + p * OT
                        wt = sWt[pos, slot]
                        # Keep the routed token id in int32, but widen before multiplying by H so
                        # a physically representable large-M call cannot overflow the flat offset.
                        base = cutlass.Int64(sTok[pos, slot]) * H
                        cute.autovec_copy(
                            cute.local_tile(sOut[wg, n, None], (4,), (q,)), v4
                        )
                        col = (
                            out_mt * self.BM + (q // 16) * 128 + wg * 64 + 4 * (q % 16)
                        )
                        _red_add_v4_f32(
                            accbuf.iterator + base + col,
                            v4[0] * wt,
                            v4[1] * wt,
                            v4[2] * wt,
                            v4[3] * wt,
                        )
            cute.arch.barrier(barrier_id=2 + wg, number_of_threads=128)
            acc.fill(0.0)

    @cute.jit
    def _claim_static(
        self, work, mS1, w13s, sItem, sDesc, sW, sSc, sched, qstate, stid, P_rows, idx
    ):
        """Publish static GEMM1 descriptor/weight scales before PDL; activation scales after it."""
        BN = self.BN
        slot = qstate.index
        sched.producer_acquire(qstate)
        j = idx // self.mt1
        mt = idx % self.mt1
        e = work[j, 0]
        row0 = work[j, 1]
        nvalid = work[j, 2]
        if stid == 0:
            sDesc[0, slot] = (e * self.mt1 + mt) * self.nkb1
            sDesc[1, slot] = row0
            sDesc[2, slot] = Int32(1)
            sDesc[3, slot] = Int32(self.nkb1)
            sDesc[4, slot] = nvalid
            sDesc[5, slot] = j
            sDesc[6, slot] = mt
            sDesc[7, slot] = Int32(0)
        for r in cutlass.range_constexpr(
            (64 + self.sched_threads - 1) // self.sched_threads
        ):
            t = stid + self.sched_threads * r
            sW[t, slot] = w13s[e, mt + (t % 2) * 4, t // 4]
        cute.arch.griddepcontrol_wait()
        for r in cutlass.range_constexpr(
            (BN * 16 + self.sched_threads - 1) // self.sched_threads
        ):
            t = stid + self.sched_threads * r
            kb = t // BN
            c = t % BN
            v = mS1[cutlass.min(row0 + c, P_rows - 1), kb]
            sSc[c, kb, slot] = v if c < nvalid else Float32(0.0)
        if stid == 0:
            sItem[slot] = idx
        sched.producer_commit(qstate)
        return idx

    @cute.jit
    def _claim_and_publish(
        self,
        meta,
        dep,
        work,
        expert_off,
        mS1,
        mS2,
        w13s,
        w2s,
        sorted_pair,
        topk_w,
        sItem,
        sClaim,
        sDesc,
        sW,
        sSc,
        sTok,
        sWt,
        sched,
        qstate,
        stid,
        n1,
        total,
        P_rows,
        static_idx,
        gdim,
        nsplit,
    ):
        """Two scheduler warps claim the next work item, wait for its GEMM1 dependency (GEMM2
        items), and cooperatively publish scales and token metadata into the item-queue slot."""
        BN = self.BN
        slot = qstate.index
        idx = static_idx
        if static_idx < 0:
            if const_expr(self.sched_threads == 32):
                if stid == 0:
                    idx = (
                        cute.arch.atomic_add(meta.iterator + 1, Int32(1), scope="gpu")
                        + gdim
                    )
                idx = cute.arch.shuffle_sync(idx, 0)
            else:
                if stid == 0:
                    sClaim[slot] = (
                        cute.arch.atomic_add(meta.iterator + 1, Int32(1), scope="gpu")
                        + gdim
                    )
                cute.arch.barrier(barrier_id=4, number_of_threads=self.sched_threads)
                idx = sClaim[slot]
        sched.producer_acquire(qstate)
        if idx < total:
            is1 = idx < n1
            i2 = idx - n1
            j = idx // self.mt1
            mt = idx % self.mt1
            kb0 = Int32(0)
            nk = Int32(self.nkb1) if is1 else Int32(self.nkb2 * self.G2)
            if not is1:
                if const_expr(self.grp_major):
                    # large M: one 256*G2-column slice of the fp32 accumulator per pass, so the
                    # scatter-add working set stays L2 resident (each w2 tile is still read once)
                    T = n1 // self.mt1
                    gq = i2 // T
                    j = i2 - gq * T
                    mt = gq * self.G2
                else:
                    nreg = (n1 // self.mt1) * self.ng2 - nsplit
                    r2 = i2
                    if i2 >= nreg:
                        hh2 = i2 - nreg
                        r2 = nreg + hh2 // 2
                        kb0 = (hh2 % 2) * 2
                        nk = Int32(2)
                    j = r2 // self.ng2
                    mt = (r2 % self.ng2) * self.G2
            e = work[j, 0]
            row0 = work[j, 1]
            nvalid = work[j, 2]
            if stid == 0:
                sDesc[0, slot] = (
                    (e * self.mt1 + mt) * self.nkb1
                    if is1
                    else (e * self.mt2 + mt) * self.nkb2 + kb0
                )
                sDesc[1, slot] = row0
                sDesc[2, slot] = Int32(1) if is1 else Int32(0)
                sDesc[3, slot] = nk
                sDesc[4, slot] = nvalid
                sDesc[5, slot] = j
                sDesc[6, slot] = mt
                sDesc[7, slot] = kb0
            if is1:
                for r in cutlass.range_constexpr(
                    (BN * 16 + self.sched_threads - 1) // self.sched_threads
                ):
                    t = stid + self.sched_threads * r
                    kb = t // BN
                    c = t % BN
                    v = mS1[cutlass.min(row0 + c, P_rows - 1), kb]
                    sSc[c, kb, slot] = v if c < nvalid else Float32(0.0)
                for r in cutlass.range_constexpr(
                    (64 + self.sched_threads - 1) // self.sched_threads
                ):
                    t = stid + self.sched_threads * r
                    sW[t, slot] = w13s[e, mt + (t % 2) * 4, t // 4]
            else:
                # GEMM2: wait until the 4 GEMM1 tiles of this token tile have stored h and its scales
                if stid == 0:
                    cnt = cute.arch.atomic_add(
                        dep.iterator + j, Int32(0), sem="acquire", scope="gpu"
                    )
                    while cnt < self.mt1:
                        cnt = cute.arch.atomic_add(
                            dep.iterator + j, Int32(0), sem="acquire", scope="gpu"
                        )
                if const_expr(self.sched_threads == 32):
                    cute.arch.sync_warp()
                else:
                    cute.arch.barrier(
                        barrier_id=4, number_of_threads=self.sched_threads
                    )
                cute.arch.fence_proxy("async.global")
                for r in cutlass.range_constexpr(
                    (BN * 4 + self.sched_threads - 1) // self.sched_threads
                ):
                    t = stid + self.sched_threads * r
                    if t < BN * 4:
                        kb = t // BN
                        c = t % BN
                        v = mS2[cutlass.min(row0 + c, P_rows - 1), kb]
                        sSc[c, kb, slot] = v if c < nvalid else Float32(0.0)
                for r in cutlass.range_constexpr(
                    (16 * self.G2 + self.sched_threads - 1) // self.sched_threads
                ):
                    t = stid + self.sched_threads * r
                    if t < 16 * self.G2:
                        # t = (mts * 4 + kb) * 4 + mi * 2 + hh
                        sW[t, slot] = w2s[
                            e, (mt + t // 16) * 2 + (t // 2) % 2, (t // 4) % 4
                        ]
                # token row offsets and router weights of this tile's columns
                for r in cutlass.range_constexpr(
                    (BN + self.sched_threads - 1) // self.sched_threads
                ):
                    c = stid + self.sched_threads * r
                    if c < BN:
                        rr = cutlass.min(row0 + c, P_rows - 1)
                        tv = sorted_pair[rr]
                        wv = topk_w[rr]
                        sTok[c, slot] = tv
                        sWt[c, slot] = wv if c < nvalid else Float32(0.0)
        if stid == 0:
            sItem[slot] = idx if idx < total else Int32(-1)
        sched.producer_commit(qstate)
        return idx


# --------------------------------------------------------------------------------------------
# host side
# --------------------------------------------------------------------------------------------
_compiled: dict = {}


def _sym(div=1):
    return cute.sym_int(divisibility=div)


def _ft(dtype, shape, stride):
    return make_fake_tensor(dtype, shape, stride, assumed_align=16)


def _get_compiled(key):
    if key in _compiled:
        return _compiled[key]
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    fp8 = cutlass.Float8E4M3FN
    kind = key[0]
    if kind == "route":
        BN = key[1]
        P = _sym(8)
        W = _sym()
        fn = cute.compile(
            _route_host,
            _ft(Int32, (P,), (1,)),
            _ft(Float32, (P,), (1,)),
            _ft(Int32, (E + 1,), (1,)),
            _ft(Int32, (W, 4), (4, 1)),
            _ft(Int32, (4,), (1,)),
            _ft(Int32, (P,), (1,)),
            _ft(Int32, (P,), (1,)),
            _ft(Float32, (P,), (1,)),
            _ft(Int32, (W,), (1,)),
            Int32(1),
            BN,
            stream,
            options="--enable-tvm-ffi",
        )
    elif kind == "route_mc":
        BN = key[1]
        P = _sym(8)
        W = _sym()
        fn = cute.compile(
            _route_mc_host,
            _ft(Int32, (P,), (1,)),
            _ft(Float32, (P,), (1,)),
            _ft(Int32, (E + 1,), (1,)),
            _ft(Int32, (W, 4), (4, 1)),
            _ft(Int32, (4,), (1,)),
            _ft(Int32, (P,), (1,)),
            _ft(Int32, (P,), (1,)),
            _ft(Float32, (P,), (1,)),
            _ft(Int32, (W,), (1,)),
            _ft(Int32, (RMC_CTAS, E), (E, 1)),
            _ft(Int32, (2,), (1,)),
            Int32(1),
            BN,
            stream,
            options="--enable-tvm-ffi",
        )
    elif kind == "quant_route":
        BN, NQ, NCH = key[1:4]
        M = _sym()
        P = _sym(8)
        PS = _sym(4)
        W = _sym()
        fn = cute.compile(
            _quant_route_host,
            _ft(cutlass.BFloat16, (M, H), (H, 1)),
            _ft(Int32, (P,), (1,)),
            _ft(Float32, (P,), (1,)),
            _ft(Int32, (E + 1,), (1,)),
            _ft(Int32, (W, 4), (4, 1)),
            _ft(Int32, (4,), (1,)),
            _ft(Int32, (P,), (1,)),
            _ft(Float32, (P,), (1,)),
            _ft(Int32, (W,), (1,)),
            _ft(fp8, (P, H), (H, 1)),
            _ft(Float32, (H // 128, PS), (PS, 1)),
            _ft(Float32, (M, H), (H, 1)),
            Int32(1),
            BN,
            NQ,
            NCH,
            stream,
            options="--enable-tvm-ffi",
        )
    elif kind == "quant":
        M = _sym()
        P = _sym()
        PS = _sym(4)
        fn = cute.compile(
            _quant_host,
            _ft(cutlass.BFloat16, (M, H), (H, 1)),
            _ft(Int32, (P,), (1,)),
            _ft(fp8, (P, H), (H, 1)),
            _ft(Float32, (H // 128, PS), (PS, 1)),
            _ft(Float32, (M, H), (H, 1)),
            Int32(1),
            stream,
            options="--enable-tvm-ffi",
        )
    elif kind == "finalize":
        parts = key[1]
        M = _sym()
        fn = cute.compile(
            _finalize_host,
            _ft(Float32, (M, H), (H, 1)),
            _ft(cutlass.BFloat16, (M, H), (H, 1)),
            _ft(Int32, (4,), (1,)),
            Int32(1),
            parts,
            stream,
            options="--enable-tvm-ffi",
        )
    elif kind == "fused":
        _, BN, stages = key[:3]
        g = MoEFused(BN, stages)
        P = _sym()
        PS = _sym(4)
        M = _sym()
        W = _sym()
        fn = cute.compile(
            g,
            _ft(fp8, (E * 2 * I * (H // 128), 128), (128, 1)),
            _ft(fp8, (P, H), (H, 1)),
            _ft(Float32, (PS, H // 128), (1, PS)),
            _ft(
                Float32,
                (E, 2 * I // 128, H // 128),
                ((2 * I // 128) * (H // 128), H // 128, 1),
            ),
            _ft(fp8, (E * H * (I // 128), 128), (128, 1)),
            _ft(fp8, (P, I), (I, 1)),
            _ft(Float32, (PS, I // 128), (1, PS)),
            _ft(
                Float32, (E, H // 128, I // 128), ((H // 128) * (I // 128), I // 128, 1)
            ),
            _ft(Int32, (E + 1,), (1,)),
            _ft(Int32, (W, 4), (4, 1)),
            _ft(Int32, (4,), (1,)),
            _ft(Int32, (W,), (1,)),
            _ft(Float32, (M, H), (H, 1)),
            _ft(cutlass.BFloat16, (M, H), (H, 1)),
            _ft(Int32, (P,), (1,)),
            _ft(Float32, (P,), (1,)),
            _ft(Int32, (P,), (1,)),
            Int32(1),
            Int32(1),
            stream,
            options="--enable-tvm-ffi",
        )
    else:
        raise ValueError(key)
    _compiled[key] = fn
    return fn


def _pick_bn(M):
    # general rule on the expected tokens per expert (M * top-k / E): small token tiles for decode,
    # wider ones once experts see tens of tokens
    if M * TOPK <= 512:
        return 16
    if M * TOPK <= 16384:
        return 32
    return 64


_SMEM_BYTES = 232448


def _pick_qstages(BN):
    return 3 if BN <= 64 else 2


def _pick_stages(BN):
    stage = 256 * 128 + BN * 128
    q = _pick_qstages(BN)
    out_smem = 2 * min(BN, 16) * (128 + 4) * 4
    extra = (
        1024
        + q * (48 + 64 * 4 + 16 * BN * 4 + 8 * BN)
        + 8 * BN * 4
        + BN * 128
        + out_smem
        + 2048
        + (2 * E + 16) * 4
    )
    return min(8, (_SMEM_BYTES - extra) // stage)
