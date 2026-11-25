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

Decode-sized batches (M < 2048) run 3 launches:
  1. quant+route : per token - dynamic 1x128 FP8 quant of x, stable expert counting sort of its 8
                   (token, k) pairs, scatter to sorted slots; CTA 0 builds the token-tile list.
  2. fused GEMM1+GEMM2 (swap-AB): weights = WGMMA M (256 rows), tokens = WGMMA N (16/32), TMA
                   pipeline, per-128-K promotion with weight x activation scales in fp32, fused
                   SiLU-GLU + 1x128 FP8 re-quant of h; GEMM2 multiplies the router weight and
                   red.adds into an fp32 [M, H] accumulator (top-k reduction in fp32).
  3. finalize    : fp32 accumulator -> bf16 out.
Large batches (M >= 2048) use a multi-CTA router + quant kernel and a non-swapped persistent GEMM
(128 sorted token rows = WGMMA M, 256 weight rows = two WGMMA N=128 halves) whose per-128-K
promotion needs only one activation scale per token row; GEMM1's epilogue does SiLU-GLU and the h
requant entirely inside each warp (gate and up halves land in the same registers).
"""

import math

import torch

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass import Float32, Int16, Int32, const_expr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

E, TOPK, H, I = 256, 8, 2048, 512
GROUP = 128
FP8_MAX = 448.0
QUANT_EPS = 1e-10

ROUTE_THREADS = 1024
QUANT_THREADS = 256  # 2048 / 8 elements per thread
QROWS = 4  # token rows per CTA of the large-batch quant kernel


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
    sorted_tok: cute.Tensor,  # (P,) int32 token * H of each sorted row
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
        meta[2] = Int32(0)
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
            sorted_tok[pos] = blk * H
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


RMC_CTAS = 48
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
            meta[2] = Int32(0)
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
            # Re-arm the reusable arrival counter before publishing the next generation.
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


@cute.kernel
def _route_mc_quantc_kernel(
    ids: cute.Tensor,
    tw: cute.Tensor,
    expert_off: cute.Tensor,
    wdesc: cute.Tensor,
    meta: cute.Tensor,
    sorted_tok: cute.Tensor,
    sorted_wt: cute.Tensor,
    dep: cute.Tensor,
    hist: cute.Tensor,
    gbar: cute.Tensor,
    x: cute.Tensor,
    xqc: cute.Tensor,
    xsc: cute.Tensor,
    P: Int32,
    grid_n: Int32,
    BN: cutlass.Constexpr,
):
    """Run the fixed 64-CTA prefill router and independent four-row quant CTAs in one launch."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    cute.arch.griddepcontrol_launch_dependents()
    # Launched with PDL: wait for the producer of ids / tw / x before reading them.
    cute.arch.griddepcontrol_wait()
    smem = utils.SmemAllocator()
    s_cnt = smem.allocate_tensor(Int32, cute.make_layout(E), byte_alignment=16)
    s_wsum = smem.allocate_tensor(Int32, cute.make_layout(16), byte_alignment=16)
    s_last = smem.allocate_tensor(Int32, cute.make_layout(2), byte_alignment=16)

    if bidx < RMC_CTAS:
        g = bidx
        nd = cute.size(dep)
        for i in cutlass.range(
            g * RMC_THREADS + tidx, nd, RMC_CTAS * RMC_THREADS, unroll=1
        ):
            dep[i] = Int32(0)
        if g == 0:
            if tidx == 0:
                # The persistent grid owns items [0, grid_n); dynamic claims begin after them.
                meta[1] = grid_n
                meta[2] = Int32(0)
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

        cute.arch.sync_threads()
        if tidx == 0:
            gen0 = cute.arch.atomic_add(
                gbar.iterator + 1, Int32(0), sem="acquire", scope="gpu"
            )
            cute.arch.fence_acq_rel_gpu()
            old = cute.arch.atomic_add(
                gbar.iterator, Int32(1), sem="acq_rel", scope="gpu"
            )
            s_last[0] = Int32(1) if (old % RMC_CTAS) == RMC_CTAS - 1 else Int32(0)
            s_last[1] = gen0
        cute.arch.sync_threads()
        if s_last[0] != 0:
            cute.arch.fence_acq_rel_gpu()
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

        s_cnt[tidx] = hist[g, tidx]
        cute.arch.sync_threads()
        wv = cute.make_rmem_tensor((TOPK,), Float32)
        for blk in cutlass.range(b0 + tidx, b1, RMC_THREADS, unroll=1):
            cute.autovec_copy(cute.local_tile(ids, (TOPK,), (blk,)), v)
            cute.autovec_copy(cute.local_tile(tw, (TOPK,), (blk,)), wv)
            for i in cutlass.range_constexpr(TOPK):
                pos = cute.arch.atomic_add(s_cnt.iterator + v[i], Int32(1), scope="cta")
                sorted_tok[pos] = blk
                sorted_wt[pos] = wv[i]
    else:
        bm = bidx - RMC_CTAS
        Mx = cute.size(x, mode=[0])
        fx_bf = cute.make_rmem_tensor(
            cute.make_layout((QROWS, 8), stride=(8, 1)), cutlass.BFloat16
        )
        for rr in cutlass.range_constexpr(QROWS):
            mq = cutlass.min(bm * QROWS + rr, Mx - 1)
            cute.autovec_copy(
                cute.local_tile(x[mq, None], (8,), (tidx,)), fx_bf[rr, None]
            )
        for rr in cutlass.range_constexpr(QROWS):
            m = bm * QROWS + rr
            fx = cute.make_rmem_tensor((8,), Float32)
            fx.store(fx_bf[rr, None].load().to(Float32))
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
            if m < Mx:
                cute.autovec_copy(q8, cute.local_tile(xqc[m, None], (8,), (tidx,)))
                if tidx % 16 == 0:
                    xsc[m, tidx // 16] = scale


@cute.jit
def _route_mc_quantc_host(
    ids,
    tw,
    expert_off,
    wdesc,
    meta,
    sorted_tok,
    sorted_wt,
    dep,
    hist,
    gbar,
    x,
    xqc,
    xsc,
    P: Int32,
    M: Int32,
    grid_n: Int32,
    BN: cutlass.Constexpr,
    stream,
):
    _route_mc_quantc_kernel(
        ids,
        tw,
        expert_off,
        wdesc,
        meta,
        sorted_tok,
        sorted_wt,
        dep,
        hist,
        gbar,
        x,
        xqc,
        xsc,
        P,
        grid_n,
        BN,
    ).launch(
        grid=[RMC_CTAS + (M + QROWS - 1) // QROWS, 1, 1],
        block=[RMC_THREADS, 1, 1],
        stream=stream,
        use_pdl=True,
    )


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
    scale = cute.arch.fmax(amax, Float32(QUANT_EPS)) / Float32(FP8_MAX)
    for i in cutlass.range_constexpr(8):
        q = fx[i] / scale
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


@cute.kernel
def _quantc_kernel(
    x: cute.Tensor,  # (M, H) bf16
    xqc: cute.Tensor,  # (M, H) fp8, token order
    xsc: cute.Tensor,  # (M, H/128) fp32, token order
):
    """Dynamic 1x128 FP8 quant of QROWS token rows per CTA, stored once in token order (the large-batch GEMM
    gathers the rows of each expert tile directly from this compact copy). All row loads of a thread are
    issued before any arithmetic so each thread keeps QROWS independent 16-byte loads in flight."""
    tidx, _, _ = cute.arch.thread_idx()
    bm, _, _ = cute.arch.block_idx()
    cute.arch.griddepcontrol_launch_dependents()
    Mx = cute.size(x, mode=[0])
    fx_bf = cute.make_rmem_tensor(
        cute.make_layout((QROWS, 8), stride=(8, 1)), cutlass.BFloat16
    )
    for rr in cutlass.range_constexpr(QROWS):
        mq = cutlass.min(bm * QROWS + rr, Mx - 1)
        cute.autovec_copy(cute.local_tile(x[mq, None], (8,), (tidx,)), fx_bf[rr, None])
    for rr in cutlass.range_constexpr(QROWS):
        m = bm * QROWS + rr
        fx = cute.make_rmem_tensor((8,), Float32)
        fx.store(fx_bf[rr, None].load().to(Float32))
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
        if m < Mx:
            cute.autovec_copy(q8, cute.local_tile(xqc[m, None], (8,), (tidx,)))
            if tidx % 16 == 0:
                xsc[m, tidx // 16] = scale
    # complete only after the routing kernel so the GEMM's dependency wait covers both launches
    cute.arch.griddepcontrol_wait()


@cute.jit
def _quantc_host(x, xqc, xsc, M: Int32, stream):
    _quantc_kernel(x, xqc, xsc).launch(
        grid=[(M + QROWS - 1) // QROWS, 1, 1],
        block=[QUANT_THREADS, 1, 1],
        stream=stream,
        use_pdl=True,
    )


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
):
    """Small-M fused quantization and stable expert counting-sort.

    Every token CTA recomputes the small expert histogram in shared memory so it can derive its
    eight sorted positions without a preceding routing launch. CTA 0 also publishes the common
    expert offsets, token-tile descriptors, and cleared dependency state.
    """
    tidx, _, _ = cute.arch.thread_idx()
    m, _, _ = cute.arch.block_idx()
    lane = tidx % 32
    warp = tidx // 32
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
    scale = cute.arch.fmax(amax, Float32(QUANT_EPS)) / Float32(FP8_MAX)
    for i in cutlass.range_constexpr(8):
        q = fx[i] / scale
        fx[i] = cute.arch.fmin(cute.arch.fmax(q, Float32(-FP8_MAX)), Float32(FP8_MAX))
    q8 = cute.make_rmem_tensor((8,), cutlass.Float8E4M3FN)
    q8.store(fx.load().to(cutlass.Float8E4M3FN))

    smem = utils.SmemAllocator()
    s_cnt = smem.allocate_tensor(Int32, cute.make_layout(E), byte_alignment=16)
    s_wsum = smem.allocate_tensor(Int32, cute.make_layout(16), byte_alignment=16)
    s_slot = smem.allocate_tensor(Int32, cute.make_layout(TOPK), byte_alignment=16)
    s_e = smem.allocate_tensor(Int32, cute.make_layout(TOPK), byte_alignment=16)
    s_lo = smem.allocate_tensor(
        Int32, cute.make_layout((8, TOPK), stride=(TOPK, 1)), byte_alignment=16
    )
    s_pr = smem.allocate_tensor(
        Int32, cute.make_layout((8, TOPK), stride=(TOPK, 1)), byte_alignment=16
    )
    s_cnt[tidx] = Int32(0)
    if tidx < TOPK:
        s_e[tidx] = ids[m * TOPK + tidx]
    cute.arch.sync_threads()

    P = cute.size(ids)
    cur = m * TOPK
    rlo = cute.make_rmem_tensor((TOPK,), Int32)
    rpr = cute.make_rmem_tensor((TOPK,), Int32)
    rlo.fill(0)
    rpr.fill(0)
    for p in cutlass.range(tidx, P, QUANT_THREADS, unroll=4):
        e = ids[p]
        if m == 0:
            cute.arch.atomic_add(s_cnt.iterator + e, Int32(1), scope="cta")
        for k in cutlass.range_constexpr(TOPK):
            ek = s_e[k]
            if e < ek:
                rlo[k] = rlo[k] + 1
            if e == ek:
                if p < cur + k:
                    rpr[k] = rpr[k] + 1
    for k in cutlass.range_constexpr(TOPK):
        lo = rlo[k]
        pr = rpr[k]
        for i in cutlass.range_constexpr(5):
            lo = lo + cute.arch.shuffle_sync_bfly(lo, 1 << i)
            pr = pr + cute.arch.shuffle_sync_bfly(pr, 1 << i)
        if lane == 0:
            s_lo[warp, k] = lo
            s_pr[warp, k] = pr
    cute.arch.sync_threads()

    if warp == 0:
        if lane < TOPK:
            pos = Int32(0)
            for w in cutlass.range_constexpr(8):
                pos = pos + s_lo[w, lane] + s_pr[w, lane]
            p = cur + lane
            s_slot[lane] = pos
            sorted_tok[pos] = m * H
            sorted_wt[pos] = tw[p]
    cute.arch.sync_threads()

    if m == 0:
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
            meta[2] = Int32(0)
        if tidx == E - 1:
            expert_off[E] = P
            meta[0] = ext + nt
    cute.arch.sync_threads()

    grp = tidx // 16
    for k in cutlass.range_constexpr(TOPK):
        pos = s_slot[k]
        cute.autovec_copy(q8, cute.local_tile(xq[pos, None], (8,), (tidx,)))
        if tidx % 16 == 0:
            xsT[grp, pos] = scale
    z = cute.make_rmem_tensor((8,), Float32)
    z.fill(0.0)
    cute.autovec_copy(z, cute.local_tile(accbuf[m, None], (8,), (tidx,)))


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
    M: Int32,
    BN: cutlass.Constexpr,
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
    ).launch(grid=[M, 1, 1], block=[QUANT_THREADS, 1, 1], stream=stream)


# --------------------------------------------------------------------------------------------
# 5. finalize fp32 -> bf16
# --------------------------------------------------------------------------------------------
@cute.kernel
def _finalize_kernel(accbuf: cute.Tensor, out: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    m, _, _ = cute.arch.block_idx()
    cute.arch.griddepcontrol_wait()
    f = cute.make_rmem_tensor((8,), Float32)
    cute.autovec_copy(cute.local_tile(accbuf[m, None], (8,), (tidx,)), f)
    o = cute.make_rmem_tensor((8,), cutlass.BFloat16)
    o.store(f.load().to(cutlass.BFloat16))
    cute.autovec_copy(o, cute.local_tile(out[m, None], (8,), (tidx,)))


@cute.jit
def _finalize_host(accbuf, out, M: Int32, stream):
    _finalize_kernel(accbuf, out).launch(
        grid=[M, 1, 1], block=[QUANT_THREADS, 1, 1], stream=stream, use_pdl=True
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
        self.ng2 = (H // 256) // self.G2
        assert BN <= 64
        self.BNS = BN
        self.NSUB = 1
        self.num_cons_threads = 256
        self.threads = (
            384  # WG0, WG1 consumers; WG2: TMA warp, scheduler warp, 2 idle warps
        )
        self.cons_regs = 224
        self.prod_regs = 56
        self.prod_warp = 8  # TMA issue
        self.sched_warp = 9  # work claiming + per-item scale / token preparation
        self.sig_warp = 10  # GEMM1 -> GEMM2 dependency release
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
        sorted_pair: cute.Tensor,  # (P,) int32 token * H per sorted row
        topk_w: cute.Tensor,  # (P,) fp32 router weight per sorted row
        ids: cute.Tensor,  # (P,) int32 flattened topk_ids
        out: cute.Tensor,  # (M, H) bf16
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
            sScl: cute.struct.Align[cute.struct.MemRange[Float32, self.BN], 16]
            sInv: cute.struct.Align[cute.struct.MemRange[Float32, self.BN], 16]
            sItem: cute.struct.Align[cute.struct.MemRange[Int32, self.qstages], 16]
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
            sH: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, self.BN * 128], 16
            ]
            sHc: cute.struct.Align[cute.struct.MemRange[Int32, E], 16]
            sToff: cute.struct.Align[cute.struct.MemRange[Int32, E + 16], 16]
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
            sorted_pair,
            topk_w,
            ids,
            out,
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
        sorted_pair: cute.Tensor,
        topk_w: cute.Tensor,
        ids: cute.Tensor,
        out: cute.Tensor,
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
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_cons_threads + 32
            ),
        )
        cute.arch.griddepcontrol_launch_dependents()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        # The token-tile table only depends on topk_ids (an input of the op), so every CTA rebuilds its
        # counts with the router's rule before griddepcontrol.wait; this lets the TMA warp stream the
        # weight boxes of its statically assigned first item while the quant / routing kernel runs.
        # Everything that reads the preceding kernels' results waits per role below.
        sHc = storage.sHc.get_tensor(cute.make_layout(E))
        sToff = storage.sToff.get_tensor(cute.make_layout(E + 16))
        if tidx < E:
            sHc[tidx] = Int32(0)
        cute.arch.sync_threads()
        P_rows = cute.size(ids)
        for p in cutlass.range(tidx, P_rows, self.threads, unroll=4):
            cute.arch.atomic_add(sHc.iterator + ids[p], Int32(1), scope="cta")
        cute.arch.sync_threads()
        if warp_idx == 0:
            # exclusive prefix of per-expert token-tile counts; lane l owns experts 8l .. 8l+7
            tv = cute.make_rmem_tensor((8,), Int32)
            run = Int32(0)
            for q in cutlass.range_constexpr(8):
                c = sHc[lane * 8 + q]
                tv[q] = (c + (BN - 1)) // BN
                run = run + tv[q]
            inc = run
            for i in cutlass.range_constexpr(5):
                o = cute.arch.shuffle_sync_up(inc, 1 << i)
                if lane >= (1 << i):
                    inc = inc + o
            ex = inc - run
            for q in cutlass.range_constexpr(8):
                sToff[lane * 8 + q] = ex
                ex = ex + tv[q]
            if lane == 31:
                sToff[E] = inc
        cute.arch.sync_threads()
        sA = storage.sA.get_tensor(a_smem_staged.outer, swizzle=a_smem_staged.inner)
        sB = storage.sB.get_tensor(b_smem_staged.outer, swizzle=b_smem_staged.inner)
        sItem = storage.sItem.get_tensor(cute.make_layout(self.qstages))
        sSig = storage.sSig.get_tensor(cute.make_layout(4))
        sScl = storage.sScl.get_tensor(cute.make_layout(BN))
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
        sH = storage.sH.get_tensor(cute.make_layout((BN, 128), stride=(128, 1)))

        T = sToff[E]
        n1 = T * self.mt1
        total = n1 + T * self.ng2

        if warp_idx >= self.prod_warp:
            cute.arch.setmaxregister_decrease(self.prod_regs)
        if warp_idx == self.sched_warp:
            # ---------------- scheduler: claim + prepare items ahead of the TMA warp ----------------
            cute.arch.griddepcontrol_wait()
            qstate = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.qstages
            )
            # first item is static (item = CTA index); later items come from the global counter + grid size
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
                sDesc,
                sW,
                sSc,
                sTok,
                sWt,
                sched,
                qstate,
                lane,
                n1,
                total,
                P_rows,
                bidx,
                gdim,
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
                    sDesc,
                    sW,
                    sSc,
                    sTok,
                    sWt,
                    sched,
                    qstate,
                    lane,
                    n1,
                    total,
                    P_rows,
                    Int32(-1),
                    gdim,
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
            # stream the weight boxes of the static first item (a GEMM1 item) before waiting on the
            # preceding kernel; its token boxes follow once the quantized activations exist
            npre = Int32(0)
            if bidx < n1:
                j0 = bidx // self.mt1
                mt0 = bidx - j0 * self.mt1
                # expert of token tile j0: the largest e with sToff[e] <= j0
                e0 = Int32(0)
                for i in cutlass.range_constexpr(8):
                    step = 128 >> i
                    if sToff[e0 + step] <= j0:
                        e0 = e0 + step
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
                    )
                    pstate.advance()
            cute.arch.griddepcontrol_wait()
            sched.consumer_wait(qs)
            idx = sItem[qs.index]
            while idx >= 0:
                slot = qs.index
                abase = sDesc[0, slot]
                row0 = sDesc[1, slot]
                is1 = sDesc[2, slot] != 0
                nkb = sDesc[3, slot]
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
                    # static first item: weight boxes of stages 0 .. npre-1 are already in flight
                    kstart = npre
                    bstate = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Producer, self.stages
                    )
                    for kb in cutlass.range(npre, unroll=1):
                        bar = mainloop.producer_get_barrier(bstate)
                        cute.copy(
                            tma_b1,
                            tB1g[(None, kb)],
                            tB1s[(None, bstate.index)],
                            tma_bar_ptr=bar,
                        )
                        bstate.advance()
                for kb in cutlass.range(kstart, nkb, unroll=1):
                    mainloop.producer_acquire(pstate)
                    bar = mainloop.producer_get_barrier(pstate)
                    if is1:
                        cute.copy(
                            tma_a1,
                            tA1g[(None, abase + kb)],
                            tA1s[(None, pstate.index)],
                            tma_bar_ptr=bar,
                        )
                        cute.copy(
                            tma_b1,
                            tB1g[(None, kb)],
                            tB1s[(None, pstate.index)],
                            tma_bar_ptr=bar,
                        )
                    else:
                        cute.copy(
                            tma_a2,
                            tA2g[(None, abase + kb)],
                            tA2s[(None, pstate.index)],
                            tma_bar_ptr=bar,
                        )
                        cute.copy(
                            tma_b2,
                            tB2g[(None, kb & 3)],
                            tB2s[(None, pstate.index)],
                            tma_bar_ptr=bar,
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
                    self._promote(acc, tmp, sx, sSc, sW, tCcC, kb, is1, slot)
                    mainloop.consumer_wait(cstate)
                    self._issue(tiled_mma, tmp, tCrA, tCrB0, cstate.index)
                    cstate.advance()
                    cute.nvgpu.warpgroup.wait_group(1)
                    mainloop.consumer_release(rstate)
                    rstate.advance()
                    self._promote(acc, tmp2, sx, sSc, sW, tCcC, kb + 1, is1, slot)
                    if not is1:
                        self._epi2(
                            acc,
                            sWt,
                            sTok,
                            accbuf,
                            tCcC,
                            rbase,
                            kb + 1,
                            mt,
                            nvalid,
                            slot,
                        )
                kbl = nkb - 2
                mainloop.consumer_wait(cstate)
                self._issue(tiled_mma, tmp2, tCrA, tCrB0, cstate.index)
                cstate.advance()
                cute.nvgpu.warpgroup.wait_group(1)
                mainloop.consumer_release(rstate)
                rstate.advance()
                self._promote(acc, tmp, sx, sSc, sW, tCcC, kbl, is1, slot)
                cute.nvgpu.warpgroup.wait_group(0)
                mainloop.consumer_release(rstate)
                rstate.advance()
                self._promote(acc, tmp2, sx, sSc, sW, tCcC, kbl + 1, is1, slot)
                if not is1:
                    self._epi2(
                        acc, sWt, sTok, accbuf, tCcC, rbase, kbl + 1, mt, nvalid, slot
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
                    # one warp finishes the per-token amax over the 8 warps and stores scale / 1/scale
                    if cwarp == 0:
                        for c0 in cutlass.range_constexpr((BN + 31) // 32):
                            col = lane + 32 * c0
                            if col < BN:
                                am = sRed[0, col]
                                for w in cutlass.range_constexpr(1, 8):
                                    am = cute.arch.fmax(am, sRed[w, col])
                                am = cute.arch.fmax(am, Float32(QUANT_EPS))
                                scale = am / Float32(FP8_MAX)
                                sScl[col] = scale
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
            # ---- grid-wide barrier (the persistent grid is co-resident), then fp32 -> bf16 output ----
            cons_bar.arrive_and_wait()
            if ctid == 0:
                cute.arch.fence_acq_rel_gpu()
                cute.arch.atomic_add(
                    meta.iterator + 2, Int32(1), sem="release", scope="gpu"
                )
                arrived = cute.arch.atomic_add(
                    meta.iterator + 2, Int32(0), sem="acquire", scope="gpu"
                )
                while arrived < gdim:
                    arrived = cute.arch.atomic_add(
                        meta.iterator + 2, Int32(0), sem="acquire", scope="gpu"
                    )
            cons_bar.arrive_and_wait()
            f8v = cute.make_rmem_tensor((8,), Float32)
            o8v = cute.make_rmem_tensor((8,), cutlass.BFloat16)
            for m in cutlass.range(bidx, cute.size(accbuf, mode=[0]), gdim, unroll=1):
                cute.autovec_copy(cute.local_tile(accbuf[m, None], (8,), (ctid,)), f8v)
                o8v.store(f8v.load().to(cutlass.BFloat16))
                cute.autovec_copy(o8v, cute.local_tile(out[m, None], (8,), (ctid,)))

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
    def _epi2(self, acc, sWt, sTok, accbuf, tCcC, rbase, kb, mt, nvalid, slot):
        """After every 4 k-blocks of a GEMM2 item: router weight + fp32 top-k reduction of one
        256-row output tile, then restart the accumulator."""
        NJS = self.BNS // 8
        if (kb & 3) == 3:
            rb2 = rbase + (mt + (kb >> 2)) * self.BM
            for jj in cutlass.range_constexpr(NJS):
                for cc in cutlass.range_constexpr(2):
                    col = tCcC[cc + 4 * jj][1]
                    if col < nvalid:
                        wt = sWt[col, slot]
                        base = sTok[col, slot] + rb2
                        for mi in cutlass.range_constexpr(2):
                            for hh in cutlass.range_constexpr(2):
                                ii = cc + 2 * hh + 4 * jj + 4 * NJS * mi
                                # fire-and-forget fp32 reduction (no returned value)
                                cute.arch.inline_ptx(
                                    "red.global.add.f32 [{$r0}], {$r1};",
                                    read_only_args=[
                                        (
                                            accbuf.iterator
                                            + (base + (mi * 128 + hh * 8))
                                        ).toint(),
                                        acc[ii] * wt,
                                    ],
                                )
            acc.fill(0.0)

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
        sDesc,
        sW,
        sSc,
        sTok,
        sWt,
        sched,
        qstate,
        lane,
        n1,
        total,
        P_rows,
        static_idx,
        gdim,
    ):
        """Producer warp: claim the next work item, wait for its GEMM1 dependency (GEMM2 items), and
        publish it with its weight block scales and activation scales into the item-queue slot."""
        BN = self.BN
        idx = static_idx
        if static_idx < 0:
            if lane == 0:
                idx = (
                    cute.arch.atomic_add(meta.iterator + 1, Int32(1), scope="gpu")
                    + gdim
                )
            idx = cute.arch.shuffle_sync(idx, 0)
        sched.producer_acquire(qstate)
        slot = qstate.index
        if idx < total:
            is1 = idx < n1
            i2 = idx - n1
            j = idx // self.mt1
            mt = idx % self.mt1
            if not is1:
                if const_expr(self.grp_major):
                    # large M: one 256*G2-column slice of the fp32 accumulator per pass, so the
                    # scatter-add working set stays L2 resident (each w2 tile is still read once)
                    T = n1 // self.mt1
                    gq = i2 // T
                    j = i2 - gq * T
                    mt = gq * self.G2
                else:
                    j = i2 // self.ng2
                    mt = (i2 % self.ng2) * self.G2
            e = work[j, 0]
            row0 = work[j, 1]
            nvalid = work[j, 2]
            if lane == 0:
                sDesc[0, slot] = (
                    (e * self.mt1 + mt) * self.nkb1
                    if is1
                    else (e * self.mt2 + mt) * self.nkb2
                )
                sDesc[1, slot] = row0
                sDesc[2, slot] = Int32(1) if is1 else Int32(0)
                sDesc[3, slot] = Int32(self.nkb1) if is1 else Int32(self.nkb2 * self.G2)
                sDesc[4, slot] = nvalid
                sDesc[5, slot] = j
                sDesc[6, slot] = mt
            if is1:
                for r in cutlass.range(BN // 16, unroll=1):
                    for r2 in cutlass.range_constexpr(8):
                        t = lane + 32 * (r * 8 + r2)
                        kb = t // BN
                        c = t % BN
                        v = mS1[cutlass.min(row0 + c, P_rows - 1), kb]
                        sSc[c, kb, slot] = v if c < nvalid else Float32(0.0)
                for r in cutlass.range_constexpr(2):
                    t = lane + 32 * r
                    sW[t, slot] = w13s[e, mt + (t % 2) * 4, t // 4]
            else:
                # GEMM2: wait until the 4 GEMM1 tiles of this token tile have stored h and its scales
                if lane == 0:
                    cnt = cute.arch.atomic_add(
                        dep.iterator + j, Int32(0), sem="acquire", scope="gpu"
                    )
                    while cnt < self.mt1:
                        cnt = cute.arch.atomic_add(
                            dep.iterator + j, Int32(0), sem="acquire", scope="gpu"
                        )
                cute.arch.sync_warp()
                cute.arch.fence_proxy("async.global")
                for r in cutlass.range_constexpr((BN + 7) // 8):
                    t = lane + 32 * r
                    if t < BN * 4:
                        kb = t // BN
                        c = t % BN
                        v = mS2[cutlass.min(row0 + c, P_rows - 1), kb]
                        sSc[c, kb, slot] = v if c < nvalid else Float32(0.0)
                for r in cutlass.range_constexpr((16 * self.G2 + 31) // 32):
                    t = lane + 32 * r
                    if t < 16 * self.G2:
                        # t = (mts * 4 + kb) * 4 + mi * 2 + hh
                        sW[t, slot] = w2s[
                            e, (mt + t // 16) * 2 + (t // 2) % 2, (t // 4) % 4
                        ]
                # token row offsets and router weights of this tile's columns
                for r in cutlass.range_constexpr((BN + 31) // 32):
                    c = lane + 32 * r
                    if c < BN:
                        rr = cutlass.min(row0 + c, P_rows - 1)
                        tv = sorted_pair[rr]
                        wv = topk_w[rr]
                        sTok[c, slot] = tv
                        sWt[c, slot] = wv if c < nvalid else Float32(0.0)
        if lane == 0:
            sItem[slot] = idx if idx < total else Int32(-1)
        sched.producer_commit(qstate)
        return idx


# --------------------------------------------------------------------------------------------
# large-M grouped block-FP8 GEMM (non-swapped: tokens = WGMMA M)
# --------------------------------------------------------------------------------------------
@dsl_user_op
def _cvt_e4m3x2(hi, lo, *, loc=None, ip=None):
    """Two fp32 -> packed e4m3 pair (RN, satfinite); `lo` lands in the low byte."""
    r = llvm.inline_asm(
        T.i16(),
        [Float32(hi).ir_value(loc=loc, ip=ip), Float32(lo).ir_value(loc=loc, ip=ip)],
        "cvt.rn.satfinite.e4m3x2.f32 $0, $1, $2;",
        "=h,f,f",
        has_side_effects=False,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )
    return Int16(r)


@dsl_user_op
def _ldsm_x4(ptr, *, loc=None, ip=None):
    """ldmatrix.x4 (b16): four 8x8 b16 matrices from 8-row x 16-byte shared segments (one row address per lane)."""
    addr = llvm.ptrtoint(T.i32(), ptr.to_llvm_ptr(loc=loc, ip=ip), loc=loc, ip=ip)
    st = llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()])
    r = llvm.inline_asm(
        st,
        [addr],
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Int32(llvm.extractvalue(T.i32(), r, [i], loc=loc, ip=ip)) for i in range(4)
    )


@dsl_user_op
def _red_add_v2_f32(ptr, a, b, *, loc=None, ip=None):
    """Fire-and-forget 8-byte fp32 reduction at an 8-byte aligned global address."""
    llvm.inline_asm(
        None,
        [
            ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
        ],
        "red.global.v2.f32.add [$0], {$1, $2};",
        "l,f,f",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _cp_async_cg16(dst, src, *, loc=None, ip=None):
    """16-byte cp.async.cg global -> shared. cutlass-dsl 4.8 dropped
    cute.arch.cp_async_shared_global, so emit the NVVM op directly."""
    from cutlass._mlir.dialects import nvvm as _nvvm
    from cutlass.cute.arch.nvvm_wrappers import _normalize_ptr

    _nvvm.cp_async_shared_global(
        _normalize_ptr(dst, loc=loc, ip=ip),
        _normalize_ptr(src, loc=loc, ip=ip),
        16,
        _nvvm.LoadCacheModifierKind.CG,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _red_add_v4_f32_keep(ptr, a, b, c, d, *, loc=None, ip=None):
    """16-byte fp32 global reduction with an L2 evict-last hint: the fp32 accumulator slice being reduced
    into stays resident while the h and weight tiles stream past it."""
    llvm.inline_asm(
        None,
        [
            ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
            Float32(c).ir_value(loc=loc, ip=ip),
            Float32(d).ir_value(loc=loc, ip=ip),
            cutlass.Int64(0x14F0000000000000).ir_value(loc=loc, ip=ip),
        ],
        "red.global.add.L2::cache_hint.v4.f32 [$0], {$1, $2, $3, $4}, $5;",
        "l,f,f,f,f,l",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _red_add_v4_f32(ptr, a, b, c, d, *, loc=None, ip=None):
    """Fire-and-forget 16-byte fp32 reduction at a 16-byte aligned global address."""
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
def _tanh_approx(x, *, loc=None, ip=None):
    r = llvm.inline_asm(
        T.f32(),
        [Float32(x).ir_value(loc=loc, ip=ip)],
        "tanh.approx.f32 $0, $1;",
        "=f,f",
        has_side_effects=False,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )
    return Float32(r)


@dsl_user_op
def _nanosleep(ns, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int32(ns).ir_value(loc=loc, ip=ip)],
        "nanosleep.u32 $0;",
        "r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


class MoEPrefill:
    """Persistent kernel running every GEMM1 and GEMM2 tile for large batches (non-swapped).

    CTA tile = 128 sorted token rows (2 consumer warpgroups x m64) x 256 weight rows, run as two
    WGMMA n128 halves per 128-K block through one shared 64-register partial (a runtime-count loop over
    the halves keeps ptxas from duplicating it), so a thread holds 2 x 64 fp32 accumulators. Weights are
    read by TMA (128x128 boxes) straight from the original w13 / w2 layouts; activation scales are
    staged in shared memory by the scheduler.
      GEMM1 item (token tile j, h group mt): halves = 128 gate rows and the matching 128 up rows, so
      gate and up of an h column land in the same thread; the item owns the whole 1x128 h quant group
      (per-row amax via quad shuffles).
      GEMM2 item (token tile j, 2 x 256 output channels): K = 512 per 256-channel sub-tile, router
      weight, fp32 red.add after each sub-tile.
    Items are claimed in order from a global counter (all GEMM1 items first; GEMM2 items
    output-slice-major so the fp32 accumulator slice being reduced into stays L2 resident); GEMM2
    items wait on a per-token-tile counter that the 4 GEMM1 items of the tile bump after storing h.

    Warps 0-7: consumers. Warp 8: TMA producer. Warp 9: scheduler. Warp 10: dependency release.
    """

    def __init__(self, stages: int, g2: int, producer_heavy: bool):
        self.BT = 128
        self.BW = 256
        self.BK = 128
        self.stages = stages
        # A third scheduler slot amortizes the higher work-item rate of the narrow g2=1 path;
        # the wider g2=2 path has fewer items and keeps the smaller two-slot queue.
        self.qstages = 3 if g2 == 1 else 2
        self.num_cons_threads = 256
        self.threads = 384
        # Once a 256-channel fp32 output slice exceeds 12 MiB, the scheduler/gather producer
        # needs more registers than the consumers need for this g2=1 path.
        self.cons_regs = 224 if producer_heavy else 232
        self.prod_regs = 56 if producer_heavy else 40
        self.prod_warp = 8
        self.sched_warp = 9
        self.sig_warp = 10
        self.fin_warp = 11  # second GEMM1 row-gather warp
        self.nkb1 = H // 128
        self.nkb2 = I // 128
        self.mt1 = (2 * I) // 256  # 128-column h groups
        self.g2 = g2  # 256-channel output sub-tiles per item; selected from the accumulator slice footprint
        self.RP = 72  # padded row pitch (floats) of the per-warp 16 x 64 GEMM2 output staging buffer
        self.HP = 144  # padded row pitch (bytes) of the 128 x 128 fp8 h staging tile
        self.mt2 = H // (256 * self.g2)  # GEMM2 items per token tile

    @cute.jit
    def __call__(
        self,
        xq: cute.Tensor,  # (M, H) fp8 activations in token order (gathered per expert tile)
        xs: cute.Tensor,  # (M, H/128) fp32 activation scales in token order
        w13: cute.Tensor,  # setup-packed [expert][h tile][kblock][gate/up 256][128]
        w13s: cute.Tensor,  # (E, 8, 16) fp32
        hq: cute.Tensor,  # (P, I) fp8 sorted intermediate
        hs: cute.Tensor,  # (PS, I/128) fp32 view, stride (1, PS)
        w2: cute.Tensor,  # setup-packed [expert][256-row output tile][kblock][256][128]
        w2s: cute.Tensor,  # (E, 16, 4) fp32
        work: cute.Tensor,
        meta: cute.Tensor,
        dep: cute.Tensor,
        accbuf: cute.Tensor,
        sorted_pair: cute.Tensor,
        topk_w: cute.Tensor,
        out: cute.Tensor,  # (M, H) bf16
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
            tiler_mn=(64, 128),
        )
        self.tiled_mma_z = sm90_utils.make_trivial_tiled_mma(
            fp8,
            fp8,
            cute.nvgpu.warpgroup.OperandMajorMode.K,
            cute.nvgpu.warpgroup.OperandMajorMode.K,
            Float32,
            (2, 1, 1),
            tiler_mn=(64, 128),
        )
        # n64 quarter MMAs: two 32-register partials alternate so one quarter's WGMMA chain runs while
        # the previous quarter is promoted
        # A (token rows) comes from registers: loaded once per 128-K block with ldmatrix and reused by
        # all four quarters, so only the weight operand is re-read from shared memory per quarter
        self.tiled_mma_q = sm90_utils.make_trivial_tiled_mma(
            fp8,
            fp8,
            cute.nvgpu.warpgroup.OperandMajorMode.K,
            cute.nvgpu.warpgroup.OperandMajorMode.K,
            Float32,
            (2, 1, 1),
            tiler_mn=(64, 64),
            a_source=cute.nvgpu.warpgroup.OperandSource.RMEM,
        )
        self.tiled_mma_qz = sm90_utils.make_trivial_tiled_mma(
            fp8,
            fp8,
            cute.nvgpu.warpgroup.OperandMajorMode.K,
            cute.nvgpu.warpgroup.OperandMajorMode.K,
            Float32,
            (2, 1, 1),
            tiler_mn=(64, 64),
            a_source=cute.nvgpu.warpgroup.OperandSource.RMEM,
        )
        ab_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(utils.LayoutEnum.ROW_MAJOR, fp8, self.BK),
            fp8,
        )
        self.a_smem_staged = cute.tile_to_shape(
            ab_atom, (self.BT, self.BK, self.stages), order=(0, 1, 2)
        )
        # B stage = two 128-row halves (index stage * 2 + half)
        self.bh_smem_staged = cute.tile_to_shape(
            ab_atom, (self.BW // 2, self.BK, 2 * self.stages), order=(0, 1, 2)
        )
        # The same bytes viewed as one 256-row stage for a single packed-weight TMA.
        self.b256_smem_staged = cute.tile_to_shape(
            ab_atom, (self.BW, self.BK, self.stages), order=(0, 1, 2)
        )
        a_one = cute.slice_(self.a_smem_staged, (None, None, 0))
        # The sparse-tail GEMM2 path deposits only the first WGMMA warpgroup's 64 h rows into
        # the existing 128-row stage.  Derive the sublayout from the full stage so its swizzle and
        # row offsets are identical to the layout consumed by WG0.
        a64_one = cute.tile_to_shape(ab_atom, (64, self.BK), order=(0, 1))
        bh_one = cute.slice_(self.bh_smem_staged, (None, None, 0))
        b256_one = cute.slice_(self.b256_smem_staged, (None, None, 0))
        g2s = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        tma_a2, tA2 = cute.nvgpu.cpasync.make_tiled_tma_atom(
            g2s, hq, a_one, (self.BT, self.BK)
        )
        tma_a64, tA64 = cute.nvgpu.cpasync.make_tiled_tma_atom(
            g2s, hq, a64_one, (64, self.BK)
        )
        tma_b1, tB1 = cute.nvgpu.cpasync.make_tiled_tma_atom(
            g2s, w13, b256_one, (self.BW, self.BK)
        )
        tma_b2, tB2 = cute.nvgpu.cpasync.make_tiled_tma_atom(
            g2s, w2, b256_one, (self.BW, self.BK)
        )
        # per stage: the weight boxes always arrive by TMA; GEMM2 adds its h box (expect_tx at issue time),
        # GEMM1 gathers its token rows with cp.async (32 producer-lane arrivals)
        self.tx_bytes = self.BW * self.BK

        @cute.struct
        class SharedStorage:
            mbar: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
            qbar: cute.struct.MemRange[cutlass.Int64, self.qstages * 2]
            sgbar: cute.struct.MemRange[cutlass.Int64, 4 * 2]
            sSig: cute.struct.Align[cute.struct.MemRange[Int32, 4], 16]
            sItem: cute.struct.Align[cute.struct.MemRange[Int32, self.qstages], 16]
            sDesc: cute.struct.Align[cute.struct.MemRange[Int32, 8 * self.qstages], 16]
            sW: cute.struct.Align[cute.struct.MemRange[Float32, 32 * self.qstages], 16]
            sTok: cute.struct.Align[
                cute.struct.MemRange[Int32, self.BT * self.qstages], 16
            ]
            sOff: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int64, self.BT * self.qstages], 16
            ]
            sWt: cute.struct.Align[
                cute.struct.MemRange[Float32, self.BT * self.qstages], 16
            ]
            sSa: cute.struct.Align[
                cute.struct.MemRange[Float32, self.BT * 16 * self.qstages], 16
            ]
            # GEMM1 h staging and GEMM2 fp32 output staging share one buffer (never live at the same time:
            # a warp reaches a GEMM2 epilogue only after every warp released the stages of the item before)
            sHR: cute.struct.Align[
                cute.struct.MemRange[
                    Float32, max(self.BT * self.HP // 4, 8 * 16 * self.RP)
                ],
                128,
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[fp8, cute.cosize(self.a_smem_staged)], 1024
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[fp8, cute.cosize(self.bh_smem_staged)], 1024
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            xq,
            tma_a2,
            tA2,
            tma_a64,
            tA64,
            tma_b1,
            tB1,
            tma_b2,
            tB2,
            xs,
            hs,
            w13s,
            w2s,
            work,
            meta,
            dep,
            hq,
            accbuf,
            sorted_pair,
            topk_w,
            out,
            self.tiled_mma,
            self.tiled_mma_z,
            self.tiled_mma_q,
            self.tiled_mma_qz,
            self.a_smem_staged,
            self.bh_smem_staged,
            self.b256_smem_staged,
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
        xqc: cute.Tensor,
        tma_a2: cute.CopyAtom,
        mA2: cute.Tensor,
        tma_a64: cute.CopyAtom,
        mA64: cute.Tensor,
        tma_b1: cute.CopyAtom,
        mB1: cute.Tensor,
        tma_b2: cute.CopyAtom,
        mB2: cute.Tensor,
        xs: cute.Tensor,
        hsg: cute.Tensor,
        w13s: cute.Tensor,
        w2s: cute.Tensor,
        work: cute.Tensor,
        meta: cute.Tensor,
        dep: cute.Tensor,
        hq: cute.Tensor,
        accbuf: cute.Tensor,
        sorted_pair: cute.Tensor,
        topk_w: cute.Tensor,
        out: cute.Tensor,
        tiled_mma: cute.TiledMma,
        tiled_mma_z: cute.TiledMma,
        tiled_mma_q: cute.TiledMma,
        tiled_mma_qz: cute.TiledMma,
        a_smem_staged: cute.ComposedLayout,
        bh_smem_staged: cute.ComposedLayout,
        b256_smem_staged: cute.ComposedLayout,
    ):
        BT = self.BT
        BW = self.BW
        BK = self.BK
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()

        if warp_idx == self.prod_warp:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_a2)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_a64)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_b1)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_b2)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        mainloop = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar.data_ptr(),
            num_stages=self.stages,
            # one elected arrive_expect_tx + one arrive per lane of the two gather warps
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 65),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 8),
            tx_count=self.tx_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        sched = pipeline.PipelineAsync.create(
            barrier_storage=storage.qbar.data_ptr(),
            num_stages=self.qstages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_cons_threads + 64
            ),
        )
        # everything below reads routing / quantized activations from the preceding kernels
        cute.arch.griddepcontrol_launch_dependents()
        cute.arch.griddepcontrol_wait()
        sA = storage.sA.get_tensor(a_smem_staged.outer, swizzle=a_smem_staged.inner)
        # Alias the first 64 rows of each full stage without changing the stage stride.
        sA64 = cute.local_tile(sA, (64, BK), (0, 0, None))
        sBh = storage.sB.get_tensor(bh_smem_staged.outer, swizzle=bh_smem_staged.inner)
        sB256 = storage.sB.get_tensor(
            b256_smem_staged.outer, swizzle=b256_smem_staged.inner
        )
        sItem = storage.sItem.get_tensor(cute.make_layout(self.qstages))
        sSig = storage.sSig.get_tensor(cute.make_layout(4))
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
        sW = storage.sW.get_tensor(cute.make_layout((32, self.qstages), stride=(1, 32)))
        sTok = storage.sTok.get_tensor(
            cute.make_layout((BT, self.qstages), stride=(1, BT))
        )
        # GEMM1 gather offsets (16-byte units) of row 4 * i + q at (i, q): each gather lane reads 16 contiguous
        sOff = storage.sOff.get_tensor(
            cute.make_layout((BT // 4, 4, self.qstages), stride=(1, BT // 4, BT))
        )
        sWt = storage.sWt.get_tensor(
            cute.make_layout((BT, self.qstages), stride=(1, BT))
        )
        # per-item activation scales [row, k-block]: GEMM1 x scales (16 blocks), GEMM2 h scales (4)
        sSa = storage.sSa.get_tensor(
            cute.make_layout((BT, 16, self.qstages), stride=(1, BT, 16 * BT))
        )

        T = meta[0]
        n1 = T * self.mt1
        total = n1 + T * self.mt2
        P_rows = cute.size(sorted_pair)
        # dep[cnt0]: number of CTAs done zeroing their rows of the fp32 accumulator
        cnt0 = cute.size(dep) - 16
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()

        if warp_idx >= self.prod_warp:
            cute.arch.setmaxregister_decrease(self.prod_regs)
        if warp_idx == self.sched_warp:
            # ---------------- scheduler: claim + prepare items ahead of the TMA warp ----------------
            qstate = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.qstages
            )
            idx = self._claim(
                meta,
                dep,
                work,
                w13s,
                w2s,
                xs,
                hsg,
                sorted_pair,
                topk_w,
                sItem,
                sDesc,
                sW,
                sTok,
                sWt,
                sSa,
                sched,
                qstate,
                lane,
                n1,
                total,
                T,
                P_rows,
                cnt0,
                gdim,
                sOff,
                bidx,
            )
            qstate.advance()
            while idx < total:
                idx = self._claim(
                    meta,
                    dep,
                    work,
                    w13s,
                    w2s,
                    xs,
                    hsg,
                    sorted_pair,
                    topk_w,
                    sItem,
                    sDesc,
                    sW,
                    sTok,
                    sWt,
                    sSa,
                    sched,
                    qstate,
                    lane,
                    n1,
                    total,
                    T,
                    P_rows,
                    cnt0,
                    gdim,
                    sOff,
                    Int32(-1),
                )
                qstate.advance()
        elif warp_idx == self.sig_warp:
            # ---------------- fp32 accumulator zeroing, then the dependency signaller ----------------
            Mt = cute.size(accbuf, mode=[0])
            z = cute.make_rmem_tensor((4,), Float32)
            z.fill(0.0)
            for row in cutlass.range(bidx, Mt, gdim, unroll=1):
                for c in cutlass.range_constexpr(H // 128):
                    cute.autovec_copy(
                        z, cute.local_tile(accbuf[row, None], (4,), (c * 32 + lane,))
                    )
            cute.arch.fence_acq_rel_gpu()
            cute.arch.sync_warp()
            if lane == 0:
                cute.arch.atomic_add(
                    dep.iterator + cnt0, Int32(1), sem="release", scope="gpu"
                )
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
        elif warp_idx == self.fin_warp:
            # ---------------- second GEMM1 row-gather warp (same stage sequence as the TMA producer) ----------------
            sA_ptr = storage.sA.data_ptr()
            c16 = lane % 8
            gq = lane // 8
            x0 = (c16 ^ gq) * 16
            x1 = (c16 ^ (gq + 4)) * 16
            offs = cute.make_rmem_tensor((BT // 8,), cutlass.Int64)
            gst = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.stages
            )
            qs = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.qstages
            )
            sched.consumer_wait(qs)
            idx = sItem[qs.index]
            while idx >= 0:
                slot = qs.index
                is1 = sDesc[2, slot] != 0
                if is1:
                    cute.autovec_copy(
                        cute.local_tile(sOff[None, gq, slot], (BT // 8,), (1,)), offs
                    )
                    for kb in cutlass.range(self.nkb1, unroll=1):
                        mainloop.sync_object_empty.wait(gst.index, gst.phase)
                        bar = mainloop.producer_get_barrier(gst)
                        self._gather(
                            sA_ptr, xqc, offs, gst.index, kb, gq, c16, x0, x1, 1
                        )
                        cute.arch.cp_async_mbarrier_arrive_noinc(bar)
                        gst.advance()
                else:
                    for kb in cutlass.range(self.g2 * self.nkb2, unroll=1):
                        mainloop.sync_object_empty.wait(gst.index, gst.phase)
                        cute.arch.mbarrier_arrive(mainloop.producer_get_barrier(gst))
                        gst.advance()
                sched.consumer_release(qs)
                qs.advance()
                sched.consumer_wait(qs)
                idx = sItem[qs.index]
        elif warp_idx == self.prod_warp:
            # ---------------- TMA producer ----------------
            hpol = {}
            if const_expr(self.g2 == 1):
                hpol = {"cache_policy": cutlass.Int64(0x12F0000000000000)}
            pstate = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.stages
            )
            qs = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.qstages
            )
            sA_ptr = storage.sA.data_ptr()
            c16 = (
                lane % 8
            )  # 16-byte chunk of a 128-byte row segment handled by this lane
            gq = lane // 8
            x0 = (c16 ^ gq) * 16
            x1 = (c16 ^ (gq + 4)) * 16
            offs = cute.make_rmem_tensor((BT // 8,), cutlass.Int64)
            gB1 = cute.local_tile(mB1, (BW, BK), (None, 0))
            gB2 = cute.local_tile(mB2, (BW, BK), (None, 0))
            tB1s, tB1g = cute.nvgpu.cpasync.tma_partition(
                tma_b1,
                0,
                cute.make_layout(1),
                cute.group_modes(sB256, 0, 2),
                cute.group_modes(gB1, 0, 2),
            )
            tB2s, tB2g = cute.nvgpu.cpasync.tma_partition(
                tma_b2,
                0,
                cute.make_layout(1),
                cute.group_modes(sB256, 0, 2),
                cute.group_modes(gB2, 0, 2),
            )
            sched.consumer_wait(qs)
            idx = sItem[qs.index]
            while idx >= 0:
                slot = qs.index
                e = sDesc[0, slot]
                row0 = sDesc[1, slot]
                is1 = sDesc[2, slot] != 0
                nvalid = sDesc[4, slot]
                mt = sDesc[6, slot]
                if not is1:
                    cute.arch.fence_proxy("async.global")
                r2 = Int32(0) if is1 else row0
                gA2 = cute.local_tile(
                    cute.domain_offset((r2, 0), mA2), (BT, BK), (0, None)
                )
                gA64 = cute.local_tile(
                    cute.domain_offset((r2, 0), mA64), (64, BK), (0, None)
                )
                tA2s, tA2g = cute.nvgpu.cpasync.tma_partition(
                    tma_a2,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sA, 0, 2),
                    cute.group_modes(gA2, 0, 2),
                )
                tA64s, tA64g = cute.nvgpu.cpasync.tma_partition(
                    tma_a64,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sA64, 0, 2),
                    cute.group_modes(gA64, 0, 2),
                )
                if is1:
                    bpanel = (e * self.mt1 + mt) * self.nkb1
                    cute.autovec_copy(
                        cute.local_tile(sOff[None, gq, slot], (BT // 8,), (0,)), offs
                    )
                    for kb in cutlass.range(self.nkb1, unroll=1):
                        mainloop.producer_acquire(pstate)
                        bar = mainloop.producer_get_barrier(pstate)
                        cute.copy(
                            tma_b1,
                            tB1g[(None, bpanel + kb)],
                            tB1s[(None, pstate.index)],
                            tma_bar_ptr=bar,
                        )
                        self._gather(
                            sA_ptr, xqc, offs, pstate.index, kb, gq, c16, x0, x1, 0
                        )
                        cute.arch.cp_async_mbarrier_arrive_noinc(bar)
                        pstate.advance()
                else:
                    for sl in cutlass.range(self.g2, unroll=1):
                        bpanel = (e * (H // self.BW) + mt * self.g2 + sl) * self.nkb2
                        for kb in cutlass.range(self.nkb2, unroll=1):
                            mainloop.producer_acquire(pstate)
                            bar = mainloop.producer_get_barrier(pstate)
                            # nvalid is a device-produced expert-tile count.  For a sparse tail WG1
                            # is already compute-elided, so only WG0's 64 h rows need to cross L2.
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_expect_tx(
                                    bar, (64 if nvalid <= Int32(64) else BT) * BK
                                )
                            cute.arch.sync_warp()
                            if nvalid <= Int32(64):
                                cute.copy(
                                    tma_a64,
                                    tA64g[(None, kb)],
                                    tA64s[(None, pstate.index)],
                                    tma_bar_ptr=bar,
                                    **hpol,
                                )
                            else:
                                cute.copy(
                                    tma_a2,
                                    tA2g[(None, kb)],
                                    tA2s[(None, pstate.index)],
                                    tma_bar_ptr=bar,
                                    **hpol,
                                )
                            cute.copy(
                                tma_b2,
                                tB2g[(None, bpanel + kb)],
                                tB2s[(None, pstate.index)],
                                tma_bar_ptr=bar,
                            )
                            cute.arch.mbarrier_arrive(bar)
                            pstate.advance()
                sched.consumer_release(qs)
                qs.advance()
                sched.consumer_wait(qs)
                idx = sItem[qs.index]
            mainloop.producer_tail(pstate)
        elif warp_idx < self.prod_warp:
            # ---------------- consumers ----------------
            cute.arch.setmaxregister_increase(self.cons_regs)
            ctid = tidx
            thr_mma = tiled_mma.get_slice(ctid)
            tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sA))
            tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sBh))
            tCcC = thr_mma.partition_C(cute.make_identity_tensor((BT, BW // 2)))
            acc0 = cute.make_rmem_tensor(tCcC.shape, Float32)
            acc1 = cute.make_rmem_tensor(tCcC.shape, Float32)
            thr_q = tiled_mma_q.get_slice(ctid)
            tCrAq = tiled_mma_q.make_fragment_A(thr_q.partition_shape_A((BT, BK)))
            rA32 = cute.recast_tensor(tCrAq, Int32)
            # ldmatrix.x4 row address of this lane: matrix lane // 8 = (row +8, k +16) quadrant, row lane % 8;
            # 128B swizzle: 16-byte chunk c of row r sits at chunk c ^ (r % 8) (r % 8 == lane % 8)
            a_row = (
                (warp_idx // 4) * 64
                + (warp_idx % 4) * 16
                + lane % 8
                + 8 * ((lane // 8) % 2)
            )
            a_lo = lane // 16
            a_x = lane % 8
            sA_base = storage.sA.data_ptr()
            # rows [0, 64) / [64, 128) of every 128-row weight half-stage
            sBq0 = cute.local_tile(sBh, (BW // 4, BK), (0, 0, None))
            sBq1 = cute.local_tile(sBh, (BW // 4, BK), (1, 0, None))
            tCrBq0 = tiled_mma_q.make_fragment_B(thr_q.partition_B(sBq0))
            tCrBq1 = tiled_mma_q.make_fragment_B(thr_q.partition_B(sBq1))
            tqC = thr_q.partition_C(cute.make_identity_tensor((BT, BW // 4)))
            tA = cute.make_rmem_tensor(tqC.shape, Float32)
            tB = cute.make_rmem_tensor(tqC.shape, Float32)
            tiled_mma_q.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
            tiled_mma_qz.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
            NV = cute.size(tCcC)
            # accumulator register v = cc + 2*hh + 4*jj holds (token row r_hh, column 8*jj + 2*(lane%4) + cc)
            # of its 128-row half (GEMM1: acc0 = gate, acc1 = up of the same h column)
            rw0 = tCcC[0][0]
            rw1 = tCcC[2][0]
            # h tile staging (128 rows x 128 fp8) so it leaves the SM as full 128-byte lines
            # padded pitches: the 8 rows of a quad-strided store land in distinct banks
            sHR_ptr = storage.sHR.data_ptr()
            sHb = cute.make_tensor(
                cute.recast_ptr(sHR_ptr, dtype=cutlass.Float8E4M3FN),
                cute.make_layout((BT, 128), stride=(self.HP, 1)),
            )
            sH16 = cute.make_tensor(
                cute.recast_ptr(sHR_ptr, dtype=Int16),
                cute.make_layout((BT, 64), stride=(self.HP // 2, 1)),
            )
            # per-warp GEMM2 output staging: 16 rows x 64 columns (+ pad) per warp
            sR = cute.make_tensor(
                sHR_ptr,
                cute.make_layout((8, 16, self.RP), stride=(16 * self.RP, self.RP, 1)),
            )
            p2 = cute.make_rmem_tensor((2,), Float32)
            wrow0 = rw0 - lane // 4  # first tile row of this warp's 16-row slab
            rv = cute.make_rmem_tensor((4,), Float32)

            cstate = pipeline.make_pipeline_state(
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
                nvalid = sDesc[4, slot]
                j = sDesc[5, slot]
                mt = sDesc[6, slot]
                nhalf = sDesc[7, slot]
                # The two WGMMA warpgroups own token rows [0, 64) and [64, 128), respectively.
                # On a <=64-row tail, WG1 has no numerically useful work.  Keep the predicate
                # warpgroup-uniform: WGMMA issue may be skipped only by the whole warpgroup.
                active_consumer = (nvalid > Int32(64)) | (warp_idx < 4)

                if is1:
                    cstate = self._kloop(
                        mainloop,
                        cstate,
                        tiled_mma_q,
                        tiled_mma_qz,
                        acc0,
                        acc1,
                        tA,
                        tB,
                        tCrAq,
                        tCrBq0,
                        tCrBq1,
                        sSa,
                        sW,
                        slot,
                        rw0,
                        rw1,
                        Int32(0),
                        self.nkb1,
                        active_consumer,
                        rA32,
                        sA_base,
                        a_row,
                        a_lo,
                        a_x,
                    )
                    # ---- SiLU-GLU into acc0, then per-token 1x128 FP8 requant of h ----
                    if active_consumer:
                        am0 = Float32(0.0)
                        am1 = Float32(0.0)
                        for v in cutlass.range_constexpr(NV):
                            g = acc0[v]
                            # silu(g) = g * sigmoid(g) = g * (0.5 + 0.5 * tanh(g / 2)): one MUFU op per element
                            gh = g * Float32(0.5)
                            hv = (gh + gh * _tanh_approx(gh)) * acc1[v]
                            acc0[v] = hv
                            if (v // 2) % 2 == 0:
                                am0 = cute.arch.fmax(am0, cute.arch.fmax(hv, -hv))
                            else:
                                am1 = cute.arch.fmax(am1, cute.arch.fmax(hv, -hv))
                        am0 = cute.arch.fmax(am0, cute.arch.shuffle_sync_bfly(am0, 1))
                        am0 = cute.arch.fmax(am0, cute.arch.shuffle_sync_bfly(am0, 2))
                        am1 = cute.arch.fmax(am1, cute.arch.shuffle_sync_bfly(am1, 1))
                        am1 = cute.arch.fmax(am1, cute.arch.shuffle_sync_bfly(am1, 2))
                        am0 = cute.arch.fmax(am0, Float32(QUANT_EPS))
                        am1 = cute.arch.fmax(am1, Float32(QUANT_EPS))
                        inv0 = Float32(FP8_MAX) / am0
                        inv1 = Float32(FP8_MAX) / am1
                        if lane % 4 == 0:
                            if rw0 < nvalid:
                                hsg[row0 + rw0, mt] = am0 / Float32(FP8_MAX)
                            if rw1 < nvalid:
                                hsg[row0 + rw1, mt] = am1 / Float32(FP8_MAX)
                        # h -> smem (packed fp8 pairs), then full-line 16-byte stores of the valid rows
                        for hh in cutlass.range_constexpr(2):
                            r = rw0 if hh == 0 else rw1
                            inv = inv0 if hh == 0 else inv1
                            for jj in cutlass.range_constexpr(NV // 4):
                                v = 2 * hh + 4 * jj
                                col = tCcC[v][1]
                                sH16[r, col >> 1] = _cvt_e4m3x2(
                                    acc0[v + 1] * inv, acc0[v] * inv
                                )
                    # Both warpgroups must arrive even when WG1 skipped the padding-only epilogue.
                    cute.arch.barrier(
                        barrier_id=1, number_of_threads=self.num_cons_threads
                    )
                    if active_consumer:
                        if nvalid <= Int32(64):
                            # The original store is striped over all 256 consumers.  Redistribute
                            # the 64 valid rows over WG0's 128 threads when WG1 is inactive.
                            for it in cutlass.range_constexpr(4):
                                q = ctid + it * (self.num_cons_threads // 2)
                                r = q // 8
                                ch = q % 8
                                if r < nvalid:
                                    cute.autovec_copy(
                                        cute.local_tile(sHb[r, None], (16,), (ch,)),
                                        cute.local_tile(
                                            hq[row0 + r, None], (16,), (mt * 8 + ch,)
                                        ),
                                    )
                        else:
                            for it in cutlass.range_constexpr(
                                BT * 8 // self.num_cons_threads
                            ):
                                q = ctid + it * self.num_cons_threads
                                r = q // 8
                                ch = q % 8
                                if r < nvalid:
                                    cute.autovec_copy(
                                        cute.local_tile(sHb[r, None], (16,), (ch,)),
                                        cute.local_tile(
                                            hq[row0 + r, None], (16,), (mt * 8 + ch,)
                                        ),
                                    )
                    # all h of this tile is stored (and sH reads done) before the GEMM1 -> GEMM2 release
                    cute.arch.barrier(
                        barrier_id=1, number_of_threads=self.num_cons_threads
                    )
                    sigq.producer_acquire(sgstate)
                    if ctid == 0:
                        sSig[sgstate.index] = j
                    sigq.producer_commit(sgstate)
                    sgstate.advance()
                else:
                    for sl in cutlass.range(self.g2, unroll=1):
                        cstate = self._kloop(
                            mainloop,
                            cstate,
                            tiled_mma_q,
                            tiled_mma_qz,
                            acc0,
                            acc1,
                            tA,
                            tB,
                            tCrAq,
                            tCrBq0,
                            tCrBq1,
                            sSa,
                            sW,
                            slot,
                            rw0,
                            rw1,
                            sl * self.nkb2,
                            self.nkb2,
                            active_consumer,
                            rA32,
                            sA_base,
                            a_row,
                            a_lo,
                            a_x,
                        )
                        # ---- router weight + fp32 top-k reduction of a 128-token x 256-channel tile:
                        # each warp stages its 16 rows x 64 columns in smem, then red.v4 covers two rows x
                        # 256 contiguous bytes per instruction (full lines instead of 8 scattered rows) ----
                        if active_consumer:
                            wt0 = sWt[rw0, slot]
                            wt1 = sWt[rw1, slot]
                            cb = (mt * self.g2 + sl) * BW
                            for qc in cutlass.range_constexpr(4):
                                for jj8 in cutlass.range_constexpr(8):
                                    for hh in cutlass.range_constexpr(2):
                                        v = 2 * hh + 4 * ((qc % 2) * 8 + jj8)
                                        lr = lane // 4 + 8 * hh
                                        c = 8 * jj8 + 2 * (lane % 4)
                                        wt = wt0 if hh == 0 else wt1
                                        if qc < 2:
                                            p2[0] = acc0[v] * wt
                                            p2[1] = acc0[v + 1] * wt
                                        else:
                                            p2[0] = acc1[v] * wt
                                            p2[1] = acc1[v + 1] * wt
                                        cute.autovec_copy(
                                            p2,
                                            cute.local_tile(
                                                sR[warp_idx, lr, None], (2,), (c // 2,)
                                            ),
                                        )
                                cute.arch.sync_warp()
                                for it in cutlass.range_constexpr(8):
                                    lr = it * 2 + lane // 16
                                    ch = lane % 16
                                    r = wrow0 + lr
                                    if r < nvalid:
                                        cute.autovec_copy(
                                            cute.local_tile(
                                                sR[warp_idx, lr, None], (4,), (ch,)
                                            ),
                                            rv,
                                        )
                                        tb = (
                                            cutlass.Int64(sTok[r, slot]) * H
                                            + cb
                                            + qc * 64
                                            + ch * 4
                                        )
                                        _red_add_v4_f32_keep(
                                            accbuf.iterator + tb,
                                            rv[0],
                                            rv[1],
                                            rv[2],
                                            rv[3],
                                        )
                                cute.arch.sync_warp()
                sched.consumer_release(qs)
                qs.advance()
                sched.consumer_wait(qs)
                idx = sItem[qs.index]
            # stop the signal warp
            sigq.producer_acquire(sgstate)
            if ctid == 0:
                sSig[sgstate.index] = Int32(-1)
            sigq.producer_commit(sgstate)

    @cute.jit
    def _gather(
        self, sA_ptr, xqc, offs, sidx, kb, gq, c16, x0, x1, half: cutlass.Constexpr
    ):
        """Half of a GEMM1 A stage (rows 4 * i + gq for i in [16 * half, 16 * half + 16)): 8 lanes per row each
        copy 16 bytes of the k-block into the 128B-swizzled K-major layout (chunk c of row r lands at chunk
        c ^ (r % 8); r % 8 = gq + 4 * (i % 2), so the two swizzled offsets are precomputed per lane)."""
        dbase = sidx * (self.BT * self.BK) + gq * self.BK
        gofs = kb * self.BK + c16 * 16
        for i in cutlass.range_constexpr(self.BT // 8):
            ii = (self.BT // 8) * half + i
            dst = sA_ptr + (dbase + ii * 4 * self.BK + (x0 if i % 2 == 0 else x1))
            src = xqc.iterator + (cutlass.Int64(offs[i]) * 16 + gofs)
            _cp_async_cg16(dst, src)

    @cute.jit
    def _lda(self, rA32, sA_base, si, a_row, a_lo, a_x):
        """A fragment of this warp's 16 token rows for one 128-K block via ldmatrix.x4 (generic-proxy reads:
        the stage's mbarrier completion already covers the cp.async / TMA writes)."""
        abase = si * (self.BT * self.BK) + a_row * self.BK
        for kk in cutlass.range_constexpr(self.BK // 32):
            q4 = _ldsm_x4(sA_base + (abase + (((2 * kk + a_lo) ^ a_x) * 16)))
            for j in cutlass.range_constexpr(4):
                rA32[j + 4 * kk] = q4[j]

    @cute.jit
    def _kloop(
        self,
        mainloop,
        cstate,
        tmq,
        tmqz,
        acc0,
        acc1,
        tA,
        tB,
        tCrAq,
        tCrBq0,
        tCrBq1,
        sSa,
        sW,
        slot,
        rw0,
        rw1,
        wbase,
        nkb: cutlass.Constexpr,
        active_consumer,
        rA32,
        sA_base,
        a_row,
        a_lo,
        a_x,
    ):
        """nkb 128-K blocks of (128 tokens) x (two 128-row weight halves) with fp32 promotion by the
        token's activation scale and the half's weight block scale (sW[(wbase + kb) * 2 + half]).
        A (token rows) is held in registers for the whole 128-K block; each weight half runs as two n64
        quarter chains alternating between two partials (tA, tB). Chains overlap promotion within a
        k-block; k-blocks are processed in pairs whose second block's first chain overlaps the first
        block's last promotion. No chain is in flight across the loop back-edge."""
        if active_consumer:
            acc0.fill(0.0)
            acc1.fill(0.0)
            for kp in cutlass.range(nkb // 2, unroll=1):
                kb = kp * 2
                si = cstate.index
                mainloop.consumer_wait(cstate)
                self._lda(rA32, sA_base, si, a_row, a_lo, a_x)
                self._mmaq(tmq, tmqz, tA, tCrAq, tCrBq0, si, 0)
                self._mmaq(tmq, tmqz, tB, tCrAq, tCrBq1, si, 0)
                sa0 = sSa[rw0, kb, slot]
                sa1 = sSa[rw1, kb, slot]
                w0 = sW[(wbase + kb) * 2, slot]
                w1 = sW[(wbase + kb) * 2 + 1, slot]
                cute.nvgpu.warpgroup.wait_group(1)
                self._promote(acc0, tA, 0, sa0 * w0, sa1 * w0)
                self._mmaq(tmq, tmqz, tA, tCrAq, tCrBq0, si, 1)
                cute.nvgpu.warpgroup.wait_group(1)
                self._promote(acc0, tB, 32, sa0 * w0, sa1 * w0)
                self._mmaq(tmq, tmqz, tB, tCrAq, tCrBq1, si, 1)
                nstate = cstate.clone()
                nstate.advance()
                sn = nstate.index
                mainloop.consumer_wait(nstate)
                cute.nvgpu.warpgroup.wait_group(1)
                self._promote(acc1, tA, 0, sa0 * w1, sa1 * w1)
                # the A registers are free once the block's last chain retires
                cute.nvgpu.warpgroup.wait_group(0)
                self._lda(rA32, sA_base, sn, a_row, a_lo, a_x)
                self._mmaq(tmq, tmqz, tA, tCrAq, tCrBq0, sn, 0)
                self._promote(acc1, tB, 32, sa0 * w1, sa1 * w1)
                mainloop.consumer_release(cstate)
                self._mmaq(tmq, tmqz, tB, tCrAq, tCrBq1, sn, 0)
                sa0 = sSa[rw0, kb + 1, slot]
                sa1 = sSa[rw1, kb + 1, slot]
                w0 = sW[(wbase + kb + 1) * 2, slot]
                w1 = sW[(wbase + kb + 1) * 2 + 1, slot]
                cute.nvgpu.warpgroup.wait_group(1)
                self._promote(acc0, tA, 0, sa0 * w0, sa1 * w0)
                self._mmaq(tmq, tmqz, tA, tCrAq, tCrBq0, sn, 1)
                cute.nvgpu.warpgroup.wait_group(1)
                self._promote(acc0, tB, 32, sa0 * w0, sa1 * w0)
                self._mmaq(tmq, tmqz, tB, tCrAq, tCrBq1, sn, 1)
                cute.nvgpu.warpgroup.wait_group(1)
                self._promote(acc1, tA, 0, sa0 * w1, sa1 * w1)
                cute.nvgpu.warpgroup.wait_group(0)
                self._promote(acc1, tB, 32, sa0 * w1, sa1 * w1)
                mainloop.consumer_release(nstate)
                cstate.advance()
                cstate.advance()
        else:
            # padding-only warpgroup: same wait/release arrivals, no math
            for kb in cutlass.range(nkb, unroll=1):
                mainloop.consumer_wait(cstate)
                mainloop.consumer_release(cstate)
                cstate.advance()
        return cstate

    @cute.jit
    def _promote(self, acc, t, off: cutlass.Constexpr, f0, f1):
        # quarter register v = cc + 2*hh + 4*jj maps to half register v + off (off = 32 for columns 64..127)
        for v in cutlass.range_constexpr(cute.size(t)):
            acc[v + off] = acc[v + off] + t[v] * (f0 if (v // 2) % 2 == 0 else f1)

    @cute.jit
    def _mmaq(self, tmq, tmqz, t, tCrAq, tCrBq, sidx, half):
        # fresh 4-instruction chain for one 128-K block of a 64-row quarter (first WGMMA overwrites t)
        cute.nvgpu.warpgroup.fence()
        for kk in cutlass.range_constexpr(cute.size(tCrAq, mode=[2])):
            cute.gemm(
                tmqz if kk == 0 else tmq,
                t,
                tCrAq[(None, None, kk)],
                tCrBq[(None, None, kk, sidx * 2 + half)],
                t,
            )
        cute.nvgpu.warpgroup.commit_group()

    @cute.jit
    def _claim(
        self,
        meta,
        dep,
        work,
        w13s,
        w2s,
        xs,
        hsg,
        sorted_pair,
        topk_w,
        sItem,
        sDesc,
        sW,
        sTok,
        sWt,
        sSa,
        sched,
        qstate,
        lane,
        n1,
        total,
        T,
        P_rows,
        cnt0,
        gdim,
        sOff,
        static_idx,
    ):
        """Scheduler warp: claim the next work item, wait for its GEMM1 dependency (GEMM2 items), and
        publish it with its weight block scales (and router info / h scales for GEMM2)."""
        idx = Int32(0)
        if lane == 0:
            if static_idx >= 0:
                idx = static_idx
            else:
                idx = cute.arch.atomic_add(meta.iterator + 1, Int32(1), scope="gpu")
        idx = cute.arch.shuffle_sync(idx, 0)
        sched.producer_acquire(qstate)
        slot = qstate.index
        if idx < total:
            is1 = idx < n1
            j = idx // self.mt1
            mt = idx % self.mt1
            if not is1:
                # output-slice-major: one 512-column slice of the fp32 accumulator per pass
                i2 = idx - n1
                mt = i2 // T
                j = i2 - mt * T
            e = work[j, 0]
            row0 = work[j, 1]
            nvalid = work[j, 2]
            if lane == 0:
                sDesc[0, slot] = e
                sDesc[1, slot] = row0
                sDesc[2, slot] = Int32(1) if is1 else Int32(0)
                sDesc[4, slot] = nvalid
                sDesc[5, slot] = j
                sDesc[6, slot] = mt
                sDesc[7, slot] = Int32(2)
            if is1:
                # t = kb * 2 + half: gate block (mt) / up block (mt + 4) scale of k-block kb
                sW[lane, slot] = w13s[e, mt + (lane % 2) * 4, lane // 2]
                sv = cute.make_rmem_tensor((4,), Float32)
                for r in cutlass.range_constexpr(self.BT // 32):
                    rr = cutlass.min(row0 + lane + 32 * r, P_rows - 1)
                    tv = sorted_pair[rr]
                    sTok[lane + 32 * r, slot] = tv
                    # row lane + 32 r = 4 * (lane // 4 + 8 r) + lane % 4
                    sOff[lane // 4 + 8 * r, lane % 4, slot] = cutlass.Int64(tv) * (
                        H // 16
                    )
                    for k4 in cutlass.range_constexpr(self.nkb1 // 4):
                        cute.autovec_copy(
                            cute.local_tile(xs[tv, None], (4,), (k4,)), sv
                        )
                        for q in cutlass.range_constexpr(4):
                            sSa[lane + 32 * r, k4 * 4 + q, slot] = sv[q]
            else:
                # Stage dependency-independent W2 and router metadata while GEMM1 finishes.
                if lane < self.g2 * self.nkb2 * 2:
                    # t = (sl * 4 + kb) * 2 + half: 128-row block (mt * g2 + sl) * 2 + half
                    sW[lane, slot] = w2s[
                        e, (mt * self.g2 + lane // 8) * 2 + lane % 2, (lane // 2) % 4
                    ]
                for r in cutlass.range_constexpr(self.BT // 32):
                    c = lane + 32 * r
                    rr = cutlass.min(row0 + c, P_rows - 1)
                    tv = sorted_pair[rr]
                    wv = topk_w[rr]
                    sTok[c, slot] = tv
                    sWt[c, slot] = wv if c < nvalid else Float32(0.0)

                # Wait only for the 4 GEMM1 tiles' h/scales and accumulator zeroing.
                if lane == 0:
                    zc = cute.arch.atomic_add(
                        dep.iterator + cnt0, Int32(0), sem="acquire", scope="gpu"
                    )
                    while zc < gdim:
                        zc = cute.arch.atomic_add(
                            dep.iterator + cnt0, Int32(0), sem="acquire", scope="gpu"
                        )
                    cnt = cute.arch.atomic_add(
                        dep.iterator + j, Int32(0), sem="acquire", scope="gpu"
                    )
                    while cnt < self.mt1:
                        cnt = cute.arch.atomic_add(
                            dep.iterator + j, Int32(0), sem="acquire", scope="gpu"
                        )
                cute.arch.sync_warp()
                cute.arch.fence_proxy("async.global")
                for r in cutlass.range_constexpr(self.BT // 32):
                    c = lane + 32 * r
                    rr = cutlass.min(row0 + c, P_rows - 1)
                    for kb in cutlass.range_constexpr(self.nkb2):
                        sSa[c, kb, slot] = hsg[rr, kb]
        if lane == 0:
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
    elif kind == "route_mc_quantc":
        BN = key[1]
        P = _sym(8)
        M = _sym()
        W = _sym()
        fn = cute.compile(
            _route_mc_quantc_host,
            _ft(Int32, (P,), (1,)),
            _ft(Float32, (P,), (1,)),
            _ft(Int32, (E + 1,), (1,)),
            _ft(Int32, (W, 4), (4, 1)),
            _ft(Int32, (4,), (1,)),
            _ft(Int32, (P,), (1,)),
            _ft(Float32, (P,), (1,)),
            _ft(Int32, (W,), (1,)),
            _ft(Int32, (RMC_CTAS, E), (E, 1)),
            _ft(Int32, (2,), (1,)),
            _ft(cutlass.BFloat16, (M, H), (H, 1)),
            _ft(fp8, (M, H), (H, 1)),
            _ft(Float32, (M, H // 128), (H // 128, 1)),
            Int32(1),
            Int32(1),
            Int32(1),
            BN,
            stream,
            options="--enable-tvm-ffi",
        )
    elif kind == "quant_route":
        BN = key[1]
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
    elif kind == "quantc":
        M = _sym()
        fn = cute.compile(
            _quantc_host,
            _ft(cutlass.BFloat16, (M, H), (H, 1)),
            _ft(fp8, (M, H), (H, 1)),
            _ft(Float32, (M, H // 128), (H // 128, 1)),
            Int32(1),
            stream,
            options="--enable-tvm-ffi",
        )
    elif kind == "finalize":
        M = _sym()
        fn = cute.compile(
            _finalize_host,
            _ft(Float32, (M, H), (H, 1)),
            _ft(cutlass.BFloat16, (M, H), (H, 1)),
            Int32(1),
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
            _ft(Int32, (P,), (1,)),
            _ft(Float32, (P,), (1,)),
            _ft(Int32, (P,), (1,)),
            _ft(cutlass.BFloat16, (M, H), (H, 1)),
            Int32(1),
            stream,
            options="--enable-tvm-ffi",
        )
    elif kind == "prefill":
        g = MoEPrefill(key[1], key[2], key[3])
        P = _sym()
        PS = _sym(4)
        M = _sym()
        W = _sym()
        fn = cute.compile(
            g,
            _ft(fp8, (M, H), (H, 1)),
            _ft(Float32, (M, H // 128), (H // 128, 1)),
            _ft(fp8, (E * (2 * I // 256) * (H // 128) * 256, 128), (128, 1)),
            _ft(
                Float32,
                (E, 2 * I // 128, H // 128),
                ((2 * I // 128) * (H // 128), H // 128, 1),
            ),
            _ft(fp8, (P, I), (I, 1)),
            _ft(Float32, (PS, I // 128), (1, PS)),
            _ft(fp8, (E * (H // 256) * (I // 128) * 256, 128), (128, 1)),
            _ft(
                Float32, (E, H // 128, I // 128), ((H // 128) * (I // 128), I // 128, 1)
            ),
            _ft(Int32, (W, 4), (4, 1)),
            _ft(Int32, (4,), (1,)),
            _ft(Int32, (W,), (1,)),
            _ft(Float32, (M, H), (H, 1)),
            _ft(Int32, (P,), (1,)),
            _ft(Float32, (P,), (1,)),
            _ft(cutlass.BFloat16, (M, H), (H, 1)),
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
    extra = (
        1024
        + q * (48 + 64 * 4 + 16 * BN * 4 + 8 * BN)
        + 8 * BN * 4
        + BN * 128
        + 2048
        + (2 * E + 16) * 4
    )
    return min(8, (_SMEM_BYTES - extra) // stage)
