"""CUDA graph accuracy coverage for Gemma 4 FP8 paged attention on SM80+."""

import math
from dataclasses import dataclass

import pytest
import torch

import flashinfer
from flashinfer.utils import get_compute_capability


DEVICE = torch.device("cuda:0")
Q_DTYPE = torch.bfloat16
KV_DTYPE = torch.float8_e4m3fn
NUM_QO_HEADS = 16
PAGE_SIZE = 16
MAX_KV_LEN = 4096
WORKSPACE_SIZE = 256 * 1024 * 1024
QUERY_PRE_ATTN_SCALE = 16
SM_SCALE = 1.0


@dataclass(frozen=True)
class Gemma4AttentionConfig:
    num_kv_heads: int
    head_dim: int
    window_left: int


GEMMA4_ATTENTION_CONFIGS = [
    pytest.param(
        Gemma4AttentionConfig(
            num_kv_heads=8,
            head_dim=256,
            window_left=1024,
        ),
        id="sliding-hd256",
    ),
    pytest.param(
        Gemma4AttentionConfig(
            num_kv_heads=2,
            head_dim=512,
            window_left=-1,
        ),
        id="full-hd512",
    ),
]


def _require_ampere_or_newer() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if get_compute_capability(DEVICE)[0] < 8:
        pytest.skip("Gemma 4 FP8 FA2 attention coverage requires SM80 or newer")


def _make_metadata(
    q_lens: list[int],
    kv_lens: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert len(q_lens) == len(kv_lens)
    assert all((q_len == 0) == (kv_len == 0) for q_len, kv_len in zip(q_lens, kv_lens))
    page_counts = [math.ceil(kv_len / PAGE_SIZE) if kv_len else 0 for kv_len in kv_lens]
    qo_indptr = torch.tensor(
        [0, *torch.tensor(q_lens).cumsum(0).tolist()], dtype=torch.int32
    )
    paged_kv_indptr = torch.tensor(
        [0, *torch.tensor(page_counts).cumsum(0).tolist()], dtype=torch.int32
    )
    paged_kv_indices = torch.arange(
        sum(page_counts), dtype=torch.int32, device=DEVICE
    )
    paged_kv_last_page_len = torch.tensor(
        [((kv_len - 1) % PAGE_SIZE) + 1 if kv_len else 0 for kv_len in kv_lens],
        dtype=torch.int32,
    )
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32)
    return (
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        seq_lens,
    )


def _plan(
    wrapper: flashinfer.BatchPrefillWithPagedKVCacheWrapper,
    metadata: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    config: Gemma4AttentionConfig,
) -> None:
    (
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        seq_lens,
    ) = metadata
    wrapper.plan(
        qo_indptr=qo_indptr,
        paged_kv_indptr=paged_kv_indptr,
        paged_kv_indices=paged_kv_indices,
        paged_kv_last_page_len=paged_kv_last_page_len,
        seq_lens=seq_lens,
        num_qo_heads=NUM_QO_HEADS,
        num_kv_heads=config.num_kv_heads,
        head_dim_qk=config.head_dim,
        page_size=PAGE_SIZE,
        causal=True,
        sm_scale=SM_SCALE,
        window_left=config.window_left,
        q_data_type=Q_DTYPE,
        kv_data_type=KV_DTYPE,
        o_data_type=Q_DTYPE,
        fixed_split_size=-1,
        disable_split_kv=False,
    )


def _refresh_inputs(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: float,
    v_scale: float,
) -> None:
    query.copy_(torch.randn_like(query) / QUERY_PRE_ATTN_SCALE)
    k_cache.copy_((torch.randn_like(k_cache, dtype=Q_DTYPE) / k_scale).to(KV_DTYPE))
    v_cache.copy_((torch.randn_like(v_cache, dtype=Q_DTYPE) / v_scale).to(KV_DTYPE))


def _reference(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    q_lens: list[int],
    kv_lens: list[int],
    config: Gemma4AttentionConfig,
    k_scale: float,
    v_scale: float,
) -> torch.Tensor:
    outputs = []
    query_offset = 0
    page_offset = 0
    group_size = NUM_QO_HEADS // config.num_kv_heads

    for q_len, kv_len in zip(q_lens, kv_lens):
        if q_len == 0:
            continue
        num_pages = math.ceil(kv_len / PAGE_SIZE)
        q = query[query_offset : query_offset + q_len].float()
        k = (
            k_cache[page_offset : page_offset + num_pages]
            .reshape(-1, config.num_kv_heads, config.head_dim)[:kv_len]
            .float()
            * k_scale
        )
        v = (
            v_cache[page_offset : page_offset + num_pages]
            .reshape(-1, config.num_kv_heads, config.head_dim)[:kv_len]
            .float()
            * v_scale
        )

        query_positions = torch.arange(q_len, device=DEVICE) + kv_len - q_len
        key_positions = torch.arange(kv_len, device=DEVICE)
        mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        if config.window_left >= 0:
            mask &= key_positions.unsqueeze(0) >= (
                query_positions.unsqueeze(1) - config.window_left
            )

        output = torch.empty_like(q)
        for kv_head in range(config.num_kv_heads):
            head_start = kv_head * group_size
            head_end = head_start + group_size
            scores = torch.matmul(
                q[:, head_start:head_end],
                k[:, kv_head].t(),
            ) * SM_SCALE
            scores.masked_fill_(~mask.unsqueeze(1), float("-inf"))
            probabilities = torch.softmax(scores, dim=-1)
            output[:, head_start:head_end] = torch.matmul(
                probabilities,
                v[:, kv_head],
            )

        outputs.append(output)
        query_offset += q_len
        page_offset += num_pages

    return torch.cat(outputs).to(Q_DTYPE)


def _run_graph_bucket(
    config: Gemma4AttentionConfig,
    max_q_len: int,
    capture_batch_size: int,
    replay_cases: list[tuple[list[int], list[int]]],
    k_scale: float,
    v_scale: float,
) -> None:
    max_pages = capture_batch_size * math.ceil(MAX_KV_LEN / PAGE_SIZE)
    num_physical_tokens = capture_batch_size * max_q_len
    query = torch.empty(
        num_physical_tokens,
        NUM_QO_HEADS,
        config.head_dim,
        dtype=Q_DTYPE,
        device=DEVICE,
    )
    k_cache = torch.empty(
        max_pages,
        PAGE_SIZE,
        config.num_kv_heads,
        config.head_dim,
        dtype=KV_DTYPE,
        device=DEVICE,
    )
    v_cache = torch.empty_like(k_cache)
    _refresh_inputs(query, k_cache, v_cache, k_scale, v_scale)

    graph_workspace = torch.empty(WORKSPACE_SIZE, dtype=torch.uint8, device=DEVICE)
    eager_workspace = torch.empty_like(graph_workspace)
    graph_output = torch.empty_like(query)
    eager_output = torch.empty_like(query)

    graph_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        graph_workspace,
        "NHD",
        use_cuda_graph=True,
        qo_indptr_buf=torch.empty(
            capture_batch_size + 1, dtype=torch.int32, device=DEVICE
        ),
        paged_kv_indptr_buf=torch.empty(
            capture_batch_size + 1, dtype=torch.int32, device=DEVICE
        ),
        paged_kv_indices_buf=torch.empty(max_pages, dtype=torch.int32, device=DEVICE),
        paged_kv_last_page_len_buf=torch.empty(
            capture_batch_size, dtype=torch.int32, device=DEVICE
        ),
        backend="fa2",
    )
    eager_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        eager_workspace,
        "NHD",
        backend="fa2",
    )

    capture_kv_lens = [1024] * capture_batch_size
    capture_q_lens = [max_q_len] * capture_batch_size
    _plan(graph_wrapper, _make_metadata(capture_q_lens, capture_kv_lens), config)
    graph_wrapper.run(
        query,
        (k_cache, v_cache),
        out=graph_output,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_wrapper.run(
            query,
            (k_cache, v_cache),
            out=graph_output,
            k_scale=k_scale,
            v_scale=v_scale,
        )

    for q_lens, kv_lens in replay_cases:
        assert len(q_lens) == capture_batch_size
        assert len(kv_lens) == capture_batch_size
        logical_batch_size = sum(q_len > 0 for q_len in q_lens)
        logical_tokens = sum(q_lens)
        assert all(0 < q_len <= max_q_len for q_len in q_lens[:logical_batch_size])
        assert all(q_len == 0 for q_len in q_lens[logical_batch_size:])
        assert all(kv_lens[i] > 0 for i in range(logical_batch_size))
        assert all(kv_lens[i] == 0 for i in range(logical_batch_size, len(kv_lens)))

        _plan(graph_wrapper, _make_metadata(q_lens, kv_lens), config)
        _refresh_inputs(query, k_cache, v_cache, k_scale, v_scale)
        graph_output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()

        active_q_lens = q_lens[:logical_batch_size]
        active_kv_lens = kv_lens[:logical_batch_size]
        _plan(
            eager_wrapper,
            _make_metadata(active_q_lens, active_kv_lens),
            config,
        )
        eager_wrapper.run(
            query[:logical_tokens],
            (k_cache, v_cache),
            out=eager_output[:logical_tokens],
            k_scale=k_scale,
            v_scale=v_scale,
        )
        torch.cuda.synchronize()

        reference_output = _reference(
            query,
            k_cache,
            v_cache,
            q_lens,
            kv_lens,
            config,
            k_scale,
            v_scale,
        )
        torch.testing.assert_close(
            eager_output[:logical_tokens],
            reference_output,
            rtol=2e-2,
            atol=2e-2,
        )
        torch.testing.assert_close(
            graph_output[:logical_tokens],
            reference_output,
            rtol=2e-2,
            atol=2e-2,
        )
        torch.testing.assert_close(
            graph_output[:logical_tokens],
            eager_output[:logical_tokens],
            rtol=1e-2,
            atol=1e-2,
        )
        if logical_tokens < num_physical_tokens:
            assert torch.isnan(graph_output[logical_tokens:]).all()


@pytest.mark.parametrize("config", GEMMA4_ATTENTION_CONFIGS)
@pytest.mark.parametrize("q_len", [1, 4], ids=["decode-q1", "mtp3-verify-q4"])
@pytest.mark.parametrize(
    ("k_scale", "v_scale"),
    [(1.0, 1.0), (0.02, 0.03)],
    ids=["unit-kv-scale", "explicit-kv-scale"],
)
def test_gemma4_fp8_kv_paged_prefill_cuda_graph_matches_eager(
    config: Gemma4AttentionConfig,
    q_len: int,
    k_scale: float,
    v_scale: float,
) -> None:
    """Compare the FP32 oracle, eager execution, and graph replay."""
    _require_ampere_or_newer()
    torch.manual_seed(20260918)

    # Four requests include the exact MTP3 transition from 16 physical query
    # rows to 12 logical rows. Eight requests cover the larger graph bucket and
    # page/window boundaries seen during serving.
    bucket4_cases = [
        ([q_len] * 4, [1024, 1201, 2049, 4095]),
        ([q_len] * 3 + [0], [1201, 2049, 4095, 0]),
        ([q_len, 0, 0, 0], [4095, 0, 0, 0]),
    ]
    bucket8_cases = [
        ([q_len] * 8, [512, 1023, 1024, 1025, 2047, 2048, 2049, 4095]),
        ([q_len] * 7 + [0], [513, 1024, 1025, 1537, 2048, 3073, 4095, 0]),
        ([q_len] * 4 + [0] * 4, [1201, 1710, 2417, 4095, 0, 0, 0, 0]),
    ]
    _run_graph_bucket(
        config,
        q_len,
        capture_batch_size=4,
        replay_cases=bucket4_cases,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    _run_graph_bucket(
        config,
        q_len,
        capture_batch_size=8,
        replay_cases=bucket8_cases,
        k_scale=k_scale,
        v_scale=v_scale,
    )
