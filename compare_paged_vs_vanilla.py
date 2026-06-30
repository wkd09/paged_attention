"""
Compare vanilla KV-cache attention with the paged-attention implementation.

Run from this directory:
    python compare_paged_vs_vanilla.py
"""

from __future__ import annotations

import math
import time
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from paged_attention.allocator import BlockAllocator
from paged_attention.decoding import ParallelSamplingManager
from paged_attention.kv_cache import PagedKVCache
from paged_attention.paged_attention import PagedAttention


def bytes_to_mib(num_bytes: int) -> float:
    return num_bytes / (1024 * 1024)


def kv_bytes(num_tokens: int, hidden_dim: int, dtype_bytes: int = 4) -> int:
    return num_tokens * hidden_dim * 2 * dtype_bytes


def build_projected_cache(
    attention: PagedAttention,
    keys: torch.Tensor,
    values: torch.Tensor,
    block_size: int,
    total_blocks: int,
) -> PagedKVCache:
    allocator = BlockAllocator(
        total_blocks=total_blocks,
        block_size=block_size,
        hidden_dim=attention.hidden_dim,
        device=str(keys.device),
    )
    cache = PagedKVCache(
        block_size=block_size,
        hidden_dim=attention.hidden_dim,
        allocator=allocator,
    )

    with torch.no_grad():
        projected_k = attention.k_proj(keys).squeeze(0)
        projected_v = attention.v_proj(values).squeeze(0)
        cache.append_token_kv_batch(projected_k, projected_v)

    return cache


def compare_outputs() -> None:
    torch.manual_seed(7)

    hidden_dim = 128
    num_heads = 8
    block_size = 16
    kv_len = 97
    q_len = 3

    attention = PagedAttention(hidden_dim, num_heads, block_size).eval()
    query = torch.randn(1, q_len, hidden_dim)
    keys = torch.randn(1, kv_len, hidden_dim)
    values = torch.randn(1, kv_len, hidden_dim)

    cache = build_projected_cache(
        attention,
        keys,
        values,
        block_size=block_size,
        total_blocks=math.ceil(kv_len / block_size) + 1,
    )

    with torch.no_grad():
        vanilla = attention.forward_vanilla(query, keys, values)
        paged = attention.forward_paged(query, cache)

    max_abs_diff = (vanilla - paged).abs().max().item()
    mean_abs_diff = (vanilla - paged).abs().mean().item()

    print("[1] Output equivalence")
    print(f"    kv_len={kv_len}, block_size={block_size}")
    print(f"    max_abs_diff : {max_abs_diff:.8f}")
    print(f"    mean_abs_diff: {mean_abs_diff:.8f}")
    print()


def compare_memory() -> None:
    hidden_dim = 4096
    block_size = 16
    max_model_len = 2048
    sequence_lengths = [31, 64, 127, 255, 513, 777, 1024]

    vanilla_reserved = len(sequence_lengths) * kv_bytes(max_model_len, hidden_dim)
    vanilla_used = sum(kv_bytes(seq_len, hidden_dim) for seq_len in sequence_lengths)

    paged_blocks = sum(math.ceil(seq_len / block_size) for seq_len in sequence_lengths)
    paged_allocated = paged_blocks * kv_bytes(block_size, hidden_dim)
    paged_waste = paged_allocated - vanilla_used

    print("[2] KV-cache memory")
    print(f"    sequences              : {sequence_lengths}")
    print(f"    hidden_dim/block_size  : {hidden_dim}/{block_size}")
    print(f"    vanilla fixed reserve  : {bytes_to_mib(vanilla_reserved):8.2f} MiB")
    print(f"    actual used tokens     : {bytes_to_mib(vanilla_used):8.2f} MiB")
    print(f"    paged allocated        : {bytes_to_mib(paged_allocated):8.2f} MiB")
    print(f"    paged internal waste   : {bytes_to_mib(paged_waste):8.2f} MiB")
    print(f"    saving vs fixed reserve: {(1 - paged_allocated / vanilla_reserved) * 100:7.2f}%")
    print()


def compare_cow_sharing() -> None:
    torch.manual_seed(11)

    hidden_dim = 4096
    block_size = 16
    prompt_len = 513
    generated_len = 32
    num_samples = 8

    total_blocks = 10000
    allocator = BlockAllocator(total_blocks, block_size, hidden_dim)
    prompt_cache = PagedKVCache(block_size, hidden_dim, allocator)
    prompt_cache.append_token_kv_batch(
        torch.randn(prompt_len, hidden_dim),
        torch.randn(prompt_len, hidden_dim),
    )

    sampling = ParallelSamplingManager(allocator, block_size, hidden_dim)
    sample_ids = sampling.create_samples(prompt_cache, num_samples)

    for sample_id in sample_ids:
        cache = sampling.get_sample_cache(sample_id)
        for _ in range(generated_len):
            cache.cow_append(torch.randn(hidden_dim), torch.randn(hidden_dim))

    used_blocks = allocator.get_status()["used_blocks"]
    paged_cow_memory = used_blocks * kv_bytes(block_size, hidden_dim)

    vanilla_copy_memory = num_samples * kv_bytes(
        prompt_len + generated_len,
        hidden_dim,
    )

    print("[3] Parallel sampling / COW sharing")
    print(f"    samples                : {num_samples}")
    print(f"    prompt/generated tokens: {prompt_len}/{generated_len}")
    print(f"    vanilla full copies    : {bytes_to_mib(vanilla_copy_memory):8.2f} MiB")
    print(f"    paged COW blocks       : {bytes_to_mib(paged_cow_memory):8.2f} MiB")
    print(f"    used physical blocks   : {used_blocks}")
    print(f"    COW copies             : {allocator.num_cow_copies}")
    print(f"    saving vs full copies  : {(1 - paged_cow_memory / vanilla_copy_memory) * 100:7.2f}%")
    print()


def compare_cpu_runtime() -> None:
    torch.manual_seed(13)

    hidden_dim = 256
    num_heads = 8
    block_size = 16
    kv_len = 512
    q_len = 1
    runs = 20

    attention = PagedAttention(hidden_dim, num_heads, block_size).eval()
    query = torch.randn(1, q_len, hidden_dim)
    keys = torch.randn(1, kv_len, hidden_dim)
    values = torch.randn(1, kv_len, hidden_dim)
    cache = build_projected_cache(
        attention,
        keys,
        values,
        block_size=block_size,
        total_blocks=math.ceil(kv_len / block_size) + 1,
    )

    with torch.no_grad():
        for _ in range(3):
            attention.forward_vanilla(query, keys, values)
            attention.forward_paged(query, cache)

        start = time.perf_counter()
        for _ in range(runs):
            attention.forward_vanilla(query, keys, values)
        vanilla_time = (time.perf_counter() - start) / runs

        start = time.perf_counter()
        for _ in range(runs):
            attention.forward_paged(query, cache)
        paged_time = (time.perf_counter() - start) / runs

    print("[4] CPU runtime sanity check")
    print(f"    kv_len/runs            : {kv_len}/{runs}")
    print(f"    vanilla dense matmul   : {vanilla_time * 1000:8.3f} ms")
    print(f"    paged python block loop: {paged_time * 1000:8.3f} ms")
    print("    note: this educational Python version optimizes memory behavior, not kernel speed.")
    print()


def main() -> None:
    compare_outputs()
    compare_memory()
    compare_cow_sharing()
    compare_cpu_runtime()


if __name__ == "__main__":
    main()
