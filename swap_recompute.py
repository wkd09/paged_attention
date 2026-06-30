"""
메모리 관리를 위한 swap 및 recompute 전략.

block을 CPU 메모리로 swap하고 KV 값을 recomputation하는 과정을 시뮬레이션한다.
"""

import torch
from typing import Dict, List, Set, Optional
import time

from .allocator import BlockAllocator, PhysicalBlock
from .kv_cache import PagedKVCache


class SwapManager:
    """
    GPU와 CPU 메모리 사이에서 block swapping을 관리한다.

    성능 분석을 위해 데이터 전송 비용을 시뮬레이션한다.
    """
    def __init__(self, allocator: BlockAllocator, gpu_to_cpu_bandwidth_gbps: float = 25.0):
        """
        swap manager를 초기화한다.

        인자:
            allocator: 블록 할당자
            gpu_to_cpu_bandwidth_gbps: PCIe bandwidth (GB/s)
        """
        self.allocator = allocator
        self.gpu_to_cpu_bandwidth_gbps = gpu_to_cpu_bandwidth_gbps

        # swap된 블록 추적
        self.swapped_blocks: Dict[int, tuple] = {} # block_id -> (cpu_key, cpu_val)

        # 통계
        self.num_swaps_out = 0
        self.num_swaps_in = 0
        self.total_swap_time = 0.0
        self.total_bytes_swapped = 0

    def swap_out_block(self, block_id: int) -> float:
        """
        block을 GPU에서 CPU로 swap out한다.

        인자:
            block_id: swap out할 block

        반환값:
            시뮬레이션된 swap 시간(초)
        """
        block = self.allocator.get_block(block_id)

        if block_id in self.swapped_blocks:
            return 0.0 # 이미 swap됨
        
        # cpu에 복사
        cpu_key = block.key_data.cpu()
        cpu_val = block.value_data.cpu()

        self.swapped_blocks[block_id] = (cpu_key, cpu_val)

        # swap 시간 시뮬레이션
        bytes_transferred = block.key_data.numel() * 4 * 2 # K + V, float32
        swap_time = bytes_transferred / (self.gpu_to_cpu_bandwidth_gbps * 1e9)

        self.num_swaps_out += 1
        self.total_swap_time += swap_time
        self.total_bytes_swapped += bytes_transferred

        return swap_time
    
    def swap_in_block(self, block_id: int) -> float:
        """
        block을 CPU에서 GPU로 swap in한다.

        인자:
            block_id: swap in할 block

        반환값:
            시뮬레이션된 swap 시간(초)
        """
        if block_id not in self.swapped_blocks:
            return 0.0 # swapped 안됨
        
        block = self.allocator.get_block(block_id)
        cpu_key, cpu_val = self.swapped_blocks[block_id]

        # gpu에 복사
        block.key_data.copy_(cpu_key)
        block.value_data.copy_(cpu_val)

        del self.swapped_blocks[block_id]

        # swap 시간 시뮬레이션
        bytes_transferred = block.key_data.numel() * 4 * 2
        swap_time = bytes_transferred / (self.gpu_to_cpu_bandwidth_gbps * 1e9)

        self.num_swaps_in += 1
        self.total_swap_time += swap_time
        self.total_bytes_swapped += bytes_transferred

        return swap_time
    
    def is_swapped(self, block_id: int) -> bool:
        """block이 현재 swap out 상태인지 확인한다."""
        return block_id in self.swapped_blocks
    
    def get_stats(self) -> Dict:
        """swap 통계 정보를 반환한다."""
        return {
            "num_swaps_out": self.num_swaps_out,
            "num_swaps_in": self.num_swaps_in,
            "total_swap_time": self.total_swap_time,
            "total_bytes_swapped": self.total_bytes_swapped,
            "swapped_blocks": len(self.swapped_blocks)
        }
    

class RecomputeManager:
    """
    KV 값을 저장하는 대신 recomputation을 관리한다.

    필요할 때 KV를 다시 계산함으로써 메모리 사용량을 줄이고,
    대신 연산 비용을 더 사용하는 방식이다.
    """
    def __init__(self, compute_time_per_token_ms: float = 0.1):
        """
        recompute manager를 초기화한다.

        인자:
            compute_time_per_token_ms: token 하나의 KV를 recompute하는 데 걸리는 시간(밀리초)
        """
        self.compute_time_per_token_ms = compute_time_per_token_ms

        # recompute되는 블록 추적
        self.recomputed_blocks: Set[int] = set()

        # 통계
        self.num_recomputes = 0
        self.total_recompute_time = 0.0
        self.tokens_recomputed = 0

    def recompute_block(self, block_id: int, num_tokens: int,
                        hidden_dim: int) -> tuple:
        """
        block의 KV를 recompute한다.

        인자:
            block_id: block 식별자
            num_tokens: block 안의 token 개수
            hidden_dim: hidden dimension

        반환값:
            (key_tensor, value_tensor, recompute_time)
        """
        # recomputation 시뮬레이션
        recompute_time = num_tokens * self.compute_time_per_token_ms / 1000.0

        # 재계산된 더미 value 생성
        key_tensor = torch.randn(num_tokens, hidden_dim)
        value_tensor = torch.randn(num_tokens, hidden_dim)

        self.recomputed_blocks.add(block_id)
        self.num_recomputes += 1
        self.total_recompute_time += recompute_time
        self.tokens_recomputed += num_tokens

        return key_tensor, value_tensor, recompute_time
    
    def mark_for_recompute(self, block_id: int):
        """block을 recomputation 대상으로 표시한다."""
        self.recomputed_blocks.add(block_id)

    def is_recomputed(self, block_id: int) -> bool:
        """block이 recomputation 대상으로 표시되어 있는지 확인한다."""
        return block_id in self.recomputed_blocks
    
    def get_stats(self) -> Dict:
        """recompute 통계 반환"""
        return {
            "num_recomputes": self.num_recomputes,
            "total_recompute_time": self.total_recompute_time,
            "tokens_recomputed": self.tokens_recomputed,
            "avg_time_per_token": (self.total_recompute_time / self.tokens_recomputed
                                   if self.tokens_recomputed > 0 else 0)
        }
    

class HybridMemoryManager:
    """
    swap과 recomputation 전략을 결합한다.

    access pattern을 기반으로 swap할지 recompute할지 결정한다.
    """
    def __init__(self, swap_manager: SwapManager, recompute_manager: RecomputeManager):
        self.swap_manager = swap_manager
        self.recompute_manager = recompute_manager

        # 접근 빈도 추적
        self.access_counts: Dict[int, int] = {}

    def evict_block(self, block_id: int, num_tokens: int) -> str:
        """
        swap할지 recompute 대상으로 표시할지 결정한다.

        인자:
            block_id: eviction할 block
            num_tokens: block 안의 token 개수

        반환값:
            사용된 전략 ('swap' 또는 'recompute')
        """
        access_count = self.access_counts.get(block_id, 0)

        # 간단한 휴리스틱: 자주 접근되는 블록은 swap
        # 드물게 접근되는 블록은 recompute 가능
        if access_count > 2:
            self.swap_manager.swap_out_block(block_id)
            return "swap"
        else:
            self.recompute_manager.mark_for_recompute(block_id)
            return "recompute"
        
    def access_block(self, block_id: int):
        """블록 접근을 기록한다."""
        self.access_counts[block_id] = self.access_counts.get(block_id, 0) + 1

    def get_combined_stats(self) -> Dict:
        """결합된 통계 정보를 반환한다."""
        return {
            "swap_stats": self.swap_manager.get_stats(),
            "recompute_stats": self.recompute_manager.get_stats(),
            "total_evictions": (self.swap_manager.num_swaps_out +
                                self.recompute_manager.num_recomputes)
        }
