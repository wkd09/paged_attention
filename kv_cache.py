"""
logical-to-physical block 매핑을 사용하는 Paged KV cache 구현.

각 sequence는 logical block을 physical block에 매핑하는 block table을 유지한다.
효율적인 append, read, copy-on-write 연산을 지원한다.
"""

import torch
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .allocator import BlockAllocator, PhysicalBlock

@dataclass
class BlockTableEntry:
    """
    sequence의 block table에 들어가는 항목.

    속성:
        logical_idx: sequence 안에서의 logical block 인덱스
        phys_block_id: allocator에서 관리하는 physical block ID
        filled: 이 블록 안에 들어 있는 유효한 토큰 수
    """
    logical_idx: int
    phys_block_id: int
    filled: int

class PagedKVCache:
    """
    paged memory를 사용하는 단일 sequence용 KV cache.

    logical block을 physical block에 매핑하는 block table을 관리한다.
    새로운 토큰을 append하고, attention 계산을 위해 block을 읽는 기능을 지원한다.
    """
    def __init__(self, block_size: int, hidden_dim: int,
                 allocator:BlockAllocator, seq_id: Optional[int] = None):
        """
        Paged KV cache를 초기화한다.

        인자:
            block_size: 블록당 토큰 슬롯 수
            hidden_dim: K/V 벡터의 차원
            allocator: 사용할 블록 할당자
            seq_id: 선택적인 sequence 식별자
        """
        self.block_size = block_size
        self.hidden_dim = hidden_dim
        self.allocator = allocator
        self.seq_id = seq_id

        # 블록 테이블: 논리 블록 매핑들의 리스트
        self.block_table: List[BlockTableEntry] = []

        # 전체 token 수 저장
        self.num_tokens = 0

    def append_token_kv(self, key_vec: torch.Tensor, val_vec:torch.Tensor):
        """
        새로운 토큰에 대한 K/V 벡터를 추가한다.

        인자:
            key_vec: Key 벡터 [hidden_dim]
            val_vec: Value 벡터 [hidden_dim]
        """
        if key_vec.shape != (self.hidden_dim,):
            raise ValueError(f"예상한 key shape은 ({self.hidden_dim},)인데, 실제 값은 {key_vec.shape}입니다.")
        if val_vec.shape != (self.hidden_dim,):
            raise ValueError(f"예상한 val shape은 ({self.hidden_dim},)인데, 실제 값은 {val_vec.shape}입니다.")
         
        # 새 블록이 필요한지 확인
        if not self.block_table or self.block_table[-1].filled >= self.block_size:
            # 새로운 블록 할당
            phys_id = self.allocator.allocate()
            logical_idx = len(self.block_table)
            self.block_table.append(BlockTableEntry(
                logical_idx=logical_idx,
                phys_block_id=phys_id,
                filled=0
            ))

        # 마지막 블록을 가져와서 추가
        entry = self.block_table[-1]
        block = self.allocator.get_block(entry.phys_block_id)

        # K/V를 블록에 기록
        block.key_data[entry.filled] = key_vec
        block.value_data[entry.filled] = val_vec
        entry.filled += 1
        block.filled = entry.filled

        self.num_tokens += 1

    def append_token_kv_batch(self, keys: torch.Tensor, values: torch.Tensor):
        """
        여러 개의 K/V 쌍을 한 번에 추가한다.

        인자:
            keys: Key 벡터들 [num_tokens, hidden_dim]
            values: Value 벡터들 [num_tokens, hidden_dim]
        """
        num_tokens = keys.shape[0]
        for i in range(num_tokens):
            self.append_token_kv(keys[i], values[i])

    def read_blocks_for_attention(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        attention 계산을 위해 모든 블록을 읽는다.

        반환값:
            (keys_list, values_list): 블록별 텐서 리스트.
            각 텐서는 [filled_slots, hidden_dim] 형태이다.
        """
        keys_list = []
        values_list = []

        for entry in self.block_table:
            block = self.allocator.get_block(entry.phys_block_id)
            # 채워진 부분만 반환
            keys_list.append(block.key_data[:entry.filled])
            values_list.append(block.value_data[:entry.filled])

        return keys_list, values_list
    
    def get_all_keys_values(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        모든 key와 value를 연속적인 텐서 형태로 가져온다. (baseline 비교용)

        반환값:
            (keys, values): 각각 [num_tokens, hidden_dim] 형태
        """
        keys_list, values_list = self.read_blocks_for_attention()

        if not keys_list:
            device = self.allocator.device
            return (torch.zeros(0, self.hidden_dim, device=device),
                    torch.zeros(0, self.hidden_dim, device=device))
        
        keys = torch.cat(keys_list, dim=0)
        values = torch.cat(values_list, dim=0)
        return keys, values
    
    def fork(self)->"PagedKVCache":
        """
        beam search 또는 parallel sampling을 위해 이 cache를 fork한다.
        reference counting을 통해 physical block을 공유한다. (copy-on-write)

        반환값:
            현재 cache와 block을 공유하는 새로운 PagedKVCache
        """
        new_cache = PagedKVCache(
            self.block_size,
            self.hidden_dim,
            self.allocator,
            seq_id=None # 새 시퀀스
        )

        # 모든 블록 공유
        for entry in self.block_table:
            self.allocator.inc_ref(entry.phys_block_id)
            new_cache.block_table.append(BlockTableEntry(
                logical_idx=entry.logical_idx,
                phys_block_id=entry.phys_block_id,
                filled=entry.filled
            ))

        new_cache.num_tokens = self.num_tokens
        return new_cache
    
    def cow_append(self, key_vec:torch.Tensor, val_vec:torch.Tensor):
        """
        copy-on-write 방식으로 append한다.
        마지막 블록이 공유 중이라면 먼저 복사한 뒤 append한다.

        인자:
            key_vec: Key 벡터 [hidden_dim]
            val_vec: Value 벡터 [hidden_dim]
        """
        if self.block_table:
            last_entry = self.block_table[-1]
            last_block = self.allocator.get_block(last_entry.phys_block_id)

            # 공유 중이고 아직 가득 차지 않았다면, 쓰기 전에 복사
            if last_block.refcount > 1 and not last_block.is_full():
                # 블록 복사
                new_block_id = self.allocator.copy_block(last_entry.phys_block_id)
                # 기존 블록의 참조 카운트 감소
                self.allocator.dec_ref(last_entry.phys_block_id)
                # 테이블 항목 업데이트
                last_entry.phys_block_id = new_block_id

        # 이제 일반 방식으로 append
        self.append_token_kv(key_vec, val_vec)

    def free_all(self):
        """이 cache의 모든 블록을 해제한다."""
        for entry in self.block_table:
            self.allocator.free(entry.phys_block_id)
        self.block_table.clear()
        self.num_tokens = 0

    def get_num_blocks(self) -> int:
        """논리 블록의 개수를 반환한다."""
        return len(self.block_table)
    
    def get_memory_usage(self)->int:
        """메모리 사용량을 바이트 단위로 반환한다. (K + V 데이터)"""
        num_blocks = len(self.block_table)
        bytes_per_element = 4 #float32
        memory = num_blocks * self.block_size * self.hidden_dim * 2 * bytes_per_element
        return memory
    
    def get_wasted_memory(self) -> int:
        """내부 단편화로 인해 낭비되는 메모리를 반환한다."""
        if not self.block_table:
            return 0
        
        total_allocated = len(self.block_table) * self.block_size
        total_used = self.num_tokens
        wasted_slots = total_allocated - total_used

        bytes_per_element = 4
        return wasted_slots * self.hidden_dim * 2 * bytes_per_element
