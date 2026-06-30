"""
참조 카운팅을 사용하는 물리 블록 할당자.

물리 메모리 블록 풀을 관리하고,
효율적인 메모리 공유를 위해 참조 카운팅을 통해
각 블록의 사용 상태를 추적한다.
"""

import torch
from collections import deque
from typing import Dict, Optional, List
from dataclasses import dataclass

@dataclass
class PhysicalBlock:
    """
    KV 쌍을 저장하기 위한 고정 크기의 메모리 블록.

    속성:
        block_id: 이 블록의 고유 식별자
        block_size: 이 블록에 들어갈 수 있는 토큰 슬롯 수
        hidden_dim: K와 V 벡터의 차원
        device: 블록 메모리가 위치한 디바이스
        key_data: key 벡터를 저장하는 텐서 [block_size, hidden_dim]
        value_data: value 벡터를 저장하는 텐서 [block_size, hidden_dim]
        filled: 현재 사용 중인 유효한 토큰 슬롯 수
        refcount: copy-on-write를 위한 참조 카운트
    """
    block_id: int
    block_size: int
    hidden_dim: int
    device: str
    key_data: torch.tensor
    value_data: torch.tensor
    filled: int = 0
    refcount: int = 0

    @staticmethod
    def create(
        block_id: int,
        block_size: int,
        hidden_dim:int,
        device: str="cpu"
        ) -> "PhysicalBlock":
        """새로운 물리 블록을 생성하는 팩토리 메서드."""
        key_data = torch.zeros((block_size, hidden_dim),
                               device=device, dtype=torch.float32)
        value_data = torch.zeros((block_size, hidden_dim),
                                 device=device, dtype=torch.float32)
        return PhysicalBlock(
            block_id=block_id,
            block_size=block_size,
            hidden_dim=hidden_dim,
            device=device,
            key_data=key_data,
            value_data=value_data,
            filled=0,
            refcount=0
        )
    
    def is_full(self)->bool:
        """블록이 꽉 찼는지 확인"""
        return self.filled >= self.block_size
    
    def is_empty(self)->bool:
        """블록이 비었는지 확인"""
        return self.filled==0
    
    def reset(self):
        """블록 상태 리셋"""
        self.filled=0
        self.refcount=0
        self.key_data.zero_()
        self.value_data.zero_()

class OOM_Error(Exception):
    pass

class BlockAllocator():
    """
    메모리 블록의 할당과 해제를 관리한다.

    copy-on-write 의미를 구현하기 위해 참조 카운팅을 사용한다.
    블록은 수정되기 전까지 여러 시퀀스에서 공유될 수 있다.
    """
    def __init__(self, total_blocks:int, block_size:int, hidden_dim:int, device:str = "cpu"):
        """
        블록 할당자를 초기화한다.

        인자:
            total_blocks: 풀에 있는 전체 물리 블록 수
            block_size: 블록당 토큰 슬롯 수
            hidden_dim: K/V 벡터의 차원
            device: 블록 메모리를 저장할 디바이스 ('cpu' 또는 'cuda')
        """
        self.total_blocks = total_blocks
        self.block_size = block_size
        self.hidden_dim = hidden_dim
        self.device = device

        # 모든 블록 미리 생성
        self.blocks: Dict[int, PhysicalBlock] = {}
        for i in range(total_blocks):
            self.blocks[i] = PhysicalBlock.create(
                i, block_size, hidden_dim, device
            )

        # 사용 가능한 블록 ID 목록
        self.free_list: deque = deque(range(total_blocks))
        
        # 통계
        self.num_allocations = 0
        self.num_frees = 0
        self.num_cow_copies = 0

    def allocate(self) -> int:
        """
        새로운 블록을 할당한다.

        반환값:
            할당된 블록의 ID

        예외:
            OOM_Error: 사용 가능한 빈 블록이 없을 경우
        """
        if not self.free_list:
            raise OOM_Error(
                f"사용 가능한 빈 블록이 없습니다. (전체: {self.total_blocks})"
            )
        
        block_id = self.free_list.popleft()
        block = self.blocks[block_id]
        block.refcount = 1
        block.filled = 0

        self.num_allocations += 1
        return block_id
    
    def free(self, block_id: int):
        """
        블록을 해제한다.
        (refcount를 감소시키고, 0이 되면 풀에 반환한다.)

        인자:
            block_id: 해제할 블록의 ID
        """
        if block_id not in self.blocks:
            raise ValueError(f"잘못된 block_id입니다: {block_id}")
        
        block = self.blocks[block_id]
        if block.refcount <= 0:
            raise ValueError(f"블록 {block_id}는 이미 해제된 상태입니다. (refcount={block.refcount})")
        
        block.refcount -= 1
        if block.refcount == 0:
            # 블록 다시 free_list로 반환
            block.reset()
            self.free_list.append(block_id)
            self.num_frees += 1

        
    def inc_ref(self, block_id: int):
        """
        참조 카운트를 증가시킨다. (공유를 위해)

        인자:
            block_id: 참조 카운트를 증가시킬 블록의 ID
        """
        if block_id not in self.blocks:
            raise ValueError(f"잘못된 block_id입니다: {block_id}")
        
        self.blocks[block_id].refcount += 1

    def dec_ref(self, block_id: int):
        """
        참조 카운트를 감소시킨다.

        인자:
            block_id: 참조 카운트를 감소시킬 블록의 ID
        """
        self.free(block_id)

    def get_block(self, block_id: int) -> PhysicalBlock:
        """
        ID로 블록을 가져온다.

        인자:
            block_id: 블록 식별자

        반환값:
            PhysicalBlock 객체
        """
        if block_id not in self.blocks:
            raise ValueError(f"잘못된 block_id입니다: {block_id}")
        return self.blocks[block_id]

    def copy_block(self, src_block_id: int) -> int:
        """
        copy-on-write를 위해 블록을 복사한다.

        인자:
            src_block_id: 복사할 원본 블록

        반환값:
            복사된 데이터를 가진 새로운 블록 ID
        """
        src_block = self.get_block(src_block_id)
        new_block_id = self.allocate()
        new_block = self.get_block(new_block_id)

        # 데이터 복사
        new_block.key_data[:src_block.filled] = src_block.key_data[:src_block.filled]
        new_block.value_data[:src_block.filled] = src_block.value_data[:src_block.filled]
        new_block.filled = src_block.filled

        self.num_cow_copies += 1
        return new_block_id
    
    def get_num_free_blocks(self) -> int:
        """사용 가능한 빈 블록의 개수를 반환한다."""
        return len(self.free_list)
    
    def get_status(self) -> Dict:
        """할당자 통계 정보를 반환한다."""
        return{
            "total_blocks": self.total_blocks,
            "free_blocks": len(self.free_list),
            "used_blocks": self.total_blocks - len(self.free_list),
            "num_allocations": self.num_allocations,
            "num_frees": self.num_frees,
            "num_cow_copies": self.num_cow_copies
        }
    
    def reset_stats(self):
        """통계 카운터를 초기화한다."""
        self.num_allocations = 0
        self.num_frees = 0
        self.num_cow_copies = 0
