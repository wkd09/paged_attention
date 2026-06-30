"""
beam search와 parallel sampling을 위한 Copy-on-Write(COW) 디코딩 관리자.

여러 decode path 간의 KV cache fork와 공유를 관리한다.
"""

from typing import List, Dict, Set, Optional
from dataclasses import dataclass
import torch

from .kv_cache import PagedKVCache
from .allocator import BlockAllocator

@dataclass
class BeamHypothesis:
    """
    beam search에서 하나의 hypothesis를 나타낸다.

    속성:
        beam_id: 고유한 beam 식별자
        parent_id: backtracking을 위한 부모 beam ID
        token_id: 이 단계에서 생성된 토큰 ID
        score: 누적 log probability
        kv_cache: 이 hypothesis의 KV cache
        is_finished: beam이 완료되었는지 여부
    """
    beam_id: int
    parent_id: Optional[int]
    token_id: int
    score: float
    kv_cache: PagedKVCache
    is_finished: bool = False

class DecodingManager:
    """
    beam search와 parallel sampling을 위한 copy-on-write 의미를 관리한다.

    cache fork, 공유 block 추적, 효율적인 메모리 사용을 처리한다.
    """
    def __init__(self, allocator: BlockAllocator, block_size:int, hidden_dim:int):
        """
        decoding manager를 초기화한다.

        인자:
            allocator: 블록 할당자
            block_size: KV 블록의 크기
            hidden_dim: 모델의 hidden dimension
        """
        self.allocator = allocator
        self.block_size = block_size
        self.hidden_dim = hidden_dim

        # beam 추적
        self.beams: Dict[int, BeamHypothesis] = {}
        self.next_beam_id = 0

        # 통계
        self.num_forks = 0
        self.num_cow_copies = 0

    def initialize_beam(self, initial_cache: PagedKVCache,
                        initial_token: int = 0) -> int:
        """
        기존 cache를 사용해 root beam을 초기화한다.

        인자:
            initial_cache: prompt의 KV cache
            initial_token: 초기 token ID

        반환값:
            Beam ID
        """
        beam_id = self.next_beam_id
        self.next_beam_id += 1

        beam = BeamHypothesis(
            beam_id=beam_id,
            parent_id=None,
            token_id=initial_token,
            score=0.0,
            kv_cache=initial_cache
        )
        self.beams[beam_id] = beam
        return beam_id
    
    def fork_beam(self, parent_beam_id: int, token_id: int,
                  score: float) -> int:
        """
        parent beam에서 새로운 beam을 fork한다. (copy-on-write)

        인자:
            parent_beam_id: fork할 부모 beam
            token_id: 이 beam의 새로운 token
            score: beam 점수

        반환값:
            새로운 beam ID
        """
        if parent_beam_id not in self.beams:
            raise ValueError(f"부모 beam {parent_beam_id}을 찾을 수 없습니다.")
        
        parent_beam = self.beams[parent_beam_id]

        # KV cache를 분기한다. (참조 카운팅을 통해 블록을 공유)
        new_cache = parent_beam.kv_cache.fork()

        new_beam_id = self.next_beam_id
        self.next_beam_id += 1

        new_beam = BeamHypothesis(
            beam_id=new_beam_id,
            parent_id=parent_beam_id,
            token_id=token_id,
            score=score,
            kv_cache=new_cache
        )

        self.beams[new_beam_id] = new_beam
        self.num_forks += 1
        return new_beam_id

    def append_token(self, beam_id: int, key_vec: torch.Tensor,
                     val_vec: torch.Tensor):
        """
        beam의 cache에 token을 append한다. (필요하면 COW 수행)

        인자:
            beam_id: append할 대상 beam
            key_vec: Key 벡터 [hidden_dim]
            val_vec: Value 벡터 [hidden_dim]
        """
        if beam_id not in self.beams:
            raise ValueError(f"Beam {beam_id}을 찾을 수 없습니다.")
        
        beam = self.beams[beam_id]

        # COW가 필요한지 확인
        old_cow_copies = self.allocator.num_cow_copies
        beam.kv_cache.cow_append(key_vec, val_vec)

        if self.allocator.num_cow_copies > old_cow_copies:
            self.num_cow_copies += 1

    def finish_beam(self, beam_id: int):
        """beam을 완료 상태로 표시한다."""
        if beam_id in self.beams:
            self.beams[beam_id].is_finished = True

    def free_beam(self, beam_id: int):
        """beam과 해당 beam이 사용하는 리소스를 해제한다."""
        if beam_id in self.beams:
            beam = self.beams[beam_id]
            beam.kv_cache.free_all()
            del self.beams[beam_id]

    def get_active_beams(self) -> List[BeamHypothesis]:
        """활성 상태인 모든 beam을 반환한다. (완료되지 않은 beam)"""
        return [b for b in self.beams.values() if not b.is_finished]
    
    def get_best_beam(self) -> Optional[BeamHypothesis]:
        """가장 높은 score를 가진 beam을 반환한다."""
        active = self.get_active_beams()
        if not active:
            return None
        return max(active, key=lambda b: b.score)
    
    def get_stats(self) -> Dict:
        "decoding 통계 반환"
        return {
            "num_beams": len(self.beams),
            "active_beams": len(self.get_active_beams()),
            "num_forks": self.num_forks,
            "num_cow_copies": self.num_cow_copies
        }
    
class ParallelSamplingManager:
    """
    parallel sampling을 관리한다. (같은 prompt에서 여러 개의 독립적인 sample 생성)

    beam search와 비슷하지만, sample들은 서로 독립적이다. (pruning 없음)
    """

    def __init__(self, allocator: BlockAllocator, block_size: int, hidden_dim: int):
        self.allocator = allocator
        self.block_size = block_size
        self.hidden_dim = hidden_dim

        self.samples: Dict[int, PagedKVCache] = {}
        self.next_sample_id = 0

        self.num_forks = 0

    def create_samples(self, prompt_cache:PagedKVCache,
                       num_samples: int) -> List[int]:
        """
        prompt cache를 공유하는 여러 개의 sample을 생성한다.

        인자:
            prompt_cache: 공유할 prompt KV cache
            num_samples: 생성할 sample 개수

        반환값:
            sample ID 리스트
        """
        sample_ids = []

        for _ in range(num_samples):
            sample_id = self.next_sample_id
            self.next_sample_id += 1

            # 이 sample을 위한 cache 분기
            sample_cache = prompt_cache.fork()
            self.samples[sample_id] = sample_cache
            sample_ids.append(sample_id)
            self.num_forks += 1

        return sample_ids
    
    def get_sample_cache(self, sample_id: int) -> PagedKVCache:
        """sample의 cache를 가져온다."""
        if sample_id not in self.samples:
            raise ValueError(f"Sample {sample_id}을 찾을 수 없습니다.")
        return self.samples[sample_id]
    
    def free_sample(self, sample_id):
        """sample이 사용하는 리소스를 해제한다."""
        if sample_id in self.samples:
            self.samples[sample_id].free_all()
            del self.samples[sample_id]

    def free_all(self):
        """모든 sample을 해제한다."""
        for sample_id in list(self.samples.keys()):
            self.free_sample(sample_id)

    def get_stats(self) -> Dict:
        """sampling 통계 정보를 반환한다."""
        return {
            "num_samples": len(self.samples),
            "num_forks": self.num_forks
        }
