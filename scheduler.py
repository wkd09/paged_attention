"""
여러 sequence를 관리하기 위한 간단한 batch scheduler.

테스트와 벤치마킹을 위해 iteration-level scheduling을 흉내 낸다.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import torch

from .kv_cache import PagedKVCache
from .allocator import BlockAllocator

@dataclass
class SequenceRequest:
    """
    단일 sequence 생성 요청을 나타낸다.

    속성:
        seq_id: 고유한 sequence 식별자
        prompt_tokens: 초기 prompt token ID들
        max_tokens: 생성할 최대 token 수
        num_samples: parallel sampling을 위한 sample 개수
        kv_cache: 연결된 KV cache
        generated_tokens: 지금까지 생성된 token들
        is_finished: 생성이 완료되었는지 여부
    """
    seq_id: int
    prompt_tokens: List[int]
    max_tokens: int
    num_samples: int = 1
    kv_cache: Optional[PagedKVCache] = None
    generated_tokens: List[int] = None
    is_finished: bool = False

    def __post__init__(self):
        if self.generated_tokens is None:
            self.generated_tokens = []

class SimpleScheduler:
    """
    batch inference를 위한 FCFS(First-Come-First-Serve) scheduler.

    여러 sequence 요청을 관리하고 실행되도록 scheduling한다.
    request forking을 통해 parallel sampling과 beam search를 지원한다.
    """
    def __init__(self, allocator: BlockAllocator, max_batch_size: int = 8):
        """
        scheduler를 초기화한다.

        인자:
            allocator: KV cache에 사용할 블록 할당자
            max_batch_size: batch에 포함할 수 있는 최대 sequence 수
        """
        self.allocator = allocator
        self.max_batch_size = max_batch_size

        # 활성 요청들
        self.requests: Dict[int, SequenceRequest] = {}
        self.next_seq_id = 0

        # 대기 큐
        self.waiting_queue: List[SequenceRequest] = []

        # 통계
        self.total_requests = 0
        self.completed_requests = 0

    def add_request(self, prompt_tokens: List[int], max_token: int,
                    num_samples: int = 1) -> int:
        """
        새로운 sequence 요청을 추가한다.

        인자:
            prompt_tokens: prompt token ID들
            max_tokens: 생성할 최대 token 수
            num_samples: parallel sampling의 sample 개수

        반환값:
            sequence ID
        """
        seq_id = self.next_seq_id
        self.next_seq_id += 1

        request = SequenceRequest(
            seq_id=seq_id,
            prompt_tokens=prompt_tokens,
            max_tokens=max_token,
            num_samples=num_samples
        )

        self.waiting_queue.append(request)
        self.total_requests += 1

        return seq_id
    
    def schedule_batch(self) -> List[SequenceRequest]:
        """
        다음 batch로 처리할 sequence들을 scheduling한다.

        반환값:
            이번 iteration에서 처리할 요청 리스트
        """
        # 공간이 있으면 대기 중인 요청을 active 상태로 이동
        while (len(self.requests) < self.max_batch_size and
               self.waiting_queue and
               self.allocator.get_num_free_blocks()>0):
            request = self.waiting_queue.pop(0)
            self.requests[request.seq_id] = request

        # 모든 활성 요청들 반환
        return list(self.requests.values())
    
    def mark_finished(self, seq_id: int):
        """sequence를 완료 상태로 표시하고, 해당 리소스를 해제한다."""
        if seq_id in self.requests:
            request = self.requests[seq_id]
            if request.kv_cache is not None:
                request.kv_cache.free_all()
            request.is_finished = True
            del self.requests[seq_id]
            self.completed_requests += 1

    def get_stats(self) -> Dict:
        """scheduler 통계 정보를 반환한다."""
        return {
            "active_requests": len(self.requests),
            "waiting_requests": len(self.waiting_queue),
            "total_requests": self.total_requests,
            "completed_requests": self.completed_requests,
            "free_blocks": self.allocator.get_num_free_blocks()
        }
    
    def is_empty(self) -> bool:
        """scheduler에 대기 중인 작업이 없는지 확인한다."""
        return len(self.requests) == 0 and len(self.waiting_queue) == 0
    

class BatchProcessor:
    """
    attention layer를 통해 sequence batch를 처리한다.
    prompt 단계와 generation 단계를 처리한다.
    """
    def __init__(self, block_size: int, hidden_dim: int):
        self.block_size = block_size
        self.hidden_dim = hidden_dim

    def process_prompt_phase(self, request: SequenceRequest,
                             allocator: BlockAllocator) -> PagedKVCache:
        """
        prompt token을 처리하고 KV cache를 초기화한다.

        인자:
            request: sequence 요청
            allocator: 블록 할당자

        반환값:
            초기화된 KV cache
        """
        cache = PagedKVCache(
            block_size=self.block_size,
            hidden_dim=self.hidden_dim,
            allocator=allocator,
            seq_id=request.seq_id
        )

        # prompt token을 cache에 추가하는 것을 시뮬레이션
        # 실제 구현에서는 attention을 통과시켜 처리한다
        num_prompt_tokens = len(request.prompt_tokens)

        for _ in range(num_prompt_tokens):
            # 더미 kv 벡터 생성
            k = torch.randn(self.hidden_dim)
            v = torch.randn(self.hidden_dim)
            cache.append_token_kv(k, v)

        request.kv_cache = cache
        return cache
    
    def process_generation_step(self, request: SequenceRequest,
                                new_token_id: int):
        """
        generation step 하나를 처리한다. (새 token을 cache에 append)

        인자:
            request: sequence 요청
            new_token_id: 생성된 token ID
        """
        if request.kv_cache is None:
            raise ValueError("KV cache가 초기화되지 않았습니다.")
        
        # 새로운 token의 K/V를 추가하는 것을 시뮬레이션
        k = torch.randn(self.hidden_dim)
        v = torch.randn(self.hidden_dim)
        request.kv_cache.append_token_kv(k, v)

        request.generated_tokens.append(new_token_id)
