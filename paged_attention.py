"""
PagedAttention: non-contiguous KV cache에 대한 blockwise attention.

고정 크기 block으로 나뉜 KV cache 위에서 동작하는 attention을 구현한다.
수치적으로 안정적인 blockwise softmax 계산을 사용한다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

from .kv_cache import PagedKVCache

class PagedAttention(nn.Module):
    """
    paged KV cache를 지원하는 multi-head attention.

    메모리에 non-contiguous하게 저장된 KV block들에 대해 attention을 계산한다.
    수치적 안정성을 위해 blockwise softmax를 사용한다.
    """
    def __init__(self, hidden_dim: int, num_heads: int, block_size: int,
                 dropout: float = 0.0):
        """
        PagedAttention을 초기화한다.

        Args:
            hidden_dim: 모델의 hidden dimension
            num_heads: attention head 개수
            block_size: KV cache block의 크기
            dropout: attention dropout 확률
        """
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim({hidden_dim})은 num_heads({num_heads})로 나누어떨어져야 합니다.")
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.block_size = block_size
        self.dropout = dropout
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Projection layer들
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None

    def _split_heads(self, x:torch.Tensor) -> torch.Tensor:
        """
        hidden dim을 여러 head로 나눈다.

        Args:
            x: [batch, seq_len, hidden_dim]

        Returns:
            [batch, num_heads, seq_len, head_dim]
        """
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)
    
    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        여러 head를 다시 hidden dim으로 합친다.

        Args:
            x: [batch, num_heads, seq_len, head_dim]

        Returns:
            [batch, seq_len, hidden_dim]
        """
        batch_size, _, seq_len, _ = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, seq_len, self.hidden_dim)
    
    def forward_vanilla(self, query: torch.Tensor, key: torch.Tensor,
                        value: torch.Tensor, mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Vanilla attention. (baseline 비교용)

        Args:
            query: [batch, q_len, hidden_dim]
            key: [batch, kv_len, hidden_dim]
            value: [batch, kv_len, hidden_dim]
            mask: 선택적인 attention mask

        Returns:
            [batch, q_len, hidden_dim]
        """
        batch_size = query.shape[0]

        # projection을 적용하고 head로 분리
        Q = self._split_heads(self.q_proj(query))
        K = self._split_heads(self.k_proj(key))
        V = self._split_heads(self.v_proj(value))

        # attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale # [batch, heads, q_len, kv_len]

        if mask is not None:
            scores = torch.masked_fill(mask==0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)

        if self.dropout_layer is not None:
            attn_weights = self.dropout_layer(attn_weights)

        # attention을 value에 적용
        output = torch.matmul(attn_weights, V) # [batch, heads, q_len, head_dim]
        output = self._merge_heads(output)     # [batch, q_len, hidden_dim]
        output = self.out_proj(output)

        return output
    
    def forward_paged(self, query: torch.Tensor, kv_cache: PagedKVCache,
                      mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        blockwise KV cache에 대해 PagedAttention을 수행한다.

        Args:
            query: [batch, q_len, hidden_dim]
            kv_cache: block들을 가진 PagedKVCache
            mask: 선택적인 attention mask

        Returns:
            [batch, q_len, hidden_dim]
        """

        batch_size, q_len, _ = query.shape

        # query를 projection
        Q = self._split_heads(self.q_proj(query))  # [batch, heads, q_len, head_dim]

        # cache에서 K/V block들을 읽기
        K_blocks_list, V_block_list = kv_cache.read_blocks_for_attention()

        if not K_blocks_list:
            # cache가 비어 있으면 zero tensor 반환
            return torch.zeros_like(query)
        
        # 수치적으로 안정적인 방식으로 block 단위 attention 계산
        output = self._blockwise_attention(Q, K_blocks_list, V_block_list, mask)

        output = self._merge_heads(output)
        output = self.out_proj(output)

        return output
    
    def _blockwise_attention(self, Q: torch.Tensor, K_blocks:list,
                             V_blocks: list, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        수치적으로 안정적인 softmax를 사용해 KV block들에 대해 attention을 계산한다.

        연결된 block 전체에 대한 softmax를 계산하기 위해,
        running max와 running sum을 추적하는 trick을 사용한다.

        Args:
            Q: [batch, heads, q_len, head_dim]
            K_blocks: [block_size_i, hidden_dim] 형태의 tensor 리스트
            V_blocks: [block_size_i, hidden_dim] 형태의 tensor 리스트
            mask: 선택적인 mask

        Returns:
            [batch, heads, q_len, head_dim]
        """
        batch_size, num_heads,q_len, head_dim = Q.shape
        device = Q.device

        # 수치적으로 안정적인 softmax를 위한 누적값 초기화
        max_score = torch.full((batch_size, num_heads, q_len, 1),
                               float('-inf'), device=device)
        sum_exp = torch.zeros((batch_size, num_heads, q_len, 1), device=device)
        weighted_values = torch.zeros((batch_size, num_heads, q_len, head_dim),
                                      device=device)
        
        current_pos = 0

        for block_idx, (K_block, V_block) in enumerate(zip(K_blocks, V_blocks)):
            # K_block: [block_len, hidden_dim], V_block: [block_len, hidden_dim]
            block_len = K_block.shape[0]

            # K_block과 V_block은 이미 projection이 적용된 tensor라고 가정한다.
            # 형태는 [block_len, hidden_dim]이며, 바로 head로 분리한다.
            K_proj = K_block.to(device).view(block_len, num_heads, head_dim).permute(1, 0, 2).unsqueeze(0)
            V_proj = V_block.to(device).view(block_len, num_heads, head_dim).permute(1, 0, 2).unsqueeze(0)
            K_proj = K_proj.expand(batch_size, -1, -1, -1)
            V_proj = V_proj.expand(batch_size, -1, -1, -1)

            # 이 block에 대한 score 계산
            block_scores = torch.matmul(Q, K_proj.transpose(-2, -1)) * self.scale
            # [batch, heads, q_len, block_len]

            if mask is not None:
                block_mask = mask[:, :, :, current_pos:current_pos + block_len]
                block_scores = block_scores.masked_fill(block_mask == 0, float('-inf'))

            # running max 업데이트
            block_max = block_scores.max(dim=-1, keepdim=True)[0] # [batch, heads, q_len, 1]
            new_max = torch.maximum(max_score, block_max)

            # 새로운 max에 맞게 이전 sum_exp 조정
            sum_exp = sum_exp * torch.exp(max_score - new_max)

            # 현재 block의 exp 값 계산
            block_exp = torch.exp(block_scores - new_max)

            # sum 업데이트
            sum_exp = sum_exp + block_exp.sum(dim=-1, keepdim=True)

            # weughted values 업데이트
            weighted_values = weighted_values * torch.exp(max_score - new_max)
            weighted_values = weighted_values + torch.matmul(block_exp, V_proj)

            # max 업데이트
            max_score = new_max
            current_pos += block_len

        # 최종 정규화
        output = weighted_values / sum_exp

        if self.dropout_layer is not None and self.training:
            # 참고: 여기서는 attention weight가 아니라 출력에 dropout을 적용한다.
            # 실제 attention dropout을 구현하려면 attention weight를 저장해야 한다.
            output = self.dropout_layer(output)

        return output
    
    def forward(self, query: torch.Tensor,
                key: Optional[torch.Tensor] = None,
                value: Optional[torch.Tensor] = None,
                kv_cache: Optional[PagedKVCache] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass - vanilla attention과 paged attention 중 적절한 방식을 자동으로 선택한다.

        Args:
            query: [batch, q_len, hidden_dim]
            key: vanilla mode에서 사용하는 선택적인 [batch, kv_len, hidden_dim]
            value: vanilla mode에서 사용하는 선택적인 [batch, kv_len, hidden_dim]
            kv_cache: paged mode에서 사용하는 선택적인 PagedKVCache
            mask: 선택적인 attention mask

        Returns:
            [batch, q_len, hidden_dim]
        """
        if kv_cache is not None:
            return self.forward_paged(query, kv_cache, mask)
        elif key is not None and value is not None:
            return self.forward_vanilla(query, key, value, mask)
        else:
            raise ValueError("kv_cache나 key, value가 제공되어야함")
        
class VanillaAttention(nn.Module):
    """
    baseline 비교를 위한 표준 multi-head attention.

    KV cache를 연속된(contiguous) 메모리에 저장한다.
    """
    def __init__(self, hidden_dim:int, num_heads: int, dropout: float = 0.0):
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim은 num_heads로 나누어떨어져야 합니다.")
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None

        # 연속된 KV cache
        self.k_cache = []
        self.v_cache = []

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)
    
    def _merge_heads(self, x:torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.hidden_dim)
    
    def append_kv(self, key:torch.Tensor, value:torch.Tensor):
        """K/V를 cache에 추가"""
        self.k_cache.append(key)
        self.v_cache.append(value)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """기본적인 attention의 forward"""
        Q = self._split_heads(self.q_proj(query))
        K = self._split_heads(self.k_proj(key))
        V = self._split_heads(self.v_proj(value))

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)

        if self.dropout_layer is not None:
            attn_weights = self.dropout_layer(attn_weights)

        output = torch.matmul(attn_weights, V)
        output = self._merge_heads(output)
        output = self.out_proj(output)

        return output
    
    def get_memory_usage(self) -> int:
        """bytes로 메모리 사용량 반환"""
        if not self.k_cache:
            return 0
        total_tokens = sum(k.shape[1] for k in self.k_cache)
        bytes_per_element = 4 # float32
        return total_tokens * self.hidden_dim * 2 * bytes_per_element
    
