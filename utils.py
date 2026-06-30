"""
워크로드 생성, 메트릭 계산, 시각화를 위한 유틸리티 함수 모음.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import torch


def generate_synthetic_workload(num_requests: int, 
                                mean_prompt_len: int = 100,
                                mean_output_len: int = 50,
                                prompt_std: int = 30,
                                output_std: int = 20) -> List[Tuple[int, int]]:
    """
    다양한 sequence 길이를 가진 synthetic workload를 생성한다.
    
    인자:
        num_requests: 생성할 request 수
        mean_prompt_len: 평균 prompt 길이
        mean_output_len: 평균 output 길이
        prompt_std: prompt 길이의 표준편차
        output_std: output 길이의 표준편차
        
    반환값:
        (prompt_len, output_len) 튜플 리스트
    """
    workload = []
    
    for _ in range(num_requests):
        prompt_len = max(1, int(np.random.normal(mean_prompt_len, prompt_std)))
        output_len = max(1, int(np.random.normal(mean_output_len, output_std)))
        workload.append((prompt_len, output_len))
    
    return workload


def compute_memory_metrics(total_allocated: int, total_used: int) -> Dict:
    """
    메모리 사용률 메트릭을 계산한다.

    인자:
        total_allocated: 전체 할당 메모리(바이트)
        total_used: 실제 사용 메모리(바이트)
        
    반환값:
        메모리 메트릭이 담긴 딕셔너리
    """
    if total_allocated == 0:
        return {
            'utilization': 0.0,
            'fragmentation': 0.0,
            'wasted_memory': 0,
            'wasted_percentage': 0.0
        }
    
    wasted = total_allocated - total_used
    utilization = total_used / total_allocated
    fragmentation = wasted / total_allocated
    
    return {
        'utilization': utilization * 100,  # 백분율
        'fragmentation': fragmentation * 100,
        'wasted_memory': wasted,
        'wasted_percentage': fragmentation * 100
    }


def plot_memory_usage(timestamps: List[float], 
                     naive_memory: List[int],
                     paged_memory: List[int],
                     title: str = "시간에 따른 메모리 사용량",
                     save_path: str = None):
    """
    naive 방식과 paged 방식의 메모리 사용량을 비교해 그린다.
    
    인자:
        timestamps: 시간 지점
        naive_memory: naive 방식의 메모리 사용량(바이트)
        paged_memory: paged 방식의 메모리 사용량(바이트)
        title: 그래프 제목
        save_path: 그림을 저장할 선택적 경로
    """
    plt.figure(figsize=(10, 6))
    
    # 바이트를 MB로 변환
    naive_mb = [m / (1024 * 1024) for m in naive_memory]
    paged_mb = [m / (1024 * 1024) for m in paged_memory]
    
    plt.plot(timestamps, naive_mb, label='Naive (연속 메모리)', 
             linewidth=2, marker='o', markersize=4)
    plt.plot(timestamps, paged_mb, label='Paged (블록)', 
             linewidth=2, marker='s', markersize=4)
    
    plt.xlabel('시간 (step)', fontsize=12)
    plt.ylabel('메모리 사용량 (MB)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_throughput(batch_sizes: List[int],
                   naive_throughput: List[float],
                   paged_throughput: List[float],
                   title: str = "처리량 비교",
                   save_path: str = None):
    """
    처리량을 비교해 그린다.
    
    인자:
        batch_sizes: 테스트한 batch 크기
        naive_throughput: naive 방식의 초당 token 수
        paged_throughput: paged 방식의 초당 token 수
        title: 그래프 제목
        save_path: 그림을 저장할 선택적 경로
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(batch_sizes, naive_throughput, label='Naive', 
             linewidth=2, marker='o', markersize=8)
    plt.plot(batch_sizes, paged_throughput, label='Paged', 
             linewidth=2, marker='s', markersize=8)
    
    plt.xlabel('Batch 크기', fontsize=12)
    plt.ylabel('처리량 (tokens/sec)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_fragmentation(block_sizes: List[int],
                      fragmentation_percentages: List[float],
                      title: str = "블록 크기에 따른 메모리 단편화",
                      save_path: str = None):
    """
    블록 크기에 따른 메모리 단편화를 그린다.
    
    인자:
        block_sizes: 테스트한 블록 크기
        fragmentation_percentages: 각 크기의 단편화 비율(%)
        title: 그래프 제목
        save_path: 그림을 저장할 선택적 경로
    """
    plt.figure(figsize=(10, 6))
    
    plt.bar(range(len(block_sizes)), fragmentation_percentages, 
            color='steelblue', alpha=0.7)
    plt.xticks(range(len(block_sizes)), block_sizes)
    
    plt.xlabel('블록 크기', fontsize=12)
    plt.ylabel('내부 단편화 (%)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_swap_vs_recompute(block_sizes: List[int],
                          swap_times: List[float],
                          recompute_times: List[float],
                          title: str = "Swap과 Recompute 오버헤드 비교",
                          save_path: str = None):
    """
    swap과 recompute 시간을 비교해 그린다.
    
    인자:
        block_sizes: 테스트한 블록 크기
        swap_times: 각 크기에서 swap에 걸린 시간(ms)
        recompute_times: 각 크기에서 recompute에 걸린 시간(ms)
        title: 그래프 제목
        save_path: 그림을 저장할 선택적 경로
    """
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(block_sizes))
    width = 0.35
    
    plt.bar(x - width/2, swap_times, width, label='Swap', alpha=0.8)
    plt.bar(x + width/2, recompute_times, width, label='Recompute', alpha=0.8)
    
    plt.xlabel('블록 크기', fontsize=12)
    plt.ylabel('시간 (ms)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(x, block_sizes)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_beam_search_memory(beam_widths: List[int],
                            naive_memory: List[float],
                            paged_memory: List[float],
                            title: str = "Beam Search 메모리 사용량",
                            save_path: str = None):
    """
    서로 다른 beam width에서 beam search의 메모리 사용량을 그린다.
    
    인자:
        beam_widths: 테스트한 beam width
        naive_memory: naive 방식의 메모리 사용량(MB)
        paged_memory: COW를 사용하는 paged 방식의 메모리 사용량(MB)
        title: 그래프 제목
        save_path: 그림을 저장할 선택적 경로
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(beam_widths, naive_memory, label='Naive (전체 복사)', 
             linewidth=2, marker='o', markersize=8)
    plt.plot(beam_widths, paged_memory, label='Paged (COW)', 
             linewidth=2, marker='s', markersize=8)
    
    plt.xlabel('Beam 너비', fontsize=12)
    plt.ylabel('메모리 사용량 (MB)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_causal_mask(seq_len: int, device: str = 'cpu') -> torch.Tensor:
    """
    causal attention mask를 생성한다.
    
    인자:
        seq_len: sequence 길이
        device: 텐서를 둘 디바이스
        
    반환값:
        mask 텐서 [seq_len, seq_len]
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask


def generate_random_embeddings(batch_size: int, seq_len: int, 
                              hidden_dim: int, device: str = 'cpu') -> torch.Tensor:
    """
    테스트용 random embedding을 생성한다.
    
    인자:
        batch_size: batch 크기
        seq_len: sequence 길이
        hidden_dim: hidden dimension
        device: 텐서를 둘 디바이스
        
    반환값:
        random tensor [batch_size, seq_len, hidden_dim]
    """
    return torch.randn(batch_size, seq_len, hidden_dim, device=device)


def calculate_attention_flops(batch_size: int, seq_len: int, 
                             hidden_dim: int, num_heads: int) -> int:
    """
    attention 계산에 필요한 FLOPs를 계산한다.
    
    인자:
        batch_size: batch 크기
        seq_len: sequence 길이
        hidden_dim: hidden dimension
        num_heads: attention head 개수
        
    반환값:
        추정 FLOPs
    """
    head_dim = hidden_dim // num_heads
    
    # Q @ K^T: batch * num_heads * seq_len * seq_len * head_dim
    qk_flops = batch_size * num_heads * seq_len * seq_len * head_dim
    
    # Softmax: 행렬곱에 비해 비용이 작으므로 무시
    
    # Attn @ V: batch * num_heads * seq_len * seq_len * head_dim
    av_flops = batch_size * num_heads * seq_len * seq_len * head_dim
    
    # Projection: batch * seq_len * hidden_dim * hidden_dim (Q, K, V, Out)
    proj_flops = 4 * batch_size * seq_len * hidden_dim * hidden_dim
    
    return qk_flops + av_flops + proj_flops


class MemoryTracker:
    """
    시각화를 위해 시간에 따른 메모리 사용량을 추적한다.
    """
    
    def __init__(self):
        self.timestamps = []
        self.memory_usage = []
        self.num_tokens = []
        self.num_blocks = []
    
    def record(self, timestamp: float, memory_bytes: int, 
              num_tokens: int = 0, num_blocks: int = 0):
        """메모리 측정값을 기록한다."""
        self.timestamps.append(timestamp)
        self.memory_usage.append(memory_bytes)
        self.num_tokens.append(num_tokens)
        self.num_blocks.append(num_blocks)
    
    def get_data(self) -> Dict:
        """추적한 데이터를 반환한다."""
        return {
            'timestamps': self.timestamps,
            'memory_usage': self.memory_usage,
            'num_tokens': self.num_tokens,
            'num_blocks': self.num_blocks
        }
    
    def plot(self, title: str = "메모리 사용량", save_path: str = None):
        """추적한 메모리 사용량을 그린다."""
        plt.figure(figsize=(10, 6))
        
        memory_mb = [m / (1024 * 1024) for m in self.memory_usage]
        
        plt.plot(self.timestamps, memory_mb, linewidth=2, marker='o', markersize=4)
        plt.xlabel('시간 (step)', fontsize=12)
        plt.ylabel('메모리 사용량 (MB)', fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def format_bytes(bytes_val: int) -> str:
    """
    바이트 값을 읽기 쉬운 문자열로 변환한다.
    
    인자:
        bytes_val: 바이트 수
        
    반환값:
        포맷된 문자열 (예: "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"


def print_stats_table(stats: Dict, title: str = "통계"):
    """
    통계 테이블을 보기 좋게 출력한다.
    
    인자:
        stats: 통계 딕셔너리
        title: 테이블 제목
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    for key, value in stats.items():
        key_formatted = key.replace('_', ' ').title()
        
        if isinstance(value, float):
            if value < 1:
                print(f"{key_formatted:<40} {value:.6f}")
            else:
                print(f"{key_formatted:<40} {value:.2f}")
        elif isinstance(value, int) and key.endswith('bytes'):
            print(f"{key_formatted:<40} {format_bytes(value)}")
        else:
            print(f"{key_formatted:<40} {value}")
    
    print(f"{'='*60}\n")
