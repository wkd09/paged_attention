# Paged Attention 구현

PagedAttention 논문의 핵심 아이디어인 paged KV cache를 간단한 PyTorch 코드로 구현한 실험용 저장소입니다. 기존 contiguous KV cache 방식과 비교할 수 있도록 vanilla attention baseline, block allocator, logical-to-physical block table, copy-on-write 기반 cache sharing을 함께 포함합니다.

## 핵심 아이디어

기존 KV cache는 sequence마다 연속된 메모리를 크게 잡아두는 방식으로 구현되는 경우가 많습니다. 이 방식은 구현은 단순하지만, sequence 길이가 서로 다르거나 beam search/parallel sampling처럼 같은 prompt에서 여러 decode path가 갈라질 때 메모리 낭비가 커집니다.

PagedAttention은 KV cache를 고정 크기 block 단위로 나누고, 각 sequence가 logical block table을 통해 physical block을 참조하게 만듭니다. OS의 virtual memory paging과 비슷하게, 실제 필요한 block만 할당하고 여러 sequence가 같은 prefix block을 공유할 수 있습니다.

이 저장소에서는 다음을 구현합니다.

- `BlockAllocator`: physical KV block pool과 reference count 관리
- `PagedKVCache`: sequence별 logical-to-physical block table
- `PagedAttention`: non-contiguous KV block에 대한 blockwise attention
- `VanillaAttention`: 비교용 contiguous KV cache baseline
- `DecodingManager`, `ParallelSamplingManager`: fork와 copy-on-write 시뮬레이션
- `SwapManager`, `RecomputeManager`: swap/recompute 전략 시뮬레이션
- `compare_paged_vs_vanilla.py`: vanilla 방식과 paged 방식을 비교하는 실행 스크립트

## 파일 구조

```text
.
├── allocator.py                  # Physical block pool, free list, refcount, COW copy
├── kv_cache.py                   # Paged KV cache와 block table
├── paged_attention.py            # Vanilla attention + paged attention
├── decoding.py                   # Beam/parallel sampling용 cache fork와 COW
├── scheduler.py                  # 간단한 batch scheduler 시뮬레이션
├── swap_recompute.py             # Swap/recompute memory strategy 시뮬레이션
├── utils.py                      # workload, metric, plotting helper
└── compare_paged_vs_vanilla.py   # 비교 실험 스크립트
```

## 실행 방법

필요한 주요 패키지는 `torch`, `numpy`, `matplotlib`입니다.

```bash
python compare_paged_vs_vanilla.py
```

Anaconda 환경을 쓰는 경우 예시는 다음과 같습니다.

```bash
/opt/anaconda3/bin/python compare_paged_vs_vanilla.py
```

## 비교 결과 예시

현재 비교 스크립트는 네 가지를 확인합니다.

1. vanilla attention과 paged attention의 출력 동일성
2. 고정 길이 reserved KV cache 대비 paged KV cache의 메모리 절약
3. parallel sampling에서 copy-on-write block sharing 효과
4. 현재 Python 구현의 CPU runtime sanity check

예시 출력:

```text
[1] Output equivalence
    kv_len=97, block_size=16
    max_abs_diff : 0.00000004
    mean_abs_diff: 0.00000001

[2] KV-cache memory
    sequences              : [31, 64, 127, 255, 513, 777, 1024]
    hidden_dim/block_size  : 4096/16
    vanilla fixed reserve  :   448.00 MiB
    actual used tokens     :    87.22 MiB
    paged allocated        :    88.00 MiB
    paged internal waste   :     0.78 MiB
    saving vs fixed reserve:   80.36%

[3] Parallel sampling / COW sharing
    samples                : 8
    prompt/generated tokens: 513/32
    vanilla full copies    :   136.25 MiB
    paged COW blocks       :    28.50 MiB
    used physical blocks   : 57
    COW copies             : 8
    saving vs full copies  :   79.08%

[4] CPU runtime sanity check
    kv_len/runs            : 512/20
    vanilla dense matmul   :    0.475 ms
    paged python block loop:    3.300 ms
    note: this educational Python version optimizes memory behavior, not kernel speed.
```

## 결과 해석

`Output equivalence`에서 max/mean absolute difference가 매우 작으므로, paged attention이 vanilla attention과 수치적으로 같은 결과를 내는 것을 확인할 수 있습니다. 비교 스크립트에서는 paged cache에 이미 projection된 K/V를 넣어 `forward_vanilla`와 `forward_paged`가 같은 Q/K/V projection 조건에서 비교되도록 맞췄습니다.

`KV-cache memory`에서는 vanilla 방식이 각 sequence에 최대 길이만큼 메모리를 reserve한다고 가정합니다. 반면 paged 방식은 필요한 block만 할당하므로, sequence 길이가 들쭉날쭉할 때 낭비가 크게 줄어듭니다. 예시에서는 약 `80.36%`의 reserved memory 절약이 나타납니다.

`Parallel sampling / COW sharing`에서는 같은 prompt에서 여러 sample을 생성할 때의 차이를 보여줍니다. vanilla 방식은 각 sample이 prompt KV cache까지 모두 복사한다고 가정하지만, paged 방식은 prefix block을 공유하고 새로 생성되는 token block만 추가합니다. prompt의 마지막 block이 partial block이면 첫 append에서 copy-on-write가 발생합니다.

`CPU runtime`은 성능 우위를 보여주기 위한 benchmark가 아닙니다. 이 저장소의 paged attention은 Python loop로 blockwise softmax를 계산하므로, CPU에서는 dense matmul 기반 vanilla attention보다 느릴 수 있습니다. 실제 PagedAttention의 serving 성능 이점은 optimized CUDA kernel과 continuous batching이 결합될 때 드러납니다.

## 간단한 사용 예시

```python
import torch

from paged_attention.allocator import BlockAllocator
from paged_attention.kv_cache import PagedKVCache
from paged_attention.paged_attention import PagedAttention

hidden_dim = 128
num_heads = 8
block_size = 16

allocator = BlockAllocator(total_blocks=128, block_size=block_size, hidden_dim=hidden_dim)
cache = PagedKVCache(block_size=block_size, hidden_dim=hidden_dim, allocator=allocator)
attention = PagedAttention(hidden_dim=hidden_dim, num_heads=num_heads, block_size=block_size)

keys = torch.randn(32, hidden_dim)
values = torch.randn(32, hidden_dim)
cache.append_token_kv_batch(keys, values)

query = torch.randn(1, 1, hidden_dim)
output = attention(query, kv_cache=cache)
print(output.shape)
```

주의: `PagedAttention.forward_paged`는 cache 안의 K/V가 이미 projection된 값이라고 가정합니다. 실제 모델에 연결하려면 token hidden state에서 `k_proj`, `v_proj`를 적용한 결과를 cache에 저장하는 흐름으로 맞춰야 합니다.

## 구현 범위와 한계

이 코드는 논문 아이디어를 이해하기 위한 educational implementation입니다.

- CUDA kernel 최적화는 포함하지 않습니다.
- 실제 tokenizer/model forward loop는 포함하지 않습니다.
- paged attention의 kernel-level memory coalescing 최적화는 구현하지 않습니다.
- scheduler는 실제 serving system이 아니라 iteration-level scheduling 개념을 단순화한 시뮬레이션입니다.
- swap/recompute는 실제 GPU 메모리 압박 상황을 재현하기보다 비용 모델을 설명하기 위한 구조입니다.

## 다음 개선 아이디어

- causal mask와 batch별 sequence length 처리를 더 엄밀하게 확장
- prefill 단계와 decode 단계 분리
- CUDA/Triton kernel 기반 blockwise attention 실험
- continuous batching scheduler 강화
- 실제 Hugging Face 모델의 KV cache 형식과 연결
- benchmark 결과를 CSV/plot으로 저장하는 옵션 추가
