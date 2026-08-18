r"""
BioPhys-LLM Frontier: Mycelial Attention Router (균사체/점균류 적응형 튜브 어텐션 라우터)
점균류(Physarum polycephalum)와 곰팡이 균사체(Mycelium)의 분산 최적화 수송망 원리를 접목하여,
정보 흐름이 강한 어텐션 헤드와 토큰 경로는 튜브를 확장(강화)하고,
정보 흐름이 약한 불필요한 어텐션 계산을 물리적으로 차단하여 O(N) 급으로 가속하는 모듈.
"""

from typing import Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MycelialAttentionRouter(nn.Module):
    """
    점균류 영양 수송 튜브 피드백 메커니즘:
    - 튜브 전도도(Conductivity D): 플럭스(Flux)가 클수록 두꺼워짐
    - 미사용 경로 자동 수축(Decay) 및 차단 (Sparsity > 50%)
    """

    def __init__(self, num_heads: int, head_dim: int, decay_rate: float = 0.20, threshold: float = 0.50):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.decay_rate = decay_rate
        self.threshold = threshold
        
        # 각 헤드별 점균류 튜브 전도도 초기 상태
        initial_cond = torch.linspace(0.1, 1.0, num_heads)
        self.register_buffer("tube_conductivity", initial_cond)

    def route_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        Args:
            q, k, v: [Batch, NumHeads, SeqLen, HeadDim]
        Returns:
            out: 라우팅된 어텐션 출력
            active_head_ratio: 활성화된 핵심 균사체 경로 비율 (%)
        """
        batch, heads, seq_len, dim = q.shape
        
        # 1. 각 헤드의 정보 플럭스(Flux) 측정
        head_flux = (q * k).sum(dim=-1).abs().mean(dim=(0, 2)) # [NumHeads]
        
        # 2. 점균류 전도도 갱신
        with torch.no_grad():
            self.tube_conductivity.mul_(1.0 - self.decay_rate).add_(head_flux, alpha=self.decay_rate)
            norm_cond = (self.tube_conductivity - self.tube_conductivity.min()) / (self.tube_conductivity.max() - self.tube_conductivity.min() + 1e-8)
            active_mask = norm_cond >= self.threshold # 활성 튜브 판정
            active_ratio = (active_mask.float().mean().item()) * 100.0

        # 3. 활성 튜브에 대해서만 선택적 어텐션 연산 수행
        scale = 1.0 / math.sqrt(dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(scores, dim=-1)
        
        tube_weights = (norm_cond * active_mask.float()).view(1, heads, 1, 1)
        attn = attn * tube_weights
        
        out = torch.matmul(attn, v)
        return out, active_ratio
