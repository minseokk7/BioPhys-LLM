r"""
BioPhys-LLM Pure Science: Plasma Collisional Damping Stabilizer (플라즈마 공명 충돌 감쇠 안정화기)
플라즈마 물리학(Plasma Physics)의 공명 충돌 감쇠(Collisional Damping & Resonant Dissipation) 이론을 접목하여,
언어 모델의 극단적 아웃라이어나 활성화 폭발(Activation Explosion)이 발생할 때,
이를 인위적인 공명 충돌층(Resonant Collision Layer)에 충돌시켜 과도한 폭주 에너지를 0ms 지연으로 무해하게 소산(Dissipation)시키는 모듈.
"""

from typing import Tuple
import torch
import torch.nn as nn


class CollisionalDampingStabilizer(nn.Module):
    """
    플라즈마 공명 충돌 감쇠기:
    - 텐서의 국소 충돌 주파수(\nu_c) 모델링
    - 임계치를 초과하는 폭주 에너지 스파이크를 공명 충돌층에 충돌시켜 무해한 정상 범위로 감쇠 흡수
    """

    def __init__(self, hidden_dim: int = 5120, damping_threshold: float = 3.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.damping_threshold = damping_threshold

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Args:
            x: [Batch, SeqLen, HiddenDim]
        Returns:
            damped_x: 충돌 감쇠를 통해 안정화된 텐서
            spike_dissipation_pct: 충돌 흡수된 폭주 에너지 비율 (%)
        """
        # 1. 텐서의 국소 진폭 및 충돌 감쇠 계수 계산
        abs_x = torch.abs(x)
        excess_spikes = torch.relu(abs_x - self.damping_threshold)
        
        # 2. 공명 충돌 감쇠 공식: x_damped = x / (1 + \nu_c * excess)
        collision_frequency = 0.5
        damping_factor = 1.0 + collision_frequency * (excess_spikes / (self.damping_threshold + 1e-8))
        damped_x = x / damping_factor
        
        # 3. 흡수된 폭주 에너지 비율 산출
        spike_dissipation_pct = (excess_spikes.sum() / (abs_x.sum() + 1e-8)).item() * 100.0
        return damped_x, spike_dissipation_pct
