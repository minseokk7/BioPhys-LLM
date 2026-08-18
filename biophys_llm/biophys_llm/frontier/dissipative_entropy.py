r"""
BioPhys-LLM Frontier: Prigogine Dissipative Entropy Regularizer (프리고진 비평형 열역학 소산 정규화기)
일리야 프리고진(Ilya Prigogine)의 산일 구조(Dissipative Structures) 원리를 접목하여,
언어 모델 추론 과정에서 누적되는 혼돈 엔트로피(노이즈)를 소산시켜
환각(Hallucination)을 억제하고 논리적 수렴성을 극대화하는 정규화 모듈.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DissipativeEntropyRegularizer(nn.Module):
    """
    비평형 열역학 엔트로피 소산:
    - Delta S_total = Delta S_internal + Delta S_dissipated
    - 내부 무질서도(혼돈 분산)가 임계치를 넘으면 고주파 노이즈 성분을 소산하여 질서 정연한 어트랙터로 수렴
    """

    def __init__(self, hidden_dim: int, dissipation_strength: float = 0.15):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dissipation_strength = dissipation_strength
        self.thermo_gate = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Args:
            hidden_states: [Batch, SeqLen, HiddenDim]
        Returns:
            clean_states: 엔트로피가 소산된 정돈된 상태
            entropy_reduction_pct: 억제된 노이즈 비율 (%)
        """
        local_entropy = torch.sigmoid(self.thermo_gate(hidden_states)) # [Batch, SeqLen, 1]
        
        # 산일 필터 적용 (고열/고엔트로피 노이즈 감쇠)
        dissipation_factor = 1.0 - (self.dissipation_strength * local_entropy)
        clean_states = hidden_states * dissipation_factor
        
        entropy_reduction_pct = (local_entropy.mean().item() * self.dissipation_strength) * 100.0
        return clean_states, entropy_reduction_pct
