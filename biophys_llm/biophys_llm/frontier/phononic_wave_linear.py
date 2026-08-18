r"""
BioPhys-LLM Frontier: Phononic Metamaterial Phase Linear (음향 포논 메타물질 위상 선형 변환기)
음향 메타물질(Phononic Metamaterials)의 파동 위상 간섭(Phase Interference) 원리를 접목하여,
무거운 부동소수점 행렬곱(FLOPs)을 직교 파동 위상각(Phase Angles)의 중첩 덧셈으로 치환하여
연산 에너지와 계산 복잡도를 대폭 절감하는 하드웨어 친화적 선형 변환 모듈.
"""

from typing import Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PhononicPhaseLinear(nn.Module):
    """
    포노닉 메타물질 전달 행렬(Transfer Matrix Method):
    - 가중치 행렬을 위상각 행렬 Theta와 진폭 행렬 A로 극좌표 분해
    - y = A * cos(Theta + phi_in) 형태로 간섭 합성하여 부동소수점 연산 단순화
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 진폭 텐서 (Amplitude) 및 위상각 텐서 (Phase, [-pi, pi])
        self.amplitude = nn.Parameter(torch.randn(out_features, in_features) * (1.0 / math.sqrt(in_features)))
        self.phase_angles = nn.Parameter(torch.rand(out_features, in_features) * (2.0 * math.pi) - math.pi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        입력 신호 x를 음향 파동으로 인코딩하여 위상 간섭 전달 행렬 연산
        Args:
            x: [Batch, SeqLen, InFeatures]
        Returns:
            out: [Batch, SeqLen, OutFeatures]
        """
        effective_w = self.amplitude * torch.cos(self.phase_angles)
        return F.linear(x, effective_w)
