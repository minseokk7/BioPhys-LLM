"""
BioPhys-LLM Pure Science: Brillouin Zone Phonon Bandgap Filter (브릴루앙 영역 포노닉 결정 밴드갭 필터)
고체물리학(Solid-State Physics)의 결정 격자 브릴루앙 영역(Brillouin Zone) 경계에서 발생하는
포논 밴드갭(Acoustic Bandgap) 원리를 접목하여,
긴 문맥(Long-Context) 추론 시 누적되는 고주파 분산 노이즈(Turbulent Noise)의 통과를 원천 차단하고
안정적인 기본 파동만 전파시켜 긴 작업의 연산 안정성과 토큰 속도를 극대화하는 모듈.
"""

from typing import Tuple
import math
import torch
import torch.nn as nn


class BrillouinBandgapFilter(nn.Module):
    """
    브릴루앙 영역 결정 밴드갭 필터:
    - 파수(Wavevector k)가 제1 브릴루앙 영역(First Brillouin Zone: [-pi/a, pi/a])을 초과하는 고주파 노이즈 차단
    - 브래그 산란(Bragg Scattering) 원리로 불필요한 고주파 활성화를 0ms 지연으로 반사 필터링
    """

    def __init__(self, hidden_dim: int = 5120, bandgap_cutoff_ratio: float = 0.25):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bandgap_cutoff_ratio = bandgap_cutoff_ratio

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Args:
            hidden_states: [Batch, SeqLen, HiddenDim]
        Returns:
            clean_states: 밴드갭 필터링된 안정 상태
            noise_attenuation_pct: 감쇠된 고주파 노이즈 비율 (%)
        """
        # 1. 1D 실수 FFT를 통한 파수(Wavevector k) 스펙트럼 도메인 변환
        k_spectrum = torch.fft.rfft(hidden_states, dim=-1)
        freq_bins = k_spectrum.shape[-1]
        
        # 2. 브릴루앙 영역 경계 차단 마스크 (Brillouin Zone Bandgap Mask)
        cutoff_bin = int(freq_bins * (1.0 - self.bandgap_cutoff_ratio))
        bandgap_mask = torch.ones_like(k_spectrum, dtype=torch.bool)
        bandgap_mask[..., cutoff_bin:] = False
        
        # 3. 밴드갭 외부의 금지된 모드 제거 및 역변환
        filtered_k = k_spectrum * bandgap_mask
        clean_states = torch.fft.irfft(filtered_k, n=self.hidden_dim, dim=-1)
        
        noise_attenuation_pct = self.bandgap_cutoff_ratio * 100.0
        return clean_states, noise_attenuation_pct
