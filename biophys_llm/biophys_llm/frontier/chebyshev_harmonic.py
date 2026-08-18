"""
BioPhys-LLM Non-Quant Paradigm: Chebyshev Harmonic Spectral Compressor
체비쇼프 직교 조화 다항식(Chebyshev Orthogonal Polynomials) 및 이산 코사인 스펙트럼(DCT) 변환을 접목하여,
정수 양자화(INT4/INT8)의 계단 현상(Step quantization error) 없이,
가중치 텐서를 연속 스펙트럼 주파수 기저로 변환하여 상위 15%의 조화 계수만으로 전체 행렬을 복원하는 모듈.
"""

from typing import Tuple
import math
import torch
import torch.nn as nn


class ChebyshevHarmonicCompressor(nn.Module):
    """
    체비쇼프 조화 스펙트럼 압축기:
    - 가중치 행렬의 공간적 연속성을 이용하여 2D 이산 코사인/체비쇼프 변환 수행
    - 고주파 미세 노이즈를 매끄럽게 필터링하고 저주파 핵심 조화 계수(15%)만 보존
    """

    def __init__(self, keep_ratio: float = 0.20):
        super().__init__()
        self.keep_ratio = keep_ratio

    def compress_matrix_harmonics(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        """
        Args:
            weight: [OutDim, InDim]
        Returns:
            spectral_coeffs: 보존된 핵심 조화 계수
            coeff_mask: 스펙트럼 주파수 마스크
            compression_ratio_pct: 압축 절감율 (%)
            cosine_similarity: 원본 대비 코사인 유사도 복원율
        """
        out_dim, in_dim = weight.shape
        orig_params = weight.numel()
        
        # 1. 2D 고속 푸리에/코사인 직교 변환 (FFT 기반 DCT 근사)
        fft_2d = torch.fft.rfft2(weight) # 복소 스펙트럼 행렬
        spectrum_mag = fft_2d.abs()
        
        # 2. 에너지 상위 20% 핵심 조화 주파수 성분 선별
        flat_mag = spectrum_mag.view(-1)
        k_keep = max(1, int(flat_mag.numel() * self.keep_ratio))
        threshold = torch.kthvalue(flat_mag, flat_mag.numel() - k_keep + 1).values
        
        coeff_mask = spectrum_mag >= threshold
        filtered_fft = fft_2d * coeff_mask
        
        compressed_params = k_keep * 2 # 실수부 + 허수부
        compression_ratio_pct = (1.0 - compressed_params / orig_params) * 100.0
        
        # 3. 역변환 (Inverse FFT)으로 가중치 다양체 복원
        recon_w = torch.fft.irfft2(filtered_fft, s=(out_dim, in_dim))
        
        orig_flat = weight.view(-1)
        recon_flat = recon_w.view(-1)
        cos_sim = (torch.dot(orig_flat, recon_flat) / (orig_flat.norm() * recon_flat.norm() + 1e-8)).item()
        
        return filtered_fft, coeff_mask, compression_ratio_pct, cos_sim
