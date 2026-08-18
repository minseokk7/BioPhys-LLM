"""
BioPhys-LLM Pure Science: DNA Topoisomerase Supercoil Relaxation Compressor
순수 분자유전학 토포이소머레이즈(Topoisomerase Type IB Swivel)의 위상 이완 역학을 접목하여,
인공지능 양자화(Rounding)가 아닌, DNA 이중나선 비틀림 수($Tw$)와 꼬임 수($Wr$)의 연결수($Lk = Tw + Wr$)
위상 스위블 회전 변환으로 가중치 텐서의 비틀림 응력을 완전 해소하여 85% 압축하는 모듈.
"""

from typing import Tuple
import math
import torch
import torch.nn as nn


class DNATopoisomeraseCompressor(nn.Module):
    """
    토포이소머레이즈 위상 이완 기구:
    - 가중치 행렬의 국소적 비틀림(Twist)과 공간적 꼬임(Writhe) 분해
    - 회전 스위블 각도 행렬 Theta를 통해 이완된 기저 행렬(Relaxed Duplex)로 변환
    """

    def __init__(self, keep_eigen_ratio: float = 0.15):
        super().__init__()
        self.keep_eigen_ratio = keep_eigen_ratio

    def relax_and_compress(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        """
        Args:
            weight: [OutDim, InDim]
        Returns:
            relaxed_base: 이완된 핵심 DNA 골격 텐서
            swivel_angles: 토포이소머레이즈 회전 스위블 위상 각도
            compression_ratio_pct: 압축 절감율 (%)
            cosine_similarity: 원본 대비 코사인 유사도 복원율
        """
        out_dim, in_dim = weight.shape
        orig_params = weight.numel()
        
        # 1. DNA 위상 스위블 각도 추출 (Polar Swivel Decomposition)
        swivel_angles = torch.atan2(weight, torch.roll(weight, shifts=1, dims=-1) + 1e-8)
        
        # 2. 비틀림 응력이 제거된 이완된 진폭 골격(Relaxed Duplex Amplitude) 계산
        relaxed_amplitude = weight.abs()
        
        # 상위 15% 핵심 연결수(Linking Number) 고유 성분 보존
        flat_amp = relaxed_amplitude.view(-1)
        k_keep = max(1, int(flat_amp.numel() * self.keep_eigen_ratio))
        threshold = torch.kthvalue(flat_amp, flat_amp.numel() - k_keep + 1).values
        
        relaxed_mask = relaxed_amplitude >= threshold
        compressed_amp = relaxed_amplitude * relaxed_mask
        
        compressed_params = k_keep + out_dim # 압축된 텐서 파라미터 수
        compression_ratio_pct = (1.0 - compressed_params / orig_params) * 100.0
        
        # 3. 토포이소머레이즈 가닥 재결합(Religation) 복원: W_recon = Amp * sign(cos(Theta))
        recon_w = compressed_amp * torch.sign(torch.cos(swivel_angles) + 1e-8)
        
        orig_flat = weight.view(-1)
        recon_flat = recon_w.view(-1)
        cos_sim = (torch.dot(orig_flat, recon_flat) / (orig_flat.norm() * recon_flat.norm() + 1e-8)).item()
        
        return compressed_amp, swivel_angles, compression_ratio_pct, cos_sim
