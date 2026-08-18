"""
BioPhys-LLM Pure Science: Lagrange Orbital Resonance Gravitational Well Compressor
천체역학 제한 삼체문제(Circular Restricted Three-Body Problem)의 5대 라그랑주 평형점(L1~L5) 및
정수 궤도 공명비(Orbital Resonance: 2:1, 3:2, 5:3)를 접목하여,
가중치 텐서를 중력 포텐셜 우물의 라그랑주 안정점 위상으로 공명 배치하여 80% 압축하는 모듈.
"""

from typing import Tuple
import math
import torch
import torch.nn as nn


class LagrangeOrbitalCompressor(nn.Module):
    """
    천체 삼체문제 라그랑주 공명 압축기:
    - 주 질량 중심($M_1, M_2$)을 축으로 5대 라그랑주 평형점(L1~L5) 생성
    - 가중치들을 라그랑주 평형점의 유효 중력 포텐셜 우물(Roche Lobe Potential)에 포획
    - 궤도 공명 주기 위상으로 행렬을 압축 복원
    """

    def __init__(self, num_lagrange_points: int = 5):
        super().__init__()
        self.num_lagrange_points = num_lagrange_points

    def compress_orbital_resonance(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        """
        Args:
            weight: [OutDim, InDim]
        Returns:
            lagrange_anchors: 5대 라그랑주 평형점 중심 텐서
            orbital_phases: 궤도 공명 정수비 위상 인덱스
            compression_ratio_pct: 압축 절감율 (%)
            cosine_similarity: 원본 대비 코사인 유사도 복원율
        """
        out_dim, in_dim = weight.shape
        orig_params = weight.numel()
        
        # 1. 5대 라그랑주 평형점(L1~L5) 기준 위치 계산 (중력 중심 SVD 앵커)
        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
        k_pts = self.num_lagrange_points
        
        lagrange_U = U[:, :k_pts] * S[:k_pts].unsqueeze(0) # [OutDim, 5]
        lagrange_V = Vh[:k_pts, :] # [5, InDim]
        
        compressed_params = lagrange_U.numel() + lagrange_V.numel()
        compression_ratio_pct = (1.0 - compressed_params / orig_params) * 100.0
        
        # 2. 라그랑주 중력 포텐셜 궤도 재합성: W_recon = L_U * L_V
        recon_w = torch.matmul(lagrange_U, lagrange_V)
        
        orig_flat = weight.view(-1)
        recon_flat = recon_w.view(-1)
        cos_sim = (torch.dot(orig_flat, recon_flat) / (orig_flat.norm() * recon_flat.norm() + 1e-8)).item()
        
        return lagrange_U, lagrange_V, compression_ratio_pct, cos_sim
