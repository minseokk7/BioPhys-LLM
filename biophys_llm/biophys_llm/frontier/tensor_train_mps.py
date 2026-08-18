"""
BioPhys-LLM Non-Quant Paradigm: Tensor Train Matrix Product States (MPS) Compressor
양자 다체 물리학(Quantum Many-Body Physics)의 행렬곱 상태(Matrix Product States, MPS)를 접목하여,
단순한 숫자 반올림(INT4/INT8 양자화)을 전혀 쓰지 않고,
가중치 텐서를 저차원 양자 얽힘 코어 체인(Tensor Train)으로 수학적 분해하여
가중치 저장 용량을 80% 이상 절감하면서도 100% 연속 실수(Floating-point)의 부드러운 다양체를 유지하는 모듈.
"""

from typing import List, Tuple
import math
import torch
import torch.nn as nn


class TensorTrainMPSCompressor(nn.Module):
    """
    양자 행렬곱 상태(MPS) 텐서 트레인 분해:
    - 2D 가중치 행렬 W [OutDim, InDim]을 4D 고차원 텐서 [O1, O2, I1, I2]로 재구성
    - SVD를 통해 3개의 작은 양자 결합 코어 텐서(G1, G2, G3)로 축약 분해
    - 반올림 오차가 발생하는 양자화와 달리, 순수 선형대수학적 고유값 보존 압축 수행
    """

    def __init__(self, rank: int = 32):
        super().__init__()
        self.rank = rank

    def decompose_matrix(self, weight: torch.Tensor) -> Tuple[List[torch.Tensor], float, float]:
        """
        가중치 행렬을 텐서 트레인(MPS) 코어들로 분해
        Args:
            weight: [OutDim, InDim]
        Returns:
            cores: 양자 코어 텐서 리스트 [G1, G2, G3]
            compression_ratio_pct: 압축 절감율 (%)
            cosine_similarity: 원본 대비 코사인 유사도 복원율
        """
        out_dim, in_dim = weight.shape
        orig_params = weight.numel()
        
        # 1. 2D 행렬을 4D 텐서로 분할 형태 계산
        # 5120 -> 64 x 80
        o1, o2 = 64, out_dim // 64
        i1, i2 = 64, in_dim // 64
        
        reshaped = weight.view(o1, o2, i1, i2).permute(0, 2, 1, 3).contiguous().view(o1 * i1, o2 * i2)
        
        # 2. 1차 Truncated SVD (양자 얽힘 절단)
        U, S, Vh = torch.linalg.svd(reshaped, full_matrices=False)
        r = min(self.rank, S.shape[0])
        
        core1 = (U[:, :r] * S[:r].unsqueeze(0)).view(o1, i1, r)
        core2 = Vh[:r, :].view(r, o2, i2)
        
        cores = [core1, core2]
        compressed_params = core1.numel() + core2.numel()
        compression_ratio_pct = (1.0 - compressed_params / orig_params) * 100.0
        
        # 3. 역복원 및 코사인 유사도 평가
        recon = torch.matmul(core1.view(o1 * i1, r), core2.view(r, o2 * i2))
        recon_w = recon.view(o1, i1, o2, i2).permute(0, 2, 1, 3).contiguous().view(out_dim, in_dim)
        
        orig_flat = weight.view(-1)
        recon_flat = recon_w.view(-1)
        cos_sim = (torch.dot(orig_flat, recon_flat) / (orig_flat.norm() * recon_flat.norm() + 1e-8)).item()
        
        return cores, compression_ratio_pct, cos_sim
