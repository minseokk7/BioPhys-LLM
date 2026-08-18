"""
BioPhys-LLM 2.0: Topological Spin Network & BEC KV Cache Compressor
양자 스핀 네트워크(위상 불변성)와 보스-아인슈타인 응축(에너지 양자화)을 접목한
실제 구동 가능한 장문 컨텍스트 KV 캐시 압축 엔진.
"""

from typing import Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TopologicalSpinKVCompressor(nn.Module):
    """
    1) BEC Energy Saliency Gate: 정보 엔트로피가 높은 핵심 토큰 자동 식별
    2) Spin Network Graph Merging: 비핵심 과거 토큰들을 위상학적 매듭(Knot) 노드로 압축 병합
    """

    def __init__(self, head_dim: int, max_knot_nodes: int = 128, compression_ratio: float = 0.25):
        super().__init__()
        self.head_dim = head_dim
        self.max_knot_nodes = max_knot_nodes
        self.compression_ratio = compression_ratio
        
        # 에너지 게이트: 각 토큰의 정보 밀도(Saliency) 평가
        self.energy_gate = nn.Linear(head_dim, 1, bias=False)
        # 위상 투영기: 스핀 매듭 노드 생성
        self.knot_projector = nn.Linear(head_dim, head_dim, bias=False)

    def compress_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        recent_window_size: int = 64
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Key/Value 캐시를 위상학적으로 압축:
        - 최근 recent_window_size 토큰은 100% 무손실 유지
        - 과거 토큰 중 에너지 상위 top-k 토큰 보존
        - 나머지 과거 토큰들을 max_knot_nodes 개수의 스핀 매듭으로 양자 응축(BEC)
        """
        # key, value shape: [batch, num_kv_heads, seq_len, head_dim]
        batch, heads, seq_len, dim = key.shape
        
        if seq_len <= recent_window_size + self.max_knot_nodes:
            return key, value, 0  # 압축할 필요 없음
            
        past_len = seq_len - recent_window_size
        past_k = key[:, :, :past_len, :]
        past_v = value[:, :, :past_len, :]
        recent_k = key[:, :, past_len:, :]
        recent_v = value[:, :, past_len:, :]
        
        # 1. BEC 에너지 스코어 계산
        energy = torch.sigmoid(self.energy_gate(past_k)).squeeze(-1) # [batch, heads, past_len]
        
        # 2. 고에너지 핵심 토큰 보존 (상위 25%)
        keep_count = max(8, int(past_len * self.compression_ratio))
        keep_count = min(keep_count, past_len - self.max_knot_nodes)
        
        _, vital_indices = torch.topk(energy, k=keep_count, dim=-1)
        vital_idx_exp = vital_indices.unsqueeze(-1).expand(-1, -1, -1, dim)
        
        vital_k = torch.gather(past_k, 2, vital_idx_exp)
        vital_v = torch.gather(past_v, 2, vital_idx_exp)
        
        # 3. 비핵심 배경 토큰들을 위상학적 스핀 매듭(Knot) 노드로 풀링(Pooling)
        # past_k 전체를 max_knot_nodes 블록으로 균등 분할하여 응축
        chunk_size = past_len // self.max_knot_nodes
        if chunk_size > 0:
            usable_len = chunk_size * self.max_knot_nodes
            reshaped_k = past_k[:, :, :usable_len, :].view(batch, heads, self.max_knot_nodes, chunk_size, dim)
            reshaped_v = past_v[:, :, :usable_len, :].view(batch, heads, self.max_knot_nodes, chunk_size, dim)
            
            knot_k = self.knot_projector(reshaped_k.mean(dim=3))
            knot_v = reshaped_v.mean(dim=3)
        else:
            knot_k = past_k[:, :, :self.max_knot_nodes, :]
            knot_v = past_v[:, :, :self.max_knot_nodes, :]
            
        # 4. 최종 압축 KV 결합: [스핀 매듭 노드 + 핵심 토큰 + 최근 윈도우 토큰]
        compressed_k = torch.cat([knot_k, vital_k, recent_k], dim=2)
        compressed_v = torch.cat([knot_v, vital_v, recent_v], dim=2)
        
        saved_tokens = seq_len - compressed_k.shape[2]
        return compressed_k, compressed_v, saved_tokens
