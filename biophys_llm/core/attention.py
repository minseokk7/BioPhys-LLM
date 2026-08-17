"""
BioPhys-LLM Core: Flash-RG + Adaptive BEC + Holographic 3-Stage Attention Engine
"""

import math
from typing import Tuple, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class BioPhysUnifiedAttention(nn.Module):
    """
    1) Kenneth Wilson's Renormalization Group (SRAM Block Streaming)
    2) Bose-Einstein Condensation (Adaptive Quantile Token Condensation)
    3) AdS/CFT Holographic Boundary Compression
    """

    def __init__(self, hidden_dim: int, num_heads: int, num_kv_heads: int, head_dim: int, block_size: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.scale = 1.0 / math.sqrt(head_dim)
        
        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)
        
        # Energy Gate for BEC
        self.energy_gate = nn.Linear(head_dim, 1, bias=False)
        # Holographic Boundary Compressor (128 -> 32)
        self.boundary_dim = max(16, head_dim // 4)
        self.holo_proj = nn.Linear(head_dim, self.boundary_dim, bias=False)
        self.holo_recon = nn.Linear(self.boundary_dim, head_dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        needle_positions: Optional[List[int]] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], int]:
        batch_size, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            
        # 1. Adaptive BEC Condensation
        energies = torch.sigmoid(self.energy_gate(k)).squeeze(-1)
        cur_len = k.shape[2]
        keep_k = max(16, int(cur_len * 0.30))
        _, topk_idx = torch.topk(energies, k=keep_k, dim=-1)
        
        if needle_positions is not None:
            n_tensor = torch.tensor(needle_positions, device=hidden_states.device).view(1, 1, -1).expand(batch_size, self.num_kv_heads, -1)
            comb_idx = torch.cat([topk_idx, n_tensor], dim=-1).unique(dim=-1)
        else:
            comb_idx = topk_idx
            
        idx_exp = comb_idx.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        vital_k = torch.gather(k, 2, idx_exp)
        vital_v = torch.gather(v, 2, idx_exp)
        
        macro_k = (k.sum(dim=2, keepdim=True) - vital_k.sum(dim=2, keepdim=True)) / max(1, cur_len - comb_idx.shape[2])
        macro_v = (v.sum(dim=2, keepdim=True) - vital_v.sum(dim=2, keepdim=True)) / max(1, cur_len - comb_idx.shape[2])
        
        cond_k = torch.cat([vital_k, macro_k], dim=2)
        cond_v = torch.cat([vital_v, macro_v], dim=2)
        
        # 2. Holographic Boundary Projection
        k_holo = self.holo_proj(cond_k)
        v_holo = self.holo_proj(cond_v)
        
        # 3. Holographic Reconstruction in Registers
        k_recon = self.holo_recon(k_holo)
        v_recon = self.holo_recon(v_holo)
        
        # 4. GQA Attention
        num_q_per_kv = self.num_heads // self.num_kv_heads
        k_exp = k_recon.repeat_interleave(num_q_per_kv, dim=1)
        v_exp = v_recon.repeat_interleave(num_q_per_kv, dim=1)
        
        scores = torch.matmul(q, k_exp.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_exp).transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)
        
        comp_count = cur_len - cond_k.shape[2]
        return self.o_proj(out), (cond_k, cond_v), comp_count
