"""
BioPhys-LLM Core: 1-Bit HDC Blended Epigenetic SwiGLU FFN + Jerne Immune Pruning
"""

import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class BioPhysUnifiedFFN(nn.Module):
    """
    1) Epigenetic DNA Methylation (1-Bit Subnetwork Masking)
    2) Hyperdimensional Computing (HDC Majority Rule Blending)
    3) Niels Jerne Idiotypic Network (Immune Anti-Hallucination Pruning)
    """

    def __init__(self, hidden_dim: int, intermediate_dim: int, num_domains: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_domains = num_domains
        
        # Frozen Base DNA Weights
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        
        # 1-Bit Epigenetic Methylation Codebook: [NumDomains, intermediate_dim]
        masks = torch.bernoulli(torch.full((num_domains, intermediate_dim), 0.5)).to(torch.uint8)
        self.register_buffer("methylation_masks", masks)
        
        # Immune Self-Pruning Matrix
        self.immune_gate = nn.Linear(intermediate_dim, intermediate_dim, bias=False)
        with torch.no_grad():
            self.immune_gate.weight.copy_(torch.eye(intermediate_dim))

    def blend_domains(self, domain_ids: List[int], weights: Optional[List[float]] = None) -> torch.Tensor:
        """HDC Majority Rule 기반 다중 도메인 실시간 비트 합성"""
        selected_masks = [self.methylation_masks[d] for d in domain_ids]
        stacked = torch.stack(selected_masks, dim=0).float()
        
        if weights is not None:
            w_t = torch.tensor(weights, device=stacked.device).view(-1, 1)
            blended = (stacked * w_t).sum(dim=0) > (sum(weights) / 2.0)
        else:
            blended = stacked.mean(dim=0) >= 0.5
        return blended.float()

    def forward(self, x: torch.Tensor, domain_ids: Optional[List[int]] = None) -> Tuple[torch.Tensor, float]:
        batch_size, seq_len, _ = x.shape
        
        # 1. SwiGLU Intermediate
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        
        # 2. Epigenetic 1-Bit Masking
        if domain_ids is not None:
            mask = self.blend_domains(domain_ids).view(1, 1, -1)
            gate = gate * mask
            
        intermediate = gate * up
        
        # 3. Jerne Immune Self-Pruning (Anti-Hallucination)
        immune_affinity = torch.sigmoid(self.immune_gate(intermediate))
        prune_mask = (immune_affinity > 0.30).float()
        clean_intermediate = intermediate * immune_affinity * prune_mask
        
        # 4. Down Projection
        out = self.down_proj(clean_intermediate)
        pruned_ratio = (1.0 - prune_mask.mean().item()) * 100.0
        
        return out, pruned_ratio
