r"""
BioPhys-LLM Pure Science: Anthropic Observer Selection Pruner (우주론적 인류원리 관측자 선택 가지치기)
우주론의 인류원리(Anthropic Principle) 및 관측자 선택 효과(Observation Selection Effect)를 접목하여,
언어 모델의 수많은 디코딩 다중 경로(Multiverse Paths) 중 인간 관측자의 의미론적 일관성(Semantic Coherence)을
만족하지 못하는 무의미한 탐색 가지를 선제적으로 제거하여 디코딩 속도를 가속하는 모듈.
"""

from typing import Tuple
import torch
import torch.nn as nn


class AnthropicObserverPruner(nn.Module):
    """
    인류원리 관측자 선택 가지치기:
    - 후보 토큰 벡터들의 관측자 일관성 포텐셜($\Psi_{observer}$) 측정
    - 관측자 일관성이 평균 이하인 붕괴 경로(Uninhabitable Branches)는 어텐션 연산 전에 즉시 사전 가지치기
    """

    def __init__(self, hidden_dim: int = 5120, top_k_ratio: float = 0.50):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.top_k_ratio = top_k_ratio
        self.observer_probe = nn.Linear(hidden_dim, 1, bias=False)

    def prune_anthropic_branches(
        self,
        candidate_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Args:
            candidate_states: [Batch, NumCandidates, HiddenDim]
        Returns:
            survived_states: 인류원리 기준 생존 상태들
            survived_indices: 생존한 후보 마스크
            pruned_ratio_pct: 제거된 무의미 가지 비율 (%)
        """
        logits = self.observer_probe(candidate_states).squeeze(-1) # [Batch, NumCandidates]
        
        # 상위 top_k_ratio 에 해당하는 관측자 적합 경로만 보존
        num_candidates = candidate_states.shape[1]
        k = max(1, int(num_candidates * self.top_k_ratio))
        
        topk_vals, topk_indices = torch.topk(logits, k=k, dim=-1)
        min_topk_val = topk_vals[..., -1:]
        
        survived_mask = logits >= min_topk_val
        pruned_ratio_pct = (1.0 - (k / num_candidates)) * 100.0
        
        return candidate_states * survived_mask.unsqueeze(-1), survived_mask, pruned_ratio_pct
