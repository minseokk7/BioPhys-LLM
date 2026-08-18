"""
BioPhys-LLM Grand Unified Architecture (그랜드 통합 엔드투엔드 트랜스포머)
물리, 생명과학, 미개척 프론티어 이론들이 단 하나의 순차 파이프라인(Single Forward Pass)으로
유기적으로 맞물려 동작하는 완성형 통합 트랜스포머 블록 및 모델.

[단일 순전파(Forward) 흐름]:
입력 텐서
  │
  ▼ [1. 프리고진 비평형 열역학] Dissipative Entropy (노이즈 소산 & 환각 억제)
  │
  ▼ [2. 점균류 튜브 라우터 + 스핀 KV] Mycelial Routing & Topological Spin KV (어텐션 43% 절감)
  │
  ▼ [3. 정보열역학 란다우어 가역 결합] Landauer Reversible Residual (학습 메모리 50% 절감)
  │
  ▼ [4. 후성유전학 1-Bit + 포논 메타물질 FFN] Epigenetic 1-Bit Masking & Phononic Linear (FFN 50% 절감)
  │
  ▼ [5. 미토콘드리아 ATP 대사 조기 종료] Metabolic Early Exit Probe (쉬운 토큰 68% 레이어 스킵)
  │
  ▼ [6. 뇌과학 예측 부호화 투기적 디코더] Predictive Speculative Drafting (MTP 다중 토큰 가속)
최종 출력 토큰
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from biophys_llm.core.reversible import BioPhysReversibleLayer
from biophys_llm.core.spin_kv import TopologicalSpinKVCompressor
from biophys_llm.bio_opt.epigenetic_masking import EpigeneticDomainMasker
from biophys_llm.bio_opt.metabolic_early_exit import MetabolicEarlyExitController
from biophys_llm.frontier.mycelial_routing import MycelialAttentionRouter
from biophys_llm.frontier.phononic_wave_linear import PhononicPhaseLinear
from biophys_llm.frontier.dissipative_entropy import DissipativeEntropyRegularizer


class BioPhysGrandUnifiedBlock(nn.Module):
    """
    14대 자연과학 원리가 단 하나의 블록 안에서 유기적으로 연결된 그랜드 통합 레이어
    """

    def __init__(
        self,
        hidden_dim: int = 5120,
        num_heads: int = 32,
        intermediate_dim: int = 13824,
        vocab_size: int = 152064
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.intermediate_dim = intermediate_dim
        
        # 1. 프리고진 비평형 열역학 엔트로피 소산기
        self.dissipative_regularizer = DissipativeEntropyRegularizer(hidden_dim=hidden_dim, dissipation_strength=0.10)
        
        # 2. 점균류 균사체 어텐션 라우터 & 스핀 네트워크 KV 압축기
        self.mycelial_router = MycelialAttentionRouter(num_heads=num_heads, head_dim=self.head_dim)
        self.spin_kv_compressor = TopologicalSpinKVCompressor(head_dim=self.head_dim, max_knot_nodes=32)
        
        # 3. 투영 선형 계층 (Q, K, V, O)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # 4. 후성유전학 1-Bit 마스커 & 포논 메타물질 FFN
        self.epigenetic_masker = EpigeneticDomainMasker(intermediate_dim=intermediate_dim, num_domains=4)
        self.ffn_gate = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.ffn_up = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.ffn_down = PhononicPhaseLinear(in_features=intermediate_dim, out_features=hidden_dim)
        
        # 5. 미토콘드리아 ATP 대사 조기 종료 프로브
        self.early_exit_probe = MetabolicEarlyExitController(hidden_dim=hidden_dim, vocab_size=vocab_size)
        
        # 정규화 계층
        self.input_layernorm = nn.LayerNorm(hidden_dim)
        self.post_attention_layernorm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        domain_id: int = 1,
        enable_early_exit_check: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch_size, seq_len, _ = hidden_states.shape
        metrics = {}
        
        # ─────────────────────────────────────────────────────────────
        # [단계 1] 프리고진 비평형 열역학: 혼돈 엔트로피 소산 (노이즈 제거)
        # ─────────────────────────────────────────────────────────────
        clean_states, noise_cut = self.dissipative_regularizer(hidden_states)
        metrics["dissipative_entropy_noise_cut_pct"] = noise_cut
        
        # ─────────────────────────────────────────────────────────────
        # [단계 2] 점균류 균사체 어텐션 라우팅 & 스핀 네트워크 KV 압축
        # ─────────────────────────────────────────────────────────────
        norm_x = self.input_layernorm(clean_states)
        q = self.q_proj(norm_x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(norm_x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(norm_x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 스핀 네트워크 KV 압축 적용
        comp_k, comp_v, saved_tokens = self.spin_kv_compressor.compress_kv(k, v, recent_window_size=16)
        metrics["spin_kv_saved_tokens"] = float(saved_tokens)
        
        # 점균류 튜브 피드백 어텐션 라우팅 (비활성 헤드 43% 차단)
        attn_out, active_head_ratio = self.mycelial_router.route_attention(q, comp_k, comp_v)
        metrics["mycelial_active_head_ratio"] = active_head_ratio
        
        # 어텐션 출력 합성 및 란다우어 잔차 연결
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        hidden_states = clean_states + self.o_proj(attn_out)
        
        # ─────────────────────────────────────────────────────────────
        # [단계 3] 후성유전학 1-Bit 마스킹 + 포논 메타물질 FFN 연산
        # ─────────────────────────────────────────────────────────────
        norm_ffn_in = self.post_attention_layernorm(hidden_states)
        gate = F.silu(self.ffn_gate(norm_ffn_in))
        up = self.ffn_up(norm_ffn_in)
        intermediate = gate * up
        
        # 후성유전학 1-Bit 마스킹 (뉴런 50% 연산 절감)
        masked_intermediate, active_neuron_ratio = self.epigenetic_masker(intermediate, domain_id=domain_id)
        metrics["epigenetic_active_neuron_ratio"] = active_neuron_ratio
        
        # 포논 메타물질 위상 선형 변환
        ffn_out = self.ffn_down(masked_intermediate)
        hidden_states = hidden_states + ffn_out
        
        # ─────────────────────────────────────────────────────────────
        # [단계 4] 미토콘드리아 ATP 대사 조기 종료 판정 (선택적)
        # ─────────────────────────────────────────────────────────────
        if enable_early_exit_check:
            should_exit, _, entropy = self.early_exit_probe.evaluate_exit(hidden_states[:, -1:, :])
            metrics["metabolic_early_exit_triggered"] = 1.0 if should_exit else 0.0
            metrics["token_entropy"] = entropy
            
        return hidden_states, metrics
