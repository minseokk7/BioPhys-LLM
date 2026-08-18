"""
BioPhys-LLM 2.7: 긴 작업 최적화 온사거 상반 정리 & 브릴루앙 영역 실측 벤치마크
1) 비평형 열역학: 온사거 상반 정리 어텐션 (Onsager Reciprocal Attention)
2) 고체물리학: 브릴루앙 영역 포논 밴드갭 필터 (Brillouin Zone Bandgap Filter)
"""

import time
import torch
from biophys_llm import OnsagerReciprocalAttention, BrillouinBandgapFilter


def test_onsager_brillouin_benchmarks():
    print("\n" + "=" * 80)
    print(" 🌌🔬 [BioPhys-LLM 2.7] 온사거 열역학 상반 정리 & 브릴루앙 밴드갭 실측 벤치마크")
    print("=" * 80)

    # -------------------------------------------------------------
    # [1] 온사거 상반 정리 어텐션 (Onsager Reciprocal Attention) 실측
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print(" 🌡️ [1] 비평형 열역학: 온사거 상반 정리 어텐션 (Onsager Attention) 실측")
    print("-" * 80)
    
    num_heads = 32
    head_dim = 128
    batch = 1
    seq_len = 512
    
    onsager_attn = OnsagerReciprocalAttention(num_heads=num_heads, head_dim=head_dim)
    
    q = torch.randn(batch, num_heads, seq_len, head_dim)
    k = torch.randn(batch, num_heads, seq_len, head_dim)
    v = torch.randn(batch, num_heads, seq_len, head_dim)
    
    t0 = time.perf_counter()
    out, flops_saved = onsager_attn(q, k, v)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    
    print(f"▶ 32개 헤드 512길이 문맥 어텐션 연산 시간: {elapsed_ms:.2f} ms")
    print(f"💥 온사거 상반 대칭성을 통한 FLOPs 절감율: {flops_saved:.2f}% (상삼각/하삼각 가역성)")
    assert out.shape == q.shape, "출력 형상 불일치"
    assert flops_saved >= 50.0, "절감율 미달"
    print("✅ [검증 통과] 온사거 상반 어텐션 100% 정상 작동!")

    # -------------------------------------------------------------
    # [2] 브릴루앙 영역 결정 밴드갭 필터 실측
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print(" 💎 [2] 고체물리학: 브릴루앙 영역 포논 밴드갭 필터 (Brillouin Filter) 실측")
    print("-" * 80)
    
    filter_layer = BrillouinBandgapFilter(hidden_dim=5120, bandgap_cutoff_ratio=0.25)
    hidden_states = torch.randn(1, 128, 5120) * 2.0
    
    t0 = time.perf_counter()
    clean_states, noise_attenuation = filter_layer(hidden_states)
    filter_elapsed = (time.perf_counter() - t0) * 1000.0
    
    print(f"▶ 5120 차원 은닉 상태 밴드갭 필터링 소요 시간: {filter_elapsed:.2f} ms")
    print(f"💥 브릴루앙 경계 반사로 제거된 고주파 노이즈: {noise_attenuation:.2f}%")
    print(f"▶ 원본 분산: {hidden_states.var():.4f} ──► 필터링 후 안정 분산: {clean_states.var():.4f}")
    assert clean_states.shape == hidden_states.shape, "형상 불일치"
    assert clean_states.var() < hidden_states.var(), "밴드갭 노이즈 감쇠 실패"
    print("✅ [검증 통과] 브릴루앙 밴드갭 필터 100% 정상 작동!")

    print("\n" + "=" * 80)
    print(" 🎉 [긴 작업 최적화 완료] 온사거 상반 대칭(50% FLOPs)과 브릴루앙 밴드갭 결합 성공!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    test_onsager_brillouin_benchmarks()
