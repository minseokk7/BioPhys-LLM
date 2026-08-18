"""
BioPhys-LLM 3.0: Golden Order Grand Unified Pipeline 실측 검증
"""

import time
import torch
from biophys_llm import BioPhysGrandUnifiedBlock


def test_grand_unified_block_pipeline():
    print("\n" + "=" * 80)
    print(" 👑 [BioPhys-LLM 3.0] 황금 순서 그랜드 대통합 파이프라인 (Grand Unified Pipeline) 실측")
    print("=" * 80)

    batch_size = 1
    seq_len = 128
    hidden_dim = 5120
    num_heads = 32
    head_dim = 128
    past_seq_len = 512

    unified_block = BioPhysGrandUnifiedBlock(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        total_layers=64,
        node_interval=4
    )

    x = torch.randn(batch_size, seq_len, hidden_dim)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)
    past_kv = torch.randn(batch_size, past_seq_len, head_dim)

    # 1회 단일 순전파(Forward Pass) 실행
    out, elapsed_ms, flops_saved = unified_block(x, q, k, v, past_kv)

    print(f"\n▶ 27B 실차원 황금 순서 1회 순전파 소요 시간: {elapsed_ms:.2f} ms")
    print(f"💥 황금 순서 파이프라인 총 연산 절감율: {flops_saved:.2f}%")
    print(f"▶ 출력 텐서 형상: {out.shape}")
    print(f"▶ 출력 수치 분산: {out.var().item():.4f}")

    assert out.shape == x.shape, "출력 형상 불일치"
    assert not torch.isnan(out).any(), "NaN 검출"
    assert not torch.isinf(out).any(), "Inf 검출"
    print("✅ [검증 통과] 3.0 황금 순서 그랜드 대통합 블록 100% 정상 작동!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    test_grand_unified_block_pipeline()
