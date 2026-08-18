"""
BioPhys-LLM 3.0: 빡센 전수 스트레스 테스트 및 최적화 파이프라인 적용 순서 변경 비교 실측
- 27B 거대 모델 실차원 (Hidden Dim: 5120, Seq Len: 1024, Head Dim: 128, 32 Heads)
- 3가지 파이프라인 순서 순열(Permutation Orders) 비교:
  1) [파이프라인 A]: 생체역학 우선 (Bio-First: 도약전도 -> 점균류 -> 온사거 -> 1-Bit -> 심플렉틱 -> 역위상충돌 -> 물관)
  2) [파이프라인 B]: 물리역학 우선 (Physics-First: 역위상충돌 -> 온사거 -> 심플렉틱 -> 도약전도 -> 1-Bit -> 물관증산)
  3) [파이프라인 C]: 유체/엔트로피 우선 (Hydro-First: 란다우초유체 -> 브릴루앙 -> 도약전도 -> 온사거 -> 공명감쇠 -> 심플렉틱)
"""

import time
import torch
from biophys_llm import (
    SaltatoryLayerConductor,
    MycelialAttentionRouter,
    OnsagerReciprocalAttention,
    EpigeneticDomainMasker,
    SymplecticHamiltonianLayer,
    DestructiveCollisionFilter,
    LandauSuperfluidConduit,
    BrillouinBandgapFilter,
    CollisionalDampingStabilizer,
    XylemCohesionTensionPuller,
)


def run_pipeline_a_bio_first(x, q, k, v, past_kv, saltatory, mycelial, onsager, epimask, symplectic, collision, xylem):
    """파이프라인 A: 생체역학 우선"""
    # 1. 랑비에 결절 도약 전도
    x_jump, _, _, _ = saltatory.saltatory_forward_pass(x)
    
    # 2. 온사거 상반 대칭 어텐션
    attn_out, _ = onsager(q, k, v)
    
    # 3. 점균류 어텐션 라우팅
    myc_out, _ = mycelial(q, k, v)
    
    # 4. 후성유전학 1-Bit SIMD 마스킹
    masked_out, _ = epimask(x_jump)
    
    # 5. 심플렉틱 류빌 체적 보존
    sym_out, _ = symplectic(masked_out)
    
    # 6. 역위상 정면 충돌 노이즈 소멸
    clean_out, _ = collision(sym_out)
    
    # 7. 물관 증산작용 흡인
    pulled_ctx, _ = xylem(clean_out[0, -1, :128], past_kv)
    
    return clean_out


def run_pipeline_b_physics_first(x, q, k, v, past_kv, saltatory, mycelial, onsager, epimask, symplectic, collision, xylem):
    """파이프라인 B: 물리역학 우선"""
    # 1. 역위상 정면 충돌 노이즈 선제 소멸
    clean_x, _ = collision(x)
    
    # 2. 온사거 상반 대칭 어텐션
    attn_out, _ = onsager(q, k, v)
    
    # 3. 심플렉틱 류빌 체적 보존
    sym_out, _ = symplectic(clean_x)
    
    # 4. 랑비에 결절 도약 전도
    x_jump, _, _, _ = saltatory.saltatory_forward_pass(sym_out)
    
    # 5. 점균류 어텐션 라우팅
    myc_out, _ = mycelial(q, k, v)
    
    # 6. 후성유전학 1-Bit 마스킹
    masked_out, _ = epimask(x_jump)
    
    # 7. 물관 증산작용 흡인
    pulled_ctx, _ = xylem(masked_out[0, -1, :128], past_kv)
    
    return masked_out


def run_pipeline_c_hydro_first(x, q, k, v, past_kv, superfluid, brillouin, saltatory, onsager, damping, symplectic):
    """파이프라인 C: 유체/엔트로피 우선"""
    # 1. 란다우 초유체 무마찰 정규화
    super_x, _ = superfluid(x)
    
    # 2. 브릴루앙 영역 포논 밴드갭 필터
    clean_x, _ = brillouin(super_x)
    
    # 3. 랑비에 결절 도약 전도
    x_jump, _, _, _ = saltatory.saltatory_forward_pass(clean_x)
    
    # 4. 온사거 상반 대칭 어텐션
    attn_out, _ = onsager(q, k, v)
    
    # 5. 플라즈마 공명 충돌 감쇠
    damped_x, _ = damping(x_jump)
    
    # 6. 심플렉틱 류빌 체적 보존
    sym_out, _ = symplectic(damped_x)
    
    return sym_out


def test_rigorous_pipeline_permutations():
    print("\n" + "=" * 85)
    print(" 🚀🔥 [BioPhys-LLM 3.0] 27B 실차원 초고강도 빡센 스트레스 & 순서 변경 벤치마크")
    print("=" * 85)

    batch = 1
    seq_len = 1024 # 1024 길이 장문 컨텍스트
    hidden_dim = 5120 # Qwen 27B 은닉 차원
    num_heads = 32
    head_dim = 128
    past_seq_len = 2048 # 과거 2048 토큰 KV 캐시

    print(f"▶ 테스트 환경: Batch={batch}, SeqLen={seq_len}, HiddenDim={hidden_dim}, Heads={num_heads} (27B Scale)")
    print(f"▶ 각 파이프라인별 10회 반복 측정 평균 산출\n")

    # 1. 모듈 인스턴스 초기화
    saltatory = SaltatoryLayerConductor(total_layers=64, node_interval=4, hidden_dim=hidden_dim)
    mycelial = MycelialAttentionRouter(num_heads=num_heads, head_dim=head_dim)
    onsager = OnsagerReciprocalAttention(num_heads=num_heads, head_dim=head_dim)
    epimask = EpigeneticDomainMasker(intermediate_dim=hidden_dim)
    symplectic = SymplecticHamiltonianLayer(hidden_dim=hidden_dim, dt=0.1)
    collision = DestructiveCollisionFilter(hidden_dim=hidden_dim)
    xylem = XylemCohesionTensionPuller(head_dim=head_dim)
    superfluid = LandauSuperfluidConduit(hidden_dim=hidden_dim)
    brillouin = BrillouinBandgapFilter(hidden_dim=hidden_dim)
    damping = CollisionalDampingStabilizer(hidden_dim=hidden_dim)

    # 2. 테스트 텐서 생성
    x = torch.randn(batch, seq_len, hidden_dim)
    q = torch.randn(batch, num_heads, seq_len, head_dim)
    k = torch.randn(batch, num_heads, seq_len, head_dim)
    v = torch.randn(batch, num_heads, seq_len, head_dim)
    past_kv = torch.randn(batch, past_seq_len, head_dim)

    # 워밍업
    _ = run_pipeline_a_bio_first(x, q, k, v, past_kv, saltatory, mycelial, onsager, epimask, symplectic, collision, xylem)
    _ = run_pipeline_b_physics_first(x, q, k, v, past_kv, saltatory, mycelial, onsager, epimask, symplectic, collision, xylem)
    _ = run_pipeline_c_hydro_first(x, q, k, v, past_kv, superfluid, brillouin, saltatory, onsager, damping, symplectic)

    num_iters = 10

    # -------------------------------------------------------------
    # [1] 파이프라인 A (생체역학 우선) 실측
    # -------------------------------------------------------------
    latencies_a = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        out_a = run_pipeline_a_bio_first(x, q, k, v, past_kv, saltatory, mycelial, onsager, epimask, symplectic, collision, xylem)
        latencies_a.append((time.perf_counter() - t0) * 1000.0)
    avg_a = sum(latencies_a) / len(latencies_a)
    var_a = out_a.var().item()

    # -------------------------------------------------------------
    # [2] 파이프라인 B (물리역학 우선) 실측
    # -------------------------------------------------------------
    latencies_b = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        out_b = run_pipeline_b_physics_first(x, q, k, v, past_kv, saltatory, mycelial, onsager, epimask, symplectic, collision, xylem)
        latencies_b.append((time.perf_counter() - t0) * 1000.0)
    avg_b = sum(latencies_b) / len(latencies_b)
    var_b = out_b.var().item()

    # -------------------------------------------------------------
    # [3] 파이프라인 C (유체/엔트로피 우선) 실측
    # -------------------------------------------------------------
    latencies_c = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        out_c = run_pipeline_c_hydro_first(x, q, k, v, past_kv, superfluid, brillouin, saltatory, onsager, damping, symplectic)
        latencies_c.append((time.perf_counter() - t0) * 1000.0)
    avg_c = sum(latencies_c) / len(latencies_c)
    var_c = out_c.var().item()

    # -------------------------------------------------------------
    # 📊 종합 비교 결과 표 출력
    # -------------------------------------------------------------
    print("-" * 85)
    print(f"{'순열 파이프라인 구성':<28} | {'평균 소요 시간(ms)':<18} | {'출력 분산(안정성)':<18} | {'상대 속도'}")
    print("-" * 85)
    print(f"{'A. 생체역학 우선 (Bio-First)':<28} | {avg_a:>14.2f} ms | {var_a:>16.4f} | 기준 (1.00x)")
    print(f"{'B. 물리역학 우선 (Physics-First)':<28} | {avg_b:>14.2f} ms | {var_b:>16.4f} | {avg_a/avg_b:>5.2f}x")
    print(f"{'C. 유체/엔트로피 우선 (Hydro-First)':<28} | {avg_c:>14.2f} ms | {var_c:>16.4f} | {avg_a/avg_c:>5.2f}x")
    print("-" * 85)

    # 안전성 및 NaN/Inf 검증
    for name, tensor in [("A", out_a), ("B", out_b), ("C", out_c)]:
        assert not torch.isnan(tensor).any(), f"파이프라인 {name}에서 NaN 검출!"
        assert not torch.isinf(tensor).any(), f"파이프라인 {name}에서 Inf 검출!"

    fastest = min([("파이프라인 A (생체역학 우선)", avg_a), ("파이프라인 B (물리역학 우선)", avg_b), ("파이프라인 C (유체/엔트로피 우선)", avg_c)], key=lambda x: x[1])
    most_stable = min([("파이프라인 A", var_a), ("파이프라인 B", var_b), ("파이프라인 C", var_c)], key=lambda x: x[1])

    print(f"\n🏆 [최고 속도 골든 순서]: {fastest[0]} ({fastest[1]:.2f} ms)")
    print(f"💎 [최고 안정성 골든 순서]: {most_stable[0]} (분산: {most_stable[1]:.4f})")
    print("✅ [검증 통과] 3가지 순열 모두 NaN/Inf 오차 0.0000%로 완벽하게 안정 작동!")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    test_rigorous_pipeline_permutations()
