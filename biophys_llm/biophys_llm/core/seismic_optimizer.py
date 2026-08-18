"""
BioPhys-LLM 2.0: Gutenberg-Richter Seismic Power-Law Optimizer
지질학 판구조론 단층 응력 및 지진 멱법칙(Self-Organized Criticality)을 접목한
실제 구동 가능한 고속 수렴 PyTorch 옵티마이저.
"""

from typing import Iterable, Dict, Any, Optional
import math
import torch
from torch.optim.optimizer import Optimizer


class SeismicOptimizer(Optimizer):
    """
    판구조론 지진 옵티마이저:
    - 각 파라미터 텐서의 누적 응력(Stress Tensor)을 모니터링
    - 응력이 지진 임계치(Threshold)를 초과할 때 멱법칙 단층 슬립(Slip) 갱신 발생
    - 미세 노이즈 그래디언트 소모를 억제하고 급격한 손실 수렴을 유도
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        stress_threshold: float = 0.01,
        power_law_alpha: float = 1.5,
        weight_decay: float = 0.01,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
            
        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            stress_threshold=stress_threshold,
            power_law_alpha=power_law_alpha,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            threshold = group["stress_threshold"]
            alpha = group["power_law_alpha"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]

                # 상태 초기화
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["accumulated_stress"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                stress = state["accumulated_stress"]

                # Weight decay 적용
                if weight_decay != 0:
                    p.mul_(1.0 - lr * weight_decay)

                # 모멘텀 및 2차 모멘트 갱신
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                step_size = lr / bias_correction1

                # 판구조론 단층 응력 누적 (Stress Accumulation)
                stress.add_(grad.abs())

                # 지진 임계치 초과 판정 (Earthquake Slip Condition)
                quake_mask = stress > threshold
                
                # 멱법칙 스케일링 계수: (Stress / Threshold)^alpha
                power_factor = torch.clamp((stress / (threshold + eps)) ** (1.0 / alpha), min=1.0, max=5.0)
                
                # 단층 슬립이 발생한 영역에 집중 업데이트
                update = (exp_avg / denom) * torch.where(quake_mask, power_factor, torch.ones_like(power_factor) * 0.5)
                p.add_(update, alpha=-step_size)

                # 발생한 에너지는 방출(Stress Reset/Relief)
                stress.mul_(torch.where(quake_mask, torch.zeros_like(stress), torch.ones_like(stress)))

        return loss
