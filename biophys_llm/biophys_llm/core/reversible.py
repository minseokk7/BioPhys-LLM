"""
BioPhys-LLM 2.0: Core Reversible Residual Block (Landauer Reversibility)
실제 PyTorch autograd.Function을 이용한 메모리 무손실 가역 역전파 엔진.
중간 활성화(Activations)를 메모리에 저장하지 않고 역전파 시 역함수로 정확하게 복원하여 학습 메모리를 최대 50% 절감.
"""

from typing import Callable, Tuple
import torch
import torch.nn as nn


class LandauerReversibleFunction(torch.autograd.Function):
    """
    정보열역학 란다우어 가역 연산:
    F, G가 비선형 함수일 때:
      x1, x2 = split(x)
      y1 = x1 + F(x2)
      y2 = x2 + G(y1)
    역연산:
      x2 = y2 - G(y1)
      x1 = y1 - F(x2)
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, f_block: nn.Module, g_block: nn.Module) -> torch.Tensor:
        ctx.f_block = f_block
        ctx.g_block = g_block
        
        # x를 채널/히든 차원 기준으로 반으로 분할
        x1, x2 = torch.chunk(x, chunks=2, dim=-1)
        
        with torch.no_grad():
            f_x2 = f_block(x2)
            y1 = x1 + f_x2
            g_y1 = g_block(y1)
            y2 = x2 + g_y1
            
        y = torch.cat([y1, y2], dim=-1)
        # 활성화 메모리를 저장하지 않고 오직 출력 텐서의 detach 형태만 컨텍스트에 보관
        ctx.save_for_backward(y.detach())
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        f_block = ctx.f_block
        g_block = ctx.g_block
        (y_detached,) = ctx.saved_tensors
        
        y1, y2 = torch.chunk(y_detached, chunks=2, dim=-1)
        dy1, dy2 = torch.chunk(grad_output, chunks=2, dim=-1)
        
        # 1. 역함수로 x2 복원: x2 = y2 - G(y1)
        with torch.no_grad():
            g_y1 = g_block(y1)
            x2 = y2 - g_y1
            
        # 2. G 블록 역전파 계산 (G(y1)에 대한 기울기)
        with torch.enable_grad():
            y1_var = y1.detach().requires_grad_(True)
            g_out = g_block(y1_var)
            g_out.backward(dy2)
            grad_y1_from_g = y1_var.grad
            
        total_dy1 = dy1 + grad_y1_from_g
        
        # 3. 역함수로 x1 복원: x1 = y1 - F(x2)
        with torch.no_grad():
            f_x2 = f_block(x2)
            x1 = y1 - f_x2
            
        # 4. F 블록 역전파 계산 (F(x2)에 대한 기울기)
        with torch.enable_grad():
            x2_var = x2.detach().requires_grad_(True)
            f_out = f_block(x2_var)
            f_out.backward(total_dy1)
            grad_x2_from_f = x2_var.grad
            
        total_dx2 = dy2 + grad_x2_from_f
        total_dx1 = total_dy1
        
        dx = torch.cat([total_dx1, total_dx2], dim=-1)
        return dx, None, None


class BioPhysReversibleLayer(nn.Module):
    """실제 학습 시 메모리를 절감하는 BioPhys 가역 잔차 트랜스포머 레이어"""

    def __init__(self, f_block: nn.Module, g_block: nn.Module):
        super().__init__()
        self.f_block = f_block
        self.g_block = g_block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and x.requires_grad:
            return LandauerReversibleFunction.apply(x, self.f_block, self.g_block)
        else:
            # 추론 시 빠른 순전파
            x1, x2 = torch.chunk(x, chunks=2, dim=-1)
            y1 = x1 + self.f_block(x2)
            y2 = x2 + self.g_block(y1)
            return torch.cat([y1, y2], dim=-1)
