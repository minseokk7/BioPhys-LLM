"""
사용자 PC 하드웨어 실시간 자동 감지 및 바이오-물리학 최적 파라미터 오토튜너
"""

import os
import psutil
import torch


def detect_and_autotune_hardware():
    """사용자 PC 하드웨어를 감지하여 최적의 바이오-물리학 파라미터를 계산합니다."""
    # 1. CPU 및 RAM 감지
    cpu_count = os.cpu_count() or 8
    physical_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
    
    # 2. GPU VRAM 감지 (NVIDIA CUDA / PyTorch)
    has_cuda = torch.cuda.is_available()
    gpu_name = "CPU / Integrated Graphics"
    vram_gb = 0.0
    
    if has_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        
    # 3. 하드웨어 맞춤형 바이오-물리학 오토튜닝 파라미터 산출
    if vram_gb >= 20.0:
        # 고성능 GPU (RTX 3090, 4090 등)
        target_model = "Qwen 3.8 27B / Gemma 4 (Full Precision Wobble-FP4)"
        quant_mode = "Wobble-FP4 Native (4-bit)"
        bec_ratio = 0.30
        speculative_lookahead = 5
        recommended_threads = min(cpu_count, 16)
    elif vram_gb >= 10.0 or physical_ram_gb >= 32.0:
        # 중고성능 환경 (RTX 4070, 3080, 32GB RAM)
        target_model = "Qwen 3.8 27B (Hybrid 2/4-bit) & Gemma 4 9B"
        quant_mode = "Wobble-FP4 Hybrid"
        bec_ratio = 0.25
        speculative_lookahead = 4
        recommended_threads = min(cpu_count, 12)
    else:
        # 일반 PC / 노트북 환경 (8GB~16GB RAM)
        target_model = "Gemma 4 2B/9B & Qwen 7B (Ultra-Compact BioPhys)"
        quant_mode = "Wobble-FP4 Ultra-Light"
        bec_ratio = 0.20
        speculative_lookahead = 3
        recommended_threads = min(cpu_count, 8)
        
    tuning_profile = {
        "cpu_cores": cpu_count,
        "physical_ram_gb": round(physical_ram_gb, 2),
        "available_ram_gb": round(available_ram_gb, 2),
        "has_cuda": has_cuda,
        "gpu_name": gpu_name,
        "vram_gb": round(vram_gb, 2),
        "target_model": target_model,
        "quant_mode": quant_mode,
        "bec_compression_ratio": bec_ratio,
        "speculative_lookahead": speculative_lookahead,
        "recommended_threads": recommended_threads,
    }
    return tuning_profile
