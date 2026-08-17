"""
BioPhys-LLM: The Grand Unified Bio-Physical Optimization Framework for Large Language Models
"""

from .core.attention import BioPhysUnifiedAttention
from .core.ffn import BioPhysUnifiedFFN
from .core.speculative import PredictiveSpeculativeEngine

__version__ = "1.0.0"
__author__ = "Advanced Agentic AI Research Initiative"

__all__ = [
    "BioPhysUnifiedAttention",
    "BioPhysUnifiedFFN",
    "PredictiveSpeculativeEngine",
]
