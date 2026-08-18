"""
BioPhys-LLM 2.2: Grand Unified Bio-Physical Optimization Framework
"""

from biophys_llm.core.reversible import LandauerReversibleFunction, BioPhysReversibleLayer
from biophys_llm.core.attention import BioPhysUnifiedAttention
from biophys_llm.core.ffn import BioPhysUnifiedFFN
from biophys_llm.core.speculative import PredictiveSpeculativeEngine
from biophys_llm.core.spin_kv import TopologicalSpinKVCompressor
from biophys_llm.core.seismic_optimizer import SeismicOptimizer

from biophys_llm.bio_opt.epigenetic_masking import EpigeneticDomainMasker
from biophys_llm.bio_opt.metabolic_early_exit import MetabolicEarlyExitController
from biophys_llm.bio_opt.crick_wobble_quant import CrickWobbleQuantizer
from biophys_llm.bio_opt.qwen38_bio_wrapper import Qwen38BioPhysAdapter

from biophys_llm.frontier.mycelial_routing import MycelialAttentionRouter
from biophys_llm.frontier.phononic_wave_linear import PhononicPhaseLinear
from biophys_llm.frontier.dissipative_entropy import DissipativeEntropyRegularizer
from biophys_llm.frontier.persistent_homology import PersistentHomologyPruner

from biophys_llm.models.unified_transformer import BioPhysGrandUnifiedBlock

__version__ = "2.2.0"
__all__ = [
    "LandauerReversibleFunction",
    "BioPhysReversibleLayer",
    "BioPhysUnifiedAttention",
    "BioPhysUnifiedFFN",
    "PredictiveSpeculativeEngine",
    "TopologicalSpinKVCompressor",
    "SeismicOptimizer",
    "EpigeneticDomainMasker",
    "MetabolicEarlyExitController",
    "CrickWobbleQuantizer",
    "Qwen38BioPhysAdapter",
    "MycelialAttentionRouter",
    "PhononicPhaseLinear",
    "DissipativeEntropyRegularizer",
    "PersistentHomologyPruner",
    "BioPhysGrandUnifiedBlock",
]
