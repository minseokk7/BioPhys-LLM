"""
BioPhys-LLM 3.6: Grand Unified Bio-Physical Optimization Framework
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
from biophys_llm.frontier.tensor_train_mps import TensorTrainMPSCompressor
from biophys_llm.frontier.chebyshev_harmonic import ChebyshevHarmonicCompressor

from biophys_llm.pure_science.dna_topoisomerase import DNATopoisomeraseCompressor
from biophys_llm.pure_science.lagrange_orbital import LagrangeOrbitalCompressor
from biophys_llm.pure_science.onsager_reciprocal import OnsagerReciprocalAttention
from biophys_llm.pure_science.brillouin_bandgap import BrillouinBandgapFilter
from biophys_llm.pure_science.verlinde_entropic import VerlindeEntropicForceOptimizer
from biophys_llm.pure_science.anthropic_observer import AnthropicObserverPruner
from biophys_llm.pure_science.symplectic_hamiltonian import SymplecticHamiltonianLayer
from biophys_llm.pure_science.superfluid_conduit import LandauSuperfluidConduit
from biophys_llm.pure_science.xylem_cohesion import XylemCohesionTensionPuller
from biophys_llm.pure_science.destructive_collision import DestructiveCollisionFilter
from biophys_llm.pure_science.collisional_damping import CollisionalDampingStabilizer
from biophys_llm.pure_science.volumetric_3d_tensor import Volumetric3DTensorRingLinear
from biophys_llm.pure_science.spherical_harmonics_3d import SphericalHarmonics3DAttention
from biophys_llm.pure_science.quantum_3d_tunneling import Quantum3DSuperpositionTunnelingLayer
from biophys_llm.pure_science.counterdiabatic_acceleration import CounterdiabaticQuantumAccelerator
from biophys_llm.pure_science.josephson_zero_bus import JosephsonZeroResistanceBus
from biophys_llm.pure_science.internal_reflection_waveguide import TotalInternalReflectionWaveguide
from biophys_llm.pure_science.supercritical_fusion import SupercriticalSinglePassEngine
from biophys_llm.pure_science.calabi_yau_compact import CalabiYau6DCompactifier
from biophys_llm.pure_science.superstring_vibration import SuperstringVibrationHarmonicDecoder
from biophys_llm.pure_science.t_duality_d_brane import TDualityDBraneAttention
from biophys_llm.pure_science.ergosphere_penrose import ErgosphereEnergyExtractor
from biophys_llm.pure_science.eyring_transition_state import EyringTransitionStateRouter
from biophys_llm.pure_science.atmospheric_jet_stream import AtmosphericJetStreamConveyor
from biophys_llm.pure_science.cryptochrome_quantum_compass import CryptochromeQuantumCompass

from biophys_llm.speed_opt.speculative_burst import NeuronalBurstDrafter
from biophys_llm.speed_opt.laminar_prefetch import LaminarPrefetchAccelerator
from biophys_llm.speed_opt.saltatory_conduction import SaltatoryLayerConductor
from biophys_llm.speed_opt.soliton_pulse_decoder import SolitonPulseDecoder

from biophys_llm.models.unified_transformer import BioPhysGrandUnifiedBlock

__version__ = "3.6.0"
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
    "TensorTrainMPSCompressor",
    "ChebyshevHarmonicCompressor",
    "DNATopoisomeraseCompressor",
    "LagrangeOrbitalCompressor",
    "OnsagerReciprocalAttention",
    "BrillouinBandgapFilter",
    "VerlindeEntropicForceOptimizer",
    "AnthropicObserverPruner",
    "SymplecticHamiltonianLayer",
    "LandauSuperfluidConduit",
    "XylemCohesionTensionPuller",
    "DestructiveCollisionFilter",
    "CollisionalDampingStabilizer",
    "Volumetric3DTensorRingLinear",
    "SphericalHarmonics3DAttention",
    "Quantum3DSuperpositionTunnelingLayer",
    "CounterdiabaticQuantumAccelerator",
    "JosephsonZeroResistanceBus",
    "TotalInternalReflectionWaveguide",
    "SupercriticalSinglePassEngine",
    "CalabiYau6DCompactifier",
    "SuperstringVibrationHarmonicDecoder",
    "TDualityDBraneAttention",
    "ErgosphereEnergyExtractor",
    "EyringTransitionStateRouter",
    "AtmosphericJetStreamConveyor",
    "CryptochromeQuantumCompass",
    "NeuronalBurstDrafter",
    "LaminarPrefetchAccelerator",
    "SaltatoryLayerConductor",
    "SolitonPulseDecoder",
    "BioPhysGrandUnifiedBlock",
]
