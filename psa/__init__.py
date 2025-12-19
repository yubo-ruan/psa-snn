"""
Predictive Sparse Assemblies (PSA)
A paradigm-shift SNN framework for sample-efficient robotics.

Key ideas:
1. Neurons predict neighbor spike rates (not exact times)
2. Assemblies form through low mutual prediction error
3. Surprise-gated plasticity (neuromodulation)
4. Fast/slow weight systems (episodic/consolidated)
5. Active inference for action selection
"""

from .neuron import PSANeuron, PSALayer
from .synapse import Synapse, DualWeightSynapse
from .assembly import AssemblyDetector
from .network import PSANetwork
from .agent import ActiveInferenceAgent
from .libero_adapter import (
    FrozenVisionEncoder,
    LanguageGating,
    HomeostaticCalibration,
    TwoTimescalePSA,
    ModularPSA,
    LIBEROPSAAgent,
)
from .metrics import (
    PredictionQualityTracker,
    SurpriseMapTracker,
    AssemblyQualityTracker,
    LifelongLearningTracker,
    PSAMetricsLogger,
)

__version__ = "0.1.0"
