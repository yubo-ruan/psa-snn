# PSA-SNN: Predictive Sparse Assemblies for Robot Manipulation

A biologically-plausible spiking neural network architecture for robot manipulation learning. PSA uses **local learning rules only** - no backpropagation required.

## Key Features

- **Predictive World Model**: LIF neurons predict next observations, not neighbor spikes
- **3-Factor Learning Rule**: Local synaptic updates with eligibility traces
- **Language Conditioning**: Task instructions modulate neural thresholds and currents
- **Action Chunking**: K-step action prediction for smoother control
- **No Backprop**: All learning is local and biologically plausible

## Results Summary

| Component | Result |
|-----------|--------|
| World Model Generalization | 4x better than linear on OOD |
| Language Conditioning | 0.39 action divergence, 87.5% success |
| Action Chunking | 7.3% less jerk, 9.2% better precision |
| Eligibility Traces | +10% success on delayed tasks |

See [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md) for detailed results.

## Architecture

```
Observation → PredictivePSALayer → Spikes → ActionReadout → Action
                    ↓                           ↑
              Next Obs Prediction         Language Embedding
```

### Core Components

1. **PredictivePSALayer** - LIF neurons with surprise-gated plasticity
2. **ThreeFactorReadout** - Action readout with eligibility traces
3. **ChunkedActionReadout** - K-step action chunks for smooth control
4. **LanguageGatedPSALayer** - Language modulates threshold/current

### Learning Rules (All Local)

| Component | Update Rule |
|-----------|-------------|
| World model | Δw = lr × error × spike_rate × surprise |
| Action readout | Δw = lr × error × eligibility_trace |
| Language gating | Fixed random projection |

## Installation

```bash
pip install torch numpy
```

For LIBERO evaluation:
```bash
pip install libero robosuite
```

## Usage

### Basic Training
```python
from psa.language_conditioned import LanguageConditionedPSA

model = LanguageConditionedPSA(
    obs_dim=10,
    action_dim=4,
    num_neurons=128,
    language_dim=32,
)

# Set task
model.set_task_embedding(task_embedding)

# Forward pass
action, next_obs_pred, info = model.forward(obs)

# Local update (no backprop!)
model.update(actual_next_obs, demo_action)
```

### Running Tests
```bash
python test_language_conditioning.py  # Multi-task test
python test_action_chunking.py        # Chunking test
python test_eligibility_traces.py     # Temporal credit test
```

## File Structure

```
psa-snn/
├── psa/
│   ├── predictive_psa.py      # Core PSA layer
│   ├── action_readout.py      # 3-factor + eligibility + chunked
│   ├── language_conditioned.py # Language gating
│   └── neuron_vectorized.py   # Vectorized PSA
├── test_*.py                  # Test scripts
├── libero_eval.py             # LIBERO-style evaluation
└── RESULTS_SUMMARY.md         # Detailed results
```

## Citation

If you use this code, please cite:
```
@misc{psa-snn,
  title={PSA-SNN: Predictive Sparse Assemblies for Robot Manipulation},
  year={2024}
}
```

## License

MIT
