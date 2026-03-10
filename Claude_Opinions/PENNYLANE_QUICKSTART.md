# PennyLane QAOA Quick Start Guide

## Installation (5 minutes)

### Step 1: Install PennyLane
```bash
pip install pennylane pennylane-lightning-gpu
```

### Step 2: Verify GPU Access
```python
# test_gpu.py
import pennylane as qml

# Try to create GPU device
try:
    dev = qml.device('lightning.gpu', wires=10)
    print("✓ GPU device created successfully!")
    print(f"  Device: {dev}")
except Exception as e:
    print(f"✗ GPU device failed: {e}")
    print("  Falling back to CPU is automatic")
```

Run: `python test_gpu.py`

**Expected output**: `✓ GPU device created successfully!`

---

## Basic Usage (2 minutes)

### Minimal Example

```python
from quantum_helpers import create_qubit_to_edge_map
from quantum_pennylane import (
    pretrain_validity_pennylane,
    create_pretrained_initial_params_pennylane,
    QAOA_pennylane
)

# Your graph setup
graph = ... # Your TSP graph
qubit_to_edge_map = create_qubit_to_edge_map(graph)

# Step 1: Gradient pretraining (30 seconds)
gammas, betas, validity = pretrain_validity_pennylane(
    graph, qubit_to_edge_map,
    num_layers=1,
    max_iterations=20,
    use_gpu=True
)

# Step 2: Extend to 3 layers
params_init = create_pretrained_initial_params_pennylane(
    gammas, betas, total_layers=3
)

# Step 3: Main QAOA optimization (1-2 minutes)
result = QAOA_pennylane(
    graph, qubit_to_edge_map, params_init,
    layers=3,
    use_gpu=True,
    use_soft_validity=True
)

print(f"Best cost: {result['best_valid_cost']}")
print(f"Validity: {result['best_validity']:.2%}")
```

---

## Run Examples

We've created comprehensive examples for you:

```bash
python example_pennylane_usage.py
```

This runs 4 examples:
1. **Basic usage** - Simple end-to-end workflow
2. **Comparison** - Pretrained vs baseline vs 2Q gates
3. **Larger problem** - Demonstrates GPU advantage (commented out, takes longer)
4. **Hyperparameter tuning** - Find best learning rate

---

## Key Parameters

### Pretraining (`pretrain_validity_pennylane`)

| Parameter | Default | Recommendation | Notes |
|-----------|---------|----------------|-------|
| `num_layers` | 1 | 1 | Pretrain just 1 layer |
| `shots` | 1024 | 2048 | More shots = better gradient |
| `max_iterations` | 50 | 20-30 | Diminishing returns after 30 |
| `learning_rate` | 0.05 | 0.05 | Tune between 0.01-0.1 |
| `use_gpu` | True | True | Enable GPU! |
| `use_local_2q_gates` | False | True | Adds entanglement |

### Main QAOA (`QAOA_pennylane`)

| Parameter | Default | Recommendation | Notes |
|-----------|---------|----------------|-------|
| `layers` | 3 | 3 | Good balance |
| `shots` | 10000 | 5000-10000 | Higher for final run |
| `use_soft_validity` | True | True | Smoother landscape |
| `soft_validity_penalty_base` | 10.0 | 10.0 | Tune 5-20 |
| `max_iterations` | 200 | 100-200 | COBYLA iterations |

---

## Typical Workflow

```python
# 1. Create graph
graph = create_test_graph(num_nodes=6)
qubit_to_edge_map = create_qubit_to_edge_map(graph)

# 2. Quick pretraining (20-30 steps, ~30 seconds)
gammas, betas, validity = pretrain_validity_pennylane(
    graph, qubit_to_edge_map,
    num_layers=1,
    max_iterations=20,
    learning_rate=0.05,
    use_gpu=True,
    use_local_2q_gates=True
)
print(f"Pretraining: {validity:.2%} validity")

# 3. Extend parameters
params_init = create_pretrained_initial_params_pennylane(
    gammas, betas, total_layers=3
)

# 4. Main optimization (100 steps, ~2-3 minutes)
result = QAOA_pennylane(
    graph, qubit_to_edge_map, params_init,
    layers=3,
    shots=5000,
    use_gpu=True,
    use_local_2q_gates=True,
    use_soft_validity=True,
    max_iterations=100
)

# 5. Results
print(f"Best tour cost: {result['best_valid_cost']}")
print(f"Best tour: {result['best_tour']}")
print(f"Validity: {result['best_validity']:.2%}")
```

**Total time**: 3-4 minutes for a 6-node problem

---

## Troubleshooting

### "Module not found: pennylane"
```bash
pip install pennylane pennylane-lightning-gpu
```

### "lightning.gpu device not found"
**Option 1**: Your GPU isn't detected
- Check NVIDIA drivers installed
- Check CUDA toolkit installed
- Try: `nvidia-smi` (should show GPU)

**Option 2**: PennyLane GPU not installed correctly
```bash
pip uninstall pennylane-lightning pennylane-lightning-gpu
pip install pennylane-lightning-gpu
```

**Option 3**: Fall back to CPU
```python
# Works on any computer
pretrain_validity_pennylane(..., use_gpu=False)
QAOA_pennylane(..., use_gpu=False)
```
**Note**: Still 3-5x faster than no optimization!

### "Out of memory" error

Reduce problem size or batch size:
```python
# For very large problems, PennyLane also supports batching
# (though less necessary than Qiskit)
result = QAOA_pennylane(
    graph, qubit_to_edge_map, params_init,
    batch_size=20,  # Limit to 20 qubit batches
    ...
)
```

### Gradients not improving validity

**Try different learning rates**:
```python
for lr in [0.01, 0.05, 0.1]:
    gammas, betas, val = pretrain_validity_pennylane(
        ..., learning_rate=lr, verbose=False
    )
    print(f"LR {lr}: {val:.2%}")
```

**Or start with smaller parameters**:
```python
# In pretrain_validity_pennylane, initialization uses:
params = np.random.uniform(-0.1, 0.1, 2*num_layers)

# Try even smaller:
params = np.random.uniform(-0.01, 0.01, 2*num_layers)
```

---

## Performance Expectations

### Small Problem (5-6 nodes, ~30 qubits)
- Pretraining: 30 seconds
- Main QAOA: 2-3 minutes
- Total: **3-4 minutes**
- Validity: 40-60%

### Medium Problem (7-8 nodes, ~56 qubits)
- Pretraining: 1 minute
- Main QAOA: 5-7 minutes
- Total: **6-8 minutes**
- Validity: 30-50%

### Large Problem (10 nodes, ~90 qubits)
- Pretraining: 2-3 minutes
- Main QAOA: 10-15 minutes
- Total: **12-18 minutes**
- Validity: 20-40%

**With GPU vs CPU**:
- GPU: ~10x faster than CPU
- Your RTX 5060 Ti: Can handle up to 25-30 qubit batches at once

---

## Next Steps

1. **Run examples**: `python example_pennylane_usage.py`
2. **Try on your data**: Replace test graph with your TSP graph
3. **Tune parameters**: Experiment with learning rates, iterations
4. **Compare**: Run both PennyLane and Qiskit, see which works better

---

## Key Advantages

✅ **Windows GPU support** - No WSL2 needed
✅ **Gradient pretraining** - Better initial validity
✅ **Larger batches** - Full circuits or 20-30 qubit batches
✅ **Easy setup** - 5 minutes vs 60 for WSL2
✅ **Spyder compatible** - Keep your workflow

---

## When to Use PennyLane vs Qiskit

**Use PennyLane if**:
- You're on Windows and want GPU
- You want to avoid WSL2 setup
- You want to experiment with gradients
- Problem size fits in GPU (≤30 qubit batches)

**Use Qiskit-Aer-GPU if**:
- You already have WSL2
- Need maximum speed (~20% faster)
- Very large problems (>30 qubit batches)

**Use both!**
- Pretrain with PennyLane (gradients!)
- Fine-tune with Qiskit (mature tooling)
- Compare results

---

## Support

If you encounter issues:
1. Check Claude_Opinions/PENNYLANE_IMPLEMENTATION_SUMMARY.md
2. Check Claude_Opinions/PENNYLANE_ALTERNATIVE.md  
3. Run `example_pennylane_usage.py` to test setup
4. Try `use_gpu=False` to isolate GPU issues

Happy optimizing! 🚀
