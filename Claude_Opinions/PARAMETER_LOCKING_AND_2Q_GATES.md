# Parameter Locking and Local 2-Qubit Gates Implementation

## Summary

Two major features have been implemented to enhance QAOA performance:

1. **Parameter Locking** - Lock pretrained layers during optimization
2. **Local 2-Qubit Gates** - Add entanglement within batches

---

## Feature 1: Parameter Locking

### Overview
Lock pretrained parameters so the optimizer only varies non-pretrained layers. This preserves learned validity while optimizing for cost.

### Usage

```python
# Pretrain layer 0
pretrained_params, validity_rates = pretrain_and_create_initial_params(
    graph, num_pretrain_layers=1, total_layers=3, shots=2048, max_iterations=30
)

# Option A: Soft lock (current default) - Allow optimizer to adjust all params
QAOA_approx(graph, ..., 
            custom_initial_params=pretrained_params,
            lock_pretrained_layers=0,  # No locking
            label='QAOA-Pretrained-Unlocked')

# Option B: Hard lock - Fix layer 0, optimize layers 1-2
QAOA_approx(graph, ...,
            custom_initial_params=pretrained_params,
            lock_pretrained_layers=1,  # Lock layer 0
            label='QAOA-Pretrained-Locked')
```

### How It Works

**Without Locking** (`lock_pretrained_layers=0`):
- Optimizer varies all 6 parameters: [γ₀, γ₁, γ₂, β₀, β₁, β₂]
- Pretrained values used as initial guess
- Optimizer may move away from pretrained solution

**With Locking** (`lock_pretrained_layers=1`):
- Layer 0 parameters fixed: γ₀ and β₀ stay at pretrained values
- Optimizer only varies 4 parameters: [γ₁, γ₂, β₁, β₂]
- Faster optimization (fewer dimensions)
- Preserves learned validity from pretraining

**Console Output**:
```
Parameter Locking Enabled:
  Locked layers: 0-0 (2 params)
  Optimizable layers: 1-2 (4 params)
  Locked gammas: [0.524]
  Locked betas: [1.047]
```

### When to Use

| Scenario | Recommendation | Reason |
|----------|---------------|--------|
| Pretraining achieves 70%+ validity | **Lock** | Preserve high validity |
| Pretraining achieves 40-60% validity | Try both | Test if locking helps |
| Pretraining achieves <40% validity | Don't lock | Not learning well enough |
| Multi-layer pretraining (2+ layers) | Lock pretrained layers | Focus optimization on new layers |

### Benefits

1. **Faster Convergence** - Fewer parameters to optimize (3 layers: 6→4 params)
2. **Stability** - Won't accidentally break learned validity
3. **Better for Sequential Training** - Lock layers 0-1, train layer 2

### Implementation Details

**Modified Files**:
- `quantum_engines.py` - Added `lock_pretrained_layers` parameter
- `quantum_engines.py::QAOA_approx()` - Splits params into locked/optimizable
- `quantum_engines.py::run_QAOA()` - Reconstructs full param array each iteration

**Parameter Format**:
- Input: `[γ₀, γ₁, γ₂, β₀, β₁, β₂]`
- Locked (layer 0): `[γ₀, β₀]`
- Optimizable (layers 1-2): `[γ₁, γ₂, β₁, β₂]`
- Reconstruction: `[γ₀, γ₁, γ₂, β₀, β₁, β₂]` (locked first, then optimizable)

---

## Feature 2: Local 2-Qubit Gates

### Overview
Add controlled-Z (CZ) gates between qubits within the same batch. This creates entanglement that may help QAOA learn better, while maintaining compatibility with batched simulation.

### Usage

```python
# Standard QAOA (no entanglement)
QAOA_approx(graph, ...,
            use_local_2q_gates=False,  # Default
            label='QAOA-1Q-Only')

# Enhanced QAOA (local entanglement)
QAOA_approx(graph, ...,
            use_local_2q_gates=True,
            qubit_batch_size=8,  # CZ gates only within each batch of 8
            label='QAOA-With-Entanglement')
```

### How It Works

**Circuit Structure** (3 layers, 10 qubits, batch_size=8):

```
Batch 0 (qubits 0-7):
  Layer 0:
    Cost: RZ(γ₀*weight) on each qubit
    Entanglement: CZ(0,1), CZ(2,3), CZ(4,5), CZ(6,7)  ← Even pairs
                  CZ(1,2), CZ(3,4), CZ(5,6)            ← Odd pairs
    Mixer: RX(β₀) on each qubit
  Layer 1: ... (same pattern)
  Layer 2: ... (same pattern)

Batch 1 (qubits 8-9):
  Layer 0:
    Cost: RZ(γ₀*weight) on qubits 8, 9
    Entanglement: CZ(8,9)  ← Only one pair in this batch
    Mixer: RX(β₀) on qubits 8, 9
  ...
```

**Key Points**:
- CZ gates **only** between qubits in same batch
- Pattern: Even pairs (0-1, 2-3, ...), then odd pairs (1-2, 3-4, ...)
- This creates a "brick wall" entangling layer
- Batches remain independent → can still simulate separately

### Why This Helps

1. **Entanglement** - Qubits can influence each other's measurements
2. **Richer Ansatz** - More expressive than single-qubit gates alone
3. **TSP Correlations** - Edges that share nodes can coordinate
4. **Still Scalable** - Batched simulation still works!

### Theory

Standard QAOA mixer is just:
```
H_mixer = Σᵢ σᵢˣ
```

With local 2-qubit gates, we get:
```
H_mixer = Σᵢ σᵢˣ + Σ_{i,j∈batch} σᵢᶻσⱼᶻ
```

This adds correlation terms between qubits in the same batch.

### Performance Expectations

**Without 2Q Gates** (baseline):
- Validity: Depends on warm-start/pretraining
- Solution quality: Good but may miss correlations
- Speed: Fastest (fewer gates)

**With 2Q Gates**:
- Validity: May improve (qubits coordinate better)
- Solution quality: Potentially better (more expressiveness)
- Speed: Slightly slower (more gates to simulate)
- Pretraining: Should work even better!

### Compatibility with Batched Simulation

**Important**: CZ gates are only added WITHIN batches, never across batch boundaries. This means:

✅ **Still works with batched simulation**
- Batch 0 (qubits 0-7) simulated independently
- Batch 1 (qubits 8-15) simulated independently
- Results combined as before

✅ **Scales to large problems**
- 100 qubits → ~13 batches of 8
- Each batch has local entanglement
- Still much faster than full simulation

❌ **Cannot add gates across batches**
- CZ(7, 8) would break batching!
- Current implementation enforces this

### Implementation Details

**Modified Files**:
- `quantum_helpers.py::create_tsp_qaoa_circuit()` - Added `use_local_2q_gates` parameter and CZ gate logic
- `quantum_helpers.py::create_warm_started_qaoa()` - Passes through 2Q parameter
- `quantum_engines.py::QAOA_approx()` - Added `use_local_2q_gates` parameter

**Gate Pattern**:
```python
# For batch with qubits [start, start+1, ..., end-1]:

# Even pairs: 0-1, 2-3, 4-5, ...
for i in range(start, end - 1, 2):
    circuit.cz(i, i + 1)

# Odd pairs: 1-2, 3-4, 5-6, ...
for i in range(start + 1, end - 1, 2):
    circuit.cz(i, i + 1)
```

This creates maximum connectivity within each batch while keeping the gates local.

---

## Combined Usage

### Example 1: Pretrained + Locked + Entanglement
```python
# Pretrain with entanglement
pretrained_params, validity_rates = pretrain_and_create_initial_params(
    graph, num_pretrain_layers=1, total_layers=3, shots=2048, max_iterations=30
)

# Use pretrained params, lock layer 0, add entanglement
QAOA_approx(graph, ...,
            custom_initial_params=pretrained_params,
            lock_pretrained_layers=1,
            use_local_2q_gates=True,
            label='QAOA-Full-Enhancement')
```

### Example 2: Warm-Start + Entanglement
```python
QAOA_approx(graph, ...,
            warm_start='nearest_neighbor',
            exploration_strength=0.2,
            use_local_2q_gates=True,
            label='QAOA-WS-Entangled')
```

### Example 3: Compare All Approaches
```python
qaoa_progress = {}

# Baseline
QAOA_approx(graph, ..., label='Baseline', qaoa_progress=qaoa_progress)

# With entanglement only
QAOA_approx(graph, ..., use_local_2q_gates=True,
            label='Entanglement-Only', qaoa_progress=qaoa_progress)

# Pretrained + Unlocked
QAOA_approx(graph, ..., custom_initial_params=pretrained_params,
            label='Pretrained-Unlocked', qaoa_progress=qaoa_progress)

# Pretrained + Locked
QAOA_approx(graph, ..., custom_initial_params=pretrained_params,
            lock_pretrained_layers=1,
            label='Pretrained-Locked', qaoa_progress=qaoa_progress)

# Full enhancement
QAOA_approx(graph, ..., custom_initial_params=pretrained_params,
            lock_pretrained_layers=1, use_local_2q_gates=True,
            label='Full-Enhancement', qaoa_progress=qaoa_progress)

# Visualize comparison
plot_qaoa_comparison(qaoa_progress)
```

---

## Testing Recommendations

### Test 1: Does Locking Help?
```python
pretrained_params, validity = pretrain_and_create_initial_params(...)

qaoa_progress = {}
QAOA_approx(graph, ..., custom_initial_params=pretrained_params,
            lock_pretrained_layers=0, label='Unlocked', qaoa_progress=qaoa_progress)
QAOA_approx(graph, ..., custom_initial_params=pretrained_params,
            lock_pretrained_layers=1, label='Locked', qaoa_progress=qaoa_progress)

plot_qaoa_comparison(qaoa_progress)
```

**Look for**:
- Does locked version maintain higher validity?
- Does locked version converge faster?
- Is final cost comparable?

### Test 2: Do 2Q Gates Help?
```python
qaoa_progress = {}
QAOA_approx(graph, ..., use_local_2q_gates=False,
            label='1Q-Only', qaoa_progress=qaoa_progress)
QAOA_approx(graph, ..., use_local_2q_gates=True,
            label='With-2Q', qaoa_progress=qaoa_progress)

plot_qaoa_comparison(qaoa_progress)
```

**Look for**:
- Higher validity with 2Q gates?
- Better final cost?
- More diverse solutions?

### Test 3: Full Comparison
Run all combinations:
- ±Warm-start
- ±Pretraining
- ±Locking (if pretrained)
- ±2Q gates

This is 16 combinations, but you can prioritize:
1. Baseline
2. Warm-start only
3. Pretrained-Locked + 2Q
4. Warm-start + 2Q

---

## Technical Notes

### Parameter Reconstruction
When `lock_pretrained_layers > 0`, the optimizer only sees the optimizable parameters, but `run_QAOA` reconstructs the full array:

```python
# Optimizer varies: [γ₁, γ₂, β₁, β₂]
# Locked: [γ₀, β₀]
# Reconstructed: [γ₀, γ₁, γ₂, β₀, β₁, β₂]
```

This is transparent to the circuit - it always receives all layer parameters.

### 2-Qubit Gate Overhead
- Each CZ gate adds minimal overhead
- For 8 qubits per batch: 7 CZ gates per layer
- For 3 layers: 21 CZ gates per batch
- Negligible compared to RZ/RX gates

### Limitations
- Cannot lock all layers (must optimize something!)
- 2Q gates only within batches (by design)
- Locking requires custom_initial_params

---

## Files Modified

| File | Changes |
|------|---------|
| `quantum_engines.py` | Added `lock_pretrained_layers` and `use_local_2q_gates` parameters |
| `quantum_helpers.py` | Added 2Q gate logic to circuit creation |
| Both | Updated function signatures and docstrings |

---

## Quick Reference

### Parameter Locking
```python
lock_pretrained_layers=0  # No locking (default)
lock_pretrained_layers=1  # Lock layer 0
lock_pretrained_layers=2  # Lock layers 0-1
```

### Local 2Q Gates
```python
use_local_2q_gates=False  # Standard QAOA (default)
use_local_2q_gates=True   # Add CZ gates within batches
```

### Typical Workflow
1. Run baseline QAOA
2. Test warm-starts (see which works best)
3. Test pretraining + locking
4. Add 2Q gates to best approach
5. Compare all with visualizations

---

Both features are ready to use! The implementations are backward-compatible (defaults preserve current behavior) and include comprehensive error checking.
