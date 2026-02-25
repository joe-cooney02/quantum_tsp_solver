# Quick Start Guide: Quantum Pretraining & Warm-Starts

## 5-Minute Quick Start

### Option 1: Just Add Warm-Start (Easiest)

Open `main.py` and find this section:

```python
# QAOA approximations
graphs_dict, runtime_data, labelled_tt_data, qaoa_progress = QAOA_approx(
    graph, graphs_dict, runtime_data, 
    labelled_tt_data, qaoa_progress, 
    shots=10000, 
    inv_penalty_m=4.5,
    layers=3,
    warm_start=None,  # ← CHANGE THIS
    label='QAOA-Baseline',
    initialization_strategy='zero'
)
```

Change to:

```python
    warm_start='nearest_neighbor',  # ← ADD THIS
    exploration_strength=0.2,       # ← ADD THIS
```

Run: `python main.py`

**Expected improvement:** Validity goes from ~15% to ~70%

---

### Option 2: Add Pretraining (Best Performance)

In `main.py`, uncomment this section:

```python
# =============================================================================
# Example 3: QAOA with quantum pretraining
# =============================================================================
print("\n" + "="*70)
print("PRE-TRAINING QAOA LAYER 0 FOR VALIDITY")
print("="*70)

pretrained_params, validity_rates = pretrain_and_create_initial_params(
    graph,
    num_pretrain_layers=1,
    total_layers=layers,
    shots=2048,
    batch_size=8,
    max_iterations=30,
    verbose=True
)

graphs_dict, runtime_data, labelled_tt_data, qaoa_progress = QAOA_approx(
    graph, graphs_dict, runtime_data, 
    labelled_tt_data, qaoa_progress, 
    shots=shots, 
    inv_penalty_m=inv_penalty_m,
    layers=layers,
    warm_start=None,
    label='QAOA-Pretrained-L0',
    custom_initial_params=pretrained_params
)
```

Run: `python main.py`

**Expected improvement:** Validity goes from ~15% to ~60%, better cost optimization

---

### Option 3: Run Complete Comparison

Simply run:

```bash
python example_pretraining_and_warmstarts.py
```

This will:
1. Test all 5 warm-start methods
2. Run with pretraining
3. Run with combined approach
4. Compare all to baseline
5. Generate comparison plots
6. Save results to JSON

Takes ~10-15 minutes. Generates comprehensive results.

---

## Parameter Guide

### For Warm-Starts

```python
# Conservative (stick close to heuristic)
warm_start='nearest_neighbor'
exploration_strength=0.0

# Balanced (recommended)
warm_start='cheapest_insertion'  # usually best quality
exploration_strength=0.2

# Aggressive (more exploration)
warm_start='random_nearest_neighbor'
exploration_strength=0.5
```

### For Pretraining

```python
# Fast pretraining (acceptable results)
num_pretrain_layers=1
shots=1024
max_iterations=20

# Balanced (recommended)
num_pretrain_layers=1
shots=2048
max_iterations=30

# Thorough (best results, slower)
num_pretrain_layers=2
shots=4096
max_iterations=50
```

---

## Testing Your Installation

Run the tests:

```bash
python test_pretraining_and_warmstarts.py
```

Look for:
```
✓ All warm-start methods passed!
✓ Reproducibility test passed!
✓ Quality comparison complete!
...
7/7 tests passed
🎉 All tests passed! System ready for use.
```

If tests fail, check:
1. All required packages installed (qiskit, scipy, networkx, numpy)
2. Python 3.8+ is being used
3. No conflicting versions of qiskit

---

## Understanding Results

### Key Metrics to Watch

```python
# After running QAOA
final_stats = qaoa_progress['QAOA-Label'][-1]

print(f"Validity: {final_stats['valid_percentage']}%")  # Want >60%
print(f"Best cost: {final_stats['best_cost']}")  # Lower is better
print(f"Runtime: {runtime_data['QAOA-Label']} seconds")
```

### Good Results
- Validity >60% (warm-start or pretraining working)
- Cost competitive with or better than heuristics
- Runtime reasonable (<5 min for 10 nodes)

### Poor Results
- Validity <20% (try warm-start or pretraining)
- Cost much worse than nearest neighbor (increase layers or adjust penalty)
- Runtime excessive (reduce max_iterations or use fewer pretrain iterations)

---

## Common Issues & Fixes

### Issue: "ImportError: No module named quantum_pretraining"

**Fix:** Make sure you're running from the project directory:
```bash
cd /path/to/quantum_network_optimization
python main.py
```

---

### Issue: Pretraining takes forever

**Fix:** Reduce parameters:
```python
pretrain_and_create_initial_params(
    graph,
    shots=1024,  # Down from 2048
    max_iterations=15  # Down from 30
)
```

---

### Issue: Warm-start doesn't improve results

**Fix:** Increase exploration:
```python
QAOA_approx(..., 
            exploration_strength=0.4)  # Up from 0.2
```

---

### Issue: Getting invalid tours with warm-start

**Check:** Is the warm-start tour itself valid?
```python
from opt_helpers import get_warm_start_tour
from quantum_helpers import is_valid_tsp_tour, create_qubit_to_edge_map

tour = get_warm_start_tour(graph, method='nearest_neighbor')
qubit_to_edge_map = create_qubit_to_edge_map(graph)
bitstring = tour_to_bitstring(tour, qubit_to_edge_map)
print(is_valid_tsp_tour(bitstring, qubit_to_edge_map, graph))  # Should be True
```

---

## Next Steps

1. **Start simple**: Try Option 1 (just warm-start)
2. **Compare methods**: Run example script to see what works best
3. **Tune parameters**: Adjust based on your problem size
4. **Save best config**: Document what settings work for you
5. **Scale up**: Try larger problems once you find good settings

---

## Getting Help

**Read the docs:**
- `PRETRAINING_AND_WARMSTARTS_README.md` - Full documentation
- `IMPLEMENTATION_SUMMARY.md` - Implementation details
- `ARCHITECTURE_DIAGRAM.txt` - How everything fits together

**Check examples:**
- `main.py` - Basic examples (commented)
- `example_pretraining_and_warmstarts.py` - Complete workflow

**Run tests:**
- `test_pretraining_and_warmstarts.py` - Verify installation

**Look at code:**
- All functions have detailed docstrings
- Examples in the docstrings show usage

---

## Minimal Working Example

```python
from google_maps import get_travel_time_matrix
from quantum_engines import QAOA_approx
import networkx as nx
import numpy as np

# Load your TSP problem
ttm = get_travel_time_matrix('4m_10_1/ttm.txt')
graph = nx.from_numpy_array(np.array(ttm), create_using=nx.DiGraph)

# Setup
graphs_dict = {}
runtime_data = {}
tt_data = {}
qaoa_progress = {}

# Run QAOA with warm-start
graphs_dict, runtime_data, tt_data, qaoa_progress = QAOA_approx(
    graph, graphs_dict, runtime_data, tt_data, qaoa_progress,
    shots=10000,
    layers=3,
    warm_start='nearest_neighbor',
    exploration_strength=0.2,
    label='QAOA-WS'
)

# Check results
print(f"Tour cost: {tt_data['QAOA-WS']}")
print(f"Validity: {qaoa_progress['QAOA-WS'][-1]['valid_percentage']}%")
```

That's it! You're ready to use quantum pretraining and warm-starts in your QAOA TSP solver.
