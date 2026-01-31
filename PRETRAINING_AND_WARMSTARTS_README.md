# Quantum Pretraining and Warm-Start Methods

This document explains the new features added to improve QAOA performance on the TSP problem.

## Overview

Two major enhancements have been implemented:

1. **Quantum Pretraining** (`quantum_pretraining.py`): Pre-train QAOA layers to maximize valid solution probability
2. **Multiple Warm-Start Methods** (`opt_helpers.py`): Various classical heuristics for initializing QAOA

## Problem Context

When using QAOA for TSP with batched simulation (limited to single-qubit gates), we face a challenge: many measurement outcomes are invalid tours. This wastes computational resources and makes optimization difficult.

**Solution Strategy:**
- Pre-train early QAOA layers to "learn" the valid solution space
- Initialize the circuit with classical heuristic solutions (warm-starts)
- Combine both approaches for best results

---

## 1. Quantum Pretraining

### What is it?

Quantum pretraining optimizes the first one or more QAOA layers to maximize the probability of measuring valid TSP tours, independent of tour cost. This teaches the quantum circuit to respect TSP constraints before optimizing for solution quality.

### Why does it work?

By dedicating early layers to validity, the optimization can:
1. Explore primarily within the valid solution space
2. Reduce wasted measurements on invalid tours
3. Allow later layers to focus purely on cost minimization

### How to use it

```python
from quantum_pretraining import pretrain_and_create_initial_params

# Pre-train first layer for validity
pretrained_params, validity_rates = pretrain_and_create_initial_params(
    graph,
    num_pretrain_layers=1,  # How many layers to pre-train
    total_layers=3,          # Total QAOA layers you'll use
    shots=2048,              # Fewer shots OK for pretraining
    max_iterations=30,       # Iterations per layer
    verbose=True
)

# Use in QAOA
graphs_dict, runtime_data, tt_data, qaoa_progress = QAOA_approx(
    graph, graphs_dict, runtime_data, tt_data, qaoa_progress,
    layers=3,
    shots=10000,
    label='QAOA-Pretrained',
    custom_initial_params=pretrained_params  # Key: use pretrained params
)
```

### Key Functions

**`pretrain_validity_layer()`**
- Trains a single layer to maximize validity
- Returns optimal gamma/beta values for that layer

**`pretrain_multiple_layers()`**
- Trains multiple layers sequentially
- Each layer builds on previous layers

**`pretrain_and_create_initial_params()`**
- Convenience function: pretrain + create full parameter set
- Ready to pass directly to QAOA

### Parameters

- `num_pretrain_layers`: How many layers to focus on validity (typically 1)
- `total_layers`: Total QAOA depth (remaining layers initialized with zeros/random)
- `shots`: Measurement shots for pretraining evaluation
- `max_iterations`: Optimization iterations per layer
- `batch_size`: Qubit batch size (8 works well for most PCs)

### When to use it

✅ **Use pretraining when:**
- Validity rate is low (<20%) with standard QAOA
- You have time for an upfront training phase
- Working with large problems where validity is hard

❌ **Skip pretraining when:**
- Warm-starts already give >80% validity
- Need very fast results
- Small problem sizes (validity already high)

---

## 2. Multiple Warm-Start Methods

### What are they?

Warm-start methods initialize the QAOA circuit in a classical heuristic solution rather than uniform superposition. This biases the search toward known good solutions while allowing quantum exploration.

### Available Methods

#### 1. **Nearest Neighbor** (Default)
```python
warm_start='nearest_neighbor'
```
- Greedy: always pick the nearest unvisited node
- Fast, usually gives good results
- Original method in the codebase

#### 2. **Farthest Insertion**
```python
warm_start='farthest_insertion'
```
- Builds tour by repeatedly inserting the farthest unvisited node
- Good for spread-out node distributions
- Can find different local optima than nearest neighbor

#### 3. **Cheapest Insertion**
```python
warm_start='cheapest_insertion'
```
- Inserts each node at the position with minimum cost increase
- Often finds better solutions than nearest neighbor
- More computation but better initial quality

#### 4. **Random Nearest Neighbor**
```python
warm_start='random_nearest_neighbor'
```
- Like nearest neighbor but picks randomly from top 3 nearest
- Adds diversity while maintaining reasonable quality
- Good for multiple runs with different seeds

#### 5. **Random**
```python
warm_start='random'
```
- Completely random valid tour
- Useful as a baseline
- Can help escape local minima

### How to use warm-starts

```python
from quantum_engines import QAOA_approx

# Basic usage - single warm-start method
graphs_dict, runtime_data, tt_data, qaoa_progress = QAOA_approx(
    graph, graphs_dict, runtime_data, tt_data, qaoa_progress,
    shots=10000,
    layers=3,
    warm_start='cheapest_insertion',  # Choose a method
    exploration_strength=0.2,          # How much to explore (0-1)
    label='QAOA-CI'
)

# Compare multiple methods
methods = ['nearest_neighbor', 'farthest_insertion', 'cheapest_insertion']
for method in methods:
    graphs_dict, runtime_data, tt_data, qaoa_progress = QAOA_approx(
        graph, graphs_dict, runtime_data, tt_data, qaoa_progress,
        warm_start=method,
        exploration_strength=0.2,
        label=f'QAOA-{method}'
    )
```

### Exploration Strength

The `exploration_strength` parameter controls how much the circuit explores away from the initial warm-start:

- `0.0`: Stay close to warm-start (minimal exploration)
- `0.1-0.3`: Balanced exploration (recommended)
- `0.5+`: High exploration (more like standard QAOA)

```python
# Conservative: stick close to warm-start
QAOA_approx(..., warm_start='nearest_neighbor', exploration_strength=0.0)

# Balanced: allow some exploration
QAOA_approx(..., warm_start='nearest_neighbor', exploration_strength=0.2)

# Aggressive: significant exploration
QAOA_approx(..., warm_start='nearest_neighbor', exploration_strength=0.5)
```

### Comparing Warm-Start Quality

You can evaluate warm-start methods before running QAOA:

```python
from opt_helpers import get_warm_start_tour

methods = ['nearest_neighbor', 'cheapest_insertion', 'farthest_insertion']

for method in methods:
    tour = get_warm_start_tour(graph, method=method)
    
    # Calculate cost
    cost = sum(graph[tour[i]][tour[i+1]]['weight'] 
               for i in range(len(tour)-1))
    
    print(f"{method}: {cost:.2f} seconds")
```

---

## 3. Combining Pretraining and Warm-Starts

You can use both techniques together for potentially best results:

```python
from quantum_pretraining import pretrain_and_create_initial_params

# Step 1: Pre-train parameters (doesn't use warm-start internally)
pretrained_params, validity_rates = pretrain_and_create_initial_params(
    graph,
    num_pretrain_layers=1,
    total_layers=3,
    shots=2048,
    max_iterations=30
)

# Step 2: Run QAOA with both pretrained params AND warm-start
graphs_dict, runtime_data, tt_data, qaoa_progress = QAOA_approx(
    graph, graphs_dict, runtime_data, tt_data, qaoa_progress,
    layers=3,
    shots=10000,
    warm_start='nearest_neighbor',      # Use warm-start
    exploration_strength=0.1,            # Small exploration
    custom_initial_params=pretrained_params,  # Use pretrained params
    label='QAOA-Combined'
)
```

**Expected behavior:**
- Warm-start biases initial state toward good solution
- Pretrained parameters guide evolution toward valid tours
- Combination can give high validity + good cost

---

## 4. Example Workflows

### Workflow 1: Quick Test
```python
# Just use warm-start with nearest neighbor
QAOA_approx(graph, ..., 
            warm_start='nearest_neighbor',
            exploration_strength=0.2)
```

### Workflow 2: Comprehensive Comparison
```python
# Test all warm-start methods
for method in ['nearest_neighbor', 'cheapest_insertion', 
               'farthest_insertion', 'random_nearest_neighbor']:
    QAOA_approx(graph, ..., 
                warm_start=method,
                exploration_strength=0.2,
                label=f'QAOA-{method}')
```

### Workflow 3: Maximum Performance
```python
# Pre-train first
pretrained_params, _ = pretrain_and_create_initial_params(
    graph, num_pretrain_layers=1, total_layers=3
)

# Then run with best warm-start
QAOA_approx(graph, ...,
            warm_start='cheapest_insertion',
            exploration_strength=0.1,
            custom_initial_params=pretrained_params)
```

---

## 5. Performance Tips

### Pretraining Tips

1. **Start with 1 layer**: Pretraining more layers takes significantly longer
2. **Use fewer shots**: 1024-2048 shots sufficient for pretraining
3. **Limit iterations**: 20-30 iterations per layer usually enough
4. **Check validity rates**: If pretraining achieves >60% validity, it's working

### Warm-Start Tips

1. **Try multiple methods**: Different problems favor different heuristics
2. **Balance exploration**: Too little (0.0) might get stuck, too much (0.5+) loses benefit
3. **Use cheapest_insertion for quality**: Often gives best initial tours
4. **Use random_nearest_neighbor for diversity**: Good for multiple independent runs

### General Tips

1. **Monitor validity percentage**: Track this in `qaoa_progress`
2. **Compare to baseline**: Always run baseline QAOA for comparison
3. **Save results**: Log everything for analysis
4. **Batch experiments**: Test multiple configurations in one run

---

## 6. Running the Examples

### Basic Example
```bash
python main.py
```
Uncomment different sections to test various configurations.

### Comprehensive Comparison
```bash
python example_pretraining_and_warmstarts.py
```
Runs all experiments and compares results.

### Quick Warm-Start Test
```python
from opt_helpers import get_warm_start_tour

# Compare warm-start qualities
for method in ['nearest_neighbor', 'cheapest_insertion']:
    tour = get_warm_start_tour(graph, method=method)
    print(f"{method}: cost = {calculate_tour_cost(tour)}")
```

---

## 7. Understanding the Results

### Key Metrics

**Validity Percentage**: Fraction of measurements that are valid tours
- <20%: Poor, most measurements wasted
- 20-50%: Moderate, room for improvement  
- 50-80%: Good, optimization working well
- >80%: Excellent, circuit learned constraints

**Final Tour Cost**: Total travel time of best tour found
- Compare to classical baselines (Brute-Force, Nearest Neighbor, etc.)
- Lower is better

**Runtime**: Time to complete QAOA optimization
- Pretraining adds upfront cost but may find better solutions faster
- Warm-starts typically don't add significant overhead

### Interpreting Plots

The `plot_qaoa_comprehensive_progress()` function shows:
1. Cost convergence over iterations
2. Validity percentage over time
3. Number of unique solutions found

Look for:
- ✅ Validity increasing or staying high
- ✅ Cost decreasing steadily
- ✅ Diverse solutions being explored

---

## 8. Troubleshooting

**Problem**: Pretraining isn't improving validity
- Solution: Try more iterations or different initialization strategy
- Check: Is the problem too constrained?

**Problem**: Warm-start gets stuck in local minimum
- Solution: Increase `exploration_strength` to 0.3-0.5
- Try: Different warm-start method

**Problem**: Too slow
- Solution: Reduce `shots` during pretraining (1024 is fine)
- Reduce `max_iterations` in pretraining (20 is often enough)
- Use only 1 pretrained layer

**Problem**: Results not reproducible
- Solution: Set `seed` parameter in warm-start methods
- Use consistent `shots` number (divisible by batch size)

---

## 9. Code Structure

```
quantum_pretraining.py
├── pretrain_validity_layer()          # Train single layer
├── pretrain_multiple_layers()         # Train multiple layers
├── create_pretrained_initial_params() # Combine pretrained + zeros
└── pretrain_and_create_initial_params() # Convenience wrapper

opt_helpers.py
└── get_warm_start_tour()
    ├── nearest_neighbor          # Greedy
    ├── farthest_insertion       # Farthest node first
    ├── cheapest_insertion       # Minimum cost increase
    ├── random_nearest_neighbor  # Randomized greedy
    └── random                   # Random tour

quantum_engines.py
└── QAOA_approx()
    └── custom_initial_params    # New parameter for pretrained values
```

---

## 10. Future Improvements

Potential enhancements:

1. **Adaptive exploration**: Adjust exploration_strength during optimization
2. **Layer-specific pretraining**: Train layer 0 for validity, layer 1 for cost structure, etc.
3. **Hybrid pretraining**: Combine warm-start with pretraining in training phase
4. **Meta-learning**: Learn which warm-start method works best for different graph structures

---

## Citation

If you use this implementation in your research, please cite:

```
Cooney, J. (2026). Quantum Pretraining and Multi-Heuristic Warm-Starts for QAOA TSP Solver.
Quantum Network Optimization Project. https://github.com/[your-repo]
```

---

## Questions?

For issues or questions:
1. Check the example files: `main.py`, `example_pretraining_and_warmstarts.py`
2. Review function docstrings in the source code
3. Open an issue on GitHub

Good luck with your quantum TSP solving! 🚀
