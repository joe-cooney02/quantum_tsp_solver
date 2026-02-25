# QAOA Visualization Guide

## Overview

New visualization functions have been added to better compare different QAOA runs with different hyperparameters (warm-starts, pretraining, etc.). These visualizations separate runs by their labels, similar to how classical algorithms are displayed.

## New Visualization Functions

### 1. `plot_qaoa_comprehensive_progress()` - Individual Run Analysis

**Purpose**: Create detailed 4-panel visualizations for each QAOA run separately.

**Usage**:
```python
from visualization_algorithms import plot_qaoa_comprehensive_progress

qaoa_progress = {
    'QAOA-Baseline': [...],  # List of stats dicts per iteration
    'QAOA-Warm-Start': [...],
    'QAOA-Pretrained': [...]
}

figs, axes = plot_qaoa_comprehensive_progress(qaoa_progress)
```

**Panels**:
1. **Cost Convergence** - Best tour cost over iterations with improvement markers
2. **Validity Rate** - Percentage of valid tours over time with average line
3. **Solution Diversity** - Number of unique bitstrings with average line
4. **Valid vs Invalid** - Stacked area showing distribution of shots

**Features**:
- ✅ Separate figure for each run (easy to save individually)
- ✅ Improvement stars on cost plot
- ✅ Average lines on validity and diversity
- ✅ Statistics summary box showing final results
- ✅ Legends on all subplots
- ✅ Professional formatting

**Returns**:
- Single run: `(fig, axes)` - one figure
- Multiple runs: `([figs], [axes])` - list of figures

---

### 2. `plot_qaoa_comparison()` - Side-by-Side Comparison

**Purpose**: Compare multiple QAOA runs on the same plots for direct comparison.

**Usage**:
```python
from visualization_algorithms import plot_qaoa_comparison

fig, axes = plot_qaoa_comparison(qaoa_progress, figsize=(16, 10))
```

**Panels**:
1. **Cost Convergence** - All runs overlaid with different colors
2. **Validity Rate** - All runs' validity percentages
3. **Solution Diversity** - All runs' unique bitstrings
4. **Valid Shot Fraction** - Alternative view of validity over time

**Features**:
- ✅ Color-coded by run for easy identification
- ✅ Summary statistics table in top-right corner
- ✅ Legends on all plots
- ✅ Consistent color scheme across all panels
- ✅ Final statistics for each run displayed

**Best For**: 
- Comparing 2-5 different QAOA configurations
- Identifying which settings work best
- Presentations and reports

---

### 3. `plot_qaoa_final_comparison_bars()` - Final Statistics Bars

**Purpose**: Bar chart comparison of final results across all runs.

**Usage**:
```python
from visualization_algorithms import plot_qaoa_final_comparison_bars

fig, (ax1, ax2, ax3) = plot_qaoa_final_comparison_bars(qaoa_progress)
```

**Charts**:
1. **Final Validity Rate** - Bar chart with 50% threshold line
2. **Final Best Cost** - Bar chart of final tour costs
3. **Average Diversity** - Bar chart of average unique bitstrings

**Features**:
- ✅ Value labels on top of each bar
- ✅ Color-coded bars
- ✅ Reference lines (e.g., 50% validity threshold)
- ✅ Clean, publication-ready format

**Best For**:
- Quick comparison of which run performed best
- Summarizing results in presentations
- Comparing many runs at once

---

## Comparison with Previous Visualizations

### Old Behavior
The previous `plot_qaoa_comprehensive_progress()` would only plot ONE run at a time and had a TODO to improve it.

### New Behavior
- **Individual plots**: Creates separate, detailed figures for each run
- **Comparison plots**: New functions for side-by-side and bar chart comparisons
- **Better organization**: Like classical algorithm plots, each QAOA run is clearly labeled

---

## Example Workflows

### Workflow 1: Analyze Single Run in Detail
```python
# Run QAOA
qaoa_progress = {}
graphs_dict, runtime_data, tt_data, qaoa_progress = QAOA_approx(
    graph, ..., label='QAOA-WS-NN', warm_start='nearest_neighbor'
)

# Plot comprehensive progress
fig, axes = plot_qaoa_comprehensive_progress(qaoa_progress)
plt.savefig('qaoa_ws_nn_progress.png')
```

### Workflow 2: Compare Multiple Configurations
```python
# Run multiple QAOA variants
qaoa_progress = {}

for method in ['nearest_neighbor', 'cheapest_insertion', 'farthest_insertion']:
    QAOA_approx(graph, ..., 
                warm_start=method, 
                label=f'QAOA-WS-{method}',
                qaoa_progress=qaoa_progress)

# Individual detailed plots
figs, axes = plot_qaoa_comprehensive_progress(qaoa_progress)
for i, fig in enumerate(figs):
    fig.savefig(f'qaoa_detail_{i}.png')

# Side-by-side comparison
fig_comp = plot_qaoa_comparison(qaoa_progress)
fig_comp[0].savefig('qaoa_comparison.png')

# Bar chart summary
fig_bars = plot_qaoa_final_comparison_bars(qaoa_progress)
fig_bars[0].savefig('qaoa_final_stats.png')
```

### Workflow 3: Compare Warm-Start vs Pretraining
```python
qaoa_progress = {}

# Baseline
QAOA_approx(graph, ..., label='Baseline', qaoa_progress=qaoa_progress)

# Warm-start
QAOA_approx(graph, ..., warm_start='nearest_neighbor', 
            label='Warm-Start', qaoa_progress=qaoa_progress)

# Pretraining
pretrained_params, _ = pretrain_and_create_initial_params(graph, ...)
QAOA_approx(graph, ..., custom_initial_params=pretrained_params,
            label='Pretrained', qaoa_progress=qaoa_progress)

# Combined
QAOA_approx(graph, ..., warm_start='nearest_neighbor',
            custom_initial_params=pretrained_params,
            label='Combined', qaoa_progress=qaoa_progress)

# Compare all four
fig_comp, axes = plot_qaoa_comparison(qaoa_progress)
plt.show()
```

---

## Understanding the Plots

### Cost Convergence
**What to look for**:
- ⭐ **Gold stars** = improvements found
- **Downward trend** = optimization working
- **Plateau** = convergence or local minimum
- **Early improvements** = good initialization

**Interpretation**:
- Warm-starts typically show fewer early improvements (already start good)
- Pretraining may show steadier convergence
- Baseline often has more erratic convergence

### Validity Rate
**What to look for**:
- **High percentage (>60%)** = good constraint learning
- **Increasing trend** = algorithm learning constraints
- **Flat/decreasing** = not learning, may need adjustments

**Interpretation**:
- Warm-starts usually show immediate high validity
- Pretraining should show gradual improvement
- Low validity (<20%) = reconsider approach

### Solution Diversity
**What to look for**:
- **High diversity** = exploring solution space well
- **Low diversity** = may be stuck in local region
- **Increasing trend** = good exploration

**Interpretation**:
- Too low = overly constrained (warm-start too strong)
- Too high = not converging
- Balance needed for good performance

### Valid vs Invalid Distribution
**What to look for**:
- **Green area growing** = validity improving
- **Red area shrinking** = fewer wasted shots
- **Proportion** = efficiency of search

---

## Customization Options

### Adjust Figure Sizes
```python
# Larger for presentations
plot_qaoa_comparison(qaoa_progress, figsize=(20, 12))

# Smaller for papers
plot_qaoa_final_comparison_bars(qaoa_progress, figsize=(10, 4))
```

### Save Individual Plots
```python
# Get all figures
figs, axes = plot_qaoa_comprehensive_progress(qaoa_progress)

# Save each one
for label, fig in zip(qaoa_progress.keys(), figs):
    fig.savefig(f'qaoa_{label}_progress.png', dpi=300, bbox_inches='tight')
```

### Extract Specific Data
```python
# After plotting comparison
fig, axes = plot_qaoa_comparison(qaoa_progress)

# Access individual axes for further customization
ax_cost, ax_validity, ax_diversity, ax_fraction = axes.flatten()
ax_cost.set_ylim(150, 300)  # Adjust y-limits
```

---

## Tips for Best Results

### 1. Consistent Labeling
Use clear, descriptive labels:
```python
# Good labels
'QAOA-Baseline'
'QAOA-WS-NN'  # Warm-Start with Nearest Neighbor
'QAOA-PT-L1'  # Pretrained Layer 1
'QAOA-WS+PT'  # Combined

# Avoid
'test1', 'run2', 'qaoa'  # Too vague
```

### 2. Run Multiple Seeds
For random methods, run multiple times:
```python
for seed in [42, 123, 456]:
    QAOA_approx(graph, ..., 
                warm_start='random_nearest_neighbor',
                label=f'QAOA-RNN-seed{seed}',
                seed=seed)
```

### 3. Save Results
Always save your `qaoa_progress` dict:
```python
import json

# After running experiments
with open('qaoa_results.json', 'w') as f:
    json.dump(qaoa_progress, f)

# Later, load and visualize
with open('qaoa_results.json', 'r') as f:
    loaded_progress = json.load(f)

plot_qaoa_comparison(loaded_progress)
```

### 4. Combine with Classical Results
```python
# Plot QAOA comparison
fig_qaoa, _ = plot_qaoa_final_comparison_bars(qaoa_progress)

# Plot classical runtime comparison
fig_classical, _ = plot_runtime_comparison(runtime_data)

# Show side-by-side
plt.show()
```

---

## Troubleshooting

### Issue: "No data to plot!"
**Cause**: Empty `qaoa_progress` dictionary
**Fix**: Make sure you're passing `qaoa_progress` to `QAOA_approx()`:
```python
qaoa_progress = {}  # Initialize
QAOA_approx(graph, ..., qaoa_progress=qaoa_progress)  # Pass it in
```

### Issue: Plots look crowded
**Cause**: Too many runs or long labels
**Fix**: 
- Use abbreviations in labels
- Increase figure size
- Plot in batches (2-3 runs at a time)

### Issue: Can't see differences between runs
**Cause**: Runs are too similar or scale is wrong
**Fix**:
- Zoom in on y-axis
- Use bar chart comparison for final stats
- Check if runs are actually different

### Issue: Missing iterations
**Cause**: QAOA converged early or failed
**Fix**:
- Check QAOA logs for errors
- Verify optimization success
- Look at number of iterations completed

---

## Quick Reference

| Function | Purpose | Best For |
|----------|---------|----------|
| `plot_qaoa_comprehensive_progress()` | Individual run details | Single run analysis, detailed examination |
| `plot_qaoa_comparison()` | Multi-run overlay | Comparing 2-5 runs, seeing trends |
| `plot_qaoa_final_comparison_bars()` | Final statistics | Quick comparison, many runs, presentations |

## Examples Location

See `visualization_examples.py` for complete working examples of all three functions.

Run examples:
```bash
python visualization_examples.py
```

This will generate all visualizations including the new QAOA comparison plots.
