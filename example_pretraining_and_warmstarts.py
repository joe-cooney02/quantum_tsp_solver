# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 2026

@author: joeco

Example script demonstrating quantum pretraining and different warm-start methods.

This file shows various ways to configure QAOA for better performance:
1. Different warm-start heuristics
2. Quantum pretraining of validity layers
3. Combinations of warm-start and pretraining
"""

from google_maps import get_travel_time_matrix
from quantum_engines import QAOA_approx
from quantum_pretraining import pretrain_and_create_initial_params
from opt_helpers import get_warm_start_tour
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import json


# Load problem data
curr_prob = '4m_10_1'
ttm = get_travel_time_matrix(f'{curr_prob}/ttm.txt')
graph = nx.from_numpy_array(np.array(ttm), create_using=nx.DiGraph)

# Common parameters
shots = 10000
inv_penalty_m = 4.5
layers = 3

# Storage for results
graphs_dict = {}
runtime_data = {}
labelled_tt_data = {}
qaoa_progress = {}


# =============================================================================
# EXPERIMENT 1: Comparing different warm-start methods
# =============================================================================
print("\n" + "="*70)
print("EXPERIMENT 1: Testing Different Warm-Start Methods")
print("="*70 + "\n")

warm_start_methods = {
    'nearest_neighbor': 'Greedy - always pick nearest unvisited node',
    'farthest_insertion': 'Build tour by inserting farthest nodes',
    'cheapest_insertion': 'Build tour by cheapest insertion cost',
    'random_nearest_neighbor': 'Randomized greedy (pick from top 3)',
    'random': 'Completely random valid tour'
}

# First, compare the warm-start tours themselves
print("Comparing warm-start tour qualities:\n")
for method, description in warm_start_methods.items():
    tour = get_warm_start_tour(graph, method=method, seed=42)
    
    # Calculate tour cost
    cost = 0
    for i in range(len(tour) - 1):
        u, v = tour[i], tour[i+1]
        cost += graph[u][v]['weight']
    
    print(f"{method:25s}: {cost:8.2f} seconds - {description}")

print("\n" + "-"*70)
print("Now running QAOA with each warm-start method...")
print("-"*70 + "\n")

# Run QAOA with each warm-start method
for method in warm_start_methods.keys():
    print(f"\nRunning QAOA with {method} warm-start...")
    
    graphs_dict, runtime_data, labelled_tt_data, qaoa_progress = QAOA_approx(
        graph, graphs_dict, runtime_data, 
        labelled_tt_data, qaoa_progress, 
        shots=shots, 
        inv_penalty_m=inv_penalty_m,
        layers=layers,
        warm_start=method,
        exploration_strength=0.2,  # Allow some exploration
        label=f'QAOA-WS-{method}',
        initialization_strategy='zero'
    )
    
    # Print results
    final_cost = labelled_tt_data[f'QAOA-WS-{method}']
    runtime = runtime_data[f'QAOA-WS-{method}']
    final_stats = qaoa_progress[f'QAOA-WS-{method}'][-1]
    validity_pct = final_stats['valid_percentage']
    
    print(f"  Final tour cost: {final_cost:.2f} seconds")
    print(f"  Runtime: {runtime:.2f} seconds")
    print(f"  Final validity rate: {validity_pct:.1f}%")


# =============================================================================
# EXPERIMENT 2: Quantum pretraining without warm-start
# =============================================================================
print("\n\n" + "="*70)
print("EXPERIMENT 2: Quantum Pretraining (No Warm-Start)")
print("="*70 + "\n")

print("Pre-training first layer to maximize validity...")
pretrained_params, validity_rates = pretrain_and_create_initial_params(
    graph,
    num_pretrain_layers=1,
    total_layers=layers,
    shots=2048,  # Use fewer shots for faster pretraining
    batch_size=8,
    max_iterations=30,
    verbose=True
)

print(f"\nLayer 0 achieved {validity_rates[0]:.2%} validity during pretraining")
print("\nNow running full QAOA with pretrained parameters...\n")

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

final_cost = labelled_tt_data['QAOA-Pretrained-L0']
runtime = runtime_data['QAOA-Pretrained-L0']
final_stats = qaoa_progress['QAOA-Pretrained-L0'][-1]
validity_pct = final_stats['valid_percentage']

print("\nResults with pretraining:")
print(f"  Final tour cost: {final_cost:.2f} seconds")
print(f"  Runtime: {runtime:.2f} seconds")
print(f"  Final validity rate: {validity_pct:.1f}%")


# =============================================================================
# EXPERIMENT 3: Combining warm-start and pretraining
# =============================================================================
print("\n\n" + "="*70)
print("EXPERIMENT 3: Combining Warm-Start AND Pretraining")
print("="*70 + "\n")

print("Pre-training with nearest_neighbor warm-start...")
# Note: To properly combine these, we'd need to modify pretraining to support warm-starts
# For now, we'll just use the pretrained params with a warm-start circuit

graphs_dict, runtime_data, labelled_tt_data, qaoa_progress = QAOA_approx(
    graph, graphs_dict, runtime_data, 
    labelled_tt_data, qaoa_progress, 
    shots=shots, 
    inv_penalty_m=inv_penalty_m,
    layers=layers,
    warm_start='nearest_neighbor',
    exploration_strength=0.1,
    label='QAOA-Combined',
    custom_initial_params=pretrained_params
)

final_cost = labelled_tt_data['QAOA-Combined']
runtime = runtime_data['QAOA-Combined']
final_stats = qaoa_progress['QAOA-Combined'][-1]
validity_pct = final_stats['valid_percentage']

print("\nResults with combined approach:")
print(f"  Final tour cost: {final_cost:.2f} seconds")
print(f"  Runtime: {runtime:.2f} seconds")
print(f"  Final validity rate: {validity_pct:.1f}%")


# =============================================================================
# EXPERIMENT 4: Baseline for comparison
# =============================================================================
print("\n\n" + "="*70)
print("EXPERIMENT 4: Baseline QAOA (No Enhancements)")
print("="*70 + "\n")

graphs_dict, runtime_data, labelled_tt_data, qaoa_progress = QAOA_approx(
    graph, graphs_dict, runtime_data, 
    labelled_tt_data, qaoa_progress, 
    shots=shots, 
    inv_penalty_m=inv_penalty_m,
    layers=layers,
    warm_start=None,
    label='QAOA-Baseline',
    initialization_strategy='zero'
)

final_cost = labelled_tt_data['QAOA-Baseline']
runtime = runtime_data['QAOA-Baseline']
final_stats = qaoa_progress['QAOA-Baseline'][-1]
validity_pct = final_stats['valid_percentage']

print("\nBaseline results:")
print(f"  Final tour cost: {final_cost:.2f} seconds")
print(f"  Runtime: {runtime:.2f} seconds")
print(f"  Final validity rate: {validity_pct:.1f}%")


# =============================================================================
# SUMMARY: Compare all methods
# =============================================================================
print("\n\n" + "="*70)
print("SUMMARY: Comparison of All Methods")
print("="*70 + "\n")

# Sort by final tour cost
results = []
for label in labelled_tt_data.keys():
    if label in qaoa_progress:
        cost = labelled_tt_data[label]
        runtime_val = runtime_data[label]
        validity = qaoa_progress[label][-1]['valid_percentage']
        results.append((label, cost, runtime_val, validity))

results.sort(key=lambda x: x[1])  # Sort by cost

print(f"{'Method':<35s} {'Cost (s)':<12s} {'Runtime (s)':<12s} {'Validity %':<12s}")
print("-" * 70)
for label, cost, runtime_val, validity in results:
    print(f"{label:<35s} {cost:<12.2f} {runtime_val:<12.2f} {validity:<12.1f}")


# =============================================================================
# Save results
# =============================================================================
print("\n\nSaving results...")

with open(f'{curr_prob}/experiments/comparison_results.json', 'w') as f:
    json.dump({
        'travel_times': labelled_tt_data,
        'runtimes': runtime_data,
        'final_validity_rates': {
            label: qaoa_progress[label][-1]['valid_percentage'] 
            for label in qaoa_progress.keys()
        }
    }, f, indent=2)

print("Results saved to experiments/comparison_results.json")


# =============================================================================
# Visualize progress for best methods
# =============================================================================
print("\nGenerating visualization...")

# Plot QAOA progress for a few key methods
from visualization_algorithms import plot_qaoa_comprehensive_progress

selected_progress = {
    'QAOA-Baseline': qaoa_progress['QAOA-Baseline'],
    'QAOA-WS-nearest_neighbor': qaoa_progress['QAOA-WS-nearest_neighbor'],
    'QAOA-Pretrained-L0': qaoa_progress['QAOA-Pretrained-L0'],
    'QAOA-Combined': qaoa_progress['QAOA-Combined']
}

fig, ax = plot_qaoa_comprehensive_progress(selected_progress)
plt.suptitle('QAOA Performance Comparison: Different Enhancement Methods', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{curr_prob}/results/qaoa_methods_comparison.png', dpi=150, bbox_inches='tight')
print(f"Visualization saved to {curr_prob}/results/qaoa_methods_comparison.png")

plt.show()

print("\n" + "="*70)
print("Experiment complete!")
print("="*70)
