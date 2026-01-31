# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 17:31:57 2025

@author: joeco
entry point to the program
"""

from google_maps import get_travel_time_matrix, get_address_set, get_directions_matrix
from experiment_logger import save_experiment_results
from optimization_engines import tsp_brute_force, Heuristic_next_closest, Heuristic_weighted_next_closest, SA_approx
from quantum_engines import QAOA_approx
from quantum_helpers import get_warm_start_tour, create_qubit_to_edge_map
from quantum_pretraining import pretrain_and_create_initial_params
from opt_helpers import graphs_to_tours
from visualization_algorithms import plot_multiple_routes_comparison, plot_route_on_map, plot_runtime_comparison
from visualization_algorithms import plot_travel_times_violin, plot_tour_comparison, plot_edge_weight_heatmap
from visualization_algorithms import plot_qaoa_comprehensive_progress, plot_valid_solution_hamming_distances
from visualization_algorithms import plot_hamming_distance_histogram
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import json
   

# get data for graphs
curr_prob = '4m_10_1'
ttm = get_travel_time_matrix(f'{curr_prob}/ttm.txt')
address_set = get_address_set(f'{curr_prob}/address-set.txt')
dirs_mat = get_directions_matrix(f'{curr_prob}/dir-mat.json')


# base graph
graph = nx.from_numpy_array(np.array(ttm), create_using=nx.DiGraph)
# graphs_dict = {'Base Graph': graph}
graphs_dict = {}
runtime_data = {}
labelled_tt_data = {}
qaoa_progress = {}


# Brute-Force search graph
graphs_dict, runtime_data, labelled_tt_data, valid_tours, all_times = tsp_brute_force(graph, graphs_dict, runtime_data, labelled_tt_data)
labelled_tt_data['Worst-Case'] = max(all_times)
print('BFS completed')

'''
# next-nearest heuristic graph
graphs_dict, runtime_data, labelled_tt_data = Heuristic_next_closest(graph, graphs_dict, runtime_data, labelled_tt_data)
print('NNH completed')


# Weighted next-nearest heuristic graph
graphs_dict, runtime_data, labelled_tt_data = Heuristic_weighted_next_closest(graph, graphs_dict, runtime_data, labelled_tt_data)
print('WNN completed')


# simulated annealing approximations (3)
graphs_dict, runtime_data, labelled_tt_data = SA_approx(graph, graphs_dict, runtime_data, labelled_tt_data, label='Simulated-Annealing-1')
print('SA-1 completed')

graphs_dict, runtime_data, labelled_tt_data = SA_approx(graph, graphs_dict, runtime_data, labelled_tt_data, label='Simulated-Annealing-2')
print('SA-2 completed')

graphs_dict, runtime_data, labelled_tt_data = SA_approx(graph, graphs_dict, runtime_data, labelled_tt_data, label='Simulated-Annealing-3')
print('SA-3 completed')
'''

# QAOA approximations
# common QAOA arguments
shots = 10000
inv_penalty_m = 4.5
layers = 5
exploration_strength = 0.1

#TODO: explore futher the quantum pre-training approach, test more aggressive pretraining.
# =============================================================================
# Example 1: QAOA with different warm-start methods
# =============================================================================
# Test different warm-start methods to see which performs best

'''
warm_start_methods = [
    'nearest_neighbor',      # Original greedy approach
    'farthest_insertion',    # Build tour by inserting farthest nodes
    'cheapest_insertion',    # Build tour by cheapest insertion cost
    'random_nearest_neighbor',  # Randomized greedy
    'random'                 # Completely random tour
]


for method in warm_start_methods:
    graphs_dict, runtime_data, labelled_tt_data, qaoa_progress = QAOA_approx(
        graph, graphs_dict, runtime_data, 
        labelled_tt_data, qaoa_progress, 
        shots=shots, 
        inv_penalty_m=inv_penalty_m,
        layers=layers,
        warm_start=method,
        exploration_strength=exploration_strength,
        label=f'QAOA-{method}',
        initialization_strategy='zero'
    )
    print(f'QAOA with {method} warm-start completed')
'''

# =============================================================================
# Example 2: QAOA with quantum pretraining
# =============================================================================
# Pre-train the first layer(s) to maximize validity, then optimize for cost


print("\n" + "="*70)
print("PRE-TRAINING QAOA FOR VALIDITY")
print("="*70)

pretrained_params, validity_rates = pretrain_and_create_initial_params(
    graph,
    num_pretrain_layers=3,
    total_layers=layers,
    shots=2048,
    batch_size=8,
    max_iterations=30,
    use_local_2q_gates=True,
    verbose=False
)

print(f"\nPre-training achieved {validity_rates[0]:.2%} validity rate")
print("Now running full QAOA with pre-trained initialization...\n")

graphs_dict, runtime_data, labelled_tt_data, qaoa_progress = QAOA_approx(
    graph, graphs_dict, runtime_data, 
    labelled_tt_data, qaoa_progress, 
    shots=shots, 
    inv_penalty_m=inv_penalty_m,
    layers=layers,
    exploration_strength=exploration_strength,
    warm_start=None,
    label='QAOA-Pretrained-L3-2Q',
    custom_initial_params=pretrained_params,
    use_soft_validity=True
)
print('QAOA with pre-trained layer 0 completed')


# =============================================================================
# Example 3: Combining warm-start AND pretraining
# =============================================================================

'''
print("\n" + "="*70)
print("COMBINING WARM-START AND PRETRAINING")
print("="*70)

# Pre-train with nearest neighbor warm-start
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
    warm_start='nearest_neighbor',
    exploration_strength=exploration_strength,
    label='QAOA-WS+Pretrained',
    custom_initial_params=pretrained_params
)
print('QAOA with warm-start AND pretraining completed')
'''

# =============================================================================
# Example 4: Standard QAOA without warm-start or pretraining (baseline)
# =============================================================================

'''
graphs_dict, runtime_data, labelled_tt_data, qaoa_progress = QAOA_approx(
    graph, graphs_dict, runtime_data, 
    labelled_tt_data, qaoa_progress, 
    shots=shots, 
    inv_penalty_m=inv_penalty_m,
    layers=layers,
    exploration_strength=exploration_strength,
    warm_start=None,
    label='QAOA-Baseline-2Q',
    initialization_strategy='zero',
    use_local_2q_gates=True
)
print('QAOA baseline completed')
'''

# =============================================================================
# Visualizations and data saving
# =============================================================================
save_all = True


experiment_results = {
    'graphs_dict': graphs_dict,
    'runtime_data': runtime_data,
    'tt_data': labelled_tt_data,
    'qaoa_progress': qaoa_progress,  # Contains ALL QAOA runs
    'valid_tours': valid_tours,
    'all_times': all_times
    }

hyperparameters = {
    'layers': layers,
    'shots': shots,
    'qubit_batch_size': 8,
    'inv_penalty_m': inv_penalty_m,
    'warm_start': 'ALL',
    'exploration_strength': exploration_strength,
    'initialization_strategy': 'zero'
    }


# =============================================================================
# NEW: QAOA Comprehensive Progress Plots (individual for each run)
# =============================================================================
if qaoa_progress:  # Only if we have QAOA results
    print("\nCreating QAOA comprehensive progress plots...")
    from visualization_algorithms import plot_qaoa_comparison, plot_qaoa_final_comparison_bars
    
    figs, axes = plot_qaoa_comprehensive_progress(qaoa_progress, figsize=(14, 10))
    
    # Check if we got single or multiple figures
    if isinstance(figs, list):
        # Multiple runs - save each one
        for i, (label, fig) in enumerate(zip(qaoa_progress.keys(), figs)):
            if save_all:
                safe_label = label.replace('/', '_').replace(' ', '_')
                fig.savefig(f'{curr_prob}/results/qaoa_progress_{safe_label}.png', 
                           dpi=150, bbox_inches='tight')
                print(f"  Saved: qaoa_progress_{safe_label}.png")
    else:
        # Single run
        if save_all:
            label = list(qaoa_progress.keys())[0]
            safe_label = label.replace('/', '_').replace(' ', '_')
            figs.savefig(f'{curr_prob}/results/qaoa_progress_{safe_label}.png',
                        dpi=150, bbox_inches='tight')
            print(f"  Saved: qaoa_progress_{safe_label}.png")


# =============================================================================
# NEW: QAOA Multi-Run Comparison (side-by-side)
# =============================================================================
if len(qaoa_progress) > 1:  # Only if we have multiple QAOA runs
    print("\nCreating QAOA multi-run comparison plot...")
    
    fig_comp, axes_comp = plot_qaoa_comparison(qaoa_progress, figsize=(16, 10))
    
    if save_all:
        fig_comp.savefig(f'{curr_prob}/results/qaoa_comparison_all_runs.png',
                        dpi=150, bbox_inches='tight')
        print("  Saved: qaoa_comparison_all_runs.png")


# =============================================================================
# OPTIONAL: QAOA Final Statistics Bar Chart
# =============================================================================
if len(qaoa_progress) > 1:  # Only if we have multiple QAOA runs
    print("\nCreating QAOA final statistics bar chart...")
    
    fig_bars, axes_bars = plot_qaoa_final_comparison_bars(qaoa_progress, figsize=(14, 5))
    
    if save_all:
        fig_bars.savefig(f'{curr_prob}/results/qaoa_final_stats_bars.png',
                        dpi=150, bbox_inches='tight')
        print("  Saved: qaoa_final_stats_bars.png")


# =============================================================================
# Classical Algorithm Visualizations
# =============================================================================
# make bar chart for runtimes
fig, ax = plot_runtime_comparison(
    runtime_data,
    title="TSP Algorithm Runtime Comparison: 10 nodes",
    ylabel="Runtime (seconds, log scale)"
)

if save_all:
    plt.savefig(f'{curr_prob}/results/sols_runtime.png')


# make violin plot for trip times
fig, ax = plot_travel_times_violin(
    all_times, 
    labeled_points=labelled_tt_data,
    title="TSP Algorithm Performance Comparison",
    ylabel="Total Travel Time (seconds)"
)

if save_all:
    plt.savefig(f'{curr_prob}/results/sols_distrib.png')
    
    
# Plot graphs comparison
fig, axes = plot_tour_comparison(graphs_dict, layout='circular')

if save_all:
    plt.savefig(f'{curr_prob}/results/sols_found.png')



# plot maps comparison
tours_dict = graphs_to_tours(graphs_dict)


# plot just brute-force
fig, ax = plot_route_on_map(tours_dict['Brute-Force'], address_set, dirs_mat, 
                            title='Brute-Force', 
                            use_map_background=True, 
                            map_style='sattelite')

if save_all:
    plt.savefig(f'{curr_prob}/results/BFS_map.png')


# plot all routes
fig, axes = plot_multiple_routes_comparison(tours_dict, address_set, dirs_mat,
                                            use_map_background=True, 
                                            map_style='sattelite')

if save_all:
    plt.savefig(f'{curr_prob}/results/all_maps.png')


# plot heatmap edge weights
fig, ax = plot_edge_weight_heatmap(graph, title="TSP Edge Weights")

if save_all:
    plt.savefig(f'{curr_prob}/results/edge_weights.png')


# plot qaoa progress
fig, ax = plot_qaoa_comprehensive_progress(qaoa_progress)

if save_all:
    plt.savefig(f'{curr_prob}/results/qaoa_progress.png')


# Get valid tours from QAOA results
qubit_to_edge_map = create_qubit_to_edge_map(graph)

# Visualize distances between all solutions
fig, ax = plot_valid_solution_hamming_distances(valid_tours, qubit_to_edge_map, graph)

if save_all:
    plt.savefig(f'{curr_prob}/results/hamming_dist_50_sols.png')


# Or compare to a reference (e.g., greedy)
greedy_tour = get_warm_start_tour(graph, method='nearest_neighbor')
fig, ax = plot_hamming_distance_histogram(valid_tours, qubit_to_edge_map, 
                                          reference_tour=greedy_tour)

if save_all:
    plt.savefig(f'{curr_prob}/results/hamming_dist_hist.png')

# show plots
plt.show()


if save_all:
    # save data
    with open(f'{curr_prob}/tours.json', 'w') as f:
        json.dump(tours_dict, f)
        
        
    with open(f'{curr_prob}/runtimes.json', 'w') as f:
        json.dump(runtime_data, f)
        
        
    with open(f'{curr_prob}/travel-times.json', 'w') as f:
        json.dump(labelled_tt_data, f)
        
    
    with open(f'{curr_prob}/all-travel-times.json', 'w') as f:
        f.writelines([f'{i}, ' for i in all_times])


    save_experiment_results(
        experiment_name="QAOA_Multi_Run_Comparison",
        problem_name=curr_prob,
        results=experiment_results,
        hyperparameters=hyperparameters,
        notes="Comparing multiple QAOA configurations with different warm-starts and pretraining"
        )
