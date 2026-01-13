# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 17:31:57 2025

@author: joeco
entry point to the program
"""

from google_maps import get_travel_time_matrix, get_address_set, get_directions_matrix
from optimization_engines import tsp_brute_force, Heuristic_next_closest, Heuristic_weighted_next_closest, SA_approx
from quantum_engines import QAOA_approx
from opt_helpers import graphs_to_tours
from visualization_algorithms import plot_multiple_routes_comparison, plot_route_on_map, plot_runtime_comparison
from visualization_algorithms import plot_travel_times_violin, plot_tour_comparison, plot_edge_weight_heatmap
from visualization_algorithms import plot_qaoa_comprehensive_progress
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


# Brute-Force search graph
graphs_dict, runtime_data, labelled_tt_data, all_travel_times = tsp_brute_force(graph, graphs_dict, runtime_data, labelled_tt_data)
labelled_tt_data['Worst-Case'] = max(all_travel_times)
print('BFS completed')


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

# QAOA approximation
# TODO: figure out how to get off the barren plateau.
graphs_dict, runtime_data, labelled_tt_data, qaoa_progress = QAOA_approx(graph, graphs_dict, runtime_data, 
                                                                         labelled_tt_data, shots=1000, 
                                                                         warm_start='nearest_neighbor',
                                                                         label='QAOA-WS-NN')
print('QAOA completed')


# visualizations and data saving
# make bar chart for runtimes
fig, ax = plot_runtime_comparison(
    runtime_data,
    title="TSP Algorithm Runtime Comparison: 10 nodes",
    ylabel="Runtime (seconds, log scale)"
)
# plt.savefig(f'{curr_prob}/sols_runtime.png')


# make violin plot for trip times
fig, ax = plot_travel_times_violin(
    all_travel_times, 
    labeled_points=labelled_tt_data,
    title="TSP Algorithm Performance Comparison",
    ylabel="Total Travel Time (seconds)"
)
# plt.savefig(f'{curr_prob}/sols_distrib.png')
    
    
# Plot graphs comparison
fig, axes = plot_tour_comparison(graphs_dict, layout='circular')
# plt.savefig(f'{curr_prob}/sols_found.png')


# plot maps comparison
tours_dict = graphs_to_tours(graphs_dict)

# plot just brute-force
fig, ax = plot_route_on_map(tours_dict['Brute-Force'], address_set, dirs_mat, 
                            title='Brute-Force', 
                            use_map_background=True, 
                            map_style='sattelite')

# plt.savefig(f'{curr_prob}/BFS_map.png')

# plot all routes
fig, axes = plot_multiple_routes_comparison(tours_dict, address_set, dirs_mat,
                                            use_map_background=True, 
                                            map_style='sattelite')

# plt.savefig(f'{curr_prob}/all_maps.png')


fig, ax = plot_edge_weight_heatmap(graph, title="TSP Edge Weights")

# plt.savefig(f'{curr_prob}/edge_weights.png')


fig, ax = plot_qaoa_comprehensive_progress(qaoa_progress)

# plt.savefig(f'{curr_prob}/qaoa_progress.png')

plt.show()

'''
# save data
with open(f'{curr_prob}/tours.json', 'w') as f:
    json.dump(tours_dict, f)
    
    
with open(f'{curr_prob}/runtimes.json', 'w') as f:
    json.dump(runtime_data, f)
    
    
with open(f'{curr_prob}/travel-times.json', 'w') as f:
    json.dump(labelled_tt_data, f)
    

with open(f'{curr_prob}/all-travel-times.json', 'w') as f:
    f.writelines([f'{i}, ' for i in all_travel_times])
    
'''













