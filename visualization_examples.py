# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 11:18:37 2025

@author: joeco
"""

# this file contains examples for the visualization algorithms.
from google_maps import get_address_set, get_directions_matrix
from visualization_algorithms import plot_multiple_routes_comparison, plot_route_on_map
from visualization_algorithms import plot_runtime_comparison, plot_tour_comparison
from visualization_algorithms import plot_travel_times_violin, plot_edge_weight_heatmap
from visualization_algorithms import plot_travel_time_matrix_from_array, plot_benchmark_results
from visualization_algorithms import plot_qaoa_validity_pie, plot_qaoa_validity_progress
from visualization_algorithms import plot_valid_solution_hamming_distances, plot_hamming_distance_histogram
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


# Example usage
if __name__ == "__main__":
    # Generate sample travel times (e.g., from random sampling or different runs)
    np.random.seed(42)
    travel_times = np.random.normal(loc=130, scale=15, size=1000)
    
    # Define specific algorithms to label
    labeled_algorithms = {
        'Brute Force': 125.5,
        'Greedy': 142.3,
        'Asadpour': 128.7,
        'Christofides': 131.2,
        'Nearest Neighbor': 138.9
    }
    
    # Create the plot
    fig, ax = plot_travel_times_violin(
        travel_times, 
        labeled_points=labeled_algorithms,
        title="TSP Algorithm Performance Comparison: Example",
        ylabel="Total Travel Distance"
    )
    
    
    # Sample runtime data for different TSP algorithms
    runtime_data = {
        'Brute Force': 0.45,
        'Greedy': 0.002,
        'Nearest Neighbor': 0.003,
        'Christofides': 0.15,
        'Asadpour': 125.8,
        'Genetic Algorithm': 2.3
    }
    
    # Create the plot
    fig0, ax0 = plot_runtime_comparison(
        runtime_data,
        title="TSP Algorithm Runtime Comparison: Example",
        ylabel="Runtime (seconds, log scale)"
    )
    
    
    # Create sample tour graphs
    
    # Base graph (complete graph for reference)
    base_G = nx.DiGraph()
    nodes = list(range(5))
    for i in nodes:
        for j in nodes:
            if i != j:
                base_G.add_edge(i, j, weight=np.random.randint(1, 20))
    
    # Brute force solution
    brute_force_G = nx.DiGraph()
    bf_tour = [0, 1, 2, 3, 4, 0]
    for i in range(len(bf_tour) - 1):
        brute_force_G.add_edge(bf_tour[i], bf_tour[i+1], weight=10)
    
    # Alternative solution 1 (slightly different)
    greedy_G = nx.DiGraph()
    greedy_tour = [0, 2, 1, 3, 4, 0]
    for i in range(len(greedy_tour) - 1):
        greedy_G.add_edge(greedy_tour[i], greedy_tour[i+1], weight=10)
    
    # Alternative solution 2 (more different)
    heuristic_G = nx.DiGraph()
    heuristic_tour = [0, 3, 1, 4, 2, 0]
    for i in range(len(heuristic_tour) - 1):
        heuristic_G.add_edge(heuristic_tour[i], heuristic_tour[i+1], weight=10)
    
    # Create dictionary of tours
    tour_dict = {
        'base_graph': base_G,
        'Brute-Force': brute_force_G,
        'Greedy': greedy_G,
        'Heuristic': heuristic_G
    }
    
    # Plot comparison
    fig1, axes1 = plot_tour_comparison(tour_dict, layout='circular')
    
    
    # Single route
    tour = [0, 1, 2, 3, 4, 0]
    directions_matrix = get_directions_matrix('dir-mat_4m_10_1.json')
    addresses = get_address_set('address-set_4m_10_1.txt')
    
    fig2, ax2 = plot_route_on_map(tour, addresses, directions_matrix, 
                             title="Brute Force Solution", 
                             use_map_background=True, 
                             map_style='sattelite')

    # Multiple routes comparison
    tours = {
        'Brute Force': [0, 1, 2, 3, 4, 0],
        'Greedy': [0, 2, 1, 4, 3, 0]
        }
    
    fig3, axes3 = plot_multiple_routes_comparison(tours, addresses, directions_matrix, 
                                                use_map_background=True, 
                                                map_style='sattelite')
    
    
    # graph 2d heatmap of edge weights (travel times)
    # Example 1: From NetworkX graph
    G = nx.DiGraph()
    edges = [
        (0, 1, 120), (0, 2, 180), (0, 3, 240),
        (1, 0, 120), (1, 2, 300), (1, 3, 200),
        (2, 0, 180), (2, 1, 300), (2, 3, 150),
        (3, 0, 240), (3, 1, 200), (3, 2, 150)
    ]
    
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)
    
    fig4, ax4 = plot_edge_weight_heatmap(G, title="TSP Edge Weights (NetworkX Graph)",
                                         cmap='YlOrRd')
    
    # Example 2: From travel time matrix array
    travel_times = [
        [0, 120, 180, 240],
        [120, 0, 300, 200],
        [180, 300, 0, 150],
        [240, 200, 150, 0]
    ]
    
    fig5, ax5 = plot_travel_time_matrix_from_array(travel_times,
                                                   node_labels=['A', 'B', 'C', 'D'],
                                                   title="Travel Time Matrix",
                                                   cmap='plasma')
    # benchmarking
    # Example 1: Benchmark results
    print("Example 1: Benchmark Results Visualization")
    benchmark_data = {
        5: {'time': 12.5, 'memory_peak_mb': 2500, 'success': True},
        8: {'time': 8.3, 'memory_peak_mb': 5100, 'success': True},
        10: {'time': 6.2, 'memory_peak_mb': 8000, 'success': True},
        12: {'time': 7.8, 'memory_peak_mb': 11500, 'success': True},
        15: {'time': 9.5, 'memory_peak_mb': 18000, 'success': True},
        20: {'time': 15.2, 'memory_peak_mb': 32000, 'success': True},
        25: {'time': 0, 'memory_peak_mb': 0, 'success': False},
    }
    
    fig6, axes6 = plot_benchmark_results(benchmark_data)
    
    # Example 2: QAOA progress with cost
    print("\nExample 2: QAOA Progress Visualization")
    qaoa_progress = [
        {'iteration': 0, 'valid_shots': 50, 'invalid_shots': 974, 'best_cost': 250},
        {'iteration': 1, 'valid_shots': 120, 'invalid_shots': 904, 'best_cost': 230},
        {'iteration': 2, 'valid_shots': 200, 'invalid_shots': 824, 'best_cost': 220},
        {'iteration': 3, 'valid_shots': 280, 'invalid_shots': 744, 'best_cost': 215},
        {'iteration': 4, 'valid_shots': 350, 'invalid_shots': 674, 'best_cost': 205},
        {'iteration': 5, 'valid_shots': 420, 'invalid_shots': 604, 'best_cost': 200},
        {'iteration': 6, 'valid_shots': 480, 'invalid_shots': 544, 'best_cost': 195},
        {'iteration': 7, 'valid_shots': 520, 'invalid_shots': 504, 'best_cost': 190},
    ]
    
    fig7, axes7 = plot_qaoa_validity_progress(qaoa_progress)
    
    # Example 3: Pie chart
    print("\nExample 3: Validity Pie Chart")
    fig8, ax8 = plot_qaoa_validity_pie(520, 504, title="Final QAOA Results")
    
    plt.show()
    
    
    '''
    # example 4: QAOA validity tours
    # Get valid tours from QAOA results
    valid_tours = [stats['best_tour'] for stats in qaoa_stats_list if stats['best_tour']]

    # Visualize distances between all solutions
    fig, ax = plot_valid_solution_hamming_distances(valid_tours, qubit_to_edge_map, G)

    # Or compare to a reference (e.g., greedy)
    greedy_tour = get_warm_start_tour(G, method='greedy')
    fig, ax = plot_hamming_distance_histogram(valid_tours, qubit_to_edge_map, 
                                              reference_tour=greedy_tour)
    '''