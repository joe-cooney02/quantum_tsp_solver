# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 11:18:37 2025

@author: joeco
"""

# this file contains examples for the visualization algorithms.
from google_maps import get_address_set, get_directions_matrix
from visualization_algorithms import plot_multiple_routes_comparison, plot_route_on_map, plot_runtime_comparison, plot_tour_comparison, plot_travel_times_violin
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
    fig, ax = plot_runtime_comparison(
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
    fig, axes = plot_tour_comparison(tour_dict, layout='circular')
    
    
    # Single route
    tour = [0, 1, 2, 3, 4, 0]
    directions_matrix = get_directions_matrix('dir-mat_4m_10_1.json')
    addresses = get_address_set('address-set_4m_10_1.txt')
    
    fig, ax = plot_route_on_map(tour, addresses, directions_matrix, 
                             title="Brute Force Solution", 
                             use_map_background=True, 
                             map_style='sattelite')

    # Multiple routes comparison
    tours = {
        'Brute Force': [0, 1, 2, 3, 4, 0],
        'Greedy': [0, 2, 1, 4, 3, 0]
        }
    
    fig, axes = plot_multiple_routes_comparison(tours, addresses, directions_matrix, 
                                                use_map_background=True, 
                                                map_style='sattelite')
    
    plt.show()