# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 13:21:27 2025

@author: joeco

This file will benchmark the PC it's running on for QAOA simulations.
"""

from quantum_benchmarking import benchmark_batch_sizes
from quantum_helpers import create_qubit_to_edge_map, create_tsp_qaoa_circuit, get_initial_parameters, bind_qaoa_parameters
import networkx as nx
from visualization_algorithms import plot_benchmark_results
import matplotlib.pyplot as plt
import json

if __name__ == '__main__':
    # make a graph with, say, 10 nodes all with weight 1

    G = nx.DiGraph()
    num_nodes = 10
    shots=2048
    
    # Add 10 nodes to the graph
    # Nodes will be labeled from 0 to 99
    G.add_nodes_from(range(num_nodes))

    # Add all possible directed edges with weight 1
    # This creates a fully-connected directed graph
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:  # Ensure no self-loops
                G.add_edge(i, j, weight=1)
    
    # Create qubit to edge mapping automatically
    print("Creating qubit-to-edge mapping...")
    qubit_to_edge_map = create_qubit_to_edge_map(G)
    print(f"Number of qubits needed: {len(qubit_to_edge_map)}")
    # print(f"Mapping: {qubit_to_edge_map}")
    
    # Create QAOA circuit with 2 layers
    print("Creating QAOA circuit...")
    qaoa_circuit = create_tsp_qaoa_circuit(G, qubit_to_edge_map, num_layers=2)
    
    # Get initial parameters and bind them
    print("\nBinding parameters...")
    gamma_vals, beta_vals = get_initial_parameters(2, strategy='random')
    print(f"Gamma values: {gamma_vals}")
    print(f"Beta values: {beta_vals}")
    
    bound_circuit = bind_qaoa_parameters(qaoa_circuit, gamma_vals, beta_vals)
    print(f"Bound circuit has {len(bound_circuit.parameters)} free parameters (should be 0)")
    
    
    # Benchmark different batch sizes
    print("\n--- Benchmarking Batch Sizes density matrix ---")
    benchmark_dm_results, optimal_dm_size = benchmark_batch_sizes(
    bound_circuit, 
    batch_sizes=[i for i in range(1, 26)],
    shots=shots,
    sim_method='density_matrix',
    verbose=False
    )
    
    # Benchmark different batch sizes
    print("\n--- Benchmarking Batch Sizes statevector ---")
    benchmark_sv_results, optimal_sv_size = benchmark_batch_sizes(
    bound_circuit, 
    batch_sizes=[i for i in range(1, 26)],
    shots=shots,
    sim_method='statevector',
    verbose=False
    )
    
    # results: on my PC, any batch size between 2-8 qubits will be fine.
        # batch size of 6 -> 4.4s DM, 4.55s statevector.
        # statevector is faster for larger batch sizes.
        
    fig0, axes0 = plot_benchmark_results(benchmark_dm_results)
    plt.savefig(f'benchmarking/{num_nodes}_nodes/density_matrix_{shots}s_results.png')
    
    fig1, axes1 = plot_benchmark_results(benchmark_sv_results)
    plt.savefig(f'benchmarking/{num_nodes}_nodes/statevector_{shots}s_results.png')

    plt.show()
    
    # save benchmarking results
    with open(f'benchmarking/{num_nodes}_nodes/density_matrix_{shots}s_results.json', 'w') as f:
        json.dump(benchmark_dm_results, f)
        
    with open(f'benchmarking/{num_nodes}_nodes/statevector_{shots}s_results.json', 'w') as f:
        json.dump(benchmark_sv_results, f)
