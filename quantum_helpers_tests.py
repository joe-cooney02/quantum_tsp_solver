# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 16:20:53 2025

@author: joeco

contains examples and/or unit tests for the quantum helper functions
"""
from quantum_helpers import is_valid_tsp_tour, postselect_best_tour, create_qubit_to_edge_map
from quantum_helpers import create_edge_to_qubit_map, create_tsp_qaoa_circuit, save_qaoa_circuit
from quantum_helpers import load_qaoa_circuit, get_initial_parameters, bind_qaoa_parameters
from quantum_helpers import split_circuit_for_simulation, simulate_large_circuit_in_batches
from quantum_benchmarking import benchmark_batch_sizes
import networkx as nx
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Create a simple TSP graph
    G = nx.DiGraph()
    edges_with_weights = [
        (0, 1, 10), (0, 2, 15), (0, 3, 20),
        (1, 0, 10), (1, 2, 35), (1, 3, 25),
        (2, 0, 15), (2, 1, 35), (2, 3, 30),
        (3, 0, 20), (3, 1, 25), (3, 2, 30)
        ]
    
    for u, v, w in edges_with_weights:
        G.add_edge(u, v, weight=w)
    
    
    # Create qubit to edge mapping automatically
    print("Creating qubit-to-edge mapping...")
    qubit_to_edge_map = create_qubit_to_edge_map(G)
    print(f"Number of qubits needed: {len(qubit_to_edge_map)}")
    print(f"Mapping: {qubit_to_edge_map}")
    
    # Also create reverse mapping
    edge_to_qubit_map = create_edge_to_qubit_map(G)
    print(f"\nReverse mapping (first 5): {dict(list(edge_to_qubit_map.items())[:5])}")
    
    
    # Test valid tour: 0->1->2->3->0
    valid_bitstring = "111100000000"  # qubits 0,1,2,3 are on
    print(f"Testing bitstring: {valid_bitstring}")
    is_valid, tour = is_valid_tsp_tour(valid_bitstring, qubit_to_edge_map, G, return_tour=True)
    print(f"Valid: {is_valid}, Tour: {tour}")
    
    # Test invalid tour (not enough edges)
    invalid_bitstring = "110000000000"
    print(f"\nTesting bitstring: {invalid_bitstring}")
    is_valid = is_valid_tsp_tour(invalid_bitstring, qubit_to_edge_map, G)
    print(f"Valid: {is_valid}")
    
    # Test postselection with multiple shots
    bitstrings = ["111100000000", "110000000000", "100010001100", "101010101010"]
    counts = [5, 3, 7, 2]
    
    print("\n--- Postselection ---")
    best_bs, best_tour, cost, success = postselect_best_tour(bitstrings, counts, 
                                                              qubit_to_edge_map, G)
    print(f"Best bitstring: {best_bs}")
    print(f"Best tour: {best_tour}")
    print(f"Tour cost: {cost}")
    print(f"Success rate: {success:.2%}")
    
    
    # Create QAOA circuit with 2 layers
    print("Creating QAOA circuit...")
    qaoa_circuit = create_tsp_qaoa_circuit(G, qubit_to_edge_map, num_layers=2)
    
    print(f"Circuit has {qaoa_circuit.num_qubits} qubits")
    print(f"Circuit has {len(qaoa_circuit.parameters)} parameters")
    print(f"Parameters: {[p.name for p in qaoa_circuit.parameters]}")
    
    # Save the circuit
    print("\nSaving circuit...")
    save_qaoa_circuit(qaoa_circuit, "tsp_qaoa.qpy")
    
    # Load the circuit back
    print("\nLoading circuit...")
    loaded_circuit = load_qaoa_circuit("tsp_qaoa.qpy")
    print(f"Loaded circuit has {loaded_circuit.num_qubits} qubits")
    
    # Get initial parameters and bind them
    print("\nBinding parameters...")
    gamma_vals, beta_vals = get_initial_parameters(2, strategy='random')
    print(f"Gamma values: {gamma_vals}")
    print(f"Beta values: {beta_vals}")
    
    bound_circuit = bind_qaoa_parameters(qaoa_circuit, gamma_vals, beta_vals)
    print(f"Bound circuit has {len(bound_circuit.parameters)} free parameters (should be 0)")
    
    # Display circuit structure
    print("\n--- Circuit Diagram (first few gates) ---")
    
    fig0 = qaoa_circuit.draw(output='mpl', fold=-1)
    plt.figure(fig0)
    
    
    # Test circuit splitting for large simulations
    print("\n--- Testing Circuit Splitting ---")
    print(f"Original circuit has {bound_circuit.num_qubits} qubits")
   
    sub_circuits = split_circuit_for_simulation(bound_circuit, max_qubits_per_batch=5)
    print(f"Split into {len(sub_circuits)} sub-circuits:")
    for idx, (sub_qc, indices) in enumerate(sub_circuits):
        print(f"  Sub-circuit {idx+1}: {len(indices)} qubits (indices {indices})")
   
    # Simulate the split circuits
    print("\nSimulating split circuits...")
    results = simulate_large_circuit_in_batches(bound_circuit, max_qubits_per_batch=5, shots=65536)
    print(f"Top 5 results: {dict(list(sorted(results.items(), key=lambda x: x[1], reverse=True))[:5])}")

    
    # Benchmark different batch sizes
    print("\n--- Benchmarking Batch Sizes ---")
    benchmark_results = benchmark_batch_sizes(
    bound_circuit, 
    batch_sizes=[4, 6, 8],  # Small values for quick demo
    shots=50,  # Fewer shots for speed
    verbose=True
    )
    
    # Display circuit structure
    print("\n--- Circuit Diagram (first few gates) ---")
    
    print("\n--- Postselection for split circuit ---")
    bitstrings = list(results.keys())
    counts = list(results.values())
    
    best_bs, best_tour, cost, success = postselect_best_tour(bitstrings, counts, 
                                                              qubit_to_edge_map, G)
    print(f"Best bitstring: {best_bs}")
    print(f"Best tour: {best_tour}")
    print(f"Tour cost: {cost}")
    print(f"Success rate: {success:.2%}")
    
    
    plt.show()
    
    
    