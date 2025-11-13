# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 09:52:41 2025

@author: joeco
"""

'''
To do TSP via quantum computing, the QAOA approach is to use one qubit per edge. (O(N^2))
Then, the edges that are selected are the qubits in the 1 position.
Maybe with an RQAOA-like approach, we could encode 1 qubit per node, and select one at each step.
    This would be like the Next-Nearest heuristic solution, and speedup could only come during the 
    comparison between all edges connected to a node.
    
    Since classical computers are good at finding a minimum from a list (O(n)), 
    the only advantage would be if we could encode extra information that made better choices.
    That total classical heuristic takes O(n^2) time: n + (n-1) + ... 1 ~ n^2 / 2
    
'''

from scipy.optimize import minimize
from quantum_helpers import create_qubit_to_edge_map
from quantum_helpers import bind_qaoa_parameters, create_tsp_qaoa_circuit
from quantum_helpers import get_initial_parameters, get_cost_expectation
from quantum_helpers import simulate_large_circuit_in_batches, postselect_best_tour


def QAOA_approx(graph, qubit_batch_size, layers, shots, inv_penalty=0, sim_method='statevector'):
    
    qubit_to_edge_map = create_qubit_to_edge_map(graph)
    circuit = create_tsp_qaoa_circuit(graph, qubit_to_edge_map, num_layers=layers)
    gamma_values, beta_values = get_initial_parameters(layers)
    
    init_params = [i for i in gamma_values] + [i for i in beta_values]
    qaoa_results_over_time = []
    
    qaoa_result = minimize(run_QAOA,
                          x0=init_params,
                          args=(circuit, qubit_batch_size, shots, sim_method,
                                layers, graph, qubit_to_edge_map, qaoa_results_over_time, 
                                inv_penalty),
                          method='COBYLA')
    
    return qaoa_result


def run_QAOA(parameters, circuit, batch_size, shots, sim_method, layers, graph, qubit_to_edge_map, results_over_time, inv_penalty=0):
    
    gamma_values = parameters[0:layers]
    beta_values = parameters[layers:-1]
    
    bound_circuit = bind_qaoa_parameters(circuit, gamma_values, beta_values)
    counts = simulate_large_circuit_in_batches(bound_circuit, batch_size, shots, sim_method)
    
    bitstrings = list(counts.keys())
    expectation_val = get_cost_expectation(bitstrings, counts, qubit_to_edge_map, graph, inv_penalty=0)
    
    results_over_time.append(postselect_best_tour(bitstrings, counts, qubit_to_edge_map, graph))
    
    return expectation_val










