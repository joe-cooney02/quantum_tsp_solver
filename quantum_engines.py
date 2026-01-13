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
import time
from quantum_helpers import create_qubit_to_edge_map
from quantum_helpers import bind_qaoa_parameters, create_tsp_qaoa_circuit
from quantum_helpers import get_initial_parameters, get_cost_expectation
from quantum_helpers import simulate_large_circuit_in_batches, get_qaoa_statistics
from quantum_helpers import create_warm_started_qaoa
from opt_helpers import tour_to_graph, get_trip_time


def QAOA_approx(graph, graphs_dict, runtime_data, tt_data, verbose=True, layers=None, shots=None, 
                qubit_batch_size=None, inv_penalty_m=1, sim_method='statevector', label='QAOA', warm_start=None, 
                exploration_strength=None):
    '''
    Parameters
    ----------
    graph : networkx DiGraph
        Graph to do TSP on.
    graphs_dict : {}
        a dictionary of graphs for the output.
    runtime_data : {}
        a dict of runtime data for output.
    tt_data : {}
        a dict of travel time data for output.
    warm_start: String or None
        wether to warm-start the QAOA, and if so, how.
    exploration_strength: None or float
        If the warm-start is being used, how strongly to explore (default 0.2)

    Returns
    -------
    graphs_dict : {}
        a dictionary of grpahs for the output.
    runtime_data : {}
        a dict of runtime data for output.
    tt_data : {}
        a dict of travel time data for output.
    qaoa_results_over_time : [{}]
        a list of dicts of simulation data at each iteration.
    '''
    
    
    # do some further testing to find a good number for this which is not overwhelming.
    # the max is 1 million by default but can change: backend._configuration.max_shots = ...
    if shots is None:
        shots = 2 ** graph.number_of_nodes()
    
    # qubit batch size of 8 works good for my PC
    if qubit_batch_size is None:
        qubit_batch_size = 8
        
    # paper found that inv_penalty (their Lambda) is best from 1.0 to 4.5 * max edge weight.
    inv_penalty = max(d['weight'] for u, v, d in graph.edges(data=True)) * inv_penalty_m
    
    # do some testing to see whats a good number of layers to use.
    if layers is None:
        layers = 3
    
    
    # create and initialize circuit.
    qubit_to_edge_map = create_qubit_to_edge_map(graph)
    gamma_values, beta_values = get_initial_parameters(layers)
    
    if warm_start == None:
        circuit = create_tsp_qaoa_circuit(graph, qubit_to_edge_map, num_layers=layers)
    else:
        circuit = create_warm_started_qaoa(graph, qubit_to_edge_map, num_layers=layers, warm_start_method=warm_start, 
                                           exploration_strength=exploration_strength)
    
    init_params = [i for i in gamma_values] + [i for i in beta_values]
    qaoa_results_over_time = []
    
    
    # do QAOA.
    start_time = time.time()
    
    qaoa_result = minimize(run_QAOA,
                          x0=init_params,
                          args=(circuit, qubit_batch_size, shots, sim_method,
                                layers, graph, qubit_to_edge_map, qaoa_results_over_time, 
                                inv_penalty),
                          method='COBYLA')
    
    end_time = time.time()
    
    tot_time = end_time - start_time
    
    
    # optional output
    if verbose:
        print(f'optimization success: {qaoa_result.success}')
        print(f'optimal parameters: {qaoa_result.x}')
        print(f'message: {qaoa_result.message}')
        print(f'objective function value (expectation value / travel time): {qaoa_result.fun}')
    
    
    # extract optimal solution
    final_result = qaoa_results_over_time[-1]
    
    print(f'final_result: {final_result}')
    
    best_tour = final_result['best_tour']
    
    print(f'best tour: {best_tour}')
    
    TSP_graph = tour_to_graph(graph, best_tour)
    
    graphs_dict[label] = TSP_graph
    runtime_data[label] = tot_time
    tt_data[label] = get_trip_time(TSP_graph)
    
    
    # yield output
    return graphs_dict, runtime_data, tt_data, qaoa_results_over_time


def run_QAOA(parameters, circuit, batch_size, shots, sim_method, layers, graph, qubit_to_edge_map, results_over_time, inv_penalty=0):
    
    gamma_values = parameters[0:layers]
    beta_values = parameters[layers:]

    
    bound_circuit = bind_qaoa_parameters(circuit, gamma_values, beta_values)
    counts = simulate_large_circuit_in_batches(bound_circuit, batch_size, shots, sim_method)
    
    bitstrings = list(counts.keys())
    expectation_val = get_cost_expectation(bitstrings, counts, qubit_to_edge_map, graph, inv_penalty=0)
    
    stats = get_qaoa_statistics(counts, qubit_to_edge_map, graph, len(results_over_time))

    results_over_time.append(stats)
    
    return expectation_val










