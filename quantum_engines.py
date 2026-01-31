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


def QAOA_approx(graph, graphs_dict, runtime_data, tt_data, qaoa_progress, verbose=True, layers=None, shots=None, 
                qubit_batch_size=None, inv_penalty_m=1, sim_method='statevector', label='QAOA', warm_start=None, 
                exploration_strength=0, initialization_strategy='zero', custom_initial_params=None, 
                lock_pretrained_layers=0, use_local_2q_gates=False, use_soft_validity=False, 
                soft_validity_penalty_base=10.0):
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
    custom_initial_params : list or None
        Custom initial parameters [γ₀...γₙ, β₀...βₙ]. If provided with lock_pretrained_layers,
        the first lock_pretrained_layers will be fixed during optimization.
    lock_pretrained_layers : int
        Number of initial layers to lock (keep fixed) during optimization.
        Only applies if custom_initial_params is provided. Default 0 (no locking).
        Example: lock_pretrained_layers=1 fixes γ₀ and β₀, optimizes γ₁, γ₂, β₁, β₂
    use_local_2q_gates : bool
        If True, add local 2-qubit entangling gates (CZ) within batches.
        This enables entanglement while maintaining batched simulation compatibility.
        Gates only connect qubits within the same batch (default: False)
    use_soft_validity : bool
        If True, use soft validity penalties that create gradients toward valid solutions.
        If False, use hard penalties (flat landscape). Default: False.
    soft_validity_penalty_base : float
        Base multiplier for soft validity penalties. Higher values push harder toward validity.
        Only used if use_soft_validity=True. Typical range: 5.0-20.0. Default: 10.0.

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
    
    # Use custom initial parameters if provided, otherwise generate them
    if custom_initial_params is not None:
        # Custom params should be [gamma_0, ..., gamma_n, beta_0, ..., beta_n]
        if len(custom_initial_params) != 2 * layers:
            raise ValueError(f"custom_initial_params must have length {2*layers} "
                           f"(got {len(custom_initial_params)})")
        init_params = custom_initial_params
        gamma_values = init_params[:layers]
        beta_values = init_params[layers:]
    else:
        gamma_values, beta_values = get_initial_parameters(layers, strategy=initialization_strategy)
        init_params = [i for i in gamma_values] + [i for i in beta_values]
    
    # Handle parameter locking
    if lock_pretrained_layers > 0:
        if custom_initial_params is None:
            raise ValueError("lock_pretrained_layers requires custom_initial_params to be provided")
        if lock_pretrained_layers >= layers:
            raise ValueError(f"lock_pretrained_layers ({lock_pretrained_layers}) must be less than total layers ({layers})")
        
        # Extract locked and optimizable parameters
        locked_gammas = init_params[:lock_pretrained_layers]
        locked_betas = init_params[layers:layers + lock_pretrained_layers]
        
        optimizable_gammas = init_params[lock_pretrained_layers:layers]
        optimizable_betas = init_params[layers + lock_pretrained_layers:]
        
        locked_params = locked_gammas + locked_betas
        optimizable_init_params = optimizable_gammas + optimizable_betas
        
        print("\nParameter Locking Enabled:")
        print(f"  Locked layers: 0-{lock_pretrained_layers-1} ({len(locked_params)} params)")
        print(f"  Optimizable layers: {lock_pretrained_layers}-{layers-1} ({len(optimizable_init_params)} params)")
        print(f"  Locked gammas: {locked_gammas}")
        print(f"  Locked betas: {locked_betas}")
    else:
        locked_params = None
        optimizable_init_params = init_params
    
    if warm_start == None:
        circuit = create_tsp_qaoa_circuit(graph, qubit_to_edge_map, num_layers=layers,
                                         use_local_2q_gates=use_local_2q_gates, batch_size=qubit_batch_size)
    else:
        circuit = create_warm_started_qaoa(graph, qubit_to_edge_map, num_layers=layers, warm_start_method=warm_start, 
                                           exploration_strength=exploration_strength,
                                           use_local_2q_gates=use_local_2q_gates, batch_size=qubit_batch_size)
    
    qaoa_results_over_time = []
    
    
    # do QAOA.
    start_time = time.time()
    
    qaoa_result = minimize(run_QAOA,
                          x0=optimizable_init_params,
                          args=(circuit, qubit_batch_size, shots, sim_method,
                                layers, graph, qubit_to_edge_map, qaoa_results_over_time, 
                                inv_penalty, locked_params, lock_pretrained_layers,
                                use_soft_validity, soft_validity_penalty_base),
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
    qaoa_progress[label] = qaoa_results_over_time
    
    
    # yield output
    return graphs_dict, runtime_data, tt_data, qaoa_progress


def run_QAOA(optimizable_parameters, circuit, batch_size, shots, sim_method, layers, graph, 
             qubit_to_edge_map, results_over_time, inv_penalty=0, locked_params=None, 
             lock_pretrained_layers=0, use_soft_validity=False, soft_validity_penalty_base=10.0):
    """
    Objective function for QAOA optimization.
    
    Parameters:
    -----------
    optimizable_parameters : array
        Parameters to optimize (excludes locked parameters)
    locked_params : list or None
        Fixed parameters from pretrained layers [locked_gammas, locked_betas]
    lock_pretrained_layers : int
        Number of layers that are locked
    use_soft_validity : bool
        Whether to use soft validity penalties
    soft_validity_penalty_base : float
        Base multiplier for soft validity penalties
    """
    
    # Reconstruct full parameter array
    if locked_params is not None and lock_pretrained_layers > 0:
        # locked_params = [gamma_0, ..., gamma_{L-1}, beta_0, ..., beta_{L-1}]
        # optimizable_parameters = [gamma_L, ..., gamma_n, beta_L, ..., beta_n]
        
        num_locked_gammas = lock_pretrained_layers
        num_optimizable_gammas = layers - lock_pretrained_layers
        
        locked_gammas = locked_params[:num_locked_gammas]
        locked_betas = locked_params[num_locked_gammas:]
        
        optimizable_gammas = optimizable_parameters[:num_optimizable_gammas]
        optimizable_betas = optimizable_parameters[num_optimizable_gammas:]
        
        # Combine: [locked_gammas, optimizable_gammas, locked_betas, optimizable_betas]
        gamma_values = list(locked_gammas) + list(optimizable_gammas)
        beta_values = list(locked_betas) + list(optimizable_betas)
    else:
        # No locking, use parameters as-is
        gamma_values = optimizable_parameters[0:layers]
        beta_values = optimizable_parameters[layers:]

    
    bound_circuit = bind_qaoa_parameters(circuit, gamma_values, beta_values)
    counts = simulate_large_circuit_in_batches(bound_circuit, batch_size, shots, sim_method)
    
    bitstrings = list(counts.keys())
    expectation_val = get_cost_expectation(bitstrings, counts, qubit_to_edge_map, graph, 
                                          inv_penalty=inv_penalty,
                                          use_soft_validity=use_soft_validity,
                                          soft_validity_penalty_base=soft_validity_penalty_base)
    
    stats = get_qaoa_statistics(counts, qubit_to_edge_map, graph, len(results_over_time))

    results_over_time.append(stats)
    
    return expectation_val










