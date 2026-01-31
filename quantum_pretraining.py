# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 11:23:39 2026

@author: joeco

Contains functions for pretraining QAOA to learn how to explore spaces.

The idea is to pre-train the first layer(s) of QAOA to maximize the probability
of measuring valid TSP solutions, independent of cost. Then, the main QAOA 
optimization can focus on minimizing cost within the valid solution space.

This is particularly useful when dealing with batched simulations where 
2-qubit gates are limited, as it helps the algorithm learn valid constraints
through single-qubit operations.
"""

import numpy as np
from scipy.optimize import minimize
from quantum_helpers import (
    create_tsp_qaoa_circuit, 
    bind_qaoa_parameters,
    simulate_large_circuit_in_batches,
    count_valid_invalid,
    create_qubit_to_edge_map
)


def pretrain_validity_layers(graph, qubit_to_edge_map, num_layers=1,
                            shots=1024, batch_size=8, sim_method='statevector',
                            max_iterations=50, verbose=True, use_local_2q_gates=False):
    """
    Pre-train QAOA layers together to maximize the probability of valid solutions.
    
    This function optimizes all gamma and beta parameters simultaneously to maximize
    the fraction of measurement outcomes that are valid TSP tours, regardless of cost.
    All layers are trained together, allowing them to coordinate from the start.
    
    Parameters:
    -----------
    graph : networkx.DiGraph
        The TSP graph
    qubit_to_edge_map : dict
        Mapping from qubit index to edge tuple
    num_layers : int, optional
        Number of QAOA layers to pre-train together. Default is 1.
    shots : int, optional
        Number of measurement shots for evaluation
    batch_size : int, optional
        Qubit batch size for simulation
    sim_method : str, optional
        Simulation method ('statevector' or 'density_matrix')
    max_iterations : int, optional
        Maximum optimization iterations
    verbose : bool, optional
        Whether to print progress
    use_local_2q_gates : bool, optional
        If True, add local 2-qubit entangling gates within batches
    
    Returns:
    --------
    tuple: (optimal_gammas, optimal_betas, final_validity_rate)
        - optimal_gammas: list of optimal gamma values (length num_layers)
        - optimal_betas: list of optimal beta values (length num_layers)
        - final_validity_rate: best validity rate achieved
    """
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Pre-training {num_layers} layer(s) together for validity")
        if use_local_2q_gates:
            print(f"  Using local 2-qubit gates (batch_size={batch_size})")
        print(f"{'='*60}")
    
    # Create circuit with ALL layers
    circuit = create_tsp_qaoa_circuit(graph, qubit_to_edge_map, num_layers=num_layers,
                                     use_local_2q_gates=use_local_2q_gates, batch_size=batch_size)
    
    # Track optimization progress
    iteration_count = [0]
    best_validity = [0.0]
    
    def validity_objective(params):
        """
        Objective function: negative validity rate (we minimize, so negate to maximize).
        
        params contains [gamma_0, ..., gamma_n, beta_0, ..., beta_n]
        """
        iteration_count[0] += 1
        
        # Split parameters
        gamma_values = params[:num_layers]
        beta_values = params[num_layers:]
        
        # Bind and simulate
        bound_circuit = bind_qaoa_parameters(circuit, gamma_values, beta_values)
        counts = simulate_large_circuit_in_batches(bound_circuit, batch_size, shots, sim_method)
        
        # Calculate validity rate
        valid_shots, invalid_shots = count_valid_invalid(counts, qubit_to_edge_map, graph)
        total_shots = valid_shots + invalid_shots
        validity_rate = valid_shots / total_shots if total_shots > 0 else 0
        
        # Track best
        if validity_rate > best_validity[0]:
            best_validity[0] = validity_rate
        
        if verbose and iteration_count[0] % 5 == 0:
            print(f"  Iteration {iteration_count[0]:3d}: Validity = {validity_rate:.2%} "
                  f"(Best: {best_validity[0]:.2%})")
        
        # Return negative because we're minimizing
        return -validity_rate
    
    # Initialize all parameters together
    initial_params = np.random.uniform(-0.1, 0.1, 2 * num_layers)
    
    if verbose:
        print(f"Optimizing {2 * num_layers} parameters together...")
    
    # Optimize
    result = minimize(
        validity_objective,
        x0=initial_params,
        method='COBYLA',
        options={'maxiter': max_iterations}
    )
    
    optimal_gammas = result.x[:num_layers]
    optimal_betas = result.x[num_layers:]
    
    if verbose:
        print(f"\n{'='*60}")
        print("Pre-training completed!")
        print(f"  Final validity rate: {best_validity[0]:.2%}")
        print(f"  Optimal gammas: {optimal_gammas}")
        print(f"  Optimal betas: {optimal_betas}")
        print(f"{'='*60}\n")
    
    return optimal_gammas.tolist(), optimal_betas.tolist(), best_validity[0]


def create_pretrained_initial_params(pretrained_gammas, pretrained_betas, 
                                     total_layers, strategy='extend_with_zeros'):
    """
    Create full initial parameter set for QAOA using pre-trained layers.
    
    Parameters:
    -----------
    pretrained_gammas : list
        Pre-trained gamma values (length = num_pretrained_layers)
    pretrained_betas : list
        Pre-trained beta values (length = num_pretrained_layers)
    total_layers : int
        Total number of QAOA layers needed
    strategy : str, optional
        How to initialize remaining layers:
        - 'extend_with_zeros': Initialize remaining layers with zeros
        - 'extend_with_small_random': Initialize with small random values
        - 'extend_with_linear': Linear interpolation from last pretrained value
    
    Returns:
    --------
    list: Full parameter list [gamma_0, ..., gamma_n, beta_0, ..., beta_n]
    """
    num_pretrained = len(pretrained_gammas)
    
    if num_pretrained > total_layers:
        raise ValueError(f"Cannot have more pre-trained layers ({num_pretrained}) "
                        f"than total layers ({total_layers})")
    
    # Start with pre-trained values
    all_gammas = list(pretrained_gammas)
    all_betas = list(pretrained_betas)
    
    # Add remaining layers
    remaining_layers = total_layers - num_pretrained
    
    if remaining_layers > 0:
        if strategy == 'extend_with_zeros':
            all_gammas.extend([0.0] * remaining_layers)
            all_betas.extend([0.0] * remaining_layers)
            
        elif strategy == 'extend_with_small_random':
            all_gammas.extend(np.random.uniform(-0.1, 0.1, remaining_layers))
            all_betas.extend(np.random.uniform(-0.1, 0.1, remaining_layers))
            
        elif strategy == 'extend_with_linear':
            # Linearly interpolate from last pretrained value to a reasonable target
            last_gamma = pretrained_gammas[-1]
            last_beta = pretrained_betas[-1]
            
            gamma_extension = np.linspace(last_gamma, np.pi/4, remaining_layers + 1)[1:]
            beta_extension = np.linspace(last_beta, np.pi/2, remaining_layers + 1)[1:]
            
            all_gammas.extend(gamma_extension)
            all_betas.extend(beta_extension)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    return all_gammas + all_betas


def pretrain_and_create_initial_params(graph, num_pretrain_layers=1, total_layers=3,
                                       shots=1024, batch_size=8, 
                                       max_iterations=50, verbose=True, use_local_2q_gates=False):
    """
    Convenience function: pre-train layers and create full initial parameters.
    
    This combines the pre-training process with parameter initialization,
    ready to be used directly in QAOA_approx.
    
    Parameters:
    -----------
    graph : networkx.DiGraph
        The TSP graph
    num_pretrain_layers : int, optional
        Number of layers to pre-train for validity (trained together)
    total_layers : int, optional
        Total number of QAOA layers to use
    shots : int, optional
        Number of measurement shots for pre-training
    batch_size : int, optional
        Qubit batch size for simulation
    max_iterations : int, optional
        Maximum iterations for pre-training
    verbose : bool, optional
        Whether to print progress
    use_local_2q_gates : bool, optional
        If True, add local 2-qubit entangling gates within batches during pretraining
    
    Returns:
    --------
    tuple: (initial_params, pretrain_validity_rate)
        - initial_params: List ready to use as custom_initial_params in QAOA optimization
          Format: [gamma_0, ..., gamma_n, beta_0, ..., beta_n]
        - pretrain_validity_rate: Validity rate achieved during pre-training
    """
    
    qubit_to_edge_map = create_qubit_to_edge_map(graph)
    
    # Pre-train specified number of layers
    pretrained_gammas, pretrained_betas, validity_rate = pretrain_validity_layers(
        graph, qubit_to_edge_map,
        num_layers=num_pretrain_layers,
        shots=shots,
        batch_size=batch_size,
        max_iterations=max_iterations,
        verbose=verbose,
        use_local_2q_gates=use_local_2q_gates
    )
    
    # Create full parameter set for all layers
    initial_params = create_pretrained_initial_params(
        pretrained_gammas, pretrained_betas,
        total_layers=total_layers,
        strategy='extend_with_zeros'
    )
    
    return initial_params, [validity_rate]


# Example usage:
"""
# Pre-train the first layer only
pretrained_params, validity_rates = pretrain_and_create_initial_params(
    graph, 
    num_pretrain_layers=1,  # Just first layer
    total_layers=3,         # Will have 3 layers total
    shots=2048,
    max_iterations=30
)

# Then use in QAOA
graphs_dict, runtime_data, labelled_tt_data, qaoa_progress = QAOA_approx(
    graph, graphs_dict, runtime_data, labelled_tt_data, qaoa_progress,
    layers=3,
    shots=10000,
    label='QAOA-Pretrained-Layer0',
    custom_initial_params=pretrained_params  # Would need to add this parameter
)
"""
