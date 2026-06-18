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
    create_qubit_to_edge_map,
    create_multi_order_qaoa_circuit,
    bind_multi_order_parameters,
    get_multi_order_initial_parameters,
    flatten_multi_order_params,
    unflatten_multi_order_params,
    compute_diversity_score,
    get_qaoa_statistics,
    estimate_gate_count,
)


def pretrain_validity_layers(graph, qubit_to_edge_map, num_layers=1,
                            shots=1024, batch_size=8, sim_method='statevector',
                            max_iterations=50, verbose=True, use_local_2q_gates=False, device='CPU'):
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
        beta_values = params[num_layers:2*num_layers]
        theta_values = params[2*num_layers:]
        
        # Bind and simulate
        bound_circuit = bind_qaoa_parameters(circuit, gamma_values, beta_values, theta_values)
        counts = simulate_large_circuit_in_batches(bound_circuit, batch_size, shots, sim_method, device=device)
        
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
    initial_params = np.random.uniform(-0.1, 0.1, 3 * num_layers)
    
    if verbose:
        print(f"Optimizing {3 * num_layers} parameters together...")
    
    # Optimize
    result = minimize(
        validity_objective,
        x0=initial_params,
        method='COBYLA',
        options={'maxiter': max_iterations}
    )
    
    optimal_gammas = result.x[:num_layers]
    optimal_betas = result.x[num_layers:2*num_layers]
    optimal_thetas = result.x[2*num_layers:]
    
    if verbose:
        print(f"\n{'='*60}")
        print("Pre-training completed!")
        print(f"  Final validity rate: {best_validity[0]:.2%}")
        print(f"  Optimal gammas: {optimal_gammas}")
        print(f"  Optimal betas: {optimal_betas}")
        print(f"{'='*60}\n")
    
    return optimal_gammas.tolist(), optimal_betas.tolist(), optimal_thetas.tolist(), best_validity[0]


def create_pretrained_initial_params(pretrained_gammas, pretrained_betas, pretrained_thetas, 
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
    all_thetas = list(pretrained_thetas)
    
    # Add remaining layers
    remaining_layers = total_layers - num_pretrained
    
    if remaining_layers > 0:
        if strategy == 'extend_with_zeros':
            all_gammas.extend([0.0] * remaining_layers)
            all_betas.extend([0.0] * remaining_layers)
            all_thetas.extend([0.0] * remaining_layers)
            
        elif strategy == 'extend_with_small_random':
            all_gammas.extend(np.random.uniform(-0.1, 0.1, remaining_layers))
            all_betas.extend(np.random.uniform(-0.1, 0.1, remaining_layers))
            all_thetas.extend(np.random.uniform[-0.1, 0.1, remaining_layers])
            
        elif strategy == 'extend_with_linear':
            # Linearly interpolate from last pretrained value to a reasonable target
            last_gamma = pretrained_gammas[-1]
            last_beta = pretrained_betas[-1]
            last_theta = pretrained_thetas[-1]
            
            gamma_extension = np.linspace(last_gamma, np.pi/4, remaining_layers + 1)[1:]
            beta_extension = np.linspace(last_beta, np.pi/2, remaining_layers + 1)[1:]
            theta_extension = np.linspace(last_theta, np.pi/4, remaining_layers + 1)[1:]
            
            all_gammas.extend(gamma_extension)
            all_betas.extend(beta_extension)
            all_thetas.extend(theta_extension)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    return all_gammas + all_betas + all_thetas


def pretrain_and_create_initial_params(graph, num_pretrain_layers=1, total_layers=3,
                                       shots=1024, batch_size=8, 
                                       max_iterations=50, verbose=True, use_local_2q_gates=False, device='CPU'):
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
    pretrained_gammas, pretrained_betas, pretrained_thetas, validity_rate = pretrain_validity_layers(
        graph, qubit_to_edge_map,
        num_layers=num_pretrain_layers,
        shots=shots,
        batch_size=batch_size,
        max_iterations=max_iterations,
        verbose=verbose,
        use_local_2q_gates=use_local_2q_gates,
        device=device
    )
    
    # Create full parameter set for all layers
    initial_params = create_pretrained_initial_params(
        pretrained_gammas, pretrained_betas, pretrained_thetas,
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


# =============================================================================
# Multi-order validity + diversity pretraining (for gate-order hyperparameter
# tuning). Generalizes pretrain_validity_layers in two ways:
#   1. Accepts an arbitrary list of entangling-gate orders (1..num_qubits)
#      instead of a single binary use_local_2q_gates flag, with one shared
#      parameter per order per layer (see create_multi_order_qaoa_circuit).
#   2. Optimizes a composite objective of validity rate AND solution
#      diversity, rather than validity alone, since the goal of this
#      pretraining stage is "valid AND varied" starting points for the main
#      cost-optimization stage.
# =============================================================================
 
def pretrain_validity_diversity_multi_order(graph, gate_orders, num_layers=1,
                                             shots=1024, batch_size=8,
                                             sim_method='statevector',
                                             max_iterations=50, verbose=True,
                                             diversity_weight=0.3,
                                             diversity_method='entropy',
                                             device='CPU', label=None):
    """
    Pre-train a multi-order QAOA circuit to maximize a composite objective of
    validity rate and solution diversity, for a given list of entangling-gate
    orders.
 
    This is the core unit of work for gate-order hyperparameter tuning. It is
    intentionally self-contained and stateless (aside from reading `graph`),
    so it can be called independently per gate-order configuration -- e.g.
    from separate worker processes for parallel sweeps.
 
    Parameters
    ----------
    graph : networkx.DiGraph
        The TSP graph.
    gate_orders : list of int
        Which entangling-gate orders to test, each in [1, num_qubits].
        E.g. [1] = single-qubit only, [4] = only 4-qubit entanglers,
        [1, 4, 8] = combination of orders 1, 4, and 8.
    num_layers : int, optional
        Number of QAOA layers to pretrain together.
    shots : int, optional
        Measurement shots per objective evaluation.
    batch_size : int, optional
        Simulation batch size. Must be >= max(gate_orders).
    sim_method : str, optional
        'statevector' or 'density_matrix'.
    max_iterations : int, optional
        Max COBYLA iterations.
    verbose : bool, optional
    diversity_weight : float, optional
        Weight on the diversity term relative to validity in the composite
        objective: composite = validity_rate + diversity_weight * diversity_score.
        Default 0.3 keeps validity as the dominant goal while still rewarding
        spread across distinct valid tours.
    diversity_method : str, optional
        Passed to compute_diversity_score ('entropy' or 'unique_fraction').
    device : str, optional
        'CPU' or 'GPU'.
    label : str, optional
        Identifier for this trial. Defaults to a string built from gate_orders
        and num_layers.
 
    Returns
    -------
    dict with keys:
        'label', 'gate_orders', 'num_layers', 'num_parameters',
        'gamma_values', 'beta_values', 'theta_values_by_order',
        'best_validity_rate', 'best_diversity_score', 'best_composite_score',
        'best_iteration_stats', 'stats_history', 'gate_count_estimate',
        'optimizer_result'
    """
    gate_orders = sorted(set(int(k) for k in gate_orders))
 
    if label is None:
        label = f"orders_{'-'.join(map(str, gate_orders))}_L{num_layers}"
 
    qubit_to_edge_map = create_qubit_to_edge_map(graph)
    num_qubits = len(qubit_to_edge_map)
 
    gate_count_estimate = estimate_gate_count(gate_orders, num_qubits, batch_size, num_layers)
 
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Pretraining [{label}]")
        print(f"  gate_orders={gate_orders}, layers={num_layers}, "
              f"params={num_layers * (2 + len(gate_orders))}, "
              f"est. entangling gates={gate_count_estimate['total']}")
        print(f"{'=' * 60}")
 
    circuit = create_multi_order_qaoa_circuit(
        graph, qubit_to_edge_map, gate_orders, num_layers=num_layers,
        batch_size=batch_size
    )
 
    results_over_time = []
    best_composite = [-np.inf]
    best_iteration_stats = [None]
 
    def objective(flat_params):
        gamma_values, beta_values, theta_values_by_order = unflatten_multi_order_params(
            flat_params, num_layers, gate_orders
        )
 
        bound_circuit = bind_multi_order_parameters(
            circuit, gamma_values, beta_values, theta_values_by_order
        )
        counts = simulate_large_circuit_in_batches(
            bound_circuit, batch_size, shots, sim_method, device=device
        )
 
        valid_shots, invalid_shots = count_valid_invalid(counts, qubit_to_edge_map, graph)
        total_shots = valid_shots + invalid_shots
        validity_rate = valid_shots / total_shots if total_shots > 0 else 0.0
 
        diversity_score, num_unique_valid = compute_diversity_score(
            counts, qubit_to_edge_map, graph, method=diversity_method
        )
 
        composite_score = validity_rate + diversity_weight * diversity_score
 
        stats = get_qaoa_statistics(counts, qubit_to_edge_map, graph, len(results_over_time))
        stats['validity_rate'] = validity_rate
        stats['diversity_score'] = diversity_score
        stats['num_unique_valid'] = num_unique_valid
        stats['composite_score'] = composite_score
        stats['gate_orders'] = gate_orders
        stats['label'] = label
 
        results_over_time.append(stats)
 
        if composite_score > best_composite[0]:
            best_composite[0] = composite_score
            best_iteration_stats[0] = dict(stats)
 
        if verbose and len(results_over_time) % 5 == 0:
            print(f"  Iter {len(results_over_time):3d}: validity={validity_rate:.2%}, "
                  f"diversity={diversity_score:.3f}, composite={composite_score:.3f} "
                  f"(best={best_composite[0]:.3f})")
 
        # Minimize negative composite score (i.e. maximize composite score)
        return -composite_score
 
    init_gamma, init_beta, init_theta = get_multi_order_initial_parameters(
        num_layers, gate_orders, strategy='random'
    )
    x0 = flatten_multi_order_params(init_gamma, init_beta, init_theta, gate_orders)
 
    if verbose:
        print(f"Optimizing {len(x0)} parameters together...")
 
    optimizer_result = minimize(
        objective, x0=x0, method='COBYLA', options={'maxiter': max_iterations}
    )
 
    final_gamma, final_beta, final_theta = unflatten_multi_order_params(
        optimizer_result.x, num_layers, gate_orders
    )
 
    if verbose:
        best = best_iteration_stats[0] or {}
        print(f"\n{'=' * 60}")
        print(f"Pretraining [{label}] completed!")
        print(f"  Best validity rate:  {best.get('validity_rate', 0):.2%}")
        print(f"  Best diversity score: {best.get('diversity_score', 0):.3f}")
        print(f"  Best composite score: {best_composite[0]:.3f}")
        print(f"{'=' * 60}\n")
 
    return {
        'label': label,
        'gate_orders': gate_orders,
        'num_layers': num_layers,
        'num_parameters': len(x0),
        'gamma_values': list(final_gamma),
        'beta_values': list(final_beta),
        'theta_values_by_order': {k: list(v) for k, v in final_theta.items()},
        'best_validity_rate': (best_iteration_stats[0] or {}).get('validity_rate', 0.0),
        'best_diversity_score': (best_iteration_stats[0] or {}).get('diversity_score', 0.0),
        'best_composite_score': best_composite[0],
        'best_iteration_stats': best_iteration_stats[0],
        'stats_history': results_over_time,
        'gate_count_estimate': gate_count_estimate,
        'optimizer_result': optimizer_result,
    }
