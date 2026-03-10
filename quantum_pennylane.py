# -*- coding: utf-8 -*-
"""
PennyLane-based QAOA implementation with GPU acceleration.

This module provides PennyLane implementations of QAOA circuits and optimization
for TSP problems. Key features:
- Native Windows GPU support (no WSL2 needed)
- Gradient-based pretraining for validity
- COBYLA optimization for cost minimization
- Fully-connected layer support (larger batch sizes)

Created: 2026-03-10
"""

import pennylane as qml
from pennylane import numpy as np
import networkx as nx
from collections import Counter
import time


def create_pennylane_device(num_qubits, use_gpu=True, shots=1024):
    """
    Create a PennyLane device for quantum simulation.
    
    Parameters:
    -----------
    num_qubits : int
        Number of qubits in the circuit
    use_gpu : bool, optional
        If True, use GPU-accelerated lightning.gpu device.
        If False, use CPU-based default.qubit. Default: True.
    shots : int, optional
        Number of measurement shots. Default: 1024.
    
    Returns:
    --------
    qml.Device: PennyLane device ready for circuit execution
    """
    if use_gpu:
        try:
            device = qml.device('lightning.gpu', wires=num_qubits, shots=shots)
            print(f"✓ Created GPU device with {num_qubits} qubits")
            return device
        except Exception as e:
            print(f"Warning: GPU device creation failed ({e})")
            print("Falling back to CPU device")
            return qml.device('default.qubit', wires=num_qubits, shots=shots)
    else:
        return qml.device('default.qubit', wires=num_qubits, shots=shots)


def create_qaoa_circuit_pennylane(graph, qubit_to_edge_map, num_layers=1, 
                                  use_local_2q_gates=False, batch_size=8):
    """
    Create a QAOA circuit for TSP using PennyLane.
    
    This creates a QNode (quantum circuit) that can be executed on GPU.
    Unlike Qiskit's batching approach, PennyLane can handle the full circuit
    on GPU, enabling larger fully-connected layers.
    
    Parameters:
    -----------
    graph : networkx.DiGraph
        The TSP graph with weighted edges
    qubit_to_edge_map : dict
        Mapping from qubit index to edge tuple
    num_layers : int, optional
        Number of QAOA layers. Default: 1.
    use_local_2q_gates : bool, optional
        If True, add local 2-qubit gates (CZ) within batches. Default: False.
    batch_size : int, optional
        Batch size for 2-qubit gate locality (only relevant if use_local_2q_gates=True).
        Default: 8.
    
    Returns:
    --------
    function: PennyLane QNode that takes parameters and returns samples
    """
    num_qubits = len(qubit_to_edge_map)
    
    def circuit(params, device):
        """
        QAOA circuit as PennyLane QNode.
        
        Parameters:
        -----------
        params : array
            Parameters [gamma_0, ..., gamma_n, beta_0, ..., beta_n]
        device : qml.Device
            PennyLane device to execute on
        
        Returns:
        --------
        array: Measurement samples (bitstrings)
        """
        # Split parameters
        gammas = params[:num_layers]
        betas = params[num_layers:]
        
        # Create QNode
        @qml.qnode(device)
        def qnode():
            # Initial state: equal superposition
            for i in range(num_qubits):
                qml.Hadamard(wires=i)
            
            # QAOA layers
            for layer in range(num_layers):
                # Cost Hamiltonian
                for qubit_idx, edge in qubit_to_edge_map.items():
                    u, v = edge
                    if graph.has_edge(u, v):
                        weight = graph[u][v]['weight']
                        qml.RZ(2 * gammas[layer] * weight, wires=qubit_idx)
                
                # Optional: Local 2-qubit gates
                if use_local_2q_gates:
                    num_batches = (num_qubits + batch_size - 1) // batch_size
                    
                    for batch_idx in range(num_batches):
                        start_qubit = batch_idx * batch_size
                        end_qubit = min(start_qubit + batch_size, num_qubits)
                        
                        # Even pairs
                        for i in range(start_qubit, end_qubit - 1, 2):
                            if i + 1 < end_qubit:
                                qml.CZ(wires=[i, i + 1])
                        
                        # Odd pairs
                        for i in range(start_qubit + 1, end_qubit - 1, 2):
                            if i + 1 < end_qubit:
                                qml.CZ(wires=[i, i + 1])
                
                # Mixer Hamiltonian
                for qubit_idx in range(num_qubits):
                    qml.RX(2 * betas[layer], wires=qubit_idx)
            
            return qml.sample()
        
        return qnode()
    
    return circuit


def samples_to_counts(samples):
    """
    Convert PennyLane samples to Qiskit-style counts dictionary.
    
    Parameters:
    -----------
    samples : array
        2D array of measurement samples from PennyLane (shots × num_qubits)
    
    Returns:
    --------
    dict: Dictionary mapping bitstrings to counts
    """
    bitstrings = [''.join(map(str, sample.tolist())) for sample in samples]
    counts = Counter(bitstrings)
    return dict(counts)


def compute_soft_validity_loss_pennylane(samples, graph, qubit_to_edge_map):
    """
    Compute differentiable soft validity loss for gradient-based optimization.
    
    This computes a smooth approximation of the validity that has gradients,
    allowing gradient-based optimization to improve validity.
    
    Parameters:
    -----------
    samples : array
        Measurement samples from circuit (shots × num_qubits)
    graph : networkx.DiGraph
        The TSP graph
    qubit_to_edge_map : dict
        Mapping from qubit index to edge tuple
    
    Returns:
    --------
    float: Average violation score (lower is better, 0.0 = valid)
    """
    num_nodes = graph.number_of_nodes()
    num_samples = len(samples)
    
    total_violation = 0.0
    
    for sample in samples:
        # Convert to Python list for indexing
        sample_list = sample.tolist() if hasattr(sample, 'tolist') else list(sample)
        
        # Violation 1: Edge count (should be exactly num_nodes)
        num_ones = sum(sample_list)
        edge_count_violation = abs(num_ones - num_nodes)
        
        # Violation 2: Degree violations
        # Get selected edges
        selected_edges = []
        for qubit_idx, bit in enumerate(sample_list):
            if bit == 1 and qubit_idx in qubit_to_edge_map:
                selected_edges.append(qubit_to_edge_map[qubit_idx])
        
        # Count degree violations
        in_degrees = {node: 0 for node in graph.nodes()}
        out_degrees = {node: 0 for node in graph.nodes()}
        
        for u, v in selected_edges:
            if u in out_degrees:
                out_degrees[u] += 1
            if v in in_degrees:
                in_degrees[v] += 1
        
        degree_violations = sum(
            abs(in_degrees[node] - 1) + abs(out_degrees[node] - 1)
            for node in graph.nodes()
        )
        
        # Weighted combination
        violation = edge_count_violation * 1.0 + degree_violations * 0.5
        total_violation += violation
    
    return total_violation / num_samples


def pretrain_validity_pennylane(graph, qubit_to_edge_map, num_layers=1,
                                shots=1024, max_iterations=50, learning_rate=0.05,
                                use_gpu=True, use_local_2q_gates=False, 
                                batch_size=None, verbose=True):
    """
    Pre-train QAOA layers using gradient-based optimization to maximize validity.
    
    This uses PennyLane's automatic differentiation to compute gradients and
    optimize parameters to maximize the probability of measuring valid TSP tours.
    
    Parameters:
    -----------
    graph : networkx.DiGraph
        The TSP graph
    qubit_to_edge_map : dict
        Mapping from qubit index to edge tuple
    num_layers : int, optional
        Number of QAOA layers to pre-train. Default: 1.
    shots : int, optional
        Number of measurement shots. Default: 1024.
    max_iterations : int, optional
        Maximum gradient descent steps. Default: 50.
    learning_rate : float, optional
        Learning rate for Adam optimizer. Default: 0.05.
    use_gpu : bool, optional
        If True, use GPU acceleration. Default: True.
    use_local_2q_gates : bool, optional
        If True, add local 2-qubit gates. Default: False.
    batch_size : int, optional
        Batch size for simulation. If None, uses full circuit (no batching).
        With GPU, can handle much larger batch sizes (20-30 qubits). Default: None.
    verbose : bool, optional
        Whether to print progress. Default: True.
    
    Returns:
    --------
    tuple: (optimal_gammas, optimal_betas, final_validity_rate)
    """
    from quantum_helpers import is_valid_tsp_tour
    
    num_qubits = len(qubit_to_edge_map)
    
    # Determine batch size
    if batch_size is None:
        # Use full circuit (no batching) - GPU can handle this!
        effective_batch_size = num_qubits
    else:
        effective_batch_size = batch_size
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Pre-training {num_layers} layer(s) with PennyLane (GPU={use_gpu})")
        print(f"  Qubits: {num_qubits}")
        print(f"  Batch size: {effective_batch_size} (full circuit, no batching!)")
        if use_local_2q_gates:
            print(f"  Using local 2-qubit gates")
        print(f"{'='*60}")
    
    # Create device
    device = create_pennylane_device(num_qubits, use_gpu=use_gpu, shots=shots)
    
    # Create circuit
    circuit_func = create_qaoa_circuit_pennylane(
        graph, qubit_to_edge_map, num_layers, 
        use_local_2q_gates, effective_batch_size
    )
    
    # Create differentiable loss function
    @qml.qnode(device, diff_method="adjoint")
    def loss_function(params):
        """Differentiable validity loss"""
        # Get circuit samples
        # Note: We need to recreate the circuit inside the QNode
        gammas = params[:num_layers]
        betas = params[num_layers:]
        
        # Build circuit
        for i in range(num_qubits):
            qml.Hadamard(wires=i)
        
        for layer in range(num_layers):
            # Cost Hamiltonian
            for qubit_idx, edge in qubit_to_edge_map.items():
                u, v = edge
                if graph.has_edge(u, v):
                    weight = graph[u][v]['weight']
                    qml.RZ(2 * gammas[layer] * weight, wires=qubit_idx)
            
            # Optional 2Q gates
            if use_local_2q_gates:
                num_batches = (num_qubits + effective_batch_size - 1) // effective_batch_size
                for batch_idx in range(num_batches):
                    start_qubit = batch_idx * effective_batch_size
                    end_qubit = min(start_qubit + effective_batch_size, num_qubits)
                    
                    for i in range(start_qubit, end_qubit - 1, 2):
                        if i + 1 < end_qubit:
                            qml.CZ(wires=[i, i + 1])
                    
                    for i in range(start_qubit + 1, end_qubit - 1, 2):
                        if i + 1 < end_qubit:
                            qml.CZ(wires=[i, i + 1])
            
            # Mixer
            for qubit_idx in range(num_qubits):
                qml.RX(2 * betas[layer], wires=qubit_idx)
        
        # Return expectation of violation (we'll minimize this)
        # Use a simple differentiable proxy: sum of squared deviations
        measurements = qml.sample()
        
        # Compute violation for each sample
        violations = []
        for sample in measurements:
            num_ones = qml.math.sum(sample)
            edge_violation = (num_ones - num_nodes)**2
            violations.append(edge_violation)
        
        return qml.math.mean(qml.math.stack(violations))
    
    # Initialize parameters (start small for stability)
    params = np.random.uniform(-0.1, 0.1, 2 * num_layers, requires_grad=True)
    
    # Use Adam optimizer
    opt = qml.AdamOptimizer(stepsize=learning_rate)
    
    # Track progress
    best_validity = 0.0
    best_params = params.copy()
    
    if verbose:
        print(f"Starting gradient-based optimization...")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Max iterations: {max_iterations}\n")
    
    # Optimization loop
    for step in range(max_iterations):
        # Gradient descent step
        params, loss = opt.step_and_cost(loss_function, params)
        
        # Evaluate actual validity every 5 steps
        if step % 5 == 0 or step == max_iterations - 1:
            # Run circuit to get samples
            samples = circuit_func(params, device)
            counts = samples_to_counts(samples)
            
            # Calculate actual validity
            total_shots = sum(counts.values())
            valid_shots = 0
            
            for bitstring, count in counts.items():
                if is_valid_tsp_tour(bitstring, qubit_to_edge_map, graph):
                    valid_shots += count
            
            validity_rate = valid_shots / total_shots if total_shots > 0 else 0
            
            # Track best
            if validity_rate > best_validity:
                best_validity = validity_rate
                best_params = params.copy()
            
            if verbose:
                print(f"  Step {step:3d}: Loss = {loss:.4f}, "
                      f"Validity = {validity_rate:.2%} (Best: {best_validity:.2%})")
    
    # Extract final parameters
    optimal_gammas = best_params[:num_layers].tolist()
    optimal_betas = best_params[num_layers:].tolist()
    
    if verbose:
        print(f"\n{'='*60}")
        print("Pre-training completed!")
        print(f"  Final validity rate: {best_validity:.2%}")
        print(f"  Optimal gammas: {optimal_gammas}")
        print(f"  Optimal betas: {optimal_betas}")
        print(f"{'='*60}\n")
    
    return optimal_gammas, optimal_betas, best_validity


def create_pretrained_initial_params_pennylane(pretrained_gammas, pretrained_betas,
                                               total_layers, strategy='extend_with_zeros'):
    """
    Create full initial parameter set using pre-trained layers.
    
    (Same as Qiskit version - reused for consistency)
    
    Parameters:
    -----------
    pretrained_gammas : list
        Pre-trained gamma values
    pretrained_betas : list
        Pre-trained beta values
    total_layers : int
        Total number of layers needed
    strategy : str, optional
        How to initialize remaining layers. Default: 'extend_with_zeros'.
    
    Returns:
    --------
    list: Full parameter list [gamma_0, ..., gamma_n, beta_0, ..., beta_n]
    """
    from quantum_pretraining import create_pretrained_initial_params
    return create_pretrained_initial_params(
        pretrained_gammas, pretrained_betas, total_layers, strategy
    )


def QAOA_pennylane(graph, qubit_to_edge_map, params_init, layers=3, shots=10000,
                   use_gpu=True, use_local_2q_gates=False, batch_size=None,
                   use_soft_validity=True, soft_validity_penalty_base=10.0,
                   max_iterations=200, verbose=True):
    """
    Run QAOA optimization using PennyLane with COBYLA.
    
    This is the main QAOA function that uses:
    - PennyLane for GPU-accelerated simulation
    - COBYLA for robust gradient-free optimization
    - Soft validity penalties for smooth cost landscape
    
    Parameters:
    -----------
    graph : networkx.DiGraph
        The TSP graph
    qubit_to_edge_map : dict
        Mapping from qubit index to edge tuple
    params_init : list or array
        Initial parameters [gamma_0, ..., gamma_n, beta_0, ..., beta_n]
    layers : int, optional
        Number of QAOA layers. Default: 3.
    shots : int, optional
        Number of measurement shots. Default: 10000.
    use_gpu : bool, optional
        If True, use GPU acceleration. Default: True.
    use_local_2q_gates : bool, optional
        If True, add local 2-qubit gates. Default: False.
    batch_size : int, optional
        Batch size (only relevant for 2Q gates). If None, uses full circuit. Default: None.
    use_soft_validity : bool, optional
        If True, use soft validity penalties. Default: True.
    soft_validity_penalty_base : float, optional
        Penalty strength for invalid solutions. Default: 10.0.
    max_iterations : int, optional
        Maximum COBYLA iterations. Default: 200.
    verbose : bool, optional
        Whether to print progress. Default: True.
    
    Returns:
    --------
    dict: Results containing optimal parameters, best tour, cost, validity, etc.
    """
    from quantum_helpers import (
        is_valid_tsp_tour, 
        extract_tour_from_edges,
        compute_soft_validity_score
    )
    from scipy.optimize import minimize
    
    num_qubits = len(qubit_to_edge_map)
    num_nodes = graph.number_of_nodes()
    
    # Determine batch size
    if batch_size is None:
        effective_batch_size = num_qubits
    else:
        effective_batch_size = batch_size
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"QAOA Optimization with PennyLane (GPU={use_gpu})")
        print(f"  Layers: {layers}")
        print(f"  Qubits: {num_qubits}")
        print(f"  Batch size: {effective_batch_size}")
        print(f"  Shots: {shots}")
        print(f"  Soft validity: {use_soft_validity}")
        print(f"{'='*70}\n")
    
    # Create device
    device = create_pennylane_device(num_qubits, use_gpu=use_gpu, shots=shots)
    
    # Create circuit
    circuit_func = create_qaoa_circuit_pennylane(
        graph, qubit_to_edge_map, layers,
        use_local_2q_gates, effective_batch_size
    )
    
    # Get max edge weight for scaling
    edge_weights = [data['weight'] for u, v, data in graph.edges(data=True)]
    max_edge_weight = max(edge_weights)
    
    # Track optimization progress
    iteration_count = [0]
    best_result = {'cost': float('inf'), 'validity': 0.0}
    results_over_time = []
    
    def cost_function(params):
        """Cost function for COBYLA optimization"""
        iteration_count[0] += 1
        
        # Run circuit
        samples = circuit_func(params, device)
        counts = samples_to_counts(samples)
        
        # Calculate cost
        total_cost = 0.0
        total_shots = sum(counts.values())
        valid_shots = 0
        best_valid_cost = float('inf')
        best_valid_tour = None
        
        for bitstring, count in counts.items():
            is_valid, tour = is_valid_tsp_tour(bitstring, qubit_to_edge_map, graph, return_tour=True)
            
            if is_valid:
                valid_shots += count
                
                # Calculate tour cost
                tour_cost = 0
                for i in range(len(tour) - 1):
                    u, v = tour[i], tour[i+1]
                    if graph.has_edge(u, v):
                        tour_cost += graph[u][v]['weight']
                
                cost = tour_cost
                
                # Track best valid tour
                if tour_cost < best_valid_cost:
                    best_valid_cost = tour_cost
                    best_valid_tour = tour
            else:
                # Invalid tour - apply penalty
                if use_soft_validity:
                    violation_score = compute_soft_validity_score(bitstring, qubit_to_edge_map, graph, fast=True)
                    cost = soft_validity_penalty_base * max_edge_weight * (1.0 + violation_score)
                else:
                    num_ones = bitstring.count('1')
                    edge_violation = abs(num_ones - num_nodes)
                    cost = soft_validity_penalty_base * max_edge_weight * edge_violation
            
            total_cost += cost * count
        
        expectation_cost = total_cost / total_shots
        validity_rate = valid_shots / total_shots if total_shots > 0 else 0
        
        # Track results
        result = {
            'iteration': iteration_count[0],
            'cost': expectation_cost,
            'validity': validity_rate,
            'best_valid_cost': best_valid_cost if best_valid_tour else None,
            'best_valid_tour': best_valid_tour,
            'params': params.tolist() if hasattr(params, 'tolist') else list(params)
        }
        results_over_time.append(result)
        
        # Update best
        if validity_rate > 0 and best_valid_cost < best_result['cost']:
            best_result.update({
                'cost': best_valid_cost,
                'validity': validity_rate,
                'tour': best_valid_tour,
                'params': params.tolist() if hasattr(params, 'tolist') else list(params)
            })
        
        # Print progress
        if verbose and iteration_count[0] % 10 == 0:
            print(f"  Iter {iteration_count[0]:3d}: Cost = {expectation_cost:7.2f}, "
                  f"Validity = {validity_rate:5.1%}, "
                  f"Best Valid Cost = {best_valid_cost if best_valid_tour else 'N/A'}")
        
        return expectation_cost
    
    # Run COBYLA optimization
    if verbose:
        print("Starting COBYLA optimization...\n")
    
    start_time = time.time()
    
    result = minimize(
        cost_function,
        x0=params_init,
        method='COBYLA',
        options={'maxiter': max_iterations}
    )
    
    elapsed_time = time.time() - start_time
    
    if verbose:
        print(f"\n{'='*70}")
        print("Optimization completed!")
        print(f"  Total time: {elapsed_time:.2f}s")
        print(f"  Iterations: {iteration_count[0]}")
        print(f"  Final cost: {result.fun:.2f}")
        print(f"  Best valid cost: {best_result['cost']}")
        print(f"  Best validity: {best_result['validity']:.2%}")
        print(f"{'='*70}\n")
    
    return {
        'optimal_params': result.x,
        'final_cost': result.fun,
        'best_valid_cost': best_result.get('cost', None),
        'best_tour': best_result.get('tour', None),
        'best_validity': best_result.get('validity', 0.0),
        'iterations': iteration_count[0],
        'runtime': elapsed_time,
        'results_over_time': results_over_time,
        'success': result.success,
        'message': result.message
    }
