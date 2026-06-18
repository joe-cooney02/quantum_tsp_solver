# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 13:50:24 2025

@author: joeco

this file will contain helper functions for QAOA algorithm(s).
thoughts on RQAOA: it will be much easier to implement in this project 
than it was in the cancer project - "fixing an edge" just means including a 
certain edge in the graph, which corresponds to one qubit. so, just trun on one qubit.
This is much easier here because travel time information 
"""

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.qpy import dump, load
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2
import numpy as np
import networkx as nx
import itertools as it
from opt_helpers import get_warm_start_tour


def check_gpu_available():
    """
    Check if GPU is available for qiskit-aer simulation.
    
    Returns:
    --------
    bool: True if GPU is available, False otherwise
    """
    try:
        simulator = AerSimulator(method='statevector', device='GPU')
        return True
    except Exception as e:
        print(f"GPU check failed: {e}")
        return False


def get_simulator(method='statevector', device='CPU', fusion=True, precision='single', shots=10000, verbose=False):
    """
    Safely create an AerSimulator with proper device selection.
    
    Parameters:
    -----------
    method : str
        Simulation method ('statevector' or 'density_matrix')
    device : str
        Requested device ('CPU' or 'GPU')
    
    Returns:
    --------
    AerSimulator: Configured simulator
    """
    
    if device.upper() == 'GPU':
        try:
            
            simulator = AerSimulator(method=method, device='GPU', fusion_enable=fusion, precision=precision, 
                                     cuStateVec_enable=True, max_parallel_experiments=6, batched_shots_gpu=True,
                                     max_parallel_threads=6)
            
            
            '''
            options = {
                "backend_options": {
                    "method": method,
                    "device": device,
                    "precision": precision,
                    "cuStateVec_enable": True,
                    "max_parallel_experiments": 6,
                    "max_parallel_threads": 6,
                    "batched_shots_gpu": True,  # Keep shots on-device
                    "default_shots": shots
                }
            }
            
            simulator = EstimatorV2(options=options)
            '''
            
            if verbose:
                print("✓ Using GPU for simulation")
            return simulator
        except Exception as e:
            print(f"⚠ GPU requested but not available: {e}")
            print("  Falling back to CPU")
            return AerSimulator(method=method, device='CPU', 
                                fusion_enable=fusion, cuStateVec_enable=True)
    else:
        return AerSimulator(method=method, device='CPU')


def create_qubit_to_edge_map(G):
    """
    Create a mapping from qubit indices to edges in a graph.
    
    Parameters:
    -----------
    G : networkx.DiGraph or networkx.Graph
        The graph to create mapping for
    
    Returns:
    --------
    dict: Mapping from qubit index (int) to edge tuple (u, v)
          e.g., {0: (0, 1), 1: (0, 2), 2: (1, 0), ...}
    """
    qubit_to_edge_map = {}
    
    # Iterate through all edges and assign qubit indices
    for idx, (u, v) in enumerate(G.edges()):
        qubit_to_edge_map[idx] = (u, v)
    
    return qubit_to_edge_map


def create_edge_to_qubit_map(G):
    """
    Create a reverse mapping from edges to qubit indices.
    
    Parameters:
    -----------
    G : networkx.DiGraph or networkx.Graph
        The graph to create mapping for
    
    Returns:
    --------
    dict: Mapping from edge tuple (u, v) to qubit index (int)
          e.g., {(0, 1): 0, (0, 2): 1, (1, 0): 2, ...}
    """
    edge_to_qubit_map = {}
    
    for idx, (u, v) in enumerate(G.edges()):
        edge_to_qubit_map[(u, v)] = idx
    
    return edge_to_qubit_map


def create_tsp_qaoa_circuit(G, qubit_to_edge_map, num_layers=1, barriers=False, warm_start_tour=None, 
                           exploration_strength=0.2, use_local_2q_gates=False, batch_size=8):
    """
    Create a QAOA circuit for TSP.
    
    Parameters:
    -----------
    G : networkx.DiGraph
        The TSP graph with weighted edges
    qubit_to_edge_map : dict
        Mapping from qubit index to edge tuple
        e.g., {0: (0, 1), 1: (0, 2), ...}
    num_layers : int, optional
        Number of QAOA layers (p parameter)
    
    barriers: bool, optional
        Whether to add barriers to the circuit. 
        Do not add barriers if you want to split the circuit.
        
    warm_start_tour : list, optional
        Initial tour to warm-start from (e.g., from greedy algorithm)
        If provided, initializes circuit in this solution instead of equal superposition
    
    use_local_2q_gates : bool, optional
        If True, add local 2-qubit gates (CZ) within batches for entanglement.
        These gates only connect qubits within the same batch, maintaining
        compatibility with batched simulation.
    
    batch_size : int, optional
        Size of batches for simulation. Only relevant if use_local_2q_gates=True.
        2-qubit gates will only connect qubits within the same batch.
    
    Returns:
    --------
    QuantumCircuit: Parameterized QAOA circuit
    """
    num_qubits = len(qubit_to_edge_map)
    
    # Create quantum circuit
    qc = QuantumCircuit(num_qubits, num_qubits)
    
    # Create parameters for each layer & purpose:
        # gamma: single-qubit rz
        # beta: mixer
        # theta: two-qubit rz
        
    gamma_params = [Parameter(f'gamma_{i}') for i in range(num_layers)]
    beta_params = [Parameter(f'beta_{i}') for i in range(num_layers)]
    theta_params = [Parameter(f'theta_{i}') for i in range(num_layers)]
    
    # Initial state
    if warm_start_tour is not None:
        # Warm-start: Initialize in a basis state corresponding to the tour
        # Convert tour to bitstring
        edge_to_qubit = {edge: qubit for qubit, edge in qubit_to_edge_map.items()}
        
        for i in range(len(warm_start_tour) - 1):
            edge = (warm_start_tour[i], warm_start_tour[i+1])
            if edge in edge_to_qubit:
                qubit_idx = edge_to_qubit[edge]
                qc.x(qubit_idx)  # Set this qubit to |1⟩
        
        # Add small superposition to allow exploration
        # This is key to avoid getting stuck in local minima
        for qubit_idx in range(num_qubits):
            qc.ry(exploration_strength, qubit_idx)  # Small rotation to add exploration
    else:
        # Standard initialization: equal superposition
        qc.h(range(num_qubits))
    
    if barriers:
        qc.barrier()
    
    # QAOA layers
    for layer in range(num_layers):
        # Cost Hamiltonian (problem Hamiltonian)
        # Apply RZ rotations based on edge weights
        for qubit_idx, edge in qubit_to_edge_map.items():
            u, v = edge
            if G.has_edge(u, v):
                weight = G[u][v]['weight']
                # RZ gate for cost function
                qc.rz(2 * gamma_params[layer] * weight, qubit_idx)
        
        # Optional: Add local 2-qubit entangling gates within batches
        if use_local_2q_gates:
            # Add CZ gates between adjacent qubits within each batch
            num_batches = (num_qubits + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches):
                # Determine qubit range for this batch
                start_qubit = batch_idx * batch_size
                end_qubit = min(start_qubit + batch_size, num_qubits)

                        
                qubits_in_batch = [i for i in range(start_qubit, end_qubit-1)]
                qubit_pairs = it.combinations(qubits_in_batch, 2)
                for pair in qubit_pairs:
                    qc.cz(pair[0], pair[1])
                    qc.rz(theta_params[layer], pair[1])
                    qc.cz(pair[1], pair[0])
        
        if barriers:
            qc.barrier()
        
        # Mixer Hamiltonian
        # Apply RX rotations (single-qubit mixer)
        for qubit_idx in range(num_qubits):
            qc.rx(2 * beta_params[layer], qubit_idx)
        
        if barriers:
            qc.barrier()
    
    
    for i in range(num_qubits):
        qc.measure(i, i)
    
    return qc


def create_warm_started_qaoa(G, qubit_to_edge_map, num_layers=1, 
                             warm_start_method=None, exploration_strength=0,
                             use_local_2q_gates=False, batch_size=8):
    """
    Create a warm-started QAOA circuit with configurable exploration.
    
    Parameters:
    -----------
    G : networkx.DiGraph
        The TSP graph
    qubit_to_edge_map : dict
        Qubit to edge mapping
    num_layers : int, optional
        Number of QAOA layers
    warm_start_method : str or a tour (list), optional
        Method to generate initial tour: 'greedy', 'random', or None for standard QAOA
    exploration_strength : float, optional
        How much to perturb the initial state (0 = pure warm-start, larger = more exploration)
        Typical range: 0.1 to 0.5
    use_local_2q_gates : bool, optional
        If True, add local 2-qubit gates within batches
    batch_size : int, optional
        Batch size for 2-qubit gate locality
    
    Returns:
    --------
    QuantumCircuit: QAOA circuit initialized with warm-start
    """
    if warm_start_method is None:
        return create_tsp_qaoa_circuit(G, qubit_to_edge_map, num_layers, warm_start_tour=None, 
                                       exploration_strength=exploration_strength,
                                       use_local_2q_gates=use_local_2q_gates, batch_size=batch_size)
    
    # if tour is provided:
    if warm_start_method is list():
        tour = warm_start_method
        
    else:
        # Get warm-start tour
        tour = get_warm_start_tour(G, method=warm_start_method)
    
    # Create circuit with warm-start
    circuit = create_tsp_qaoa_circuit(G, qubit_to_edge_map, num_layers, warm_start_tour=tour, 
                                      exploration_strength=exploration_strength,
                                      use_local_2q_gates=use_local_2q_gates, batch_size=batch_size)
    
    return circuit


def split_circuit_for_simulation(circuit, max_qubits_per_batch=10):
    """
    Split a circuit into smaller sub-circuits for simulation.
    
    This function intelligently handles circuits with local 2-qubit gates by:
    1. Detecting which qubits are entangled via 2-qubit gates
    2. Ensuring entangled qubits stay in the same batch
    3. Splitting only at boundaries where no entanglement crosses
    
    For circuits with only single-qubit gates, splits freely every max_qubits_per_batch.
    For circuits with local 2-qubit gates (within batches), splits at batch boundaries.
    
    Parameters:
    -----------
    circuit : QuantumCircuit
        The circuit to split
    max_qubits_per_batch : int, optional
        Maximum number of qubits per sub-circuit (used as batch size for 2Q gates)
    
    Returns:
    --------
    list: List of tuples (sub_circuit, qubit_indices)
          where sub_circuit is a QuantumCircuit and qubit_indices is the list
          of original qubit indices it corresponds to
    
    Raises:
    -------
    ValueError: If circuit has 2-qubit gates that cross expected batch boundaries
    """
    
    # First, detect if circuit has any 2-qubit gates
    has_2q_gates = False
    two_qubit_connections = set()  # Set of (qubit_i, qubit_j) pairs
    
    for instruction in circuit.data:
        if len(instruction.qubits) > 1:
            has_2q_gates = True
            # Get qubit indices
            qubit_indices = [circuit.qubits.index(q) for q in instruction.qubits]
            # Store as sorted tuple to avoid duplicates
            two_qubit_connections.add(tuple(sorted(qubit_indices)))
    
    if not has_2q_gates:
        # Simple case: no 2Q gates, split freely
        return _split_circuit_simple(circuit, max_qubits_per_batch)
    
    # Complex case: has 2Q gates, need to respect entanglement boundaries
    return _split_circuit_with_2q_gates(circuit, max_qubits_per_batch, two_qubit_connections)


def _split_circuit_simple(circuit, max_qubits_per_batch):
    """
    Helper function: Split circuit with only single-qubit gates.
    
    This is the original splitting logic - works for any single-qubit-only circuit.
    """
    num_qubits = circuit.num_qubits
    num_batches = (num_qubits + max_qubits_per_batch - 1) // max_qubits_per_batch
    
    sub_circuits = []
    
    for batch_idx in range(num_batches):
        start_qubit = batch_idx * max_qubits_per_batch
        end_qubit = min(start_qubit + max_qubits_per_batch, num_qubits)
        batch_qubit_indices = list(range(start_qubit, end_qubit))
        
        sub_qc = QuantumCircuit(len(batch_qubit_indices))
        qubit_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(batch_qubit_indices)}
        
        for instruction in circuit.data:
            qubit = instruction.qubits[0]
            qubit_idx = circuit.qubits.index(qubit)
            
            if qubit_idx in batch_qubit_indices:
                new_qubit_idx = qubit_map[qubit_idx]
                operation = instruction.operation
                
                if operation.name == 'measure':
                    sub_qc.measure_all()
                    break
                elif operation.name == 'barrier':
                    sub_qc.barrier()
                else:
                    sub_qc.append(operation, [new_qubit_idx])
        
        sub_circuits.append((sub_qc, batch_qubit_indices))
    
    return sub_circuits


def _split_circuit_with_2q_gates(circuit, batch_size, two_qubit_connections):
    """
    Helper function: Split circuit with local 2-qubit gates.
    
    Assumes 2Q gates only connect qubits within the same batch of size `batch_size`.
    Verifies this assumption and raises error if violated.
    """
    num_qubits = circuit.num_qubits
    
    # Verify all 2Q gates are within batch boundaries
    for q1, q2 in two_qubit_connections:
        batch1 = q1 // batch_size
        batch2 = q2 // batch_size
        
        if batch1 != batch2:
            raise ValueError(
                f"Circuit has 2-qubit gate between qubits {q1} and {q2}, "
                f"which are in different batches (batch {batch1} and {batch2}). "
                f"Cannot split this circuit safely. "
                f"Batch size is {batch_size}, so qubits {batch1*batch_size}-{(batch1+1)*batch_size-1} "
                f"and {batch2*batch_size}-{(batch2+1)*batch_size-1} are in separate batches."
            )
    
    # If we get here, all 2Q gates are within batches, so we can split at batch boundaries
    num_batches = (num_qubits + batch_size - 1) // batch_size
    sub_circuits = []
    
    for batch_idx in range(num_batches):
        start_qubit = batch_idx * batch_size
        end_qubit = min(start_qubit + batch_size, num_qubits)
        batch_qubit_indices = list(range(start_qubit, end_qubit))
        
        sub_qc = QuantumCircuit(len(batch_qubit_indices))
        qubit_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(batch_qubit_indices)}
        
        for instruction in circuit.data:
            qubits_involved = [circuit.qubits.index(q) for q in instruction.qubits]
            
            # Check if all qubits in this instruction are in our batch
            if all(q_idx in batch_qubit_indices for q_idx in qubits_involved):
                operation = instruction.operation
                
                if operation.name == 'measure':
                    sub_qc.measure_all()
                    break
                elif operation.name == 'barrier':
                    sub_qc.barrier()
                else:
                    # Map to new qubit indices
                    new_qubit_indices = [qubit_map[q_idx] for q_idx in qubits_involved]
                    sub_qc.append(operation, new_qubit_indices)
        
        sub_circuits.append((sub_qc, batch_qubit_indices))
    
    return sub_circuits


def simulate_split_circuits(sub_circuits, shots=1024, sim_method='statevector', device='CPU'):
    """
    Simulate split sub-circuits and combine results into full bitstrings.
    
    Parameters:
    -----------
    sub_circuits : list
        List of (sub_circuit, qubit_indices) tuples from split_circuit_for_simulation
    shots : int, optional
        Number of shots for simulation
    
    Returns:
    --------
    dict: Dictionary mapping full bitstrings to counts
    """
    
    # Simulate each sub-circuit
    transpiled_circuits = []
    circuit_qubit_indices = []
    sub_results = []
    
    simulator = get_simulator(method=sim_method, device=device, shots=shots)
    
    for sub_circuit, qubit_indices in sub_circuits:
        
        transpiled_circuit = transpile(sub_circuit, simulator)
        transpiled_circuits.append(transpiled_circuit)
        
        circuit_qubit_indices.append(qubit_indices)
        
        
    # we set this to run in parellel with max 8 threads above
    # change this because now it is an EstimatorV2 (needs PUB's)
    job = simulator.run(transpiled_circuits, shots=shots)
    result = job.result()
    
    for i in range(len(transpiled_circuits)):
        counts = result.get_counts(transpiled_circuits[i])
        sub_results.append((counts, circuit_qubit_indices[i])) 
    
    # Combine results
    # For single-qubit gates only, each qubit's measurement is independent
    # So we randomly sample from each sub-circuit and concatenate
    # TODO: figure out what this is actually doing and how to best accelerate
        # for high-shot simulations, the EstimatorV2 precision=0 option calculates the exp val <Y|O|Y> on the GPU.
        # this would be great if we had an observable O that accurately reflected the qubit contributions to travel times.
        # alternatively, figure out how to calculate the get_cost_expectation either in parallel on cpu or on gpu.
        # I don't think we have to worry about the statevector being loaded on and off the GPU as, with 2-qubit gates,
        # this is minimal compared to the circuit execution time. also, the exp val calculation may need the VRAM anyway.
    combined_counts = {}
    
    for shot_idx in range(shots):
        full_bitstring = ['0'] * sum(len(indices) for _, indices in sub_results)
        
        # Sample from each sub-circuit
        for counts, qubit_indices in sub_results:
            # Sample a bitstring from this sub-circuit's distribution
            bitstrings = list(counts.keys())
            probabilities = np.array(list(counts.values())) / sum(counts.values())
            sampled_bitstring = np.random.choice(bitstrings, p=probabilities)
            
            # Place bits in correct positions
            for i, qubit_idx in enumerate(qubit_indices):
                full_bitstring[qubit_idx] = sampled_bitstring[-(i+1)]  # Qiskit uses reverse order
        
        # Convert to string and count
        full_bitstring_str = ''.join(full_bitstring)
        combined_counts[full_bitstring_str] = combined_counts.get(full_bitstring_str, 0) + 1
    
    return combined_counts


def simulate_large_circuit_in_batches(circuit, max_qubits_per_batch=10, shots=1024, sim_method='statevector', verbose=False, device='CPU'):
    """
    Convenience function to split and simulate a large circuit in one call.
    
    Parameters:
    -----------
    circuit : QuantumCircuit
        The large circuit to simulate
    max_qubits_per_batch : int, optional
        Maximum qubits per batch
    shots : int, optional
        Number of shots
    verbose: Bool
        wether to print progress messages
    
    Returns:
    --------
    dict: Combined measurement results (bitstring -> count)
    """
    
    if verbose:
        print(f"Splitting circuit with {circuit.num_qubits} qubits into batches of {max_qubits_per_batch}...")
    
    sub_circuits = split_circuit_for_simulation(circuit, max_qubits_per_batch)
    
    if verbose:
        print(f"Created {len(sub_circuits)} sub-circuits")
    
        for idx, (sub_qc, indices) in enumerate(sub_circuits):
            print(f"  Batch {idx+1}: qubits {indices[0]}-{indices[-1]} ({len(indices)} qubits)")
    
    if verbose:
        print(f"\nSimulating with {shots} shots...")
    
    results = simulate_split_circuits(sub_circuits, shots, sim_method, device=device)
    
    if verbose:
        print(f"Simulation complete! Got {len(results)} unique bitstrings")
    
    return results


def save_qaoa_circuit(circuit, filename):
    """
    Save a QAOA circuit to a .qpy file.
    
    Parameters:
    -----------
    circuit : QuantumCircuit
        The QAOA circuit to save
    filename : str
        Output filename (should end with .qpy)
    
    Returns:
    --------
    None
    """
    if not filename.endswith('.qpy'):
        filename += '.qpy'
    
    with open(filename, 'wb') as f:
        dump(circuit, f)
    
    print(f"Circuit saved to {filename}")


def load_qaoa_circuit(filename):
    """
    Load a QAOA circuit from a .qpy file.
    
    Parameters:
    -----------
    filename : str
        Input filename (should end with .qpy)
    
    Returns:
    --------
    QuantumCircuit: The loaded circuit
    """
    if not filename.endswith('.qpy'):
        filename += '.qpy'
    
    with open(filename, 'rb') as f:
        circuits = load(f)
    
    print(f"Circuit loaded from {filename}")
    return circuits[0]


def bind_qaoa_parameters(circuit, gamma_values, beta_values, theta_values):
    """
    Bind parameter values to a QAOA circuit.
    
    Parameters:
    -----------
    circuit : QuantumCircuit
        Parameterized QAOA circuit
    gamma_values : list or array
        Values for gamma parameters (one per layer)
    beta_values : list or array
        Values for beta parameters (one per layer)
    
    Returns:
    --------
    QuantumCircuit: Circuit with bound parameters
    """
    num_layers = len(gamma_values)
    
    # Create parameter dictionary
    param_dict = {}
    for i in range(num_layers):
        param_dict[f'gamma_{i}'] = gamma_values[i]
        param_dict[f'beta_{i}'] = beta_values[i]
        param_dict[f'theta_{i}'] = theta_values[i]
    
    # Bind parameters
    bound_circuit = circuit.assign_parameters(param_dict)
    
    return bound_circuit


def get_initial_parameters(num_layers, strategy='random', total_time=1.0):
    """
    Generate initial parameter values for QAOA optimization.
    
    Parameters:
    -----------
    num_layers : int
        Number of QAOA layers
    strategy : str, optional
        Strategy for initialization: 'random', 'linear', 'zero'
    
    Returns:
    --------
    tuple: (gamma_values, beta_values)
    """
    if strategy == 'random':
        gamma_values = np.random.uniform(0, 2*np.pi, num_layers)
        beta_values = np.random.uniform(0, np.pi, num_layers)
        theta_values = np.random.uniform(0, 2*np.pi, num_layers)
        
    elif strategy == 'linear':
        # Linear interpolation from 0 to optimal-ish values
        gamma_values = np.linspace(0, np.pi/4, num_layers)
        beta_values = np.linspace(0, np.pi/2, num_layers)
        theta_values = np.linspace(0, np.pi/4, num_layers)
        
    elif strategy == 'zero':
        gamma_values = np.zeros(num_layers)
        beta_values = np.zeros(num_layers)
        theta_values = gamma_values = np.zeros(num_layers)
        
    elif strategy == 'tqa':
        # Trotterized Quantum Annealing (proven to help)
        s = np.linspace(0, 1, num_layers + 1)[1:]
        gamma_values = (s * total_time).tolist()
        beta_values = ((1 - s) * total_time).tolist()
        theta_values = gamma_values = (s * total_time).tolist()
        
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return gamma_values, beta_values, theta_values


def create_and_save_qaoa_circuit(G, qubit_to_edge_map, num_layers, filename):
    """
    Convenience function to create and save a QAOA circuit in one step.
    
    Parameters:
    -----------
    G : networkx.DiGraph
        The TSP graph
    qubit_to_edge_map : dict
        Qubit to edge mapping
    num_layers : int
        Number of QAOA layers
    filename : str
        Output filename
    
    Returns:
    --------
    QuantumCircuit: The created circuit (also saved to file)
    """
    circuit = create_tsp_qaoa_circuit(G, qubit_to_edge_map, num_layers)
    save_qaoa_circuit(circuit, filename)
    return circuit



def is_valid_tsp_tour(bitstring, qubit_to_edge_map, G, return_tour=False):
    """
    Check if a QAOA bitstring represents a valid TSP tour.
    
    A valid tour must:
    1. Form a connected path through all nodes
    2. Visit each node exactly once
    3. Return to the starting node
    
    Parameters:
    -----------
    bitstring : str or list
        Binary string where 1 indicates edge is in the tour, 0 indicates not in tour
        e.g., "101100" or [1, 0, 1, 1, 0, 0]
    qubit_to_edge_map : dict
        Mapping from qubit index to edge tuple
        e.g., {0: (0, 1), 1: (0, 2), 2: (1, 2), ...}
    G : networkx.DiGraph
        The graph representing the TSP problem
    return_tour : bool, optional
        If True, also return the tour sequence if valid
    
    Returns:
    --------
    bool (or tuple): True if valid tour, False otherwise
                     If return_tour=True, returns (is_valid, tour_list or None)
    """
    # Convert bitstring to list if it's a string
    if isinstance(bitstring, str):
        bits = [int(b) for b in bitstring]
    else:
        bits = list(bitstring)
    
    # Get the edges selected by the bitstring
    selected_edges = []
    for qubit_idx, bit in enumerate(bits):
        if bit == 1:
            if qubit_idx in qubit_to_edge_map:
                selected_edges.append(qubit_to_edge_map[qubit_idx])
    
    # Create a graph from selected edges
    tour_graph = nx.DiGraph()
    tour_graph.add_edges_from(selected_edges)
    
    n_nodes = G.number_of_nodes()
  
    # Check 0: Must have same nodes in each graph
    if set(tour_graph.nodes) != set(G.nodes):
        return (False, None) if return_tour else False
    
    # Check 1: Must have exactly n edges for a tour of n nodes
    if len(selected_edges) != n_nodes:
        return (False, None) if return_tour else False
    
    # Check 2: Each node must have exactly one incoming and one outgoing edge
    for node in G.nodes():
        in_degree = tour_graph.in_degree(node)
        out_degree = tour_graph.out_degree(node)
        
        if in_degree != 1 or out_degree != 1:
            return (False, None) if return_tour else False
    
    # Check 3: Must form a single cycle (not multiple disconnected cycles)
    # Find strongly connected components
    components = list(nx.strongly_connected_components(tour_graph))
    
    if len(components) != 1:
        # Multiple cycles or disconnected
        return (False, None) if return_tour else False
    
    # Check 4: Verify it's actually a cycle that visits all nodes
    if tour_graph.number_of_nodes() != n_nodes:
        return (False, None) if return_tour else False
    
    # If we need to return the tour, extract it
    if return_tour:
        tour = extract_tour_from_edges(selected_edges)
        return (True, tour)
    
    return True


def extract_tour_from_edges(edges):
    """
    Extract ordered tour from a list of edges.
    
    Parameters:
    -----------
    edges : list of tuples
        List of (from_node, to_node) edges forming a tour
    
    Returns:
    --------
    list: Ordered sequence of nodes in the tour
    """
    if not edges:
        return []
    
    # Build adjacency dict
    next_node = {}
    for u, v in edges:
        next_node[u] = v
    
    # Start from any node
    start = edges[0][0]
    tour = [start]
    current = start
    
    # Follow the path
    while True:
        if current not in next_node:
            break
        next_n = next_node[current]
        if next_n == start:
            tour.append(start)  # Complete the cycle
            break
        tour.append(next_n)
        current = next_n
    
    return tour


def filter_valid_shots(bitstrings, qubit_to_edge_map, G, return_tours=False):
    """
    Filter a list of QAOA shots to keep only valid tours.
    
    Parameters:
    -----------
    bitstrings : list
        List of bitstrings from QAOA measurements
    qubit_to_edge_map : dict
        Mapping from qubit index to edge tuple
    G : networkx.DiGraph
        The graph representing the TSP problem
    return_tours : bool, optional
        If True, return tours along with bitstrings
    
    Returns:
    --------
    list or tuple: Valid bitstrings (and tours if return_tours=True)
    """
    valid_bitstrings = []
    valid_tours = []
    
    for bitstring in bitstrings:
        if return_tours:
            is_valid, tour = is_valid_tsp_tour(bitstring, qubit_to_edge_map, G, return_tour=True)
            if is_valid:
                valid_bitstrings.append(bitstring)
                valid_tours.append(tour)
        else:
            if is_valid_tsp_tour(bitstring, qubit_to_edge_map, G):
                valid_bitstrings.append(bitstring)
    
    if return_tours:
        return valid_bitstrings, valid_tours
    return valid_bitstrings


def postselect_best_tour(bitstrings, counts, qubit_to_edge_map, G):
    """
    Postselect the best valid tour from QAOA results.
    
    Parameters:
    -----------
    bitstrings : list
        List of unique bitstrings from QAOA measurements
    counts : list or dict
        Counts/frequencies for each bitstring (same order as bitstrings)
        Can be a list of counts or dict mapping bitstring to count
    qubit_to_edge_map : dict
        Mapping from qubit index to edge tuple
    G : networkx.DiGraph
        The graph representing the TSP problem
    
    Returns:
    --------
    tuple: (best_bitstring, best_tour, tour_cost, success_rate)
           success_rate is the fraction of shots that were valid tours
    """
    # Convert counts to dict if needed
    if isinstance(counts, list):
        counts_dict = {bs: c for bs, c in zip(bitstrings, counts)}
    else:
        counts_dict = counts
        
    # print(counts_dict)
    
    total_shots = sum(counts_dict.values())
    valid_shots = 0
    best_cost = float('inf')
    best_bitstring = None
    best_tour = None
    
    for bitstring in bitstrings:
        is_valid, tour = is_valid_tsp_tour(bitstring, qubit_to_edge_map, G, return_tour=True)
        
        if is_valid:
            valid_shots += counts_dict.get(bitstring, 0)
            
            # Calculate tour cost
            cost = 0
            for i in range(len(tour) - 1):
                u, v = tour[i], tour[i+1]
                if G.has_edge(u, v):
                    cost += G[u][v]['weight']
            
            # Update best if this is better
            if cost < best_cost:
                best_cost = cost
                best_bitstring = bitstring
                best_tour = tour
                
    
    success_rate = valid_shots / total_shots if total_shots > 0 else 0
    
    return best_bitstring, best_tour, best_cost, success_rate


def compute_soft_validity_score(bitstring, qubit_to_edge_map, graph, fast=True):
    """
    Compute a soft validity score measuring how close a bitstring is to being valid.
    
    Returns a violation score where:
    - 0.0 = perfectly valid tour
    - Higher values = more violations
    
    This provides a gradient toward validity, avoiding barren plateaus.
    
    Parameters:
    -----------
    bitstring : str
        Binary string representing edge selections
    qubit_to_edge_map : dict
        Mapping from qubit index to edge tuple
    graph : networkx.DiGraph
        The TSP graph
    fast : bool, optional
        If True, skip disconnection check (faster). Default True.
    
    Returns:
    --------
    float: Violation score (0.0 = valid, higher = more invalid)
    """
    # Convert bitstring if needed
    if isinstance(bitstring, str):
        bits = [int(b) for b in bitstring]
    else:
        bits = list(bitstring)
    
    # Get selected edges
    selected_edges = []
    for qubit_idx, bit in enumerate(bits):
        if bit == 1 and qubit_idx in qubit_to_edge_map:
            selected_edges.append(qubit_to_edge_map[qubit_idx])
    
    num_nodes = graph.number_of_nodes()
    
    # Violation 1: Wrong number of edges
    # Should have exactly N edges for N nodes
    edge_count_violation = abs(len(selected_edges) - num_nodes)
    
    # Violation 2: Degree violations
    # Each node should have in-degree = 1 and out-degree = 1
    in_degrees = {node: 0 for node in graph.nodes()}
    out_degrees = {node: 0 for node in graph.nodes()}
    
    for u, v in selected_edges:
        if u in out_degrees:  # Safety check
            out_degrees[u] += 1
        if v in in_degrees:  # Safety check
            in_degrees[v] += 1
    
    degree_violations = sum(
        abs(in_degrees[node] - 1) + abs(out_degrees[node] - 1)
        for node in graph.nodes()
    )
    
    # Violation 3: Disconnection penalty (optional, expensive)
    disconnection_penalty = 0
    if not fast and edge_count_violation == 0 and degree_violations == 0:
        # Only check if we have right structure
        tour_graph = nx.DiGraph()
        tour_graph.add_edges_from(selected_edges)
        
        # Count strongly connected components
        num_components = len(list(nx.strongly_connected_components(tour_graph)))
        disconnection_penalty = num_components - 1  # Should be 1 component
    
    # Weighted combination
    # Edge count is most important (need right number)
    # Degree violations next (need right structure)
    # Disconnection last (only matters if structure is right)
    total_violation = (
        edge_count_violation * 1.0 +
        degree_violations * 0.5 +
        disconnection_penalty * 2.0
    )
    
    return total_violation


def get_cost_expectation(bitstrings, counts, qubit_to_edge_map, G, inv_penalty=0, 
                        use_soft_validity=False, soft_validity_penalty_base=10.0):
    """
    get expectation of cost value from QAOA results.
        
    Parameters:
    -----------
    bitstrings : list
        List of unique bitstrings from QAOA measurements
    counts : list or dict
        Counts/frequencies for each bitstring (same order as bitstrings)
        Can be a list of counts or dict mapping bitstring to count
    qubit_to_edge_map : dict
        Mapping from qubit index to edge tuple
    G : networkx.DiGraph
        The graph representing the TSP problem
    inv_penalty : float
        Penalty term for invalid solution bitstrings.
        If negative, will be |inv_penalty| * (max edge weight)
        Only used if use_soft_validity=False. Default: 0.
    use_soft_validity : bool
        If True, use soft validity penalties (gradient toward validity).
        If False, use hard penalty (flat landscape). Default: False.
    soft_validity_penalty_base : float
        Base multiplier for soft validity penalties. Only used if use_soft_validity=True.
        Penalty = base * max_edge_weight * (1 + violation_score). Default: 10.0.
    
    Returns:
    --------
    float: cost_expectation
        The expectation value to use in optimization loop.
    """
    # Convert counts to dict if needed
    if isinstance(counts, list):
        counts_dict = {bs: c for bs, c in zip(bitstrings, counts)}
    else:
        counts_dict = counts
        
    total_shots = sum(counts_dict.values())
    total_cost = 0
    num_nodes = G.number_of_nodes()
    
    # Get max edge weight for scaling
    edge_weights = [data['weight'] for u, v, data in G.edges(data=True)]
    max_edge_weight = max(edge_weights)
    
    # Get penalty term for hard penalty mode
    if inv_penalty < 0:
        inv_penalty = abs(inv_penalty) * max_edge_weight
    
    # iterate through bitstrings
    for bitstring in bitstrings:
        is_valid, tour = is_valid_tsp_tour(bitstring, qubit_to_edge_map, G, return_tour=True)
        
        if is_valid:               
            # Calculate tour cost
            cost = 0
            for i in range(len(tour) - 1):
                u, v = tour[i], tour[i+1]
                if G.has_edge(u, v):
                    cost += G[u][v]['weight']
        else:
            # Invalid tour - apply penalty
            if use_soft_validity:
                # Soft penalty: scales with violation severity
                violation_score = compute_soft_validity_score(bitstring, qubit_to_edge_map, G, fast=True)
                cost = soft_validity_penalty_base * max_edge_weight * (1.0 + violation_score)
            else:
                # Hard penalty: fixed penalty with simple Hamming distance
                doi = abs(bitstring.count('1') - num_nodes)
                cost = inv_penalty * doi if doi > 0 else inv_penalty
                    
        total_cost += cost * counts_dict[bitstring]
            
    cost_expectation = total_cost / total_shots
    
    return cost_expectation 


def count_valid_invalid(counts, qubit_to_edge_map, G):
    """
    Count the number of valid and invalid tour shots from QAOA results.
    
    Parameters:
    -----------
    counts : dict
        Dictionary mapping bitstrings to counts (from simulation results)
    qubit_to_edge_map : dict
        Mapping from qubit index to edge tuple
    G : networkx.DiGraph
        The graph representing the TSP problem
    
    Returns:
    --------
    tuple: (valid_shots, invalid_shots)
    """
    valid_shots = 0
    invalid_shots = 0
    
    for bitstring, count in counts.items():
        is_valid = is_valid_tsp_tour(bitstring, qubit_to_edge_map, G)
        
        if is_valid:
            valid_shots += count
        else:
            invalid_shots += count
    
    return valid_shots, invalid_shots


def get_best_cost(counts, qubit_to_edge_map, G):
    """
    Get the best (lowest) tour cost from QAOA results.
    
    Only considers valid tours.
    
    Parameters:
    -----------
    counts : dict
        Dictionary mapping bitstrings to counts (from simulation results)
    qubit_to_edge_map : dict
        Mapping from qubit index to edge tuple
    G : networkx.DiGraph
        The graph representing the TSP problem
    
    Returns:
    --------
    float or None: Best tour cost, or None if no valid tours found
    """
    best_cost = float('inf')
    found_valid = False
    
    for bitstring in counts.keys():
        is_valid, tour = is_valid_tsp_tour(bitstring, qubit_to_edge_map, G, return_tour=True)
        
        if is_valid:
            found_valid = True
            # Calculate tour cost
            cost = 0
            for i in range(len(tour) - 1):
                u, v = tour[i], tour[i+1]
                if G.has_edge(u, v):
                    cost += G[u][v]['weight']
            
            if cost < best_cost:
                best_cost = cost
    
    return best_cost if found_valid else None


def get_qaoa_statistics(counts, qubit_to_edge_map, G, iteration):
    """
    Get comprehensive statistics from QAOA results.
    This is really a wrapper function for postselect_best_tour.
    
    Parameters:
    -----------
    counts : dict
        Dictionary mapping bitstrings to counts (from simulation results)
    qubit_to_edge_map : dict
        Mapping from qubit index to edge tuple
    G : networkx.DiGraph
        The graph representing the TSP problem
    
    Returns:
    --------
    dict: Statistics including valid/invalid counts, best cost, success rate, etc.
    """
    
    # Get best tour details
    best_bitstring, best_tour, best_cost, success_rate = postselect_best_tour(
        list(counts.keys()), counts, qubit_to_edge_map, G
    )
    
    total_shots = sum(counts.values())
    valid_shots = total_shots * success_rate
    invalid_shots = total_shots - valid_shots
    
    if best_tour is None:
        # this occurs when there is no valid solution found.
        best_tour = nx.DiGraph()
    
    stats = {
        'iteration': iteration,
        'total_shots': total_shots,
        'valid_shots': valid_shots,
        'invalid_shots': invalid_shots,
        'valid_percentage': 100 * success_rate,
        'best_cost': best_cost,
        'best_bitstring': best_bitstring,
        'best_tour': best_tour,
        'num_unique_bitstrings': len(counts)
    }
    
    return stats


def hamming_distance(bitstring1, bitstring2):
    """
    Calculate Hamming distance between two bitstrings.
    
    Parameters:
    -----------
    bitstring1 : str
        First bitstring
    bitstring2 : str
        Second bitstring
    
    Returns:
    --------
    int: Number of positions where the bitstrings differ
    """
    if len(bitstring1) != len(bitstring2):
        raise ValueError("Bitstrings must have the same length")
    
    return sum(b1 != b2 for b1, b2 in zip(bitstring1, bitstring2))


def tour_to_bitstring(tour, qubit_to_edge_map):
    """
    Convert a tour (list of nodes) to a bitstring using the qubit-to-edge mapping.
    
    Parameters:
    -----------
    tour : list
        Ordered list of nodes representing the tour (including return to start)
        e.g., [0, 1, 2, 3, 0]
    qubit_to_edge_map : dict
        Mapping from qubit index to edge tuple
        e.g., {0: (0, 1), 1: (0, 2), ...}
    
    Returns:
    --------
    str: Bitstring where '1' indicates edge is in tour, '0' indicates not in tour
    """
    # Create reverse mapping: edge -> qubit
    edge_to_qubit = {edge: qubit for qubit, edge in qubit_to_edge_map.items()}
    
    num_qubits = len(qubit_to_edge_map)
    bitstring = ['0'] * num_qubits
    
    # Set bits for edges in the tour
    for i in range(len(tour) - 1):
        edge = (tour[i], tour[i + 1])
        if edge in edge_to_qubit:
            qubit_idx = edge_to_qubit[edge]
            bitstring[qubit_idx] = '1'
    
    return ''.join(bitstring)


def bitstring_to_tour(bitstring, qubit_to_edge_map):
    """
    Convert a bitstring to a tour if it represents a valid tour.
    
    Parameters:
    -----------
    bitstring : str
        Binary string representing edge selections
    qubit_to_edge_map : dict
        Mapping from qubit index to edge tuple
    
    Returns:
    --------
    list or None: Tour as ordered list of nodes, or None if invalid
    """
    # Get selected edges
    selected_edges = []
    for qubit_idx, bit in enumerate(bitstring):
        if bit == '1' and qubit_idx in qubit_to_edge_map:
            selected_edges.append(qubit_to_edge_map[qubit_idx])
    
    if not selected_edges:
        return None
    
    # Try to build a tour from the edges
    return extract_tour_from_edges(selected_edges)


# =============================================================================
# Multi-order gate infrastructure (added for gate-order hyperparameter tuning)
#
# Generalizes the binary use_local_2q_gates flag into an arbitrary list of
# gate "orders" (k-qubit entangling gates), with ONE shared parameter per
# order per layer (not one parameter per individual gate). This is a
# deliberate choice based on the symmetry of the validity constraint: since
# the TSP one-hot/permutation validity check doesn't depend on edge weights,
# qubits playing equivalent structural roles are in the same symmetry orbit,
# so tying their parameters together loses little-to-no expressivity for the
# validity objective while keeping the parameter count (and thus shot budget)
# from exploding. See create_multi_order_qaoa_circuit for details.
# =============================================================================
 
def _apply_parity_ladder_gate(qc, qubits, theta_param):
    """
    Apply a k-qubit Pauli-Z-string rotation exp(-i*theta * Z_q0 Z_q1 ... Z_q(k-1))
    using the standard CNOT-staircase ("parity ladder") construction.
 
    This is the natural generalization of a single-qubit RZ (k=1, no CNOTs needed)
    up to an arbitrary k-qubit entangling phase gate. For k=2 this is the
    standard two-qubit ZZ-interaction gate.
 
    Parameters
    ----------
    qc : QuantumCircuit
        Circuit to append the gate to (modified in place).
    qubits : list of int
        Qubit indices the gate acts on. Order matters only for the CNOT
        ladder construction, not for the resulting operator.
    theta_param : Parameter or float
        Rotation angle (shared across all gates of this order/layer).
    """
    k = len(qubits)
 
    if k == 1:
        qc.rz(2 * theta_param, qubits[0])
        return
 
    # Compute parity of qubits[0..k-2] onto qubits[k-1]
    for i in range(k - 1):
        qc.cx(qubits[i], qubits[i + 1])
 
    qc.rz(2 * theta_param, qubits[-1])
 
    # Uncompute parity
    for i in range(k - 2, -1, -1):
        qc.cx(qubits[i], qubits[i + 1])
 
 
def estimate_gate_count(gate_orders, num_qubits, batch_size, num_layers=1):
    """
    Estimate the number of multi-qubit entangling gates a multi-order circuit
    will contain, before building it. Useful for sanity-checking a hyperparameter
    sweep configuration, since the number of k-qubit groups per batch grows
    combinatorially (C(batch_size, k)).
 
    Parameters
    ----------
    gate_orders : list of int
        Gate orders to be used (1 through num_qubits).
    num_qubits : int
        Total number of qubits in the problem.
    batch_size : int
        Simulation batch size (a k-qubit gate must fit within one batch).
    num_layers : int, optional
        Number of QAOA layers.
 
    Returns
    -------
    dict: {order: total_gate_count_across_all_layers, ...}, plus 'total' key.
    """
    from math import comb
 
    num_batches = (num_qubits + batch_size - 1) // batch_size
    counts = {}
 
    for k in sorted(set(int(o) for o in gate_orders)):
        per_layer = 0
        for batch_idx in range(num_batches):
            start_qubit = batch_idx * batch_size
            end_qubit = min(start_qubit + batch_size, num_qubits)
            batch_qubits = end_qubit - start_qubit
            if batch_qubits >= k:
                per_layer += comb(batch_qubits, k) if k > 1 else batch_qubits
        counts[k] = per_layer * num_layers
 
    counts['total'] = sum(counts.values())
    return counts
 
 
def create_multi_order_qaoa_circuit(G, qubit_to_edge_map, gate_orders, num_layers=1,
                                    barriers=False, warm_start_tour=None,
                                    exploration_strength=0.2, batch_size=8):
    """
    Create a QAOA circuit generalizing create_tsp_qaoa_circuit's single
    use_local_2q_gates flag into an arbitrary set of entangling-gate orders.
 
    For each layer, applies (in order): the standard weighted cost RZ layer
    (gamma), then for every order k in gate_orders, a k-qubit parity-ladder
    entangling gate (see _apply_parity_ladder_gate) applied to every group of
    k qubits within each simulation batch, sharing ONE parameter theta_k per
    layer across all groups of that order. Finally the standard RX mixer
    (beta) is applied.
 
    Parameters
    ----------
    G : networkx.DiGraph
        TSP graph with weighted edges.
    qubit_to_edge_map : dict
        Mapping from qubit index to edge tuple.
    gate_orders : list of int
        Which gate orders (k-qubit entangling blocks) to include in every
        layer, each in [1, num_qubits]. E.g. [1] = single-qubit phase only,
        [2] = pairwise entangling (like use_local_2q_gates=True), [1,3,6] =
        combination of single, triple, and six-qubit entanglers.
    num_layers : int, optional
        Number of QAOA layers (p).
    barriers : bool, optional
        Whether to add barriers (don't use if you intend to split the circuit).
    warm_start_tour : list, optional
        Initial tour to warm-start from.
    exploration_strength : float, optional
        Perturbation strength when warm-starting.
    batch_size : int, optional
        Simulation batch size. Every k-qubit gate is confined to a single
        batch so the circuit remains splittable via
        simulate_large_circuit_in_batches. Consequently max(gate_orders)
        must be <= batch_size.
 
    Returns
    -------
    QuantumCircuit: Parameterized circuit with one Parameter per (order, layer)
        named 'theta_order{k}_{layer}', plus the usual 'gamma_{layer}' and
        'beta_{layer}' parameters.
 
    Raises
    ------
    ValueError: if any requested order is < 1, exceeds num_qubits, or exceeds
        batch_size (a k-qubit gate cannot span multiple batches).
    """
    num_qubits = len(qubit_to_edge_map)
 
    gate_orders = sorted(set(int(k) for k in gate_orders))
 
    if any(k < 1 for k in gate_orders):
        raise ValueError("All gate_orders must be >= 1.")
    if max(gate_orders) > num_qubits:
        raise ValueError(
            f"gate_orders contains order {max(gate_orders)}, which exceeds "
            f"the number of qubits ({num_qubits})."
        )
    if max(gate_orders) > batch_size:
        raise ValueError(
            f"gate_orders contains order {max(gate_orders)}, which exceeds "
            f"batch_size ({batch_size}). A k-qubit gate must fit entirely "
            f"within one simulation batch to remain compatible with "
            f"simulate_large_circuit_in_batches. Increase batch_size to at "
            f"least {max(gate_orders)} to test this order (note: this also "
            f"increases simulation cost for that batch)."
        )
 
    qc = QuantumCircuit(num_qubits, num_qubits)
 
    gamma_params = [Parameter(f'gamma_{i}') for i in range(num_layers)]
    beta_params = [Parameter(f'beta_{i}') for i in range(num_layers)]
    order_params = {
        k: [Parameter(f'theta_order{k}_{i}') for i in range(num_layers)]
        for k in gate_orders
    }
 
    # Initial state
    if warm_start_tour is not None:
        edge_to_qubit = {edge: qubit for qubit, edge in qubit_to_edge_map.items()}
        for i in range(len(warm_start_tour) - 1):
            edge = (warm_start_tour[i], warm_start_tour[i + 1])
            if edge in edge_to_qubit:
                qc.x(edge_to_qubit[edge])
        for qubit_idx in range(num_qubits):
            qc.ry(exploration_strength, qubit_idx)
    else:
        qc.h(range(num_qubits))
 
    if barriers:
        qc.barrier()
 
    num_batches = (num_qubits + batch_size - 1) // batch_size
 
    for layer in range(num_layers):
        # Cost Hamiltonian (same as create_tsp_qaoa_circuit)
        for qubit_idx, edge in qubit_to_edge_map.items():
            u, v = edge
            if G.has_edge(u, v):
                weight = G[u][v]['weight']
                qc.rz(2 * gamma_params[layer] * weight, qubit_idx)
 
        # Multi-order entangling gates: one shared parameter per order per layer
        for k in gate_orders:
            theta_k = order_params[k][layer]
            for batch_idx in range(num_batches):
                start_qubit = batch_idx * batch_size
                end_qubit = min(start_qubit + batch_size, num_qubits)
                qubits_in_batch = list(range(start_qubit, end_qubit))
 
                if len(qubits_in_batch) < k:
                    continue
 
                if k == 1:
                    for q in qubits_in_batch:
                        _apply_parity_ladder_gate(qc, [q], theta_k)
                else:
                    for group in it.combinations(qubits_in_batch, k):
                        _apply_parity_ladder_gate(qc, list(group), theta_k)
 
        if barriers:
            qc.barrier()
 
        # Mixer Hamiltonian
        for qubit_idx in range(num_qubits):
            qc.rx(2 * beta_params[layer], qubit_idx)
 
        if barriers:
            qc.barrier()
 
    for i in range(num_qubits):
        qc.measure(i, i)
 
    return qc
 
 
def bind_multi_order_parameters(circuit, gamma_values, beta_values, theta_values_by_order):
    """
    Bind parameter values to a circuit built with create_multi_order_qaoa_circuit.
 
    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit produced by create_multi_order_qaoa_circuit.
    gamma_values : list or array
        One value per layer.
    beta_values : list or array
        One value per layer.
    theta_values_by_order : dict
        {order_k: [theta values, one per layer], ...}
 
    Returns
    -------
    QuantumCircuit: Circuit with all parameters bound.
    """
    num_layers = len(gamma_values)
 
    param_dict = {}
    for i in range(num_layers):
        param_dict[f'gamma_{i}'] = gamma_values[i]
        param_dict[f'beta_{i}'] = beta_values[i]
 
    for k, theta_values in theta_values_by_order.items():
        for i in range(num_layers):
            param_dict[f'theta_order{k}_{i}'] = theta_values[i]
 
    return circuit.assign_parameters(param_dict)
 
 
def get_multi_order_initial_parameters(num_layers, gate_orders, strategy='random'):
    """
    Generate initial parameter values for a multi-order QAOA circuit.
 
    Parameters
    ----------
    num_layers : int
    gate_orders : list of int
    strategy : str, optional
        'random' or 'zero'.
 
    Returns
    -------
    tuple: (gamma_values, beta_values, theta_values_by_order)
    """
    gate_orders = sorted(set(int(k) for k in gate_orders))
 
    if strategy == 'random':
        gamma_values = np.random.uniform(0, 2 * np.pi, num_layers)
        beta_values = np.random.uniform(0, np.pi, num_layers)
        theta_values_by_order = {
            k: np.random.uniform(-0.1, 0.1, num_layers) for k in gate_orders
        }
    elif strategy == 'zero':
        gamma_values = np.zeros(num_layers)
        beta_values = np.zeros(num_layers)
        theta_values_by_order = {k: np.zeros(num_layers) for k in gate_orders}
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
 
    return gamma_values, beta_values, theta_values_by_order
 
 
def flatten_multi_order_params(gamma_values, beta_values, theta_values_by_order, gate_orders):
    """
    Flatten gamma/beta/theta-by-order parameters into a single 1D array
    suitable for scipy.optimize.minimize. Order is:
    [gamma_0..gamma_L, beta_0..beta_L, theta_order[0]_0..L, theta_order[1]_0..L, ...]
    with gate_orders taken in sorted order.
    """
    gate_orders = sorted(set(int(k) for k in gate_orders))
    flat = list(gamma_values) + list(beta_values)
    for k in gate_orders:
        flat += list(theta_values_by_order[k])
    return np.array(flat, dtype=float)
 
 
def unflatten_multi_order_params(flat_params, num_layers, gate_orders):
    """
    Inverse of flatten_multi_order_params.
 
    Returns
    -------
    tuple: (gamma_values, beta_values, theta_values_by_order)
    """
    gate_orders = sorted(set(int(k) for k in gate_orders))
 
    gamma_values = flat_params[0:num_layers]
    beta_values = flat_params[num_layers:2 * num_layers]
 
    theta_values_by_order = {}
    offset = 2 * num_layers
    for k in gate_orders:
        theta_values_by_order[k] = flat_params[offset:offset + num_layers]
        offset += num_layers
 
    return gamma_values, beta_values, theta_values_by_order
 
 
def compute_diversity_score(counts, qubit_to_edge_map, G, method='entropy'):
    """
    Quantify the diversity of VALID solutions found in a set of measurement
    counts. Intended as a secondary objective alongside validity rate during
    pretraining, since high validity with all shots collapsing onto one
    bitstring is much less useful than high validity spread across many
    distinct valid tours.
 
    Parameters
    ----------
    counts : dict
        Bitstring -> count, as returned by simulate_large_circuit_in_batches.
    qubit_to_edge_map : dict
    G : networkx.DiGraph
    method : str, optional
        'entropy' (default): normalized Shannon entropy of the distribution
            over valid bitstrings, in [0, 1]. 1 = perfectly uniform spread
            across all observed valid bitstrings, 0 = all valid shots are
            the same bitstring.
        'unique_fraction': fraction of valid shots that landed on a bitstring
            not yet counted (i.e. unique valid bitstrings / total valid shots).
 
    Returns
    -------
    tuple: (diversity_score, num_unique_valid_bitstrings)
        diversity_score is 0.0 if there are no valid shots at all.
    """
    valid_bitstrings = []
    valid_counts = []
 
    for bitstring, count in counts.items():
        if is_valid_tsp_tour(bitstring, qubit_to_edge_map, G):
            valid_bitstrings.append(bitstring)
            valid_counts.append(count)
 
    if not valid_bitstrings:
        return 0.0, 0
 
    total_valid = sum(valid_counts)
    num_unique = len(valid_bitstrings)
 
    if method == 'entropy':
        probs = np.array(valid_counts, dtype=float) / total_valid
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        max_entropy = np.log2(num_unique) if num_unique > 1 else 1.0
        diversity_score = entropy / max_entropy if max_entropy > 0 else 0.0
    elif method == 'unique_fraction':
        diversity_score = num_unique / total_valid
    else:
        raise ValueError(f"Unknown diversity method: {method}")
 
    return diversity_score, num_unique

