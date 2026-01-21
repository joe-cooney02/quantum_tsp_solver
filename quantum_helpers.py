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
import numpy as np
import networkx as nx
from opt_helpers import get_warm_start_tour


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


def create_tsp_qaoa_circuit(G, qubit_to_edge_map, num_layers=1, barriers=False, warm_start_tour=None, exploration_strength=0.2):
    """
    Create a QAOA circuit for TSP using only single-qubit gates.
    
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
        Wether to add barriers to the circuit. 
        Do not add barriers if you want to split the circuit.
        
    warm_start_tour : list, optional
        Initial tour to warm-start from (e.g., from greedy algorithm)
        If provided, initializes circuit in this solution instead of equal superposition
    
    Returns:
    --------
    QuantumCircuit: Parameterized QAOA circuit
    """
    num_qubits = len(qubit_to_edge_map)
    
    # Create quantum circuit
    qc = QuantumCircuit(num_qubits, num_qubits)
    
    # Create parameters for each layer
    gamma_params = [Parameter(f'gamma_{i}') for i in range(num_layers)]
    beta_params = [Parameter(f'beta_{i}') for i in range(num_layers)]
    
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
                             warm_start_method=None, exploration_strength=0):
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
    
    Returns:
    --------
    QuantumCircuit: QAOA circuit initialized with warm-start
    """
    if warm_start_method is None:
        return create_tsp_qaoa_circuit(G, qubit_to_edge_map, num_layers, warm_start_tour=None, 
                                       exploration_strength=exploration_strength)
    
    # if tour is provided:
    if warm_start_method is list():
        tour = warm_start_method
        
    else:
        # Get warm-start tour
        tour = get_warm_start_tour(G, method=warm_start_method)
    
    # Create circuit with warm-start
    circuit = create_tsp_qaoa_circuit(G, qubit_to_edge_map, num_layers, warm_start_tour=tour, 
                                      exploration_strength=exploration_strength)
    
    return circuit


def split_circuit_for_simulation(circuit, max_qubits_per_batch=10):
    """
    Split a large single-qubit-only circuit into smaller sub-circuits for simulation.
    
    This works because circuits with only single-qubit gates can be simulated 
    independently for each qubit, then the results combined.
    
    Parameters:
    -----------
    circuit : QuantumCircuit
        The circuit to split (must contain only single-qubit gates)
    max_qubits_per_batch : int, optional
        Maximum number of qubits per sub-circuit
    
    Returns:
    --------
    list: List of tuples (sub_circuit, qubit_indices)
          where sub_circuit is a QuantumCircuit and qubit_indices is the list
          of original qubit indices it corresponds to
    """
    num_qubits = circuit.num_qubits
    
    # Check if circuit has 2-qubit gates (shouldn't for our QAOA)
    for instruction in circuit.data:
        if len(instruction.qubits) > 1:
            raise ValueError(f"Circuit contains multi-qubit gate: {instruction.operation.name}. "
                           "Can only split circuits with single-qubit gates.")
    
    # Calculate number of batches needed
    num_batches = (num_qubits + max_qubits_per_batch - 1) // max_qubits_per_batch
    
    sub_circuits = []
    
    for batch_idx in range(num_batches):
        # Determine which qubits go in this batch
        start_qubit = batch_idx * max_qubits_per_batch
        end_qubit = min(start_qubit + max_qubits_per_batch, num_qubits)
        batch_qubit_indices = list(range(start_qubit, end_qubit))
        
        # Create new sub-circuit
        sub_qc = QuantumCircuit(len(batch_qubit_indices))
        
        # Map original qubit indices to sub-circuit indices
        qubit_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(batch_qubit_indices)}
        
        # Copy operations for these qubits
        for instruction in circuit.data:
            qubit = instruction.qubits[0]
            qubit_idx = circuit.qubits.index(qubit)
            
            # Only add if this qubit is in our batch
            if qubit_idx in batch_qubit_indices:
                new_qubit_idx = qubit_map[qubit_idx]
                operation = instruction.operation
                
                # Add the operation to the sub-circuit
                if operation.name == 'measure':
                    sub_qc.measure_all()
                    break  # Don't add more after measure_all
                elif operation.name == 'barrier':
                    sub_qc.barrier()
                else:
                    sub_qc.append(operation, [new_qubit_idx])
        
        sub_circuits.append((sub_qc, batch_qubit_indices))
    
    return sub_circuits


def simulate_split_circuits(sub_circuits, shots=1024, sim_method='statevector'):
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
    sub_results = []
    
    for sub_circuit, qubit_indices in sub_circuits:
        simulator = AerSimulator(method=sim_method)
        
        transpiled_circuit = transpile(sub_circuit, simulator)
        
        job = simulator.run(transpiled_circuit, shots=shots)
        result = job.result()
        counts = result.get_counts()
        sub_results.append((counts, qubit_indices))
    
    # Combine results
    # For single-qubit gates only, each qubit's measurement is independent
    # So we randomly sample from each sub-circuit and concatenate
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


def simulate_large_circuit_in_batches(circuit, max_qubits_per_batch=10, shots=1024, sim_method='statevector', verbose=False):
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
    
    results = simulate_split_circuits(sub_circuits, shots, sim_method)
    
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


def bind_qaoa_parameters(circuit, gamma_values, beta_values):
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
        
    elif strategy == 'linear':
        # Linear interpolation from 0 to optimal-ish values
        gamma_values = np.linspace(0, np.pi/4, num_layers)
        beta_values = np.linspace(0, np.pi/2, num_layers)
        
    elif strategy == 'zero':
        gamma_values = np.zeros(num_layers)
        beta_values = np.zeros(num_layers)
        
    elif strategy == 'tqa':
        # Trotterized Quantum Annealing (proven to help)
        s = np.linspace(0, 1, num_layers + 1)[1:]
        gamma_values = (s * total_time).tolist()
        beta_values = ((1 - s) * total_time).tolist()
        
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return gamma_values, beta_values


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


def get_cost_expectation(bitstrings, counts, qubit_to_edge_map, G, inv_penalty=0):
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
    inv_penalty: float
    penalty term for invalid solution bitstrings.
    If this is negative, this will be |inv_penalty| * (max edge weight of G)
    Default: 0.
    
    Returns:
    --------
    float: cost_expectation
    the expectation value to use in optimization loop.
    """
    # Convert counts to dict if needed
    if isinstance(counts, list):
        counts_dict = {bs: c for bs, c in zip(bitstrings, counts)}
    else:
        counts_dict = counts
        
    # print(counts_dict)
        
    total_shots = sum(counts_dict.values())
    total_cost = 0
    
    # get penalty term
    if inv_penalty < 0:    
        edge_weights = [data['weight'] for u, v, data in G.edges(data=True)]
        inv_penalty = abs(inv_penalty) * max(edge_weights)
    
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
                    
            total_cost += cost * counts_dict[bitstring]
            
        else:
            # apply penalty
            total_cost += inv_penalty * counts_dict[bitstring]
            
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



