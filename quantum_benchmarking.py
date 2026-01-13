# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 12:51:39 2025

@author: joeco

this file contains benchmarking and hyperparameter tuning functions for QAOA implementations
"""

from quantum_helpers import simulate_large_circuit_in_batches, bind_qaoa_parameters
from quantum_helpers import get_initial_parameters, count_valid_invalid, get_best_cost
import time
import tracemalloc

def benchmark_batch_sizes(circuit, batch_sizes=None, shots=1024, verbose=True, sim_method='statevector'):
    """
    Benchmark different batch sizes to find the optimal configuration for your system.
    
    Parameters:
    -----------
    circuit : QuantumCircuit
        The circuit to benchmark (should have bound parameters)
    batch_sizes : list, optional
        List of batch sizes to test. If None, uses [5, 8, 10, 12, 15, 20]
    shots : int, optional
        Number of shots for each test
    verbose : bool, optional
        Whether to print progress information
    
    Returns:
    --------
    dict: Dictionary with batch size as key and dict of metrics as value
              {'batch_size': {'time': float, 'memory_peak': float, 'success': bool}}
    """

    
    if batch_sizes is None:
        # Default batch sizes to test
        batch_sizes = [5, 8, 10, 12, 15, 20]
    
    # Filter batch sizes that make sense
    batch_sizes = [bs for bs in batch_sizes if bs <= circuit.num_qubits]
    
    results = {}
    
    print(f"Benchmarking batch sizes for {circuit.num_qubits}-qubit circuit")
    print(f"Testing batch sizes: {batch_sizes}")
    print(f"Shots per test: {shots}")
    print("=" * 60)
    
    for batch_size in batch_sizes:
        if verbose:
            print(f"\n--- Testing batch_size={batch_size} ---")
        
        # Start memory tracking
        tracemalloc.start()
        start_time = time.time()
        
        success = True
        error_msg = None
        
        try:
           # run sim     
           sim_results = simulate_large_circuit_in_batches(
               circuit, 
               max_qubits_per_batch=batch_size, 
               shots=shots,
               sim_method=sim_method,
               verbose=verbose
               )
            
        except MemoryError as e:
            success = False
            error_msg = "Out of Memory"
            if verbose:
                print(f"ERROR: {error_msg}")
                
        except Exception as e:
            success = False
            error_msg = str(e)
            if verbose:
                print(f"ERROR: {error_msg}")
        
        # Stop timing and memory tracking
        elapsed_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Store results
        results[batch_size] = {
            'time': elapsed_time,
            'memory_peak_mb': peak / 1024 / 1024,
            'memory_current_mb': current / 1024 / 1024,
            'success': success,
            'error': error_msg
        }
        
        if verbose:
            if success:
                print("✓ Success!")
                print(f"  Time: {elapsed_time:.2f} seconds")
                print(f"  Peak Memory: {peak / 1024 / 1024:.1f} MB")
            else:
                print(f"✗ Failed: {error_msg}")
    
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"{'Batch Size':<12} {'Status':<10} {'Time (s)':<12} {'Peak Mem (MB)':<15}")
    print("-" * 60)
    
    for batch_size in sorted(results.keys()):
        r = results[batch_size]
        status = "✓ OK" if r['success'] else "✗ FAIL"
        time_str = f"{r['time']:.2f}" if r['success'] else "N/A"
        mem_str = f"{r['memory_peak_mb']:.1f}" if r['success'] else "N/A"
        print(f"{batch_size:<12} {status:<10} {time_str:<12} {mem_str:<15}")
    
    # Find optimal batch size
    successful = {bs: r for bs, r in results.items() if r['success']}
    if successful:
        # Optimal is the fastest sim that works.
        optimal_bs = min(successful, key=lambda outer_key: successful[outer_key]['time'])
        print("\n" + "=" * 60)
        print(f"RECOMMENDED: Use batch_size={optimal_bs}")
        print(f"  Expected time: {successful[optimal_bs]['time']:.2f}s for {shots} shots")
        print(f"  Peak memory: {successful[optimal_bs]['memory_peak_mb']:.1f} MB")
        print("=" * 60)
    else:
        print("\n⚠ WARNING: No batch sizes succeeded. Circuit may be too large.")
    
    return results, optimal_bs


def benchmark_shot_counts(circuit, qubit_to_edge_map, G, batch_size=10,
                          shot_counts=None, sim_method='automatic', verbose=True):
    """
    Benchmark different shot counts to find optimal trade-off between runtime and solution quality.
    
    Parameters:
    -----------
    circuit : QuantumCircuit
        Parameterized, bound QAOA circuit
    qubit_to_edge_map : dict
        Mapping from qubits to edges
    G : networkx.DiGraph
        The TSP graph
    batch_size : int, optional
        Batch size for simulation
    shot_counts : list, optional
        List of shot counts to test. If None, uses [256, 512, 1024, 2048, 4096, 8192, 16384]
    sim_method : str, optional
        Simulation method
    verbose : bool, optional
        Print progress
    
    Returns:
    --------
    dict: Results with shot count as key and metrics as value
    """
    import time
    
    if shot_counts is None:
        shot_counts = [256, 512, 1024, 2048, 4096, 8192, 16384]
    
    results = {}
    
    print(f"Benchmarking shot counts for {circuit.num_qubits}-qubit circuit")
    print(f"Testing shot counts: {shot_counts}")
    print("=" * 60)
    
    for shots in shot_counts:
        if verbose:
            print(f"\n--- Testing shots={shots} ---")
        
        start_time = time.time()
        
        try:
            # Run simulation
            counts = simulate_large_circuit_in_batches(
                circuit,
                max_qubits_per_batch=batch_size,
                shots=shots,
                sim_method=sim_method
            )
            
            # Get statistics
            valid, invalid = count_valid_invalid(counts, qubit_to_edge_map, G)
            best_cost = get_best_cost(counts, qubit_to_edge_map, G)
            
            elapsed_time = time.time() - start_time
            
            results[shots] = {
                'time': elapsed_time,
                'valid_shots': valid,
                'invalid_shots': invalid,
                'valid_percentage': 100 * valid / shots,
                'best_cost': best_cost,
                'unique_bitstrings': len(counts),
                'success': True
            }
            
            if verbose:
                print("✓ Complete!")
                print(f"  Time: {elapsed_time:.2f}s")
                print(f"  Valid tours: {valid}/{shots} ({100*valid/shots:.1f}%)")
                print(f"  Best cost: {best_cost}")
                print(f"  Unique bitstrings: {len(counts)}")
        
        except Exception as e:
            results[shots] = {
                'success': False,
                'error': str(e)
            }
            if verbose:
                print(f"✗ Failed: {e}")
    
    print("\n" + "=" * 60)
    print("SHOT COUNT BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"{'Shots':<10} {'Time (s)':<12} {'Valid %':<12} {'Best Cost':<12} {'Unique BS':<12}")
    print("-" * 60)
    
    for shots in sorted(results.keys()):
        r = results[shots]
        if r['success']:
            print(f"{shots:<10} {r['time']:<12.2f} {r['valid_percentage']:<12.1f} "
                  f"{r['best_cost'] if r['best_cost'] else 'None':<12} {r['unique_bitstrings']:<12}")
        else:
            print(f"{shots:<10} FAILED: {r['error']}")
    
    # Recommend optimal shot count
    successful = {s: r for s, r in results.items() if r['success'] and r['best_cost'] is not None}
    if successful:
        # Find diminishing returns point
        print("\n" + "=" * 60)
        print("ANALYSIS:")
        
        # Check if more shots improve the solution
        best_costs = [(s, r['best_cost']) for s, r in sorted(successful.items())]
        
        optimal_shots = None
        for i in range(len(best_costs) - 1):
            current_shots, current_cost = best_costs[i]
            next_shots, next_cost = best_costs[i + 1]
            
            # If cost doesn't improve significantly, use current shots
            if next_cost >= current_cost * 0.99:  # Less than 1% improvement
                optimal_shots = current_shots
                break
        
        if optimal_shots is None:
            optimal_shots = max(successful.keys())
        
        print(f"RECOMMENDED: Use shots={optimal_shots}")
        print(f"  Best cost achieved: {successful[optimal_shots]['best_cost']}")
        print(f"  Valid tour rate: {successful[optimal_shots]['valid_percentage']:.1f}%")
        print(f"  Time per run: {successful[optimal_shots]['time']:.2f}s")
        print("=" * 60)
    
    return results