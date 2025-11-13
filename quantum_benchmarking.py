# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 12:51:39 2025

@author: joeco

this file contains benchmarking and hyperparameter tuning functions for QAOA implementations
"""

from quantum_helpers import simulate_large_circuit_in_batches
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