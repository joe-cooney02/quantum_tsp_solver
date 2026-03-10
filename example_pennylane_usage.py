# -*- coding: utf-8 -*-
"""
Example usage of PennyLane QAOA with gradient-based pretraining.

This script demonstrates:
1. Gradient-based validity pretraining (20-50 steps)
2. COBYLA cost optimization (100-200 steps)
3. GPU acceleration on Windows (no WSL2 needed!)
4. Comparison with baseline approaches

Created: 2026-03-10
"""

import networkx as nx
import numpy as np
from quantum_helpers import create_qubit_to_edge_map
from quantum_pennylane import (
    pretrain_validity_pennylane,
    create_pretrained_initial_params_pennylane,
    QAOA_pennylane
)
from visualization_algorithms import plot_qaoa_comparison
import time


def create_test_graph(num_nodes=8):
    """Create a test TSP graph."""
    G = nx.complete_graph(num_nodes, create_using=nx.DiGraph())
    
    # Add random weights
    np.random.seed(42)
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.randint(10, 100)
    
    return G


def example_basic_usage():
    """
    Example 1: Basic PennyLane QAOA with gradient pretraining.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic PennyLane QAOA with Gradient Pretraining")
    print("="*80)
    
    # Create test graph
    graph = create_test_graph(num_nodes=6)
    qubit_to_edge_map = create_qubit_to_edge_map(graph)
    
    print(f"\nProblem: {graph.number_of_nodes()}-node TSP")
    print(f"Qubits needed: {len(qubit_to_edge_map)}")
    
    # Stage 1: Gradient-based pretraining
    print("\n" + "-"*80)
    print("STAGE 1: Gradient-Based Validity Pretraining")
    print("-"*80)
    
    pretrained_gammas, pretrained_betas, validity = pretrain_validity_pennylane(
        graph,
        qubit_to_edge_map,
        num_layers=1,                # Pretrain just 1 layer
        shots=2048,
        max_iterations=30,           # 30 gradient steps
        learning_rate=0.05,
        use_gpu=True,                # Enable GPU!
        use_local_2q_gates=False,    # Start simple
        verbose=True
    )
    
    print(f"\n✓ Pretraining completed: {validity:.2%} validity achieved")
    
    # Extend to 3 layers
    params_init = create_pretrained_initial_params_pennylane(
        pretrained_gammas,
        pretrained_betas,
        total_layers=3,
        strategy='extend_with_zeros'
    )
    
    # Stage 2: COBYLA optimization
    print("\n" + "-"*80)
    print("STAGE 2: COBYLA Cost Optimization")
    print("-"*80)
    
    result = QAOA_pennylane(
        graph,
        qubit_to_edge_map,
        params_init,
        layers=3,
        shots=5000,
        use_gpu=True,
        use_local_2q_gates=False,
        use_soft_validity=True,
        soft_validity_penalty_base=10.0,
        max_iterations=100,
        verbose=True
    )
    
    # Print results
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Best tour cost: {result['best_valid_cost']}")
    print(f"Best tour: {result['best_tour']}")
    print(f"Final validity: {result['best_validity']:.2%}")
    print(f"Total runtime: {result['runtime']:.2f}s")
    print(f"Optimization successful: {result['success']}")
    
    return result


def example_compare_approaches():
    """
    Example 2: Compare different optimization approaches.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Comparing Different Approaches")
    print("="*80)
    
    # Create test graph
    graph = create_test_graph(num_nodes=5)
    qubit_to_edge_map = create_qubit_to_edge_map(graph)
    
    results = {}
    
    # Approach 1: No pretraining (baseline)
    print("\n" + "-"*80)
    print("Approach 1: Baseline (No Pretraining)")
    print("-"*80)
    
    params_random = np.random.uniform(0, 0.1, 6)  # 3 layers * 2 params
    
    results['baseline'] = QAOA_pennylane(
        graph,
        qubit_to_edge_map,
        params_random,
        layers=3,
        shots=5000,
        use_gpu=True,
        use_soft_validity=True,
        max_iterations=100,
        verbose=False
    )
    
    print(f"✓ Baseline: cost={results['baseline']['best_valid_cost']}, "
          f"validity={results['baseline']['best_validity']:.2%}, "
          f"time={results['baseline']['runtime']:.1f}s")
    
    # Approach 2: Gradient pretraining
    print("\n" + "-"*80)
    print("Approach 2: Gradient-Based Pretraining")
    print("-"*80)
    
    gammas, betas, _ = pretrain_validity_pennylane(
        graph, qubit_to_edge_map,
        num_layers=1, max_iterations=20,
        use_gpu=True, verbose=False
    )
    
    params_pretrained = create_pretrained_initial_params_pennylane(
        gammas, betas, total_layers=3
    )
    
    results['pretrained'] = QAOA_pennylane(
        graph, qubit_to_edge_map,
        params_pretrained,
        layers=3, shots=5000,
        use_gpu=True, use_soft_validity=True,
        max_iterations=100, verbose=False
    )
    
    print(f"✓ Pretrained: cost={results['pretrained']['best_valid_cost']}, "
          f"validity={results['pretrained']['best_validity']:.2%}, "
          f"time={results['pretrained']['runtime']:.1f}s")
    
    # Approach 3: Gradient pretraining + 2Q gates
    print("\n" + "-"*80)
    print("Approach 3: Gradient Pretraining + 2Q Gates")
    print("-"*80)
    
    gammas_2q, betas_2q, _ = pretrain_validity_pennylane(
        graph, qubit_to_edge_map,
        num_layers=1, max_iterations=20,
        use_gpu=True, use_local_2q_gates=True,
        verbose=False
    )
    
    params_2q = create_pretrained_initial_params_pennylane(
        gammas_2q, betas_2q, total_layers=3
    )
    
    results['pretrained_2q'] = QAOA_pennylane(
        graph, qubit_to_edge_map,
        params_2q,
        layers=3, shots=5000,
        use_gpu=True, use_local_2q_gates=True,
        use_soft_validity=True,
        max_iterations=100, verbose=False
    )
    
    print(f"✓ Pretrained+2Q: cost={results['pretrained_2q']['best_valid_cost']}, "
          f"validity={results['pretrained_2q']['best_validity']:.2%}, "
          f"time={results['pretrained_2q']['runtime']:.1f}s")
    
    # Summary comparison
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Approach':<25} {'Cost':>10} {'Validity':>10} {'Time':>10}")
    print("-"*80)
    
    for name, result in results.items():
        cost = result['best_valid_cost'] if result['best_valid_cost'] else float('inf')
        print(f"{name:<25} {cost:>10.2f} {result['best_validity']:>9.1%} "
              f"{result['runtime']:>9.1f}s")
    
    # Find winner
    best_approach = min(results.items(), 
                       key=lambda x: x[1]['best_valid_cost'] if x[1]['best_valid_cost'] else float('inf'))
    
    print("\n" + "="*80)
    print(f"🏆 Winner: {best_approach[0]}")
    print(f"   Cost: {best_approach[1]['best_valid_cost']}")
    print(f"   Validity: {best_approach[1]['best_validity']:.2%}")
    print("="*80)
    
    return results


def example_larger_problem():
    """
    Example 3: Larger problem demonstrating GPU advantage.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Larger Problem (GPU Advantage)")
    print("="*80)
    
    # Create larger graph
    graph = create_test_graph(num_nodes=8)  # 8 nodes = 56 edges = 56 qubits
    qubit_to_edge_map = create_qubit_to_edge_map(graph)
    
    print(f"\nProblem size: {graph.number_of_nodes()} nodes")
    print(f"Qubits: {len(qubit_to_edge_map)}")
    print(f"State space: 2^{len(qubit_to_edge_map)} = {2**len(qubit_to_edge_map):,}")
    
    # With GPU, we can handle the full circuit!
    print("\n💡 GPU enables full circuit simulation (no batching needed!)")
    
    # Pretrain with gradient descent
    print("\n" + "-"*80)
    print("Gradient Pretraining (takes advantage of full circuit)")
    print("-"*80)
    
    start = time.time()
    gammas, betas, validity = pretrain_validity_pennylane(
        graph, qubit_to_edge_map,
        num_layers=1,
        shots=2048,
        max_iterations=30,
        learning_rate=0.05,
        use_gpu=True,
        use_local_2q_gates=True,  # Can use full entanglement!
        batch_size=None,          # No batching!
        verbose=True
    )
    pretrain_time = time.time() - start
    
    print(f"\n✓ Pretraining: {validity:.2%} validity in {pretrain_time:.1f}s")
    
    # Main optimization
    params_init = create_pretrained_initial_params_pennylane(
        gammas, betas, total_layers=2  # Use 2 layers for larger problem
    )
    
    result = QAOA_pennylane(
        graph, qubit_to_edge_map,
        params_init,
        layers=2,
        shots=5000,
        use_gpu=True,
        use_local_2q_gates=True,
        use_soft_validity=True,
        max_iterations=100,
        verbose=True
    )
    
    print("\n" + "="*80)
    print("FINAL RESULTS (Larger Problem)")
    print("="*80)
    print(f"Best cost: {result['best_valid_cost']}")
    print(f"Validity: {result['best_validity']:.2%}")
    print(f"Total time: {pretrain_time + result['runtime']:.1f}s")
    print(f"  Pretraining: {pretrain_time:.1f}s")
    print(f"  Main QAOA: {result['runtime']:.1f}s")
    
    return result


def example_hyperparameter_tuning():
    """
    Example 4: Tuning gradient pretraining hyperparameters.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Hyperparameter Tuning for Gradient Pretraining")
    print("="*80)
    
    graph = create_test_graph(num_nodes=5)
    qubit_to_edge_map = create_qubit_to_edge_map(graph)
    
    # Test different learning rates
    learning_rates = [0.01, 0.05, 0.1]
    results = {}
    
    print("\nTesting different learning rates...")
    print("-"*80)
    
    for lr in learning_rates:
        print(f"\nLearning rate: {lr}")
        gammas, betas, validity = pretrain_validity_pennylane(
            graph, qubit_to_edge_map,
            num_layers=1,
            shots=1024,
            max_iterations=20,
            learning_rate=lr,
            use_gpu=True,
            verbose=False
        )
        results[f'lr_{lr}'] = validity
        print(f"  Validity achieved: {validity:.2%}")
    
    # Find best learning rate
    best_lr = max(results.items(), key=lambda x: x[1])
    
    print("\n" + "="*80)
    print("LEARNING RATE COMPARISON")
    print("="*80)
    for name, validity in results.items():
        lr = name.replace('lr_', '')
        marker = " ← BEST" if name == best_lr[0] else ""
        print(f"LR {lr:>5s}: {validity:>6.2%}{marker}")
    
    print("\n💡 Recommendation: Use learning_rate=" + best_lr[0].replace('lr_', ''))
    
    return results


def main():
    """
    Run all examples.
    """
    print("\n" + "="*80)
    print("PennyLane QAOA Examples")
    print("Gradient-Based Pretraining + COBYLA Optimization")
    print("="*80)
    print("\nThese examples demonstrate:")
    print("  1. Basic usage with gradient pretraining")
    print("  2. Comparing different approaches")
    print("  3. Larger problems (GPU advantage)")
    print("  4. Hyperparameter tuning")
    print("\nNote: GPU acceleration requires 'pip install pennylane-lightning-gpu'")
    print("="*80)
    
    # Run examples
    try:
        # Example 1: Basic usage
        result1 = example_basic_usage()
        
        # Example 2: Compare approaches
        result2 = example_compare_approaches()
        
        # Example 3: Larger problem
        # Uncomment to run (takes longer)
        # result3 = example_larger_problem()
        
        # Example 4: Hyperparameter tuning
        result4 = example_hyperparameter_tuning()
        
        print("\n" + "="*80)
        print("✓ All examples completed successfully!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Install PennyLane: pip install pennylane pennylane-lightning-gpu")
        print("  2. Check GPU drivers are installed")
        print("  3. Try use_gpu=False to test on CPU first")
        raise


if __name__ == "__main__":
    main()
