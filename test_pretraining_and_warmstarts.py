# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 2026

@author: joeco

Unit tests for quantum pretraining and warm-start methods.
"""

import networkx as nx
import numpy as np
from opt_helpers import get_warm_start_tour
from quantum_helpers import create_qubit_to_edge_map, is_valid_tsp_tour, tour_to_bitstring
from quantum_pretraining import (
    pretrain_validity_layer,
    pretrain_multiple_layers,
    create_pretrained_initial_params,
    pretrain_and_create_initial_params
)


def create_test_graph(n_nodes=5):
    """Create a small test graph for quick testing."""
    # Create random distance matrix
    np.random.seed(42)
    distances = np.random.randint(10, 100, size=(n_nodes, n_nodes))
    
    # Make diagonal zero (no self-loops)
    np.fill_diagonal(distances, 0)
    
    # Create directed graph
    G = nx.from_numpy_array(distances, create_using=nx.DiGraph)
    
    return G


def test_warm_start_methods():
    """Test all warm-start methods produce valid tours."""
    print("\n" + "="*70)
    print("TEST 1: Warm-Start Methods Validity")
    print("="*70)
    
    graph = create_test_graph(n_nodes=6)
    qubit_to_edge_map = create_qubit_to_edge_map(graph)
    
    methods = ['nearest_neighbor', 'farthest_insertion', 'cheapest_insertion',
               'random_nearest_neighbor', 'random']
    
    all_passed = True
    
    for method in methods:
        print(f"\nTesting {method}...")
        
        # Generate tour
        tour = get_warm_start_tour(graph, method=method, seed=42)
        
        # Convert to bitstring
        bitstring = tour_to_bitstring(tour, qubit_to_edge_map)
        
        # Check validity
        is_valid = is_valid_tsp_tour(bitstring, qubit_to_edge_map, graph)
        
        # Calculate cost
        cost = sum(graph[tour[i]][tour[i+1]]['weight'] 
                  for i in range(len(tour)-1))
        
        status = "✓ PASS" if is_valid else "✗ FAIL"
        print(f"  {status}: Valid={is_valid}, Cost={cost:.2f}")
        
        if not is_valid:
            print(f"  ERROR: Generated invalid tour: {tour}")
            all_passed = False
    
    print("\n" + "-"*70)
    if all_passed:
        print("✓ All warm-start methods passed!")
    else:
        print("✗ Some warm-start methods failed!")
    print("-"*70)
    
    return all_passed


def test_warm_start_reproducibility():
    """Test that warm-start methods are reproducible with seeds."""
    print("\n" + "="*70)
    print("TEST 2: Warm-Start Reproducibility")
    print("="*70)
    
    graph = create_test_graph(n_nodes=6)
    
    # Test with random methods
    methods = ['random', 'random_nearest_neighbor']
    
    all_passed = True
    
    for method in methods:
        print(f"\nTesting {method} reproducibility...")
        
        # Generate same tour twice with same seed
        tour1 = get_warm_start_tour(graph, method=method, seed=42)
        tour2 = get_warm_start_tour(graph, method=method, seed=42)
        
        # Generate different tour with different seed
        tour3 = get_warm_start_tour(graph, method=method, seed=123)
        
        same_seed_match = tour1 == tour2
        diff_seed_match = tour1 == tour3
        
        status = "✓ PASS" if same_seed_match and not diff_seed_match else "✗ FAIL"
        print(f"  {status}: Same seed match={same_seed_match}, "
              f"Different seed differs={not diff_seed_match}")
        
        if not (same_seed_match and not diff_seed_match):
            all_passed = False
    
    print("\n" + "-"*70)
    if all_passed:
        print("✓ Reproducibility test passed!")
    else:
        print("✗ Reproducibility test failed!")
    print("-"*70)
    
    return all_passed


def test_warm_start_quality_comparison():
    """Compare quality of different warm-start methods."""
    print("\n" + "="*70)
    print("TEST 3: Warm-Start Quality Comparison")
    print("="*70)
    
    graph = create_test_graph(n_nodes=8)
    
    methods = ['nearest_neighbor', 'farthest_insertion', 'cheapest_insertion',
               'random_nearest_neighbor']
    
    results = []
    
    for method in methods:
        tour = get_warm_start_tour(graph, method=method, seed=42)
        cost = sum(graph[tour[i]][tour[i+1]]['weight'] 
                  for i in range(len(tour)-1))
        results.append((method, cost))
    
    results.sort(key=lambda x: x[1])
    
    print("\nWarm-start quality ranking (lower cost = better):")
    print(f"{'Rank':<6} {'Method':<30} {'Cost':<10}")
    print("-"*50)
    for rank, (method, cost) in enumerate(results, 1):
        print(f"{rank:<6} {method:<30} {cost:<10.2f}")
    
    print("\n✓ Quality comparison complete!")
    
    return True


def test_pretraining_basic():
    """Test basic pretraining functionality."""
    print("\n" + "="*70)
    print("TEST 4: Basic Pretraining (Single Layer)")
    print("="*70)
    
    graph = create_test_graph(n_nodes=5)  # Small for speed
    qubit_to_edge_map = create_qubit_to_edge_map(graph)
    
    print("\nRunning pretraining on layer 0...")
    print("(This may take 30-60 seconds)")
    
    try:
        gamma, beta, validity = pretrain_validity_layer(
            graph, qubit_to_edge_map,
            layer_idx=0,
            shots=512,  # Reduced for faster testing
            batch_size=8,
            max_iterations=10,  # Reduced for faster testing
            verbose=False
        )
        
        print("\nResults:")
        print(f"  Gamma: {gamma}")
        print(f"  Beta: {beta}")
        print(f"  Validity rate: {validity:.2%}")
        
        # Check that we got some parameters
        params_exist = len(gamma) > 0 and len(beta) > 0
        validity_positive = validity > 0
        
        status = "✓ PASS" if params_exist and validity_positive else "✗ FAIL"
        print(f"\n{status}: Parameters generated and validity > 0")
        
        return params_exist and validity_positive
        
    except Exception as e:
        print(f"\n✗ FAIL: Pretraining raised exception: {e}")
        return False


def test_pretraining_multiple_layers():
    """Test pretraining multiple layers."""
    print("\n" + "="*70)
    print("TEST 5: Multiple Layer Pretraining")
    print("="*70)
    
    graph = create_test_graph(n_nodes=5)
    qubit_to_edge_map = create_qubit_to_edge_map(graph)
    
    print("\nPretraining 2 layers...")
    print("(This may take 1-2 minutes)")
    
    try:
        gammas, betas, validities = pretrain_multiple_layers(
            graph, qubit_to_edge_map,
            num_layers_to_pretrain=2,
            shots=512,
            batch_size=8,
            max_iterations=5,  # Very limited for testing
            verbose=False
        )
        
        print("\nResults:")
        print(f"  Gammas: {gammas}")
        print(f"  Betas: {betas}")
        print(f"  Validity rates: {[f'{v:.2%}' for v in validities]}")
        
        correct_length = len(gammas) == 2 and len(betas) == 2
        all_positive = all(v > 0 for v in validities)
        
        status = "✓ PASS" if correct_length and all_positive else "✗ FAIL"
        print(f"\n{status}: Correct number of layers and all validities > 0")
        
        return correct_length and all_positive
        
    except Exception as e:
        print(f"\n✗ FAIL: Multiple layer pretraining raised exception: {e}")
        return False


def test_create_pretrained_params():
    """Test parameter creation for full QAOA."""
    print("\n" + "="*70)
    print("TEST 6: Create Full Parameter Set")
    print("="*70)
    
    # Mock pretrained values
    pretrained_gammas = [0.1, 0.2]
    pretrained_betas = [0.3, 0.4]
    total_layers = 5
    
    print(f"\nPretrained layers: {len(pretrained_gammas)}")
    print(f"Total layers needed: {total_layers}")
    
    # Test different strategies
    strategies = ['extend_with_zeros', 'extend_with_small_random', 'extend_with_linear']
    
    all_passed = True
    
    for strategy in strategies:
        print(f"\nTesting strategy: {strategy}")
        
        params = create_pretrained_initial_params(
            pretrained_gammas, pretrained_betas,
            total_layers, strategy=strategy
        )
        
        # Check length
        expected_length = 2 * total_layers  # gamma and beta for each layer
        correct_length = len(params) == expected_length
        
        # Check that pretrained values are preserved
        gammas = params[:total_layers]
        betas = params[total_layers:]
        
        pretrained_preserved = (
            gammas[0] == pretrained_gammas[0] and
            gammas[1] == pretrained_gammas[1] and
            betas[0] == pretrained_betas[0] and
            betas[1] == pretrained_betas[1]
        )
        
        status = "✓ PASS" if correct_length and pretrained_preserved else "✗ FAIL"
        print(f"  {status}: Length={len(params)} (expected {expected_length}), "
              f"Pretrained preserved={pretrained_preserved}")
        
        if not (correct_length and pretrained_preserved):
            all_passed = False
    
    return all_passed


def test_integration():
    """Test complete workflow: pretraining + QAOA initialization."""
    print("\n" + "="*70)
    print("TEST 7: Integration Test")
    print("="*70)
    
    graph = create_test_graph(n_nodes=5)
    
    print("\nRunning complete pretraining workflow...")
    print("(This may take 1 minute)")
    
    try:
        params, validities = pretrain_and_create_initial_params(
            graph,
            num_pretrain_layers=1,
            total_layers=3,
            shots=512,
            max_iterations=5,
            verbose=False
        )
        
        print("\nResults:")
        print(f"  Parameter length: {len(params)} (expected 6)")
        print(f"  Validity rates: {[f'{v:.2%}' for v in validities]}")
        
        # Check that we can use these in QAOA_approx
        correct_length = len(params) == 6  # 3 layers * 2 params
        has_validities = len(validities) > 0
        
        status = "✓ PASS" if correct_length and has_validities else "✗ FAIL"
        print(f"\n{status}: Ready for QAOA_approx")
        
        if correct_length and has_validities:
            print("\nExample usage:")
            print("  graphs_dict, runtime_data, tt_data, qaoa_progress = QAOA_approx(")
            print("      graph, ..., layers=3,")
            print("      custom_initial_params=params")
            print("  )")
        
        return correct_length and has_validities
        
    except Exception as e:
        print(f"\n✗ FAIL: Integration test raised exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "#"*70)
    print("# QUANTUM PRETRAINING AND WARM-START UNIT TESTS")
    print("#"*70)
    
    tests = [
        ("Warm-Start Validity", test_warm_start_methods),
        ("Warm-Start Reproducibility", test_warm_start_reproducibility),
        ("Warm-Start Quality", test_warm_start_quality_comparison),
        ("Basic Pretraining", test_pretraining_basic),
        ("Multiple Layer Pretraining", test_pretraining_multiple_layers),
        ("Parameter Creation", test_create_pretrained_params),
        ("Integration Test", test_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n✗ {test_name} CRASHED: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n\n" + "#"*70)
    print("# TEST SUMMARY")
    print("#"*70)
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    for test_name, passed_flag in results:
        status = "✓ PASS" if passed_flag else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System ready for use.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review output above.")
    
    print("#"*70)


if __name__ == "__main__":
    run_all_tests()
