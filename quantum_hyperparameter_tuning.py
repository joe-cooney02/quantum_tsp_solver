#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 17:32:21 2026

@author: joecooney

Orchestration and visualization for gate-order hyperparameter tuning.

Background
----------
quantum_pretraining.pretrain_validity_diversity_multi_order pretrains a QAOA
circuit (via quantum_helpers.create_multi_order_qaoa_circuit) to maximize
validity + diversity for ONE gate-order configuration (e.g. gate_orders=[4]
means "use only 4-qubit entangling gates"). This module orchestrates running
that function across MANY gate-order configurations and visualizing how
validity, diversity, and parameter/gate count trade off against each other.

Parallelization
----------------
run_single_hyperparameter_trial is the unit of parallelizable work: it is
fully self-contained given (graph, gate_orders, ...), so you can dispatch it
yourself across processes (multiprocessing.Pool, joblib, etc.) using your own
function to generate the list of gate_orders configurations to test. The
sequential run_hyperparameter_sweep wrapper is provided for convenience when
parallelism isn't needed, or for aggregating results that were computed
elsewhere (see aggregate_external_trial_results).
"""

import time
import matplotlib.pyplot as plt

from quantum_pretraining import pretrain_validity_diversity_multi_order
from quantum_helpers import (
    create_qubit_to_edge_map,
    create_multi_order_qaoa_circuit,
    bind_multi_order_parameters,
    simulate_large_circuit_in_batches,
    filter_valid_shots,
)
from visualization_algorithms import (
    plot_qaoa_comprehensive_progress,
    plot_qaoa_comparison,
    plot_qaoa_final_comparison_bars,
    plot_valid_solution_hamming_distances,
)


# =============================================================================
# Orchestration
# =============================================================================

def run_single_hyperparameter_trial(graph, gate_orders, num_layers=1, shots=1024,
                                     batch_size=8, sim_method='statevector',
                                     max_iterations=100, diversity_weight=0.3,
                                     diversity_method='entropy', device='CPU',
                                     verbose=True, label=None):
    """
    Run one gate-order hyperparameter-tuning trial.

    This is the parallelizable unit of work. Call this once per gate_orders
    configuration -- e.g. from your own worker-dispatch function -- since
    each call is independent and stateless aside from reading `graph`.

    Parameters
    ----------
    graph : networkx.DiGraph
        The TSP graph.
    gate_orders : list of int
        Entangling-gate orders to test in this trial, each in
        [1, num_qubits]. E.g. [3] tests only 3-qubit gates; [1, 6] tests a
        combination of single-qubit and 6-qubit gates.
    num_layers : int, optional
        Number of QAOA layers to pretrain together.
    shots : int, optional
    batch_size : int, optional
        Must be >= max(gate_orders).
    sim_method : str, optional
    max_iterations : int, optional
        COBYLA iteration budget.
    diversity_weight : float, optional
        Weight of the diversity term in the composite objective.
    diversity_method : str, optional
        'entropy' or 'unique_fraction'.
    device : str, optional
        'CPU' or 'GPU'.
    verbose : bool, optional
    label : str, optional
        Identifier for this trial; auto-generated from gate_orders/num_layers
        if not given.

    Returns
    -------
    dict: Trial result (see pretrain_validity_diversity_multi_order), with an
        added 'runtime_seconds' key.
    """
    start_time = time.time()

    result = pretrain_validity_diversity_multi_order(
        graph, gate_orders, num_layers=num_layers, shots=shots,
        batch_size=batch_size, sim_method=sim_method,
        max_iterations=max_iterations, verbose=verbose,
        diversity_weight=diversity_weight, diversity_method=diversity_method,
        device=device, label=label
    )

    result['runtime_seconds'] = time.time() - start_time
    return result


def run_hyperparameter_sweep(graph, gate_orders_list, num_layers=1, shots=1024,
                              batch_size=8, sim_method='statevector',
                              max_iterations=100, diversity_weight=0.3,
                              diversity_method='entropy', device='CPU',
                              verbose=True):
    """
    Sequentially run a hyperparameter sweep across multiple gate-order
    configurations.

    For parallel execution, don't use this function -- instead call
    run_single_hyperparameter_trial directly from your own dispatch logic
    (e.g. multiprocessing.Pool.map, joblib.Parallel) for each entry in
    gate_orders_list, then pass the collected results to
    aggregate_external_trial_results for visualization.

    Parameters
    ----------
    graph : networkx.DiGraph
    gate_orders_list : list of list of int
        Each inner list is one gate_orders configuration to test. Typically
        produced by your own generator function so the sweep can be
        distributed across workers. See example_single_order_sweep_configs
        for a simple example generator.
    num_layers, shots, batch_size, sim_method, max_iterations,
    diversity_weight, diversity_method, device, verbose :
        Common QAOA / pretraining hyperparameters applied to every trial in
        the sweep (see run_single_hyperparameter_trial).

    Returns
    -------
    dict: {trial_label: trial_result_dict, ...}, in the same insertion order
        as gate_orders_list.
    """
    sweep_results = {}

    for i, gate_orders in enumerate(gate_orders_list):
        sorted_orders = sorted(set(int(k) for k in gate_orders))
        label = f"orders_{'-'.join(map(str, sorted_orders))}_L{num_layers}"

        if verbose:
            print(f"\n[{i + 1}/{len(gate_orders_list)}] Running trial: {label}")

        trial_result = run_single_hyperparameter_trial(
            graph, gate_orders, num_layers=num_layers, shots=shots,
            batch_size=batch_size, sim_method=sim_method,
            max_iterations=max_iterations, diversity_weight=diversity_weight,
            diversity_method=diversity_method, device=device, verbose=verbose,
            label=label
        )
        sweep_results[label] = trial_result

    return sweep_results


def aggregate_external_trial_results(trial_results):
    """
    Build a sweep_results dict (label -> trial_result) from a list of
    trial_result dicts that were computed elsewhere (e.g. returned from
    parallel workers running run_single_hyperparameter_trial). Use this to
    feed externally-parallelized results into the visualization functions
    below.

    Parameters
    ----------
    trial_results : list of dict
        Each element should be a dict returned by
        run_single_hyperparameter_trial / pretrain_validity_diversity_multi_order.

    Returns
    -------
    dict: {trial_result['label']: trial_result, ...}
    """
    return {result['label']: result for result in trial_results}


def example_single_order_sweep_configs(num_qubits):
    """
    Convenience example generator for gate_orders_list: tests one gate order
    at a time, from 1 through num_qubits. This is a simple default -- write
    your own generator (e.g. to test combinations of orders, or to chunk
    work across parallel workers) and pass its output directly to
    run_hyperparameter_sweep or to your own parallel dispatcher.

    Parameters
    ----------
    num_qubits : int

    Returns
    -------
    list of list of int: [[1], [2], ..., [num_qubits]]
    """
    return [[k] for k in range(1, num_qubits + 1)]


def get_valid_tours_from_trial(graph, trial_result, shots=2048,
                                sim_method='statevector', batch_size=None,
                                device='CPU'):
    """
    Re-simulate a trial's circuit at its final (optimized) parameters and
    extract the resulting valid tours. Useful for feeding into
    plot_valid_solution_hamming_distances / plot_hamming_distance_histogram
    to inspect the diversity of a specific trial's solutions in more detail
    than the scalar diversity_score captures.

    Parameters
    ----------
    graph : networkx.DiGraph
    trial_result : dict
        A trial result from run_single_hyperparameter_trial /
        pretrain_validity_diversity_multi_order.
    shots : int, optional
        Shots to use for this re-simulation (can differ from the shots used
        during optimization, e.g. to get a larger, more representative
        sample once parameters are fixed).
    sim_method : str, optional
    batch_size : int, optional
        Defaults to max(gate_orders) from the trial if not given (the
        minimum valid batch size for that configuration).
    device : str, optional

    Returns
    -------
    tuple: (valid_tours, valid_bitstrings)
    """
    gate_orders = trial_result['gate_orders']
    num_layers = trial_result['num_layers']

    if batch_size is None:
        batch_size = max(gate_orders)

    qubit_to_edge_map = create_qubit_to_edge_map(graph)

    circuit = create_multi_order_qaoa_circuit(
        graph, qubit_to_edge_map, gate_orders, num_layers=num_layers,
        batch_size=batch_size
    )

    bound_circuit = bind_multi_order_parameters(
        circuit,
        trial_result['gamma_values'],
        trial_result['beta_values'],
        trial_result['theta_values_by_order'],
    )

    counts = simulate_large_circuit_in_batches(
        bound_circuit, batch_size, shots, sim_method, device=device
    )

    valid_bitstrings, valid_tours = filter_valid_shots(
        list(counts.keys()), qubit_to_edge_map, graph, return_tours=True
    )

    return valid_tours, valid_bitstrings


# =============================================================================
# Visualization
# =============================================================================

def _is_single_order_sweep(sweep_results):
    """Check whether every trial in the sweep tested exactly one gate order."""
    return all(len(result['gate_orders']) == 1 for result in sweep_results.values())


def plot_validity_diversity_vs_order(sweep_results, figsize=(12, 5)):
    """
    For a single-order sweep (each trial tests exactly one gate order k),
    plot best validity rate and best diversity score as a function of k.

    Falls back to a label-indexed bar version (via plot_qaoa_final_comparison_bars,
    reused from visualization_algorithms) if the sweep mixes multiple orders
    per trial, since "gate order" isn't a single well-defined x-axis value
    in that case.

    Parameters
    ----------
    sweep_results : dict
        {label: trial_result, ...} as returned by run_hyperparameter_sweep
        or aggregate_external_trial_results.
    figsize : tuple, optional

    Returns
    -------
    fig, axes : matplotlib figure and axes objects
    """
    if not sweep_results:
        print("No sweep results to plot!")
        return None, None

    if not _is_single_order_sweep(sweep_results):
        print("Sweep mixes multiple gate orders per trial; falling back to "
              "label-indexed bar chart (plot_qaoa_final_comparison_bars).")
        labelled_stats = {label: r['stats_history'] for label, r in sweep_results.items()}
        return plot_qaoa_final_comparison_bars(labelled_stats, figsize=figsize)

    # Sort trials by their (single) gate order
    ordered = sorted(sweep_results.values(), key=lambda r: r['gate_orders'][0])
    orders = [r['gate_orders'][0] for r in ordered]
    validities = [100 * r['best_validity_rate'] for r in ordered]
    diversities = [r['best_diversity_score'] for r in ordered]
    num_params = [r['num_parameters'] for r in ordered]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.plot(orders, validities, 'o-', linewidth=2.5, markersize=8, color='#2ca02c')
    ax1.set_xlabel('Gate Order (k-qubit entangler)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Best Validity Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Validity vs. Gate Order', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(orders)

    ax2.plot(orders, diversities, 's-', linewidth=2.5, markersize=8, color='#ff7f0e')
    ax2.set_xlabel('Gate Order (k-qubit entangler)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Best Diversity Score', fontsize=12, fontweight='bold')
    ax2.set_title('Diversity vs. Gate Order', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(orders)
    ax2.set_ylim(0, 1.05)

    # Annotate parameter count on the validity panel as a secondary reference
    ax1_twin = ax1.twinx()
    ax1_twin.plot(orders, num_params, '^--', color='gray', alpha=0.5, markersize=6,
                  label='# Parameters')
    ax1_twin.set_ylabel('# Parameters', fontsize=10, color='gray')
    ax1_twin.tick_params(axis='y', labelcolor='gray')
    ax1_twin.legend(loc='lower right', fontsize=9)

    plt.suptitle('Gate-Order Hyperparameter Sweep: Validity & Diversity',
                  fontsize=15, fontweight='bold')
    plt.tight_layout()

    return fig, (ax1, ax2)


def plot_validity_vs_parameter_count(sweep_results, figsize=(9, 7)):
    """
    Scatter plot of best validity rate vs. number of tunable parameters
    across all trials in a sweep, colored by diversity score. Directly
    visualizes whether adding parameters (higher gate orders, or more
    orders combined per layer) is helping or hurting at a fixed shot budget
    -- the practical question motivating this sweep.

    Parameters
    ----------
    sweep_results : dict
        {label: trial_result, ...}
    figsize : tuple, optional

    Returns
    -------
    fig, ax : matplotlib figure and axes objects
    """
    if not sweep_results:
        print("No sweep results to plot!")
        return None, None

    labels = list(sweep_results.keys())
    num_params = [sweep_results[l]['num_parameters'] for l in labels]
    validities = [100 * sweep_results[l]['best_validity_rate'] for l in labels]
    diversities = [sweep_results[l]['best_diversity_score'] for l in labels]
    gate_counts = [sweep_results[l]['gate_count_estimate']['total'] for l in labels]

    fig, ax = plt.subplots(figsize=figsize)

    # Marker size reflects estimated entangling-gate count (circuit complexity)
    max_gc = max(gate_counts) if max(gate_counts) > 0 else 1
    sizes = [80 + 320 * (gc / max_gc) for gc in gate_counts]

    scatter = ax.scatter(num_params, validities, c=diversities, s=sizes,
                          cmap='viridis', edgecolor='black', linewidth=1.2,
                          vmin=0, vmax=1, alpha=0.85)

    for x, y, label in zip(num_params, validities, labels):
        short_label = label.replace('orders_', '').replace(f"_L{sweep_results[label]['num_layers']}", '')
        ax.annotate(short_label, (x, y), fontsize=8, xytext=(5, 5),
                    textcoords='offset points')

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Diversity Score', rotation=270, labelpad=20, fontsize=12)

    ax.set_xlabel('Number of Tunable Parameters', fontsize=12, fontweight='bold')
    ax.set_ylabel('Best Validity Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Validity vs. Parameter Count\n(marker size = estimated entangling-gate count)',
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax


def plot_sweep_progress(sweep_results, figsize=(14, 10)):
    """
    Reuse plot_qaoa_comprehensive_progress (from visualization_algorithms) on
    every trial's iteration history, giving per-trial convergence panels
    (best cost, validity %, diversity/unique-bitstring count, valid vs.
    invalid shots) exactly as used for the main QAOA runs in main.py.

    Parameters
    ----------
    sweep_results : dict
        {label: trial_result, ...}
    figsize : tuple, optional

    Returns
    -------
    figs, axes : as returned by plot_qaoa_comprehensive_progress (single
        figure/axes if one trial, else lists)
    """
    labelled_stats = {label: r['stats_history'] for label, r in sweep_results.items()}
    return plot_qaoa_comprehensive_progress(labelled_stats, figsize=figsize)


def plot_sweep_comparison(sweep_results, figsize=(16, 10)):
    """
    Reuse plot_qaoa_comparison (from visualization_algorithms) for a
    side-by-side comparison of every trial's convergence in the sweep.
    """
    labelled_stats = {label: r['stats_history'] for label, r in sweep_results.items()}
    return plot_qaoa_comparison(labelled_stats, figsize=figsize)


def plot_sweep_final_bars(sweep_results, figsize=(14, 5)):
    """
    Reuse plot_qaoa_final_comparison_bars (from visualization_algorithms) to
    show final validity rate, cost, and diversity bars across all trials.
    Note: the "cost" panel reflects the underlying graph weights even though
    this pretraining stage doesn't directly optimize for cost -- it's
    informational only, since validity is graph-weight-independent.
    """
    labelled_stats = {label: r['stats_history'] for label, r in sweep_results.items()}
    return plot_qaoa_final_comparison_bars(labelled_stats, figsize=figsize)


def plot_best_trial_solution_diversity(graph, sweep_results, metric='best_composite_score',
                                       shots=2048, sim_method='statevector', device='CPU',
                                       max_solutions=50, figsize=(10, 8)):
    """
    Identify the best trial in a sweep (by the given metric), re-simulate it
    to collect a fresh sample of valid tours, and visualize the Hamming
    distances between those solutions using
    plot_valid_solution_hamming_distances (reused from
    visualization_algorithms).

    Parameters
    ----------
    graph : networkx.DiGraph
    sweep_results : dict
        {label: trial_result, ...}
    metric : str, optional
        Key in trial_result to rank trials by (default: 'best_composite_score').
        Other useful options: 'best_validity_rate', 'best_diversity_score'.
    shots, sim_method, device : passed to get_valid_tours_from_trial.
    max_solutions : int, optional
        Passed to plot_valid_solution_hamming_distances.
    figsize : tuple, optional

    Returns
    -------
    tuple: (best_label, fig, ax)
    """
    if not sweep_results:
        print("No sweep results to plot!")
        return None, None, None

    best_label = max(sweep_results, key=lambda l: sweep_results[l][metric])
    best_trial = sweep_results[best_label]

    print(f"Best trial by {metric}: {best_label} "
          f"({metric}={best_trial[metric]:.3f}, gate_orders={best_trial['gate_orders']})")

    valid_tours, _ = get_valid_tours_from_trial(
        graph, best_trial, shots=shots, sim_method=sim_method, device=device
    )

    if len(valid_tours) < 2:
        print(f"Only {len(valid_tours)} valid tour(s) found for {best_label}; "
              f"need at least 2 to compute Hamming distances.")
        return best_label, None, None

    qubit_to_edge_map = create_qubit_to_edge_map(graph)
    fig, ax = plot_valid_solution_hamming_distances(
        valid_tours, qubit_to_edge_map, G=graph,
        max_solutions=max_solutions, figsize=figsize
    )

    return best_label, fig, ax