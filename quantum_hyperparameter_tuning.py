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
import numpy as np
import os
import json
import datetime
import csv
import warnings

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

# =============================================================================
# Data saving, loading, and aggregation
#
# Design notes
# ------------
# JSON is the storage format: human-readable, git-diffable, and requires no
# extra dependencies. Two serialization problems need handling:
#
#   1. NumPy types (np.float64, np.ndarray, np.int64) are not JSON-native.
#      _to_python() recursively converts them to Python scalars/lists.
#
#   2. JSON requires all dict keys to be strings. Two dicts in trial results
#      have integer keys: theta_values_by_order ({2: [...]}) and
#      gate_count_estimate ({2: 15, 'total': 15}). These are serialized with
#      string keys and restored to integers on load via _restore_numeric_keys().
#
#   3. scipy.optimize.OptimizeResult is not serializable. Only the fields
#      needed for downstream analysis are extracted and stored.
#
# File layout produced by save_sweep_results:
#   save_dir/
#     {sweep_name}_index.json    -- summary scalars for every trial (fast to load)
#     {sweep_name}_summary.csv   -- same content, spreadsheet-friendly
#     trials/
#       {label}.json             -- full trial data including stats_history
# =============================================================================
 
# --- Serialization helpers ---------------------------------------------------
 
def _to_python(obj):
    """
    Recursively convert an object tree containing NumPy scalars, arrays,
    scipy OptimizeResult objects, and networkx graphs into plain Python types
    suitable for json.dump.
 
    scipy OptimizeResult is reduced to a plain dict of the fields useful for
    downstream analysis: x (optimal parameters), fun (final objective value),
    success, message, and nfev (number of function evaluations).
 
    networkx DiGraph / Graph (which can appear as best_tour in stats returned
    by get_qaoa_statistics) is serialized as a list of (u, v, weight) triples
    so the tour structure is fully recoverable.
    """
    # networkx graphs -- check before dict since Graph is dict-like internally
    try:
        import networkx as nx
        if isinstance(obj, (nx.DiGraph, nx.Graph)):
            return {
                '__type__': 'DiGraph',
                'edges': [[u, v, d] for u, v, d in obj.edges(data=True)],
                'nodes': list(obj.nodes()),
            }
    except ImportError:
        pass
 
    # scipy OptimizeResult
    try:
        from scipy.optimize import OptimizeResult
        if isinstance(obj, OptimizeResult):
            return {
                '__type__': 'OptimizeResult',
                'x': _to_python(list(obj.x)),
                'fun': float(obj.fun),
                'success': bool(obj.success),
                'message': str(obj.message),
                'nfev': int(obj.nfev),
            }
    except ImportError:
        pass
 
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float64, np.float32, np.floating)):
        return float(obj)
    if isinstance(obj, (np.int64, np.int32, np.integer)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    return obj
 
 
def _restore_numeric_keys(obj):
    """
    Recursively walk a decoded JSON object and convert any dict key that
    looks like an integer back to int. JSON encodes all dict keys as strings;
    this restores the original int-keyed dicts (theta_values_by_order,
    gate_count_estimate). Keys that are not pure digits (e.g. 'total', 'label')
    are left as strings.
    """
    if isinstance(obj, dict):
        restored = {}
        for k, v in obj.items():
            new_key = int(k) if isinstance(k, str) and k.lstrip('-').isdigit() else k
            restored[new_key] = _restore_numeric_keys(v)
        return restored
    if isinstance(obj, list):
        return [_restore_numeric_keys(v) for v in obj]
    return obj
 
 
# --- Single trial I/O --------------------------------------------------------
 
def save_trial_result(trial_result, save_dir, filename=None, save_history=True):
    """
    Save a single trial result dict to a JSON file.
 
    Parameters
    ----------
    trial_result : dict
        Result from run_single_hyperparameter_trial /
        pretrain_validity_diversity_multi_order.
    save_dir : str
        Directory to write the file into (created if it doesn't exist).
    filename : str or None, optional
        JSON filename (without directory). Defaults to '{label}.json'.
    save_history : bool, optional
        Whether to include stats_history (full per-iteration data) in the
        saved file. Set False to produce smaller files when you only need
        the scalar summary metrics. Default True.
 
    Returns
    -------
    str: Full path to the saved file.
    """
    os.makedirs(save_dir, exist_ok=True)
 
    if filename is None:
        safe_label = trial_result['label'].replace('/', '_').replace('\\', '_')
        filename = f"{safe_label}.json"
 
    filepath = os.path.join(save_dir, filename)
 
    data = _to_python(trial_result)
    if not save_history:
        data.pop('stats_history', None)
        data.pop('best_iteration_stats', None)
 
    data['_saved_at'] = datetime.now().isoformat(timespec='seconds')
 
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
 
    return filepath
 
 
def load_trial_result(filepath):
    """
    Load a trial result from a JSON file saved by save_trial_result.
 
    Parameters
    ----------
    filepath : str
        Path to the JSON file.
 
    Returns
    -------
    dict: Trial result with integer dict keys restored on theta_values_by_order
        and gate_count_estimate. stats_history and best_iteration_stats are
        included only if they were saved (i.e. save_history=True at save time).
 
    Notes
    -----
    optimizer_result is stored as a plain dict (not a scipy OptimizeResult),
    with the same keys: 'x', 'fun', 'success', 'message', 'nfev'.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
 
    return _restore_numeric_keys(data)
 
 
# --- Full sweep I/O ----------------------------------------------------------
 
_TRIAL_SUBDIR = 'trials'
_INDEX_SUFFIX = '_index.json'
_CSV_SUFFIX = '_summary.csv'
 
# Scalar fields written to the index JSON and summary CSV.
# These cover everything needed to compare trials and make plots without
# loading full per-iteration history.
_SUMMARY_FIELDS = [
    'label', 'gate_orders', 'num_layers', 'num_parameters',
    'best_validity_rate', 'best_diversity_score', 'best_composite_score',
    'runtime_seconds',
    ('gate_count_estimate', 'total'),  # nested field: (outer_key, inner_key)
]
 
 
def _extract_summary_row(trial_result):
    """Extract the scalar summary fields from a trial result into a flat dict."""
    row = {}
    for field in _SUMMARY_FIELDS:
        if isinstance(field, tuple):
            outer, inner = field
            col_name = f"{outer}.{inner}"
            row[col_name] = trial_result.get(outer, {}).get(inner)
        else:
            val = trial_result.get(field)
            # gate_orders is a list; represent as a compact string for CSV
            row[field] = str(val) if isinstance(val, list) else val
    return row
 
 
def save_sweep_results(sweep_results, save_dir, sweep_name=None, save_history=True):
    """
    Save a complete sweep (dict of label -> trial_result) to disk.
 
    Produces three artifacts under save_dir:
 
    - trials/{label}.json        Full trial data per trial (one file each).
    - {sweep_name}_index.json    Summary scalars for all trials, fast to load
                                 without reading every trial file.
    - {sweep_name}_summary.csv   Same content as the index, spreadsheet-friendly.
 
    Parameters
    ----------
    sweep_results : dict
        {label: trial_result, ...} as returned by run_hyperparameter_sweep or
        aggregate_external_trial_results.
    save_dir : str
        Root directory for this sweep's outputs (created if needed).
    sweep_name : str or None, optional
        Prefix for the index and CSV files. Defaults to
        'sweep_{YYYYMMDD_HHMMSS}'.
    save_history : bool, optional
        Whether to save per-iteration stats_history in each trial file.
        The index JSON and CSV always contain only scalar summaries regardless
        of this flag.
 
    Returns
    -------
    dict with keys:
        'save_dir', 'sweep_name', 'index_path', 'csv_path',
        'trial_paths' (dict of label -> filepath)
    """
    if sweep_name is None:
        sweep_name = 'sweep_' + datetime.now().strftime('%Y%m%d_%H%M%S')
 
    trials_dir = os.path.join(save_dir, _TRIAL_SUBDIR)
    os.makedirs(trials_dir, exist_ok=True)
 
    trial_paths = {}
    summary_rows = []
 
    for label, trial_result in sweep_results.items():
        path = save_trial_result(trial_result, trials_dir, save_history=save_history)
        trial_paths[label] = path
        summary_rows.append(_extract_summary_row(trial_result))
 
    # Index JSON
    index_path = os.path.join(save_dir, sweep_name + _INDEX_SUFFIX)
    index_data = {
        'sweep_name': sweep_name,
        'saved_at': datetime.now().isoformat(timespec='seconds'),
        'num_trials': len(sweep_results),
        'trial_files': {label: os.path.relpath(path, save_dir)
                        for label, path in trial_paths.items()},
        'summaries': {row['label']: row for row in summary_rows},
    }
    with open(index_path, 'w') as f:
        json.dump(_to_python(index_data), f, indent=2)
 
    # Summary CSV
    csv_path = os.path.join(save_dir, sweep_name + _CSV_SUFFIX)
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(_to_python(summary_rows))
 
    return {
        'save_dir': save_dir,
        'sweep_name': sweep_name,
        'index_path': index_path,
        'csv_path': csv_path,
        'trial_paths': trial_paths,
    }
 
 
def load_sweep_results(save_dir, sweep_name=None, load_history=True):
    """
    Load a complete sweep from disk into a sweep_results dict.
 
    If sweep_name is given, loads the corresponding index file and uses it
    to locate trial files. If sweep_name is None, searches save_dir for an
    index file automatically (raises ValueError if there is more than one).
 
    Falls back to scanning the trials/ subdirectory for any .json files if
    no index file is found.
 
    Parameters
    ----------
    save_dir : str
        Root directory written by save_sweep_results.
    sweep_name : str or None, optional
    load_history : bool, optional
        If False, omit stats_history and best_iteration_stats from each
        loaded trial (they'll simply be absent if not saved, or dropped here
        if they were saved). Useful for quickly loading just summary data.
 
    Returns
    -------
    dict: {label: trial_result, ...}, suitable for passing directly to all
        visualization functions.
 
    Raises
    ------
    FileNotFoundError: if save_dir doesn't exist.
    ValueError: if sweep_name is None and multiple index files are found.
    """
    if not os.path.isdir(save_dir):
        raise FileNotFoundError(f"save_dir not found: {save_dir}")
 
    # Resolve index file
    if sweep_name is not None:
        index_path = os.path.join(save_dir, sweep_name + _INDEX_SUFFIX)
    else:
        candidates = [f for f in os.listdir(save_dir) if f.endswith(_INDEX_SUFFIX)]
        if len(candidates) == 1:
            index_path = os.path.join(save_dir, candidates[0])
        elif len(candidates) > 1:
            raise ValueError(
                f"Multiple index files found in {save_dir}: {candidates}. "
                f"Specify sweep_name to disambiguate."
            )
        else:
            index_path = None
 
    # Collect trial file paths
    if index_path and os.path.isfile(index_path):
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        trial_relpaths = index_data.get('trial_files', {})
        trial_paths = {
            label: os.path.join(save_dir, relpath)
            for label, relpath in trial_relpaths.items()
        }
    else:
        # Fallback: scan trials/ subdirectory
        trials_dir = os.path.join(save_dir, _TRIAL_SUBDIR)
        if not os.path.isdir(trials_dir):
            raise FileNotFoundError(
                f"No index file and no trials/ subdirectory found in {save_dir}."
            )
        trial_paths = {
            os.path.splitext(fname)[0]: os.path.join(trials_dir, fname)
            for fname in os.listdir(trials_dir)
            if fname.endswith('.json')
        }
 
    sweep_results = {}
    for label, path in trial_paths.items():
        if not os.path.isfile(path):
            warnings.warn(f"Trial file not found, skipping: {path}")
            continue
        trial = load_trial_result(path)
        if not load_history:
            trial.pop('stats_history', None)
            trial.pop('best_iteration_stats', None)
        sweep_results[trial.get('label', label)] = trial
 
    return sweep_results
 
 
def load_sweep_index(save_dir, sweep_name=None):
    """
    Load only the lightweight index file from a saved sweep, without reading
    any individual trial files. Returns the summary scalars for all trials,
    suitable for quick inspection or deciding which trials to load in full.
 
    Parameters
    ----------
    save_dir : str
    sweep_name : str or None, optional
 
    Returns
    -------
    dict: The raw index data, including 'summaries' (label -> scalar dict),
        'sweep_name', 'saved_at', 'num_trials', and 'trial_files'.
    """
    if sweep_name is not None:
        index_path = os.path.join(save_dir, sweep_name + _INDEX_SUFFIX)
    else:
        candidates = [f for f in os.listdir(save_dir) if f.endswith(_INDEX_SUFFIX)]
        if len(candidates) == 1:
            index_path = os.path.join(save_dir, candidates[0])
        elif len(candidates) > 1:
            raise ValueError(
                f"Multiple index files in {save_dir}: {candidates}. "
                f"Specify sweep_name."
            )
        else:
            raise FileNotFoundError(f"No index file found in {save_dir}.")
 
    with open(index_path, 'r') as f:
        return json.load(f)
 
 
# --- Aggregation -------------------------------------------------------------
 
def merge_sweep_results(*sweep_dicts, on_duplicate='warn'):
    """
    Merge two or more sweep_results dicts (e.g. results from parallel workers
    that each ran a subset of trials) into one combined dict.
 
    Parameters
    ----------
    *sweep_dicts : dicts
        Any number of {label: trial_result} dicts, in merge-priority order
        (last writer wins on duplicates when on_duplicate='keep_last').
    on_duplicate : str, optional
        What to do when the same label appears in more than one input dict:
        'warn'        (default) Keep the first occurrence, emit a warning.
        'keep_last'   Overwrite with the later occurrence silently.
        'error'       Raise a ValueError.
 
    Returns
    -------
    dict: {label: trial_result, ...}
    """
    merged = {}
    for sweep in sweep_dicts:
        for label, trial in sweep.items():
            if label in merged:
                if on_duplicate == 'error':
                    raise ValueError(f"Duplicate trial label during merge: '{label}'")
                elif on_duplicate == 'warn':
                    warnings.warn(
                        f"Duplicate trial label '{label}' during merge; "
                        f"keeping first occurrence. Use on_duplicate='keep_last' "
                        f"to override."
                    )
                    continue
                # 'keep_last': fall through to overwrite
            merged[label] = trial
    return merged
 
 
# --- Summary table -----------------------------------------------------------
 
def get_sweep_summary(sweep_results, sort_by='best_composite_score', ascending=False):
    """
    Extract a list of scalar-summary dicts for all trials in a sweep,
    suitable for quick inspection (print as a table) or loading into pandas.
 
    Parameters
    ----------
    sweep_results : dict
        {label: trial_result, ...}
    sort_by : str, optional
        Field to sort rows by. Default 'best_composite_score'.
    ascending : bool, optional
        Sort direction. Default False (best score first).
 
    Returns
    -------
    list of dict: One dict per trial, with keys:
        label, gate_orders, num_layers, num_parameters,
        best_validity_rate (%), best_diversity_score, best_composite_score,
        gate_count_total, runtime_seconds.
 
    Example
    -------
    >>> rows = get_sweep_summary(sweep_results)
    >>> for row in rows:
    ...     print(row)
    # or: import pandas as pd; pd.DataFrame(rows)
    """
    rows = []
    for label, trial in sweep_results.items():
        rows.append({
            'label': label,
            'gate_orders': trial.get('gate_orders', []),
            'num_layers': trial.get('num_layers'),
            'num_parameters': trial.get('num_parameters'),
            'best_validity_%': round(100 * trial.get('best_validity_rate', 0), 2),
            'best_diversity_score': round(float(trial.get('best_diversity_score', 0)), 4),
            'best_composite_score': round(float(trial.get('best_composite_score', 0)), 4),
            'gate_count_total': trial.get('gate_count_estimate', {}).get('total'),
            'runtime_seconds': round(trial.get('runtime_seconds', float('nan')), 2),
        })
 
    if sort_by in (rows[0] if rows else {}):
        rows.sort(key=lambda r: (r[sort_by] is None, r[sort_by]),
                  reverse=not ascending)
 
    return rows
 
 
def print_sweep_summary(sweep_results, sort_by='best_composite_score', ascending=False):
    """
    Print get_sweep_summary as a formatted table directly to stdout.
 
    Parameters
    ----------
    sweep_results : dict
    sort_by, ascending : passed to get_sweep_summary.
    """
    rows = get_sweep_summary(sweep_results, sort_by=sort_by, ascending=ascending)
    if not rows:
        print("No trials to summarize.")
        return
 
    col_widths = {k: max(len(str(k)), max(len(str(r[k])) for r in rows))
                  for k in rows[0]}
 
    header = '  '.join(str(k).ljust(col_widths[k]) for k in rows[0])
    print(header)
    print('-' * len(header))
    for row in rows:
        print('  '.join(str(row[k]).ljust(col_widths[k]) for k in row))