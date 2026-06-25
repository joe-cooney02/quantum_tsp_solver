# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 08:06:38 2026

@author: joeco


Runs a gate-order hyperparameter sweep using Ray for parallelism across nodes,
and saves all results into the problem directory structure:

    4m_4_1/
      experiments/   <- sweep index, CSV summary, and per-trial JSON files
      results/       <- saved plot images from this run

Running on a single machine
----------------------------
Just run the script directly. Ray will spin up local workers using all
available CPU cores (one worker per trial).

Running on a Ray cluster
-------------------------
Start the cluster head node first:
    ray start --head --port=6379

Then on each worker node:
    ray start --address='<head-node-ip>:6379'

Then set RAY_ADDRESS before running this script:
    RAY_ADDRESS='ray://<head-node-ip>:10001' python hyperparameter_tuning_test.py

Or call ray.init(address='ray://<head-node-ip>:10001') directly in the script
by setting CLUSTER_ADDRESS below.

Note on modules across nodes
------------------------------
For a multi-node cluster, quantum_hyperparameter_tuning, quantum_pretraining,
and quantum_helpers must be importable on every worker node (i.e. on the same
path, or installed as a package). The simplest approach is to keep a copy of
the project directory on each node at the same path and let Ray's working-dir
runtime_env handle the rest -- see the ray.init() call below.

FOR RAY: NEED PYTHON 3.11 OR 3.12
"""

import os
import itertools as it
import networkx as nx
import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend; safe across Ray workers
import matplotlib.pyplot as plt
import ray

from quantum_hyperparameter_tuning import (
    run_single_hyperparameter_trial,
    aggregate_external_trial_results,
    save_sweep_results,
    load_sweep_results,
    print_sweep_summary,
    plot_validity_diversity_vs_order,
    plot_validity_vs_parameter_count,
    plot_sweep_final_bars,
)
from google_maps import get_travel_time_matrix, get_address_set, get_directions_matrix


# =============================================================================
# Configuration
# =============================================================================

CURR_PROB      = '4m_4_1'
EXPERIMENTS_DIR = os.path.join(CURR_PROB, 'experiments')
RESULTS_DIR     = os.path.join(CURR_PROB, 'results')

NUM_QUBITS     = 12
NUM_LAYERS     = 3
SHOTS          = 16384
MAX_ITERATIONS = 100
NUM_TO_TEST    = 5          # number of randomly-sampled gate_orders configs to test

# Set to 'ray://<head-ip>:10001' to connect to an existing cluster,
# or leave as None to start a local Ray instance on this machine.
CLUSTER_ADDRESS = None


# =============================================================================
# Ray remote trial function
#
# @ray.remote makes this run in a separate Ray worker process (local or remote
# node). Each invocation is fully independent, which is exactly what we need:
# run_single_hyperparameter_trial is stateless given (graph, gate_orders).
#
# num_cpus=1 tells the Ray scheduler each trial needs one CPU slot, so it
# will run at most len(available_cpus) trials in parallel automatically --
# no need to specify a pool size like multiprocessing.Pool requires.
# Increase num_cpus if Qiskit Aer's internal threading can usefully claim
# more cores per trial (depends on batch_size and sim_method).
# =============================================================================

@ray.remote(num_cpus=1)
def run_trial_remote(graph, gate_orders, num_qubits):
    """
    Single hyperparameter trial, executed as a Ray task.

    Parameters are passed by value (Ray serializes with pickle), so
    the networkx graph and gate_orders list are safely copied to each worker.
    verbose=False keeps worker stdout clean; Ray captures and can display
    worker logs via `ray logs` if needed.
    """
    return run_single_hyperparameter_trial(
        graph, gate_orders,
        num_layers=NUM_LAYERS,
        shots=SHOTS,
        batch_size=num_qubits,
        max_iterations=MAX_ITERATIONS,
        verbose=False,
    )


# =============================================================================
# Helpers
# =============================================================================

def build_full_gate_orders_list(num_qubits):
    """
    Build the complete list of gate_orders configurations to sweep over:
    all single orders [1]..[num_qubits], then all pairs, triples, etc.
    This grows combinatorially -- use NUM_TO_TEST to sample a manageable
    subset for any given run.
    """
    all_orders = list(range(1, num_qubits + 1))
    configs = [[k] for k in all_orders]             # single-order configs first
    for r in range(2, num_qubits + 1):
        for combo in it.combinations(all_orders, r):
            configs.append(list(combo))
    return configs


def save_plots(sweep_results, results_dir, sweep_name):
    """
    Generate and save all sweep visualization plots to results_dir.
    Reuses the existing visualization functions from
    quantum_hyperparameter_tuning, which in turn reuse visualization_algorithms.

    Saved files:
      {sweep_name}_validity_diversity_vs_order.png
      {sweep_name}_validity_vs_param_count.png
      {sweep_name}_final_bars.png
    """
    os.makedirs(results_dir, exist_ok=True)
    saved = []

    # Validity & diversity vs gate order (or fallback bar chart for mixed configs)
    try:
        fig, _ = plot_validity_diversity_vs_order(sweep_results)
        if fig is not None:
            path = os.path.join(results_dir, f'{sweep_name}_validity_diversity_vs_order.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            saved.append(path)
    except Exception as e:
        print(f"  [warn] plot_validity_diversity_vs_order failed: {e}")

    # Validity vs parameter count scatter (colored by diversity)
    try:
        fig, _ = plot_validity_vs_parameter_count(sweep_results)
        if fig is not None:
            path = os.path.join(results_dir, f'{sweep_name}_validity_vs_param_count.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            saved.append(path)
    except Exception as e:
        print(f"  [warn] plot_validity_vs_parameter_count failed: {e}")

    # Final comparison bars
    try:
        fig, _ = plot_sweep_final_bars(sweep_results)
        if fig is not None:
            path = os.path.join(results_dir, f'{sweep_name}_final_bars.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            saved.append(path)
    except Exception as e:
        print(f"  [warn] plot_sweep_final_bars failed: {e}")

    return saved


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # 1. Load problem data
    # -------------------------------------------------------------------------
    print(f"Loading problem: {CURR_PROB}")
    ttm         = get_travel_time_matrix(f'{CURR_PROB}/ttm.txt')
    address_set = get_address_set(f'{CURR_PROB}/address-set.txt')
    dirs_mat    = get_directions_matrix(f'{CURR_PROB}/dir-mat.json')

    graph = nx.from_numpy_array(np.array(ttm), create_using=nx.DiGraph)
    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges, "
          f"{NUM_QUBITS} qubits")

    # -------------------------------------------------------------------------
    # 2. Sample gate_orders configurations to test
    # -------------------------------------------------------------------------
    full_gate_orders_list = build_full_gate_orders_list(NUM_QUBITS)
    print(f"Total possible configs: {len(full_gate_orders_list)}, "
          f"sampling {NUM_TO_TEST} at random")

    rng = np.random.default_rng()   # seeded by OS entropy; reproducible runs
    indices = rng.choice(len(full_gate_orders_list), size=NUM_TO_TEST, replace=False)
    sampled_configs = [full_gate_orders_list[i] for i in sorted(indices)]

    print("Sampled gate_orders configs:")
    for cfg in sampled_configs:
        print(f"  {cfg}")

    # -------------------------------------------------------------------------
    # 3. Initialize Ray
    #
    # runtime_env distributes the local source files to every worker node so
    # imports work without manual installation on each node. Remove or adjust
    # 'working_dir' if your project is already installed as a package.
    # -------------------------------------------------------------------------
    ray.init(
        address=CLUSTER_ADDRESS,          # None = local; set for cluster
        runtime_env={
            'working_dir': '.',           # ships current directory to workers
            'pip': ['qiskit', 'qiskit-aer', 'scipy', 'networkx', 'numpy'],
        },
        ignore_reinit_error=True,
    )
    print(f"\nRay initialized: {ray.cluster_resources()}")

    # -------------------------------------------------------------------------
    # 4. Dispatch trials to Ray workers and collect results
    #
    # ray.remote returns an ObjectRef (a future). ray.get() blocks until all
    # futures are resolved. Ray schedules work across available workers
    # automatically -- no pool size to set.
    # -------------------------------------------------------------------------
    print(f"\nDispatching {NUM_TO_TEST} trials to Ray workers...")
    futures = [
        run_trial_remote.remote(graph, orders, NUM_QUBITS)
        for orders in sampled_configs
    ]
    results = ray.get(futures)      # blocks until all workers finish
    ray.shutdown()

    sweep_results = aggregate_external_trial_results(results)
    print(f"\nAll {len(sweep_results)} trials complete.")

    # -------------------------------------------------------------------------
    # 5. Print summary to stdout for quick inspection
    # -------------------------------------------------------------------------
    print()
    print_sweep_summary(sweep_results)

    # -------------------------------------------------------------------------
    # 6. Save experiment data to 4m_4_1/experiments/
    #
    # save_sweep_results writes:
    #   experiments/trials/{label}.json     -- full per-trial data
    #   experiments/{sweep_name}_index.json -- scalar summary (fast to reload)
    #   experiments/{sweep_name}_summary.csv
    #
    # The sweep_name embeds the problem ID so experiments from different
    # problems never collide in the same directory.
    # -------------------------------------------------------------------------
    from datetime import datetime
    sweep_name = f"{CURR_PROB}_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\nSaving experiment data to {EXPERIMENTS_DIR}/")
    save_info = save_sweep_results(
        sweep_results,
        save_dir=EXPERIMENTS_DIR,
        sweep_name=sweep_name,
        save_history=True,      # keep full per-iteration stats for later plotting
    )
    print(f"  index:  {os.path.relpath(save_info['index_path'])}")
    print(f"  csv:    {os.path.relpath(save_info['csv_path'])}")
    print(f"  trials: {list(save_info['trial_paths'].keys())}")

    # -------------------------------------------------------------------------
    # 7. Save plots to 4m_4_1/results/
    # -------------------------------------------------------------------------
    print(f"\nSaving plots to {RESULTS_DIR}/")
    saved_plots = save_plots(sweep_results, RESULTS_DIR, sweep_name)
    for p in saved_plots:
        print(f"  {os.path.relpath(p)}")

    # -------------------------------------------------------------------------
    # 8. Demonstrate reloading (verifies the saved data is intact)
    # -------------------------------------------------------------------------
    print(f"\nVerifying reload from {EXPERIMENTS_DIR}/...")
    reloaded = load_sweep_results(EXPERIMENTS_DIR, sweep_name=sweep_name)
    assert set(reloaded.keys()) == set(sweep_results.keys()), "Reload label mismatch!"
    print(f"  OK -- {len(reloaded)} trials reloaded successfully.")
    print(f"\nDone. Results in {CURR_PROB}/")
