#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 18:05:51 2026

@author: joecooney
"""

import networkx as nx
import numpy as np
import itertools as it
from quantum_hyperparameter_tuning import run_single_hyperparameter_trial, aggregate_external_trial_results, plot_validity_diversity_vs_order
from google_maps import get_travel_time_matrix, get_address_set, get_directions_matrix
from multiprocessing import Pool


def run_trial(args):
    graph, gate_orders, num_qubits = args
    return run_single_hyperparameter_trial(
        graph, gate_orders,
        num_layers=3, shots=16384, batch_size=num_qubits,
        max_iterations=100, verbose=False
    )


if __name__ == '__main__':
    
    # get data for graphs
    curr_prob = '4m_4_1'
    ttm = get_travel_time_matrix(f'{curr_prob}/ttm.txt')
    address_set = get_address_set(f'{curr_prob}/address-set.txt')
    dirs_mat = get_directions_matrix(f'{curr_prob}/dir-mat.json')
    save_all = True

    # base graph
    graph = nx.from_numpy_array(np.array(ttm), create_using=nx.DiGraph)
    # graphs_dict = {'Base Graph': graph}
    graphs_dict = {}
    runtime_data = {}
    labelled_tt_data = {}
    qaoa_progress = {}

    num_qubits = 12


    all_orders = [i for i in range(1, num_qubits+1)]
    full_gate_orders_list = [i for i in range(1, num_qubits+1)]

    for i in range(2, num_qubits+1):
        combos = list(it.combinations(all_orders, i))
        for combo in combos:
            full_gate_orders_list.append(list(combo))


    results_list = []
    num_to_test = len(full_gate_orders_list)
    gate_orders_indices = np.random.choice(len(full_gate_orders_list), num_to_test, replace=False)
    gate_orders_list = [full_gate_orders_list[i] for i in gate_orders_indices]
    
    configs = gate_orders_list
    
    with Pool(processes=20) as pool:
        results = pool.map(run_trial, [(graph, orders, num_qubits) for orders in configs])
    
    sweep_results = aggregate_external_trial_results(results)

    plot_validity_diversity_vs_order(sweep_results)



