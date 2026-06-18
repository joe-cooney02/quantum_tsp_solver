#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 18:05:51 2026

@author: joecooney
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import itertools as it
from quantum_hyperparameter_tuning import run_single_hyperparameter_trial, aggregate_external_trial_results, plot_validity_diversity_vs_order
from google_maps import get_travel_time_matrix, get_address_set, get_directions_matrix


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


all_orders = [i for i in range(1, 13)]
full_gate_orders_list = [i for i in range(1, 13)]

for i in range(2, 13):
    combos = list(it.combinations(all_orders, i))
    for combo in combos:
        full_gate_orders_list.append(list(combo))


results_list = []
gate_orders_indices = np.random.choice(len(full_gate_orders_list), 5, replace=False)
gate_orders_list = [full_gate_orders_list[i] for i in gate_orders_indices]


for orders in gate_orders_list:
    # ask claude: Why is this running in single-thread instead of all threads?
    results_list.append(run_single_hyperparameter_trial(graph, orders, shots=10000, batch_size=12, num_layers=1, label=f'{orders} order gates'))
    
    
results_for_vis = aggregate_external_trial_results(results_list)

plot_validity_diversity_vs_order(results_for_vis)



