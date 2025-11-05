# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 12:04:02 2025

@author: joeco
This file does optimization lifting.
"""

import networkx as nx
import networkx.algorithms.approximation as approx
from itertools import permutations
import time
from opt_helpers import get_trip_time, tour_to_graph
'''
from openqaoa.problems import TSP
from openqaoa import QAOA, create_device
import openqaoa as oq
'''

# these all should have the following input/output:
    # Input: a networkx DiGraph, must be connected.
        # because this uses google maps to get the travel times, "connecting back through" a node is built in. 
        # EX: Connecting through Denver is the fastest way to get to Aspen and Green River,
        # so, this connection is built in to that travel time, even if you have to also go through Denver.
        # Basically, this allows the usual problem constraints to be used.
        
    # Output: a networkx DiGraph, but optimized so there is exactly one edge coming in/out of each node.
    # Output: the time taken.
    
    
def Heuristic_next_closest(graph, graphs_dict: {}, runtime_data: {}, tt_data: {}, label='Nearest-Neighbor'):
    '''
    Parameters
    ----------
    graph : networkx DiGraph
        Graph to do TSP on.
    graphs_dict : {}
        a dictionary of graphs for the output.
    runtime_data : {}
        a dict of runtime data for output.
    tt_data : {}
        a dict of travel time data for output.

    Returns
    -------
    graphs_dict : {}
        a dictionary of grpahs for the output.
    runtime_data : {}
        a dict of runtime data for output.
    tt_data : {}
        a dict of travel time data for output.
    '''
    
    start_time = time.time()
    
    # setup
    nodes = list(graph.nodes)
    curr_node = nodes[0]
    start_node = nodes[0]
    visited = set([curr_node])
    nodes = set(nodes)
    
    TSP_graph = nx.DiGraph()
    TSP_graph.add_nodes_from(nodes)
    
    while visited != nodes:
        # list of remaining nodes - index needs to match the time index, so need list.
        unvisited = list(nodes - visited)
        travel_times = []
        
        # look at unvisited nodes and see travel times to them.
        for node in unvisited:
            travel_times.append(graph[curr_node][node]['weight'])
        
        # find min time and select that node
        min_time = min(travel_times)
        min_time_ind = travel_times.index(min_time)
        next_node = unvisited[min_time_ind]
        
        # add that node/edge to graph # update loops
        TSP_graph.add_edge(curr_node, next_node, weight=min_time)
        curr_node = next_node
        visited.add(next_node)
        
    # connect graph back to start
    TSP_graph.add_edge(curr_node, start_node, weight=graph[curr_node][start_node]['weight'])
    
    end_time = time.time()
    tot_time = end_time - start_time
    
    graphs_dict[label] = TSP_graph
    runtime_data[label] = tot_time
    tt_data[label] = get_trip_time(TSP_graph)

    return graphs_dict, runtime_data, tt_data


def Heuristic_weighted_next_closest(graph, graphs_dict: {}, runtime_data: {}, tt_data: {}, label='Weighted-Nearest-Neighbor'):
    '''
    Parameters
    ----------
    graph : networkx DiGraph
        Graph to do TSP on.
    graphs_dict : {}
        a dictionary of graphs for the output.
    runtime_data : {}
        a dict of runtime data for output.
    tt_data : {}
        a dict of travel time data for output.

    Returns
    -------
    graphs_dict : {}
        a dictionary of grpahs for the output.
    runtime_data : {}
        a dict of runtime data for output.
    tt_data : {}
        a dict of travel time data for output.
    '''
    
    start_time = time.time()
    
    # setup
    nodes = list(graph.nodes)
    curr_node = nodes[0]
    start_node = nodes[0]
    visited = set([curr_node])
    nodes = set(nodes)
    
    TSP_graph = nx.DiGraph()
    TSP_graph.add_nodes_from(nodes)
    
    while visited != nodes:
        # list of remaining nodes - index needs to match the time index, so need list.
        unvisited = list(nodes - visited)
        travel_times = []
        weighted_travel_times = []
        
        # look at unvisited nodes and see travel times to them.
        for node in unvisited:
            base_time = graph[curr_node][node]['weight']
            
            # add a penalty for having a lot of high-weight neighbors
            penalty = 0
            other_unvisited = list(set(unvisited) - set([curr_node]))
            
            for node2 in other_unvisited:
                penalty += graph[curr_node][node2]['weight']
                
            weighted_travel_times.append(base_time + (0.1 * penalty))
            travel_times.append(base_time)
        
        # find min time and select that node
        min_time = min(weighted_travel_times)
        min_time_ind = weighted_travel_times.index(min_time)
        next_node = unvisited[min_time_ind]
        
        # add that node/edge to graph # update loops
        TSP_graph.add_edge(curr_node, next_node, weight=travel_times[min_time_ind])
        curr_node = next_node
        visited.add(next_node)
        
    # connect graph back to start
    TSP_graph.add_edge(curr_node, start_node, weight=graph[curr_node][start_node]['weight'])
    
    end_time = time.time()
    tot_time = end_time - start_time
    
    graphs_dict[label] = TSP_graph
    runtime_data[label] = tot_time
    tt_data[label] = get_trip_time(TSP_graph)

    return graphs_dict, runtime_data, tt_data


def SA_approx(graph, graphs_dict: {}, runtime_data: {}, tt_data: {}, label='Simulated-Annealing'):
    '''
    Parameters
    ----------
    graph : networkx DiGraph
        Graph to do TSP on.
    graphs_dict : {}
        a dictionary of graphs for the output.
    runtime_data : {}
        a dict of runtime data for output.
    tt_data : {}
        a dict of travel time data for output.

    Returns
    -------
    graphs_dict : {}
        a dictionary of grpahs for the output.
    runtime_data : {}
        a dict of runtime data for output.
    tt_data : {}
        a dict of travel time data for output.
    '''
    
    start_time = time.time()
    cycle = approx.simulated_annealing_tsp(graph, "greedy")
    end_time = time.time()
    tot_time = end_time - start_time
    
    # re-make graph from cycle (use helper function).
    SA_graph = tour_to_graph(graph, cycle)
    
    graphs_dict[label] = SA_graph
    runtime_data[label] = tot_time
    tt_data[label] = get_trip_time(SA_graph)
    
    return graphs_dict, runtime_data, tt_data


def tsp_brute_force(G, graphs_dict: {}, runtime_data: {}, tt_data: {}, start_node=None, label='Brute-Force'):
    """
    Solve TSP using brute force approach for directed or undirected graphs.
    THIS IS FROM CLAUDE.
    
    Parameters:
    -----------
    G : networkx.Graph or networkx.DiGraph
        A weighted graph with edge weights representing distances
    start_node : optional
        The node to start the tour from. If None, uses the first node.
    
    Returns:
    --------
    tuple: (min_distance, best_tour, tour_graph)
        min_distance: float - the shortest tour distance
        best_tour: list - the nodes in order of the optimal tour
        tour_graph: networkx.DiGraph - a graph containing only the edges in the optimal tour
    """
    start_time = time.time()
    
    nodes = list(G.nodes())
    all_tour_times = []
    
    if len(nodes) < 2:
        return 0, nodes
    
    # Use specified start node or first node
    if start_node is None:
        start_node = nodes[0]
    elif start_node not in nodes:
        raise ValueError(f"Start node {start_node} not in graph")
    
    # Get all other nodes
    other_nodes = [n for n in nodes if n != start_node]
    
    min_distance = float('inf')
    best_tour = None
    
    # Try all permutations of the other nodes
    for perm in permutations(other_nodes):
        # Create full tour: start -> permutation -> back to start
        tour = [start_node] + list(perm) + [start_node]
        
        # Calculate total distance
        distance = 0
        valid_tour = True
        
        for i in range(len(tour) - 1):
            # For directed graphs, check if edge exists in the correct direction
            if G.has_edge(tour[i], tour[i+1]):
                distance += G[tour[i]][tour[i+1]]['weight']
            else:
                # No edge exists in this direction, tour is invalid
                valid_tour = False
                break
        
        # Update best tour if this one is better
        if valid_tour:
            # record the distance for analysis - might as well!
            all_tour_times.append(distance)
            
            if distance < min_distance:
                min_distance = distance
                best_tour = tour
    
    if best_tour is None:
        raise ValueError("No valid tour found - graph may not be fully connected")
    
    # Create a DiGraph with the optimal tour
    tour_graph = nx.DiGraph()
    for i in range(len(best_tour) - 1):
        u, v = best_tour[i], best_tour[i+1]
        tour_graph.add_edge(u, v, weight=G[u][v]['weight'])
    
    end_time = time.time()
    tot_time = end_time - start_time
    
    graphs_dict[label] = tour_graph
    runtime_data[label] = tot_time
    tt_data[label] = get_trip_time(tour_graph)
    
    return graphs_dict, runtime_data, tt_data, all_tour_times
    
        



 
        
        
        
            