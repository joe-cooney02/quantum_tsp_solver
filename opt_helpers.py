# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 12:13:46 2025

@author: joeco
"""

import networkx as nx
import numpy as np

# contains helper functions for TSP problems
def get_trip_time(graph):
    """
    Calculate the total distance/weight of a tour graph.
    
    Parameters:
    -----------
    graph : networkx.DiGraph
        A graph representing a tour (one edge in/out from each node!)
    
    Returns:
    --------
    float: The total distance of the tour
    """
    total_distance = sum(data['weight'] for u, v, data in graph.edges(data=True))
    
    return total_distance


def tour_to_graph(G, tour):
    """
    Create a DiGraph from a tour list using edges from the original graph.
    
    Parameters:
    -----------
    G : networkx.Graph or networkx.DiGraph
        The original graph containing all edges with weights
    tour : list
        A list of nodes representing the tour path
    
    Returns:
    --------
    networkx.DiGraph: A graph containing only the edges in the tour
    """
    tour_graph = nx.DiGraph()
    
    for i in range(len(tour) - 1):
        u, v = tour[i], tour[i+1]
        if not G.has_edge(u, v):
            raise ValueError(f"Edge ({u}, {v}) does not exist in the original graph")
        tour_graph.add_edge(u, v, weight=G[u][v]['weight'])
    
    return tour_graph


def graphs_to_tours(tour_graphs_dict):
    """
    Convert a dictionary of tour DiGraphs to a dictionary of tour lists.
    
    Parameters:
    -----------
    tour_graphs_dict : dict
        Dictionary mapping names to networkx DiGraph tour objects
        e.g. {'Brute Force': tour_graph1, 'Greedy': tour_graph2}
    
    Returns:
    --------
    dict: Dictionary mapping names to tour lists (ordered node sequences)
    """
    tours_dict = {}
    
    for name, G in tour_graphs_dict.items():
        if G.number_of_nodes() == 0:
            tours_dict[name] = []
            continue
        
        # Start from any node
        start_node = list(G.nodes())[0]
        tour = [start_node]
        current = start_node
        
        # Follow edges to build tour
        while True:
            neighbors = list(G.successors(current))
            if not neighbors:
                break
            next_node = neighbors[0]
            if next_node == start_node:
                tour.append(next_node)
                break
            tour.append(next_node)
            current = next_node
        
        tours_dict[name] = tour
    
    return tours_dict


def get_warm_start_tour(G, method='nearest_neighbor', start_node=None, seed=None):
    """
    Generate a warm-start tour using a classical heuristic.
    
    Parameters:
    -----------
    G : networkx.DiGraph
        The TSP graph
    method : str, optional
        Heuristic method: 'nearest_neighbor', 'random', 'farthest_insertion', 
        'cheapest_insertion', 'random_nearest_neighbor'
    start_node : optional
        Starting node for the tour
    seed : int, optional
        Random seed for reproducibility (for random methods)
    
    Returns:
    --------
    list: Tour as ordered list of nodes (including return to start)
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    nodes = list(G.nodes())
    if start_node is None:
        start_node = nodes[0]
    
    if method == 'nearest_neighbor':
        # Greedy: always pick the shortest available edge
        # unfortunately, we can't use the alg from optimization_engines - circular imports.
        # conveniently, this is pretty short code.
        
        unvisited = set(nodes)
        tour = [start_node]
        unvisited.remove(start_node)
        
        current = start_node
        while unvisited:
            # Find nearest unvisited neighbor
            best_next = None
            best_weight = float('inf')
            
            for next_node in unvisited:
                if G.has_edge(current, next_node):
                    weight = G[current][next_node]['weight']
                    if weight < best_weight:
                        best_weight = weight
                        best_next = next_node
            
            if best_next is None:
                # No edge available, pick any unvisited node
                best_next = unvisited.pop()
                unvisited.add(best_next)
            
            tour.append(best_next)
            unvisited.remove(best_next)
            current = best_next
        
        tour.append(start_node)  # Return to start
        
        return tour
    
    
    elif method == 'random':
        # Random valid tour
        tour = nodes.copy()
        np.random.shuffle(tour)
        # Make sure start_node is first
        tour.remove(start_node)
        tour = [start_node] + tour + [start_node]
        return tour
    
    elif method == 'farthest_insertion':
        # Start with a partial tour and iteratively add the farthest unvisited node
        tour = [start_node]
        unvisited = set(nodes)
        unvisited.remove(start_node)
        
        # Find the farthest node from start to add first
        if unvisited:
            farthest_node = max(unvisited, 
                              key=lambda n: G[start_node][n]['weight'] if G.has_edge(start_node, n) else 0)
            tour.append(farthest_node)
            unvisited.remove(farthest_node)
        
        # Build tour by inserting farthest remaining node at best position
        while unvisited:
            # Find the farthest node from current tour
            farthest_node = None
            max_min_dist = -float('inf')
            
            for node in unvisited:
                # Find minimum distance to tour
                min_dist = float('inf')
                for tour_node in tour:
                    if G.has_edge(tour_node, node):
                        dist = G[tour_node][node]['weight']
                        min_dist = min(min_dist, dist)
                
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    farthest_node = node
            
            # Find best position to insert this node
            best_pos = 1
            best_cost_increase = float('inf')
            
            for i in range(len(tour)):
                # Cost to insert between tour[i] and tour[(i+1) % len(tour)]
                curr_node = tour[i]
                next_node = tour[(i + 1) % len(tour)]
                
                # Current edge cost
                curr_cost = G[curr_node][next_node]['weight'] if G.has_edge(curr_node, next_node) else float('inf')
                
                # New edges cost
                new_cost = 0
                if G.has_edge(curr_node, farthest_node):
                    new_cost += G[curr_node][farthest_node]['weight']
                else:
                    new_cost = float('inf')
                    
                if G.has_edge(farthest_node, next_node):
                    new_cost += G[farthest_node][next_node]['weight']
                else:
                    new_cost = float('inf')
                
                cost_increase = new_cost - curr_cost
                
                if cost_increase < best_cost_increase:
                    best_cost_increase = cost_increase
                    best_pos = i + 1
            
            tour.insert(best_pos, farthest_node)
            unvisited.remove(farthest_node)
        
        tour.append(start_node)  # Return to start
        return tour
    
    elif method == 'cheapest_insertion':
        # Start with a partial tour and iteratively add node with cheapest insertion cost
        tour = [start_node]
        unvisited = set(nodes)
        unvisited.remove(start_node)
        
        # Add nearest node to start
        if unvisited:
            nearest_node = min(unvisited,
                             key=lambda n: G[start_node][n]['weight'] if G.has_edge(start_node, n) else float('inf'))
            tour.append(nearest_node)
            unvisited.remove(nearest_node)
        
        # Build tour by inserting cheapest node at best position
        while unvisited:
            best_node = None
            best_pos = 1
            best_cost_increase = float('inf')
            
            # Try each unvisited node
            for node in unvisited:
                # Try each position in tour
                for i in range(len(tour)):
                    curr_node = tour[i]
                    next_node = tour[(i + 1) % len(tour)]
                    
                    # Current edge cost
                    curr_cost = G[curr_node][next_node]['weight'] if G.has_edge(curr_node, next_node) else float('inf')
                    
                    # New edges cost
                    new_cost = 0
                    if G.has_edge(curr_node, node):
                        new_cost += G[curr_node][node]['weight']
                    else:
                        new_cost = float('inf')
                        
                    if G.has_edge(node, next_node):
                        new_cost += G[node][next_node]['weight']
                    else:
                        new_cost = float('inf')
                    
                    cost_increase = new_cost - curr_cost
                    
                    if cost_increase < best_cost_increase:
                        best_cost_increase = cost_increase
                        best_node = node
                        best_pos = i + 1
            
            tour.insert(best_pos, best_node)
            unvisited.remove(best_node)
        
        tour.append(start_node)  # Return to start
        return tour
    
    elif method == 'random_nearest_neighbor':
        # Like nearest neighbor but with randomness - pick from top K nearest
        unvisited = set(nodes)
        tour = [start_node]
        unvisited.remove(start_node)
        
        current = start_node
        while unvisited:
            # Find K nearest neighbors (or all if fewer than K)
            K = min(3, len(unvisited))  # Consider top 3 nearest
            
            # Get distances to all unvisited nodes
            candidates = []
            for next_node in unvisited:
                if G.has_edge(current, next_node):
                    weight = G[current][next_node]['weight']
                    candidates.append((next_node, weight))
            
            if not candidates:
                # No edges available, pick any
                best_next = unvisited.pop()
                unvisited.add(best_next)
            else:
                # Sort by weight and pick randomly from top K
                candidates.sort(key=lambda x: x[1])
                top_k = candidates[:K]
                best_next = np.random.choice([node for node, _ in top_k])
            
            tour.append(best_next)
            unvisited.remove(best_next)
            current = best_next
        
        tour.append(start_node)  # Return to start
        return tour
    
    else:
        raise ValueError(f"Unknown method: {method}. Available methods: "
                        f"'nearest_neighbor', 'random', 'farthest_insertion', "
                        f"'cheapest_insertion', 'random_nearest_neighbor'")