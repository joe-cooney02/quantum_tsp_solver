# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 12:13:46 2025

@author: joeco
"""

import networkx as nx

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