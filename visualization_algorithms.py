# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 10:25:15 2025

@author: joeco
pretty much everything in here was made using Claude.
"""

# this file contains useful algorithms for visualization
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import networkx as nx
import polyline
import contextily as ctx


def plot_travel_times_violin(travel_times, labeled_points=None, title="Travel Time Distribution", 
                              ylabel="Travel Time", figsize=(10, 6)):
    """
    Create a half violin plot of travel times with labeled specific points.
    
    Parameters:
    -----------
    travel_times : array-like
        Array of travel time values to plot
    labeled_points : dict, optional
        Dictionary mapping labels to travel time values, e.g. {'Brute Force': 125.5, 'Greedy': 140.2}
    title : str, optional
        Title for the plot
    ylabel : str, optional
        Label for the y-axis
    figsize : tuple, optional
        Figure size as (width, height)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create violin plot (we'll modify it to show only right side)
    parts = ax.violinplot([travel_times], positions=[0], widths=0.7,
                          showmeans=True, showmedians=True)
    
    # Modify violin to show only right side
    for pc in parts['bodies']:
        # Get the vertices of the violin
        vertices = pc.get_paths()[0].vertices
        # Keep only the right half (x >= 0)
        mid_idx = len(vertices) // 2
        vertices[:mid_idx, 0] = 0  # Set left side x-coordinates to center
        pc.set_facecolor('#8dd3c7')
        pc.set_alpha(0.7)
    
    parts['cmeans'].set_color('#d62728')
    parts['cmedians'].set_color('#2ca02c')
    
    # Add labeled points if provided
    if labeled_points:
        # Sort points by value to help with spacing
        sorted_points = sorted(labeled_points.items(), key=lambda x: x[1])
        
        # Calculate vertical spacing to avoid overlaps
        values = [v for _, v in sorted_points]
        min_spacing = (max(travel_times) - min(travel_times)) * 0.08  # Minimum spacing
        
        # Adjust positions to avoid overlaps
        adjusted_positions = []
        for i, (label, value) in enumerate(sorted_points):
            if i == 0:
                adjusted_positions.append(value)
            else:
                # Check if too close to previous
                prev_pos = adjusted_positions[-1]
                if value - prev_pos < min_spacing:
                    adjusted_positions.append(prev_pos + min_spacing)
                else:
                    adjusted_positions.append(value)
        
        # Plot points and labels
        for i, ((label, value), text_y) in reversed(list(enumerate(zip(sorted_points, adjusted_positions)))):
            # Plot the point at actual value
            ax.plot(0, value, 'o', markersize=10, color='#ff7f0e', zorder=5)
            
            # Position text box on the left side
            x_offset = -0.4
            
            # Add text box with arrow
            bbox_props = dict(boxstyle='round,pad=0.5', facecolor='white', 
                            edgecolor='#ff7f0e', linewidth=2)
            ax.annotate(f'{label}: {value} sec', 
                       xy=(0, value), 
                       xytext=(x_offset, text_y),
                       bbox=bbox_props,
                       arrowprops=dict(arrowstyle='-', color='#ff7f0e', linewidth=1.5),
                       fontsize=10,
                       ha='left',
                       va='center')
    
    # Customize plot
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.set_xlim(-0.5, 0.5)  # Adjust x-limits to accommodate labels on right
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add legend for mean and median
    legend_elements = [
        Line2D([0], [0], color='#d62728', marker='_', linestyle='-', 
               markersize=10, label='Mean', linewidth=2),
        Line2D([0], [0], color='#2ca02c', marker='_', linestyle='-', 
               markersize=10, label='Median', linewidth=2)
    ]
    if labeled_points:
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e', 
                   markersize=10, label='Labeled Algorithms')
        )
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    return fig, ax


def plot_runtime_comparison(runtime_data, title="Algorithm Runtime Comparison", 
                           ylabel="Runtime (seconds)", figsize=(10, 6)):
    """
    Create a bar chart of algorithm runtimes on a log scale.
    
    Parameters:
    -----------
    runtime_data : dict
        Dictionary mapping algorithm names to runtime values (in seconds)
        e.g. {'Brute Force': 0.45, 'Greedy': 0.002, 'Asadpour': 120.5}
    title : str, optional
        Title for the plot
    ylabel : str, optional
        Label for the y-axis
    figsize : tuple, optional
        Figure size as (width, height)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract data
    algorithms = list(runtime_data.keys())
    runtimes = list(runtime_data.values())
    
    # Create color palette
    colors = plt.cm.Set3(np.linspace(0, 1, len(algorithms)))
    
    # Create bar chart
    bars = ax.bar(algorithms, runtimes, color=colors, edgecolor='black', linewidth=1.5)
    
    # Set log scale
    ax.set_yscale('log')
    
    # Add dotted black line at y=1 (or another reference if needed)
    ax.axhline(y=1, color='black', linestyle='--', linewidth=1.5, label='1 second', zorder=0)
    
    # Customize plot
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--', which='both')
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of bars
    for bar, runtime in zip(bars, runtimes):
        height = bar.get_height()
        if runtime >= 1:
            label_text = f'{runtime:.2f}s'
        elif runtime >= 0.001:
            label_text = f'{runtime*1000:.1f}ms'
        else:
            label_text = f'{runtime*1e6:.1f}μs'
        
        ax.text(bar.get_x() + bar.get_width()/2., height,
                label_text,
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.legend()
    plt.tight_layout()
    return fig, ax


def compare_tour_orders(base_tour, other_tour):
    """
    Compare two tours and identify nodes that are in different relative positions.
    
    Parameters:
    -----------
    base_tour : list
        The reference tour (from brute force)
    other_tour : list
        The tour to compare against the base
    
    Returns:
    --------
    set: Nodes that appear in different relative order
    """
    # Remove the last node if it's a return to start
    if len(base_tour) > 1 and base_tour[0] == base_tour[-1]:
        base_tour = base_tour[:-1]
    if len(other_tour) > 1 and other_tour[0] == other_tour[-1]:
        other_tour = other_tour[:-1]
    
    # Find the index of each node in both tours
    base_indices = {node: i for i, node in enumerate(base_tour)}
    other_indices = {node: i for i, node in enumerate(other_tour)}
    
    # Check for nodes that are out of order
    out_of_order = set()
    
    for i, node1 in enumerate(base_tour):
        for j, node2 in enumerate(base_tour):
            if i < j:  # node1 comes before node2 in base tour
                # Check if the relative order is preserved in other tour
                if node1 in other_indices and node2 in other_indices:
                    if other_indices[node1] > other_indices[node2]:
                        # Order is reversed in other tour
                        out_of_order.add(node1)
                        out_of_order.add(node2)
    
    return out_of_order


def extract_tour_from_graph(G):
    """
    Extract the tour as an ordered list of nodes from a tour graph.
    
    Parameters:
    -----------
    G : networkx.DiGraph
        A tour graph with edges forming a cycle
    
    Returns:
    --------
    list: Ordered list of nodes in the tour
    """
    if G.number_of_nodes() == 0:
        return []
    
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
    
    return tour


def plot_tour_comparison(tour_graphs, base_graph_key='base_graph', 
                        brute_force_key='Brute-Force',
                        figsize=None, layout='circular'):
    """
    Visualize multiple tour graphs in a multi-panel plot, highlighting differences from brute force.
    
    Parameters:
    -----------
    tour_graphs : dict
        Dictionary mapping names to networkx DiGraph tour objects
        e.g. {'base_graph': G1, 'Brute-Force': G2, 'Greedy': G3, ...}
    base_graph_key : str, optional
        Key for the base graph (won't be colored for differences)
    brute_force_key : str, optional
        Key for the brute force solution (reference for comparison)
    figsize : tuple, optional
        Figure size as (width, height). If None, auto-calculated
    layout : str, optional
        Layout algorithm: 'circular', 'spring', or 'shell'
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes objects
    """
    n_graphs = len(tour_graphs)
    
    # Calculate grid dimensions
    n_cols = min(3, n_graphs)
    n_rows = (n_graphs + n_cols - 1) // n_cols
    
    if figsize is None:
        figsize = (5 * n_cols, 5 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes for easier iteration
    if n_graphs == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else axes
    
    # Get the brute force tour for comparison
    brute_force_tour = None
    if brute_force_key in tour_graphs:
        brute_force_tour = extract_tour_from_graph(tour_graphs[brute_force_key])
    
    # Get node ordering from brute force solution
    if brute_force_tour and layout == 'circular':
        # Remove duplicate end node if present
        if brute_force_tour[0] == brute_force_tour[-1]:
            ordered_nodes = brute_force_tour[:-1]
        else:
            ordered_nodes = brute_force_tour
        
        # Create circular positions based on brute force order
        n = len(ordered_nodes)
        pos = {}
        for i, node in enumerate(ordered_nodes):
            angle = 2 * np.pi * i / n
            pos[node] = (np.cos(angle), np.sin(angle))
    else:
        # Fallback to standard layout
        base_G = tour_graphs.get(base_graph_key, list(tour_graphs.values())[0])
        if layout == 'circular':
            pos = nx.circular_layout(base_G)
        elif layout == 'spring':
            pos = nx.spring_layout(base_G, seed=42)
        elif layout == 'shell':
            pos = nx.shell_layout(base_G)
        else:
            pos = nx.circular_layout(base_G)
    
    # Plot each graph
    for idx, (name, G) in enumerate(tour_graphs.items()):
        ax = axes[idx]
        
        # Determine node colors
        if name == base_graph_key or brute_force_tour is None or name == brute_force_key:
            # No coloring for base graph or brute force itself
            node_colors = ['#1f77b4'] * G.number_of_nodes()
        else:
            # Compare with brute force
            current_tour = extract_tour_from_graph(G)
            out_of_order = compare_tour_orders(brute_force_tour, current_tour)
            
            # Color nodes: red if out of order, blue otherwise
            node_colors = ['#d62728' if node in out_of_order else '#1f77b4' 
                          for node in G.nodes()]
        
        # Draw the graph
        nx.draw_networkx(G, pos, ax=ax, 
                        node_color=node_colors,
                        node_size=700,
                        font_size=12,
                        font_color='white',
                        font_weight='bold',
                        edge_color='black',
                        width=2,
                        arrowsize=20,
                        arrowstyle='->')
        
        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(n_graphs, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig, axes


def extract_route_coordinates(directions_result):
    """
    Extract coordinates from Google Maps directions result.
    
    Parameters:
    -----------
    directions_result : dict
        Google Maps directions API result
    
    Returns:
    --------
    list: List of (lat, lon) tuples for the route
    """
    if not directions_result or len(directions_result) == 0:
        return []
    
    # Get the polyline from the overview_polyline
    encoded_polyline = directions_result[0]['overview_polyline']['points']
    
    # Decode the polyline to get coordinates
    coordinates = polyline.decode(encoded_polyline)
    
    return coordinates


def plot_route_on_map(tour, addresses, directions_matrix, title="TSP Route", 
                      figsize=(12, 10), show_labels=True, use_map_background=True,
                      map_style='default'):
    """
    Plot a TSP tour on a map using actual Google Maps routes.
    
    Parameters:
    -----------
    tour : list
        Ordered list of indices representing the tour
    addresses : list
        List of address strings
    directions_matrix : list of lists
        Matrix of Google Maps directions results from make_travel_time_matrix
    title : str, optional
        Title for the plot
    figsize : tuple, optional
        Figure size as (width, height)
    show_labels : bool, optional
        Whether to show address labels
    use_map_background : bool, optional
        Whether to use map tiles as background (requires contextily)
    map_style : str, optional
        Map style: 'default', 'satellite', 'terrain', 'streets'
        Options: ctx.providers.OpenStreetMap.Mapnik (default)
                 ctx.providers.Esri.WorldImagery (satellite)
                 ctx.providers.Stamen.Terrain (terrain)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract all route segments for the tour
    all_lats = []
    all_lons = []
    
    # Get location coordinates (start/end points)
    location_coords = {}
    
    for i in range(len(tour) - 1):
        from_idx = tour[i]
        to_idx = tour[i + 1]
        
        directions = directions_matrix[from_idx][to_idx]
        
        if directions and directions != 0:
            # Get start location if we don't have it
            if from_idx not in location_coords:
                start_loc = directions[0]['legs'][0]['start_location']
                location_coords[from_idx] = (start_loc['lat'], start_loc['lng'])
            
            # Get end location
            if to_idx not in location_coords:
                end_loc = directions[0]['legs'][0]['end_location']
                location_coords[to_idx] = (end_loc['lat'], end_loc['lng'])
            
            # Get route coordinates
            route_coords = extract_route_coordinates(directions)
            
            if route_coords:
                # Plot the route segment
                lats, lons = zip(*route_coords)
                ax.plot(lons, lats, 'b-', linewidth=3, alpha=0.8, zorder=2)
                
                all_lats.extend(lats)
                all_lons.extend(lons)
    
    # Add map background if available and requested
    if use_map_background and all_lats and all_lons:
        # Set bounds first
        lat_margin = (max(all_lats) - min(all_lats)) * 0.1
        lon_margin = (max(all_lons) - min(all_lons)) * 0.1
        
        ax.set_xlim(min(all_lons) - lon_margin, max(all_lons) + lon_margin)
        ax.set_ylim(min(all_lats) - lat_margin, max(all_lats) + lat_margin)
        
        # Choose map source based on style
        if map_style == 'satellite':
            source = ctx.providers.Esri.WorldImagery
        elif map_style == 'terrain':
            source = ctx.providers.USGS.USTopo
        elif map_style == 'streets':
            source = ctx.providers.CartoDB.Positron
        else:
            source = ctx.providers.OpenStreetMap.Mapnik
        
        # Add basemap (contextily uses Web Mercator, lat/lon are in WGS84)
        ctx.add_basemap(ax, crs='EPSG:4326', source=source, zoom='auto', alpha=0.8)
    
    # Plot location markers
    if location_coords:
        for idx, (lat, lon) in location_coords.items():
            # Determine marker properties
            if idx == tour[0]:
                # Start/end location (assuming tour returns to start)
                color = 'green'
                marker = 's'
                size = 250
                label = 'Start/End'
            else:
                color = 'red'
                marker = 'o'
                size = 200
                label = f'Stop {idx}'
            
            ax.scatter(lon, lat, c=color, marker=marker, s=size, 
                      edgecolors='white', linewidth=3, zorder=3, alpha=0.95)
            
            # Add labels if requested
            if show_labels:
                ax.annotate(f"{idx}", 
                           xy=(lon, lat),
                           xytext=(0, 0),
                           textcoords='offset points',
                           fontsize=14,
                           fontweight='bold',
                           color='white',
                           ha='center',
                           va='center',
                           zorder=4)
    
    # Set map bounds with some padding if not already set
    if not (use_map_background) and all_lats and all_lons:
        lat_margin = (max(all_lats) - min(all_lats)) * 0.1
        lon_margin = (max(all_lons) - min(all_lons)) * 0.1
        
        ax.set_xlim(min(all_lons) - lon_margin, max(all_lons) + lon_margin)
        ax.set_ylim(min(all_lats) - lat_margin, max(all_lats) + lat_margin)
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    if not (use_map_background):
        ax.grid(True, alpha=0.3)
    
    ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig, ax



def plot_multiple_routes_comparison(tours_dict, addresses, directions_matrix, 
                                   figsize=None, use_map_background=True, map_style='default'):
    """
    Plot multiple TSP solutions side by side for comparison.
    
    Parameters:
    -----------
    tours_dict : dict
        Dictionary mapping algorithm names to tour lists
        e.g. {'Brute Force': [0, 1, 2, 3, 0], 'Greedy': [0, 2, 1, 3, 0]}
    addresses : list
        List of address strings
    directions_matrix : list of lists
        Matrix of Google Maps directions results
    figsize : tuple, optional
        Figure size. If None, auto-calculated
    use_map_background : bool, optional
        Whether to use map tiles as background
    map_style : str, optional
        Map style: 'default', 'satellite', 'terrain', 'streets'
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes objects
    """
    n_tours = len(tours_dict)
    n_cols = min(2, n_tours)
    n_rows = (n_tours + n_cols - 1) // n_cols
    
    if figsize is None:
        figsize = (12 * n_cols, 10 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    if n_tours == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else axes
    
    # Choose map source
    if use_map_background:
        if map_style == 'satellite':
            source = ctx.providers.Esri.WorldImagery
        elif map_style == 'terrain':
            source = ctx.providers.Stamen.Terrain
        elif map_style == 'streets':
            source = ctx.providers.CartoDB.Positron
        else:
            source = ctx.providers.OpenStreetMap.Mapnik
    
    for idx, (name, tour) in enumerate(tours_dict.items()):
        ax = axes[idx]
        
        # Extract all route segments for the tour
        all_lats = []
        all_lons = []
        location_coords = {}
        
        for i in range(len(tour) - 1):
            from_idx = tour[i]
            to_idx = tour[i + 1]
            
            directions = directions_matrix[from_idx][to_idx]
            
            if directions and directions != 0:
                if from_idx not in location_coords:
                    start_loc = directions[0]['legs'][0]['start_location']
                    location_coords[from_idx] = (start_loc['lat'], start_loc['lng'])
                
                if to_idx not in location_coords:
                    end_loc = directions[0]['legs'][0]['end_location']
                    location_coords[to_idx] = (end_loc['lat'], end_loc['lng'])
                
                route_coords = extract_route_coordinates(directions)
                
                if route_coords:
                    lats, lons = zip(*route_coords)
                    ax.plot(lons, lats, 'b-', linewidth=3, alpha=0.8, zorder=2)
                    all_lats.extend(lats)
                    all_lons.extend(lons)
        
        # Set bounds
        if all_lats and all_lons:
            lat_margin = (max(all_lats) - min(all_lats)) * 0.1
            lon_margin = (max(all_lons) - min(all_lons)) * 0.1
            ax.set_xlim(min(all_lons) - lon_margin, max(all_lons) + lon_margin)
            ax.set_ylim(min(all_lats) - lat_margin, max(all_lats) + lat_margin)
        
        # Add map background
        if use_map_background and all_lats and all_lons:
            ctx.add_basemap(ax, crs='EPSG:4326', source=source, zoom='auto', alpha=0.8)
        
        # Plot location markers
        if location_coords:
            for loc_idx, (lat, lon) in location_coords.items():
                if loc_idx == tour[0]:
                    color = 'green'
                    marker = 's'
                    size = 250
                else:
                    color = 'red'
                    marker = 'o'
                    size = 200
                
                ax.scatter(lon, lat, c=color, marker=marker, s=size,
                          edgecolors='white', linewidth=3, zorder=3, alpha=0.95)
                
                ax.annotate(f"{loc_idx}",
                           xy=(lon, lat),
                           xytext=(0, 0),
                           textcoords='offset points',
                           fontsize=14,
                           fontweight='bold',
                           color='white',
                           ha='center',
                           va='center',
                           zorder=4)
        
        ax.set_xlabel('Longitude', fontsize=10)
        ax.set_ylabel('Latitude', fontsize=10)
        ax.set_title(name, fontsize=12, fontweight='bold')
        
        if not (use_map_background):
            ax.grid(True, alpha=0.3)
        
        ax.set_aspect('equal')
    
    # Hide unused subplots
    for idx in range(n_tours, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig, axes


def plot_edge_weight_heatmap(G, title="Travel Time Matrix", figsize=(10, 8), 
                              cmap='viridis', show_values=True, format_time=True):
    """
    Create a heatmap visualization of edge weights (travel times) in a graph.
    
    Parameters:
    -----------
    G : networkx.DiGraph or networkx.Graph
        Graph with weighted edges
    title : str, optional
        Title for the plot
    figsize : tuple, optional
        Figure size as (width, height)
    cmap : str, optional
        Colormap to use (e.g., 'viridis', 'plasma', 'RdYlGn_r', 'coolwarm')
    show_values : bool, optional
        Whether to display edge weight values in cells
    format_time : bool, optional
        Whether to format values as time (seconds to min:sec)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    # Get nodes and create matrix
    nodes = sorted(list(G.nodes()))
    n = len(nodes)
    
    # Create weight matrix
    weight_matrix = np.zeros((n, n))
    
    for i, node_i in enumerate(nodes):
        for j, node_j in enumerate(nodes):
            if G.has_edge(node_i, node_j):
                weight_matrix[i, j] = G[node_i][node_j].get('weight', 0)
            elif i == j:
                weight_matrix[i, j] = 0  # Diagonal is 0
            else:
                weight_matrix[i, j] = np.nan  # No edge
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(weight_matrix, cmap=cmap, aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(nodes)
    ax.set_yticklabels(nodes)
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    if format_time:
        cbar.set_label('Travel Time (seconds)', rotation=270, labelpad=20, fontsize=12)
    else:
        cbar.set_label('Edge Weight', rotation=270, labelpad=20, fontsize=12)
    
    # Add text annotations if requested
    if show_values:
        for i in range(n):
            for j in range(n):
                value = weight_matrix[i, j]
                if not np.isnan(value):
                    if format_time and value > 0:
                        # Format as min:sec
                        mins = int(value // 60)
                        secs = int(value % 60)
                        text = f"{mins}:{secs:02d}"
                    else:
                        text = f"{value:.0f}" if value == int(value) else f"{value:.1f}"
                    
                    # Choose text color based on background
                    # text_color = "white" if value > np.nanmax(weight_matrix) * 0.5 else "black"
                    text_color = "black"
                    ax.text(j, i, text, ha="center", va="center", 
                           color=text_color, fontsize=9, fontweight='bold')
    
    # Labels
    ax.set_xlabel('Destination Node', fontsize=12)
    ax.set_ylabel('Origin Node', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig, ax


def plot_travel_time_matrix_from_array(travel_time_matrix, node_labels=None,
                                       title="Travel Time Matrix", figsize=(10, 8),
                                       cmap='viridis', show_values=True, format_time=True):
    """
    Create a heatmap from a travel time matrix (2D array).
    
    Parameters:
    -----------
    travel_time_matrix : 2D array or list of lists
        Matrix of travel times between nodes
    node_labels : list, optional
        Labels for nodes. If None, uses indices.
    title : str, optional
        Title for the plot
    figsize : tuple, optional
        Figure size as (width, height)
    cmap : str, optional
        Colormap to use
    show_values : bool, optional
        Whether to display values in cells
    format_time : bool, optional
        Whether to format values as time (seconds to min:sec)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    # Convert to numpy array
    weight_matrix = np.array(travel_time_matrix)
    n = weight_matrix.shape[0]
    
    # Default labels
    if node_labels is None:
        node_labels = list(range(n))
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap (mask diagonal or zeros if needed)
    im = ax.imshow(weight_matrix, cmap=cmap, aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(node_labels)
    ax.set_yticklabels(node_labels)
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    if format_time:
        cbar.set_label('Travel Time (seconds)', rotation=270, labelpad=20, fontsize=12)
    else:
        cbar.set_label('Edge Weight', rotation=270, labelpad=20, fontsize=12)
    
    # Add text annotations if requested
    if show_values:
        max_val = np.max(weight_matrix[weight_matrix > 0]) if np.any(weight_matrix > 0) else 1
        
        for i in range(n):
            for j in range(n):
                value = weight_matrix[i, j]
                if value > 0 or i == j:  # Show diagonal zeros but not missing edges
                    if format_time and value > 0:
                        # Format as min:sec
                        mins = int(value // 60)
                        secs = int(value % 60)
                        text = f"{mins}:{secs:02d}"
                    else:
                        text = f"{value:.0f}" if value == int(value) else f"{value:.1f}"
                    
                    # Choose text color based on background
                    # text_color = "white" if value > max_val * 0.5 else "black"
                    text_color = "black"
                    ax.text(j, i, text, ha="center", va="center",
                           color=text_color, fontsize=9, fontweight='bold')
    
    # Labels
    ax.set_xlabel('Destination Node', fontsize=12)
    ax.set_ylabel('Origin Node', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig, ax


def plot_benchmark_results(benchmark_results, figsize=(12, 5)):
    """
    Visualize benchmarking results showing time and memory usage vs batch size.
    
    Parameters:
    -----------
    benchmark_results : dict
        Results from benchmark_batch_sizes function
        Format: {batch_size: {'time': float, 'memory_peak_mb': float, 'success': bool}}
    figsize : tuple, optional
        Figure size as (width, height)
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes objects
    """
    # Extract successful results
    successful = {bs: r for bs, r in benchmark_results.items() if r['success']}
    failed = {bs: r for bs, r in benchmark_results.items() if not r['success']}
    
    if not successful:
        print("No successful benchmarks to plot!")
        return None, None
    
    batch_sizes = sorted(successful.keys())
    times = [successful[bs]['time'] for bs in batch_sizes]
    memories = [successful[bs]['memory_peak_mb'] for bs in batch_sizes]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Execution Time vs Batch Size
    ax1.plot(batch_sizes, times, 'o-', linewidth=2, markersize=8, color='#1f77b4', label='Successful')
    
    # Mark failed batch sizes
    if failed:
        failed_sizes = sorted(failed.keys())
        ax1.scatter(failed_sizes, [max(times)] * len(failed_sizes), 
                   marker='x', s=200, color='red', linewidths=3, 
                   label='Failed', zorder=5)
    
    ax1.set_xlabel('Batch Size (qubits)', fontsize=12)
    ax1.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax1.set_title('Simulation Time vs Batch Size', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Highlight optimal (minimum time)
    optimal_idx = np.argmin(times)
    optimal_bs = batch_sizes[optimal_idx]
    ax1.axvline(optimal_bs, color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax1.text(optimal_bs, max(times) * 0.9, f'Optimal\n({optimal_bs})', 
            ha='center', fontweight='bold', color='green')
    
    # Plot 2: Peak Memory vs Batch Size
    ax2.plot(batch_sizes, memories, 'o-', linewidth=2, markersize=8, color='#ff7f0e', label='Peak Memory')
    
    # Mark failed batch sizes
    if failed:
        ax2.scatter(failed_sizes, [max(memories)] * len(failed_sizes), 
                   marker='x', s=200, color='red', linewidths=3, 
                   label='Failed', zorder=5)
    
    ax2.set_xlabel('Batch Size (qubits)', fontsize=12)
    ax2.set_ylabel('Peak Memory (MB)', fontsize=12)
    ax2.set_title('Memory Usage vs Batch Size', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add horizontal line for typical memory limit
    # ax2.axhline(32000, color='red', linestyle=':', linewidth=2, alpha=0.5, label='32GB RAM')
    
    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_qaoa_validity_progress(iteration_data, figsize=(10, 6)):
    """
    Plot the progression of valid vs invalid tours during QAOA optimization.
    
    Parameters:
    -----------
    iteration_data : list of dict
        List where each element represents one iteration/evaluation with keys:
        - 'iteration': int (iteration number)
        - 'valid_shots': int (number of valid tour shots)
        - 'invalid_shots': int (number of invalid tour shots)
        - 'best_cost': float (optional, best tour cost found so far)
    figsize : tuple, optional
        Figure size as (width, height)
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes objects
    """
    iterations = [d['iteration'] for d in iteration_data]
    valid_shots = [d['valid_shots'] for d in iteration_data]
    invalid_shots = [d['invalid_shots'] for d in iteration_data]
    total_shots = [v + i for v, i in zip(valid_shots, invalid_shots)]
    valid_percent = [100 * v / t if t > 0 else 0 for v, t in zip(valid_shots, total_shots)]
    
    # Check if cost data is available
    has_cost = 'best_cost' in iteration_data[0]
    
    if has_cost:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=figsize)
        ax2 = None
    
    # Plot 1: Stacked area chart of valid vs invalid
    ax1.fill_between(iterations, 0, valid_shots, alpha=0.7, color='#2ca02c', label='Valid Tours')
    ax1.fill_between(iterations, valid_shots, total_shots, alpha=0.7, color='#d62728', label='Invalid Tours')
    
    ax1.set_ylabel('Number of Shots', fontsize=12)
    ax1.set_title('QAOA Valid vs Invalid Tours Over Iterations', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add percentage line on secondary axis
    ax1_twin = ax1.twinx()
    ax1_twin.plot(iterations, valid_percent, 'o-', color='blue', linewidth=2, 
                 markersize=6, label='Valid %', alpha=0.8)
    ax1_twin.set_ylabel('Valid Tour Percentage (%)', fontsize=12, color='blue')
    ax1_twin.tick_params(axis='y', labelcolor='blue')
    ax1_twin.set_ylim(0, 105)
    
    # Plot 2: Best cost progression (if available)
    if has_cost and ax2 is not None:
        best_costs = [d['best_cost'] for d in iteration_data]
        ax2.plot(iterations, best_costs, 'o-', linewidth=2, markersize=8, 
                color='#9467bd', label='Best Tour Cost')
        
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Best Tour Cost', fontsize=12)
        ax2.set_title('Best Solution Quality Over Iterations', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Highlight improvements
        for i in range(1, len(best_costs)):
            if best_costs[i] < best_costs[i-1]:
                ax2.plot(iterations[i], best_costs[i], 'g*', markersize=15, zorder=5)
    
    if not has_cost:
        ax1.set_xlabel('Iteration', fontsize=12)
    
    plt.tight_layout()
    return fig, (ax1, ax2) if has_cost else (fig, ax1)


def plot_qaoa_validity_pie(valid_shots, invalid_shots, title="QAOA Tour Validity", 
                           figsize=(8, 6)):
    """
    Create a pie chart showing the split between valid and invalid tours.
    
    Parameters:
    -----------
    valid_shots : int
        Number of valid tour measurements
    invalid_shots : int
        Number of invalid tour measurements
    title : str, optional
        Title for the plot
    figsize : tuple, optional
        Figure size as (width, height)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    total = valid_shots + invalid_shots
    sizes = [valid_shots, invalid_shots]
    labels = [f'Valid Tours\n({valid_shots} shots, {100*valid_shots/total:.1f}%)',
              f'Invalid Tours\n({invalid_shots} shots, {100*invalid_shots/total:.1f}%)']
    colors = ['#2ca02c', '#d62728']
    explode = (0.05, 0)  # Slightly separate the valid slice
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                       explode=explode, autopct='',
                                       shadow=True, startangle=90)
    
    # Make text bold and larger
    for text in texts:
        text.set_fontsize(12)
        text.set_fontweight('bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig, ax
