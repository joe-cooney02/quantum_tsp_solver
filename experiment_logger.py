# -*- coding: utf-8 -*-
"""
Created on January 22, 2026

@author: Joe Cooney

Experiment logging and results management for QAOA TSP experiments.
"""

import json
import datetime
import networkx as nx
from pathlib import Path
import pandas as pd
import numpy as np



def serialize_graph(G):
    """Convert NetworkX graph to JSON-serializable dict."""
    return {
        'nodes': list(G.nodes()),
        'edges': [(u, v, G[u][v]) for u, v in G.edges()]
    }


def deserialize_graph(graph_dict):
    """Reconstruct NetworkX graph from dict."""
    G = nx.DiGraph()
    G.add_nodes_from(graph_dict['nodes'])
    for u, v, data in graph_dict['edges']:
        G.add_edge(u, v, **data)
    return G


def save_experiment_results(experiment_name, problem_name, results, hyperparameters, 
                            notes="", base_dir=None):
    """
    Save complete experiment results with metadata.
    
    Parameters:
    -----------
    experiment_name : str
        Descriptive name for this experiment (e.g., "QAOA_warmstart_zero_init")
    problem_name : str
        Problem instance name (e.g., "4m_10_1")
    results : dict
        Dictionary containing:
        - graphs_dict: {algorithm_name: tour_graph}
        - runtime_data: {algorithm_name: runtime_seconds}
        - tt_data: {algorithm_name: travel_time}
        - qaoa_progress: {label: [qaoa_stats_list]} (IMPORTANT: Include all QAOA runs)
        - valid_tours: list of tours (optional)
        - all_times: list of all travel times (optional)
    hyperparameters : dict
        QAOA hyperparameters used:
        - layers, shots, qubit_batch_size, inv_penalty_m, 
        - sim_method, warm_start, exploration_strength, 
        - initialization_strategy, etc.
        Note: For multi-run experiments, this can be a dict of dicts keyed by run label
    notes : str, optional
        Any additional notes about this experiment
    base_dir : str, optional
        Base directory for saving (default: problem_name/experiments/)
    
    Returns:
    --------
    str: Path to saved file
    """
    # Create directory structure
    if base_dir is None:
        base_dir = Path(problem_name) / "experiments"
    else:
        base_dir = Path(base_dir)
    
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = base_dir / f"{experiment_name}_{timestamp}.json"
    
    # Prepare data structure
    experiment_data = {
        "metadata": {
            "experiment_name": experiment_name,
            "problem_name": problem_name,
            "timestamp": timestamp,
            "datetime": datetime.datetime.now().isoformat(),
            "notes": notes
        },
        "hyperparameters": hyperparameters,
        "results": {
            "algorithms": list(results.get('graphs_dict', {}).keys()),
            "runtime_data": results.get('runtime_data', {}),
            "travel_time_data": results.get('tt_data', {}),
        }
    }
    
    # Serialize graphs
    if 'graphs_dict' in results:
        experiment_data["results"]["tour_graphs"] = {
            name: serialize_graph(G) 
            for name, G in results['graphs_dict'].items()
        }
    
    # Add QAOA progress if available - ENHANCED VERSION
    if 'qaoa_progress' in results and results['qaoa_progress']:
        # Convert any non-serializable objects in qaoa_progress
        serializable_progress = {}
        qaoa_summary = {}  # Summary statistics for each run
        
        for label, stats_list in results['qaoa_progress'].items():
            serializable_stats = []
            
            for stats in stats_list:
                # Create a copy and handle non-serializable fields
                stat_copy = dict(stats)
                
                # Convert tour to list if it's a graph
                if 'best_tour' in stat_copy and isinstance(stat_copy['best_tour'], nx.DiGraph):
                    # Extract tour from graph
                    if stat_copy['best_tour'].number_of_nodes() > 0:
                        start = list(stat_copy['best_tour'].nodes())[0]
                        tour = [start]
                        current = start
                        while True:
                            neighbors = list(stat_copy['best_tour'].successors(current))
                            if not neighbors:
                                break
                            next_node = neighbors[0]
                            if next_node == start:
                                tour.append(start)
                                break
                            tour.append(next_node)
                            current = next_node
                        stat_copy['best_tour'] = tour
                    else:
                        stat_copy['best_tour'] = []
                
                serializable_stats.append(stat_copy)
            
            serializable_progress[label] = serializable_stats
            
            # Create summary for this run
            if serializable_stats:
                final_stats = serializable_stats[-1]
                initial_cost = next((s['best_cost'] for s in serializable_stats 
                                   if s.get('best_cost') is not None), None)
                final_cost = final_stats.get('best_cost')
                
                qaoa_summary[label] = {
                    'num_iterations': len(serializable_stats),
                    'final_validity_pct': final_stats.get('valid_percentage', 0),
                    'final_best_cost': final_cost,
                    'initial_cost': initial_cost,
                    'improvement_pct': ((initial_cost - final_cost) / initial_cost * 100) 
                                      if (initial_cost and final_cost) else 0,
                    'avg_validity_pct': np.mean([s.get('valid_percentage', 0) 
                                                 for s in serializable_stats]),
                    'avg_diversity': np.mean([s.get('num_unique_bitstrings', 0) 
                                             for s in serializable_stats]),
                    'max_validity_pct': max([s.get('valid_percentage', 0) 
                                            for s in serializable_stats]),
                    'min_cost': min([s.get('best_cost', float('inf')) 
                                    for s in serializable_stats 
                                    if s.get('best_cost') is not None], default=None)
                }
        
        experiment_data["results"]["qaoa_progress"] = serializable_progress
        experiment_data["results"]["qaoa_summary"] = qaoa_summary
        
        # Add comparison if multiple QAOA runs
        if len(qaoa_summary) > 1:
            best_run = min(qaoa_summary.items(), 
                          key=lambda x: x[1]['final_best_cost'] 
                          if x[1]['final_best_cost'] else float('inf'))
            
            experiment_data["results"]["qaoa_comparison"] = {
                'best_run_by_cost': best_run[0],
                'best_final_cost': best_run[1]['final_best_cost'],
                'best_run_by_validity': max(qaoa_summary.items(), 
                                           key=lambda x: x[1]['final_validity_pct'])[0],
                'all_final_costs': {label: summary['final_best_cost'] 
                                   for label, summary in qaoa_summary.items()},
                'all_final_validities': {label: summary['final_validity_pct'] 
                                        for label, summary in qaoa_summary.items()}
            }
    
    # Add valid tours summary if available
    if 'valid_tours' in results:
        experiment_data["results"]["num_valid_tours"] = len(results['valid_tours'])
        # Optionally save first few tours as examples
        experiment_data["results"]["example_valid_tours"] = results['valid_tours'][:5]
    
    # Add all times statistics if available
    if 'all_times' in results:
        experiment_data["results"]["brute_force_statistics"] = {
            "num_tours": len(results['all_times']),
            "min": float(np.min(results['all_times'])),
            "max": float(np.max(results['all_times'])),
            "mean": float(np.mean(results['all_times'])),
            "median": float(np.median(results['all_times'])),
            "std": float(np.std(results['all_times']))
        }
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(experiment_data, f, indent=2)
    
    print(f"\nExperiment results saved to: {filename}")
    
    # Print summary to console
    if 'qaoa_summary' in experiment_data["results"]:
        print("\n" + "="*70)
        print("QAOA RUNS SUMMARY")
        print("="*70)
        for label, summary in experiment_data["results"]["qaoa_summary"].items():
            print(f"\n{label}:")
            print(f"  Final Validity: {summary['final_validity_pct']:.1f}%")
            print(f"  Final Cost: {summary['final_best_cost']:.2f}s" if summary['final_best_cost'] else "  Final Cost: N/A")
            print(f"  Improvement: {summary['improvement_pct']:.1f}%")
            print(f"  Avg Validity: {summary['avg_validity_pct']:.1f}%")
            print(f"  Avg Diversity: {summary['avg_diversity']:.1f}")
        
        if 'qaoa_comparison' in experiment_data["results"]:
            comp = experiment_data["results"]["qaoa_comparison"]
            print(f"\n{'='*70}")
            print(f"Best run by cost: {comp['best_run_by_cost']} ({comp['best_final_cost']:.2f}s)")
            print(f"Best run by validity: {comp['best_run_by_validity']}")
            print("="*70)
    
    return str(filename)


def load_experiment_results(filename):
    """
    Load experiment results from JSON file.
    
    Parameters:
    -----------
    filename : str
        Path to saved experiment file
    
    Returns:
    --------
    dict: Experiment data with deserialized graphs
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Deserialize graphs if present
    if 'tour_graphs' in data.get('results', {}):
        data['results']['graphs_dict'] = {
            name: deserialize_graph(graph_data)
            for name, graph_data in data['results']['tour_graphs'].items()
        }
    
    return data


def create_experiment_summary(experiments_dir):
    """
    Create a summary CSV of all experiments in a directory.
    
    Parameters:
    -----------
    experiments_dir : str
        Directory containing experiment JSON files
    
    Returns:
    --------
    pandas.DataFrame: Summary of all experiments
    """
    
    experiments_dir = Path(experiments_dir)
    experiment_files = list(experiments_dir.glob("*.json"))
    
    summaries = []
    
    for filepath in experiment_files:
        try:
            data = load_experiment_results(filepath)
            
            # Extract key metrics
            summary = {
                'experiment_name': data['metadata']['experiment_name'],
                'timestamp': data['metadata']['datetime'],
                'problem': data['metadata']['problem_name'],
            }
            
            # Add hyperparameters
            summary.update(data['hyperparameters'])
            
            # Add results
            for algo, runtime in data['results'].get('runtime_data', {}).items():
                summary[f'runtime_{algo}'] = runtime
            
            for algo, tt in data['results'].get('travel_time_data', {}).items():
                summary[f'travel_time_{algo}'] = tt
            
            # Add QAOA-specific metrics if available
            if 'qaoa_progress' in data['results']:
                for label, progress in data['results']['qaoa_progress'].items():
                    if progress:
                        final_stats = progress[-1]
                        summary[f'{label}_final_valid_pct'] = final_stats.get('valid_percentage', 0)
                        summary[f'{label}_final_best_cost'] = final_stats.get('best_cost', None)
            
            summaries.append(summary)
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    
    df = pd.DataFrame(summaries)
    
    # Save summary
    summary_file = experiments_dir / "experiments_summary.csv"
    df.to_csv(summary_file, index=False)
    print(f"Summary saved to: {summary_file}")
    
    return df


# Example usage
if __name__ == "__main__":
    # Example of saving experiment results
    example_results = {
        'graphs_dict': {},  # Your tour graphs
        'runtime_data': {'QAOA': 45.2, 'Brute-Force': 0.5},
        'tt_data': {'QAOA': 1250, 'Brute-Force': 1180},
        'qaoa_progress': {'QAOA-NN-Zero': []}  # Your QAOA progress data
    }
    
    example_hyperparams = {
        'layers': 3,
        'shots': 10000,
        'qubit_batch_size': 8,
        'inv_penalty_m': 4.5,
        'warm_start': 'nearest_neighbor',
        'exploration_strength': 0.0,
        'initialization_strategy': 'zero'
    }
    
    saved_path = save_experiment_results(
        experiment_name="QAOA_warmstart_test",
        problem_name="4m_10_1",
        results=example_results,
        hyperparameters=example_hyperparams,
        notes="Testing warm-start with zero initialization"
    )
    
    # Load it back
    loaded_data = load_experiment_results(saved_path)
    print(f"\nLoaded experiment: {loaded_data['metadata']['experiment_name']}")
