# Quantum Network Optimization - Code Structure Chart

## Project Overview
TSP solver comparing classical and quantum (QAOA) algorithms using real Google Maps travel time data.

---

## Module Hierarchy

```
main.py (Entry Point)
│
├── google_maps.py (Data Acquisition)
│   ├── get_travel_time_matrix()
│   ├── get_address_set()
│   └── get_directions_matrix()
│
├── optimization_engines.py (Classical Algorithms)
│   ├── tsp_brute_force()
│   ├── Heuristic_next_closest()
│   ├── Heuristic_weighted_next_closest()
│   └── SA_approx()
│
├── quantum_engines.py (Quantum Algorithms)
│   ├── QAOA_approx()
│   └── run_QAOA()
│
├── quantum_helpers.py (QAOA Support Functions)
│   ├── Circuit Construction
│   │   ├── create_tsp_qaoa_circuit()
│   │   ├── create_warm_started_qaoa()
│   │   ├── create_qubit_to_edge_map()
│   │   └── create_edge_to_qubit_map()
│   │
│   ├── Circuit Management
│   │   ├── bind_qaoa_parameters()
│   │   ├── get_initial_parameters()
│   │   ├── save_qaoa_circuit()
│   │   └── load_qaoa_circuit()
│   │
│   ├── Simulation
│   │   ├── split_circuit_for_simulation()
│   │   ├── simulate_split_circuits()
│   │   └── simulate_large_circuit_in_batches()
│   │
│   ├── Validation & Analysis
│   │   ├── is_valid_tsp_tour()
│   │   ├── extract_tour_from_edges()
│   │   ├── filter_valid_shots()
│   │   ├── postselect_best_tour()
│   │   ├── count_valid_invalid()
│   │   ├── get_best_cost()
│   │   ├── get_cost_expectation()
│   │   └── get_qaoa_statistics()
│   │
│   └── Bitstring Operations
│       ├── tour_to_bitstring()
│       ├── bitstring_to_tour()
│       └── hamming_distance()
│
├── opt_helpers.py (Shared Utilities)
│   ├── get_trip_time()
│   ├── tour_to_graph()
│   ├── graphs_to_tours()
│   └── get_warm_start_tour()
│
├── visualization_algorithms.py (All Visualizations)
│   ├── Tour Visualizations
│   │   ├── plot_tour_comparison()
│   │   ├── plot_route_on_map()
│   │   └── plot_multiple_routes_comparison()
│   │
│   ├── Performance Visualizations
│   │   ├── plot_runtime_comparison()
│   │   ├── plot_travel_times_violin()
│   │   └── plot_edge_weight_heatmap()
│   │
│   ├── QAOA Progress Visualizations
│   │   ├── plot_qaoa_comprehensive_progress()
│   │   ├── plot_qaoa_cost_convergence()
│   │   ├── plot_qaoa_validity_progress()
│   │   └── plot_qaoa_validity_pie()
│   │
│   ├── Solution Analysis
│   │   ├── plot_valid_solution_hamming_distances()
│   │   ├── plot_hamming_distance_histogram()
│   │   └── plot_benchmark_results()
│   │
│   └── Helper Functions
│       ├── compare_tour_orders()
│       ├── extract_tour_from_graph()
│       └── extract_route_coordinates()
│
└── quantum_pretraining.py (NEW - Option 2 Implementation)
    └── [To be implemented]

```

---

## Data Flow Diagram

```
Google Maps API
    ↓
[google_maps.py]
    ↓
Travel Time Matrix + Directions + Addresses
    ↓
NetworkX DiGraph (base_graph)
    ↓
    ├─→ [optimization_engines.py] → Classical Solutions
    │       ↓
    │   Tour DiGraphs
    │
    └─→ [quantum_engines.py] → QAOA Solution
            ↓
        [quantum_helpers.py]
            ↓
        - Circuit Creation
        - Simulation (batched)
        - Validation
        - Statistics Collection
            ↓
        Tour DiGraph + Progress Data
    
All Solutions
    ↓
[opt_helpers.py] → Convert formats
    ↓
[visualization_algorithms.py] → Generate plots
    ↓
Saved visualizations & data files
```

---

## Key Data Structures

### Input Data
- **`travel_time_matrix`**: 2D list of travel times (seconds)
- **`addresses`**: List of address strings
- **`directions_matrix`**: 2D list of Google Maps API responses

### Graph Representations
- **`base_graph`**: NetworkX DiGraph with all possible edges
- **`tour_graph`**: NetworkX DiGraph with only tour edges (one in/out per node)

### QAOA-Specific
- **`qubit_to_edge_map`**: Dict mapping `{qubit_idx: (u, v)}`
- **`bitstring`**: String of '0'/'1' representing edge selections
- **`counts`**: Dict mapping `{bitstring: occurrence_count}`
- **`qaoa_stats`**: Dict with keys: `iteration`, `valid_shots`, `invalid_shots`, `best_cost`, `best_tour`, etc.

### Output Data
- **`graphs_dict`**: Dict mapping `{algorithm_name: tour_graph}`
- **`runtime_data`**: Dict mapping `{algorithm_name: runtime_seconds}`
- **`tt_data`**: Dict mapping `{algorithm_name: total_travel_time}`
- **`qaoa_progress`**: Dict mapping `{label: [qaoa_stats_list]}`

---

## Function Dependencies

### Core Optimization Flow
```
main.py
├─ tsp_brute_force() → (graphs_dict, runtime_data, tt_data, valid_tours, all_times)
├─ Heuristic_next_closest() → (graphs_dict, runtime_data, tt_data)
├─ SA_approx() → (graphs_dict, runtime_data, tt_data)
└─ QAOA_approx() → (graphs_dict, runtime_data, tt_data, qaoa_progress)
    └─ run_QAOA() (called iteratively by scipy.minimize)
        ├─ bind_qaoa_parameters()
        ├─ simulate_large_circuit_in_batches()
        │   ├─ split_circuit_for_simulation()
        │   └─ simulate_split_circuits()
        ├─ get_cost_expectation()
        └─ get_qaoa_statistics()
            └─ postselect_best_tour()
                ├─ is_valid_tsp_tour()
                └─ extract_tour_from_edges()
```

### Visualization Flow
```
main.py
├─ graphs_to_tours() → tours_dict
├─ plot_tour_comparison(graphs_dict)
├─ plot_multiple_routes_comparison(tours_dict, addresses, dirs_mat)
├─ plot_runtime_comparison(runtime_data)
├─ plot_travel_times_violin(all_times, labeled_points=tt_data)
├─ plot_edge_weight_heatmap(base_graph)
├─ plot_qaoa_comprehensive_progress(qaoa_progress)
├─ plot_valid_solution_hamming_distances(valid_tours, qubit_to_edge_map, graph)
└─ plot_hamming_distance_histogram(valid_tours, qubit_to_edge_map, greedy_tour)
```

---

## External Dependencies

### Python Libraries
- **qiskit**: Quantum circuit construction
- **qiskit_aer**: Quantum circuit simulation
- **networkx**: Graph data structures and algorithms
- **scipy**: Optimization (minimize function for QAOA)
- **numpy**: Numerical operations
- **matplotlib**: Plotting
- **googlemaps**: Google Maps API client
- **polyline**: Decode Google Maps route polylines
- **contextily**: Map tile backgrounds

### API Keys Required
- **Google Maps API**: Set via `GMAPS_API_KEY` environment variable

---

## File I/O

### Input Files (per problem instance in `{problem_name}/` folder)
- `address-set.txt`: List of addresses (one per line)
- `ttm.txt`: Travel time matrix (CSV format)
- `dir-mat.json`: Directions matrix (JSON)

### Output Files
- `tours.json`: Dictionary of tours by algorithm
- `runtimes.json`: Runtime data by algorithm
- `travel-times.json`: Travel time results by algorithm
- `all-travel-times.json`: All valid tour times from brute force
- `*.png`: Various visualization plots
- `tsp_qaoa.qpy`: Saved QAOA circuit

---

## Configuration Parameters

### QAOA Hyperparameters (in `quantum_engines.py`)
- **`layers`**: Number of QAOA layers (default: 3)
- **`shots`**: Number of measurement shots (default: based on graph size)
- **`qubit_batch_size`**: Qubits per simulation batch (default: 8)
- **`inv_penalty_m`**: Invalid solution penalty multiplier (default: 4.5)
- **`sim_method`**: Qiskit simulation method (default: 'statevector')
- **`warm_start`**: Warm-start method ('nearest_neighbor', 'random', or None)
- **`exploration_strength`**: Warm-start exploration parameter (default: 0.0)
- **`initialization_strategy`**: Parameter initialization ('zero', 'random', 'linear', 'tqa')

---

## Future Modules

### quantum_pretraining.py (To be implemented)
Purpose: Pre-train QAOA layers to improve validity rate
- Layer 0: Train to maximize validity (ignore cost)
- Layers 1+: Train to minimize cost (assuming validity from layer 0)

---

*Last Updated: January 2026*
*Maintained by: Joe Cooney*
