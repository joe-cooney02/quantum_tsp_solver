# Soft Validity Measures for QAOA Cost Function

## Overview

This module provides computationally efficient soft validity measures that can be used in the QAOA cost function to create a smoother optimization landscape, avoiding barren plateaus of invalid solutions.

## Key Insight

For TSP, valid solutions have a specific structure:
- Each node must have exactly 1 incoming and 1 outgoing edge
- Total of N edges selected (for N nodes)
- Forms a single connected cycle

Rather than binary valid/invalid, we measure **how many violations** a bitstring has.

## Proposed Measure: Violation Score

```python
def compute_soft_validity_score(bitstring, qubit_to_edge_map, graph):
    """
    Compute a soft validity score for a bitstring.
    
    Returns a violation score where:
    - 0.0 = perfectly valid tour
    - Higher values = more violations
    
    This can be used in the cost function to create gradients toward validity.
    """
    
    # Get selected edges
    selected_edges = []
    for qubit_idx, bit in enumerate(bitstring):
        if bit == '1' and qubit_idx < len(qubit_to_edge_map):
            selected_edges.append(qubit_to_edge_map[qubit_idx])
    
    num_nodes = graph.number_of_nodes()
    
    # Violation 1: Wrong number of edges
    # Should have exactly N edges for N nodes
    edge_count_violation = abs(len(selected_edges) - num_nodes)
    
    # Violation 2: Degree violations
    # Each node should have in-degree = 1 and out-degree = 1
    in_degrees = {node: 0 for node in graph.nodes()}
    out_degrees = {node: 0 for node in graph.nodes()}
    
    for u, v in selected_edges:
        out_degrees[u] += 1
        in_degrees[v] += 1
    
    degree_violations = 0
    for node in graph.nodes():
        degree_violations += abs(in_degrees[node] - 1)  # Should be 1
        degree_violations += abs(out_degrees[node] - 1)  # Should be 1
    
    # Violation 3: Disconnection penalty
    # Count number of connected components
    # (Only compute if we have the right number of edges and no degree violations)
    disconnection_penalty = 0
    if edge_count_violation == 0 and degree_violations == 0:
        # Create subgraph from selected edges
        import networkx as nx
        tour_graph = nx.DiGraph()
        tour_graph.add_edges_from(selected_edges)
        
        # Count strongly connected components
        num_components = len(list(nx.strongly_connected_components(tour_graph)))
        disconnection_penalty = num_components - 1  # Should be 1 component
    
    # Total violation score
    # Weight the violations appropriately
    total_violation = (
        edge_count_violation * 1.0 +      # Each wrong edge counts as 1
        degree_violations * 0.5 +          # Each degree violation counts as 0.5
        disconnection_penalty * 2.0        # Disconnection is more severe
    )
    
    return total_violation
```

## Alternative: Faster Approximate Measure

For even faster computation (skip disconnection check):

```python
def compute_fast_validity_score(bitstring, qubit_to_edge_map, graph):
    """
    Fast validity score using only degree constraints.
    
    Computes:
    - Edge count violation
    - In/out degree violations
    
    Skips disconnection check (expensive for large graphs).
    """
    
    selected_edges = []
    for qubit_idx, bit in enumerate(bitstring):
        if bit == '1' and qubit_idx < len(qubit_to_edge_map):
            selected_edges.append(qubit_to_edge_map[qubit_idx])
    
    num_nodes = graph.number_of_nodes()
    edge_count_violation = abs(len(selected_edges) - num_nodes)
    
    in_degrees = {node: 0 for node in graph.nodes()}
    out_degrees = {node: 0 for node in graph.nodes()}
    
    for u, v in selected_edges:
        out_degrees[u] += 1
        in_degrees[v] += 1
    
    degree_violations = sum(
        abs(in_degrees[node] - 1) + abs(out_degrees[node] - 1)
        for node in graph.nodes()
    )
    
    return edge_count_violation + 0.5 * degree_violations
```

## Using in Cost Function

### Current Cost Function
```python
def get_cost_expectation(bitstrings, counts, qubit_to_edge_map, graph, inv_penalty=0):
    total_cost = 0
    total_shots = sum(counts.values())
    
    for bitstring in bitstrings:
        is_valid, tour = is_valid_tsp_tour(bitstring, qubit_to_edge_map, graph, return_tour=True)
        
        if is_valid:
            cost = sum(graph[tour[i]][tour[i+1]]['weight'] for i in range(len(tour)-1))
            total_cost += cost * counts[bitstring]
        else:
            total_cost += inv_penalty * counts[bitstring]  # Hard penalty
    
    return total_cost / total_shots
```

### Enhanced Cost Function with Soft Validity
```python
def get_cost_expectation_soft(bitstrings, counts, qubit_to_edge_map, graph, 
                              max_edge_weight, base_penalty=10.0, use_fast=True):
    """
    Cost function with soft validity penalties.
    
    Parameters:
    -----------
    max_edge_weight : float
        Maximum edge weight in graph (for scaling)
    base_penalty : float
        Penalty multiplier (higher = stronger push toward validity)
    use_fast : bool
        If True, use fast approximation (no disconnection check)
    """
    total_cost = 0
    total_shots = sum(counts.values())
    
    validity_fn = compute_fast_validity_score if use_fast else compute_soft_validity_score
    
    for bitstring in bitstrings:
        is_valid, tour = is_valid_tsp_tour(bitstring, qubit_to_edge_map, graph, return_tour=True)
        
        if is_valid:
            # Valid tour: use actual cost
            cost = sum(graph[tour[i]][tour[i+1]]['weight'] for i in range(len(tour)-1))
        else:
            # Invalid tour: penalize based on violation severity
            violation_score = validity_fn(bitstring, qubit_to_edge_map, graph)
            
            # Penalty scales with violation score
            # Use max_edge_weight for scaling to make penalties comparable to tour costs
            cost = base_penalty * max_edge_weight * (1 + violation_score)
        
        total_cost += cost * counts[bitstring]
    
    return total_cost / total_shots
```

## Complexity Analysis

| Measure | Time Complexity | Notes |
|---------|----------------|-------|
| **Binary (current)** | O(E + N) | Fast but no gradient |
| **Soft (full)** | O(E + N + C) | C = connected components (small) |
| **Soft (fast)** | O(E + N) | Same as binary, but gives gradient |

Where:
- E = number of selected edges (~N for valid tours)
- N = number of nodes
- C = cost of finding connected components (typically very small)

**Recommendation**: Use fast version during optimization, full version for debugging.

## Why This Helps

### Current Problem
```
Invalid bitstrings → All get same penalty (inv_penalty)
→ Flat landscape (barren plateau)
→ Optimizer has no gradient to follow
```

### With Soft Validity
```
Invalid bitstrings → Get different penalties based on violations
→ Gradient toward validity
→ Optimizer can "climb down" the violation landscape

Example:
- 10 edges, 5 degree violations → High penalty
- 9 edges, 3 degree violations → Medium penalty  
- 8 edges, 1 degree violation → Low penalty
- 8 edges, 0 degree violations, disconnected → Medium penalty
- Valid tour → Actual tour cost
```

## Implementation Strategy

### Step 1: Add Soft Validity Function

Add to `quantum_helpers.py`:

```python
def compute_soft_validity_score(bitstring, qubit_to_edge_map, graph, fast=True):
    # Implementation as above
    pass
```

### Step 2: Modify Cost Function

Update `get_cost_expectation()` to accept a `soft_penalty` parameter:

```python
def get_cost_expectation(bitstrings, counts, qubit_to_edge_map, graph, 
                        inv_penalty=0, soft_penalty=False):
    if soft_penalty:
        return get_cost_expectation_soft(bitstrings, counts, ...)
    else:
        return get_cost_expectation_hard(bitstrings, counts, ...)
```

### Step 3: Add to QAOA_approx

```python
def QAOA_approx(..., use_soft_validity=False, validity_penalty_strength=10.0):
    # ...
    expectation_val = get_cost_expectation(
        bitstrings, counts, qubit_to_edge_map, graph,
        inv_penalty=inv_penalty,
        soft_penalty=use_soft_validity,
        penalty_strength=validity_penalty_strength
    )
```

## Advanced: Adaptive Penalty

For even better results, adaptively adjust penalty during optimization:

```python
def adaptive_penalty(iteration, max_iterations, initial_penalty=5.0, final_penalty=20.0):
    """
    Start with low penalty (explore invalid space)
    End with high penalty (enforce validity)
    """
    progress = iteration / max_iterations
    return initial_penalty + (final_penalty - initial_penalty) * progress
```

This creates an "annealing" effect: early exploration → later exploitation.

## Tuning Recommendations

1. **Start with fast version**: `use_fast=True` for initial experiments
2. **Tune penalty strength**: Try `base_penalty` in range [5.0, 20.0]
3. **Compare to hard penalty**: Measure if soft penalty increases validity
4. **Adaptive if needed**: If still stuck, try adaptive penalty schedule

## Expected Results

With soft validity penalties:
- ✅ Smoother optimization landscape
- ✅ More iterations before convergence (good sign!)
- ✅ Higher validity rates (hopefully!)
- ✅ Better final tour costs
- ⚠️ Slightly slower per iteration (extra computation)

The extra computation per iteration should be negligible compared to the benefit of escaping barren plateaus.
