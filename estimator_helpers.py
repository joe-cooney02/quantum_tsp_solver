#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 21:27:26 2026

@author: joecooney
"""

"""
GPU-accelerated expectation value calculation for TSP QAOA using EstimatorV2.

The key idea:
  Instead of sampling shots and calling get_cost_expectation() over bitstrings,
  we express the TSP cost Hamiltonian as a sum of Pauli-Z observables and let
  EstimatorV2 compute <ψ|H_cost|ψ> analytically from the statevector on the GPU.
  This removes shot noise entirely (precision=0 → exact statevector expval) and
  avoids the Python-side bitstring loop that dominates wall time at high shot counts.

Observable construction:
  Each qubit k encodes edge (u,v) with weight w_uv.
  The cost Hamiltonian for the TSP routing cost is:

      H_cost = sum_{(u,v) in E}  w_uv * (I - Z_k) / 2

  where (I - Z_k)/2 is the projector onto |1⟩ on qubit k.
  This is a *diagonal* Hamiltonian whose expectation equals the average
  tour cost weighted by the probability that each edge is selected.

  Note: This only captures the routing cost. Constraint violations
  (invalid tours) are not penalised by this Hamiltonian — use the
  `penalty_strength` argument to add a soft Z^2-style penalty that
  pushes the optimiser toward valid-degree subspaces.

Penalty term (optional):
  For each node n, the TSP constraint is that in-degree = out-degree = 1.
  We approximate the degree-violation penalty as:

      H_penalty = P * sum_n [ (sum_{k: qubit k leaves n} Z_k)^2
                             + (sum_{k: qubit k enters n} Z_k)^2 ]

  Expanding the squares gives weight-2 ZZ terms plus single-qubit Z terms,
  all of which are expressible as SparsePauliOp.
"""

from __future__ import annotations

import numpy as np
import networkx as nx

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2


# ---------------------------------------------------------------------------
# Observable construction
# ---------------------------------------------------------------------------

def build_tsp_cost_observable(
    G: nx.DiGraph,
    qubit_to_edge_map: dict,
    penalty_strength: float = 0.0,
) -> SparsePauliOp:
    """
    Build the TSP cost Hamiltonian as a SparsePauliOp.

    Parameters
    ----------
    G : nx.DiGraph
        TSP graph with 'weight' attributes on edges.
    qubit_to_edge_map : dict
        {qubit_index: (u, v)} mapping produced by create_qubit_to_edge_map().
    penalty_strength : float
        Coefficient P for the degree-violation penalty Hamiltonian.
        Set to 0 to use pure cost (no penalty). A good starting value when
        enabling the penalty is ~(max_edge_weight * num_nodes).

    Returns
    -------
    SparsePauliOp
        Hermitian observable whose expectation value equals the average
        TSP routing cost (plus penalty, if requested) under the circuit's
        output state.
    """
    num_qubits = len(qubit_to_edge_map)
    edge_to_qubit = {edge: q for q, edge in qubit_to_edge_map.items()}

    pauli_list: list[tuple[str, complex]] = []

    # ------------------------------------------------------------------
    # Cost part:  H_cost = sum_{(u,v)} w_uv * (I - Z_k) / 2
    # ------------------------------------------------------------------
    # Constant offset from the "I" terms (added as identity Pauli)
    cost_offset = 0.0

    for qubit_idx, (u, v) in qubit_to_edge_map.items():
        if G.has_edge(u, v):
            w = G[u][v]["weight"]
            cost_offset += w / 2.0
            # -w/2 * Z_k
            z_str = _single_z_pauli(qubit_idx, num_qubits)
            pauli_list.append((z_str, -w / 2.0))

    # Add the constant identity term
    identity_str = "I" * num_qubits
    pauli_list.append((identity_str, cost_offset))

    # ------------------------------------------------------------------
    # Penalty part (optional):
    #   For each node n, penalise (out-degree deviation)^2 + (in-degree deviation)^2.
    #
    #   Let S_out(n) = sum_{k: edge k leaves n} (I - Z_k)/2   (= out-degree operator)
    #   Penalty = P * sum_n [ (S_out(n) - 1)^2 + (S_in(n) - 1)^2 ]
    #
    #   (S - 1)^2 = S^2 - 2S + I
    #   S^2 involves ZZ and Z terms; we expand fully below.
    # ------------------------------------------------------------------
    if penalty_strength != 0.0:
        for node in G.nodes():
            # Qubits for edges leaving / entering this node
            out_qubits = [
                edge_to_qubit[(node, v)]
                for v in G.successors(node)
                if (node, v) in edge_to_qubit
            ]
            in_qubits = [
                edge_to_qubit[(u, node)]
                for u in G.predecessors(node)
                if (u, node) in edge_to_qubit
            ]

            for qubit_group in (out_qubits, in_qubits):
                # (S - 1)^2 where S = sum_k (I - Z_k)/2
                # Expand using algebra; collect coefficients of I, Z_k, Z_j Z_k
                _add_degree_penalty_terms(
                    pauli_list, qubit_group, num_qubits, penalty_strength
                )

    # ------------------------------------------------------------------
    # Combine into a single SparsePauliOp (Qiskit uses little-endian qubit order)
    # ------------------------------------------------------------------
    if not pauli_list:
        # Edge case: empty graph
        return SparsePauliOp.from_list([("I" * num_qubits, 0.0)])

    observable = SparsePauliOp.from_list(pauli_list).simplify()
    return observable


def _single_z_pauli(qubit_idx: int, num_qubits: int) -> str:
    """
    Return a Pauli string with Z on qubit_idx and I elsewhere.
    Qiskit SparsePauliOp uses little-endian ordering (qubit 0 is rightmost).
    """
    chars = ["I"] * num_qubits
    chars[num_qubits - 1 - qubit_idx] = "Z"
    return "".join(chars)


def _add_degree_penalty_terms(
    pauli_list: list,
    qubits: list[int],
    num_qubits: int,
    P: float,
) -> None:
    """
    Expand P * (S - 1)^2 where S = sum_{k in qubits} (I - Z_k)/2
    and accumulate Pauli terms into pauli_list.

    S       = (n/2)*I  -  (1/2)*sum_k Z_k          (n = len(qubits))
    S^2     = n^2/4 * I  - n/2 * sum_k Z_k  + 1/4 * sum_{j,k} Z_j Z_k
    (S-1)^2 = S^2 - 2S + I
            = (n^2/4 - n + 1)*I + (n/2 - 1)* sum_k Z_k + 1/4 * sum_{j≠k} Z_j Z_k + n/4 * I
    Simplified:
            = (n^2/4 - n + 1 + n/4)*I + (n/2 - 1)*sum_k Z_k + 1/4 * sum_{j<k} 2*Z_jZ_k + 1/4*n*I
    We just expand term by term to keep the code readable.
    """
    n = len(qubits)
    if n == 0:
        return

    identity_str = "I" * num_qubits

    # Constant from (n/2*I - sum Z/2 - 1)^2
    # Coefficient of I from S^2: n^2/4
    # From -2S: adds 2*n/2 = n to constant (from -2 * n/2 * I)
    # From +I: +1
    # Total I coeff = n^2/4 - n + 1
    const_coeff = (n ** 2) / 4.0 - n + 1.0
    pauli_list.append((identity_str, P * const_coeff))

    # Linear Z terms from S^2 (-n/2 * sum Z_k) and -2S (+sum Z_k)
    # Net: (-n/2 + 1) per Z_k
    z_coeff = -n / 2.0 + 1.0
    for k in qubits:
        z_str = _single_z_pauli(k, num_qubits)
        pauli_list.append((z_str, P * z_coeff))

    # ZZ terms from S^2: 1/4 * sum_{j,k} Z_j Z_k (j != k gives factor 2 for j<k)
    for i, j in zip(*np.triu_indices(n, k=1)):  # upper triangle j < k
        qi, qj = qubits[i], qubits[j]
        zz_str = _zz_pauli(qi, qj, num_qubits)
        pauli_list.append((zz_str, P * 0.5))  # 2 * 1/4 = 1/2 for each pair


def _zz_pauli(q1: int, q2: int, num_qubits: int) -> str:
    """Return Pauli string with Z on q1 and q2, I elsewhere (little-endian)."""
    chars = ["I"] * num_qubits
    chars[num_qubits - 1 - q1] = "Z"
    chars[num_qubits - 1 - q2] = "Z"
    return "".join(chars)


# ---------------------------------------------------------------------------
# Estimator-based expectation value
# ---------------------------------------------------------------------------

def get_cost_expectation_estimator(
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
    estimator_options: dict | None = None,
) -> float:
    """
    Compute <circuit|observable|circuit> using Aer's EstimatorV2 (GPU-ready).

    EstimatorV2 accepts Primitive Unified Blocs (PUBs): tuples of
    (circuit, observables, parameter_values, precision).

    Setting precision=0 triggers exact statevector expectation value —
    no shot sampling — which is the key speedup for high-shot experiments.

    Parameters
    ----------
    circuit : QuantumCircuit
        A fully-bound (no free Parameters) quantum circuit WITHOUT measurements.
        Use bind_qaoa_parameters() from quantum_helpers.py first, then remove
        measurements if any (EstimatorV2 does not accept measurement gates).
    observable : SparsePauliOp
        The cost Hamiltonian returned by build_tsp_cost_observable().
    estimator_options : dict, optional
        Options forwarded to EstimatorV2. Defaults to GPU statevector with
        cuStateVec and single precision. Pass an empty dict to use CPU defaults.

    Returns
    -------
    float
        The exact expectation value <ψ|H_cost|ψ>.

    Example
    -------
    >>> obs = build_tsp_cost_observable(G, qubit_to_edge_map, penalty_strength=5.0)
    >>> bound_qc = bind_qaoa_parameters(qc, gamma, beta, theta)
    >>> qc_no_meas = remove_measurements(bound_qc)
    >>> cost = get_cost_expectation_estimator(qc_no_meas, obs)
    """
    if estimator_options is None:
        estimator_options = _default_gpu_options()

    estimator = EstimatorV2(options=estimator_options)

    # PUB = (circuit, observables, parameter_values=None, precision=0)
    # precision=0 → exact statevector expval (no shot noise)
    pub = (circuit, [observable], None, 0)

    job = estimator.run([pub])
    result = job.result()

    # result[0] is the PubResult for the first (only) PUB
    # .data.evs contains the expectation values array; we want the first element
    expval = float(result[0].data.evs[0])
    return expval


def _default_gpu_options() -> dict:
    """
    Return EstimatorV2 options that enable GPU-accelerated statevector simulation.

    Key settings:
    - device='GPU'            → use CUDA-enabled AerSimulator backend
    - cuStateVec_enable=True  → use NVIDIA cuStateVec library (faster on Hopper/Ampere)
    - precision='single'      → FP32 statevector fits more qubits in VRAM
    - batched_shots_gpu=True  → keep shots on-device (irrelevant for precision=0
                                 but good practice for mixed workflows)
    """
    return {
        "backend_options": {
            "method": "statevector",
            "device": "GPU",
            "precision": "single",
            "cuStateVec_enable": True,
            "batched_shots_gpu": True,
            "max_parallel_experiments": 1,  # EstimatorV2 handles batching internally
        }
    }


def _default_cpu_options() -> dict:
    """Fallback options for CPU simulation (useful for testing)."""
    return {
        "backend_options": {
            "method": "statevector",
            "device": "CPU",
            "precision": "double",
            "fusion_enable": True,
        }
    }


# ---------------------------------------------------------------------------
# Utility: strip measurements from a circuit
# ---------------------------------------------------------------------------

def remove_measurements(circuit: QuantumCircuit) -> QuantumCircuit:
    """
    Return a copy of `circuit` with all measurement (and reset) instructions removed.

    EstimatorV2 requires circuits WITHOUT measurements; it computes the
    expectation value directly from the statevector before any collapse.

    Parameters
    ----------
    circuit : QuantumCircuit
        The circuit to strip (may contain or lack measurements).

    Returns
    -------
    QuantumCircuit
        A new circuit with the same gates but no measurement operations.
    """
    qc_copy = circuit.copy()
    qc_copy.remove_final_measurements(inplace=True)
    # remove_final_measurements only removes trailing ones; drop any remaining
    data_filtered = [
        instr for instr in qc_copy.data
        if instr.operation.name not in ("measure", "reset", "barrier")
    ]
    qc_out = QuantumCircuit(circuit.num_qubits)
    for instr in data_filtered:
        qc_out.append(instr.operation, instr.qubits, instr.clbits)
    return qc_out


# ---------------------------------------------------------------------------
# Drop-in wrapper: same signature as get_cost_expectation() in quantum_helpers
# ---------------------------------------------------------------------------

def get_cost_expectation_gpu(
    circuit: QuantumCircuit,
    G: nx.DiGraph,
    qubit_to_edge_map: dict,
    penalty_strength: float = 0.0,
    use_gpu: bool = True,
) -> float:
    """
    High-level drop-in for get_cost_expectation() using EstimatorV2.

    Compared to the shot-based version this function:
      1. Builds the cost+penalty Hamiltonian as a SparsePauliOp once.
      2. Runs EstimatorV2 with precision=0 (exact statevector expval).
      3. Returns the scalar expectation value — no shot noise, no bitstring loop.

    Call this from your QAOA optimisation loop wherever you previously called
    get_cost_expectation(bitstrings, counts, qubit_to_edge_map, G, ...).

    Parameters
    ----------
    circuit : QuantumCircuit
        Fully-bound QAOA circuit (parameters assigned, measurements OK — they
        will be stripped automatically).
    G : nx.DiGraph
        TSP graph with edge weights.
    qubit_to_edge_map : dict
        {qubit_index: (u, v)} from create_qubit_to_edge_map().
    penalty_strength : float
        Degree-violation penalty coefficient (see build_tsp_cost_observable).
        Start with 0 and tune upward if the optimiser finds too many invalid tours.
    use_gpu : bool
        If True, use GPU options; if False, fall back to CPU (handy for testing).

    Returns
    -------
    float
        Exact expectation value of the TSP cost Hamiltonian.

    Example (inside QAOA optimisation loop)
    ----------------------------------------
    >>> def objective(params):
    ...     gamma, beta, theta = unpack(params)
    ...     bound_qc = bind_qaoa_parameters(qaoa_circuit, gamma, beta, theta)
    ...     return get_cost_expectation_gpu(bound_qc, G, qubit_to_edge_map,
    ...                                     penalty_strength=5.0)
    >>> result = scipy.optimize.minimize(objective, x0, method='COBYLA')
    """
    observable = build_tsp_cost_observable(G, qubit_to_edge_map, penalty_strength)
    qc_no_meas = remove_measurements(circuit)
    options = _default_gpu_options() if use_gpu else _default_cpu_options()
    return get_cost_expectation_estimator(qc_no_meas, observable, options)


# ---------------------------------------------------------------------------
# Batched variant: evaluate multiple parameter sets in one EstimatorV2 call
# ---------------------------------------------------------------------------

def get_cost_expectations_batched(
    circuits: list[QuantumCircuit],
    G: nx.DiGraph,
    qubit_to_edge_map: dict,
    penalty_strength: float = 0.0,
    use_gpu: bool = True,
) -> list[float]:
    """
    Evaluate expectation values for a *batch* of circuits in a single EstimatorV2 call.

    This is the most GPU-efficient pattern: all circuits share one observable and
    are dispatched together, letting cuStateVec amortise kernel launch overhead.
    Useful for gradient estimation (parameter shift rule) or population-based
    optimisers (CMA-ES, differential evolution) where you evaluate many candidates
    per iteration.

    Parameters
    ----------
    circuits : list[QuantumCircuit]
        List of fully-bound circuits (measurements will be stripped).
    G, qubit_to_edge_map, penalty_strength, use_gpu
        Same as get_cost_expectation_gpu().

    Returns
    -------
    list[float]
        Expectation values in the same order as `circuits`.

    Example (gradient via parameter shift)
    ----------------------------------------
    >>> shift = np.pi / 2
    >>> shifted_circuits = [
    ...     bind_qaoa_parameters(qc, gamma + shift*e_i, beta, theta),
    ...     bind_qaoa_parameters(qc, gamma - shift*e_i, beta, theta),
    ... ]
    >>> e_plus, e_minus = get_cost_expectations_batched(shifted_circuits, G, q2e)
    >>> gradient_i = (e_plus - e_minus) / 2
    """
    observable = build_tsp_cost_observable(G, qubit_to_edge_map, penalty_strength)
    options = _default_gpu_options() if use_gpu else _default_cpu_options()
    estimator = EstimatorV2(options=options)

    pubs = [
        (remove_measurements(qc), [observable], None, 0)
        for qc in circuits
    ]

    job = estimator.run(pubs)
    results = job.result()

    return [float(results[i].data.evs[0]) for i in range(len(circuits))]