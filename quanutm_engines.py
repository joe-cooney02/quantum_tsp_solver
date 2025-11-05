# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 09:52:41 2025

@author: joeco
"""

'''
To do TSP via quantum computing, the QAOA approach is to use one qubit per edge. (O(N^2))
Then, the edges that are selected are the qubits in the 1 position.
Maybe with an RQAOA-like approach, we could encode 1 qubit per node, and select one at each step.
    This would be like the Next-Nearest heuristic solution, and speedup could only come during the 
    comparison between all edges connected to a node.
    
    Since classical computers are good at finding a minimum from a list (O(n)), 
    the only advantage would be if we could encode extra information that made better choices.
    That total classical heuristic takes O(n^2) time: n + (n-1) + ... 1 ~ n^2 / 2
    
'''


def QAOA_approx(graph, A=10, B=1, p=3, shots=8000, alpha=0.85, maxi=50, tol=0.05):
    '''
    This only works in python 3.8-3.10, not the 3.13 the rest of this is coded with.
    To use this function, you will need to make another conda environment.

    Parameters
    ----------
    graph : TYPE
        DESCRIPTION.
    A : TYPE, optional
        DESCRIPTION. The default is 10.
    B : TYPE, optional
        DESCRIPTION. The default is 1.
    p : TYPE, optional
        DESCRIPTION. The default is 3.
    shots : TYPE, optional
        DESCRIPTION. The default is 8000.
    alpha : TYPE, optional
        DESCRIPTION. The default is 0.85.
    maxi : TYPE, optional
        DESCRIPTION. The default is 50.
    tol : TYPE, optional
        DESCRIPTION. The default is 0.05.

    Returns
    -------
    None.

    '''
    '''
    tsp_prob = TSP(G=graph, A=A, B=B)
    tsp_qubo = tsp_prob.qubo
    
    q = QAOA()
    
    qiskit_sv = create_device(location='local', name='qiskit.qasm_simulator')
    q.set_device(qiskit_sv)
    
    # circuit properties
    q.set_circuit_properties(p=p, param_type='standard', init_type='rand', mixer_hamiltonian='x')

    # backend properties
    q.set_backend_properties(init_hadamard=True, n_shots=shots, cvar_alpha=alpha)

    # classical optimizer properties
    q.set_classical_optimizer(method='cobyla', maxiter=maxi, tol=tol)
    
    q.compile(tsp_qubo)
    q.optimize()
    
    print(q.result.optimized)
    '''
    return None


