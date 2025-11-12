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


def QAOA_approx(graph, circuit, shots):
    '''
    
    '''
    return None


