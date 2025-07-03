# half_edge.py

import numpy as np

class HalfEdge:
    def __init__(self):
        self.Endpoints = np.zeros(2, dtype=int)  
        self.EdgeVec = [0.0, 0.0, 0.0]              
        self.OppositePoint = 0              
        self.BelongFacet = 0                 
        self.InverseIdx = -1                   

    def __lt__(self, other):
        if self.Endpoints[0] != other.Endpoints[0]:
            return self.Endpoints[0] < other.Endpoints[0]
        return self.Endpoints[1] < other.Endpoints[1]