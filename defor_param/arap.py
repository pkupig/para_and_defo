#arap.py

import numpy as np
from scipy.sparse import coo_matrix
from half_edge import HalfEdge


beta = 50000
inf_for_bfr = 1e20
eps = 1e-6
Uniform = 0
MeanValue = 1
Cotangent_1 = 2
Cotangent_2 = 3

ARAP = 0
ASAP = 1
Hybrid = 2
LSCM = 3
MVC = 4
Laplacian = 1
PARAM = 0
DEFORM = 1


def getLaplace(half_edges, weights, fix, Laplace, func):
    """
    to build Laplace matrix
    """
    triplet_list = []
    row_sum = np.zeros(Laplace.shape[0])
    
    for i in fix:
        row_sum[i] = 2 * beta
    
    for i in range(len(half_edges)):
        a = half_edges[i].Endpoints[0]
        b = half_edges[i].Endpoints[1]
        inv_idx = half_edges[i].InverseIdx
        w = 0
        
        if func == PARAM:
            w = weights[i]
        elif func == DEFORM:
            w = weights[a * Laplace.shape[0] + b]
        
        if inv_idx != -1:
            if func == PARAM:
                w += weights[inv_idx]
            row_sum[a] += w
            triplet_list.append((a, b, -w))
        else:
            row_sum[a] += w
            row_sum[b] += w
            triplet_list.append((a, b, -w))
            triplet_list.append((b, a, -w))
    
    for i in range(Laplace.shape[0]):
        triplet_list.append((i, i, row_sum[i]))
    rows, cols, data = zip(*triplet_list)
    Laplace = coo_matrix((data, (rows, cols)), shape=Laplace.shape).tocsr()
    return Laplace

def getRHS(R, half_edges, weights, fix, fix_vec, RHS, func, is_first):
    """
    to build the right vector
    """
    RHS.fill(0)
    for i in range(len(fix)):
        RHS[fix[i], :] = 2 * beta * fix_vec[i].T
    
    if (func == PARAM and not is_first) or func == DEFORM:
        for i in range(len(half_edges)):
            a = half_edges[i].Endpoints[0]
            b = half_edges[i].Endpoints[1]
            inv_idx = half_edges[i].InverseIdx
            edge_vec = half_edges[i].EdgeVec
            
            if inv_idx != -1:
                if func == PARAM:
                    coeR = weights[i] * R[i // 3]
                    term = coeR @ edge_vec
                    inv_term = weights[inv_idx] * R[inv_idx // 3] @ half_edges[inv_idx].EdgeVec
                    RHS[a, :] += (term - inv_term).T                    
                elif func == DEFORM:
                    term = ( 0.5 * weights[a * RHS.shape[0] + b] * (R[a] + R[b])) @ edge_vec
                    RHS[a, :] += term.T
            else:
                if func == PARAM:
                    coeR1 = weights[i] * R[i // 3]
                    term = coeR1 @ edge_vec
                    RHS[a, :] += term.T
                    RHS[b, :] -= term.T
                elif func == DEFORM:
                    term = 0.5 * weights[a * RHS.shape[0] + b] * (R[a] + R[b]) @ edge_vec
                    RHS[a, :] += term.T
                    RHS[b, :] -= term.T


def local_phase_param(R, half_edges, res, weights, area, method, lamda, distortion_per_unit, aread_per_unit, angled_per_unit, distortion, aread, angled):
    """
    """
    distortion = 0.0
    aread = 0.0
    angled = 0.0

    for i in range(len(half_edges) // 3):  
        J = getJacobian2x2(half_edges, res, i)  
        U, singular_values, Vt = np.linalg.svd(J, full_matrices=False)
        SI = np.zeros((2, 2))
        SI[0, 0] = SI[1, 1] = 0.5 * (singular_values[0] + singular_values[1])

        if method == ARAP or method == ASAP:
            R[i] = U @ Vt
            if method == ASAP:
                R[i] = U @ SI @ Vt
            if np.linalg.det(R[i]) < 0:
                newVt = Vt.copy()
                newVt[1, :] *= -1  
                SI[0, 0] = SI[1, 1] = 0.5 * (singular_values[0] - singular_values[1])

                if method == ARAP:
                    R[i] = U @ newVt
                elif method == ASAP:
                    R[i] = U @ SI @ newVt

        elif method == Hybrid:
            C1, C2, C3 = 0, 0, 0
            for j in range(3):
                v = half_edges[i * 3 + j].EdgeVec
                a, b = half_edges[i * 3 + j].Endpoints
                u = (res[a] - res[b]).T

                C1 += weights[i * 3 + j] * (v[0] ** 2 + v[1] ** 2)
                C2 += weights[i * 3 + j] * (u[0] * v[0] + u[1] * v[1])
                C3 += weights[i * 3 + j] * (u[0] * v[1] - u[1] * v[0])

            def cubic_fun(x):
                return (
                    2 * lamda * (1 + C3 ** 2 / C2 ** 2) * x ** 3
                    + (C1 - 2 * lamda) * x
                    - C2
                )
            entry_1 = binary_find_root(cubic_fun)
            R[i] = np.array([
                [entry_1, entry_1 * C3 / C2],
                [-entry_1 * C3 / C2, entry_1]
            ])

        distortion_per_unit[i] = area[i] * np.trace((J - R[i]).T @ (J - R[i]))
        aread_per_unit[i] = area[i] * (
            singular_values[0] * singular_values[1]
            + 1.0 / (singular_values[0] * singular_values[1])
        )
        angled_per_unit[i] = area[i] * (
            singular_values[0] / singular_values[1]
            + singular_values[1] / singular_values[0]
        )

        distortion += distortion_per_unit[i]
        aread += aread_per_unit[i]
        angled += angled_per_unit[i]
    

def global_phase(R, half_edges, weights, fix, fix_vec, RHS, func, first, res, dir_solver):
    """
    dir_solver: Laplace * res = RHS
    """
    getRHS(R, half_edges, weights, fix, fix_vec, RHS, func, first)
    res[:] = dir_solver.solve(RHS)
    
def local_phase_deform(R, neighbors, verts, res, weights):
    """
    Compute rotation matrices 
    """
    vert_num = verts.shape[1]
    for i in range(vert_num):
        S = getCovariance3x3(neighbors[i], verts, res, weights, i)
        U, S_vals, Vh = np.linalg.svd(S, full_matrices=False)
        R_i = U @ Vh
        det = np.linalg.det(R_i)
        if det < 0:
            smallest_sv = np.inf
            smallest_idx = -1
            for j in range(len(S_vals)):
                if S_vals[j] < smallest_sv:
                    smallest_sv = S_vals[j]
                    smallest_idx = j
            new_Vh = Vh.copy()
            new_Vh[smallest_idx, :] *= -1
            R_i = U @ new_Vh
        R[i] = R_i
        assert np.linalg.det(R[i]) > 0, f"Rotation matrix for vertex {i} has non-positive determinant."

def getCovariance3x3(neighbor, verts, now_res, weights, cur):
    """
    """
    S = np.zeros((3, 3))
    for i in range(len(neighbor)):
        w = weights[cur * verts.shape[1] + neighbor[i]]
        S += w * (now_res[cur] - now_res[neighbor[i]]).T @ (verts[:, cur] - verts[:, neighbor[i]]).T
    return S


def getJacobian2x2(half_edges, now_res, cur):
    """
    """
    JT = np.zeros((2, 2))
    Origin = np.zeros((2, 2))
    RHS = np.zeros((2, 2))

    a = half_edges[cur * 3].Endpoints[0]
    b = half_edges[cur * 3].Endpoints[1]
    c = half_edges[cur * 3 + 1].Endpoints[0]
    d = half_edges[cur * 3 + 1].Endpoints[1]

    Origin[0, :] = half_edges[cur * 3].EdgeVec.T
    Origin[1, :] = half_edges[cur * 3 + 1].EdgeVec.T
    RHS[0, :] = now_res[a] - now_res[b]
    RHS[1, :] = now_res[c] - now_res[d]

    JT = np.linalg.solve(Origin, RHS)
    return JT.T


def getCovariance2x2(half_edges, now_res, weights, cur):
    """
    """
    S = np.zeros((2, 2))
    for i in range(3):
        a = half_edges[cur * 3 + i].Endpoints[0]
        b = half_edges[cur * 3 + i].Endpoints[1]
        U = (now_res[a] - now_res[b]).T
        X = half_edges[cur * 3 + i].EdgeVec
        S += weights[cur * 3 + i] * U @ X.T
    return S


def binary_find_root(fun):
    """
    """
    x = 0
    y = fun(x)
    left, right = 0, 0
    if y < -eps:
        left, right = x, inf_for_bfr
    elif y > eps:
        left, right = -inf_for_bfr, x
    else:
        return x

    while right - left > eps:
        x = (left + right) / 2
        y = fun(x)
        if y < -eps:
            left = x
        elif y > eps:
            right = x
        else:
            return x
        

class ARAP_energy:
    def __init__(self, half_edges, weights, area, method, lamda):
        self.half_edges = half_edges
        self.weights = weights
        self.area = area
        self.method = method
        self.lamda = lamda

    def __call__(self, res):
        """
        """
        E = 0  
        for i in range(len(self.half_edges) // 3):  
            J = getJacobian2x2(self.half_edges, res, i)
            U, singular_values, Vt = np.linalg.svd(J, full_matrices=False)
            SI = np.zeros((2, 2))
            SI[0, 0] = SI[1, 1] = 0.5 * (singular_values[0] + singular_values[1])

            if self.method == ARAP or self.method == ASAP:
                L = U @ Vt
                if self.method == ASAP:
                    L = U @ SI @ Vt
                if np.linalg.det(L) < 0:
                    newVt = Vt.copy()
                    newVt[1, :] *= -1  
                    SI[0, 0] = SI[1, 1] = 0.5 * (singular_values[0] - singular_values[1])
                    if self.method == ARAP:
                        L = U @ newVt
                    elif self.method == ASAP:
                        L = U @ SI @ newVt

            elif self.method == Hybrid:
                C1, C2, C3 = 0, 0, 0
                for j in range(3):
                    v = self.half_edges[i * 3 + j].EdgeVec
                    a, b = self.half_edges[i * 3 + j].Endpoints
                    u = (res[a] - res[b]).T
                    C1 += self.weights[i * 3 + j] * (v[0] ** 2 + v[1] ** 2)
                    C2 += self.weights[i * 3 + j] * (u[0] * v[0] + u[1] * v[1])
                    C3 += self.weights[i * 3 + j] * (u[0] * v[1] - u[1] * v[0])

                def cubic_fun(x):
                    return (
                        2 * self.lamda * (1 + C3 ** 2 / C2 ** 2) * x ** 3
                        + (C1 - 2 * self.lamda) * x
                        - C2
                    )
                entry_1 = binary_find_root(cubic_fun)
                L = np.array([
                    [entry_1, entry_1 * C3 / C2],
                    [-entry_1 * C3 / C2, entry_1]
                ])
            E += self.area[i] * np.trace((J - L).T @ (J - L))

        return E
    