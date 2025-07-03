# graph_process.py

import numpy as np


Uniform = 0
# MeanValue = 1
Cotangent_1 = 2
Cotangent_2 = 3



def getWeights(half_edges, verts, weights, Type):
    """
    """
    if Type == Uniform:
        weights.fill(1.0)
    elif Type == Cotangent_1:
        weights.fill(0.0)
        for i in range(len(half_edges)):
            c = half_edges[i].OppositePoint
            a, b = half_edges[i].Endpoints
            edge_a = verts[c] - verts[a]
            edge_b = verts[c] - verts[b]
            
            norm_a = np.linalg.norm(edge_a)
            norm_b = np.linalg.norm(edge_b)
            if norm_a < 1e-8 or norm_b < 1e-8:
                weights[i] = 0.0
                continue
            edge_a /= norm_a
            edge_b /= norm_b
            
            cos_theta = np.dot(edge_a, edge_b)
            denominator = np.sqrt(1 - cos_theta**2)
            if denominator < 1e-8:
                weights[i] = 0.0
            else:
                weights[i] = cos_theta / denominator
    elif Type == Cotangent_2:
        weights.fill(0.0)
        for i in range(len(half_edges)):
            c = half_edges[i].OppositePoint
            inv_idx = half_edges[i].InverseIdx
            a, b = half_edges[i].Endpoints
            
            edge_a = verts[c] - verts[a]
            edge_b = verts[c] - verts[b]
            norm_a = np.linalg.norm(edge_a)
            norm_b = np.linalg.norm(edge_b)
            if norm_a < 1e-8 or norm_b < 1e-8:
                current_val = 0.0
            else:
                edge_a /= norm_a
                edge_b /= norm_b
                cos_theta = abs(np.dot(edge_a, edge_b))
                current_val = 0.5 * cos_theta / np.sqrt(1 - cos_theta**2)
            weights[i] += current_val
            
            # inverse_edge
            if inv_idx != -1:
                c_inv = half_edges[inv_idx].OppositePoint
                edge_a_inv = verts[c_inv] - verts[a]
                edge_b_inv = verts[c_inv] - verts[b]
                norm_a_inv = np.linalg.norm(edge_a_inv)
                norm_b_inv = np.linalg.norm(edge_b_inv)
                if norm_a_inv < 1e-8 or norm_b_inv < 1e-8:
                    current_val_inv = 0.0
                else:
                    edge_a_inv /= norm_a_inv
                    edge_b_inv /= norm_b_inv
                    cos_theta_inv = abs(np.dot(edge_a_inv, edge_b_inv))
                    current_val_inv = 0.5 * cos_theta_inv / np.sqrt(1 - cos_theta_inv**2)
                weights[i] += current_val_inv
                
            # bound_edge
            if inv_idx == -1:
                weights[b * verts.shape[0] + a] = weights[i] 
                pass
            


def mapTo2DBoundary(verts, boundary_points, fix_vec, alpha):
    """
    """
    L = 0.0
    bsize = len(boundary_points)
    Lk = np.zeros(bsize)
    
    for i in range(bsize):
        curr_idx = boundary_points[i]
        next_idx = boundary_points[(i + 1) % bsize]
        curr_vert = verts[curr_idx]
        next_vert = verts[next_idx]
        
        distance = np.linalg.norm(np.array(curr_vert) - np.array(next_vert))
        L += pow(distance, alpha)
        Lk[i] = L
    
    for i in range(bsize):
        x = 1.0 / np.sqrt(np.pi) * np.cos(2.0 * np.pi * Lk[i] / L)
        y = 1.0 / np.sqrt(np.pi) * np.sin(2.0 * np.pi * Lk[i] / L)
        fix_vec.append(np.array([x, y]))

def isometricProj(half_edges):
    """
    """
    for i in range(0, len(half_edges), 3):
        a = half_edges[i].Endpoints[0]
        b = half_edges[i].Endpoints[1]
        c = half_edges[i + 1].Endpoints[0]
        c1 = half_edges[i + 1].Endpoints[1]
        flag = False
        if c == a or c == b:
            c, c1 = c1, c
            flag = True
        dist_ab = np.linalg.norm(half_edges[i].EdgeVec)
        Sin, Cos = -2, -2
        if c1 == a:
            dist_ac = np.linalg.norm(half_edges[i + 1].EdgeVec)
            Cos = np.dot(half_edges[i].EdgeVec, half_edges[i + 1].EdgeVec) / (dist_ab * dist_ac)
            if not flag:
                Cos *= -1.0
            Sin = np.sqrt(1 - Cos**2)
            half_edges[i].EdgeVec = np.array([-dist_ab, 0])
            half_edges[i + 1].EdgeVec = np.array([-dist_ac * Cos, -dist_ac * Sin])
            if not flag:
                half_edges[i + 1].EdgeVec *= -1.0
            q = half_edges[i + 2].Endpoints[1]
            half_edges[i + 2].EdgeVec = np.array([dist_ac * Cos - dist_ab, dist_ac * Sin])
            if q == c:
                half_edges[i + 2].EdgeVec *= -1.0
        elif c1 == b:
            dist_ac = np.linalg.norm(half_edges[i + 2].EdgeVec)
            p = half_edges[i + 2].Endpoints[0]
            q = half_edges[i + 2].Endpoints[1]
            Cos = np.dot(half_edges[i].EdgeVec, half_edges[i + 2].EdgeVec) / (dist_ab * dist_ac)
            if p == c:
                Cos *= -1.0
            Sin = np.sqrt(1 - Cos**2)
            half_edges[i].EdgeVec = np.array([-dist_ab, 0])
            half_edges[i + 2].EdgeVec = np.array([dist_ac * Cos, dist_ac * Sin])
            if q == c:
                half_edges[i + 2].EdgeVec *= -1.0
            half_edges[i + 1].EdgeVec = np.array([dist_ac * Cos - dist_ab, dist_ac * Sin])
            if flag:
                half_edges[i + 1].EdgeVec *= -1.0

def findBoundary(half_edges, edges_dict, fix):
    """
    """
    boundary_found = False
    start_vertex = -1
    for he_idx in range(len(half_edges)):
        if half_edges[he_idx].InverseIdx == -1:
            a, b = half_edges[he_idx].Endpoints
            fix.extend([a, b])
            start_vertex = a
            prev_vertex = a
            current_vertex = b
            boundary_found = True
            break

    if not boundary_found:
        if len(half_edges) == 0:
            print("无法找到有效半边！")
            return
        first_he = half_edges[0]
        a, b = first_he.Endpoints
        c = first_he.OppositePoint
        fix.extend([a, b, c])
        start_vertex = a
        prev_vertex = b
        current_vertex = c
        boundary_found = True

    while current_vertex != start_vertex:
        next_vertex = -1
        for (u, v), he_idx in edges_dict.items():
            if u == current_vertex and v != prev_vertex:
                if half_edges[he_idx].InverseIdx == -1:
                    next_vertex = v
                    break
        if next_vertex == -1:
            print("Bad boundary!")
            return
        
        fix.append(next_vertex)
        prev_vertex = current_vertex
        current_vertex = next_vertex

    if len(fix) < 3:
        print("Less than 3 boundary_vertices！")
        return
    if fix[-1] == start_vertex:
        fix.pop()

def normalize_to_one2D(res):
    """
    """
    normalized_res = np.zeros((res.shape[0], 2))
    xmax, ymax, xmin, ymin = -1e16, -1e16, 1e16, 1e16
    for i in range(res.shape[0]):
        if res[i, 0] < xmin:
            xmin = res[i, 0]
        if res[i, 0] > xmax:
            xmax = res[i, 0]
        if res[i, 1] < ymin:
            ymin = res[i, 1]
        if res[i, 1] > ymax:
            ymax = res[i, 1]
    Min = np.array([xmin, ymin])
    scale = 1.0 / max(xmax - xmin, ymax - ymin)
    for i in range(res.shape[0]):
        normalized_res[i] = (res[i] - Min) * scale
        if max(xmax - xmin, ymax - ymin) == xmax - xmin:
            normalized_res[i, 1] += 0.5 * (1.0 - scale * (ymax - ymin))
        else:
            normalized_res[i, 0] += 0.5 * (1.0 - scale * (xmax - xmin))
    return normalized_res

def flip_avoiding_line_search(F, res, new_res, energy_func):
    """
    """
    alpha = 1.0
    min_alpha = 1e-6  
    reduction = 0.5   

    E0 = energy_func(res)  
    while alpha > min_alpha:
        current_res = res + alpha * (new_res - res)
        if not check_flipping(current_res, F):
            E_candidate = energy_func(current_res)
            if E_candidate <= E0:
                return alpha 
        alpha *= reduction
    return 0.0

def check_flipping(V_candidate, F):
    """
    """
    for face in F:
        a, b, c = face
        v0 = V_candidate[a]
        v1 = V_candidate[b]
        v2 = V_candidate[c]
        e1 = v1 - v0
        e2 = v2 - v0
        area = e1[0] * e2[1] - e1[1] * e2[0]
        if area < 0:
            return True
    return False

def get_neighbors(half_edges, neighbors):
    for edge in half_edges:
        a = edge.Endpoints[0]
        b = edge.Endpoints[1]
        neighbors[a].append(b)
        if edge.InverseIdx == -1:
            neighbors[b].append(a)