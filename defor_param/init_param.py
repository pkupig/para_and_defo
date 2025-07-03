# init_param.py

from half_edge import HalfEdge  
import numpy as np

ARAP = 0
ASAP = 1
Hybrid = 2
LSCM = 3
MVC = 4



def readObj(input_name):
    """
    """
    verts = []
    edges = {}
    half_edges = []
    area = []
    F = []
    with open(input_name, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    vert_num = 0
    face_num = 0
    for line in lines:
        if line.startswith('v '):
            vert_num += 1
        elif line.startswith('f '):
            parts = line.split()
            if len(parts) != 4:
                print("Only triangle mesh are supported!")
                assert False
            face_num += 1
    F.clear()
    verts.clear()
    edges.clear()
    half_edges.clear()
    area.clear()
    cur_vert_num = 0
    cur_face_num = 0
    total_area = 0.0

    for line in lines:
        if line.startswith('v '):
            parts = line.split()
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
            verts.append([x, y, z])
            cur_vert_num += 1
        elif line.startswith('f '):
            parts = line.split()
            a = int(parts[1].split('/')[0])-1
            b = int(parts[2].split('/')[0])-1
            c = int(parts[3].split('/')[0])-1
            F.append([a, b, c])

            vec_ab = [verts[a][0]-verts[b][0], 
                     verts[a][1]-verts[b][1], 
                     verts[a][2]-verts[b][2]]
            vec_ac = [verts[a][0]-verts[c][0], 
                     verts[a][1]-verts[c][1], 
                     verts[a][2]-verts[c][2]]
            cross = np.cross(vec_ab, vec_ac)
            area_val = 0.5 * np.linalg.norm(cross)
            area.append(area_val)
            total_area += area_val

            # a→b
            edge_ab = (a, b)
            edge_ba = (b, a)
            if edge_ab not in edges:
                # a→b
                he = HalfEdge()
                he.Endpoints = (a, b)
                he.OppositePoint = c
                he.BelongFacet = cur_face_num
                he.EdgeVec = [verts[a][0]-verts[b][0], 
                             verts[a][1]-verts[b][1], 
                             verts[a][2]-verts[b][2]]
                half_edges.append(he)
                edges[edge_ab] = len(half_edges)-1
                if edge_ba in edges:
                    he.InverseIdx = edges[edge_ba]
                    inv_he_idx = edges[edge_ba]
                    half_edges[inv_he_idx].InverseIdx = edges[edge_ab]
            else:
                print("Input is not a manifold!")
                assert False
            
            # b→c
            edge_bc = (b, c)
            edge_cb = (c, b)
            if edge_bc not in edges:
                he = HalfEdge()
                he.Endpoints = (b, c)
                he.OppositePoint = a
                he.BelongFacet = cur_face_num
                he.EdgeVec = [verts[b][0]-verts[c][0], 
                             verts[b][1]-verts[c][1], 
                             verts[b][2]-verts[c][2]]
                half_edges.append(he)
                edges[edge_bc] = len(half_edges)-1
                
                if edge_cb in edges:
                    he.InverseIdx = edges[edge_cb]
                    inv_he_idx = edges[edge_cb]
                    half_edges[inv_he_idx].InverseIdx = edges[edge_bc]
            else:
                print("Input is not a manifold!")
                assert False
            
            # c→a
            edge_ca = (c, a)
            edge_ac = (a, c)
            if edge_ca not in edges:
                he = HalfEdge()
                he.Endpoints = (c, a)
                he.OppositePoint = b
                he.BelongFacet = cur_face_num
                he.EdgeVec = [verts[c][0]-verts[a][0], 
                             verts[c][1]-verts[a][1], 
                             verts[c][2]-verts[a][2]]
                half_edges.append(he)
                edges[edge_ca] = len(half_edges)-1
                
                if edge_ac in edges:
                    he.InverseIdx = edges[edge_ac]
                    inv_he_idx = edges[edge_ac]
                    half_edges[inv_he_idx].InverseIdx = edges[edge_ca]
            else:
                print("Input is not a manifold!")
                assert False
            
            cur_face_num += 1

    assert cur_vert_num == vert_num and cur_face_num == face_num
    reg_param = np.sqrt(total_area)
    for i in range(len(area)):
        area[i] /= total_area
    for he in half_edges:
        vec = he.EdgeVec
        he.EdgeVec = [v/reg_param for v in vec]
    for v in range(len(verts)):
        verts[v] = [coord/reg_param for coord in verts[v]]

    return cur_vert_num, verts, F, edges, half_edges, area


def genOutputName(input_name, method, slamda, flip_avoid):
    """
    """
    output_name = input_name[:input_name.find(".obj")]
    if method == ARAP:
        output_name += "_ARAP"
    elif method == ASAP:
        output_name += "_ASAP"
    elif method == Hybrid:
        output_name += f"_Hybrid_{slamda}"
    elif method == LSCM:
        output_name += "_LSCM"
    elif method == MVC:
        output_name += "_MVC"
    if flip_avoid:
        output_name += "_noflip"
    return output_name

