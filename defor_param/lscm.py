# lscm.py

import numpy as np
import shutil
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr
from graph_process import findBoundary
from init_param import readObj, genOutputName
from image import (printObjModel, create_checkerboard, 
                  printFile, printImage
)
import os

def run_lscm_parameterization(file_path, method=3, itrs=1, flip_avoid=False, lamda=0.0, output_dir='results'):
    output_dir = os.path.abspath(output_dir)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    full_name = os.path.basename(file_path)
    input_name = os.path.splitext(full_name)[0]
    output_base_name = genOutputName(input_name, method, "0", flip_avoid)
    base_output_path = os.path.join(output_dir, output_base_name)   

    vert_num, verts, F, edges, half_edges, area = readObj(file_path)
    vertices = np.array(verts)  
    # no need for lambda
    fix = []
    findBoundary(half_edges, edges, fix)
    if len(fix) < 2:
        print("Error: Boundary has less than 2 points.")
        return None

    max_dist = -1
    fix0, fix1 = fix[0], fix[0]
    for i in range(len(fix)):
        for j in range(i+1, len(fix)):
            dist = np.linalg.norm(vertices[fix[i]] - vertices[fix[j]])
            if dist > max_dist:
                max_dist = dist
                fix0, fix1 = fix[i], fix[j]
    
    fix = [fix0, fix1]
    fix_coords = [np.array([0.0, 0.0]), np.array([1.0, 0.0])]
    
    vpmap = [False] * vert_num
    for idx in fix:
        vpmap[idx] = True
    non_fix = [i for i in range(vert_num) if not vpmap[i]]
    fvi = {}
    for idx in non_fix:
        fvi[idx] = len(fvi)  
    
    res = np.zeros((vert_num, 2))
    res[fix0] = fix_coords[0]
    res[fix1] = fix_coords[1]
    
    A_data = []
    A_row_ind = []
    A_col_ind = []
    B = []
    row_count = 0
    
    for face in F:
        row_count = setup_triangle_relations(
            vertices, face, vpmap, fvi, res,
            A_data, A_row_ind, A_col_ind, B, row_count
        )
    
    num_rows = row_count
    num_cols = 2 * len(non_fix)  
    A = coo_matrix((A_data, (A_row_ind, A_col_ind)), shape=(num_rows, num_cols)).tocsr()
    B = np.array(B)
    
    solution = lsqr(A, B, atol=1e-5, btol=1e-5, iter_lim=500)[0]
    
    for idx in non_fix:
        vidx = fvi[idx]
        res[idx, 0] = solution[2 * vidx]
        res[idx, 1] = solution[2 * vidx + 1]
    
    normalized_res = res

    aread = compute_area_distortion(vertices, F, res)
    angled = compute_angle_distortion(vertices, F, res)
    distortion_path = os.path.join(output_dir, f"{output_base_name}_distortion.txt")
    distortion_file = open(distortion_path, "w")
    
    printFile(
        print_pic=True,
        print_vtkfile=False,       # so we can disable VTK
        print_txtfile=True,
        print_each_frame=True,  
        now_itr=0,  
        distortion = 0,  
        aread=aread,
        angled=angled,
        output_name=base_output_path,
        distortion_per_unit=0,  
        aread_per_unit=0,
        angled_per_unit=0,
        half_edges=half_edges,
        F=F,
        res=res,
        distortion_file=distortion_file
    )  
    normalized_res = res

    texture_path = os.path.join(output_dir, f"{output_base_name}_texture.png")
    obj_path = os.path.join(output_dir, f"{output_base_name}.obj")
    param_image_path = os.path.join(output_dir, f"{output_base_name}_param.tga")
    mtl_path = os.path.join(output_dir, f"{output_base_name}.mtl")

    create_checkerboard((512, 512), 8, save_path=texture_path)
    printObjModel(vertices, normalized_res, F, obj_path, edges, mtl_path, texture_path, grid_size=1.0, texture_size=(256, 256))
    printImage(half_edges, res, param_image_path)
    
    # no distortion in lscm
    return {
        'vertices': vertices.tolist(),
        'faces': F,
        'param_coords': normalized_res.tolist(),
        'boundary': fix,
        'texture_path': texture_path,
        'vtk_path': None,
        'obj_path': obj_path,
        'mtl_path': mtl_path,
        'param_image': param_image_path,
        'metrics': {
            'distortion': 0.0,  
            'area_distortion': aread,
            'angle_distortion': angled
        },
        'half_edges': half_edges
    }

def setup_triangle_relations(vertices, face, vpmap, fvi, uv,
                             A_data, A_row_ind, A_col_ind, B, row_count):
    v0, v1, v2 = face
    p0 = vertices[v0]
    p1 = vertices[v1]
    p2 = vertices[v2]
    
    e1 = p1 - p0
    e2 = p2 - p0
    normal = np.cross(e1, e2)
    normal_norm = np.linalg.norm(normal)
    if normal_norm < 1e-10:
        z0 = np.array([0.0, 0.0])
        z1 = np.array([1.0, 0.0])
        z2 = np.array([0.0, 1.0])
    else:
        x_axis = e1 / np.linalg.norm(e1)
        z_axis = normal / normal_norm
        y_axis = np.cross(z_axis, x_axis)
        z0 = np.array([0.0, 0.0])
        z1 = np.array([np.linalg.norm(e1), 0.0])  
        z2 = np.array([np.dot(p2 - p0, x_axis), np.dot(p2 - p0, y_axis)])
    
    a = z1[0]  
    b = z1[1] 
    c = z2[0]
    d = z2[1]

    eq1_coeffs = {
        v0: (-a + c, -d),   
        v1: (-c, d),
        v2: (a, 0)
    }

    eq2_coeffs = {
        v0: (d, -a + c),
        v1: (-d, -c),
        v2: (0, a)
    }
    
    rhs1 = 0.0
    for v, (coeff_u, coeff_v) in eq1_coeffs.items():
        if vpmap[v]:
            rhs1 -= coeff_u * uv[v, 0] + coeff_v * uv[v, 1]
        else:
            vidx = fvi[v]
            if coeff_u != 0:
                A_data.append(coeff_u)
                A_row_ind.append(row_count)
                A_col_ind.append(2 * vidx)  
            if coeff_v != 0:
                A_data.append(coeff_v)
                A_row_ind.append(row_count)
                A_col_ind.append(2 * vidx + 1)  
    B.append(rhs1)
    
    rhs2 = 0.0
    for v, (coeff_u, coeff_v) in eq2_coeffs.items():
        if vpmap[v]:
            rhs2 -= coeff_u * uv[v, 0] + coeff_v * uv[v, 1]
        else:
            vidx = fvi[v]
            if coeff_u != 0:
                A_data.append(coeff_u)
                A_row_ind.append(row_count + 1)
                A_col_ind.append(2 * vidx)  
            if coeff_v != 0:
                A_data.append(coeff_v)
                A_row_ind.append(row_count + 1)
                A_col_ind.append(2 * vidx + 1)  
    B.append(rhs2)
    
    return row_count + 2  

def compute_area_distortion(vertices, faces, param_coords):
    area_distortion = 0.0
    for face in faces:
        v0, v1, v2 = vertices[face]
        a = np.linalg.norm(v1 - v0)
        b = np.linalg.norm(v2 - v0)
        c = np.linalg.norm(v2 - v1)
        s = (a + b + c) / 2
        original_area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        p0, p1, p2 = param_coords[face]
        area = 0.5 * np.abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]))
        area_distortion += abs(original_area - area)

    return area_distortion

def compute_angle_distortion(vertices, faces, param_coords):
    angle_distortion = 0.0
    
    for face in faces:
        v0_idx, v1_idx, v2_idx = face
        v0 = vertices[v0_idx]
        v1 = vertices[v1_idx]
        v2 = vertices[v2_idx]
        p0 = param_coords[v0_idx]
        p1 = param_coords[v1_idx]
        p2 = param_coords[v2_idx]
        angle0_original = compute_angle(v0, v1, v2)
        angle1_original = compute_angle(v1, v2, v0)
        angle2_original = compute_angle(v2, v0, v1)
        angle0_param = compute_angle(p0, p1, p2)
        angle1_param = compute_angle(p1, p2, p0)
        angle2_param = compute_angle(p2, p0, p1)
        
        angle_distortion += abs(angle0_original - angle0_param)
        angle_distortion += abs(angle1_original - angle1_param)
        angle_distortion += abs(angle2_original - angle2_param)
    
    return angle_distortion

def compute_angle(a, b, c, eps=1e-8):
    ab = b - a
    ac = c - a
    ab_norm = np.linalg.norm(ab)
    ac_norm = np.linalg.norm(ac)

    if ab_norm < eps or ac_norm < eps:
        return 0.0

    cos_theta = np.dot(ab, ac) / (ab_norm * ac_norm)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  
    
    return np.arccos(cos_theta)