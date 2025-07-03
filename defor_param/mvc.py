import numpy as np
import shutil
import os
from scipy.sparse import lil_matrix, eye
from scipy.sparse.linalg import spsolve, lsqr, bicgstab
from init_param import readObj, genOutputName
from image import (printVTK, printObjModel, 
            create_checkerboard, printFile, printImage
)
from graph_process import (
    findBoundary, mapTo2DBoundary, normalize_to_one2D
)

class TangentWeight:
    def __init__(self, p, q, r):
        """
        Tangent_weight(p, q, r)
        """
        p = np.array(p)
        q = np.array(q)
        r = np.array(r)
        
        v_pr = r - q  # q->r
        v_pq = p - q  # q->p
        self.d_r = np.linalg.norm(v_pr)
        self.d_p = np.linalg.norm(v_pq)
        
        cross_product = np.cross(v_pr, v_pq)
        self.A = np.linalg.norm(cross_product) / 2.0  
        self.S = np.dot(v_pr, v_pq)
        denominator = self.d_r * self.d_p + self.S
        if abs(denominator) < 1e-12:
            self.w_base = 0.0
        else:
            self.w_base = -(2.0 * self.A) / denominator
    
    def get_w_r(self):
        if self.d_r < 1e-12:
            return 0.0
        return self.w_base / self.d_r
    def get_w_p(self):
        if self.d_p < 1e-12:
            return 0.0
        return self.w_base / self.d_p

def build_mvc_matrix_cgal_style(verts, F):
    """
    """
    n = len(verts)
    A = lil_matrix((n, n), dtype=np.float64)
    
    for tri in F:
        a, b, c = tri
        p_a = verts[a]
        p_b = verts[b]
        p_c = verts[c]
        
        tw_a = TangentWeight(p_c, p_a, p_b)  
        w_ab = tw_a.get_w_r()  
        w_ac = tw_a.get_w_p()  
        
        A[a, b] += w_ab
        A[a, c] += w_ac
        A[a, a] -= (w_ab + w_ac)
        
        tw_b = TangentWeight(p_a, p_b, p_c) 
        w_bc = tw_b.get_w_r() 
        w_ba = tw_b.get_w_p()  
        
        A[b, c] += w_bc
        A[b, a] += w_ba
        A[b, b] -= (w_bc + w_ba)
        
        tw_c = TangentWeight(p_b, p_c, p_a)  
        w_ca = tw_c.get_w_r() 
        w_cb = tw_c.get_w_p()  
        
        A[c, a] += w_ca
        A[c, b] += w_cb
        A[c, c] -= (w_ca + w_cb)
    
    return A

def run_mvc_parameterization(file_path, method = 4, itrs=1, flip_avoid=False, lamda=0.0, output_dir='results'):

    output_dir = os.path.abspath(output_dir)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    full_name = os.path.basename(file_path)
    input_name = os.path.splitext(full_name)[0]
    output_base_name = genOutputName(input_name, 4, "0", flip_avoid)
    base_output_path = os.path.join(output_dir, output_base_name)
    
    vert_num, verts, F, edges, half_edges, area = readObj(file_path)
    res = np.zeros((vert_num, 2))  
    verts = np.array(verts)
    fix = []  
    fix_vec = []  
    # no need for lambda

    findBoundary(half_edges, edges, fix)
    mapTo2DBoundary(verts, fix, fix_vec, 1)  
    
    uvmap = {}
    for i in range(vert_num):
        uvmap[i] = res[i]
    for i, idx in enumerate(fix):
        uvmap[idx] = fix_vec[i]
        res[idx] = fix_vec[i]
    
    n = vert_num
    A = build_mvc_matrix_cgal_style(verts, F)
    Bu = np.zeros(n)
    Bv = np.zeros(n)
    
    original_rows = {}
    for idx in fix:
        original_rows[idx] = A[idx].copy()
    
    for idx in fix:
        A[idx] = 0.0
        A[idx, idx] = 1.0
        Bu[idx] = uvmap[idx][0]
        Bv[idx] = uvmap[idx][1]
    
    A_csr = A.tocsr()
    
    try:
        M = eye(A.shape[0])
        Xu, info_u = bicgstab(A_csr, Bu, atol=1e-10, maxiter=1000, M=M)
        Xv, info_v = bicgstab(A_csr, Bv, atol=1e-10, maxiter=1000, M=M)
        
        if info_u != 0 or info_v != 0:
            raise RuntimeError(f"BiCGSTAB failed to converge: info_u={info_u}, info_v={info_v}")
    except Exception as e:
        print(f"BiCGSTAB failed, falling back to direct solver: {str(e)}")
        try:
            Xu = spsolve(A_csr, Bu)
            Xv = spsolve(A_csr, Bv)
        except:
            Xu, exit_code = lsqr(A_csr, Bu, atol=1e-8, btol=1e-8)[:2]
            Xv, exit_code = lsqr(A_csr, Bv, atol=1e-8, btol=1e-8)[:2]
            if exit_code < 0:
                raise RuntimeError(f"LSQR failure: exit_code = {exit_code}")
    for i in range(vert_num):
        res[i] = [Xu[i], Xv[i]]
    
    distortion_per_unit = np.zeros(len(half_edges) // 3)
    aread_per_unit = np.zeros(len(half_edges) // 3)
    angled_per_unit = np.zeros(len(half_edges) // 3)
    
    distortion, aread, angled = compute_distortion_metrics(
        half_edges, area, res, verts,
        distortion_per_unit, aread_per_unit, angled_per_unit
    )
    distortion_path = os.path.join(output_dir, f"{output_base_name}_distortion.txt")
    distortion_file = open(distortion_path, "w")

    printFile(
        print_pic=True,
        print_vtkfile=True,
        print_txtfile=True,
        print_each_frame=True,
        now_itr=0,
        distortion=distortion,
        aread=aread,
        angled=angled,
        output_name=base_output_path,
        distortion_per_unit=distortion_per_unit,
        aread_per_unit=aread_per_unit,
        angled_per_unit=angled_per_unit,
        half_edges=half_edges,
        F=F,
        res=res,
        distortion_file=distortion_file
    )
    distortion_file.close()
    normalized_res = normalize_to_one2D(res)
    
    texture_path = os.path.join(output_dir, f"{output_base_name}_texture.png")
    vtk_path = os.path.join(output_dir, f"{output_base_name}_distortion.vtk")
    obj_path = os.path.join(output_dir, f"{output_base_name}.obj")
    param_image_path = os.path.join(output_dir, f"{output_base_name}_param.tga")
    mtl_path = os.path.join(output_dir, f"{output_base_name}.mtl")
    
    create_checkerboard((512, 512), 8, save_path=texture_path)
    printVTK(half_edges, F, distortion_per_unit, aread_per_unit, angled_per_unit, res, vtk_path[:-4])
    printObjModel(verts, normalized_res, F, obj_path, edges, mtl_path, texture_path, grid_size=1.0, texture_size=(256, 256))
    printImage(half_edges, res, param_image_path)
    # print(len(half_edges))

    return {
        'vertices': verts.tolist(),
        'faces': F,
        'param_coords': normalized_res.tolist(),
        'boundary': fix,
        'texture_path': texture_path,
        'vtk_path': vtk_path,
        'obj_path': obj_path,
        'mtl_path': mtl_path,
        'param_image': param_image_path,
        'metrics': {
            'distortion': distortion,
            'area_distortion': aread,
            'angle_distortion': angled
        },
        'half_edges': half_edges
    }

def compute_distortion_metrics(half_edges, area, res, vertices_3d,
                              distortion_per_unit, aread_per_unit, angled_per_unit):
    """
    """
    distortion = 0.0
    aread = 0.0
    angled = 0.0
    for i in range(len(half_edges) // 3):
        he1 = half_edges[3*i]
        he2 = half_edges[3*i+1]
        he3 = half_edges[3*i+2]
        v0 = he1.Endpoints[0]
        v1 = he1.Endpoints[1]
        v2 = he2.Endpoints[0] if he2.Endpoints[0] != v1 else he2.Endpoints[1]
        
        p0_3d = np.array(vertices_3d[v0])
        p1_3d = np.array(vertices_3d[v1])
        p2_3d = np.array(vertices_3d[v2])
        
        p0_2d = np.array(res[v0])
        p1_2d = np.array(res[v1])
        p2_2d = np.array(res[v2])
        
        area_3d = area[i]
        vec1_2d = p1_2d - p0_2d
        vec2_2d = p2_2d - p0_2d
        area_2d = 0.5 * abs(vec1_2d[0]*vec2_2d[1] - vec1_2d[1]*vec2_2d[0])
        
        if area_3d < 1e-8:
            area_ratio = 0
        else:
            area_ratio = area_2d / area_3d
        
        area_dist = abs(area_ratio - 1.0)
        aread_per_unit[i] = area_dist * area_3d  
        aread += aread_per_unit[i]
        
        def compute_angles(p0, p1, p2):
            vec1 = p1 - p0
            vec2 = p2 - p0
            vec3 = p2 - p1
            
            len1 = np.linalg.norm(vec1)
            len2 = np.linalg.norm(vec2)
            len3 = np.linalg.norm(vec3)
            
            if len1 < 1e-8 or len2 < 1e-8 or len3 < 1e-8:
                return [0, 0, 0]
            
            angle0 = np.arccos(np.clip(np.dot(vec1, vec2) / (len1 * len2), -1.0, 1.0))
            angle1 = np.arccos(np.clip(np.dot(-vec1, vec3) / (len1 * len3), -1.0, 1.0))
            angle2 = np.pi - angle0 - angle1  
            
            return [angle0, angle1, angle2]
        
        angles_3d = compute_angles(p0_3d, p1_3d, p2_3d)
        angles_2d = compute_angles(p0_2d, p1_2d, p2_2d)
        
        angle_diff = 0
        for idx in range(3):
            angle_diff += abs(angles_3d[idx] - angles_2d[idx])
        
        angled_per_unit[i] = angle_diff * area_3d  
        angled += angled_per_unit[i]
        
        distortion_per_unit[i] = area_dist * area_3d
        distortion += distortion_per_unit[i]
    
    return distortion, aread, angled