#-----------------------------------------------------------------
# app.py
# This is a Flask-based backend program aimed at enabling 
# visualization and user interactivity for parametric design and 
# mesh deformation. 
#-----------------------------------------------------------------



import os
import shutil
import json
import numpy as np
import threading
import uuid
import time
import socket
import signal
import logging
from contextlib import contextmanager
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from scipy.sparse import coo_matrix, linalg, vstack, lil_matrix, diags
from scipy.sparse.linalg import splu
from arap import (
    local_phase_param, global_phase, getLaplace, getRHS,
    local_phase_deform, ARAP_energy
)
from init_param import readObj, genOutputName
from image import (
    printVTK, printObjModel, create_checkerboard, 
    printFile, printImage
)
from graph_process import (
    findBoundary, mapTo2DBoundary, getWeights,
    isometricProj, flip_avoiding_line_search, get_neighbors
)
from lscm import run_lscm_parameterization
from mvc import run_mvc_parameterization



logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


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

DEFORM_METHODS = {
    0: "ARAP Deformation",
    1: "Laplacian Deformation",
}



@contextmanager
def timeout(seconds):
    def raise_timeout(signum, frame):
        raise TimeoutError(f"操作超时，超过 {seconds} 秒")
    
    if hasattr(signal, 'SIGALRM') and hasattr(signal, 'alarm'):
        original_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, raise_timeout)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, original_handler)
    else:
        start_time = time.time()
        try:
            yield
        finally:
            if time.time() - start_time > seconds:
                raise TimeoutError(f"操作超时，超过 {seconds} 秒")

class DeformationSession:
    def __init__(self, vertices, faces, param_coords, boundary, metrics, half_edges):
        self.vertices = np.array(vertices)  
        self.faces = np.array(faces)        
        self.param_coords = np.array(param_coords)  
        self.boundary = boundary            
        self.metrics = metrics              
        self.half_edges = half_edges       
        
        self.current_vertices = np.copy(self.vertices)
        self.handles = {}  
        self.selected_handle = -1  
        self.method = ARAP  
        
        self.neighbors = [[] for _ in range(len(self.vertices))]
        get_neighbors(self.half_edges, self.neighbors)
        self.weights = np.zeros(len(self.vertices)*len(self.vertices))
        getWeights(self.half_edges, self.vertices, self.weights, Cotangent_2)
        self.R = [np.eye(3) for _ in range(len(self.vertices))]
        
        self.placing_handles = True
        self.first_deform = True
        self.current_iteration = 0
        self.moved = False
        self.changed = False
        
        self.laplacian_matrix = None
        self.dir_solver = None
        self._precompute_solver()
    
    
     
    def _precompute_solver(self):
        """ 
        """ 
        fix = list(self.handles.keys()) if self.handles else self.boundary
        n_vertices = len(self.vertices)
        laplace = lil_matrix((n_vertices, n_vertices))
        
        self.laplacian_matrix = getLaplace(
            self.half_edges, 
            self.weights, 
            fix, 
            laplace, 
            DEFORM
        )
    
        diag_reg = 1e-8 * np.ones(n_vertices)
        reg_matrix = diags(diag_reg, 0)
        regularized_matrix = self.laplacian_matrix + reg_matrix
    
        try:
            self.dir_solver = linalg.factorized(regularized_matrix.tocsc())
        except RuntimeError as e:
            logger.error(f"矩阵分解失败: {str(e)}")
            from scipy.sparse.linalg import svds
            U, s, Vt = svds(regularized_matrix, k=min(regularized_matrix.shape)-1)
            s_inv = np.zeros_like(s)
            s_inv[s > 1e-10] = 1/s[s > 1e-10]
            self.dir_solver = (Vt.T @ np.diag(s_inv) @ U.T).dot
            


    def set_handle(self, vertex_id, position):
        """
        """
        self.handles[vertex_id] = np.array(position)
        self.selected_handle = vertex_id
        self.placing_handles = False
        self.first_deform = True 
        self._precompute_solver()  
    
    def update_handle_position(self, vertex_id, new_position):
        """
        """
        if vertex_id in self.handles:
            self.handles[vertex_id] = np.array(new_position)
            self.moved = True
            self.current_iteration = 0
            # self.changed = True
            self.first_deform = False
    
    def deform_iteration(self, method=ARAP):
        """
        """
        if not self.handles or not self.moved:
            return None
        
        self.method = method  
        
        if method == Laplacian:
            return self._laplacian_deform()
        else:  
            return self._arap_deform()
    
    def _arap_deform(self):

        if self.first_deform:
            self._precompute_solver()  
        self.first_deform = False

        n_vertices = len(self.current_vertices)
        RHS = np.zeros((n_vertices, 3))
        fix = list(self.handles.keys())
        fix_vec = [self.handles[i] for i in fix]
    
        # 使用预计算的余切权重
        local_phase_deform(
            self.R, 
            self.neighbors, 
            self.vertices, 
            self.current_vertices, 
            self.weights
        )
    
        getRHS(
            self.R, 
            self.half_edges, 
            self.weights, 
            fix, 
            fix_vec, 
            RHS, 
            DEFORM, 
            self.first_deform
        )
    
        # 向量化求解
        try:
            new_vertices = self.dir_solver(RHS)
        except Exception as e:
            logger.error(f"求解失败: {str(e)}")
            return None
    
        self.current_vertices = new_vertices
        self.current_iteration += 1
        self.moved = False
        self.first_deform = False
    
        return self._compute_deformation_metrics()
    
    def _laplacian_deform(self):
        """
        """
        n_vertices = len(self.current_vertices)
        fix = list(self.handles.keys())
        
        A = self.laplacian_matrix
        b = np.zeros((n_vertices, 3))
        
        for i in range(n_vertices):
            if i not in self.handles:
                b[i] = self.vertices[i]
        
        constraint_matrix = coo_matrix((0, n_vertices))
        constraint_rhs = np.zeros((0, 3))
        
        for vertex_id, position in self.handles.items():
            row = coo_matrix(([1], ([0], [vertex_id])), shape=(1, n_vertices))
            constraint_matrix = vstack([constraint_matrix, row])
            constraint_rhs = np.vstack([constraint_rhs, position])
        
        A_augmented = vstack([A, constraint_matrix])
        b_augmented = np.vstack([b, constraint_rhs])
        
        new_vertices = np.zeros((n_vertices, 3))
        for dim in range(3):
            solution, _ = linalg.lsqr(A_augmented, b_augmented[:, dim])
            new_vertices[:, dim] = solution
        
        self.current_vertices = new_vertices
        self.current_iteration += 1
        self.moved = False
        self.first_deform = False
        
        return self._compute_deformation_metrics()
    
    def _compute_deformation_metrics(self):
        """
        """
        n_vertices = len(self.current_vertices)
        max_displacement = 0.0
        distortion = 0.0
        
        for i in range(n_vertices):
            displacement = np.linalg.norm(self.current_vertices[i] - self.vertices[i])
            if displacement > max_displacement:
                max_displacement = displacement
            
            for j in self.neighbors[i]:
                orig_edge = np.linalg.norm(self.vertices[j] - self.vertices[i])
                new_edge = np.linalg.norm(self.current_vertices[j] - self.current_vertices[i])
                distortion += abs(orig_edge - new_edge)
        
        return {
            'vertices': self.current_vertices.tolist(),
            'metrics': {
                'distortion': distortion,
                'max_displacement': max_displacement
            },
            'deform_method': DEFORM_METHODS.get(self.method, "Unknown")
        }
    
    def auto_deform(self):
        if self.moved and not self.placing_handles:
            self.deform_iteration()
            self.moved = False

    def get_session(self, session_id):
        session = self.sessions[session_id]
        if not hasattr(session, 'auto_timer'):
            session.auto_timer = threading.Timer(0.05, self.auto_deform, [session_id])
            session.auto_timer.start()
        return session

    def auto_deform(self, session_id):
        try:
            session = self.get_session(session_id)
            session.auto_deform()
        finally:
            if session_id in self.sessions:
                session.auto_timer = threading.Timer(0.05, self.auto_deform, [session_id])
                session.auto_timer.start()



class DeformationServer:
    def __init__(self):
        self.sessions = {}
        self.lock = threading.Lock()
    
    def init_session(self, session_id, vertices, faces, param_coords, boundary, metrics, half_edges):
        """
        """
        with self.lock:
            if session_id in self.sessions:
                raise ValueError(f"Session {session_id} already exists")
                
            session = DeformationSession(
                vertices,
                faces,
                param_coords,
                boundary,
                metrics,
                half_edges
            )
            self.sessions[session_id] = session
            return session
    
    def get_session(self, session_id):
        """
        """
        with self.lock:
            session = self.sessions.get(session_id)
            if not session:
                raise KeyError(f"Session {session_id} not found")
            return session
    
    def set_handle(self, session_id, vertex_id, position):
        """
        """
        try:
            session = self.get_session(session_id)
            session.set_handle(vertex_id, position)
            return True
        except Exception as e:
            logger.error(f"Error setting handle: {e}")
            return False
    
    def update_handle(self, session_id, vertex_id, position):
        """
        """
        try:
            session = self.get_session(session_id)
            session.update_handle_position(vertex_id, position)
            return True
        except Exception as e:
            logger.error(f"Error updating handle: {e}")
            return False
    
    def deform(self, session_id, method):
        """
        """
        try:
            session = self.get_session(session_id)
            result = session.deform_iteration(method)
            return result
        except Exception as e:
            logger.error(f"Error deforming mesh: {e}")
            return None




def run_arap_parameterization(file_path, method, itrs, flip_avoid, lamda=0.0, output_dir='results'):
    output_dir = os.path.abspath(output_dir)
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    full_name = os.path.basename(file_path)
    input_name = os.path.splitext(full_name)[0]
    flip_avoid=False
    itrs=5

    output_base_name = genOutputName(input_name, method, "0", flip_avoid)
    base_output_path = os.path.join(output_dir, output_base_name)
    
    verts_list = []          
    edges = {}          
    half_edges = []     
    area = []         
    F = []   
    vert_num, verts_list, F, edges, half_edges, area = readObj(file_path)
    vertices = np.array(verts_list)
    
    res = np.zeros((vert_num, 2))
    RHS = np.zeros((vert_num, 2))
    weights = np.zeros(len(half_edges))
    Laplace = lil_matrix((vert_num, vert_num))
    R = None
    print_txtfile = True
    fix = []
    fix_vec = []
    PARAM = 0

    findBoundary(half_edges, edges, fix)
    mapTo2DBoundary(vertices, fix, fix_vec, 1)
    weights = np.zeros(len(half_edges))
    getWeights(half_edges, vertices, weights, 0)
    Laplace = getLaplace(half_edges, weights, fix, Laplace, PARAM)
    RHS = np.zeros((vert_num, 2))
    getRHS(R, half_edges, weights, fix, fix_vec, RHS, PARAM, True)
    dir_solver = splu(Laplace.tocsc())
    res = dir_solver.solve(RHS)
    
    isometricProj(half_edges)
    getWeights(half_edges, vertices, weights, 2)
    R = [np.zeros((2, 2)) for _ in range(len(half_edges) // 3)]

    distortion = 0.0
    aread = 0.0
    angled = 0.0
    distortion_per_unit = np.zeros(len(half_edges) // 3)
    aread_per_unit = np.zeros(len(half_edges) // 3)
    angled_per_unit = np.zeros(len(half_edges) // 3)    

    distortion_path = os.path.join(output_dir, f"{output_base_name}_distortion.txt")
    if print_txtfile:
        distortion_file = open(distortion_path, "w")
    
    fix.clear()
    fix_vec.clear()
    np.random.seed(42)
    fix.append(np.random.randint(vert_num))
    for i in range(len(fix)):
        fix_vec.append(res[fix[i], :].reshape(-1, 1))
    
    Laplace1 = getLaplace(half_edges, weights, fix, Laplace, PARAM)
    dir_solver = splu(Laplace1.tocsc())
    E = ARAP_energy(half_edges, weights, area, method, lamda)
    new_res = np.zeros((vert_num, 2))
    
    for now_itr in range(itrs):
        local_phase_param(
            R, half_edges, res, weights, area, method, lamda,
            distortion_per_unit, aread_per_unit, angled_per_unit,
            distortion, aread, angled
        )
        
        printFile(
            print_pic=True,
            print_vtkfile=True,
            print_txtfile=True,
            print_each_frame=True,
            now_itr=now_itr,
            distortion=distortion,
            aread=aread,
            angled=angled,
            output_name=output_base_name,
            distortion_per_unit=distortion_per_unit,
            aread_per_unit=aread_per_unit,
            angled_per_unit=angled_per_unit,
            half_edges=half_edges,
            F=F,
            res=res,
            distortion_file=open(distortion_path, "a")
        )
        
        global_phase(R, half_edges, weights, fix, fix_vec, RHS, PARAM, False, new_res, dir_solver)
        if flip_avoid:
            flip_avoiding_line_search(F, res, new_res, E)
        else:
            res = new_res.copy()
        
        for i in range(len(fix)):
            fix_vec[i] = res[fix[i], :].reshape(-1, 1)
    local_phase_param(
        R, half_edges, res, weights, area, method, lamda,
        distortion_per_unit, aread_per_unit, angled_per_unit,
        distortion, aread, angled
    )
    if distortion_file is not None:
        distortion_file.close()
    flip_avoid = True
    distortion = np.sum(distortion_per_unit)
    aread = np.sum(aread_per_unit)
    angled = np.sum(angled_per_unit)
    texture_path = os.path.join(output_dir, f"{output_base_name}_texture.png")
    vtk_path = os.path.join(output_dir, f"{output_base_name}_distortion.vtk")
    obj_path = os.path.join(output_dir, f"{output_base_name}.obj")
    param_image_path = os.path.join(output_dir, f"{output_base_name}_param.tga")
    mtl_path = os.path.join(output_dir, f"{output_base_name}.mtl")

    # tex_path =  f"{output_base_name}_texture.png"
    # v_path = output_dir, f"{output_base_name}_distortion.vtk"
    # ob_path =  f"{output_base_name}.obj"
    # param_path =  f"{output_base_name}_param.tga"
    # mt_path = f"{output_base_name}.mtl"

    create_checkerboard((512, 512), save_path=texture_path)
    printVTK(half_edges, F, distortion_per_unit, aread_per_unit, angled_per_unit, res, vtk_path)
    printObjModel(vertices, res, F, obj_path, edges, mtl_path, texture_path, grid_size=1.0, texture_size=(256, 256))
    printImage(half_edges, res, param_image_path)
    
    return {
        'vertices': vertices.tolist(),
        'faces': F,
        'param_coords': res.tolist(),
        'boundary': fix,
        'texture_path': texture_path,
        'vtk_path': vtk_path,
        'obj_path': obj_path,
        'mtl_path': mtl_path,
        'param_image': param_image_path,
        'metrics': {
            'distortion': float(distortion),
            'area_distortion': float(aread),
            'angle_distortion': float(angled)
        },
        'half_edges': half_edges
    }



r''''''
def run_param(file_path, method, itrs, flip_avoid, lamda, output_base):
    # print(f"method: {method}") 
    logger.info(f"运行参数化: 方法={PARAM_METHODS.get(method, method)}")
    if method in [ARAP, ASAP, Hybrid]:
        return run_arap_parameterization(file_path, method, itrs=5, flip_avoid=True, lamda = 0.0, output_dir=output_base+str(method))
    elif method == LSCM:
        return run_lscm_parameterization(file_path, method, itrs=1, flip_avoid=False, lamda=0.0, output_dir=output_base+str(method))
    elif method == MVC:
        return run_mvc_parameterization(file_path, method, itrs=1, flip_avoid=False, lamda=0.0, output_dir=output_base+str(method))
    else:
        raise ValueError(f"Unsupported method: {method}")





deformation_server = DeformationServer()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "supports_credentials": True}})

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
TEXTURES_FOLDER = 'textures'
VTK_FOLDER = 'vtk'
PARAM_IMAGES = 'param_images'
MTL_FOLDER = 'mtl'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['TEXTURES_FOLDER'] = TEXTURES_FOLDER
app.config['VTK_FOLDER'] = VTK_FOLDER
app.config['PARAM_IMAGES'] = PARAM_IMAGES
app.config['MTL_FOLDER'] = MTL_FOLDER

for folder in [UPLOAD_FOLDER, RESULTS_FOLDER, TEXTURES_FOLDER, VTK_FOLDER, PARAM_IMAGES, MTL_FOLDER]:
    os.makedirs(folder, exist_ok=True)
    os.chmod(folder, 0o755)


PARAM_METHODS = {
    0: "ARAP",
    1: "ASAP",
    2: "Hybrid",
    3: "LSCM",
    4: "MVC",
}

DEFORM_METHODS = {
    0: "ARAP Deformation",
    1: "Laplacian Deformation",
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'obj'


"""
def save_temp_obj(vertices, faces, session_id):
    r'''
    临时obj文件
    '''
    temp_file = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{session_id}.obj")
    with open(temp_file, 'w') as f:
        for vert in vertices:
            f.write(f"v {vert[0]} {vert[1]} {vert[2]}\n")
        
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
    return temp_file
"""


session_files = {}

@app.route('/upload_mesh', methods=['POST'])
def upload_mesh():
    """
    """
    try:
        logger.info("Received mesh upload request")
        
        if 'file' not in request.files:
            logger.error('No file part in request')
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.error('No selected file')
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            # 生成唯一会话ID
            session_id = str(uuid.uuid4())
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
            file.save(file_path)
            logger.info(f"File saved to: {file_path}")
            
            session_files[session_id] = file_path
            vert_num, verts_list, F, _, _, _ = readObj(file_path)
            
            return jsonify({
                'session_id': session_id,
                'vertices': verts_list,
                'faces': F,
                'status': 'loaded'
            })
        else:
            logger.error(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Invalid file type'}), 400

    except Exception as e:
        logger.exception("Unhandled exception in mesh upload endpoint")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/parameterize', methods=['POST'])
def parameterize():
    try:
        logger.info("Received parameterization request")
        data = request.json
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({'error': 'Missing session_id'}), 400
        
        if session_id not in session_files:
            return jsonify({'error': 'Session expired or invalid'}), 400
        
        file_path = session_files[session_id]
        
        method = int(data.get('method', 0))
        itrs = int(data.get('iterations', 5))
        flip_avoid = bool(data.get('flip_avoid', False))
        lamda = float(data.get('lamda', 0.0))
        output_base = app.config['RESULTS_FOLDER']
        
        logger.info(f"Parameters: method={method}, iterations={itrs}, flip_avoid={flip_avoid}, lambda={lamda}")
        
        try:
            with timeout(600):  # 5-min timeout
                result = run_param(file_path, method, itrs, flip_avoid, lamda, output_base)
        except TimeoutError as e:
            logger.error(f"Parameterization timed out: {str(e)}")
            return jsonify({'error': 'Parameterization timed out'}), 500
        except Exception as e:
            logger.error(f"Parameterization failed: {str(e)}")
            return jsonify({'error': f'Parameterization failed: {str(e)}'}), 500
        
        try:
            if len(result['param_coords']) != len(result['vertices']):
                raise ValueError("参数化坐标数与顶点数不匹配")
                
            deformation_server.init_session(
                session_id,
                result['vertices'],
                result['faces'],
                result['param_coords'],
                result['boundary'],
                result['metrics'],
                result['half_edges']
            )
        except Exception as e:
            logger.error(f"Failed to initialize deformation session: {str(e)}")
            return jsonify({'error': f'Session initialization failed: {str(e)}'}), 500
        
        moved_files = {}
        for file_type, target_folder in [
            ('texture_path', app.config['TEXTURES_FOLDER']),
            ('vtk_path', app.config['VTK_FOLDER']),
            ('obj_path', app.config['RESULTS_FOLDER']),
            ('mtl_path', app.config['MTL_FOLDER']),
            ('param_image', app.config['PARAM_IMAGES'])
        ]:
            if file_type in result and result[file_type]:
                try:
                    src = result[file_type]
                    if os.path.exists(src):
                        filename = os.path.basename(src)
                        dest_path = os.path.join(target_folder, filename)
                        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                        
                        shutil.copy2(src, dest_path)
                        
                        moved_files[file_type] = filename
                        logger.info(f"Copied {file_type} from {src} to {dest_path}")
                    else:
                        logger.warning(f"Source file not found: {src}")
                except Exception as e:
                    logger.error(f"Failed to copy file: {str(e)}")
        
        response = {
            'session_id': session_id,
            'vertices': result['vertices'],
            'faces': result['faces'],
            'param_coords': result['param_coords'],
            'boundary': result['boundary'],
            'metrics': result.get('metrics', {}),
            'param_method': PARAM_METHODS.get(method, "Unknown"),
            'param_method_id': method,
            'status': 'parameterized'
        }
        
        # 添加文件路径
        for file_type in moved_files:
            response[file_type] = moved_files[file_type]
        
        logger.info(f"Parameterization completed for session: {session_id}")
        return jsonify(response)
    except Exception as e:
        logger.exception("Unhandled exception in parameterize endpoint")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

# 变形相关路由
@app.route('/set_handle', methods=['POST'])
def set_handle():
    data = request.json
    session_id = data.get('session_id')
    vertex_id = data.get('vertex_id')
    position = data.get('position')
    
    if not all([session_id, vertex_id, position]):
        return jsonify({'error': 'Missing parameters'}), 400
    
    try:
        success = deformation_server.set_handle(session_id, vertex_id, position)
        return jsonify({'success': success})
    except Exception as e:
        logger.error(f"Error setting handle: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/update_handle', methods=['POST'])
def update_handle():
    data = request.json
    session_id = data.get('session_id')
    vertex_id = data.get('vertex_id')
    position = data.get('position')
    
    if not all([session_id, vertex_id, position]):
        return jsonify({'error': 'Missing parameters'}), 400
    
    try:
        success = deformation_server.update_handle(session_id, vertex_id, position)
        return jsonify({'success': success})
    except Exception as e:
        logger.error(f"Error updating handle: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/deform', methods=['POST'])
def deform_mesh():
    data = request.json
    session_id = data.get('session_id')
    method = data.get('method', 0)
    iterations = int(data.get('iterations', 4))  
    
    session = deformation_server.get_session(session_id)
    results = []
    for _ in range(iterations):
        result = session.deform_iteration(method)
        results.append(result)
    
    return jsonify({
        'vertices': results[-1]['vertices'],
        'metrics': results[-1]['metrics'],
        'iterations': iterations
    })

# 清理会话
@app.route('/cleanup', methods=['POST'])
def cleanup_session():
    data = request.json
    session_id = data.get('session_id')
    
    if not session_id:
        return jsonify({'error': 'Missing session_id'}), 400
    
    try:
        with deformation_server.lock:
            # 删除变形会话
            if session_id in deformation_server.sessions:
                del deformation_server.sessions[session_id]
            
            # 删除上传的文件
            if session_id in session_files:
                try:
                    os.remove(session_files[session_id])
                except Exception as e:
                    logger.warning(f"Failed to remove file: {str(e)}")
                del session_files[session_id]
            
            # 清理结果文件
            for folder in [app.config['UPLOAD_FOLDER'], 
                          app.config['RESULTS_FOLDER'],
                          app.config['TEXTURES_FOLDER'],
                          app.config['VTK_FOLDER'],
                          app.config['PARAM_IMAGES'],
                          app.config['MTL_FOLDER']]:
                for file in os.listdir(folder):
                    if file.startswith(session_id):
                        try:
                            os.remove(os.path.join(folder, file))
                        except Exception as e:
                            logger.warning(f"Failed to remove file: {str(e)}")
        
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error cleaning up session: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
# 文件服务路由
@app.route('/textures/<path:filename>')
def serve_texture(filename):
    return send_from_directory(app.config['TEXTURES_FOLDER'], filename)

@app.route('/vtk/<path:filename>')
def serve_vtk(filename):
    return send_file(
        os.path.join(app.config['VTK_FOLDER'], filename),
        mimetype='text/vtk',
        as_attachment=False
    )

@app.route('/obj/<path:filename>')
def serve_obj(filename):
    return send_file(
        os.path.join(app.config['RESULTS_FOLDER'], filename),
        mimetype='text/plain',
        as_attachment=False
    )

@app.route('/mtl/<path:filename>')
def serve_mtl(filename):
    return send_file(
        os.path.join(app.config['MTL_FOLDER'], filename),
        mimetype='text/plain',
        as_attachment=False
    )

@app.route('/param_image/<path:filename>')
def serve_param_image(filename):
    return send_from_directory(app.config['PARAM_IMAGES'], filename)

# 获取方法列表
@app.route('/param_methods', methods=['GET'])
def get_param_methods():
    return jsonify(PARAM_METHODS)

@app.route('/deform_methods', methods=['GET'])
def get_deform_methods():
    return jsonify(DEFORM_METHODS)

@app.route('/toggle_mode', methods=['POST'])
def toggle_mode():
    data = request.json
    session_id = data.get('session_id')
    
    session = deformation_server.get_session(session_id)
    session.placing_handles = not session.placing_handles
    
    if not session.placing_handles and session.handles:
        session._precompute_solver()
    
    return jsonify({
        'placing_handles': session.placing_handles
    })


@app.route('/')
def index():
    return send_from_directory('templates', 'app.html')

# 实时数据流Socket服务器
def start_socket_server():
    HOST = 'localhost'
    PORT = 5001
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            s.bind((HOST, PORT))
            s.listen()
            print(f"Socket server listening on {HOST}:{PORT}")
            
            while True:
                conn, addr = s.accept()
                with conn:
                    print(f"Connected by {addr}")
                    while True:
                        data = conn.recv(1024)
                        if not data:
                            break
                        
                        try:
                            request_data = json.loads(data.decode())
                            session_id = request_data.get('session_id')
                            if session_id in deformation_server.sessions:
                                session = deformation_server.sessions[session_id]
                                response = {
                                    'vertices': session.current_vertices.tolist(),
                                    'handles': {k: v.tolist() for k, v in session.handles.items()}
                                }
                                conn.sendall(json.dumps(response).encode())
                        except Exception as e:
                            conn.sendall(json.dumps({'error': str(e)}).encode())
        except OSError as e:
            print(f"无法启动Socket服务器: {e}")
            try:
                PORT = 5555
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((HOST, PORT))
                s.listen()
                print(f"Socket服务器现在监听于 {HOST}:{PORT}")
                
                while True:
                    conn, addr = s.accept()
                    with conn:
                        print(f"连接来自 {addr}")
                        while True:
                            data = conn.recv(1024)
                            if not data:
                                break
                            try:
                                request_data = json.loads(data.decode())
                                session_id = request_data.get('session_id')
                                if session_id in deformation_server.sessions:
                                    session = deformation_server.sessions[session_id]
                                    response = {
                                        'vertices': session.current_vertices.tolist(),
                                        'handles': {k: v.tolist() for k, v in session.handles.items()}
                                    }
                                    conn.sendall(json.dumps(response).encode())
                            except Exception as e:
                                conn.sendall(json.dumps({'error': str(e)}).encode())
            except OSError as e2:
                print(f"备用端口也无法使用: {e2}")




if __name__ == '__main__':
    r'''
    '''
    socket_thread = threading.Thread(target=start_socket_server)
    socket_thread.daemon = True
    socket_thread.start()
    
    app.run(host='0.0.0.0', port=5000, debug=True)



