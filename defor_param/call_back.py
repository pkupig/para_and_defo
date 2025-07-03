# call_back.py

#----------------------------------------------------------------------
# revision from the stantard implementation , but there is an error 
# in pyigl and igl so the manners did badly.

# the original implementation uses pointers so the transformed 
# one takes advantage of list to simulate the have-been behaviours.
#----------------------------------------------------------------------

import numpy as np
import igl
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from math import isinf
from arap import local_phase_deform, global_phase, getLaplace
from image import printObjModel



DEFORM = 1
def updateViewer(
    viewer, 
    itr,  
    placing_handles, 
    R, 
    neighbors, 
    verts, 
    half_edges, 
    weights, 
    fix,  
    fix_vec,  
    RHS, 
    first,  
    moved,  
    res, 
    dir_solver
):
    
    YELLOW = np.array([1.0, 0.9, 0.2])
    GREEN = np.array([0.2, 0.6, 0.3])
    BLUE = np.array([0.2, 0.3, 0.8])
    ORANGE = np.array([1.0, 0.7, 0.2])

    if placing_handles:
        viewer.data().set_vertices(res)
        viewer.data().set_colors(YELLOW)
        CV = np.vstack(fix_vec)
        viewer.data().set_points(CV, GREEN)
    else:
        if moved[0]:
            if not first[0]:
                local_phase_deform(R, neighbors, verts, res, weights)
            global_phase(R, half_edges, weights, fix, fix_vec, RHS, DEFORM, first, res, dir_solver)
            if first[0]:
                first[0] = False
            for i in range(len(fix)):
                fix_vec[i] = res[fix[i]]  
            itr[0] += 1  

        viewer.data().set_vertices(res)
        viewer.data().set_colors(BLUE)
        CU = np.vstack(fix_vec)
        viewer.data().set_points(CU, ORANGE)

    viewer.data().compute_normals()

class CallbackMouseDown:
    def __init__(self, placing_handles, itr, last_mouse, res, fix, fix_vec, R, neighbors, verts, half_edges, weights, RHS, F, first, moved, changed, sel, dir_solver):
        self.placing_handles = placing_handles
        self.itr = itr
        self.last_mouse = last_mouse
        self.res = res
        self.fix = fix
        self.fix_vec = fix_vec
        self.R = R
        self.neighbors = neighbors
        self.verts = verts
        self.half_edges = half_edges
        self.weights = weights
        self.RHS = RHS
        self.F = F
        self.first = first
        self.moved = moved
        self.changed = changed
        self.sel = sel
        self.dir_solver = dir_solver

    def __call__(self, viewer, button, modifier):
        viewport = viewer.core().viewport
        self.last_mouse[0] = viewer.current_mouse_x
        self.last_mouse[1] = viewport[3] - viewer.current_mouse_y
        self.last_mouse[2] = 0.0

        if self.placing_handles[0]:
            fid, bc = igl.unproject_onto_mesh(
                self.last_mouse[:2], 
                viewer.core().view, 
                viewer.core().proj, 
                viewport, 
                self.res, 
                self.F
            )
            if fid != -1:
                c = np.argmax(bc)
                new_c = self.res[self.F[fid, c]]
                CV = np.vstack(self.fix_vec) if self.fix[0] else np.empty((0,3))
                if not self.fix[0] or np.min(np.linalg.norm(CV - new_c, axis=1)) > 1e-6:
                    self.fix[0].append(self.F[fid, c])
                    self.fix_vec[0].append(new_c.copy())
                    self.changed[0] = True
                    updateViewer(
                        viewer, self.itr[0], self.placing_handles[0], 
                        self.R, self.neighbors, self.verts, self.half_edges, 
                        self.weights, self.fix[0], self.fix_vec[0], self.RHS, 
                        self.first[0], self.moved[0], self.res, self.dir_solver
                    )
                    return True
        else:
            if self.fix[0]:
                CU = np.vstack(self.fix_vec[0])
                CP = igl.project(CU.astype(np.float32), 
                                 viewer.core().view, 
                                 viewer.core().proj, 
                                 viewport)
                if CP.size > 0:
                    D = np.linalg.norm(CP - self.last_mouse[:2], axis=1)
                    min_idx = np.argmin(D)
                    if D[min_idx] < 30:
                        self.sel[0] = min_idx
                        self.last_mouse[2] = CP[min_idx, 2]
                        return True
                    else:
                        self.sel[0] = -1
        return False

class CallbackMouseMove:
    def __init__(self, fix_vec, last_mouse, moved, sel, itr):
        self.fix_vec = fix_vec
        self.last_mouse = last_mouse
        self.moved = moved
        self.sel = sel
        self.itr = itr

    def __call__(self, viewer, dx, dy):
        if self.sel[0] != -1 and self.sel[0] < len(self.fix_vec):
            viewport = viewer.core().viewport
            drag_mouse = np.array([
                viewer.current_mouse_x,
                viewport[3] - viewer.current_mouse_y,
                self.last_mouse[2]
            ], dtype=np.float32)
            
            drag_scene = igl.unproject(
                drag_mouse, 
                viewer.core().view, 
                viewer.core().proj, 
                viewport
            )
            last_scene = igl.unproject(
                self.last_mouse, 
                viewer.core().view, 
                viewer.core().proj, 
                viewport
            )
            
            self.fix_vec[self.sel[0]] += (drag_scene - last_scene)
            self.last_mouse[:] = drag_mouse
            self.itr[0] = 0
            self.moved[0] = True
            return True
        return False

class CallbackKeyPressed:
    def __init__(self, placing_handles, itr, num, res, fix, fix_vec, R, neighbors, verts, half_edges, weights, RHS, F, first, moved, changed, Laplace, dir_solver, output_name):
        self.placing_handles[0] = not self.placing_handles[0]
        self.itr = itr
        self.num = num
        self.res = res
        self.fix = fix
        self.fix_vec = fix_vec
        self.R = R
        self.neighbors = neighbors
        self.verts = verts
        self.half_edges = half_edges
        self.weights = weights
        self.RHS = RHS
        self.F = F
        self.first = first
        self.Laplace = Laplace
        self.dir_solver = dir_solver
        self.moved = moved
        self.changed = changed
        self.output_name = output_name

    def __call__(self, viewer, key, modifiers):
        if key == ord('U') or key == ord('u'):
            updateViewer(
                viewer, self.itr[0], self.placing_handles, 
                self.R, self.neighbors, self.verts, self.half_edges, 
                self.weights, self.fix, self.fix_vec, self.RHS, 
                self.first[0], self.moved[0], self.res, self.dir_solver
            )
            return True
        elif key == ord(' '):
            self.placing_handles = not self.placing_handles
            if not self.placing_handles and self.fix:
                getLaplace(self.half_edges, self.weights, self.fix, self.Laplace, DEFORM)
                if self.changed[0]:
                    self.dir_solver = splu(self.Laplace.tocsc())
                    self.changed[0] = False
                else:
                    self.dir_solver = splu(self.Laplace.tocsc())
            updateViewer(
                viewer, self.itr[0], self.placing_handles, 
                self.R, self.neighbors, self.verts, self.half_edges, 
                self.weights, self.fix, self.fix_vec, self.RHS, 
                self.first[0], self.moved[0], self.res, self.dir_solver
            )
            return True
        elif key == ord('G') or key == ord('g'):
            printObjModel(self.res, self.F, self.output_name, self.num[0])
            self.num[0] += 1
            return True
        return False

class CallbackPreDraw:
    def __init__(self, placing_handles, itr, res, fix, fix_vec, R, neighbors, verts, half_edges, weights, RHS, F, first, moved, pause, inf_itr, dir_solver):
        self.placing_handles = placing_handles  
        self.itr = itr  
        self.res = res
        self.fix = fix
        self.fix_vec = fix_vec
        self.R = R
        self.neighbors = neighbors
        self.verts = verts
        self.half_edges = half_edges
        self.weights = weights
        self.RHS = RHS
        self.F = F
        self.first = first 
        self.moved = moved  
        self.pause = pause  
        self.inf_itr = inf_itr  
        self.dir_solver = dir_solver

    def __call__(self, viewer):
        if viewer.core().is_animating and not self.placing_handles[0]:
            if self.pause[0]:
                if not self.moved[0]:
                    updateViewer(
                        viewer, self.itr[0], self.placing_handles[0], 
                        self.R, self.neighbors, self.verts, self.half_edges, 
                        self.weights, self.fix, self.fix_vec, self.RHS, 
                        self.first[0], self.moved[0], self.res, self.dir_solver
                    )
            elif not self.inf_itr[0]:
                if self.itr[0] < 4:
                    updateViewer(
                        viewer, self.itr[0], self.placing_handles[0], 
                        self.R, self.neighbors, self.verts, self.half_edges, 
                        self.weights, self.fix, self.fix_vec, self.RHS, 
                        self.first[0], self.moved[0], self.res, self.dir_solver
                    )
            else:
                updateViewer(
                    viewer, self.itr[0], self.placing_handles[0], 
                    self.R, self.neighbors, self.verts, self.half_edges, 
                    self.weights, self.fix, self.fix_vec, self.RHS, 
                    self.first[0], self.moved[0], self.res, self.dir_solver
                )
        return False
