# image.py

import numpy as np
from graph_process import normalize_to_one2D
from PIL import ImageDraw
from PIL import Image as Im
import os

class Image:

    def __init__(self, w, h):
        self.width = w
        self.height = h
        self.data = np.zeros((h, w, 3), dtype=np.uint8)  

    def __del__(self):
        pass  

    def Width(self):
        return self.width

    def Height(self):
        return self.height

    def GetPixel(self, x, y):
        assert 0 <= x < self.width and 0 <= y < self.height
        return self.data[y, x]

    def SetAllPixels(self, color):
        self.data[:, :] = color

    def SetPixel(self, x, y, color):
        assert 0 <= x < self.width and 0 <= y < self.height
        self.data[y, x] = color

    @staticmethod
    def LoadPPM(filename):
        assert filename is not None
        assert filename.endswith(".ppm")
        with open(filename, "rb") as file:
            header = file.readline().decode('utf-8').strip()
            assert header == "P6"
            comment = file.readline().decode('utf-8').strip()
            assert comment.startswith("#")
            width, height = map(int, file.readline().decode('utf-8').strip().split())
            max_val = int(file.readline().decode('utf-8').strip())
            assert max_val == 255
            data = np.frombuffer(file.read(), dtype=np.uint8)
            data = data.reshape((height, width, 3))
            img = Image(width, height)
            for y in range(height - 1, -1, -1):
                for x in range(width):
                    r, g, b = data[height - 1 - y, x]
                    img.SetPixel(x, y, np.array([r / 255.0, g / 255.0, b / 255.0]))
        return img

    def SavePPM(self, filename):
        assert filename is not None
        assert filename.endswith(".ppm")
        with open(filename, "wb") as file:
            file.write(b"P6\n")
            file.write(b"# Creator: Image::SavePPM()\n")
            file.write(f"{self.width} {self.height}\n".encode('utf-8'))
            file.write(b"255\n")
            for y in range(self.height - 1, -1, -1):
                for x in range(self.width):
                    r, g, b = self.GetPixel(x, y)
                    file.write(bytes([int(ClampColorComponent(r)),
                                      int(ClampColorComponent(g)),
                                      int(ClampColorComponent(b))]))

    def SaveTGA(self, filename):
        assert filename.endswith(".tga"), "Filename must end with .tga"
        with open(filename, "wb") as file:
            # Write header
            for i in range(18):
                if i == 2:
                    file.write(bytes([2]))  
                elif i == 12:
                    file.write(bytes([self.width % 256]))
                elif i == 13:
                    file.write(bytes([self.width // 256]))
                elif i == 14:
                    file.write(bytes([self.height % 256]))
                elif i == 15:
                    file.write(bytes([self.height // 256]))
                elif i == 16:
                    file.write(bytes([24]))  
                elif i == 17:
                    file.write(bytes([32]))  
                else:
                    file.write(bytes([0]))
            for y in range(self.height - 1, -1, -1):
                for x in range(self.width):
                    r, g, b = self.GetPixel(x, y)
                    file.write(bytes([b, g, r]))

    @staticmethod
    def LoadTGA(filename):
        assert filename.endswith(".tga"), "Filename must end with .tga"
        with open(filename, "rb") as file:
            for i in range(18):
                tmp = file.read(1)[0]
                if i == 12:
                    width = tmp
                elif i == 13:
                    width += 256 * tmp
                elif i == 14:
                    height = tmp
                elif i == 15:
                    height += 256 * tmp
            image = Image(width, height)
            for y in range(height - 1, -1, -1):
                for x in range(width):
                    b = file.read(1)[0]
                    g = file.read(1)[0]
                    r = file.read(1)[0]
                    image.SetPixel(x, y, [r, g, b])

        return image

    @staticmethod
    def Compare(img1, img2):
        assert img1.Width() == img2.Width()
        assert img1.Height() == img2.Height()

        img3 = Image(img1.Width(), img1.Height())
        for x in range(img1.Width()):
            for y in range(img1.Height()):
                color1 = img1.GetPixel(x, y)
                color2 = img2.GetPixel(x, y)
                color3 = np.abs(color1 - color2)
                img3.SetPixel(x, y, color3)
        return img3

"""    @staticmethod
    def LoadImage(filename):
        img = cv2.imread(filename, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Could not read the image: {filename}")
            assert False
        height, width = img.shape[:2]
        answer = Image(width, height)
        for y in range(height - 1, -1, -1):
            for x in range(width):
                b, g, r = img[y, x]
                reColor = np.array([r / 255.0, g / 255.0, b / 255.0])
                answer.SetPixel(x, y, reColor)
        return answer
"""

def ClampColorComponent(c):
    tmp = int(c * 255)
    tmp = max(0, min(tmp, 255))
    return tmp

def printVTK(half_edges, F, distortion, aread, angled, res, output_name):
    res_3d = np.c_[res, np.zeros(res.shape[0])] 
    output_name += ".vtk"
    with open(output_name, "w") as vtkfile:
        vtkfile.write("# vtk DataFile Version 3.0\n")
        vtkfile.write(f"{output_name}\n")
        vtkfile.write("ASCII\n")
        vtkfile.write("DATASET POLYDATA\n")
        vtkfile.write(f"POINTS {res.shape[0]} double\n")
        for i in range(res.shape[0]):
            vtkfile.write(f"{res_3d[i, 0]} {res_3d[i, 1]} {res_3d[i, 2]} \n")
        vtkfile.write("\n")
        vtkfile.write(f"POLYGONS {len(F)} {4 * len(F)}\n")
        for face in F:
            vtkfile.write(f"3 {face[0]} {face[1]} {face[2]}\n")
        vtkfile.write("\n")
        vtkfile.write(f"CELL_DATA {len(F)}\n")
        vtkfile.write("SCALARS distortion double 1\n")
        vtkfile.write("LOOKUP_TABLE distortion_table\n")
        for i in range(len(F)):
            vtkfile.write(f"{distortion[i]}\n")
        vtkfile.write("\n")
        vtkfile.write("SCALARS aread double 1\n")
        vtkfile.write("LOOKUP_TABLE aread_table\n")
        for i in range(len(F)):
            vtkfile.write(f"{aread[i]}\n")
        vtkfile.write("\n")
        vtkfile.write("SCALARS angled double 1\n")
        vtkfile.write("LOOKUP_TABLE angled_table\n")
        for i in range(len(F)):
            vtkfile.write(f"{angled[i]}\n")
        vtkfile.write("\n")


def twoDDDA(a, b, pic, r):
    distance = b - a
    dx = int(distance[0])
    dy = int(distance[1])
    steps = max(abs(dx), abs(dy))
    if steps == 0:
        return
    x, y = a[0], a[1]
    xinc = dx / steps
    yinc = dy / steps
    for _ in range(steps + 1):
        x_clamped = int(round(x))
        y_clamped = int(round(y))
        if 0 <= x_clamped < pic.width and 0 <= y_clamped < pic.height:
            pic.SetPixel(x_clamped, y_clamped, np.array([r, 0, 0], dtype=np.uint8))
        x += xinc
        y += yinc

def printImage(half_edges, res, output_name):
    width, height = 4000, 4000
    pic = Image(width, height)
    white = np.array([255, 255, 255], dtype=np.uint8)
    pic.SetAllPixels(white)
    normalized_res = normalize_to_one2D(res)
    screen_size = min(width, height)
    scale = screen_size - 1  
    center_x = width // 2
    center_y = height // 2
    for i in range(res.shape[0]):
        x_normalized = normalized_res[i, 0]
        y_normalized = normalized_res[i, 1]
        x = int(x_normalized * scale) - (scale // 2) + center_x
        y = int(y_normalized * scale) - (scale // 2) + center_y
        x_clamped = max(0, min(x, width - 1))
        y_clamped = max(0, min(y, height - 1))
        normalized_res[i] = [x_clamped, y_clamped]  
    vis = np.zeros(len(half_edges), dtype=bool)
    for i in range(len(half_edges)):
        if not vis[i]:
            a_idx = half_edges[i].Endpoints[0]
            b_idx = half_edges[i].Endpoints[1]
            a = normalized_res[a_idx]
            b = normalized_res[b_idx]
            is_boundary = (half_edges[i].InverseIdx == -1)
            color_val = 255 if is_boundary else 128
            twoDDDA(a, b, pic, color_val)
            vis[i] = True
            if half_edges[i].InverseIdx != -1:
                inv_idx = half_edges[i].InverseIdx
                vis[inv_idx] = True
    output_name += ".tga"
    pic.SaveTGA(output_name)


def printFile(print_pic, print_vtkfile, print_txtfile, print_each_frame, now_itr, distortion, aread, angled,
              output_name, distortion_per_unit, aread_per_unit, angled_per_unit, half_edges, F, res, distortion_file):
    itr = str(now_itr)
    if print_pic and print_each_frame:
        printImage(half_edges, res, f"{output_name}_{itr}")
    if print_vtkfile and print_each_frame:
        printVTK(half_edges, F, distortion_per_unit, aread_per_unit, angled_per_unit, res, f"{output_name}_{itr}")
    if print_txtfile:
        distortion_file.write(f"After the {now_itr}th iterations\n")
        distortion_file.write(f"Total Energy: {distortion}\n")
        distortion_file.write(f"Total Area Distortion: {aread}\n")
        distortion_file.write(f"Total Angle Distortion: {angled}\n\n")


def printObjModel(verts, res, F, output_name, edges, mtl_path, texture_path, grid_size=1.0, texture_size=(256, 256)):
    obj_filename = f"{output_name}.obj"
    create_checkerboard(texture_size, square_size=int(texture_size[0]/16), save_path=texture_path)
    
    mtl_dir = os.path.dirname(mtl_path)
    texture_rel_to_mtl = os.path.relpath(texture_path, mtl_dir)
    with open(mtl_path, "w") as mtlfile:
        mtlfile.write("newmtl Checkerboard\n")
        mtlfile.write(f"map_Kd {texture_rel_to_mtl}\n")
    
    obj_dir = os.path.dirname(obj_filename)
    mtl_rel_path = os.path.relpath(mtl_path, obj_dir)
    
    with open(obj_filename, "w") as Out:
        for v in range(verts.shape[0]):
            Out.write(f"v {verts[v][0]:.6f} {verts[v][1]:.6f} {verts[v][2]:.6f}\n")
        for i in range(res.shape[0]):
            Out.write(f"vt {res[i][0]:.6f} {res[i][1]:.6f}\n")
        Out.write(f"\nmtllib {mtl_rel_path}\n")
        Out.write("usemtl Checkerboard\n")
        for face in F:
            face_str = " ".join([f"{vtx+1}/{vtx+1}" for vtx in face])
            Out.write(f"f {face_str}\n")
        for edge in edges:
            Out.write(f"l {edge[0]+1} {edge[1]+1}\n")
    

def create_checkerboard(size=(256, 256), square_size=8, save_path="checkerboard.png"):
    img = Im.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for y in range(0, size[1], square_size):
        for x in range(0, size[0], square_size):
            if (x // square_size + y // square_size) % 2 == 0:
                draw.rectangle([x, y, x + square_size, y + square_size], fill="black")
    img.save(save_path)
        
    