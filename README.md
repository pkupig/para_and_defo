# Para And Defo

A lightweight teaching and research demo for triangle-mesh parameterization and mesh deformation.

The project provides:

- several parameterization methods: `ARAP`, `ASAP`, `Hybrid`, `LSCM`, `MVC`
- interactive deformation in the browser with `ARAP` and `Laplacian` deformation
- synchronized 3D mesh view and 2D parameter-domain view
- distortion metrics for parameterization and deformation

## Repository Layout

- `defor_param/app.py`: Flask backend and deformation session manager
- `defor_param/templates/app.html`: browser UI built with Three.js
- `defor_param/arap.py`, `lscm.py`, `mvc.py`: core geometry algorithms
- `defor_param/environment.yml`: reference Conda environment

## Environment

The code expects Python 3.10 and the following major dependencies:

- `flask`
- `flask_cors`
- `numpy`
- `scipy`
- `pillow`
- `igl`

Recommended setup:

```bash
cd defor_param
conda env create -f environment.yml
conda activate opencv
```

If you prefer an existing environment, make sure `flask_cors` is installed in addition to `flask`.

## Run

From the repository root:

```bash
cd defor_param
python app.py
```

The backend serves the UI at:

- `http://127.0.0.1:5000`

The app also starts a lightweight socket thread for live session data.

## How To Use

1. Upload a triangle mesh in `.obj` format.
2. Choose a parameterization method and click `执行参数化`.
3. Inspect the 2D parameter domain and the reported distortion metrics.
4. Click a vertex in the 3D view or 2D parameter view to select it.
5. Click `应用控制点并求解` to create a deformation handle and solve once.
6. Drag an existing handle in the 3D view to move it, then solve again.
7. Use `重置变形` to restore the original mesh and clear all handles.

## Interaction Notes

- The 3D view is for geometry inspection and handle dragging.
- The 2D view is for parameter-domain inspection and vertex picking.
- Boundary vertices in the 2D view are shown in red.
- Active or existing handles are highlighted in the interface.

## Current Behavior

This repository now includes a fix for the deformation visualization path:

- dragging a handle no longer directly overwrites mesh vertex positions on the frontend
- handle positions are maintained separately from the displayed mesh
- deformation reset now clears backend session state correctly
- vertex index `0` is accepted correctly when creating or updating handles

These changes make the system more reliable for classroom demos and research presentation.

## Known Limitations

- Input is limited to triangle meshes in `.obj` format.
- The UI currently depends on CDN-hosted Three.js assets.
- Large meshes may still be slow because the backend solves deformation on demand.
- The project structure is still prototype-like rather than production-packaged.

## Output Files

When parameterization runs, the backend may generate intermediate or result files such as:

- textures
- distortion text files
- `.vtk` exports
- `.obj` and `.mtl` outputs
- parameter-domain images

These runtime artifacts are ignored by `.gitignore`.
