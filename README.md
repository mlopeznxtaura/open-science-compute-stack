# Open Science Compute Stack

Cluster 20 of the NextAura 500 SDKs / 25 Clusters project.

Physics simulation, FEM solving, and CFD on GPU — research-grade, free. Open-source alternative to ANSYS, COMSOL, and MATLAB.

## Architecture

- FEniCS (dolfinx) for finite element method (FEM) solving
- OpenFOAM for computational fluid dynamics (CFD)
- OpenMM for molecular dynamics simulation
- GROMACS for biomolecular simulation
- NVIDIA Warp for GPU physics kernels
- JAX for differentiable scientific computing
- VTK for 3D scientific visualization
- SciPy + NumPy for numerical methods
- SymPy for symbolic mathematics
- AstroPy for astrophysics calculations
- Plotly Dash for interactive science dashboards

## SDKs Used

OpenFOAM SDK, FEniCS SDK, OpenMM SDK, GROMACS SDK, OpenSim SDK, VTK SDK, ITK SDK, NVIDIA Warp, CuPy SDK, JAX, NumPy, SciPy, SymPy, AstroPy SDK, Polars, DuckDB, Apache Arrow, Weights & Biases, FastAPI, Plotly Dash SDK

## Quickstart

```bash
pip install -r requirements.txt

# Solve Poisson equation with FEM
python main.py --mode fem --problem poisson --resolution 64

# Run molecular dynamics
python main.py --mode md --system water --steps 10000

# GPU physics with Warp
python main.py --mode warp --sim fluid --particles 100000

# Astrophysics calculations
python main.py --mode astro --target m31

# Launch science dashboard
python main.py --mode dash
```

## Structure

```
fem/
  fenics_solver.py    FEniCS FEM solver (Poisson, heat, elasticity)
  vtk_export.py       VTK 3D visualization export
md/
  openmm_sim.py       OpenMM molecular dynamics
  gromacs_runner.py   GROMACS biomolecular simulation wrapper
gpu/
  warp_physics.py     NVIDIA Warp GPU physics kernels
  jax_science.py      JAX differentiable scientific computing
numerics/
  scipy_solvers.py    SciPy ODE/PDE/optimization solvers
  sympy_analysis.py   SymPy symbolic mathematics
astro/
  astropy_tools.py    AstroPy astrophysics calculations
dashboard/
  dash_app.py         Plotly Dash interactive science viz
api/
  server.py           FastAPI compute API
main.py               Entry point
```
