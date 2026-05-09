"""
FEniCS (dolfinx) FEM solver.
Entry point per spec: solve the Poisson equation on a unit square domain,
output as numpy array and VTK file.
SDKs: FEniCS/dolfinx, NumPy, VTK
"""
import numpy as np
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
from dataclasses import dataclass


@dataclass
class FEMResult:
    problem: str
    n_dofs: int
    solution: np.ndarray
    mesh_coords: np.ndarray
    residual: float
    solve_time_sec: float
    vtk_path: Optional[str] = None


class FEniCSPoisson:
    """
    Solve the Poisson equation: -Δu = f on domain Ω, u = 0 on ∂Ω.
    Uses dolfinx (FEniCS v0.7+) with Lagrange elements.
    Falls back to a finite difference solver when dolfinx is unavailable.
    """

    def __init__(self, resolution: int = 32, degree: int = 1):
        self.resolution = resolution
        self.degree = degree
        self._dolfinx_available = self._check_dolfinx()

    def _check_dolfinx(self) -> bool:
        try:
            import dolfinx
            return True
        except ImportError:
            print("[FEniCS] dolfinx not available. Install: conda install -c conda-forge fenics-dolfinx")
            print("[FEniCS] Using finite difference fallback solver.")
            return False

    def solve_poisson(
        self,
        f_expr: str = "10*exp(-(pow(x[0]-0.5,2)+pow(x[1]-0.5,2))/0.02)",
        output_vtk: Optional[str] = None,
    ) -> FEMResult:
        """
        Solve -Δu = f on unit square [0,1]^2 with homogeneous Dirichlet BC.
        f_expr: source term expression (dolfinx syntax)
        """
        if self._dolfinx_available:
            return self._dolfinx_solve(f_expr, output_vtk)
        return self._fd_solve(output_vtk)

    def _dolfinx_solve(self, f_expr: str, output_vtk: Optional[str]) -> FEMResult:
        """Full dolfinx FEM solution."""
        import time
        from dolfinx import mesh, fem, plot
        from dolfinx.fem.petsc import LinearProblem
        from dolfinx.io import VTKFile
        from mpi4py import MPI
        import ufl
        from petsc4py import PETSc

        t0 = time.time()
        msh = mesh.create_unit_square(
            MPI.COMM_WORLD, self.resolution, self.resolution,
            mesh.CellType.triangle
        )
        V = fem.functionspace(msh, ("Lagrange", self.degree))

        # Boundary condition: u = 0
        def boundary(x):
            return (np.isclose(x[0], 0) | np.isclose(x[0], 1) |
                    np.isclose(x[1], 0) | np.isclose(x[1], 1))

        dofs = fem.locate_dofs_geometrical(V, boundary)
        bc = fem.dirichletbc(PETSc.ScalarType(0), dofs, V)

        # Variational form: a(u,v) = (grad u, grad v), L(v) = (f, v)
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        x = ufl.SpatialCoordinate(msh)
        f = eval(f"lambda x: {f_expr.replace('x[0]','x[0]').replace('x[1]','x[1]')}")(x)
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = f * v * ufl.dx

        problem = LinearProblem(a, L, bcs=[bc])
        uh = problem.solve()

        sol_array = uh.x.array.copy()
        coords = msh.geometry.x[:, :2]
        elapsed = time.time() - t0

        vtk_path = None
        if output_vtk:
            Path(output_vtk).parent.mkdir(parents=True, exist_ok=True)
            with VTKFile(MPI.COMM_WORLD, output_vtk, "w") as f_out:
                f_out.write_mesh(msh)
                f_out.write_function(uh)
            vtk_path = output_vtk

        print(f"[FEniCS] Solved Poisson | {len(sol_array)} DOFs | {elapsed:.3f}s")
        return FEMResult(
            problem="poisson", n_dofs=len(sol_array),
            solution=sol_array, mesh_coords=coords,
            residual=float(np.linalg.norm(sol_array)),
            solve_time_sec=elapsed, vtk_path=vtk_path,
        )

    def _fd_solve(self, output_vtk: Optional[str]) -> FEMResult:
        """
        Finite difference fallback: solve -Δu = f on [0,1]^2 with zero BC.
        5-point stencil, direct sparse solve via scipy.
        """
        import time
        from scipy.sparse import lil_matrix
        from scipy.sparse.linalg import spsolve

        n = self.resolution
        h = 1.0 / (n + 1)
        N = n * n
        t0 = time.time()

        # Build 5-point Laplacian
        A = lil_matrix((N, N))
        for i in range(n):
            for j in range(n):
                k = i * n + j
                A[k, k] = 4.0 / h**2
                if j > 0:     A[k, k-1] = -1.0 / h**2
                if j < n-1:   A[k, k+1] = -1.0 / h**2
                if i > 0:     A[k, k-n] = -1.0 / h**2
                if i < n-1:   A[k, k+n] = -1.0 / h**2

        # RHS: f = Gaussian source at center
        x_coords = np.linspace(h, 1-h, n)
        xx, yy = np.meshgrid(x_coords, x_coords)
        f = 10 * np.exp(-((xx-0.5)**2 + (yy-0.5)**2) / 0.02)
        b = f.ravel()

        u = spsolve(A.tocsr(), b)
        elapsed = time.time() - t0

        coords = np.column_stack([xx.ravel(), yy.ravel()])

        if output_vtk:
            self._write_vtk_fd(u, coords, output_vtk, n)

        print(f"[FEM-FD] Solved Poisson (FD) | {N} DOFs | {elapsed:.3f}s | max={u.max():.4f}")
        return FEMResult(
            problem="poisson_fd", n_dofs=N,
            solution=u, mesh_coords=coords,
            residual=float(np.linalg.norm(b - A.tocsr().dot(u))),
            solve_time_sec=elapsed, vtk_path=output_vtk,
        )

    def _write_vtk_fd(self, u: np.ndarray, coords: np.ndarray, path: str, n: int):
        """Write FD solution to a basic VTK file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write("# vtk DataFile Version 3.0
")
            f.write("Poisson FD Solution
ASCII
DATASET UNSTRUCTURED_GRID
")
            f.write(f"POINTS {len(coords)} float
")
            for x, y in coords:
                f.write(f"{x:.6f} {y:.6f} 0.0
")
            f.write(f"
POINT_DATA {len(u)}
")
            f.write("SCALARS u float 1
LOOKUP_TABLE default
")
            for val in u:
                f.write(f"{val:.6f}
")
        print(f"[FEM-FD] VTK written: {path}")

    def solve_heat_equation(
        self, t_end: float = 1.0, dt: float = 0.01,
        alpha: float = 0.01,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve heat equation: ∂u/∂t = α Δu
        Simple explicit finite differences in time, FD in space.
        Returns (time_array, solution_snapshots).
        """
        n = self.resolution
        h = 1.0 / (n + 1)
        r = alpha * dt / h**2
        if r > 0.5:
            print(f"[Heat] Warning: stability criterion violated (r={r:.3f} > 0.5). Reducing dt.")
            dt = 0.25 * h**2 / alpha
            r = alpha * dt / h**2

        x = np.linspace(h, 1-h, n)
        xx, yy = np.meshgrid(x, x)

        # Initial condition: Gaussian
        u = np.exp(-10 * ((xx-0.5)**2 + (yy-0.5)**2))
        times = [0.0]
        snapshots = [u.copy()]

        t = 0.0
        while t < t_end:
            u_new = u.copy()
            u_new[1:-1, 1:-1] = u[1:-1, 1:-1] + r * (
                u[2:, 1:-1] + u[:-2, 1:-1] +
                u[1:-1, 2:] + u[1:-1, :-2] - 4*u[1:-1, 1:-1]
            )
            u_new[0, :] = u_new[-1, :] = u_new[:, 0] = u_new[:, -1] = 0
            u = u_new
            t += dt
            if len(snapshots) % 10 == 0:
                snapshots.append(u.copy())
                times.append(t)

        print(f"[Heat] Solved {len(times)} time steps | max_u={u.max():.4f}")
        return np.array(times), np.array(snapshots)
