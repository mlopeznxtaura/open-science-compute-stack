"""
open-science-compute-stack — Entry Point

GPU-accelerated scientific computing: FEM, MD, fluid simulation, astrophysics.

Usage:
  python main.py --mode fem --problem poisson --resolution 64
  python main.py --mode md --system water --steps 10000
  python main.py --mode warp --sim fluid --particles 50000
  python main.py --mode astro --target m31
  python main.py --mode ode --system lorenz
  python main.py --mode demo
"""
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Open Science Compute Stack")
    parser.add_argument("--mode", required=True,
                        choices=["fem", "md", "warp", "astro", "ode", "sympy", "demo"])
    parser.add_argument("--problem", default="poisson", choices=["poisson", "heat"])
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--system", default="water",
                        choices=["water", "lorenz", "pendulum", "double_pendulum"])
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--sim", default="fluid", choices=["fluid"])
    parser.add_argument("--particles", type=int, default=10_000)
    parser.add_argument("--target", default="m31")
    parser.add_argument("--output", default="./output")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def mode_fem(args):
    from fem.fenics_solver import FEniCSPoisson
    import pathlib
    pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)

    solver = FEniCSPoisson(resolution=args.resolution)
    if args.problem == "poisson":
        result = solver.solve_poisson(output_vtk=f"{args.output}/poisson.vtu")
        print(f"
Poisson solved:")
        print(f"  DOFs: {result.n_dofs:,}")
        print(f"  Max value: {result.solution.max():.4f}")
        print(f"  Solve time: {result.solve_time_sec:.3f}s")
    elif args.problem == "heat":
        times, snapshots = solver.solve_heat_equation(t_end=1.0, dt=0.001)
        print(f"
Heat equation solved:")
        print(f"  Time steps: {len(times)}")
        print(f"  Final max temperature: {snapshots[-1].max():.4f}")


def mode_md(args):
    from md.openmm_sim import OpenMMSimulator
    sim = OpenMMSimulator()
    result = sim.simulate_water_box(n_steps=args.steps, output_dir=f"{args.output}/md")
    print(f"
MD simulation complete:")
    print(f"  Atoms: {result.n_atoms}")
    print(f"  Steps: {result.n_steps}")
    print(f"  Potential energy: {result.final_potential_energy:.1f} kJ/mol")
    print(f"  Time: {result.elapsed_sec:.1f}s")


def mode_warp(args):
    from gpu.warp_physics import WarpFluidSim
    sim = WarpFluidSim(n_particles=args.particles)
    result = sim.run(n_steps=500)
    print(f"
GPU particle sim complete:")
    print(f"  Particles: {result.n_particles:,}")
    print(f"  Steps: {result.n_steps}")
    print(f"  KE: {result.kinetic_energy:.2f}")
    print(f"  Time: {result.elapsed_sec:.2f}s")
    print(f"  Rate: {result.n_particles * result.n_steps / result.elapsed_sec / 1e6:.2f}M particle-steps/sec")


def mode_astro(args):
    from astro.astropy_tools import AstrophysicsTools
    astro = AstrophysicsTools()

    info = astro.object_info(args.target)
    print(f"
Object: {info.get('name', args.target)}")
    for k, v in info.items():
        if k != "name":
            print(f"  {k}: {v}")

    if "ra_deg" in info:
        gal = astro.coordinate_transform(info["ra_deg"], info["dec_deg"])
        print(f"  Galactic: l={gal.get('l_deg', '?'):.2f}°, b={gal.get('b_deg', '?'):.2f}°")

    z = np.array([0.01, 0.1, 0.5, 1.0, 2.0])
    dists = astro.cosmological_distances(z)
    print(f"
Cosmological distances:")
    for i, zi in enumerate(z):
        print(f"  z={zi}: d_L={dists['luminosity_Mpc'][i]:.1f} Mpc" if "luminosity_Mpc" in dists
              else f"  z={zi}: d_c={dists['comoving_Mpc'][i]:.1f} Mpc")


def mode_ode(args):
    from numerics.scipy_solvers import ScientificSolvers
    solvers = ScientificSolvers()

    if args.system == "lorenz":
        sol = solvers.lorenz_attractor()
        print(f"
Lorenz attractor:")
        print(f"  Points: {len(sol.t)}")
        print(f"  Final state: x={sol.y[0,-1]:.2f}, y={sol.y[1,-1]:.2f}, z={sol.y[2,-1]:.2f}")
    elif args.system == "double_pendulum":
        sol = solvers.double_pendulum()
        print(f"
Double pendulum:")
        print(f"  Points: {len(sol.t)}")
        print(f"  Success: {sol.success}")


def mode_sympy(args):
    from numerics.scipy_solvers import SymPyAnalysis
    sym = SymPyAnalysis()
    eqm = sym.derive_lagrangian()
    print(f"
Lagrangian mechanics:
  {eqm}")
    energies = sym.solve_schrodinger_box(n=5)
    print(f"
Particle in box eigenvalues:")
    for k, v in energies.items():
        print(f"  {k} = {v:.4f}")


def mode_demo(args):
    print("Running open science demo...
")
    mode_fem(args)
    mode_astro(args)
    mode_ode(args)
    print("
Demo complete.")


def main():
    args = parse_args()
    print("=" * 60)
    print("  Open Science Compute Stack")
    print(f"  Mode: {args.mode.upper()}")
    print("=" * 60)

    dispatch = {
        "fem": mode_fem,
        "md": mode_md,
        "warp": mode_warp,
        "astro": mode_astro,
        "ode": mode_ode,
        "sympy": mode_sympy,
        "demo": mode_demo,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
