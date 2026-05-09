"""
OpenMM molecular dynamics simulation.
Water box equilibration, energy minimization, NVT/NPT ensembles.
SDKs: OpenMM, NumPy
"""
import time
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass


@dataclass
class MDResult:
    system: str
    n_atoms: int
    n_steps: int
    final_potential_energy: float
    final_kinetic_energy: float
    temperature_K: float
    elapsed_sec: float
    trajectory_path: Optional[str] = None


class OpenMMSimulator:
    """
    OpenMM molecular dynamics for small molecule and protein systems.
    Supports AMBER, CHARMM, and OpenFF forcefields.
    """

    def __init__(self, platform: str = "CUDA"):
        self._available = self._check_openmm()
        self.platform_name = platform

    def _check_openmm(self) -> bool:
        try:
            import openmm
            return True
        except ImportError:
            print("[OpenMM] Not available. Install: conda install -c conda-forge openmm")
            return False

    def simulate_water_box(
        self,
        n_molecules: int = 216,
        temperature_K: float = 300.0,
        n_steps: int = 10_000,
        output_dir: str = "./md_output",
    ) -> MDResult:
        """
        Simulate a TIP3P water box — classic MD benchmark.
        """
        if self._available:
            return self._openmm_water(n_molecules, temperature_K, n_steps, output_dir)
        return self._stub_water(n_molecules, temperature_K, n_steps)

    def _openmm_water(self, n_molecules, temperature_K, n_steps, output_dir) -> MDResult:
        import openmm as mm
        import openmm.app as app
        import openmm.unit as unit

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        # Build water box using modeller
        forcefield = app.ForceField("tip3p.xml")
        modeller = app.Modeller(app.Topology(), [])
        modeller.addSolvent(forcefield, model="tip3p",
                           boxSize=mm.Vec3(3, 3, 3) * unit.nanometers)

        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * unit.nanometers,
            constraints=app.HBonds,
        )
        integrator = mm.LangevinMiddleIntegrator(
            temperature_K * unit.kelvin,
            1 / unit.picosecond,
            0.002 * unit.picoseconds,
        )

        try:
            platform = mm.Platform.getPlatformByName(self.platform_name)
        except Exception:
            platform = mm.Platform.getPlatformByName("CPU")

        simulation = app.Simulation(modeller.topology, system, integrator, platform)
        simulation.context.setPositions(modeller.positions)
        simulation.minimizeEnergy()

        dcd_path = str(Path(output_dir) / "trajectory.dcd")
        simulation.reporters.append(app.DCDReporter(dcd_path, 100))
        simulation.reporters.append(app.StateDataReporter(
            str(Path(output_dir) / "md.log"), 1000,
            step=True, time=True, potentialEnergy=True, temperature=True
        ))
        simulation.step(n_steps)

        state = simulation.context.getState(getEnergy=True)
        n_atoms = simulation.topology.getNumAtoms()
        elapsed = time.time() - t0

        print(f"[OpenMM] Water box: {n_atoms} atoms | {n_steps} steps | {elapsed:.1f}s")
        return MDResult(
            system="water_box", n_atoms=n_atoms, n_steps=n_steps,
            final_potential_energy=state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole),
            final_kinetic_energy=state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole),
            temperature_K=temperature_K, elapsed_sec=elapsed,
            trajectory_path=dcd_path,
        )

    def _stub_water(self, n_molecules, temperature_K, n_steps) -> MDResult:
        """Stub simulation when OpenMM is unavailable."""
        import time
        rng = np.random.default_rng(42)
        t0 = time.time()
        n_atoms = n_molecules * 3
        # Simulate energy trajectory
        pe = -1000.0
        for _ in range(n_steps // 100):
            pe += rng.normal(0, 2.0)
        elapsed = time.time() - t0
        print(f"[OpenMM-Stub] {n_atoms} atoms | {n_steps} steps | {elapsed:.3f}s")
        return MDResult(
            system="water_box_stub", n_atoms=n_atoms, n_steps=n_steps,
            final_potential_energy=pe, final_kinetic_energy=abs(pe) * 0.6,
            temperature_K=temperature_K, elapsed_sec=elapsed,
        )
