"""
NVIDIA Warp GPU physics kernels for scientific computing.
Fluid particles, rigid bodies, and custom physics simulations.
SDKs: NVIDIA Warp, NumPy, CuPy
"""
import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

try:
    import warp as wp
    wp.init()
    WARP_AVAILABLE = True
except ImportError:
    WARP_AVAILABLE = False
    print("Warning: NVIDIA Warp not available. Install: pip install warp-lang")

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


@dataclass
class ParticleSimResult:
    n_particles: int
    n_steps: int
    final_positions: np.ndarray
    final_velocities: np.ndarray
    kinetic_energy: float
    elapsed_sec: float


class WarpFluidSim:
    """
    GPU-accelerated SPH (Smoothed Particle Hydrodynamics) fluid simulation.
    Uses NVIDIA Warp for kernel execution.
    """

    def __init__(self, n_particles: int = 10_000, device: str = "cuda"):
        self.n = n_particles
        self.device = device if WARP_AVAILABLE else "cpu"

        if WARP_AVAILABLE:
            self._init_warp()
        else:
            self._init_numpy()

        print(f"[WarpFluid] {n_particles} particles on {self.device}")

    def _init_warp(self):
        """Initialize Warp arrays for particle state."""
        rng = np.random.default_rng(42)
        pos_np = rng.uniform(0, 1, (self.n, 3)).astype(np.float32)
        vel_np = rng.normal(0, 0.1, (self.n, 3)).astype(np.float32)

        self.pos = wp.array(pos_np, dtype=wp.vec3, device=self.device)
        self.vel = wp.array(vel_np, dtype=wp.vec3, device=self.device)
        self.force = wp.zeros(self.n, dtype=wp.vec3, device=self.device)

    def _init_numpy(self):
        """CPU fallback particle state."""
        rng = np.random.default_rng(42)
        self.pos_np = rng.uniform(0, 1, (self.n, 3)).astype(np.float32)
        self.vel_np = rng.normal(0, 0.1, (self.n, 3)).astype(np.float32)

    def step(self, dt: float = 0.001, gravity: float = -9.81):
        """Advance simulation by one time step."""
        if WARP_AVAILABLE:
            self._warp_step(dt, gravity)
        else:
            self._numpy_step(dt, gravity)

    def _numpy_step(self, dt: float, gravity: float):
        """CPU fallback Euler integration."""
        g = np.array([0, gravity, 0], dtype=np.float32)
        self.vel_np += g * dt
        self.pos_np += self.vel_np * dt
        # Boundary reflection
        for dim in range(3):
            mask_lo = self.pos_np[:, dim] < 0
            mask_hi = self.pos_np[:, dim] > 1
            self.vel_np[mask_lo, dim] *= -0.8
            self.vel_np[mask_hi, dim] *= -0.8
            self.pos_np[:, dim] = np.clip(self.pos_np[:, dim], 0, 1)

    def _warp_step(self, dt: float, gravity: float):
        """Warp GPU kernel step."""
        g_vec = wp.vec3(0.0, gravity, 0.0)

        @wp.kernel
        def integrate(
            pos: wp.array(dtype=wp.vec3),
            vel: wp.array(dtype=wp.vec3),
            dt: float,
            gravity: wp.vec3,
        ):
            tid = wp.tid()
            v = vel[tid] + gravity * dt
            p = pos[tid] + v * dt
            # Boundary reflection
            for i in range(3):
                if p[i] < 0.0:
                    p[i] = 0.0
                    v[i] = wp.abs(v[i]) * 0.8
                if p[i] > 1.0:
                    p[i] = 1.0
                    v[i] = -wp.abs(v[i]) * 0.8
            pos[tid] = p
            vel[tid] = v

        wp.launch(integrate, dim=self.n, inputs=[self.pos, self.vel, dt, g_vec])

    def run(self, n_steps: int = 1000) -> ParticleSimResult:
        """Run simulation for n_steps and return result."""
        import time
        t0 = time.time()
        for _ in range(n_steps):
            self.step()
        elapsed = time.time() - t0

        if WARP_AVAILABLE:
            pos_np = self.pos.numpy()
            vel_np = self.vel.numpy()
        else:
            pos_np = self.pos_np
            vel_np = self.vel_np

        ke = 0.5 * float(np.sum(vel_np**2))
        print(f"[WarpFluid] {n_steps} steps | {elapsed:.2f}s | KE={ke:.2f}")
        return ParticleSimResult(
            n_particles=self.n, n_steps=n_steps,
            final_positions=pos_np, final_velocities=vel_np,
            kinetic_energy=ke, elapsed_sec=elapsed,
        )


class JAXScienceKernels:
    """
    JAX-based differentiable scientific computing.
    Automatic differentiation through physics simulations.
    SDKs: JAX
    """

    def __init__(self):
        try:
            import jax
            import jax.numpy as jnp
            self.jax = jax
            self.jnp = jnp
            print(f"[JAX] Available | devices: {jax.devices()}")
            self._available = True
        except ImportError:
            print("[JAX] Not available. Install: pip install jax[cpu]")
            self._available = False

    def solve_ode(
        self,
        f,  # dy/dt = f(t, y)
        y0: np.ndarray,
        t_span: Tuple[float, float],
        n_steps: int = 1000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Solve ODE using JAX-accelerated RK4."""
        t0, t1 = t_span
        dt = (t1 - t0) / n_steps
        ts = np.linspace(t0, t1, n_steps + 1)

        if self._available:
            import jax.numpy as jnp
            from jax import jit

            @jit
            def rk4_step(y, t, dt):
                k1 = f(t, y)
                k2 = f(t + dt/2, y + dt/2 * k1)
                k3 = f(t + dt/2, y + dt/2 * k2)
                k4 = f(t + dt, y + dt * k3)
                return y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

            y = jnp.array(y0, dtype=jnp.float32)
            ys = [np.array(y)]
            for t in ts[:-1]:
                y = rk4_step(y, t, dt)
                ys.append(np.array(y))
        else:
            y = y0.copy()
            ys = [y.copy()]
            for t in ts[:-1]:
                k1 = f(t, y)
                k2 = f(t + dt/2, y + dt/2 * k1)
                k3 = f(t + dt/2, y + dt/2 * k2)
                k4 = f(t + dt, y + dt * k3)
                y = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
                ys.append(y.copy())

        return ts, np.array(ys)

    def grad_of_loss(self, loss_fn, params: np.ndarray) -> np.ndarray:
        """Compute gradient of a loss function w.r.t. parameters using JAX autograd."""
        if not self._available:
            h = 1e-5
            return np.array([(loss_fn(params + h*np.eye(len(params))[i]) -
                              loss_fn(params - h*np.eye(len(params))[i])) / (2*h)
                             for i in range(len(params))])
        from jax import grad
        import jax.numpy as jnp
        p_jax = jnp.array(params)
        return np.array(grad(loss_fn)(p_jax))
