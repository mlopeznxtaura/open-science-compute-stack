"""
SciPy numerical solvers: ODE integration, optimization, signal processing.
SDKs: SciPy, NumPy, SymPy
"""
import numpy as np
from typing import Callable, Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from scipy import integrate, optimize, signal, linalg, fft


@dataclass
class ODESolution:
    t: np.ndarray
    y: np.ndarray
    method: str
    n_steps: int
    success: bool
    message: str


class ScientificSolvers:
    """
    SciPy-based numerical methods for scientific computing.
    ODE integration, boundary value problems, optimization, signal processing.
    """

    # ---- ODE solvers ----

    def solve_ivp(
        self,
        f: Callable,
        t_span: Tuple[float, float],
        y0: np.ndarray,
        method: str = "RK45",
        rtol: float = 1e-6,
        atol: float = 1e-9,
        dense_output: bool = False,
    ) -> ODESolution:
        """Solve initial value problem dy/dt = f(t, y)."""
        result = integrate.solve_ivp(
            f, t_span, y0,
            method=method, rtol=rtol, atol=atol,
            dense_output=dense_output,
        )
        return ODESolution(
            t=result.t, y=result.y,
            method=method, n_steps=len(result.t),
            success=result.success, message=result.message,
        )

    def lorenz_attractor(
        self,
        sigma: float = 10.0, rho: float = 28.0, beta: float = 8/3,
        t_end: float = 50.0, y0: Optional[np.ndarray] = None,
    ) -> ODESolution:
        """Solve the Lorenz system — classic chaotic ODE."""
        if y0 is None:
            y0 = np.array([1.0, 1.0, 1.0])

        def lorenz(t, y):
            x, y_, z = y
            return [sigma*(y_-x), x*(rho-z)-y_, x*y_-beta*z]

        sol = self.solve_ivp(lorenz, (0, t_end), y0, method="RK45")
        print(f"[SciPy] Lorenz: {len(sol.t)} points | success={sol.success}")
        return sol

    def double_pendulum(
        self,
        m1: float = 1.0, m2: float = 1.0,
        L1: float = 1.0, L2: float = 1.0,
        t_end: float = 20.0,
    ) -> ODESolution:
        """Double pendulum equations of motion."""
        g = 9.81

        def equations(t, y):
            th1, w1, th2, w2 = y
            d = th2 - th1
            denom1 = (m1 + m2) * L1 - m2 * L1 * np.cos(d)**2
            denom2 = (L2/L1) * denom1

            dth1 = w1
            dw1 = (m2*L1*w1**2*np.sin(d)*np.cos(d) +
                   m2*g*np.sin(th2)*np.cos(d) +
                   m2*L2*w2**2*np.sin(d) -
                   (m1+m2)*g*np.sin(th1)) / denom1

            dth2 = w2
            dw2 = (-m2*L2*w2**2*np.sin(d)*np.cos(d) +
                   (m1+m2)*g*np.sin(th1)*np.cos(d) -
                   (m1+m2)*L1*w1**2*np.sin(d) -
                   (m1+m2)*g*np.sin(th2)) / denom2

            return [dth1, dw1, dth2, dw2]

        y0 = [np.pi/4, 0, np.pi/2, 0]
        return self.solve_ivp(equations, (0, t_end), y0, method="DOP853")

    # ---- Optimization ----

    def minimize_scalar(
        self, f: Callable, bounds: Tuple[float, float], method: str = "bounded"
    ) -> Dict[str, Any]:
        result = optimize.minimize_scalar(f, bounds=bounds, method=method)
        return {"x": result.x, "fun": result.fun, "success": result.success}

    def minimize_nd(
        self, f: Callable, x0: np.ndarray, method: str = "L-BFGS-B",
        bounds: Optional[List[Tuple]] = None,
    ) -> Dict[str, Any]:
        result = optimize.minimize(f, x0, method=method, bounds=bounds,
                                   options={"maxiter": 1000})
        return {"x": result.x, "fun": result.fun, "success": result.success,
                "n_iter": result.nit}

    # ---- Signal processing ----

    def spectral_analysis(
        self, signal_data: np.ndarray, sample_rate: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """FFT, PSD, and spectrogram of a 1D signal."""
        freqs = fft.rfftfreq(len(signal_data), 1/sample_rate)
        spectrum = np.abs(fft.rfft(signal_data))
        f_psd, psd = signal.welch(signal_data, fs=sample_rate, nperseg=256)
        return {"freqs": freqs, "spectrum": spectrum, "psd_freqs": f_psd, "psd": psd}

    def bandpass_filter(
        self, data: np.ndarray, low_hz: float, high_hz: float, fs: float
    ) -> np.ndarray:
        sos = signal.butter(4, [low_hz, high_hz], btype="band", fs=fs, output="sos")
        return signal.sosfiltfilt(sos, data)


class SymPyAnalysis:
    """SymPy symbolic mathematics for physics derivations."""

    def __init__(self):
        import sympy as sp
        self.sp = sp

    def derive_lagrangian(self) -> str:
        """Derive equations of motion for simple pendulum via Lagrangian mechanics."""
        sp = self.sp
        t = sp.Symbol("t")
        m, L, g = sp.symbols("m L g", positive=True)
        theta = sp.Function("theta")(t)

        T = sp.Rational(1, 2) * m * L**2 * sp.diff(theta, t)**2
        V = -m * g * L * sp.cos(theta)
        Lagrangian = T - V

        EL = (sp.diff(sp.diff(Lagrangian, sp.diff(theta, t)), t) -
              sp.diff(Lagrangian, theta))
        eq = sp.simplify(EL)
        return f"Equation of motion: {eq} = 0"

    def solve_schrodinger_box(self, n: int = 3, L_val: float = 1.0) -> Dict:
        """Energy eigenvalues for particle in 1D box (analytical)."""
        sp = self.sp
        n_sym, m, hbar, L = sp.symbols("n m hbar L", positive=True)
        E_n = (n_sym * sp.pi * hbar)**2 / (2 * m * L**2)
        energies = {}
        for i in range(1, n+1):
            energies[f"E_{i}"] = float(E_n.subs([(n_sym, i), (m, 1), (hbar, 1), (L, L_val)]))
        return energies
