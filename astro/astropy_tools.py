"""
AstroPy astrophysics calculations.
Coordinate transforms, spectral analysis, cosmological distances, galaxy data.
SDKs: AstroPy, NumPy, SciPy
"""
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

try:
    from astropy import units as u, constants as const
    from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_body
    from astropy.time import Time
    from astropy.cosmology import FlatLambdaCDM, Planck18
    from astropy.io import fits
    from astropy.table import Table
    from astropy.visualization import ZScaleInterval
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False
    print("Warning: astropy not available. Install: pip install astropy")


@dataclass
class CelestialObject:
    name: str
    ra_deg: float         # Right Ascension in degrees
    dec_deg: float        # Declination in degrees
    distance_mpc: Optional[float] = None
    redshift: Optional[float] = None
    magnitude: Optional[float] = None


FAMOUS_OBJECTS = {
    "m31": CelestialObject("Andromeda Galaxy", 10.6847, 41.2692, distance_mpc=0.785, magnitude=3.44),
    "m87": CelestialObject("Messier 87", 187.7059, 12.3911, distance_mpc=16.4, redshift=0.00436),
    "crab": CelestialObject("Crab Nebula", 83.6220, 22.0145, distance_mpc=0.002),
    "sgr_a": CelestialObject("Sagittarius A*", 266.4168, -29.0078, distance_mpc=0.0083),
    "proxima": CelestialObject("Proxima Centauri", 217.4292, -62.6795, distance_mpc=1.295e-6),
}


class AstrophysicsTools:
    """
    AstroPy-based astrophysics calculations and coordinate transforms.
    """

    def __init__(self):
        if not ASTROPY_AVAILABLE:
            print("[AstroPy] Stub mode — astropy not installed")

    def object_info(self, target: str) -> Dict[str, Any]:
        """Get info for a named astronomical object."""
        key = target.lower().replace(" ", "_").replace("-", "_")
        obj = FAMOUS_OBJECTS.get(key)
        if not obj:
            return {"error": f"Unknown target: {target}. Try: {list(FAMOUS_OBJECTS.keys())}"}

        result = {
            "name": obj.name,
            "ra_deg": obj.ra_deg,
            "dec_deg": obj.dec_deg,
        }
        if obj.distance_mpc:
            result["distance_mpc"] = obj.distance_mpc
            result["distance_ly"] = obj.distance_mpc * 3.262e6
        if obj.magnitude:
            result["apparent_magnitude"] = obj.magnitude
        return result

    def coordinate_transform(
        self, ra_deg: float, dec_deg: float,
        frame_from: str = "icrs", frame_to: str = "galactic"
    ) -> Dict[str, float]:
        """Transform between coordinate frames."""
        if not ASTROPY_AVAILABLE:
            return {"error": "astropy not available"}
        coord = SkyCoord(ra=ra_deg*u.degree, dec=dec_deg*u.degree, frame=frame_from)
        transformed = coord.transform_to(frame_to)
        if frame_to == "galactic":
            return {"l_deg": float(transformed.l.deg), "b_deg": float(transformed.b.deg)}
        elif frame_to == "geocentrictrueecliptic":
            return {"lon_deg": float(transformed.lon.deg), "lat_deg": float(transformed.lat.deg)}
        return {"ra_deg": float(transformed.ra.deg), "dec_deg": float(transformed.dec.deg)}

    def angular_separation(
        self, ra1: float, dec1: float, ra2: float, dec2: float
    ) -> float:
        """Angular separation between two sky positions in degrees."""
        if ASTROPY_AVAILABLE:
            c1 = SkyCoord(ra=ra1*u.degree, dec=dec1*u.degree, frame="icrs")
            c2 = SkyCoord(ra=ra2*u.degree, dec=dec2*u.degree, frame="icrs")
            return float(c1.separation(c2).deg)
        # Haversine fallback
        r1, d1 = np.radians(ra1), np.radians(dec1)
        r2, d2 = np.radians(ra2), np.radians(dec2)
        return np.degrees(np.arccos(np.sin(d1)*np.sin(d2) + np.cos(d1)*np.cos(d2)*np.cos(r1-r2)))

    def cosmological_distances(self, redshifts: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute luminosity, comoving, and angular diameter distances."""
        if ASTROPY_AVAILABLE:
            cosmo = Planck18
            dl = cosmo.luminosity_distance(redshifts).to(u.Mpc).value
            dc = cosmo.comoving_distance(redshifts).to(u.Mpc).value
            da = cosmo.angular_diameter_distance(redshifts).to(u.Mpc).value
            return {"redshift": redshifts, "luminosity_Mpc": dl,
                    "comoving_Mpc": dc, "angular_diameter_Mpc": da}
        # Hubble law approximation
        H0 = 67.4  # km/s/Mpc
        c = 3e5    # km/s
        dc = c * redshifts / H0
        return {"redshift": redshifts, "comoving_Mpc": dc}

    def stellar_magnitude_flux(self, magnitudes: np.ndarray, m_ref: float = 0.0) -> np.ndarray:
        """Convert magnitudes to flux ratios relative to reference."""
        return 10**((m_ref - magnitudes) / 2.5)

    def planck_spectrum(
        self, temperature_K: float, wavelengths_nm: np.ndarray
    ) -> np.ndarray:
        """Planck blackbody spectrum B_lambda(T) in W/m^2/sr/nm."""
        if ASTROPY_AVAILABLE:
            from astropy.modeling.models import BlackBody
            bb = BlackBody(temperature=temperature_K * u.K)
            wav = wavelengths_nm * u.nm
            return bb(wav).to(u.W / u.m**2 / u.sr / u.nm).value
        h = 6.626e-34
        c = 3e8
        k = 1.381e-23
        lam = wavelengths_nm * 1e-9
        return (2*h*c**2 / lam**5) / (np.exp(h*c / (lam*k*temperature_K)) - 1) * 1e-9  # per nm
