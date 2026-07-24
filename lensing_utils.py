import numpy as np
from universe import UnivPlanck15
from halo_fit import Halofit
from weight import WeightLensSingle
from pn_2d import P2dAuto


def build_kappa_power_spectrum(ell_min=100, ell_max=1e5, clkg_scale=1.0):
    """
    Build the lensing kappa power spectrum using theoretical predictions.

    Uses UnivPlanck15 cosmology with Halofit to compute C_L^kappa for CMB lensing.
    The amplitude can be tuned with clkg_scale parameter.

    Parameters
    ----------
    ell_min : float
        Minimum ell for the power spectrum
    ell_max : float
        Maximum ell for the power spectrum
    clkg_scale : float
        Amplitude scaling factor (default 1.0). Multiplies the full power spectrum.

    Returns
    -------
    f_kappa : callable
        Function that evaluates C_L^kappa(ell) with amplitude scaling applied.
        Handles both single values and arrays.
    """

    u = UnivPlanck15()
    halofit = Halofit(u, save=False)
    w_cmblens = WeightLensSingle(u, z_source=1100., name="cmblens")
    p2d_cmblens = P2dAuto(u, halofit, w_cmblens, save=False)

    lrange = np.arange(ell_min, int(ell_max) + 1)
    clk_base = p2d_cmblens.fPinterp(lrange)

    def f_kappa(ell):
        """Evaluate scaled kappa power spectrum at given ell values."""
        ell = np.atleast_1d(ell)
        result = np.zeros_like(ell, dtype=float)

        within_range = (ell >= ell_min) & (ell <= ell_max)
        if np.any(within_range):
            ell_int = np.rint(ell[within_range]).astype(int)
            ell_int = np.clip(ell_int, int(ell_min), int(ell_max))
            result[within_range] = clk_base[ell_int - int(ell_min)]

        beyond_max = ell > ell_max
        if np.any(beyond_max):
            result[beyond_max] = clk_base[-1]

        below_min = ell < ell_min
        if np.any(below_min):
            result[below_min] = clk_base[0]

        return clkg_scale * result

    return f_kappa


def generate_kappa_realization(baseMap, f_kappa, amplitude=1.0, seed=12345, test=False):
    """
    Generate a kappa realization from the lensing power spectrum.

    Parameters
    ----------
    baseMap : FlatMap
        FlatMap object with genGRF method
    f_kappa : callable
        Power spectrum function f_kappa(ell) returning C_L^kappa
    amplitude : float
        Amplitude scaling for the power spectrum (default 1.0)
    seed : int
        Seed for reproducibility (default 12345)
    test : bool
        Debug output flag

    Returns
    -------
    kappa_fourier : ndarray
        2D Fourier space kappa field (complex)
    """

    np.random.seed(seed)

    def f_kappa_scaled(ell):
        return amplitude * f_kappa(ell)

    kappa_fourier = baseMap.genGRF(f_kappa_scaled, test=test)

    return kappa_fourier
