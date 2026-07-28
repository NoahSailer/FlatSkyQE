import numpy as np
from universe import UnivPlanck15
from halo_fit import Halofit
from weight import WeightLensSingle
from pn_2d import P2dAuto
from astropy.cosmology import FlatLambdaCDM



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

def build_fg_power_spectrum(ell_min=100, ell_max=1e5, alpha=2.0, A=1e-7):
    """
    Build a foreground power spectrum with a power-law form: C_L ~ A * (ell / ell_ref)^(-alpha).

    Parameters
    ----------
    ell_min : float
        Minimum ell for the power spectrum
    ell_max : float
        Maximum ell for the power spectrum
    alpha : float
        Power-law exponent (default 2.0)
    A : float
        Amplitude of the foreground power spectrum at reference ell (default 1e-7)

    Returns
    -------
    f_fg : callable
        Function that evaluates C_L^fg(ell) with amplitude scaling applied.
        Handles both single values and arrays.
    """

    ell_ref = 1000.0  # Reference ell for normalization

    def f_fg(ell):
        """Evaluate foreground power spectrum at given ell values."""
        ell = np.atleast_1d(ell)
        result = np.zeros_like(ell, dtype=float)

        within_range = (ell >= ell_min) & (ell <= ell_max)
        if np.any(within_range):
            result[within_range] = A * (ell[within_range] / ell_ref) ** (-alpha)

        beyond_max = ell > ell_max
        if np.any(beyond_max):
            result[beyond_max] = A * (ell_max / ell_ref) ** (-alpha)

        below_min = ell < ell_min
        if np.any(below_min):
            result[below_min] = A * (ell_min / ell_ref) ** (-alpha)

        return result

    return f_fg



def genGRF_from_Cl(self, fCl, seed=None):
    rng = np.random.default_rng(seed)
    A = self.sizeX * self.sizeY

    # full complex array shape for rfft grid
    shape = self.l.shape
    z = rng.normal(size=shape) + 1j * rng.normal(size=shape)

    # enforce pure-real modes on self-conjugate lines (ky=0 and Nyquist if present)
    z[:, 0] = rng.normal(size=self.nX) + 0j
    if self.nY % 2 == 0:
        z[:, -1] = rng.normal(size=self.nX) + 0j

    Cl = np.array(fCl(self.l))
    Cl = np.nan_to_num(Cl, nan=0.0, posinf=0.0, neginf=0.0)
    Cl[self.l == 0] = 0.0

    kfour = np.sqrt(0.5 * A * Cl) * z
    kfour[:, 0] *= np.sqrt(2.0)
    if self.nY % 2 == 0:
        kfour[:, -1] *= np.sqrt(2.0)

    return kfour



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

    kappa_fourier = baseMap.genGRF_rf(f_kappa_scaled, test=test)

    return kappa_fourier


def lensing_kernel_weights(z_s, z_bins):
    cosmo = FlatLambdaCDM(H0=70, Om0=0.28)
    chi_s = cosmo.comoving_distance(z_s).value  # Mpc
    weights = []

    for z in z_bins:
        chi = cosmo.comoving_distance(z).value
        a = 1.0 / (1 + z)
        if chi >= chi_s:
            weights.append(0.0)
        else:
            w = (chi / a) * (chi_s - chi) / chi_s
            weights.append(w)
    
    prefac = (3/2) * (cosmo.H0.value / 3e5)**2 * cosmo.Om0  # in units of 1/Mpc^2
    return prefac * np.array(weights)  


# def combine_kappa_fields(grf_list, z_bins, z_s=2.0):
    
#     weights = lensing_kernel_weights(z_s, z_bins)  # shape [n_slices]

#     weights /= np.sum(weights)
#     print('weights:', weights)
#     kappa_map = np.zeros_like(grf_list[0])
#     for delta_i, w_i in zip(grf_list, weights):
#         kappa_map += w_i * delta_i
    
#     return kappa_map

def combine_kappa_fields(grf_list, z_bins, z_s=2.0):
    weights = lensing_kernel_weights(z_s, z_bins)   # keep physical amplitude
    kappa_map = np.zeros_like(grf_list[0])
    for (one_plus_delta_i), w_i in zip(grf_list, weights):
        delta_i = one_plus_delta_i - 1.0
        delta_i = delta_i - np.mean(delta_i)
        kappa_map += w_i * delta_i
    return kappa_map

def get_lensing_fields_from_kappa(kappa_map, fmap):
    kappaF = fmap.fourier(kappa_map)
    dx, dy = fmap.deflectionFromKappa(kappaF)   # in map-coordinate units used by FlatMap
    mu = 1.0 + 2.0 * kappa_map                  # weak-lensing magnification
    return dx, dy, mu

def lens_positions_periodic(x, y, dx, dy, nx, ny):
    # nearest-pixel sample of deflection at source locations
    ix = np.mod(np.floor(x).astype(int), nx)
    iy = np.mod(np.floor(y).astype(int), ny)

    x_l = x + dx[iy, ix]
    y_l = y + dy[iy, ix]

    # periodic wrap
    x_l = np.mod(x_l, nx)
    y_l = np.mod(y_l, ny)
    return x_l, y_l