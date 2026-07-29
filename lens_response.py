import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.optimize import curve_fit


def C_ell(ell, A, B, n):
    """C_ell = A + B * ell^n"""
    return A + B * ell**n


def log_derivs(ell, A, B, n):
    """
    Returns (s, s2) where
        s  = d ln C / d ln ell
        s2 = d^2 ln C / (d ln ell)^2
    """
    Ce = C_ell(ell, A, B, n)
    s  = n * B * ell**n / Ce
    s2 = s * (n - s)
    return s, s2

def response_components(ellx, elly, L, A, B, n):
    """
    Compute the isotropic and quadrupolar pieces of f_{ell, L-ell}.

    Parameters
    ----------
    ellx, elly : 2D arrays
        Fourier-space grid coordinates.
    L : float
        Large-scale multipole magnitude; taken along +x, so L_vec = (L, 0).
    A, B, n : float
        Power spectrum parameters: C_ell = A + B * ell^n.

    Returns
    -------
    f_iso      : 2D array, isotropic (m=0) response.
    f_quad_2d  : 2D array, full quadrupolar field = cos(2theta) * f_quad_amp.
    f_quad_amp : 2D array, amplitude of the cos(2theta) piece (useful for 
                 understanding the shear-only weight function).
    """
    ell = np.sqrt(ellx**2 + elly**2)
    ell = np.maximum(ell, 1.0)          # floor to avoid division by zero

    cos_2th = (ellx**2 - elly**2) / ell**2

    Ce       = C_ell(ell, A, B, n)
    s, s2    = log_derivs(ell, A, B, n)

    # D+C and D-C (as absolute quantities, not divided by C)
    Dplus  = Ce * (s  + s**2 + s2)     # ell^2 d^2C/dell^2 + dC/dlnell
    Dminus = Ce * (s**2 + s2 - s)      # ell^2 d^2C/dell^2 - dC/dlnell

    ratio = L / ell                     # L / |ell|

    # Isotropic piece: constant-in-angle terms from Appendix B Eq. B1
    #   first bracket:  2 + s  (cos2theta*s goes to f_quad)
    #   second bracket: D+C    (4s*costheta averages to zero; cos2theta*D- to f_quad)
    f_iso = Ce * (2.0 + s) - 0.5 * ratio * Dplus

    # Quadrupolar amplitude: coefficient of cos(2theta)
    #   first bracket:  +C * s
    #   second bracket: -1/2*(L/ell)*D- C
    f_quad_amp = Ce * s - 0.5 * ratio * Dminus

    f_quad_2d = cos_2th * f_quad_amp

    return f_iso, f_quad_2d, f_quad_amp

def response_powerlaw_spin2_only(ellx, elly, L, B, n):
    """
    The pure spin-2 (cos 2θ) component extracted from the same model.

    Keep only cos2θ terms:
      f_spin2 ≈ Bℓ^n * [ n*cos2θ  - 1/2*(L/ℓ)*n(n-2)*cos2θ ].

    This highlights the quadrupolar/shear-like response pattern.
    """
    ell = np.sqrt(ellx**2 + elly**2)
    ell = np.maximum(ell, 1e-12)

    cos_2th = (ellx**2 - elly**2) / ell**2

    return (B * ell**n) * (n * cos_2th - 0.5 * (L / ell) * n * (n - 2.0) * cos_2th)


def both_legs_mask(ellx, elly, L, ellmin, ellmax):
    """
    Boolean mask: True where both |ell| and |L - ell| lie in [ellmin, ellmax].
    L is taken along +x so L_vec = (L, 0).
    """
    ell1 = np.sqrt(ellx**2          + elly**2)
    ell2 = np.sqrt((L - ellx)**2    + elly**2)
    in1  = (ell1 >= ellmin) & (ell1 <= ellmax)
    in2  = (ell2 >= ellmin) & (ell2 <= ellmax)
    return in1 & in2


def make_ell_grid(ellmax, N):
    ax = np.linspace(-ellmax, ellmax, N)
    ellx, elly = np.meshgrid(ax, ax, indexing="xy")
    return ax, ellx, elly


def annulus_mask(ellx, elly, ellmin, ellmax):
    ell = np.sqrt(ellx**2 + elly**2)
    return (ell >= ellmin) & (ell <= ellmax)


def plot_response_2d(ax_grid, f2d, title, L, ellmin=None, ellmax=None, cmap="RdBu_r"):
    # Symmetric-ish color scaling around 0 (robust to outliers)
    finite = np.isfinite(f2d)
    if not np.any(finite):
        raise ValueError("No finite pixels to plot (mask too aggressive?)")
    v = np.nanpercentile(np.abs(f2d[finite]), 99.0)
    norm = TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        f2d,
        origin="lower",
        extent=[ax_grid[0], ax_grid[-1], ax_grid[0], ax_grid[-1]],
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        aspect="equal",
    )
    cb = plt.colorbar(im, ax=ax)
    cb.set_label(r"$f_{\ell,\,\mathbf{L}-\ell}$ (arb. units)")

    ax.set_xlabel(r"$\ell_x$")
    ax.set_ylabel(r"$\ell_y$")
    ax.set_title(title)

    # Show L direction
    ax.arrow(0, 0, L, 0, width=0.0, head_width=0.03 * ax_grid[-1], head_length=0.05 * ax_grid[-1],
             length_includes_head=True, color="k")
    ax.text(0.02 * ax_grid[-1], 0.05 * ax_grid[-1], rf"$\mathbf{{L}}=({L},0)$", color="k")

    # Optional circles for annulus
    if ellmin is not None and ellmax is not None:
        th = np.linspace(0, 2*np.pi, 400)
        for r, ls in [(ellmin, "--"), (ellmax, "--")]:
            ax.plot(r*np.cos(th), r*np.sin(th), "k", lw=1, ls=ls, alpha=0.6)

    plt.tight_layout()
    return fig, ax


def diverging_imshow(ax, data, extent, vmax=None, cmap="RdBu_r",
                     cbar_label="", title="", text_fs=14):
    finite = data[np.isfinite(data)]
    if vmax is None:
        vmax = np.percentile(np.abs(finite), 99.5) if len(finite) else 1.0

    norm     = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="0.88")

    im = ax.imshow(
        data,
        origin="lower",
        extent=extent,
        cmap=cmap_obj,
        norm=norm,
        interpolation="nearest",
        aspect="equal",
    )

    cb = ax.figure.colorbar(
        im, ax=ax,
        orientation="horizontal",
        location="top",
        fraction=0.046,
        pad=0.12,
    )
    cb.set_label(cbar_label, fontsize=10)

    # Text in top-left corner, in axes coordinates (0,0)=bottom-left (1,1)=top-right
    ax.text(
        0.03, 0.97, title,
        transform=ax.transAxes,
        fontsize=text_fs,
        va="top", ha="left",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=2.0),
    )

    return im

def add_annulus_and_arrow(ax, L, ellmin, ellmax, xmax):
    """
    Overlay dashed annulus circles (upper half only) and an arrow for L.
    """
    th = np.linspace(0.0, np.pi, 400)
    for r in [ellmin, ellmax]:
        ax.plot(r * np.cos(th), r * np.sin(th),
                color="k", lw=0.8, ls="--", alpha=0.55)

    arrow_len = 0.13 * xmax
    ax.annotate(
        "", xy=(arrow_len, 0.0), xytext=(0.0, 0.0),
        arrowprops=dict(arrowstyle="->", color="k", lw=1.5)
    )
    ax.text(arrow_len * 1.06, xmax * 0.025,
            r"$\mathbf{L}$", fontsize=12, va="bottom")
    

def C_ell(ell, A, B, n):
    return A + B * ell**n

def log_derivs(ell, A, B, n):
    Ce = C_ell(ell, A, B, n)
    s  = n * B * ell**n / Ce
    s2 = s * (n - s)
    return s, s2

def fit_smooth_spectrum(lb, cl, model='powerlaw_plus_shot', n_components=1,
                        smooth_factor=None, degree=2):
    """
    Fit a smooth parametric model to binned power spectrum data and return
    a callable cell_smooth(ell).

    Parameters
    ----------
    lb, cl         : 1-D arrays   bandpower centres and values.
    model          : str
        'powerlaw_plus_shot'  — C = A + B * ell^n  (3 params, default)
        'multi_powerlaw'      — C = A + sum_i B_i * ell^{n_i}  (shot + n_components power laws)
        'logpoly_plus_shot'   — Estimate A_shot from high-ell bins, then fit
                                log(C - A_shot) as a degree-d polynomial in log ell
        'spline'              — Smooth univariate spline on log C vs log ell
    n_components   : int    number of independent power-law components (multi_powerlaw only)
    smooth_factor  : float  UnivariateSpline smoothing parameter s (spline only; None = auto)
    degree         : int    polynomial degree (logpoly_plus_shot only)

    Returns
    -------
    cell_smooth : callable   ell -> C_ell  (extrapolates as power law outside data range)
    params      : dict       fitted parameter values
    """
    lb = np.asarray(lb, dtype=float)
    cl = np.asarray(cl, dtype=float)

    if model == 'powerlaw_plus_shot':
        def _fn(ell, A, B, n):
            return A + B * ell**n
        A0 = np.percentile(cl, 10)          # rough shot-noise floor
        popt, _ = curve_fit(_fn, lb, cl,
                            p0=[A0, cl[0], -2.0],
                            bounds=([0, 0, -6], [np.inf, np.inf, 0]),
                            maxfev=20000)
        A, B, n_fit = popt
        Bc = B  # capture for lambda
        nf = n_fit
        cell_smooth = lambda ell, _A=A, _B=Bc, _n=nf: _A + _B * np.maximum(ell, 1.0)**_n
        params = dict(A=A, B=B, n=n_fit)

    elif model == 'multi_powerlaw':
        def _fn(ell, *p):
            out = np.full_like(ell, p[0], dtype=float)          # shot noise A
            for k in range(n_components):
                out += p[1 + 2*k] * ell**p[2 + 2*k]
            return out
        A0 = np.percentile(cl, 10)
        p0    = [A0] + [cl[0] / n_components, -2.0] * n_components
        lo    = [0]  + [0,        -6]  * n_components
        hi    = [np.inf] + [np.inf, 0] * n_components
        popt, _ = curve_fit(_fn, lb, cl, p0=p0, bounds=(lo, hi), maxfev=40000)
        A  = popt[0]
        Bs = popt[1::2].copy()
        ns = popt[2::2].copy()
        def cell_smooth(ell, _A=A, _Bs=Bs, _ns=ns):
            ell = np.maximum(np.asarray(ell, dtype=float), 1.0)
            out = np.full_like(ell, _A)
            for _B, _n in zip(_Bs, _ns):
                out = out + _B * ell**_n
            return out
        params = dict(A=A, Bs=Bs, ns=ns)

    elif model == 'logpoly_plus_shot':
        # Estimate shot-noise floor from the flattest high-ell quartile
        A_shot = np.percentile(cl[-max(len(cl)//4, 3):], 25)
        cl_sub = cl - A_shot
        # Only fit bins where clustering term > 20% of total (avoids log(-eps) divergence)
        mask   = cl_sub > 0.2 * cl
        if mask.sum() < degree + 1:
            mask = cl_sub > 0  # fallback: all positive-residual bins
        coeffs = np.polyfit(np.log(lb[mask]), np.log(cl_sub[mask]), degree)
        poly   = np.poly1d(coeffs)
        def cell_smooth(ell, _A=A_shot, _p=poly):
            ell = np.maximum(np.asarray(ell, dtype=float), 1.0)
            return _A + np.maximum(np.exp(_p(np.log(ell))), 0.0)
        params = dict(A_shot=A_shot, poly_coeffs=coeffs, mask=mask)

    elif model == 'spline':
        lnl = np.log(lb)
        lnc = np.log(cl)
        spl = UnivariateSpline(lnl, lnc, k=3, s=smooth_factor)
        def cell_smooth(ell, _spl=spl):
            return np.exp(_spl(np.log(np.maximum(ell, 1.0))))
        params = dict(spline=spl)

    else:
        raise ValueError(
            f"Unknown model '{model}'. "
            "Choose from 'powerlaw_plus_shot', 'multi_powerlaw', "
            "'logpoly_plus_shot', 'spline'."
        )

    return cell_smooth, params


def load_ciber_f25_spectrum(inst=1, basepath=None, smooth=False,
                            smooth_model='powerlaw_plus_shot', fpath=None, **smooth_kwargs):
    """
    Load the CIBER F25B field-averaged auto power spectrum for TM1 (J, inst=1)
    or TM2 (H, inst=2).  Returns a callable cell_interp(ell) and the raw arrays.

    Parameters
    ----------
    inst         : int   1 = 1.1 um / J-band (TM1),  2 = 1.8 um / H-band (TM2)
    basepath     : str   directory containing ciber_auto_*.npz files; defaults to
                         the ciber repo data/ folder relative to this file.
    smooth       : bool  If True, replace the raw log-log linear interpolation with
                         a smooth fitted model (no jagged bin artefacts).
    smooth_model : str   One of:
                         'powerlaw_plus_shot'  C = A + B*ell^n   (default)
                         'multi_powerlaw'       C = A + sum B_i*ell^{n_i}
                         'logpoly_plus_shot'    polynomial in log-log after subtracting shot noise
                         'spline'               smooth UnivariateSpline in log-log
    **smooth_kwargs      Extra keyword arguments forwarded to fit_smooth_spectrum()
                         (e.g. n_components=2, degree=3, smooth_factor=0.5).

    Returns
    -------
    cell_interp : callable  ell -> C_ell
    lb          : 1-D array bandpower centres
    cl          : 1-D array field-averaged C_ell values
    """
    import os
    band = {1: 'J', 2: 'H'}[inst]
    if basepath is None:
        basepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'data')
    if fpath is not None:
        fname = fpath
    else:
        fname = f'../data/feder25_ciber_spitzer/ciber_auto_{band}lt16.0_F25B.npz'

    d  = np.load(fname)
    lb = d['lb']
    cl = d['fieldav_cl']

    if smooth:
        cell_interp, _ = fit_smooth_spectrum(lb, cl, model=smooth_model, **smooth_kwargs)
    else:
        # Raw log-log linear interpolation (power-law between bins)
        _log_interp = interp1d(np.log(lb), np.log(cl), kind='linear',
                               bounds_error=False, fill_value='extrapolate')
        cell_interp = lambda ell: np.exp(_log_interp(np.log(np.maximum(ell, 1.0))))

    return cell_interp, lb, cl


def log_derivs_numerical(ell, cell_interp, dlnell=0.005):
    """
    Numerical log-derivatives of a tabulated spectrum.

    Returns (s, s2) via central finite differences on ln(ell):
        s  = d ln C / d ln ell
        s2 = d^2 ln C / (d ln ell)^2
    """
    ell = np.maximum(np.asarray(ell, dtype=float), 1.0)
    ell_p = ell * np.exp( dlnell)
    ell_m = ell * np.exp(-dlnell)
    lnC_p = np.log(cell_interp(ell_p))
    lnC_0 = np.log(cell_interp(ell))
    lnC_m = np.log(cell_interp(ell_m))
    s  = (lnC_p - lnC_m) / (2.0 * dlnell)
    s2 = (lnC_p - 2.0 * lnC_0 + lnC_m) / dlnell**2
    return s, s2


def response_amplitudes_1d(ell, L, A, B, n, cell_interp=None):
    """
    Returns f_iso(ell,L) and f_quad_amp(ell,L) as 1D functions of |ell|.
    The full 2D quadrupolar field is cos(2theta)*f_quad_amp.

    If cell_interp is provided (a callable returning C(ell)) it overrides A, B, n.
    """
    ell  = np.maximum(ell, 1.0)
    if cell_interp is not None:
        Ce    = cell_interp(ell)
        s, s2 = log_derivs_numerical(ell, cell_interp)
    else:
        Ce    = C_ell(ell, A, B, n)
        s, s2 = log_derivs(ell, A, B, n)

    Dplus  = Ce * (s  + s**2 + s2)
    Dminus = Ce * (s**2 + s2 - s)

    ratio      = L / ell
    f_iso      = Ce * (2.0 + s) - 0.5 * ratio * Dplus
    f_quad_amp = Ce * s          - 0.5 * ratio * Dminus

    return f_iso, f_quad_amp

def fisher_integrands(ell_arr, L, A, B, n, C_tot_arr=None, cell_interp=None):
    """
    Compute the per-(ln ell) Fisher information contributed by each piece.

    The reconstruction noise satisfies
        N_L^{-1} = int d^2ell/(2pi)^2  f^2 / (2 C^tot_ell C^tot_{|L-ell|})

    In the squeezed limit |L-ell| ~ ell, so C^tot_{|L-ell|} ~ C^tot_ell and:
        N_L^{-1} ~ int (ell dell dtheta)/(2pi)^2 * f^2 / (2 (C^tot)^2)

    Integrating over theta:
        iso  piece: <f_iso^2>_theta       = f_iso^2          (no theta dependence)
        quad piece: <f_quad_2d^2>_theta   = f_quad_amp^2 / 2 (<cos^2(2theta)>=1/2)

    Returned integrand is d(N^{-1})/d(ln ell) = ell^2 / (2pi) * f^2 / (2 C^tot^2)
    (up to the squeezed-limit approximation C^tot_{L-ell} ~ C^tot_ell).

    If cell_interp is provided it overrides A, B, n.
    """
    f_iso, f_quad_amp = response_amplitudes_1d(ell_arr, L=L, A=A, B=B, n=n,
                                               cell_interp=cell_interp)

    if C_tot_arr is None:
        C_tot_arr = cell_interp(ell_arr) if cell_interp is not None else C_ell(ell_arr, A, B, n)

    prefactor = ell_arr**2 / (2.0 * np.pi) / (2.0 * C_tot_arr**2)

    dFiso  = prefactor * f_iso**2
    dFquad = prefactor * f_quad_amp**2 / 2.0   # factor 1/2 from <cos^2(2theta)>

    return dFiso, dFquad

def cumulative_fisher(ell_arr, dFiso, dFquad):
    """Cumulative Fisher information (trapz in ln ell) and shear fraction."""
    dlnell = np.gradient(np.log(ell_arr))

    Fiso_cum  = np.cumsum(dFiso  * dlnell)
    Fquad_cum = np.cumsum(dFquad * dlnell)
    Ftot_cum  = Fiso_cum + Fquad_cum

    shear_frac = np.where(Ftot_cum > 0, Fquad_cum / Ftot_cum, 0.0)
    return Fiso_cum, Fquad_cum, Ftot_cum, shear_frac

def plot_profiles_and_fisher(L_values, A, B, n, ellmin, ellmax, Nell=500):
    ell_arr = np.geomspace(ellmin, ellmax, Nell)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        rf"$C_\ell = A + B\ell^n$,  $A={A}$,  $B={B}$,  $n={n}$"
        rf"  |  annulus $\ell\in[{int(ellmin)},{int(ellmax)}]$",
        fontsize=12
    )

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(L_values)))

    for L, col in zip(L_values, colors):
        f_iso, f_quad_amp = response_amplitudes_1d(ell_arr, L=L, A=A, B=B, n=n)
        f_quad_rms = np.abs(f_quad_amp) / np.sqrt(2.0)   # RMS over angle

        dFiso, dFquad = fisher_integrands(ell_arr, L=L, A=A, B=B, n=n)
        _, _, Ftot_cum, shear_frac = cumulative_fisher(ell_arr, dFiso, dFquad)

        # Panel 1: response amplitudes
        axes[0].plot(ell_arr, np.abs(f_iso),      color=col, ls="-",
                     label=rf"$f^{{(0)}}$, $L={L}$")
        axes[0].plot(ell_arr, f_quad_rms,          color=col, ls="--",
                     label=rf"$f^{{(2)}}/\sqrt{{2}}$, $L={L}$")

        # Panel 2: cumulative shear fraction
        axes[1].plot(ell_arr, shear_frac, color=col,
                     label=rf"$L={L}$")

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$\ell$", fontsize=12)
    axes[0].set_ylabel("Response amplitude (arb. units)", fontsize=11)
    axes[0].set_title("Response amplitudes: iso (solid) vs shear RMS (dashed)")
    axes[0].legend(fontsize=9, ncol=2)

    axes[1].axhline(0.5, color="k", lw=0.8, ls=":", alpha=0.6,
                    label="50% shear")
    axes[1].set_xscale("log")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_xlabel(r"$\ell_{\rm max}$ (upper integration limit)", fontsize=12)
    axes[1].set_ylabel(r"Cumulative shear fraction  $F^{\rm quad}/F^{\rm tot}$",
                       fontsize=11)
    axes[1].set_title("Shear fraction of total Fisher information vs integration limit")
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 2: shear fraction as a function of spectral index n
# ---------------------------------------------------------------------------

def plot_shear_fraction_vs_n(n_values, A, B, L, ellmin, ellmax, Nell=500):
    """
    For each n, compute the total-integrated shear fraction
    (integrating over the full annulus [ellmin, ellmax]).
    """
    ell_arr    = np.geomspace(ellmin, ellmax, Nell)
    fracs      = []

    for nv in n_values:
        dFiso, dFquad = fisher_integrands(ell_arr, L=L, A=A, B=B, n=nv)
        dlnell        = np.gradient(np.log(ell_arr))
        Fiso          = np.trapz(dFiso  * dlnell)
        Fquad         = np.trapz(dFquad * dlnell)
        fracs.append(Fquad / (Fiso + Fquad) if (Fiso + Fquad) > 0 else 0.0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(n_values, fracs, "k-o", ms=4)
    ax.axhline(0.5, color="grey", lw=0.8, ls="--", label="50%")
    ax.axvline(-2.0, color="steelblue", lw=0.8, ls="--",
               label=r"$n=-2$ ($f_{\rm iso}\to 0$)")
    ax.set_xlabel(r"Power-law index $n$", fontsize=12)
    ax.set_ylabel(r"Shear fraction  $F^{\rm quad}/F^{\rm tot}$", fontsize=12)
    ax.set_title(
        rf"Total shear fraction vs spectral slope  |  $A={A}$,  $L={L}$"
        rf"  |  $\ell\in[{int(ellmin)},{int(ellmax)}]$",
        fontsize=11
    )
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Full 2D Fisher integrands (no squeezed-limit approximation)
#
# At each |ell|, integrate over angle theta to get
#   dN_L^{-1}/d(ln ell) = ell^2/(2pi)^2 * int_0^{2pi} dtheta
#       * [f_iso^2 + 2 f_iso f_quad cos2theta + f_quad^2 cos^2(2theta)]
#         / (2 C_ell * C_{|L - ell(theta)|})
# Iso and quad amplitudes are functions of |ell| only; the only angular
# dependence comes from cos(2theta) in the response and C_{|L-ell|} in
# the denominator (the squeezed limit sets C_{|L-ell|} -> C_ell).
# ---------------------------------------------------------------------------

def fisher_integrands_full2d(ell_arr, L, A, B, n, Ntheta=400, cell_interp=None):
    """
    Full (non-squeezed) per-lnell Fisher integrands for iso, quad, and cross.

    Returns
    -------
    dF_iso, dF_quad, dF_cross : 1-D arrays of shape (len(ell_arr),)
        dF_iso  : magnification (isotropic) contribution
        dF_quad : shear (quadrupolar) contribution, = f_quad^2 <cos^2 2theta / C1 C2>
        dF_cross: interference term 2 f_iso f_quad <cos 2theta / C1 C2>;
                  vanishes in squeezed limit but is nonzero when L ~ ell.

    If cell_interp is provided it overrides A, B, n.
    """
    theta  = np.linspace(0.0, 2.0 * np.pi, Ntheta, endpoint=False)
    dtheta = 2.0 * np.pi / Ntheta
    cos2th = np.cos(2.0 * theta)                   # shape (Ntheta,)

    ell_safe = np.maximum(ell_arr, 1.0)            # floor to avoid /0

    # |L - ell_vec(theta)|  ->  shape (Nell, Ntheta)
    ell2_sq = (L**2 + ell_safe[:, None]**2
               - 2.0 * L * ell_safe[:, None] * np.cos(theta[None, :]))
    ell2    = np.sqrt(np.maximum(ell2_sq, 1.0))

    if cell_interp is not None:
        C1 = cell_interp(ell_safe)                 # (Nell,)
        C2 = cell_interp(ell2)                     # (Nell, Ntheta)
    else:
        C1 = C_ell(ell_safe, A, B, n)              # (Nell,)
        C2 = C_ell(ell2,     A, B, n)              # (Nell, Ntheta)

    inv_denom = 1.0 / (2.0 * C1[:, None] * C2)    # (Nell, Ntheta)

    # Angular integrals
    I_iso   = np.sum(             inv_denom, axis=1) * dtheta  # (Nell,)
    I_quad  = np.sum(cos2th**2 *  inv_denom, axis=1) * dtheta
    I_cross = np.sum(cos2th    *  inv_denom, axis=1) * dtheta

    f_iso, f_quad = response_amplitudes_1d(ell_safe, L=L, A=A, B=B, n=n,
                                           cell_interp=cell_interp)
    prefactor = ell_safe**2 / (2.0 * np.pi)**2

    dF_iso   = prefactor * f_iso**2           * I_iso
    dF_quad  = prefactor * f_quad**2          * I_quad
    dF_cross = prefactor * f_iso * f_quad * 2 * I_cross   # factor 2 from expansion

    return dF_iso, dF_quad, dF_cross


def fisher_integrands_full2d_exact(ell_arr, L, A, B, n, Ntheta=400, 
                                   Nphi=100, cell_interp=None):
    """
    Fully angle-resolved Fisher integrand — response computed at each (ell, phi)
    point on the annulus, eliminating exact-zero artifacts from 1D reduction.

    This integrates the response function f(ell_x, ell_y, L) over the full annulus
    at each |ell|, where f depends on the actual 2D position not just the magnitude.
    
    Returns
    -------
    dF_total : 1-D array of shape (len(ell_arr),)
        Total Fisher integrand per ln(ell), averaged over all angles phi (azimuth
        around the |ell| annulus) and theta (orientation of conjugate mode).
        
    Notes
    -----
    This is ~Nphi times slower than fisher_integrands_full2d but removes the
    unphysical exact zeros that occur when the 1D response amplitude f(|ell|, L)
    changes sign.
    """
    phi_arr = np.linspace(0.0, 2.0 * np.pi, Nphi, endpoint=False)
    dphi    = 2.0 * np.pi / Nphi
    
    ell_safe = np.maximum(ell_arr, 1.0)
    
    # Build lensing mode positions: L along +x axis
    Lx, Ly = L, 0.0
    
    # Output array
    dF_total = np.zeros(len(ell_safe))
    
    for i, ell_mag in enumerate(ell_safe):
        # Azimuthal angles around the |ell| = ell_mag annulus
        ellx = ell_mag * np.cos(phi_arr)   # (Nphi,)
        elly = ell_mag * np.sin(phi_arr)
        
        # Conjugate leg
        ell2x = Lx - ellx
        ell2y = Ly - elly
        ell2  = np.sqrt(ell2x**2 + ell2y**2)
        
        # Power spectra at each point
        if cell_interp is not None:
            C1 = cell_interp(ell_mag)       # scalar (same for all phi)
            C2 = cell_interp(ell2)          # (Nphi,)
            # Log derivatives: evaluate at ell_mag (approximate — same around annulus)
            s, s2 = log_derivs_numerical(ell_mag, cell_interp)
        else:
            C1 = C_ell(ell_mag, A, B, n)
            C2 = C_ell(ell2, A, B, n)
            s  = n * B * ell_mag**n / C1
            s2 = s * (n - s)
        
        # Response amplitudes at each (ellx, elly) point
        # D± depend on spectrum derivatives (approximated as constant around annulus)
        Dplus  = C1 * (s + s**2 + s2)
        Dminus = C1 * (s**2 + s2 - s)
        ratio  = L / ell_mag
        
        cos_2phi = np.cos(2.0 * phi_arr)
        
        # Angle-dependent response (KEY: varies with phi even at fixed |ell|)
        f_iso_phi   = C1 * (2.0 + s) - 0.5 * ratio * Dplus
        f_quad_phi  = (C1 * s - 0.5 * ratio * Dminus) * cos_2phi
        
        # Total response at each phi
        f_total = f_iso_phi + f_quad_phi
        
        # Fisher integrand = f^2 / (2 C1 C2), averaged over phi
        integrand_phi = f_total**2 / (2.0 * C1 * C2)
        
        # Integrate over azimuth
        dF_total[i] = (ell_mag**2 / (2.0 * np.pi)**2) * np.sum(integrand_phi) * dphi
    
    return dF_total


# ---------------------------------------------------------------------------
# Plot 3: SNR integrand  d[SNR(phi_L)^2] / d(ln ell)  vs ell for several L
#
# SNR(phi_L)^2 = C_L^{phi phi} * N_L^{-1}
# Since C_L^{phi phi} is independent of ell (the small-scale mode), the ell
# dependence of SNR^2 is entirely from N_L^{-1}, whose integrand is
#   dN_L^{-1}/d(ln ell) = dF_iso/d(ln ell) + dF_quad/d(ln ell)
# Plotting this shows which small-scale modes drive the reconstruction SNR.
# Curves are normalised to their peak so shapes across L are directly comparable.
# ---------------------------------------------------------------------------

def plot_snr_integrand(
    L_values,
    A,
    B,
    n,
    ellmin,
    ellmax,
    Nell=600,
    normalize=True,
    N_tris=1e-9,
    C_L=1.0,
):
    """
    Plot d[SNR(phi_L)^2]/d(ln ell) vs ell for each L in L_values.

    Parameters
    ----------
    normalize : bool
        If True, normalise each curve to its peak (shape comparison).
        If False, show raw amplitudes (arb. units).
    """
    ell_arr = np.geomspace(ellmin, ellmax, Nell)
    colors  = plt.cm.plasma(np.linspace(0.15, 0.80, len(L_values)))

    print('C_L is ', C_L)

    fig, ax = plt.subplots(figsize=(7, 4))

    for L, col in zip(L_values, colors):
        dFiso, dFquad = fisher_integrands(ell_arr, L=L, A=A, B=B, n=n)
        g = dFiso + dFquad          # total  dN_L^{-1}/d(ln ell)

        # Cumulative Gaussian inverse-noise up to each ell:
        #   I_cum(ell) = int_{ellmin}^{ell} g(ell') d(ln ell')
        dlnell = np.gradient(np.log(ell_arr))
        I_cum  = np.cumsum(g * dlnell)

        # SNR^2(ell_max) = C_L * I_cum / (1 + N_tris * I_cum)
        # Differentiating with respect to ln(ell_max):
        #   d[SNR^2]/d(ln ell) = C_L * g(ell) / (1 + N_tris * I_cum(ell))^2
        # Turnover when N_tris * I_cum ~ 1, i.e. I_cum ~ 1/N_tris.
        snr_integrand = C_L * g / (1.0 + N_tris * I_cum)**2

        integrand = snr_integrand
        if normalize:
            peak = np.nanmax(integrand)
            if peak > 0:
                integrand = integrand / peak

        ax.plot(ell_arr, integrand, color=col, lw=1.8, label=rf"$L={L}$")

    # Mark L values on the x-axis for reference
    for L, col in zip(L_values, colors):
        ax.axvline(L, color=col, lw=0.7, ls="--", alpha=0.45)

    ax.set_xscale("log")
    if not normalize:
        ax.set_yscale("log")
    ax.set_xlabel(r"$\ell$", fontsize=12)
    ylabel = (r"$d[\mathrm{SNR}(\phi_L)^2]/d\ln\ell$  (normalised to peak)"
              if normalize else
              r"$d[\mathrm{SNR}(\phi_L)^2]/d\ln\ell$  (arb. units)")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def snr_from_fisher(ell_arr, dF, dF_total, C_L, N_tris):
    """
    Compute the SNR integrand d[SNR^2]/d(ln ell) from the Fisher integrand dF.
    """

    dlnell = np.gradient(np.log(ell_arr))
    # dlnell = np.gradient(np.log(dF_total))
    I_cum  = np.cumsum(dF_total * dlnell)
    snr_integrand = C_L * dF / (1.0 + N_tris * I_cum)**2
    return snr_integrand

def compute_snr_integrand(ell_arr, L, C_L,  A, B, n, Ntheta, Nphi, cell_interp, exact_response, N_tris=None):
    if N_tris is None:
        N_tris = [0.]
    if exact_response:
        dF_total = fisher_integrands_full2d_exact(
            ell_arr, L=L, A=A, B=B, n=n, Ntheta=Ntheta, Nphi=Nphi,
            cell_interp=cell_interp, N_tris=N_tris,
        )
        snr_total = snr_from_fisher(ell_arr, dF_total, dF_total, C_L, N_tris)
        return snr_total
    else:
        dF_iso, dF_quad, dF_cross = fisher_integrands_full2d(
            ell_arr, L=L, A=A, B=B, n=n, Ntheta=Ntheta, cell_interp=cell_interp,
        )
        dF_total  = dF_iso + dF_quad + dF_cross
        snr_quad = [snr_from_fisher(ell_arr, dF_quad, dF_total, C_L, N_tris_indiv) for N_tris_indiv in N_tris]
        snr_iso  = [snr_from_fisher(ell_arr, dF_iso, dF_total, C_L, N_tris_indiv) for N_tris_indiv in N_tris]

        return snr_quad, snr_iso


def plot_snr_shear_vs_mag(
    L_values,
    A,
    B,
    n,
    ellmin,
    ellmax,
    Nell=300,
    Ntheta=400,
    N_tris=1e-9,
    C_L=1.0,
    show_squeezed=True,
    figsize=(6, 5),
    cell_interp=None,
    exact_response=False,
    Nphi=100,
    ylim=None,
    bbox_anchor=(0., 1.0),
    legend_fs=10,
    textypos=1e8,
    lab_fs=14,
):
    """
    Single plot of d[SNR(phi_L)^2]/d(ln ell) split into shear and magnification,
    using the full 2D integral (no squeezed-limit approximation).

    Each L gets its own color.  Solid = shear (quad), dashed = magnification (iso).
    Curves start at ell = L (no contribution from ell < L).
    If show_squeezed=True, squeezed-limit curves are overlaid in grey.
    Trispectrum suppression via N_tris uses the cumulative Fisher from ell=L onward.

    Parameters
    ----------
    cell_interp : callable, optional
        If provided, use this interpolated spectrum C(ell) instead of A+B*ell^n.
        Obtain with load_ciber_f25_spectrum() or any callable ell -> C_ell.
        A, B, n are ignored when cell_interp is set.
    exact_response : bool, optional
        If True, use fisher_integrands_full2d_exact which computes the response
        at every (ell, phi) point on the annulus. This removes the exact-zero
        artifacts but is slower (~Nphi times). Default: False.
    Nphi : int, optional
        Number of azimuthal samples around each |ell| annulus for exact_response.
        Only used if exact_response=True. Default: 100.
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.plasma(np.linspace(0.15, 0.80, len(L_values)))

    # ax.text(350, textypos, '$N_{L}^{\\rm tris} = $'+str(N_tris), fontsize=14)

    # for logy in [5, 6, 7, 8, 9, 10]:
        # ax.axhline(10**logy, lw=0.7, ls="solid", alpha=0.2, color='k')

    for i, (L, col) in enumerate(zip(L_values, colors)):
        # Only use ell >= L
        ell_arr = np.geomspace(max(ellmin, L), ellmax, Nell)

        # ax.axvline(L, color='k', lw=1.5, ls="solid", alpha=0.45)

        snr_total = compute_snr_integrand(ell_arr, L, C_L, A, B, n, Ntheta, Nphi, cell_interp, exact_response, N_tris=N_tris)
        lbl = rf"$(L={L})$"

        # --- Full 2D integrands ---
        if exact_response: 

            # Use fully angle-resolved response (no exact zeros)

            snr_total = compute_snr_integrand(ell_arr, L, C_L, A, B, n, Ntheta, Nphi, cell_interp, exact_response, N_tris=N_tris)

            ax.plot(ell_arr, snr_total, color=col, ls="-", lw=1.8, label=lbl)
        else:
            # Original: 1D response amplitudes (can have exact zeros)

            snr_quad, snr_iso = compute_snr_integrand(ell_arr, L, C_L, A, B, n, Ntheta, Nphi, cell_interp, exact_response, N_tris=N_tris)

            # Determine colors based on L value (varying shades of blue and red)
            blue_color = plt.cm.Blues(0.4 + 0.4 * (i / max(len(L_values)-1, 1)))
            red_color = plt.cm.Reds(0.4 + 0.4 * (i / max(len(L_values)-1, 1)))
            
            # Plot shear (snr_quad) cases in blue with solid lines

            linestyles = ['-', '--', '-.', ':']


            for j, snr_quad_case in enumerate(snr_quad):
                ls = linestyles[j % len(linestyles)]

                label = rf"Shear {lbl}" if j == 0 else None
                ax.plot(ell_arr, snr_quad_case, color=blue_color, ls=ls, lw=1.8, label=label)
            
            # Plot magnification (snr_iso) cases in red with different line styles for each N_tris case
            for j, snr_iso_case in enumerate(snr_iso):
                ls = linestyles[j % len(linestyles)]
                label = rf"Magnification {lbl}" if j == 0 else None
                ax.plot(ell_arr, snr_iso_case, color=red_color, ls=ls, lw=1.8, label=label)

        # --- Squeezed-limit overlay ---
        if show_squeezed and not exact_response:
            dFiso_sq, dFquad_sq = fisher_integrands(ell_arr, L=L, A=A, B=B, n=n,
                                                    cell_interp=cell_interp)
            g_sq = dFiso_sq + dFquad_sq

            dlnell = np.gradient(np.log(ell_arr))
            I_sq = np.cumsum(g_sq * dlnell)
            w_sq = C_L / (1.0 + N_tris * I_sq)**2
            ax.plot(ell_arr, w_sq * dFquad_sq, color="0.65", ls="-",  lw=0.9,
                    label="shear (squeezed)" if i == 0 else None)
            ax.plot(ell_arr, w_sq * dFiso_sq,  color="0.65", ls="--", lw=0.9,
                    label="mag (squeezed)"   if i == 0 else None)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\ell$", fontsize=lab_fs)
    ax.set_ylabel(r"$d[\mathrm{SNR}(\phi_L)^2]/d\ln\ell$", fontsize=lab_fs)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=legend_fs, ncol=2, loc=2, bbox_to_anchor=bbox_anchor)
    if ylim is not None:
        ax.set_ylim(ylim)
    # plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Comparison plot: 1D response (with exact zeros) vs exact 2D response
# ---------------------------------------------------------------------------

def plot_response_comparison_1d_vs_exact2d(
    L=300,
    ellmin=100,
    ellmax=50000,
    Nell=300,
    Ntheta=400,
    Nphi=100,
    N_tris=1e-9,
    C_L=1.0,
    cell_interp=None,
    A=None,
    B=None,
    n=None,
    figsize=(12, 5),
    ylim=None
):
    """
    Side-by-side comparison of SNR integrand computed with:
    - Left: 1D response (fisher_integrands_full2d) - shows exact-zero dips
    - Right: Exact 2D response (fisher_integrands_full2d_exact) - dips filled in
    
    This demonstrates that the unphysical exact zeros from the 1D approximation
    are replaced by shallow (but real) minima when using the full angle-dependent
    response at each point on the |ell| annulus.
    
    Parameters
    ----------
    L : float
        Lens multipole
    N_tris : float
        Trispectrum noise level for cumulative suppression (default: 1e-9)
    C_L : float
        Lensing potential power spectrum normalization (default: 1.0)
    cell_interp : callable, optional
        Interpolated C(ell) function. If None, uses A + B*ell^n
    Nphi : int
        Number of azimuthal samples for exact 2D (default: 100)
        Higher = more accurate but slower
    """
    ell_arr = np.geomspace(max(ellmin, L), ellmax, Nell)
    dlnell = np.gradient(np.log(ell_arr))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # --- Left panel: 1D response (can have exact zeros) ---
    dF_iso_1d, dF_quad_1d, dF_cross_1d = fisher_integrands_full2d(
        ell_arr, L=L, A=A, B=B, n=n, Ntheta=Ntheta, cell_interp=cell_interp
    )
    dF_total_1d = dF_iso_1d + dF_quad_1d + dF_cross_1d
    
    # Apply trispectrum suppression
    I_cum_1d = np.cumsum(dF_total_1d * dlnell)
    w_1d = C_L / (1.0 + N_tris * I_cum_1d)**2
    
    snr_quad_1d = w_1d * dF_quad_1d
    snr_iso_1d = w_1d * dF_iso_1d
    snr_total_1d = w_1d * dF_total_1d
    
    ax1.plot(ell_arr, snr_quad_1d, color='C0', ls='-', lw=2, label='Shear')
    ax1.plot(ell_arr, snr_iso_1d, color='C1', ls='--', lw=2, label='Magnification')
    ax1.plot(ell_arr, snr_total_1d, color='k', ls=':', lw=1.5, label='Total')
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$\ell$', fontsize=14)
    ax1.set_ylabel(r'$d[\mathrm{SNR}(\phi_L)^2]/d\ln\ell$', fontsize=14)
    ax1.set_title(f'1D Response (exact zeros)\n$L={L}$, $N_{{\\rm tris}}={N_tris:.1e}$', fontsize=13)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=11)
    
    # Highlight exact zeros with vertical lines
    # Find where integrand drops below numerical floor
    zero_threshold = 1e-10 * np.nanmax(snr_total_1d)
    zero_mask = (snr_total_1d < zero_threshold) & (ell_arr > 2*L)
    if np.any(zero_mask):
        zero_ells = ell_arr[zero_mask]
        for ze in zero_ells[:3]:  # Mark first 3 zeros
            ax1.axvline(ze, color='red', alpha=0.3, ls='--', lw=1)
    
    # --- Right panel: Exact 2D response (zeros filled in) ---
    print(f"Computing exact 2D response with Nphi={Nphi} (this may take ~{Nphi//10} sec)...")
    dF_total_2d = fisher_integrands_full2d_exact(
        ell_arr, L=L, A=A, B=B, n=n, Ntheta=Ntheta, Nphi=Nphi,
        cell_interp=cell_interp
    )
    
    # Apply trispectrum suppression
    I_cum_2d = np.cumsum(dF_total_2d * dlnell)
    w_2d = C_L / (1.0 + N_tris * I_cum_2d)**2
    
    snr_total_2d = w_2d * dF_total_2d
    
    ax2.plot(ell_arr, snr_total_2d, color='C2', ls='-', lw=2, label='Total (azimuth-avg)')
    # Overlay 1D total for comparison
    ax2.plot(ell_arr, snr_total_1d, color='0.6', ls=':', lw=1.5, alpha=0.7,
             label='1D total (reference)')
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel(r'$\ell$', fontsize=14)
    ax2.set_ylabel(r'$d[\mathrm{SNR}(\phi_L)^2]/d\ln\ell$', fontsize=14)
    ax2.set_title(f'Exact 2D Response (dips filled)\n$L={L}$, $N_\\phi={Nphi}$', fontsize=13)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=11)
    if ylim is not None:
        ax1.set_ylim(ylim)
        ax2.set_ylim(ylim)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 4: Fisher integrand per ln-ell shell (where does info come from?)
# ---------------------------------------------------------------------------

def plot_fisher_integrand_per_shell(L_values, A, B, n, ellmin, ellmax, Nell=500):
    ell_arr = np.geomspace(ellmin, ellmax, Nell)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors  = plt.cm.viridis(np.linspace(0.15, 0.85, len(L_values)))

    for L, col in zip(L_values, colors):
        dFiso, dFquad = fisher_integrands(ell_arr, L=L, A=A, B=B, n=n)
        ax.plot(ell_arr, dFiso,  color=col, ls="-",
                label=rf"iso $L={L}$")
        ax.plot(ell_arr, dFquad, color=col, ls="--",
                label=rf"shear $L={L}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\ell$", fontsize=12)
    ax.set_ylabel(r"$dN_L^{-1}/d\ln\ell$  (arb. units)", fontsize=11)
    ax.set_title(
        r"Fisher information per $\ln\ell$ shell: iso (solid) vs shear (dashed)"
    )
    ax.legend(fontsize=9, ncol=2)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 4: shear vs. magnification information as a function of ell_max
#
# Spectrum: C_ell = A + B * ell^{-2},  pivot at ell_piv  (A = B / ell_piv^2)
# As ell_max increases past ell_piv the power-law regime (n=-2, s->-2) takes
# over, where f_iso ~ (2+s) -> 0 and magnification information is suppressed.
# ---------------------------------------------------------------------------

def plot_shear_vs_mag_vs_ellmax(
    L_values,
    ellmin,
    ellmax_arr,
    ell_pivot=5000.0,
    B=1.0,
    Nell=600,
):
    """
    Show how shear (quadrupolar) and magnification (isotropic) Fisher information
    split as a function of the upper integration limit ell_max.

    Spectrum: C_ell = A + B * ell^{-2}  with  A = B / ell_pivot^2
    (the two terms contribute equally at ell = ell_pivot).

    Parameters
    ----------
    L_values   : list of floats  – large-scale multipoles to show.
    ellmin     : float           – lower integration limit (held fixed).
    ellmax_arr : 1-D array       – upper limits to sweep over.
    ell_pivot  : float           – pivot scale where A = B * ell_pivot^{-2}.
    B          : float           – overall amplitude (ratio is independent of B).
    Nell       : int             – grid points per ell_arr evaluation.
    """
    n = -2.0
    A = B / ell_pivot**2

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # --- left panel: mock power spectrum D_ell ---
    ell_plot = np.geomspace(max(ellmin, 10), ellmax_arr[-1] * 1.2, 500)
    D_ell    = ell_plot * (ell_plot + 1) * C_ell(ell_plot, A, B, n) / (2.0 * np.pi)
    axes[0].plot(ell_plot, D_ell, "k-", lw=1.8)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$\ell$", fontsize=12)
    axes[0].set_ylabel(r"$D_\ell \equiv \ell(\ell+1)C_\ell/2\pi$", fontsize=11)
    axes[0].set_ylim(1e-2, 1e3)
    # --- right panel: shear / mag fractions vs ell_max ---
    colors = plt.cm.plasma(np.linspace(0.15, 0.80, len(L_values)))

    for i, (L, col) in enumerate(zip(L_values, colors)):
        # Only compute for ell_max >= L (need modes up to L to reconstruct L)
        valid_ellmax = ellmax_arr[ellmax_arr >= L]
        
        shear_fracs = []
        mag_fracs   = []

        for ellmax in valid_ellmax:
            if ellmax <= ellmin:
                shear_fracs.append(np.nan)
                mag_fracs.append(np.nan)
                continue

            ell_arr        = np.geomspace(ellmin, ellmax, Nell)
            dFmag, dFshear = fisher_integrands(ell_arr, L=L, A=A, B=B, n=n)
            dlnell         = np.gradient(np.log(ell_arr))
            Fmag           = np.trapz(dFmag   * dlnell)
            Fshear         = np.trapz(dFshear * dlnell)
            Ftot           = Fmag + Fshear

            shear_fracs.append(Fshear / Ftot if Ftot > 0 else np.nan)
            mag_fracs.append(  Fmag   / Ftot if Ftot > 0 else np.nan)

        # Condensed labels: first L gets "shear (L=...)" and "mag (L=...)", rest just "L=..."
        if i == 0:
            shear_lbl = rf"shear ($L={L}$)"
            mag_lbl   = rf"mag ($L={L}$)"
        else:
            shear_lbl = rf"$L={L}$"
            mag_lbl   = None  # Skip label for mag on subsequent L values
        
        axes[1].plot(valid_ellmax, shear_fracs, color=col, ls="-",  lw=1.8, label=shear_lbl)
        axes[1].plot(valid_ellmax, mag_fracs,   color=col, ls="--", lw=1.8, label=mag_lbl)

    axes[1].set_xscale("log")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_xlabel(r"$\ell_{\rm max}$", fontsize=12)
    axes[1].set_ylabel(r"Fraction of total Fisher information", fontsize=12)
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=10, ncol=2, bbox_to_anchor=[1.0, 1.3])

#     plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Three-panel plot: Shear SNR | Magnification SNR | Fractional Fisher vs ell_max
# ---------------------------------------------------------------------------

def plot_snr_shear_mag_fisher_threepanel(
    L_values,
    ellmin,
    ellmax,
    ellmax_arr=None,
    A=None,
    B=None,
    n=None,
    Nell=300,
    Ntheta=400,
    N_tris=1e-9,
    C_L=1.0,
    show_squeezed=True,
    figsize=(16, 5),
    cell_interp=None,
    exact_response=False,
    Nphi=100,
    ylim_snr=None,
    legend_fs=13,
    lab_fs=16,
    plot_xmin=2000,
    shear_lmax=4e4,
    mag_lmin = 2000,
    textxpos=1.5e4, 
    textypos=1e9,
    text_fs=18,
    figsize_single=(7, 4.5),
    bbox_to_anchor=[-0.05, 1.2]
):
    """
    Three-panel plot showing shear/magnification SNR integrands and magnification fraction vs ell_max.

    Parameters
    ----------
    L_values : list of floats
        Large-scale multipoles to plot.
    A, B, n : float
        Power spectrum parameters: C_ell = A + B * ell^n (used for Fisher panel if cell_interp not provided).
    ellmin, ellmax : float
        Multipole range for SNR panels.
    ellmax_arr : 1-D array, optional
        Upper integration limits for magnification fraction panel. Default: np.geomspace(ellmin, ellmax, 50).
    Nell : int
        Number of grid points per evaluation.
    Ntheta : int
        Number of angular samples for 2D Fisher computation.
    N_tris : float or list
        Trispectrum noise level(s) for SNR suppression.
    C_L : float
        Lensing potential power spectrum normalization.
    cell_interp : callable, optional
        Interpolated power spectrum C(ell). If provided, overrides A, B, n for Fisher panel.
    exact_response : bool
        Use exact 2D response (slower, no exact zeros).
    Nphi : int
        Azimuthal samples for exact response.
    ylim_snr : tuple, optional
        y-limits for SNR panels.
    legend_fs, lab_fs : int
        Font sizes.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Two-panel SNR figure
    fig_fraction : matplotlib.figure.Figure
        Magnification fraction vs ell_max figure
    fig_cumul : matplotlib.figure.Figure
        Cumulative Fisher information figure
    """
    if ellmax_arr is None:
        ellmax_arr = np.geomspace(ellmin, ellmax, 50)

    # Determine N_tris values to iterate over
    if isinstance(N_tris, list):
        N_tris_list = N_tris
    else:
        N_tris_list = [N_tris]

    # Pre-compute SNR integrands for all L values and N_tris variations
    # Dictionary: snr_data[(L, n_idx)] = {'ell_arr': ..., 'snr_quad': [...], 'snr_iso': [...]}
    snr_data = {}
    colors = plt.cm.plasma(np.linspace(0.15, 0.80, len(L_values)))
    linestyles = ['-', '--', '-.', ':']

    ell_arr_full = np.geomspace(max(ellmin, min(L_values)), ellmax, Nell)

    snr_data_notris = {}

    for i, L in enumerate(L_values):
        ell_arr = np.geomspace(max(ellmin, L), ellmax, Nell)
        snr_quad, snr_iso = compute_snr_integrand(
            ell_arr, L, C_L, A, B, n, Ntheta, Nphi, cell_interp, exact_response, N_tris=N_tris_list
        )
        snr_data[L] = {'ell_arr': ell_arr, 'snr_quad': snr_quad, 'snr_iso': snr_iso}

        snr_quad_notris, snr_iso_notris = compute_snr_integrand(
            ell_arr, L, C_L, A, B, n, Ntheta, Nphi, cell_interp, exact_response, N_tris=[0.0]
        )
        snr_data_notris[L] = {'ell_arr': ell_arr, 'snr_quad': snr_quad_notris, 'snr_iso': snr_iso_notris}

    # --- Panel 1: Shear SNR integrand ---

    linewidth=2.5
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    ax_shear = axes[0]

    # for shear, discard high ell values which have numerical issues. 



    for i, (L, col) in enumerate(zip(L_values, colors)):
        blue_color = plt.cm.Blues(0.4 + 0.4 * (i / max(len(L_values) - 1, 1)))
        ell_arr = snr_data[L]['ell_arr']
        snr_quad = snr_data[L]['snr_quad']

        for j, snr_quad_case in enumerate(snr_quad):
            ls = linestyles[j % len(linestyles)]
            label = rf"$L={L}$" if j == 0 else None

            ax_shear.plot(ell_arr[ell_arr <= shear_lmax], snr_quad_case[ell_arr <= shear_lmax], color=blue_color, ls=ls, lw=linewidth, label=label)

    if plot_xmin is not None:
        ax_shear.set_xlim(plot_xmin, ellmax)

    ax_shear.set_xscale("log")
    ax_shear.set_yscale("log")
    ax_shear.set_xlabel(r"$\ell$", fontsize=lab_fs)
    ax_shear.set_ylabel(r"$d[\mathrm{SNR}(\phi_L)^2]/d\ln\ell$", fontsize=lab_fs)
    ax_shear.set_title("Shear", fontsize=lab_fs)
    ax_shear.grid(alpha=0.3)
    ax_shear.legend(fontsize=legend_fs, ncol=1, loc=2)

    if N_tris_list[0]==0.:
        ax_shear.text(textxpos, textypos, r"$N_{L}^{\rm \kappa, NG} = 0$", fontsize=lab_fs)
    else:
        ax_shear.text(textxpos, textypos, r"$N_{L}^{\rm \kappa, NG} = 10^{"+str(int(np.log10(N_tris)))+"}$", fontsize=lab_fs)
    if ylim_snr is not None:
        ax_shear.set_ylim(ylim_snr)

    # --- Panel 2: Magnification SNR integrand ---
    ax_mag = axes[1]

    for i, (L, col) in enumerate(zip(L_values, colors)):
        red_color = plt.cm.Reds(0.4 + 0.4 * (i / max(len(L_values) - 1, 1)))
        ell_arr = snr_data[L]['ell_arr']
        snr_iso = snr_data[L]['snr_iso']

        for j, snr_iso_case in enumerate(snr_iso):
            ls = linestyles[j % len(linestyles)]
            label = rf"$L={L}$" if j == 0 else None
            ax_mag.plot(ell_arr[ell_arr > mag_lmin], snr_iso_case[ell_arr > mag_lmin], color=red_color, ls=ls, lw=linewidth, label=label)

            if i==len(L_values)-1 and j==0:
                print('L, j', L, j)
            #     # Overlay the N_tris=0 case for comparison
                snr_iso_notris_case = snr_data_notris[300]['snr_iso'][0]
                ell_arr = snr_data_notris[300]['ell_arr']
                label = '$(N_L^{\\rm \\kappa, NG} = 0)$'
                ax_mag.plot(ell_arr[ell_arr > mag_lmin], snr_iso_notris_case[ell_arr > mag_lmin], color='grey', ls='dashed', lw=linewidth, label=label, zorder=-5)


    if plot_xmin is not None:
        ax_mag.set_xlim(plot_xmin, ellmax)
    ax_mag.set_xscale("log")
    ax_mag.set_yscale("log")
    ax_mag.set_xlabel(r"$\ell$", fontsize=lab_fs)
    # ax_mag.set_ylabel(r"$d[\mathrm{SNR}(\phi_L)^2]/d\ln\ell$", fontsize=lab_fs)
    ax_mag.set_title("Magnification", fontsize=lab_fs)
    ax_mag.grid(alpha=0.3)
    ax_mag.legend(fontsize=legend_fs, ncol=1, loc=2)
    if ylim_snr is not None:
        ax_mag.set_ylim(ylim_snr)

    if N_tris_list[0]==0.:
        ax_mag.text(textxpos, textypos, r"$N_{L}^{\rm \kappa, NG} = 0$", fontsize=lab_fs)
    else:
        ax_mag.text(textxpos, textypos, r"$N_{L}^{\rm \kappa, NG} = 10^{"+str(int(np.log10(N_tris_list[0])))+"}$", fontsize=lab_fs)

    plt.subplots_adjust(wspace=0.1)




    for ax in [ax_shear, ax_mag]:
        ax.tick_params(labelsize=12)


    # single panel figure version

    fig_single, ax = plt.subplots(figsize=figsize_single, sharey=True)

    # for shear, discard high ell values which have numerical issues. 

    for i, (L, col) in enumerate(zip(L_values, colors)):
        blue_color = plt.cm.Blues(0.4 + 0.4 * (i / max(len(L_values) - 1, 1)))
        ell_arr = snr_data[L]['ell_arr']
        snr_quad = snr_data[L]['snr_quad']

        for j, snr_quad_case in enumerate(snr_quad):
            ls = linestyles[j % len(linestyles)]
            label = rf"$L={L}$" if j == 0 else None

            ax.plot(ell_arr[ell_arr <= shear_lmax], snr_quad_case[ell_arr <= shear_lmax], color=blue_color, ls=ls, lw=linewidth, label=label)

    if plot_xmin is not None:
        ax.set_xlim(plot_xmin, ellmax)

    ax.legend(fontsize=legend_fs, ncol=1, loc=2)
    ax.tick_params(labelsize=12)

    if ylim_snr is not None:
        ax.set_ylim(ylim_snr)

    # --- Panel 2: Magnification SNR integrand ---

    for i, (L, col) in enumerate(zip(L_values, colors)):
        red_color = plt.cm.Reds(0.4 + 0.4 * (i / max(len(L_values) - 1, 1)))
        ell_arr = snr_data[L]['ell_arr']
        snr_iso = snr_data[L]['snr_iso']

        for j, snr_iso_case in enumerate(snr_iso):
            ls = linestyles[j % len(linestyles)]
            label = rf"$L={L}$" if j == 0 else None
            ax.plot(ell_arr[ell_arr > mag_lmin], snr_iso_case[ell_arr > mag_lmin], color=red_color, ls=ls, lw=linewidth, label=label)

            if i==len(L_values)-1 and j==0:
                print('L, j', L, j)
            #     # Overlay the N_tris=0 case for comparison
                snr_iso_notris_case = snr_data_notris[300]['snr_iso'][0]
                ell_arr = snr_data_notris[300]['ell_arr']
                # label = '$(N_L^{\\rm \\kappa, NG} = 0)$'
                label = None
                ax.plot(ell_arr[ell_arr > mag_lmin], snr_iso_notris_case[ell_arr > mag_lmin], color='grey', ls='dashed', lw=linewidth, label=label, zorder=-5)


    if plot_xmin is not None:
        ax.set_xlim(plot_xmin, ellmax)

    x = N_tris_list[0]
    sci = f"{x:.0e}"              # e.g. '5e-09'
    mant, exp = sci.split('e')    # mant='5', exp='-09'
    exp = int(exp)  
    label = rf"$N_{{L}}^{{\rm \kappa, NG}} = {mant}\times10^{{{exp}}}$"


    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\ell$", fontsize=lab_fs)
    ax.set_ylabel(r"$d[\mathrm{SNR}(\phi_L)^2]/d\ln\ell$", fontsize=lab_fs)
    ax.text(1.1*plot_xmin, 5e8, "Shear", fontsize=text_fs, color=plt.cm.Blues(0.8))
    ax.text(1.1*plot_xmin, 3e9, "Magnification ("+label+")", fontsize=text_fs, color=plt.cm.Reds(0.8))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=legend_fs, ncol=4, loc=2, bbox_to_anchor=bbox_to_anchor)
    if ylim_snr is not None:
        ax.set_ylim(ylim_snr)

    ax.text(textxpos, textypos, r"$N_{L}^{\rm \kappa, NG} = 0$", fontsize=lab_fs, rotation=25, color='grey')

    # if N_tris_list[0]==0.:
    #     ax.text(textxpos, textypos, r"$N_{L}^{\rm \kappa, NG} = 0$", fontsize=lab_fs, rotation=45)
    # else:
    #     ax.text(textxpos, textypos, r"$N_{L}^{\rm \kappa, NG} = 10^{"+str(int(np.log10(N_tris_list[0])))+"}$", fontsize=lab_fs)

    plt.subplots_adjust(wspace=0.1)




    # --- Panel 3: Magnification fraction vs ell_max (for all N_tris variations) ---
    fig_fraction, ax_fisher = plt.subplots(figsize=(6, 4))

    for i, L in enumerate(L_values):
        valid_ellmax = ellmax_arr[ellmax_arr >= L]
        red_color = plt.cm.Reds(0.3 + 0.4 * (i / max(len(L_values) - 1, 1)))
        blue_color = plt.cm.Blues(0.3 + 0.4 * (i / max(len(L_values) - 1, 1)))
        ell_arr_base = snr_data[L]['ell_arr']
        snr_quad_base = snr_data[L]['snr_quad']
        snr_iso_base = snr_data[L]['snr_iso']

        for n_idx, N_tris_val in enumerate(N_tris_list):
            mag_fracs = []

            for ellmax in valid_ellmax:
                if ellmax <= ellmin:
                    mag_fracs.append(np.nan)
                    continue

                # Interpolate precomputed SNR to this ellmax
                mask = ell_arr_base <= ellmax
                ell_subset = ell_arr_base[mask]
                snr_quad_subset = snr_quad_base[n_idx][mask]
                snr_iso_subset = snr_iso_base[n_idx][mask]

                # Integrate SNR components over log-space
                Fshear = np.trapz(snr_quad_subset, np.log(ell_subset))
                Fmag = np.trapz(snr_iso_subset, np.log(ell_subset))
                Ftot = Fshear + Fmag

                mag_fracs.append(Fmag / Ftot if Ftot > 0 else np.nan)

            ls = linestyles[n_idx % len(linestyles)]
            label = rf"$L={L}$" if n_idx == 0 else None
            ax_fisher.plot(valid_ellmax, mag_fracs, color=red_color, ls=ls, lw=linewidth, label=label)
            ax_fisher.plot(valid_ellmax, np.ones_like(mag_fracs)-mag_fracs, color=blue_color, ls=ls, lw=linewidth, label=label)

    ax_fisher.set_xscale("log")
    if plot_xmin is not None:
        ax_fisher.set_xlim(plot_xmin, 1e5)
    ax_fisher.set_ylim(-0.02, 1.02)
    ax_fisher.set_xlabel(r"$\ell_{\rm max}$", fontsize=lab_fs)
    ax_fisher.set_ylabel(r"$F(\phi_L)_{\rm mag}/F(\phi_L)_{\rm tot}$", fontsize=lab_fs)
    ax_fisher.grid(alpha=0.3)
    ax_fisher.legend(fontsize=legend_fs, ncol=4, loc=2, bbox_to_anchor=(-0.1, 1.2))
    ax_fisher.tick_params(labelsize=12)

    # --- Panel 4: Cumulative Fisher information vs ell_max ---
    fig_cumul, ax_cumul = plt.subplots(figsize=(6, 4))

    for i, L in enumerate(L_values):
        valid_ellmax = ellmax_arr[ellmax_arr >= L]
        blue_color = plt.cm.Blues(0.4 + 0.4 * (i / max(len(L_values) - 1, 1)))
        red_color = plt.cm.Reds(0.4 + 0.4 * (i / max(len(L_values) - 1, 1)))
        ell_arr_base = snr_data[L]['ell_arr']
        snr_quad_base = snr_data[L]['snr_quad']
        snr_iso_base = snr_data[L]['snr_iso']

        # Use first N_tris variation
        n_idx = 0
        shear_cumul = []
        mag_cumul = []

        for ellmax in valid_ellmax:
            if ellmax <= ellmin:
                shear_cumul.append(np.nan)
                mag_cumul.append(np.nan)
                continue

            mask = ell_arr_base <= ellmax
            ell_subset = ell_arr_base[mask]
            snr_quad_subset = snr_quad_base[n_idx][mask]
            snr_iso_subset = snr_iso_base[n_idx][mask]

            Fshear = np.trapz(snr_quad_subset, np.log(ell_subset))
            Fmag = np.trapz(snr_iso_subset, np.log(ell_subset))

            shear_cumul.append(Fshear)
            mag_cumul.append(Fmag)

        ax_cumul.plot(valid_ellmax, shear_cumul, color=blue_color, ls='solid', lw=linewidth,
                      label=rf"Shear $L={L}$")
        ax_cumul.plot(valid_ellmax, mag_cumul, color=red_color, ls='solid', lw=linewidth, 
                      label=rf"Magnification $L={L}$")

    ax_cumul.set_xscale("log")
    ax_cumul.set_yscale("log")
    if plot_xmin is not None:
        ax_cumul.set_xlim(plot_xmin, 1e5)
    ax_cumul.set_xlabel(r"$\ell_{\rm max}$", fontsize=lab_fs)
    ax_cumul.set_ylabel(r"Cumulative Fisher information", fontsize=lab_fs)
    # ax_cumul.set_title("Cumulative SNR Integrand", fontsize=lab_fs)
    ax_cumul.grid(alpha=0.3)
    ax_cumul.legend(fontsize=legend_fs, ncol=2, loc=2, bbox_to_anchor=(-0.05, 1.45))
    ax_cumul.tick_params(labelsize=12)
    
    return fig, fig_fraction, fig_cumul, fig_single


