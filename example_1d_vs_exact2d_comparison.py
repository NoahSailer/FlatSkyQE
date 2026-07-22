#!/usr/bin/env python
"""
Example: Comparing 1D response (with exact-zero artifacts) 
         vs exact 2D response (dips filled in)

This demonstrates how the unphysical exact zeros from the 1D approximation
are replaced by shallow minima when computing the full angle-dependent
response at each (ell_x, ell_y) point on the annulus.
"""

import numpy as np
import matplotlib.pyplot as plt
from lens_response import (
    load_ciber_f25_spectrum,
    plot_response_comparison_1d_vs_exact2d,
    plot_snr_shear_vs_mag,
)

# ---------------------------------------------------------------------------
# Example 1: Direct comparison of Fisher integrands (1D vs exact 2D)
# ---------------------------------------------------------------------------

print("=" * 70)
print("Example 1: Side-by-side comparison of 1D vs exact 2D response")
print("=" * 70)

# Load CIBER F25B power spectrum (J-band, inst=1)
# Use spline smoothing to avoid jagged interpolation
cell_interp, lb, cl = load_ciber_f25_spectrum(
    inst=1, 
    smooth=True, 
    smooth_model='spline'
)

# Compare at L=300 (typical scales for lensing reconstruction)
fig1 = plot_response_comparison_1d_vs_exact2d(
    L=300,
    ellmin=100,
    ellmax=50000,
    Nell=300,
    Ntheta=400,
    Nphi=100,  # 100 azimuthal samples (takes ~10 sec)
    N_tris=1e-9,  # Trispectrum suppression
    C_L=1.0,
    cell_interp=cell_interp,
)
fig1.savefig('comparison_1d_vs_exact2d_L300.png', dpi=150, bbox_inches='tight')
print("\nSaved: comparison_1d_vs_exact2d_L300.png")
print("  Left panel: 1D response shows exact-zero dips (red dashed lines)")
print("  Right panel: Exact 2D response has shallow minima instead\n")

plt.show()

# ---------------------------------------------------------------------------
# Example 2: SNR curves with 1D response (default behavior)
# ---------------------------------------------------------------------------

print("=" * 70)
print("Example 2: SNR curves with 1D response (default)")
print("=" * 70)

fig2 = plot_snr_shear_vs_mag(
    L_values=[100, 200, 300],
    A=None, B=None, n=None,  # Parameters ignored when cell_interp is set
    ellmin=100,
    ellmax=50000,
    Nell=300,
    Ntheta=400,
    cell_interp=cell_interp,
    exact_response=False,  # Use 1D response (faster, but has exact zeros)
)
fig2.savefig('snr_shear_vs_mag_1d_response.png', dpi=150, bbox_inches='tight')
print("\nSaved: snr_shear_vs_mag_1d_response.png")
print("  May show sharp dips to zero at certain ell (unphysical)\n")

plt.show()

# ---------------------------------------------------------------------------
# Example 3: SNR curves with exact 2D response (artifact-free)
# ---------------------------------------------------------------------------

print("=" * 70)
print("Example 3: SNR curves with exact 2D response (artifact-free)")
print("=" * 70)
print("WARNING: This is ~100x slower than Example 2!")
print("Recommended: use Nell=100, Nphi=50 for testing\n")

# Reduced resolution for speed demonstration
fig3 = plot_snr_shear_vs_mag(
    L_values=[300],  # Single L for speed
    A=None, B=None, n=None,
    ellmin=100,
    ellmax=50000,
    Nell=100,  # Reduced from 300
    Ntheta=200,  # Reduced from 400
    cell_interp=cell_interp,
    exact_response=True,  # Use exact 2D (slower, but no artifacts)
    Nphi=50,  # Reduced from 100 for speed
    show_squeezed=False,  # Squeezed overlay not compatible with exact_response
)
fig3.savefig('snr_shear_vs_mag_exact2d_response.png', dpi=150, bbox_inches='tight')
print("\nSaved: snr_shear_vs_mag_exact2d_response.png")
print("  Dips are now shallow minima (physical)\n")

plt.show()

# ---------------------------------------------------------------------------
# Example 4: Quick analytical test (no CIBER data)
# ---------------------------------------------------------------------------

print("=" * 70)
print("Example 4: Analytical power-law comparison (A + B*ell^n)")
print("=" * 70)

fig4 = plot_response_comparison_1d_vs_exact2d(
    L=200,
    ellmin=50,
    ellmax=30000,
    Nell=200,
    Nphi=80,
    N_tris=1e-9,
    cell_interp=None,  # Use analytical spectrum
    A=4e-6,  # Poisson shot noise floor
    B=6e-4,  # Clustering amplitude
    n=-1.5,  # Power-law index
)
fig4.savefig('comparison_analytical_powerlaw.png', dpi=150, bbox_inches='tight')
print("\nSaved: comparison_analytical_powerlaw.png")
print("  C(ell) = 4e-6 + 6e-4 * ell^{-1.5}\n")

plt.show()

print("=" * 70)
print("All examples complete!")
print("=" * 70)
print("\nKey takeaways:")
print("  1. Exact zeros in 1D are artifacts of |ell|-only response")
print("  2. Exact 2D computes response at each (ell_x, ell_y) point")
print("  3. Dips fill in to shallow minima (physically meaningful)")
print("  4. Use exact_response=True for publication plots")
print("  5. Use exact_response=False for fast exploration")
print("  6. Typical speed: 1D ~0.1 sec, exact 2D ~10 sec (Nphi=100)")
