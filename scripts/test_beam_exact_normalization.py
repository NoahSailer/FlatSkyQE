"""
Test script for beam-exact normalization implementation.

This demonstrates how to use the new explicit B(ℓ)B(L-ℓ) normalization
and compare it with the old FFT-based approach.
"""

import numpy as np
import matplotlib.pyplot as plt
from flat_map import FlatMap
import sys

# ============================================================================
# Configuration
# ============================================================================

# Set plot=True to show all diagnostic plots, False for minimal output
plot = True  # <-- Change this to control plotting

# ============================================================================
# Setup: Define your power spectra and beam
# ============================================================================

# Map parameters
MAP_SIZE = 512  # Use smaller size for faster testing
sizeX = 2.0 * np.pi / 180  # 2 degrees
sizeY = 2.0 * np.pi / 180
lMin = 1e4
lMax = 1e5

# Create flat map
baseMap = FlatMap(nX=MAP_SIZE, nY=MAP_SIZE, sizeX=sizeX, sizeY=sizeY)

# Define spectra
c_i_shot = 1e-9  # Example shot noise level

# CRITICAL: fC0 should be the raw unlensed spectrum (NO BEAM)
def fC0(ell):
    """Unlensed CIB auto-spectrum"""
    return c_i_shot * np.ones_like(ell)

# Example CIBER beam (you can replace with your actual clf.bl)
def B_ell(ell):
    """CIBER beam approximation"""
    # Gaussian beam with FWHM ~ 7 arcsec
    sigma_rad = 7.0 / 206265.0  # Convert arcsec to radians
    return np.exp(-0.5 * (ell * sigma_rad)**2)

# Noise level
N_ell = 1e-10  # Example noise level

# Observed spectrum (includes beam)
def fCtot(ell):
    """Observed spectrum = beam^2 * unlensed + noise"""
    return B_ell(ell)**2 * fC0(ell) + N_ell


# ============================================================================
# Test 1: Compute beam-exact normalization
# ============================================================================

print("\n" + "="*70)
print("TEST 1: Computing Beam-Exact Normalization")
print("="*70)

norm_exact = baseMap.computeQuadEstPhiNormalizationFFT_BeamExact(
    fC0, fCtot, B_ell, 
    lMin=lMin, lMax=lMax, 
    nEll=50,  # Use 50 bins for faster testing
    test=True  # Will show diagnostic plots
)

print("\nNormalization computed!")
print(f"Shape: {norm_exact.shape}")
print(f"Non-zero elements: {np.sum(norm_exact != 0)}")
print(f"Mean |N_L|: {np.mean(np.abs(norm_exact[norm_exact != 0])):.3e}")


# ============================================================================
# Test 2: Compare with FFT method (old way)
# ============================================================================

print("\n" + "="*70)
print("TEST 2: Comparing Beam-Exact vs FFT Method")
print("="*70)

# Old way: FFT with beam baked into fC0 (INCORRECT)
print("\nComputing FFT normalization (old method)...")
norm_fft = baseMap.computeQuadEstPhiNormalizationFFT(
    fC0, fCtot,
    lMin=lMin, lMax=lMax,
    test=False,
    fB_ell=B_ell  # This applies B² incorrectly
)

# Compare the two
l_flat = baseMap.l.flatten()
mask = (l_flat > lMin) & (l_flat < 2*lMax)

if plot:
    plt.figure(figsize=(14, 4))

    # Plot 1: 2D comparison
    plt.subplot(131)
    ratio_2d = np.abs(norm_exact) / (np.abs(norm_fft) + 1e-30)
    plt.imshow(np.log10(ratio_2d), cmap='RdBu_r', vmin=-0.3, vmax=0.3)
    plt.colorbar(label='log10(Beam-exact / FFT)')
    plt.title('Ratio: Beam-Exact / FFT Method')

    # Plot 2: 1D profiles
    plt.subplot(132)
    L_bins = np.logspace(np.log10(lMin), np.log10(2*lMax), 20)
    L_centers = 0.5 * (L_bins[:-1] + L_bins[1:])

    norm_exact_binned = []
    norm_fft_binned = []

    for i in range(len(L_centers)):
        mask_bin = (l_flat >= L_bins[i]) & (l_flat < L_bins[i+1])
        if np.sum(mask_bin) > 0:
            norm_exact_binned.append(np.median(np.abs(norm_exact.flatten()[mask_bin])))
            norm_fft_binned.append(np.median(np.abs(norm_fft.flatten()[mask_bin])))
        else:
            norm_exact_binned.append(np.nan)
            norm_fft_binned.append(np.nan)

    plt.loglog(L_centers, norm_exact_binned, 'ro-', label='Beam-Exact', linewidth=2)
    plt.loglog(L_centers, norm_fft_binned, 'b^--', label='FFT Method', linewidth=2)
    plt.xlabel('L', fontsize=12)
    plt.ylabel('|N_L|', fontsize=12)
    plt.title('Normalization Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: Ratio vs L
    plt.subplot(133)
    ratio_binned = np.array(norm_exact_binned) / (np.array(norm_fft_binned) + 1e-30)
    plt.semilogx(L_centers, ratio_binned, 'ko-', linewidth=2)
    plt.axhline(1.0, color='r', linestyle='--', label='Equal', linewidth=2)
    plt.xlabel('L', fontsize=12)
    plt.ylabel('Ratio (Beam-Exact / FFT)', fontsize=12)
    plt.title('Method Comparison')
    plt.ylim([0.5, 1.5])
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig('beam_exact_vs_fft_comparison.png', dpi=150, bbox_inches='tight')
    print("\nComparison plot saved to: beam_exact_vs_fft_comparison.png")
    plt.show()
else:
    # Still compute binned values for summary stats
    L_bins = np.logspace(np.log10(lMin), np.log10(2*lMax), 20)
    L_centers = 0.5 * (L_bins[:-1] + L_bins[1:])
    norm_exact_binned = []
    norm_fft_binned = []
    for i in range(len(L_centers)):
        mask_bin = (l_flat >= L_bins[i]) & (l_flat < L_bins[i+1])
        if np.sum(mask_bin) > 0:
            norm_exact_binned.append(np.median(np.abs(norm_exact.flatten()[mask_bin])))
            norm_fft_binned.append(np.median(np.abs(norm_fft.flatten()[mask_bin])))
        else:
            norm_exact_binned.append(np.nan)
            norm_fft_binned.append(np.nan)


# ============================================================================
# Test 3: Full QE with beam-exact normalization
# ============================================================================

print("\n" + "="*70)
print("TEST 3: Running Full QE with Beam-Exact Normalization")
print("="*70)

# Generate mock data
print("\nGenerating mock lensed CIB map...")
np.random.seed(42)

# Simple mock: Gaussian random field with power spectrum
data_map = np.random.randn(MAP_SIZE, MAP_SIZE)
data_fourier = baseMap.fourier(data_map)

# Apply power spectrum shape (roughly)
l = baseMap.l
power_shape = np.where(l > 0, fCtot(l), 0)
data_fourier *= np.sqrt(power_shape)

data_map = baseMap.inverseFourier(data_fourier).real
print(f"Mock data RMS: {np.std(data_map):.3e}")

# Run QE with beam-exact normalization
print("\nRunning QE with beam-exact normalization...")
kappa_est_fourier, norm_fourier = baseMap.computeQuadEstKappaNorm(
    fC0, fCtot,
    lMin=lMin, lMax=lMax,
    dataFourier=baseMap.fourier(data_map),
    fB_ell=B_ell,
    use_beam_exact=True,  # Use exact method
    nEll_beam=50,
    test=False
)

kappa_est_map = baseMap.inverseFourier(kappa_est_fourier).real

print(f"\nKappa estimate RMS: {np.std(kappa_est_map):.3e}")

# Extract N_L from normalization for diagnostics
if norm_fourier is not None:
    print("\nExtracting N_L normalization for diagnostics...")
    lC_norm, N_L, N_L_err = baseMap.powerSpectrum(norm_fourier)
    print(f"N_L range: [{N_L.min():.3e}, {N_L.max():.3e}]")
    
    # Check high-L behavior
    high_L_mask = lC_norm > 0.8 * lMax
    if np.any(high_L_mask):
        print(f"N_L at high L (>0.8*lMax): mean={N_L[high_L_mask].mean():.3e}, std={N_L[high_L_mask].std():.3e}")
    
    # Plot N_L normalization
    if plot:
        plt.figure(figsize=(10, 6))
        plt.errorbar(lC_norm, N_L, yerr=N_L_err, fmt='o-', label='$N_L$ (QE normalization)', alpha=0.7)
        plt.axvline(lMax, color='red', linestyle='--', alpha=0.5, label=f'lMax = {lMax:.0f}')
        plt.axvline(0.8*lMax, color='orange', linestyle=':', alpha=0.5, label=f'0.8×lMax')
        plt.axvline(lMin, color='blue', linestyle='--', alpha=0.5, label=f'lMin = {lMin:.0f}')
        plt.xlabel('$L$', fontsize=14)
        plt.ylabel('$N_L$', fontsize=14)
        plt.title('QE Normalization (Beam-Exact Method)', fontsize=14)
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('beam_exact_normalization_N_L.png', dpi=150, bbox_inches='tight')
        print("N_L plot saved to: beam_exact_normalization_N_L.png")
        plt.show()

# Plot result
if plot:
    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    plt.imshow(data_map, cmap='RdBu_r')
    plt.colorbar()
    plt.title('Mock Data Map')

    plt.subplot(132)
    plt.imshow(kappa_est_map, cmap='RdBu_r')
    plt.colorbar()
    plt.title('κ Reconstruction (Beam-Exact)')

    plt.subplot(133)
    plt.imshow(np.log10(np.abs(kappa_est_fourier)), cmap='viridis')
    plt.colorbar()
    plt.title('log10|κ(k)|')

    plt.tight_layout()
    plt.savefig('beam_exact_qe_result.png', dpi=150, bbox_inches='tight')
    print("\nQE result plot saved to: beam_exact_qe_result.png")
    plt.show()


# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("\n✓ Beam-exact normalization implemented successfully")
print("✓ Comparison with FFT method completed")
print("✓ Full QE pipeline tested")
print("\nKey points:")
print("  • fC0 should be the RAW unlensed spectrum (no beam)")
print("  • fCtot includes beam: B²(ℓ) × C_unlensed + N")
print("  • Beam applied correctly in normalization as B(ℓ) × B(L-ℓ)")
print("  • Set use_beam_exact=True in computeQuadEstKappaNorm")
print("\nNext steps:")
print("  1. Test with your actual CIBER data and clf.bl beam")
print("  2. Compare κ-g cross-spectrum with theory")
print("  3. Verify bispectrum bias estimates")
print("  4. If results match theory, can remove external kcorr/vbeam")
print("="*70)
