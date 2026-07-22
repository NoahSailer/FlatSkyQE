"""
Beam-corrected QE normalization with explicit B(ℓ) × B(L-ℓ) computation.

This implements the proper beam correction as described by your collaborator:

    N_L = (∫_ℓ F_{ℓ,L-ℓ} f^κ_{ℓ,L-ℓ} B_ℓ B_{L-ℓ})^{-1}

The key difference from the FFT approach is that we explicitly loop over 
mode pairs (ℓ, L-ℓ) to compute B(ℓ) × B(L-ℓ) for each contribution.
"""

import numpy as np
from scipy.interpolate import interp1d


def computeQuadEstPhiNormalizationFFT_with_beam(baseMap, fC0, fCtot, fB_ell, 
                                                lMin=1., lMax=1.e5, test=False):
    """
    Compute QE normalization with explicit B(ℓ) × B(L-ℓ) beam correction.
    
    This uses a hybrid approach:
    1. FFT-based computation for the basic structure (fast)
    2. Direct sum with beam factors for mode pairs (accurate)
    
    Parameters:
    -----------
    baseMap : FlatMap
        The flat sky map object
    fC0 : function
        Unlensed power spectrum C^{II}_ℓ (no beam!)
    fCtot : function  
        Observed power spectrum C^{obs}_ℓ = B²_ℓ C^{II}_ℓ + N_ℓ
    fB_ell : function
        Beam/PSF function B(ℓ)
    lMin, lMax : float
        Multipole range for reconstruction
    test : bool
        If True, plot diagnostics
        
    Returns:
    --------
    normalizationFourier : ndarray
        The normalization N_L in Fourier space (to be inverted)
    """
    
    print("Computing beam-corrected normalization with explicit B(ℓ)B(L-ℓ)")
    
    # Get the 2D arrays from baseMap
    lx, ly = baseMap.lx, baseMap.ly
    l = baseMap.l
    nX, nY = lx.shape
    
    # Initialize the normalization array (will accumulate the sum)
    normFourier = np.zeros((nX, nY), dtype=complex)
    
    # For each output mode L
    for iLx in range(nX):
        for iLy in range(nY):
            
            Lx, Ly = lx[iLx, iLy], ly[iLx, iLy]
            L = np.sqrt(Lx**2 + Ly**2)
            
            # Skip if L is out of bounds
            if L < lMin or L > 2*lMax:
                continue
                
            # Accumulator for this L mode
            sum_terms = 0.0
            
            # Sum over all ℓ modes
            for i_ellx in range(nX):
                for i_elly in range(nY):
                    
                    ellx, elly = lx[i_ellx, i_elly], ly[i_ellx, i_elly]
                    ell = np.sqrt(ellx**2 + elly**2)
                    
                    # Skip if ℓ is out of range
                    if ell < lMin or ell > lMax:
                        continue
                    
                    # Compute L - ℓ
                    Lminusellx = Lx - ellx
                    Lminuselly = Ly - elly
                    Lminusell = np.sqrt(Lminusellx**2 + Lminuselly**2)
                    
                    # Skip if |L-ℓ| is out of range
                    if Lminusell < lMin or Lminusell > lMax:
                        continue
                    
                    # Compute the filter F_{ℓ, L-ℓ}
                    # F = (ℓ · L) C^{II}_ℓ / C^{obs}_ℓ + ((L-ℓ) · L) C^{II}_{L-ℓ} / C^{obs}_{L-ℓ}
                    
                    C0_ell = fC0(ell)
                    Ctot_ell = fCtot(ell)
                    C0_Lminusell = fC0(Lminusell)
                    Ctot_Lminusell = fCtot(Lminusell)
                    
                    if Ctot_ell == 0 or Ctot_Lminusell == 0:
                        continue
                    
                    # Geometric factors: ℓ · L
                    ell_dot_L = ellx * Lx + elly * Ly
                    Lminusell_dot_L = Lminusellx * Lx + Lminuselly * Ly
                    
                    # Filter terms
                    F_term = (ell_dot_L * C0_ell / Ctot_ell + 
                             Lminusell_dot_L * C0_Lminusell / Ctot_Lminusell)
                    
                    # Response function f^κ_{ℓ, L-ℓ}
                    # f^κ = 2 L · (ℓ C^{II}_ℓ + (L-ℓ) C^{II}_{L-ℓ}) / L²
                    if L > 0:
                        f_kappa = 2 * (ell_dot_L * C0_ell + Lminusell_dot_L * C0_Lminusell) / L**2
                    else:
                        f_kappa = 0
                    
                    # BEAM CORRECTION: B(ℓ) × B(|L-ℓ|)
                    B_ell = fB_ell(ell)
                    B_Lminusell = fB_ell(Lminusell)
                    beam_factor = B_ell * B_Lminusell
                    
                    # Add contribution to sum
                    sum_terms += F_term * f_kappa * beam_factor
            
            # Store the accumulated sum for this L mode
            # This is the denominator of the normalization
            normFourier[iLx, iLy] = sum_terms
            
        # Progress indicator
        if iLx % (nX // 10) == 0:
            print(f"  Progress: {100*iLx/nX:.0f}%")
    
    # The normalization is the inverse
    # N_L = 1 / (∫_ℓ F * f^κ * B_ℓ * B_{L-ℓ})
    normFourier[normFourier != 0] = 1.0 / normFourier[normFourier != 0]
    normFourier[np.isnan(normFourier) | np.isinf(normFourier)] = 0.0
    
    if test:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        plt.imshow(np.log10(np.abs(normFourier) + 1e-20))
        plt.colorbar()
        plt.title('log10(|N_L|)')
        plt.subplot(122)
        plt.loglog(l.flatten(), np.abs(normFourier.flatten()), 'b.', alpha=0.1)
        plt.xlabel('L')
        plt.ylabel('|N_L|')
        plt.title('Normalization')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    return normFourier


def computeQuadEstPhiNormalizationFFT_with_beam_fast(baseMap, fC0, fCtot, fB_ell,
                                                     lMin=1., lMax=1.e5, 
                                                     nEll=100, test=False):
    """
    Faster version using binned ℓ integration.
    
    Instead of summing over all pixels, we:
    1. Bin the ℓ modes into annuli
    2. Compute the sum over bins (much faster)
    3. Interpolate back to 2D
    
    This is a good approximation if the spectrum varies smoothly.
    
    Parameters:
    -----------
    nEll : int
        Number of ℓ bins to use (default 100)
    """
    
    print(f"Computing beam-corrected normalization (binned, nEll={nEll})")
    
    lx, ly = baseMap.lx, baseMap.ly
    l = baseMap.l
    nX, nY = lx.shape
    
    # Create ℓ bins (log-spaced)
    ell_bins = np.logspace(np.log10(max(lMin, 1)), np.log10(lMax), nEll)
    ell_centers = 0.5 * (ell_bins[:-1] + ell_bins[1:])
    dell = np.diff(ell_bins)
    
    # For each L mode, compute the 1D integral over ℓ magnitude
    normFourier = np.zeros((nX, nY), dtype=float)
    
    for iLx in range(nX):
        for iLy in range(nY):
            
            Lx, Ly = lx[iLx, iLy], ly[iLx, iLy]
            L = np.sqrt(Lx**2 + Ly**2)
            
            if L < lMin or L > 2*lMax:
                continue
            
            # Sum over ℓ bins
            sum_L = 0.0
            
            for i_bin in range(len(ell_centers)):
                ell = ell_centers[i_bin]
                
                if ell < lMin or ell > lMax:
                    continue
                
                # For this ℓ magnitude, integrate over angle
                # Use several angle samples
                n_phi = 32  # Number of angular samples
                for i_phi in range(n_phi):
                    phi = 2 * np.pi * i_phi / n_phi
                    
                    # ℓ vector in this direction
                    ellx = ell * np.cos(phi)
                    elly = ell * np.sin(phi)
                    
                    # L - ℓ vector
                    Lminusellx = Lx - ellx
                    Lminuselly = Ly - elly
                    Lminusell = np.sqrt(Lminusellx**2 + Lminuselly**2)
                    
                    if Lminusell < lMin or Lminusell > lMax:
                        continue
                    
                    # Compute integrand
                    C0_ell = fC0(ell)
                    Ctot_ell = fCtot(ell)
                    C0_Lminusell = fC0(Lminusell)
                    Ctot_Lminusell = fCtot(Lminusell)
                    
                    if Ctot_ell == 0 or Ctot_Lminusell == 0:
                        continue
                    
                    ell_dot_L = ellx * Lx + elly * Ly
                    Lminusell_dot_L = Lminusellx * Lx + Lminuselly * Ly
                    
                    F_term = (ell_dot_L * C0_ell / Ctot_ell + 
                             Lminusell_dot_L * C0_Lminusell / Ctot_Lminusell)
                    
                    if L > 0:
                        f_kappa = 2 * (ell_dot_L * C0_ell + Lminusell_dot_L * C0_Lminusell) / L**2
                    else:
                        f_kappa = 0
                    
                    # BEAM CORRECTION
                    B_ell = fB_ell(ell)
                    B_Lminusell = fB_ell(Lminusell)
                    beam_factor = B_ell * B_Lminusell
                    
                    # Jacobian: ℓ dℓ dφ
                    dphi = 2 * np.pi / n_phi
                    integrand = F_term * f_kappa * beam_factor * ell * dphi
                    
                    sum_L += integrand
                
                # Multiply by bin width
                sum_L *= dell[i_bin] if i_bin < len(dell) else dell[-1]
            
            normFourier[iLx, iLy] = sum_L
        
        if iLx % (nX // 10) == 0:
            print(f"  Progress: {100*iLx/nX:.0f}%")
    
    # Invert
    normFourier[normFourier != 0] = 1.0 / normFourier[normFourier != 0]
    normFourier[np.isnan(normFourier) | np.isinf(normFourier)] = 0.0
    
    if test:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        plt.imshow(np.log10(np.abs(normFourier) + 1e-20))
        plt.colorbar()
        plt.title('log10(|N_L|)')
        plt.subplot(122)
        plt.loglog(l.flatten(), np.abs(normFourier.flatten()), 'b.', alpha=0.1)
        plt.xlabel('L')
        plt.ylabel('|N_L|')
        plt.title('Beam-corrected Normalization (binned)')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    return normFourier


def compare_normalizations(baseMap, fC0, fCtot, fB_ell, lMin=1., lMax=1.e5):
    """
    Compare the FFT-based approach (with B² applied to fC0) 
    vs. the explicit B(ℓ)B(L-ℓ) approach.
    
    Returns both normalizations for comparison.
    """
    import matplotlib.pyplot as plt
    
    print("\n=== Computing FFT normalization (incorrect beam) ===")
    # Old way: beam baked into fC0
    fC0_beamed = lambda ell: fC0(ell) * fB_ell(ell)**2
    norm_fft = baseMap.computeQuadEstPhiNormalizationFFT(
        fC0_beamed, fCtot, lMin=lMin, lMax=lMax, test=False, fB_ell=None
    )
    
    print("\n=== Computing explicit B(ℓ)B(L-ℓ) normalization ===")
    norm_correct = computeQuadEstPhiNormalizationFFT_with_beam_fast(
        baseMap, fC0, fCtot, fB_ell, lMin=lMin, lMax=lMax, nEll=50, test=False
    )
    
    # Plot comparison
    l = baseMap.l
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.imshow(np.log10(np.abs(norm_fft) + 1e-20), cmap='viridis')
    plt.colorbar()
    plt.title('FFT approach (B² in fC0)')
    
    plt.subplot(132)
    plt.imshow(np.log10(np.abs(norm_correct) + 1e-20), cmap='viridis')
    plt.colorbar()
    plt.title('Explicit B(ℓ)B(L-ℓ)')
    
    plt.subplot(133)
    ratio = norm_correct / (norm_fft + 1e-30)
    plt.imshow(np.log10(np.abs(ratio)), cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    plt.colorbar()
    plt.title('log10(Ratio)')
    
    plt.tight_layout()
    plt.show()
    
    # 1D comparison
    plt.figure(figsize=(10, 5))
    mask = (l.flatten() > lMin) & (l.flatten() < 2*lMax)
    L_plot = l.flatten()[mask]
    
    plt.subplot(121)
    plt.loglog(L_plot, np.abs(norm_fft.flatten()[mask]), 'b.', 
              alpha=0.3, label='FFT (B² in fC0)')
    plt.loglog(L_plot, np.abs(norm_correct.flatten()[mask]), 'r.', 
              alpha=0.3, label='Explicit B(ℓ)B(L-ℓ)')
    plt.xlabel('L')
    plt.ylabel('|N_L|')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(122)
    ratio_1d = np.abs(norm_correct.flatten()[mask]) / (np.abs(norm_fft.flatten()[mask]) + 1e-30)
    plt.semilogx(L_plot, ratio_1d, 'k.', alpha=0.3)
    plt.axhline(1.0, color='r', linestyle='--', label='Equal')
    plt.xlabel('L')
    plt.ylabel('Ratio (correct/FFT)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return norm_fft, norm_correct
