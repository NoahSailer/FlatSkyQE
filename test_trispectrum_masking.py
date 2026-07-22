"""
Test how bright source masking affects the trispectrum contribution to C_L^kg noise.

The trispectrum of the intensity field contributes non-Gaussian noise to kappa-galaxy
cross-correlations. Bright sources likely dominate this, so masking them should reduce
the trispectrum contribution.

This module provides tools to:
1. Generate mock intensity maps with/without bright sources
2. Apply different magnitude-based masks
3. Measure collapsed trispectrum for each configuration
4. Compare N_L^{kg} with/without trispectrum contributions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import sys
sys.path.append('/Users/richardfeder/Documents/ciber/FlatSkyQE/')
from flat_map import FlatMap


def generate_mock_intensity_map_with_sources(flatmap, n_sources_per_deg2, mag_min, mag_max,
                                              flux_normalization=1e7, m0=18, add_diffuse=True,
                                              clkg_scale=1.0, seed=None):
    """
    Generate mock CIBER-like intensity map with point sources and optional diffuse component.
    
    Parameters
    ----------
    flatmap : FlatMap
        FlatMap object defining geometry
    n_sources_per_deg2 : float
        Surface density of sources [deg^-2]
    mag_min, mag_max : float
        Magnitude range for source population
    flux_normalization : float
        Scaling factor for flux = flux_norm * 10^(-0.4*m)
    m0 : float
        Reference magnitude for number counts dN/dm ∝ (m-m0)^2
    add_diffuse : bool
        Add diffuse CIB component with power-law C_ell
    clkg_scale : float
        Scaling for diffuse C_ell amplitude
    seed : int or None
        Random seed
        
    Returns
    -------
    intensity_map : array
        2D intensity map [nW/m^2/sr-like units]
    source_positions : array
        (x, y, flux, mag) for each source
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Map geometry
    dimx, dimy = flatmap.nX, flatmap.nY
    pixsize_arcsec = flatmap.pixScaleX * 3600  # Convert deg to arcsec
    area_deg2 = (dimx * pixsize_arcsec / 3600) * (dimy * pixsize_arcsec / 3600)
    
    # Number of sources
    n_sources = int(n_sources_per_deg2 * area_deg2)
    
    # Generate magnitudes: dN/dm ∝ (m - m0)^2
    # CDF: F(m) ∝ (m - m0)^3 / 3
    u = np.random.uniform(0, 1, n_sources)
    F_max = (mag_max - m0)**3 / 3
    F_min = (mag_min - m0)**3 / 3
    mags = m0 + ((F_min + u * (F_max - F_min)) * 3)**(1./3.)
    
    # Convert to fluxes
    fluxes = flux_normalization * 10**(-0.4 * mags)
    
    # Random positions (pixel coordinates)
    x_pos = np.random.uniform(0, dimx, n_sources)
    y_pos = np.random.uniform(0, dimy, n_sources)
    
    # Create intensity map by placing point sources
    intensity_map = np.zeros((dimx, dimy))
    for x, y, flux in zip(x_pos, y_pos, fluxes):
        ix, iy = int(x), int(y)
        if 0 <= ix < dimx and 0 <= iy < dimy:
            intensity_map[ix, iy] += flux
    
    # Add diffuse component if requested
    if add_diffuse:
        # Power-law C_ell ∝ ell^-1.5
        def cl_diffuse(ell):
            ell0 = 1000.
            cl0 = 1e-8 * clkg_scale  # Adjust amplitude
            return cl0 * (ell / ell0)**(-1.5) * (ell > 100)
        
        diffuse_fourier = flatmap.genGRF(cl_diffuse, test=False)
        diffuse_real = flatmap.inverseFourier(diffuse_fourier)
        intensity_map += diffuse_real
    
    # Mean subtract
    intensity_map -= np.mean(intensity_map)
    
    source_positions = np.column_stack([x_pos, y_pos, fluxes, mags])
    
    return intensity_map, source_positions


def apply_magnitude_mask(intensity_map, source_positions, mag_cut, flatmap, mask_radius_arcsec=7):
    """
    Create mask removing sources brighter than mag_cut.
    
    Parameters
    ----------
    intensity_map : array
        2D intensity map
    source_positions : array
        (x, y, flux, mag) for each source
    mag_cut : float
        Mask sources with m < mag_cut (brighter than cut)
    flatmap : FlatMap
        For geometry
    mask_radius_arcsec : float
        Radius to mask around each bright source [arcsec]
        
    Returns
    -------
    masked_map : array
        Intensity map with bright sources masked
    mask : array
        Binary mask (1=keep, 0=masked)
    n_masked : int
        Number of sources masked
    """
    dimx, dimy = intensity_map.shape
    mask = np.ones((dimx, dimy))
    
    # Find bright sources
    bright_sources = source_positions[source_positions[:, 3] < mag_cut]
    n_masked = len(bright_sources)
    
    # Mask radius in pixels
    pixsize_arcsec = flatmap.pixScaleX * 3600
    mask_radius_pix = mask_radius_arcsec / pixsize_arcsec
    
    # Mask around each bright source
    for x, y, flux, mag in bright_sources:
        ix, iy = int(x), int(y)
        # Create circular mask
        yy, xx = np.ogrid[:dimx, :dimy]
        circle_mask = ((xx - ix)**2 + (yy - iy)**2) <= mask_radius_pix**2
        mask[circle_mask] = 0
    
    masked_map = intensity_map * mask
    masked_map -= np.mean(masked_map[mask > 0.5])  # Re-mean subtract
    
    return masked_map, mask, n_masked


def test_trispectrum_vs_masking(n_sources=1e5, mag_cuts=[15, 16, 17, 18, 19, 20],
                                mag_min=15, mag_max=25, m0=18,
                                flux_normalization=1e8,
                                map_size_deg=2.0, pixsize_arcsec=7.0,
                                mask_radius_arcsec=7.0,
                                nBins=25, seed=42):
    """
    Test how magnitude-based masking affects the measured trispectrum.
    
    Parameters
    ----------
    n_sources : float
        Total number of sources per deg^2
    mag_cuts : list
        List of magnitude cuts to test (mask m < cut)
    mag_min, mag_max : float
        Magnitude range for full population
    m0 : float
        Reference magnitude for dN/dm distribution
    flux_normalization : float
        Flux scaling
    map_size_deg : float
        Map side length [deg]
    pixsize_arcsec : float
        Pixel size [arcsec]
    mask_radius_arcsec : float
        Masking radius around bright sources [arcsec]
    nBins : int
        Number of bins for collapsed trispectrum
    seed : int
        Random seed
        
    Returns
    -------
    results : dict
        Dictionary containing trispectrum measurements for each mask
    fig : matplotlib figure
    """
    
    # Create FlatMap
    npix = int(map_size_deg * 3600 / pixsize_arcsec)
    pixsize_deg = pixsize_arcsec / 3600
    flatmap = FlatMap(nX=npix, nY=npix, pixScaleX=pixsize_deg, pixScaleY=pixsize_deg)
    
    print(f"Map: {npix}x{npix} pixels, {map_size_deg:.2f} deg on side")
    print(f"Generating mock with {n_sources:.0e} sources/deg^2, {mag_min} < m < {mag_max}")
    
    # Generate mock intensity map
    intensity_map, source_positions = generate_mock_intensity_map_with_sources(
        flatmap, n_sources, mag_min, mag_max,
        flux_normalization=flux_normalization, m0=m0,
        add_diffuse=True, seed=seed
    )
    
    print(f"Generated {len(source_positions)} sources")
    print(f"Flux range: {np.min(source_positions[:,2]):.2e} to {np.max(source_positions[:,2]):.2e}")
    
    # Also generate Gaussian reference with same C_ell
    intensity_fourier = flatmap.fourier(intensity_map)
    lCen, Cl_intensity, _ = flatmap.powerSpectrum(intensity_fourier, nBins=50, plot=False)
    cl_interp = interp1d(lCen, Cl_intensity, bounds_error=False, fill_value=0)
    gaussian_fourier = flatmap.genGRF(cl_interp, test=False)
    
    # Storage for results
    results = {
        'mag_cuts': mag_cuts,
        'n_masked_sources': [],
        'mask_fractions': [],
        'trispectrum_ng': {},  # Non-Gaussian (data)
        'trispectrum_g': {},   # Gaussian reference
        'trispectrum_excess': {},  # NG - G
        'l_bins': None,
        'l_mean_values': None
    }
    
    # Test each masking level
    for mag_cut in mag_cuts:
        print(f"\n{'='*60}")
        print(f"Testing magnitude cut: m < {mag_cut} (masking brighter sources)")
        print(f"{'='*60}")
        
        # Apply mask
        masked_map, mask, n_masked = apply_magnitude_mask(
            intensity_map, source_positions, mag_cut, flatmap,
            mask_radius_arcsec=mask_radius_arcsec
        )
        
        mask_frac = np.sum(mask) / mask.size
        results['n_masked_sources'].append(n_masked)
        results['mask_fractions'].append(mask_frac)
        
        print(f"Masked {n_masked} sources brighter than m={mag_cut}")
        print(f"Mask fraction: {mask_frac:.3f}")
        
        # Apply same mask to Gaussian
        gaussian_real = flatmap.inverseFourier(gaussian_fourier)
        masked_gaussian = gaussian_real * mask
        masked_gaussian -= np.mean(masked_gaussian[mask > 0.5])
        
        # Fourier transforms
        masked_fourier = flatmap.fourier(masked_map)
        masked_gaussian_fourier = flatmap.fourier(masked_gaussian)
        
        # Compute collapsed trispectrum
        print("Computing trispectrum for non-Gaussian map...")
        flatmap.saveTrispectrum(dataFourier=masked_fourier, 
                               gaussDataFourier=masked_gaussian_fourier,
                               path=f"./output/trispec_magcut_{mag_cut}_",
                               nBins=nBins)
        
        # Load results (saveTrispectrum saves to file, need to load)
        # For now, store the key parameters
        key = f"m<{mag_cut}"
        # You would load the actual trispectrum here
        # results['trispectrum_ng'][key] = ...
        
    return results, flatmap


def plot_trispectrum_vs_masking(results):
    """
    Plot how trispectrum changes with masking.
    
    Parameters
    ----------
    results : dict
        Output from test_trispectrum_vs_masking
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel 1: Number of masked sources vs magnitude cut
    ax = axes[0, 0]
    ax.plot(results['mag_cuts'], results['n_masked_sources'], 'o-', 
            linewidth=2, markersize=8, color='blue')
    ax.set_xlabel('Magnitude cut', fontsize=12)
    ax.set_ylabel('Number of masked sources', fontsize=12)
    ax.set_title('Bright source removal', fontsize=12)
    ax.grid(alpha=0.3)
    
    # Panel 2: Mask fraction vs magnitude cut
    ax = axes[0, 1]
    ax.plot(results['mag_cuts'], results['mask_fractions'], 'o-',
            linewidth=2, markersize=8, color='red')
    ax.set_xlabel('Magnitude cut', fontsize=12)
    ax.set_ylabel('Unmasked fraction', fontsize=12)
    ax.set_title('Sky coverage vs masking', fontsize=12)
    ax.grid(alpha=0.3)
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.5)
    
    # Panel 3: Trispectrum amplitude vs masking
    # (Would show actual trispectrum values here)
    ax = axes[1, 0]
    ax.set_xlabel('Magnitude cut', fontsize=12)
    ax.set_ylabel('Trispectrum amplitude', fontsize=12)
    ax.set_title('Non-Gaussian contribution', fontsize=12)
    ax.grid(alpha=0.3)
    ax.text(0.5, 0.5, 'Trispectrum\nmeasurements\nwould go here',
            transform=ax.transAxes, ha='center', va='center',
            fontsize=14, alpha=0.3)
    
    # Panel 4: Relative reduction in trispectrum
    ax = axes[1, 1]
    ax.set_xlabel('Magnitude cut', fontsize=12)
    ax.set_ylabel('Trispectrum reduction factor', fontsize=12)
    ax.set_title('Masking efficiency', fontsize=12)
    ax.grid(alpha=0.3)
    ax.text(0.5, 0.5, 'Trispectrum\nreduction\nwould go here',
            transform=ax.transAxes, ha='center', va='center',
            fontsize=14, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def estimate_nlkg_with_trispectrum(flatmap, cl_kappa, cl_galaxy, n_bar_deg2,
                                   trispectrum_amplitude, L_range=None):
    """
    Estimate N_L^{kg} including trispectrum contribution from intensity field.
    
    The noise includes Gaussian terms plus trispectrum:
    N_L^{kg} ≈ (C_L^κκ + N_L^κ) * (C_L^gg + 1/n̄) + (trispectrum contribution)
    
    Parameters
    ----------
    flatmap : FlatMap
        For computing conversion factor
    cl_kappa : callable
        C_L^κκ(L)
    cl_galaxy : callable
        C_L^gg(L)
    n_bar_deg2 : float
        Galaxy surface density [deg^-2]
    trispectrum_amplitude : float or callable
        Measured trispectrum, either constant or function of L
    L_range : array or None
        L values to evaluate at
        
    Returns
    -------
    L : array
        Multipoles
    nlkg_gaussian : array
        Gaussian N_L^{kg}
    nlkg_with_trispec : array
        Total N_L^{kg} including trispectrum
    trispec_contribution : array
        Trispectrum term alone
    """
    
    if L_range is None:
        L_range = np.logspace(2, 4, 50)
    
    # Gaussian noise (standard formula)
    nlkg_gaussian = np.zeros_like(L_range)
    for i, L in enumerate(L_range):
        nlkg_gaussian[i] = np.sqrt((cl_kappa(L)) * (cl_galaxy(L) + 1./n_bar_deg2))
    
    # Trispectrum contribution (simplified - real calculation more complex)
    if callable(trispectrum_amplitude):
        trispec_contribution = trispectrum_amplitude(L_range)
    else:
        trispec_contribution = trispectrum_amplitude * np.ones_like(L_range)
    
    nlkg_with_trispec = np.sqrt(nlkg_gaussian**2 + trispec_contribution**2)
    
    return L_range, nlkg_gaussian, nlkg_with_trispec, trispec_contribution


if __name__ == "__main__":
    print("Testing trispectrum sensitivity to bright source masking")
    print("="*70)
    
    # Run test
    results, flatmap = test_trispectrum_vs_masking(
        n_sources=1e5,
        mag_cuts=[15, 16, 17, 18, 19, 20],
        mag_min=15,
        mag_max=25,
        flux_normalization=1e8,
        map_size_deg=2.0,
        pixsize_arcsec=7.0,
        nBins=25,
        seed=42
    )
    
    # Plot results
    fig = plot_trispectrum_vs_masking(results)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("Magnitude cuts tested:", results['mag_cuts'])
    print("Sources masked:", results['n_masked_sources'])
    print("Mask fractions:", [f"{f:.3f}" for f in results['mask_fractions']])
    print("\nNext steps:")
    print("1. Extract actual trispectrum measurements from saved files")
    print("2. Compare trispectrum amplitude vs masking level")
    print("3. Compute impact on N_L^{kg} using estimate_nlkg_with_trispectrum()")
    print("4. Test on real CIBER data with actual bright source catalogs")
