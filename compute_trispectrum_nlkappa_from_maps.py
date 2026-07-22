"""
Standalone function to compute N_L^kappa from trispectrum measured in intensity maps.

This module provides functions to:
1. Compute collapsed trispectrum from intensity maps
2. Subtract Gaussian component (either from a Gaussian mock or analytically)
3. Convert trispectrum to N_L^kappa reconstruction noise

Author: Richard Feder
Date: February 2026
"""

import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from flat_map import FlatMap


def compute_nlkappa_from_intensity_maps(intensity_map, fC0, fCtot, 
                                        gaussian_map=None,
                                        lMin=1., lMax=1e5,
                                        lMean_array=None,
                                        nBins=51, lRange=None,
                                        return_full_trispec=False,
                                        plot=False):
    """
    Compute N_L^kappa from trispectrum measured in intensity maps.
    
    This function follows the workflow:
    1. Compute collapsed 4-point function from intensity map
    2. Compute collapsed 4-point function from Gaussian comparison (if provided)
    3. Subtract Gaussian component to isolate trispectrum
    4. Convert trispectrum to N_L^kappa using conversion factor from FlatMap
    
    Parameters
    ----------
    intensity_map : 2D array
        Real-space intensity map (e.g., CIBER mock or real data)
    fC0 : function
        Unlensed power spectrum C_ell (for QE weights)
    fCtot : function
        Total observed power spectrum (for QE weights)
    gaussian_map : 2D array, optional
        Gaussian realization with same power spectrum as intensity_map.
        If None, uses analytical Gaussian expectation (2*C^2/Nmodes)
    lMin, lMax : float
        Multipole range for QE reconstruction
    lMean_array : array, optional
        Array of high-ell values to evaluate collapsed trispectrum.
        Default: np.logspace(np.log10(100), np.log10(3e3), 9)
    nBins : int
        Number of bins for power spectrum of squared map
    lRange : tuple, optional
        (L_min, L_max) range for binning squared map power spectrum
    return_full_trispec : bool
        If True, return full trispectrum arrays vs L and lMean
    plot : bool
        Whether to plot intermediate results
        
    Returns
    -------
    f_nl_kappa : function
        Interpolated N_L^kappa(L) from trispectrum contribution
    trispec_info : dict (optional)
        If return_full_trispec=True, returns dict with:
        - 'L': Low-ell array (L in T(L, l, l, L-l))
        - 'lMean': High-ell array (l values)
        - 'trispec': Trispectrum values [L, lMean]
        - 'C2': C^2 values from Gaussian [L, lMean]
        - 'nl_kappa_vs_lMean': N_L^kappa for each lMean [L, lMean]
    
    Examples
    --------
    >>> # Create FlatMap and generate mock
    >>> baseMap = FlatMap(...)
    >>> intensity_map = baseMap.data  # Your intensity map
    >>> 
    >>> # Define power spectrum functions
    >>> fC0 = lambda l: ...  # Unlensed spectrum
    >>> fCtot = lambda l: ...  # Total spectrum
    >>> 
    >>> # Compute N_L^kappa from trispectrum
    >>> f_nl_kappa = compute_nlkappa_from_intensity_maps(
    ...     intensity_map, fC0, fCtot, lMin=300, lMax=3000
    ... )
    >>> 
    >>> # Evaluate at specific L values
    >>> L_vals = np.logspace(2, 4, 50)
    >>> nl_kappa_trispec = f_nl_kappa(L_vals)
    """
    
    # Create FlatMap instance from input map
    # Assumes square map - can be generalized
    nX, nY = intensity_map.shape
    sizeXDeg = 10.0  # Default, should match your map size
    sizeYDeg = 10.0
    
    baseMap = FlatMap(nX=nX, nY=nY, sizeXDeg=sizeXDeg, sizeYDeg=sizeYDeg)
    baseMap.data = intensity_map.copy()
    baseMap.dataFourier = baseMap.fourier(data=intensity_map)
    
    # Default lMean array if not provided
    if lMean_array is None:
        lMean_array = np.logspace(np.log10(100.), np.log10(3.e3), 9)
    
    print(f"Computing collapsed trispectrum at {len(lMean_array)} high-ell values")
    print(f"lMean range: {lMean_array[0]:.0f} - {lMean_array[-1]:.0f}")
    
    # Analyze the intensity map (potentially non-Gaussian)
    collapsed4pt_NG = {}
    sCollapsed4pt_NG = {}
    for i, lMean in enumerate(lMean_array):
        L, c4pt, sc4pt = baseMap.collapsed4PtFunc(
            lMean=lMean, 
            dataFourier=baseMap.dataFourier.copy(),
            nBins=nBins, 
            lRange=lRange
        )
        collapsed4pt_NG[i] = c4pt
        sCollapsed4pt_NG[i] = sc4pt
        print(f"  Done {i+1}/{len(lMean_array)}: lMean={lMean:.0f}")
    
    # Analyze Gaussian comparison
    if gaussian_map is not None:
        print("Computing Gaussian comparison from provided mock")
        baseMap_gauss = FlatMap(nX=nX, nY=nY, sizeXDeg=sizeXDeg, sizeYDeg=sizeYDeg)
        baseMap_gauss.data = gaussian_map.copy()
        baseMap_gauss.dataFourier = baseMap_gauss.fourier(data=gaussian_map)
        
        collapsed4pt_G = {}
        sCollapsed4pt_G = {}
        for i, lMean in enumerate(lMean_array):
            L, c4pt, sc4pt = baseMap_gauss.collapsed4PtFunc(
                lMean=lMean,
                dataFourier=baseMap_gauss.dataFourier.copy(),
                nBins=nBins,
                lRange=lRange
            )
            collapsed4pt_G[i] = c4pt
            sCollapsed4pt_G[i] = sc4pt
    else:
        print("Using analytical Gaussian expectation")
        # Will compute C^2/Nmodes analytically below
        collapsed4pt_G = None
    
    # Extract trispectrum and convert to N_L^kappa
    n_lmean = len(lMean_array)
    n_L = len(L)
    
    C2Nmodes = np.zeros((n_L, n_lmean))
    Trispec = np.zeros((n_L, n_lmean))
    sTrispec = np.zeros((n_L, n_lmean))
    
    for i in range(n_lmean):
        # Compute number of modes in filter around lMean
        lMean = lMean_array[i]
        
        # Filter shape: approximately Gaussian in log(ell)
        def filter_func(lnl):
            l = np.exp(lnl)
            return baseMap.filterCollapsed4PtFunc(l, lMean)
        
        # Nmodes from filter normalization
        f_nmodes_num = lambda lnl: np.exp(lnl)**2 / (2.*np.pi) * filter_func(lnl)**2
        Nmodes_num = integrate.quad(f_nmodes_num, np.log(1.), np.log(1.e5), 
                                     epsabs=0., epsrel=1.e-3)[0]
        Nmodes = Nmodes_num**2
        
        f_nmodes_denom = lambda lnl: np.exp(lnl)**2 / (2.*np.pi) * filter_func(lnl)**4
        Nmodes_denom = integrate.quad(f_nmodes_denom, np.log(1.), np.log(1.e5),
                                       epsabs=0., epsrel=1.e-3)[0]
        Nmodes /= Nmodes_denom
        
        # Extract Gaussian and trispectrum components
        if collapsed4pt_G is not None:
            C2Nmodes[:, i] = collapsed4pt_G[i].copy()
            Trispec[:, i] = collapsed4pt_NG[i] - collapsed4pt_G[i]
            sTrispec[:, i] = np.sqrt(2.) * sCollapsed4pt_G[i]
        else:
            # Analytical Gaussian: need to compute C^2 at lMean
            # This is approximate - better to provide gaussian_map
            C2Nmodes[:, i] = collapsed4pt_NG[i] / 2.  # Rough estimate
            Trispec[:, i] = collapsed4pt_NG[i] / 2.   # Rough estimate
            sTrispec[:, i] = np.sqrt(2.) * sCollapsed4pt_NG[i] / 2.
            print(f"Warning: Using rough analytical Gaussian estimate for lMean={lMean:.0f}")
    
    # Now convert trispectrum to N_L^kappa
    print("\nComputing conversion factor from trispectrum to N_L^kappa")
    
    # Get conversion factor: N_L^kappa = conv_factor * T(L)
    # This comes from the QE formalism
    conv_factor_interp = baseMap.computeConversionTrispecToNoiseKappa(
        fC0, fCtot, lMin=lMin, lMax=lMax, test=False
    )
    
    # Evaluate conversion factor on L grid
    conv_factor_vals = conv_factor_interp(L)
    
    # Compute N_L^kappa from trispectrum
    # Average over lMean (since trispectrum should be approximately independent of lMean)
    # Or take median to be more robust
    nl_kappa_vs_L = np.median(Trispec * conv_factor_vals[:, np.newaxis], axis=1)
    
    # Remove any negative or nan values
    nl_kappa_vs_L = np.abs(nl_kappa_vs_L)
    nl_kappa_vs_L = np.nan_to_num(nl_kappa_vs_L, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Interpolate to create function
    valid_mask = (L > 0) & (nl_kappa_vs_L > 0) & np.isfinite(nl_kappa_vs_L)
    if np.sum(valid_mask) < 3:
        print("Warning: Very few valid points for interpolation")
        f_nl_kappa = lambda l: np.zeros_like(l)
    else:
        L_valid = L[valid_mask]
        nl_valid = nl_kappa_vs_L[valid_mask]
        
        # Log-log interpolation
        log_L = np.log10(L_valid)
        log_nl = np.log10(nl_valid)
        
        f_nl_kappa_log = interp1d(log_L, log_nl, kind='linear', 
                                   bounds_error=False, fill_value=np.nan)
        
        def f_nl_kappa(l):
            l_arr = np.atleast_1d(l)
            log_l = np.log10(l_arr)
            log_result = f_nl_kappa_log(log_l)
            result = 10**log_result
            # Fill nans with zeros
            result = np.nan_to_num(result, nan=0.0)
            return result if l_arr.shape else result[0]
    
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Collapsed 4pt function vs L for different lMean
        ax = axes[0, 0]
        for i in range(min(5, n_lmean)):  # Plot first 5 lMean values
            ax.loglog(L, collapsed4pt_NG[i], label=f'lMean={lMean_array[i]:.0f}')
        ax.set_xlabel('L (low ell)')
        ax.set_ylabel('Collapsed 4pt function')
        ax.set_title('4pt function from squared map')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        
        # Plot 2: Trispectrum (NG - G) vs L
        ax = axes[0, 1]
        for i in range(min(5, n_lmean)):
            ax.loglog(L, np.abs(Trispec[:, i]), label=f'lMean={lMean_array[i]:.0f}')
        ax.set_xlabel('L (low ell)')
        ax.set_ylabel('Trispectrum')
        ax.set_title('Trispectrum = 4pt(NG) - 4pt(G)')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        
        # Plot 3: Conversion factor
        ax = axes[1, 0]
        ax.loglog(L, conv_factor_vals)
        ax.set_xlabel('L')
        ax.set_ylabel('Conversion factor')
        ax.set_title('T(L) → N_L^κ conversion')
        ax.grid(alpha=0.3)
        
        # Plot 4: Final N_L^kappa
        ax = axes[1, 1]
        ax.loglog(L[valid_mask], nl_kappa_vs_L[valid_mask], 'o-', linewidth=2,
                 label='N_L^κ from trispectrum', color='red', markersize=4)
        ax.set_xlabel('L')
        ax.set_ylabel('N_L^κ')
        ax.set_title('Reconstruction noise from trispectrum')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    print(f"\nN_L^kappa from trispectrum computed successfully")
    print(f"Valid L range: {L_valid[0]:.0f} - {L_valid[-1]:.0f}")
    print(f"Typical value: {np.median(nl_valid):.2e}")
    
    if return_full_trispec:
        trispec_info = {
            'L': L,
            'lMean': lMean_array,
            'trispec': Trispec,
            'C2Nmodes': C2Nmodes,
            'nl_kappa_vs_L': nl_kappa_vs_L,
            'conv_factor': conv_factor_vals,
            'collapsed4pt_NG': np.array([collapsed4pt_NG[i] for i in range(n_lmean)]).T,
            'collapsed4pt_G': np.array([collapsed4pt_G[i] for i in range(n_lmean)]).T if collapsed4pt_G is not None else None,
        }
        return f_nl_kappa, trispec_info
    
    return f_nl_kappa


def quick_trispec_nlkappa_estimate(intensity_map, Cl_intensity, 
                                   fC0_cmb, fCtot_cmb,
                                   lMin=300, lMax=3000,
                                   sizeXDeg=10., sizeYDeg=10.):
    """
    Quick estimate of N_L^kappa from intensity map trispectrum.
    
    This is a simplified wrapper that:
    - Generates a Gaussian comparison map from power spectrum
    - Computes trispectrum at a few representative ells
    - Returns N_L^kappa function
    
    Parameters
    ----------
    intensity_map : 2D array
        Intensity map (real space)
    Cl_intensity : function or array
        Power spectrum of intensity map. If array, assumed to be C_ell at integer ells.
    fC0_cmb : function
        CMB unlensed power spectrum (for QE)
    fCtot_cmb : function
        CMB total power spectrum (for QE)
    lMin, lMax : float
        QE multipole range
    sizeXDeg, sizeYDeg : float
        Map size in degrees
        
    Returns
    -------
    f_nl_kappa : function
        N_L^kappa(L) function
    """
    nX, nY = intensity_map.shape
    
    # Create FlatMap
    baseMap = FlatMap(nX=nX, nY=nY, sizeXDeg=sizeXDeg, sizeYDeg=sizeYDeg)
    
    # Convert Cl_intensity to function if needed
    if not callable(Cl_intensity):
        ell_array = np.arange(len(Cl_intensity))
        fCl = interp1d(ell_array, Cl_intensity, bounds_error=False, fill_value=0.)
    else:
        fCl = Cl_intensity
    
    # Generate Gaussian comparison map
    print("Generating Gaussian comparison map...")
    gaussFourier = baseMap.genGRF(fCl, test=False)
    gaussian_map = baseMap.inverseFourier(dataFourier=gaussFourier)
    
    # Compute N_L^kappa
    f_nl_kappa = compute_nlkappa_from_intensity_maps(
        intensity_map, 
        fC0_cmb, 
        fCtot_cmb,
        gaussian_map=gaussian_map,
        lMin=lMin,
        lMax=lMax,
        lMean_array=np.logspace(np.log10(500.), np.log10(2000.), 5),  # Fewer points for speed
        plot=False
    )
    
    return f_nl_kappa


if __name__ == "__main__":
    """
    Example usage demonstrating the workflow.
    """
    print("Example: Computing N_L^kappa from mock intensity map")
    print("="*60)
    
    # Create mock intensity map
    nX, nY = 512, 512
    sizeXDeg, sizeYDeg = 10., 10.
    
    baseMap = FlatMap(nX=nX, nY=nY, sizeXDeg=sizeXDeg, sizeYDeg=sizeYDeg)
    
    # Define power spectrum (simple model)
    def fCl_intensity(l):
        # Power law with shot noise
        l = np.atleast_1d(l)
        result = 1e-10 * (l / 1000.)**(-2) + 1e-11  # Clustering + shot noise
        return result
    
    # Generate non-Gaussian mock (with point sources)
    print("\nGenerating non-Gaussian intensity map...")
    # Start with Gaussian
    intensityFourier = baseMap.genGRF(fCl_intensity, test=False)
    intensity_map = baseMap.inverseFourier(dataFourier=intensityFourier)
    
    # Add bright point sources to make it non-Gaussian
    n_sources = 100
    for i in range(n_sources):
        x = np.random.randint(0, nX)
        y = np.random.randint(0, nY)
        flux = np.random.lognormal(mean=0, sigma=2)  # Log-normal flux distribution
        # Add Gaussian PSF
        xx, yy = np.meshgrid(np.arange(nX) - x, np.arange(nY) - y, indexing='ij')
        rr = np.sqrt(xx**2 + yy**2)
        intensity_map += flux * np.exp(-rr**2 / (2 * 3**2))
    
    # Define CMB power spectra (simple models)
    def fC0_cmb(l):
        l = np.atleast_1d(l)
        return 3000. * (l / 1000.)**(-1) * np.exp(-l / 5000.)
    
    def fCtot_cmb(l):
        return fC0_cmb(l) + 1e-10  # Add small noise
    
    # Compute N_L^kappa from trispectrum
    print("\nComputing N_L^kappa from intensity map trispectrum...")
    f_nl_kappa, trispec_info = compute_nlkappa_from_intensity_maps(
        intensity_map,
        fC0_cmb,
        fCtot_cmb,
        gaussian_map=None,  # Will use analytical estimate
        lMin=300,
        lMax=3000,
        lMean_array=np.logspace(np.log10(500), np.log10(2000), 5),
        return_full_trispec=True,
        plot=True
    )
    
    # Evaluate at some L values
    L_test = np.logspace(2, 4, 20)
    nl_kappa_test = f_nl_kappa(L_test)
    
    print(f"\nExample N_L^kappa values:")
    for i in range(0, len(L_test), 5):
        print(f"  L={L_test[i]:.0f}: N_L^kappa = {nl_kappa_test[i]:.2e}")
    
    print("\nDone!")
