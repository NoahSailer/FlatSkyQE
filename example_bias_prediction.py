"""
Example: Using power-spectra-based 2-halo bispectrum prediction

This demonstrates the SIMPLE approach: assume biases and extract P_matter from C_ℓ^gg.
No need for CAMB/CLASS!
"""

import numpy as np
from kappa_auto_cross_fns import run_flatskyqe_ciber

# Simple mode: Assume biases, extract P_matter from galaxy auto-spectrum
def run_with_assumed_biases():
    """
    Run CIBER lensing analysis with assumed biases (RECOMMENDED - simplest approach).
    
    This mode:
    1. Uses your measured C_ℓ^gg (galaxy auto-spectrum)
    2. Assumes a linear bias b_g (e.g., 1.5 for low-z galaxies)
    3. Extracts matter power spectrum: P_matter = C_ℓ^gg / b_g²
    4. Predicts 2-halo bispectrum using assumed b_I and b_g
    
    No CAMB/CLASS required!
    """
    
    # Set assumed biases based on typical values
    b_g_assumed = 1.5   # Low-z galaxies typically have b ~ 1-2
    b_I_assumed = 2.0   # CIB sources at z~1 typically have b ~ 2-3
    
    # Set effective redshifts based on your samples
    z_eff_g = 0.5   # Median redshift of galaxy sample (e.g., unWISE)
    z_eff_I = 0.8   # Median redshift of CIB sources (from SED models)
    
    # Run analysis
    results_dict = run_flatskyqe_ciber(
        tailstr='test_bias_pred_simple',
        inst0=1,
        inst1=2,
        ifield_list=[4, 5, 6, 7, 8],
        catname='unWISE',
        addstr='unWISE_neo8',
        lMin=1e4,
        lMax=1e5,
        mag_lim=17.5,
        single_band=True,
        calc_ciber_cross=False,
        apply_FW=False,
        cut_lxly=False,
        mode='qe_kappa_norm',
        compute_bis=True,
        skew_filter_mode='wiener',
        # Simple mode parameters:
        b_g_assumed=b_g_assumed,   # Assumed galaxy bias
        b_I_assumed=b_I_assumed,   # Assumed CIB bias  
        z_eff_g=z_eff_g,
        z_eff_I=z_eff_I
    )
    
    return results_dict


# Advanced mode: Use provided P_matter from CAMB/CLASS (if you have it)
def load_matter_power_spectrum():
    """
    Load matter power spectrum from CAMB or CLASS (OPTIONAL - advanced users only).
    
    Returns a function P(k, z) where:
    - k is in units of h/Mpc
    - z is redshift
    - P(k,z) is in units of (Mpc/h)³
    """
    # Example: Load from pre-computed file
    # data = np.load('matter_power_spectrum.npz')
    # k_array = data['k']  # h/Mpc
    # z_array = data['z']
    # P_kz = data['P_kz']  # (Mpc/h)³, shape (len(k), len(z))
    
    # Or use CAMB directly:
    # from camb import get_matter_power_interpolator
    # PK = get_matter_power_interpolator(...)
    # return lambda k, z: PK.P(z, k)
    
    # For now, use a simple power-law approximation
    def P_matter_func(k, z):
        """
        Simple power-law approximation for testing.
        Replace with actual CAMB/CLASS output!
        """
        k = np.atleast_1d(k)
        P_0 = 2e4 * (k / 0.1)**(-3.0)
        growth = 1.0 / (1 + z)
        return P_0 * growth**2
    
    return P_matter_func


def run_with_camb_power_spectrum():
    """
    Run CIBER lensing analysis with provided P_matter (ADVANCED - only if you have CAMB/CLASS).
    
    This mode extracts biases from the data and uses your provided P_matter.
    """
    
    # Load matter power spectrum
    P_matter_func = load_matter_power_spectrum()
    
    # Set effective redshifts
    z_eff_g = 0.5
    z_eff_I = 0.8
    
    # Run analysis
    results_dict = run_flatskyqe_ciber(
        tailstr='test_bias_pred_advanced',
        inst0=1,
        inst1=2,
        ifield_list=[4, 5, 6, 7, 8],
        catname='unWISE',
        addstr='unWISE_neo8',
        lMin=1e4,
        lMax=1e5,
        mag_lim=17.5,
        single_band=True,
        calc_ciber_cross=False,
        apply_FW=False,
        cut_lxly=False,
        mode='qe_kappa_norm',
        compute_bis=True,
        skew_filter_mode='wiener',
        # Advanced mode parameters:
        P_matter_func=P_matter_func,
        z_eff_g=z_eff_g,
        z_eff_I=z_eff_I
    )
    
    return results_dict


def analyze_bias_comparison(results_dict):
    """
    Analyze and plot comparison between measured and predicted bias.
    """
    import matplotlib.pyplot as plt
    
    # Extract results from first field as example
    lC = results_dict['lC']
    
    # Get field-averaged cross-spectra and bias
    clkg_avg = results_dict['all_fieldav_clkg'][0]  # First path (TM1)
    clkg_err = results_dict['all_fieldav_clkgerr'][0]
    
    # Get measured bispectrum bias (field-averaged)
    if results_dict['all_dclkg_iig'] is not None:
        bias_measured = np.mean(results_dict['all_dclkg_iig'], axis=0)
        bias_err = np.std(results_dict['all_dclkg_iig'], axis=0) / np.sqrt(len(results_dict['ifield_list']))
    else:
        print("No bispectrum results available")
        return
    
    # Get 2-halo prediction (if available)
    # Note: This would need to be stored in results_dict
    # For now, we'd need to re-run the prediction function
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    
    # Top panel: Cross-spectrum with bias
    ax1.errorbar(lC, clkg_avg, yerr=clkg_err, fmt='o', label='Measured C_L^κg')
    ax1.errorbar(lC, bias_measured, yerr=bias_err, fmt='s', label='Bispectrum bias')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('L')
    ax1.set_ylabel('C_L')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_title('Cross-spectrum and Primary Bispectrum Bias')
    
    # Bottom panel: Ratio
    ratio = clkg_avg / bias_measured
    ratio_err = ratio * np.sqrt((clkg_err/clkg_avg)**2 + (bias_err/bias_measured)**2)
    
    ax2.errorbar(lC, ratio, yerr=ratio_err, fmt='o')
    ax2.axhline(1, color='k', linestyle='--', label='Bias = Signal')
    ax2.axhline(10, color='r', linestyle='--', alpha=0.5, label='Bias = 10× Signal')
    ax2.set_xscale('log')
    ax2.set_xlabel('L')
    ax2.set_ylabel('Signal / Bias')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_title('Signal-to-Bias Ratio')
    
    plt.tight_layout()
    plt.savefig('bias_comparison.pdf')
    print("Saved bias_comparison.pdf")
    
    return fig


if __name__ == '__main__':
    # Example usage
    print("="*60)
    print("SIMPLE MODE (RECOMMENDED): Assume biases, extract P_matter from data")
    print("="*60)
    print("\nTo use in your analysis:")
    print("1. Choose reasonable bias values (b_g ~ 1.5, b_I ~ 2.0)")
    print("2. Pass b_g_assumed and b_I_assumed to run_flatskyqe_ciber")
    print("3. The pipeline extracts P_matter from your C_ℓ^gg automatically")
    print("4. No CAMB/CLASS needed!")
    print("\nExample:")
    print("  results = run_flatskyqe_ciber(..., b_g_assumed=1.5, b_I_assumed=2.0)")
    
    # Uncomment to run simple mode:
    # results = run_with_assumed_biases()
    
    print("\n" + "="*60)
    print("ADVANCED MODE (OPTIONAL): Use CAMB/CLASS P_matter")
    print("="*60)
    print("\nOnly use this if you have specific cosmology requirements.")
    print("Example:")
    print("  P_matter_func = load_from_camb()")
    print("  results = run_flatskyqe_ciber(..., P_matter_func=P_matter_func)")
    
    # Uncomment to run advanced mode:
    # results = run_with_camb_power_spectrum()

