import numpy as np
import os
import config

import universe
import pn_2d
import flat_map
# import weight

from flat_map import *
from pn_2d import *
from universe import *

from scipy.integrate import quad

parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(parent_dir)

from ciber.mocks.cib_mocks import *


def radial_bin(ell2d, W2d, ell_bins):
	# Create histogram to bin ell2d and W2d 
	# Returning 1D W1d values as azimuthally averaged window 
	W1d, bin_edges = np.histogram(ell2d, bins=ell_bins, weights=W2d)
	counts, _ = np.histogram(ell2d, bins=ell_bins)  # Count number of entries for normalization
	W1d[counts > 0] /= counts[counts > 0]  # Normalize by counts
	return W1d

def calculate_analytic_bias(n_cib_per_pixel, n_g_per_pixel, 
							fluxes_cib, fluxes_g, pix_area=1.0, 
							m_max_cutsrc=None, inst=1):
	"""
	Calculates the analytic self-lensing bias ΔC_L^{κg} in the Poisson limit
	and trispectrum contributions to reconstruction noise.

	This implements the formula: ΔĈ = (A_pix * n_g * <s^2>_g) / (2 * n_cib * <s^2>_cib)
	
	Trispectrum noise: N_L^kappa = (1/4) * (<s^4> / <s^2>^2) * (1/nbar)

	Args:
		n_cib_per_pixel (float): The number density of all CIB sources (per pixel).
		n_g_per_pixel (float): The number density of the tracer galaxies (per pixel).
		fluxes_cib (np.ndarray): Array of fluxes for the entire CIB population.
		fluxes_g (np.ndarray): Array of fluxes for the subset of CIB sources 
							   that are also the galaxy tracers.
		pix_area (float): The area of a single pixel (A_pix). Units are arbitrary 
						  but must be consistent. Defaults to 1.0.

	Returns:
		tuple: (delta_c, c_i_shot, c_i2_g_shot, galshot, trispec_noise_cib, trispec_noise_g)
			delta_c: The calculated scale-independent bias term, ΔĈ_L^{κg}
			c_i_shot: CIB shot noise power
			c_i2_g_shot: Cross shot noise term
			galshot: Galaxy shot noise
			trispec_noise_cib: N_L^kappa from CIB trispectrum
			trispec_noise_g: N_L^kappa from galaxy trispectrum
	"""
	if len(fluxes_cib) == 0:
		return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

	if m_max_cutsrc is not None:
		print('Culling sources with mag <', m_max_cutsrc, 'for analytic bias calculation')
		cmock = ciber_mock()
		flux_min_cutsrc = np.array(cmock.mag_2_nu_Inu(m_max_cutsrc, band=inst-1).value)

		mask_cib = (fluxes_cib < flux_min_cutsrc)
		mask_g = (fluxes_g < flux_min_cutsrc)

		frac_retain_cib = np.sum(mask_cib)/len(mask_cib)
		frac_retain_g = np.sum(mask_g)/len(mask_g)

		print('FRAC RETAIN CIB SOURCES = ', frac_retain_cib)
		fluxes_cib = fluxes_cib[mask_cib]
		fluxes_g = fluxes_g[mask_g]

		n_cib_per_pixel *= frac_retain_cib
		# n_g_per_pixel *= frac_retain_g

		
	# Calculate the second moment of the flux for all CIB sources
	mean_s2_cib = np.mean(fluxes_cib**2)
	
	# Calculate the second moment of the flux for the galaxy tracer subset
	mean_s2_g = np.mean(fluxes_g**2)
	
	# Fourth moments for trispectrum
	mean_s4_cib = np.mean(fluxes_cib**4)
	mean_s4_g = np.mean(fluxes_g**4)
	
	# Number densities (per unit area, not per pixel)
	ncib = n_cib_per_pixel / pix_area
	ng = n_g_per_pixel / pix_area
	
	# C^I = n_cib * <s^2>_cib
	# Note: n_cib here is a density, so C^I is the power per unit area (or per pixel area)
	c_i_shot = n_cib_per_pixel * mean_s2_cib * pix_area
	
	c_i2_g_shot = mean_s2_g * pix_area
	
	# Galaxy shot noise (per unit area)
	galshot = (pix_area / n_g_per_pixel)
	
	# Trispectrum contributions to N_L^kappa
	# N_L^kappa = (1/4) * (<s^4> / <s^2>^2) * (1/nbar)
	trispec_noise_cib = 0.25 * (mean_s4_cib / mean_s2_cib**2) / ncib
	trispec_noise_g = 0.25 * (mean_s4_g / mean_s2_g**2) / ng
	
	# Avoid division by zero if there are no CIB sources
	if c_i_shot == 0:
		return 0.0, 0.0, 0.0, galshot, 0.0, trispec_noise_g
		
	# Final bias
	delta_c = (pix_area * c_i2_g_shot) / (2 * c_i_shot)
	
	return delta_c, c_i_shot, c_i2_g_shot, galshot, trispec_noise_cib, trispec_noise_g

def effective_beam_skew_I2g_correct(L_vals, lMin, lMax, B_ell_fn=None, W_ell_fn=None, n_ell_grid=200, n_theta_grid=32):
	"""
	Effective beam suppression for C_L^{I^2 x g} in Poisson limit, with proper 2D vector geometry.

	For each output multipole L (scalar magnitude), integrate over all 2D ℓ vectors such that:
	1. |ℓ| ∈ [lMin, lMax]
	2. |L - ℓ| ∈ [lMin, lMax]  (where L - ℓ is vector subtraction)

	The partner magnitude |L - ℓ| = √(L² + ℓ² - 2Lℓ cos θ) depends on the angle θ between L and ℓ.

	Parameters
	----------
	L_vals : array_like
		Array of output multipoles (one per bandpower).
	lMin, lMax : float
		Bandpass annulus edges [lMin, lMax].
	B_ell_fn : callable, optional
		Function returning beam factor B(ell). If None, returns all 1.0s (no beam).
	W_ell_fn : callable, optional
		Optional weight function W(ell) to apply to integrand.
	n_ell_grid : int
		Number of integration points for ell magnitude in [lMin, lMax].
	n_theta_grid : int
		Number of integration points for angle θ in the valid range.

	Returns
	-------
	Beff_skew : ndarray
		Effective beam × mode-overlap per output L (length = len(L_vals)).
	"""

	# Create radial grid for ℓ magnitude in the bandpass
	ell_grid = np.logspace(np.log10(lMin), np.log10(lMax), n_ell_grid)
	d_ell = ell_grid[1:] - ell_grid[:-1]
	ell_centers = 0.5 * (ell_grid[:-1] + ell_grid[1:])

	# Beam function for small-scale leg ell
	if B_ell_fn is not None:
		B_ell = B_ell_fn(ell_centers)
	else:
		B_ell = np.ones_like(ell_centers)

	# Weight function
	if W_ell_fn is not None:
		W_ell = W_ell_fn(ell_centers)
	else:
		W_ell = np.ones_like(ell_centers)

	# Total 2D modes in the annulus: ∫∫ ℓ dℓ dθ from ℓ=lMin to lMax, θ=0 to 2π
	total_modes_2d = np.pi * (lMax**2 - lMin**2)

	n_L = len(L_vals)
	Beff_skew = np.zeros(n_L)

	for i, L in enumerate(L_vals):
		if L == 0:
			Beff_skew[i] = 0.0
			continue

		valid_modes_integral = 0.0
		beam_weighted_integral = 0.0

		for j, ell in enumerate(ell_centers):
			# For this radial ℓ, find the angular range where |L - ℓ| ∈ [lMin, lMax]
			# |L - ℓ|² = L² + ℓ² - 2Lℓ cos(θ)

			# Constraints: lMin² ≤ L² + ℓ² - 2Lℓ cos(θ) ≤ lMax²
			# Rearrange: (L² + ℓ² - lMax²)/(2Lℓ) ≤ cos(θ) ≤ (L² + ℓ² - lMin²)/(2Lℓ)

			cos_lower = (L**2 + ell**2 - lMax**2) / (2.0 * L * ell)
			cos_upper = (L**2 + ell**2 - lMin**2) / (2.0 * L * ell)

			# Clamp to [-1, 1] (physical range for cosine)
			cos_lower = np.clip(cos_lower, -1.0, 1.0)
			cos_upper = np.clip(cos_upper, -1.0, 1.0)

			if cos_lower > cos_upper:
				continue  # No valid angular range

			# Valid angle range in [0, π] where cos is monotonic
			theta_lower = np.arccos(cos_upper)  # where |L-ℓ| is maximum
			theta_upper = np.arccos(cos_lower)  # where |L-ℓ| is minimum

			# Total valid angle measure (accounting for both sides of unit circle)
			delta_theta = 2.0 * (theta_upper - theta_lower)

			if delta_theta <= 0:
				continue

			# 2D mode weight contribution: ℓ dℓ × delta_θ
			mode_weight = ell * delta_theta * d_ell[j]
			valid_modes_integral += mode_weight

			# For beam weighting, integrate over the valid θ range
			theta_samples = np.linspace(theta_lower, theta_upper, n_theta_grid)

			B_partner_avg = 0.0
			for theta in theta_samples:
				partner_mag = np.sqrt(L**2 + ell**2 - 2.0*L*ell*np.cos(theta))
				if B_ell_fn is not None:
					B_partner_avg += B_ell_fn(partner_mag)
				else:
					B_partner_avg += 1.0

			B_partner_avg /= len(theta_samples)

			# Beam weighted contribution
			beam_product = B_ell[j] * B_partner_avg
			beam_weighted_integral += W_ell[j] * beam_product * mode_weight

		if valid_modes_integral > 0:
			mode_overlap_frac = valid_modes_integral / total_modes_2d
			beam_weight_avg = beam_weighted_integral / valid_modes_integral
			Beff_skew[i] = mode_overlap_frac * beam_weight_avg
		else:
			Beff_skew[i] = 0.0

	return Beff_skew


def effective_beam_skew_I2g(L_vals, lEdges, ell_grid, flat_sky=True, fwhm_I=None, B_ell_fn=None, W_ell_fn=None):
	"""
	Effective beam suppression for C_L^{I^2 x g} in Poisson limit,
	with beam on I, no beam on g.

	Parameters
	----------
	L_vals : array_like
		Array of output multipoles (one per bandpower).
	lEdges : ndarray
		Bandpower edges for the skew spectrum bins.
	fwhm_I : float
		Gaussian beam FWHM of intensity map (arcsec).
	ell_grid : ndarray
		1D array of ell magnitudes for Fourier modes (flat-sky), should cover smaller-scale modes contributing to skew contraction.
	flat_sky : bool
		Use flat-sky or full-sky beam definition.
	B_ell_fn : callable
		Function returning beam factor B(ell) as alternative to fwhm_I.
	W_ell_fn : callable
		Optional weight function W(ell) to apply to the integrand.

	Returns
	-------
	Beff_skew_bins : ndarray
		Effective beam per skew band (I^2 leg has two beams).
	"""

	if fwhm_I is not None:
		B_I = gaussian_beam_window(ell_grid, fwhm_I, flat_sky=flat_sky)
	elif B_ell_fn is not None:
		B_I = B_ell_fn(ell_grid)
	else:
		# return np.ones(len(lEdges) - 1)
		B_I = np.ones_like(ell_grid)

	# Apply optional weighting
	if W_ell_fn is not None:
		W_ell_vals = W_ell_fn(ell_grid)
	else:
		W_ell_vals = np.ones_like(ell_grid)

	weights = ell_grid if flat_sky else (2*ell_grid + 1)

	Nbins = len(lEdges) - 1
	Beff_skew_bins = np.zeros(Nbins)

	for i in range(Nbins):
		L_low, L_high = lEdges[i], lEdges[i+1]
		# Mean L for bin center (used for the second leg)
		L_center = 0.5 * (L_low + L_high)

		# For Poisson, integrate suppression from two I legs with weighting:
		# integrand = W(ell) * B(ell) * B(|L-ell|)
		B_Lminusell = np.interp(np.abs(L_center - ell_grid), ell_grid, B_I)
		prod_B = W_ell_vals * B_I * B_Lminusell

		sel = (ell_grid >= L_low) & (ell_grid < L_high)
		Beff_skew_bins[i] = np.sum(weights[sel] * prod_B[sel]) / np.sum(weights[sel])

	return Beff_skew_bins


def gaussian_beam_window(ell, fwhm_arcsec, flat_sky=False):
	"""
	Gaussian beam window function B_ell for given multipoles.
	
	Parameters
	----------
	ell : array_like
		Multipole values (can be float or ndarray).
	fwhm_arcsec : float
		Gaussian beam FWHM in arcseconds.
	flat_sky : bool, optional
		If True, use flat-sky formula with ell^2 instead of ell(ell+1).
	
	Returns
	-------
	B_ell : ndarray
		Beam transfer function.
	"""
	# Convert FWHM from arcsec to radians
	fwhm_rad = fwhm_arcsec * np.pi / (180. * 3600.)
	
	# Gaussian sigma of the beam (in radians)
	sigma_b = fwhm_rad / np.sqrt(8.0 * np.log(2.0))
	
	if flat_sky:
		return np.exp(-0.5 * (ell**2) * sigma_b**2)
	else:
		return np.exp(-0.5 * ell * (ell + 1.0) * sigma_b**2)

def effective_beam_bins(ell, fwhm_arcsec, lEdges, flat_sky=True):
	"""
	Compute effective Gaussian beam factor per ℓ-bin given bin edges, 
	by averaging over ℓ in each bin with proper mode-count weighting.
	
	Parameters
	----------
	ell : array_like
		ℓ values for your modes (can be Fourier grid radii or 1D ℓ array).
	fwhm_arcsec : float
		Gaussian beam FWHM in arcseconds.
	lEdges : ndarray
		Array of bin edges (length Nbins+1).
	flat_sky : bool
		If True, use flat-sky beam formula and weight ∝ ℓ;
		If False, use full-sky formula and weight ∝ (2ℓ+1).
	
	Returns
	-------
	Beff_bins : ndarray
		Effective beam factor for each bin (length Nbins).
	"""
	# Compute B_ell for all modes
	B_ell = gaussian_beam_window(np.asarray(ell), fwhm_arcsec, flat_sky=flat_sky)
	
	# Select weights
	if flat_sky:
		weights = np.asarray(ell)
	else:
		weights = 2.0 * np.asarray(ell) + 1.0
	
	Nbins = len(lEdges) - 1
	Beff_bins = np.zeros(Nbins)
	
	# Loop over bins defined by lEdges
	for i in range(Nbins):
		low = lEdges[i]
		high = lEdges[i+1]
		sel = (ell >= low) & (ell < high)
		
		if np.any(sel):
			Beff_bins[i] = np.sum(weights[sel] * B_ell[sel]) / np.sum(weights[sel])
		else:
			Beff_bins[i] = np.nan  # or 1.0 if you want empty bins set to unity
	
	return Beff_bins

def calculate_cib_poisson_terms(n_cib_per_pixel, n_g_per_pixel, 
                            fluxes_cib, fluxes_g, pix_area=1.0):
    """
    Calculates the analytic self-lensing bias ΔC_L^{κg} in the Poisson limit.

    This implements the formula: ΔĈ = (A_pix * n_g * <s^2>_g) / (2 * n_cib * <s^2>_cib)

    Args:
        n_cib_per_pixel (float): The number density of all CIB sources (per pixel).
        n_g_per_pixel (float): The number density of the tracer galaxies (per pixel).
        fluxes_cib (np.ndarray): Array of fluxes for the entire CIB population.
        fluxes_g (np.ndarray): Array of fluxes for the subset of CIB sources 
                               that are also the galaxy tracers.
        pix_area (float): The area of a single pixel (A_pix). Units are arbitrary 
                          but must be consistent. Defaults to 1.0.

    Returns:
        float: The calculated scale-independent bias term, ΔĈ_L^{κg}.
    """
    if len(fluxes_cib) == 0:
        return 0.0
    
    
    ncib = n_cib_per_pixel/pix_area
    ng = n_g_per_pixel / pix_area
    
    cibshot_n = 1./ncib
    galshot = 1./ng
    

    # Calculate the second moment of the flux for all CIB sources
    mean_s2_cib = np.mean(fluxes_cib**2)
    mean_s2_g = np.mean(fluxes_g**2)

    # fourth moment for trispectrum
    mean_s4_cib = np.mean(fluxes_cib**4)
    mean_s4_g = np.mean(fluxes_g**4)


    # C^I = n_cib * <s^2>_cib
    # Note: n_cib here is a density, so C^I is the power per unit area (or per pixel area)
    c_i_shot = ncib * mean_s2_cib
    c_i2_g_shot = mean_s2_g * pix_area

    trispec_noise_cib = 0.25*(mean_s4_cib/mean_s2_cib**2)*(n_cib_per_pixel * pix_area)**(-1)
    trispec_noise_g = 0.25*(mean_s4_g/mean_s2_g**2)*galshot
    
    # Avoid division by zero if there are no CIB sources
    if c_i_shot == 0:
        return 0.0

    # Final bias
    delta_clkg = (pix_area * c_i2_g_shot) / (2 * c_i_shot)
        
    res_dict = {'delta_clkg':delta_clkg, 'galshot':galshot, 'c_i_shot':c_i_shot, 'c_i2_g_shot':c_i2_g_shot, \
               'mean_s2_cib':mean_s2_cib, 'mean_s2_g':mean_s2_g, 'mean_s4_cib':mean_s4_cib, 'mean_s2_g':mean_s2_g, \
               'trispec_noise_cib':trispec_noise_cib, 'trispec_noise_g':trispec_noise_g}
    
    return res_dict


def psf_pix_fwhm_to_sigma_rad(pixel_size_arcsec, psf_fwhm_pixels):
	
	# Constants
	arcsec_to_rad = 206265.0  # Conversion factor from arcseconds to radians
	# ln2 = np.log(2)
	
	# Convert pixel size and FWHM to radians
	pixel_size_rad = pixel_size_arcsec / arcsec_to_rad
	psf_fwhm_rad = psf_fwhm_pixels * pixel_size_rad  # FWHM in radians
	
	# Calculate the Gaussian beam standard deviation (σ_b)
	sigma_b = psf_fwhm_rad / np.sqrt(8 * np.log(2))
	
	return sigma_b

def compute_beam_correction(pixel_size_arcsec, psf_fwhm_pixels, ell_min=1e4, ell_max=8e4, B_ell=None):
	
	''' Correction to bispectrum for Gaussian beam with specified FWHM or empirical B(ell) '''
	
	k_correct_denom = ell_max**2 - ell_min**2

	if B_ell is not None:
		def B_ell_sq(ell):
			return ell*B_ell(ell)**2

		k_correct_num, _ = quad(B_ell_sq, ell_min, ell_max)
		k_correct_denom *= 0.5

	else:
		sigma_b = psf_pix_fwhm_to_sigma_rad(pixel_size_arcsec, psf_fwhm_pixels)
		k_correct_num = (np.exp(-(ell_min*sigma_b)**2) - np.exp(-(ell_max*sigma_b)**2))/sigma_b**2
	
	k_correct = k_correct_num / k_correct_denom
	
	return k_correct

def compute_beam_correction_num(B_ell, ell_min=1e4, ell_max=8e4, W_ell=None, P_ell_sq=None):
	''' 
	Correction to bispectrum for Gaussian beam with specified FWHM or empirical B(ell).
	
	Parameters:
		pixel_size_arcsec (float): The pixel size in arcseconds.
		psf_fwhm_pixels (float): PSF Full Width Half Maximum in pixels.
		ell_min (float): Minimum multipole value.
		ell_max (float): Maximum multipole value.
		B_ell (callable): A 1D function for the beam (e.g., interp1d).
		W_ell (callable): A 1D function for the weights (e.g., filtering weights).
	
	Returns:
		float: The correction factor for the primary bispectrum.
	'''

	if W_ell is None:
		W_ell = lambda ell: 1.

		k_correct_denom = 0.5*(ell_max**2 - ell_min**2)  # This is the simple difference of squares
	else:
		def integrand_denom(ell):
			return ell * W_ell(ell)
		k_correct_denom, _ = quad(integrand_denom, ell_min, ell_max)

	if P_ell_sq is None:
		P_ell_sq = lambda ell: 1.

	def integrand_num(ell):
		return ell * W_ell(ell) * P_ell_sq(ell) * B_ell(ell)**2

	# # numerator

	k_correct_num, _ = quad(integrand_num, ell_min, ell_max)

	k_correct = k_correct_num / k_correct_denom

	return k_correct

