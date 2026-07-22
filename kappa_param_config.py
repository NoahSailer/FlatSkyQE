import numpy as np

def build_param_dict(nX, nY, sizeX, sizeY, lMin, lMax, nBins, ciber_inst, 
                     mean_cl_sky=None, Apix=1.15e-9):
    """
    Build standardized parameter dictionary for CIBER lensing analysis.
    
    Parameters:
    -----------
    nX, nY : int
        Map dimensions in pixels
    sizeX, sizeY : float
        Map size in degrees
    lMin, lMax : float
        Multipole range for analysis
    nBins : int
        Number of multipole bins
    ciber_inst : int
        CIBER instrument (1 or 2)
    mean_cl_sky : float, optional
        Mean CIB power spectrum level (shot noise)
    Apix : float
        Pixel area in steradians
        
    Returns:
    --------
    param_dict : dict
    """
    pixel_size_arcsec = 3600. * (sizeX / nX)
    print('pixel size arcsec is ', pixel_size_arcsec)
    l_nyquist = np.pi / (pixel_size_arcsec / 206265.0)
    
    param_dict = {
        'nX': nX,
        'nY': nY,
        'sizeX': sizeX,
        'sizeY': sizeY,
        'lMin': lMin,
        'lMax': lMax,
        'nBins': nBins,
        'ciber_inst': ciber_inst,
        'pixel_size_arcsec': pixel_size_arcsec,
        'l_nyquist': l_nyquist,
        'Apix': Apix
    }
    
    if mean_cl_sky is not None:
        param_dict['c_i_shot'] = mean_cl_sky
        
    return param_dict

def build_config_dict(apply_mask=True, cut_lxly=False, mode='qe_kappa_norm',
                     calc_ciber_cross=False, single_band=True, apply_FW=False,
                     compute_bis=False, pixel_fn_correct=False, add_noise=True,
                     grab_cib_sim=False, skew_filter_mode='bandpass',
                     P_matter_func=None, z_eff_g=0.5, z_eff_I=0.8,
                     b_g_assumed=None, b_I_assumed=2.0):
    """
    Build standardized configuration dictionary for CIBER lensing analysis.
    
    Parameters:
    -----------
    apply_mask : bool
        Whether to apply masks to maps
    cut_lxly : bool
        Whether to cut lx=ly modes in QE
    mode : str
        QE mode ('qe_kappa_norm' or 'qe_kappa_shearonly')
    calc_ciber_cross : bool
        Whether to compute cross-band products
    single_band : bool
        Whether analyzing single band only
    apply_FW : bool
        Whether to apply Fourier weights
    compute_bis : bool
        Whether to compute bispectrum
    pixel_fn_correct : bool
        Whether to apply pixel window correction
    add_noise : bool
        Whether data includes noise (always True for real data)
    grab_cib_sim : bool
        Whether loading from simulation (False for real data)
    skew_filter_mode : str
        Bispectrum filtering mode ('wiener' or 'bandpass'). Default 'bandpass' matches mocks.
    P_matter_func : callable, optional
        Matter power spectrum function P(k, z) for 2-halo bias prediction (advanced mode).
        Should accept k in h/Mpc and z as arguments.
        If None and b_g_assumed is None, 2-halo prediction will be skipped.
    z_eff_g : float
        Effective redshift of galaxy tracer sample (default 0.5)
    z_eff_I : float
        Effective redshift of CIB source sample (default 0.8)
    b_g_assumed : float, optional
        Assumed linear bias of galaxy sample (e.g., 1.5 for low-z galaxies).
        If provided, P_matter will be extracted from C_ℓ^gg (SIMPLE MODE - recommended).
        If None, must provide P_matter_func to use advanced mode.
    b_I_assumed : float
        Assumed linear bias of CIB sources (default 2.0).
        Only used if b_g_assumed is provided.
        
    Returns:
    --------
    config_dict : dict
    """
    config_dict = {
        'apply_mask': apply_mask,
        'cut_lxly': cut_lxly,
        'mode': mode,
        'calc_ciber_cross': calc_ciber_cross,
        'single_band': single_band,
        'apply_FW': apply_FW,
        'compute_bis': compute_bis,
        'pixel_fn_correct': pixel_fn_correct,
        'add_noise': add_noise,
        'grab_cib_sim': grab_cib_sim,
        'skew_filter_mode': skew_filter_mode,
        'P_matter_func': P_matter_func,
        'z_eff_g': z_eff_g,
        'z_eff_I': z_eff_I,
        'b_g_assumed': b_g_assumed,
        'b_I_assumed': b_I_assumed
    }
    
    return config_dict

def build_map_dict(ld, galdens, galdensFourier):
    """
    Build map dictionary for CIBER analysis.
    
    Parameters:
    -----------
    ld : lens_data
        Lens data object with maps and Fourier transforms
    galdens : ndarray
        Processed galaxy density map
    galdensFourier : ndarray
        Fourier transform of galaxy density map
        
    Returns:
    --------
    map_dict : dict
        Dictionary with all necessary maps
    """
    map_dict = {
        'dataFourier': ld.dataFourier,
        'mask': ld.mask,
        'galdens': galdens,
        'galdensFourier': galdensFourier
    }
    
    # Add second band if available
    if ld.dataFourier2 is not None:
        map_dict['dataFourier2'] = ld.dataFourier2
        
    return map_dict