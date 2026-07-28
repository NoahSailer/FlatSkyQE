import numpy as np
import os
import config

import universe
import pn_2d
import flat_map

from flat_map import *
from pn_2d import *
from universe import *

from prep_ciber_dat_lens import *
from kappa_plotting_fns import *
from forecast_cib_lens import *
from lens_data_file_utils import *

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
# Add the parent directory to sys.path
sys.path.append(parent_dir)
from ciber.core.powerspec_pipeline import *
from ciber.core.ps_pipeline_go import *
from ciber.io.ciber_data_utils import *
from ciber.plotting.plotting_fns import plot_map
from map_clus_utils import *
from bias_modl import *

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
        print('Setting c_i_shot to ', mean_cl_sky)
        param_dict['c_i_shot'] = mean_cl_sky
        
    return param_dict


def build_config_dict(apply_mask=True, cut_lxly=False, mode='qe_kappa_norm',
                     calc_ciber_cross=False, single_band=True, apply_FW=False,
                     compute_bis=False, pixel_fn_correct=False, add_noise=True,
                     grab_cib_sim=False):
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
        'grab_cib_sim': grab_cib_sim
    }
    
    return config_dict


def build_cl_fns_and_corr_facs(clf, param_dict, config_dict, ciber_unlensed_auto, 
                                ciber_obs_auto, mask):
    """
    Build filter functions and correction factors for CIBER real data.
    
    This is a wrapper that handles real data where we have pre-computed
    unlensed and observed auto-spectra, as opposed to simulations where
    we build them from scratch.
    
    Parameters:
    -----------
    clf : ciber_lens_forecast
        Contains beam function
    param_dict : dict
        Parameter dictionary from build_param_dict
    config_dict : dict
        Configuration dictionary from build_config_dict
    ciber_unlensed_auto : callable
        Unlensed CIB auto-spectrum function
    ciber_obs_auto : callable
        Observed (pseudo-Cl) auto-spectrum function
    mask : ndarray
        Mask array for computing unmask_frac
        
    Returns:
    --------
    cl_fns : dict
        Dictionary with filter functions
    corr_facs : dict
        Dictionary with correction factors
    """
    
    # Build filter function
    def W_ell(ell):
        return ciber_unlensed_auto(ell) / ciber_obs_auto(ell)
    
    # Compute correction factors
    kcorr = compute_beam_correction_num(clf.bl, ell_min=param_dict['lMin'], 
                                       ell_max=param_dict['lMax'], W_ell=W_ell)
    vbeam = compute_beam_correction_num(clf.bl, ell_min=param_dict['lMin'], 
                                       ell_max=param_dict['lMax'])
    modefrac = (param_dict['lMax']**2 - param_dict['lMin']**2) / param_dict['l_nyquist']**2
    
    # Apply mode fraction to vbeam
    vbeam_corrected = vbeam * modefrac
    
    # Get unmask fraction
    unmask_frac = np.mean(mask)
    
    print('kcorr is ', kcorr)
    print('vbeam is ', vbeam)
    print('vbeam * modefrac is ', vbeam_corrected)
    print('mode frac is ', modefrac)
    print('unmask frac is ', unmask_frac)
    
    cl_fns = {
        'cib_unlensed_auto': ciber_unlensed_auto,
        'obs_auto': ciber_obs_auto,
        'W_ell': W_ell,
        'B_ell': clf.bl
    }
    
    corr_facs = {
        'kcorr': kcorr,
        'vbeam': vbeam_corrected,
        'unmask_frac': unmask_frac,
        'modefrac': modefrac
    }
    
    return cl_fns, corr_facs


def compute_ciber_kappa_products(baseMap, map_dict, cl_fns, param_dict, 
                                  config_dict, corr_facs, paths):
    """
    Unified function to compute kappa estimates and power spectra for CIBER data.
    Analogous to compute_lensing_ps_quantities_v2 but for real data pipeline.
    
    Parameters:
    -----------
    baseMap : FlatMap
        Flat map object for transformations
    map_dict : dict
        Dictionary with 'dataFourier', 'dataFourier2' (optional), 'galdensFourier', 'mask'
    cl_fns : dict
        Dictionary with 'cib_unlensed_auto', 'obs_auto', 'W_ell', 'B_ell'
    param_dict : dict
        Dictionary with 'lMin', 'lMax', 'Apix', 'c_i_shot', etc.
    config_dict : dict
        Dictionary with 'apply_mask', 'cut_lxly', 'mode', 'compute_bis', etc.
    corr_facs : dict
        Dictionary with 'kcorr', 'vbeam', 'unmask_frac'
    paths : list
        List of output paths for kappa estimates
    
    Returns:
    --------
    results : dict
        Dictionary with all computed power spectra:
        - 'lC': multipole bins
        - 'all_cl': kappa auto-spectra (one per path)
        - 'all_clerr': errors on kappa auto-spectra
        - 'clxs': kappa-galaxy cross-spectra (one per path)
        - 'clxerrs': errors on cross-spectra
        - 'cl_bis': bispectrum (if compute_bis=True)
        - 'sCl_bis': bispectrum errors
        - 'clkg_bias': bias estimate from bispectrum
    """
    from map_clus_utils import proc_clkg, proc_skewspec, compute_skew_cl_I2G_simp
    
    # Run kappa estimation for each path
    n_paths = len(paths)
    for k in range(n_paths):
        # Select appropriate data Fourier
        if k == 0:
            dataFourier_use = map_dict['dataFourier']
            dataFourier2_use = None
        elif k == 1 and 'dataFourier2' in map_dict:
            dataFourier_use = map_dict['dataFourier2']
            dataFourier2_use = None
        elif k == 2 and 'dataFourier2' in map_dict:
            # Cross-band case
            dataFourier_use = map_dict['dataFourier']
            dataFourier2_use = map_dict['dataFourier2']
        else:
            dataFourier_use = map_dict['dataFourier']
            dataFourier2_use = None
            
        run_kappa_est(baseMap, cl_fns['cib_unlensed_auto'], cl_fns['obs_auto'],
                     param_dict, dataFourier=dataFourier_use, 
                     dataFourier2=dataFourier2_use, test=False,
                     path=paths[k], cut_lxly=config_dict['cut_lxly'], 
                     mode=config_dict['mode'], fB_ell=cl_fns.get('B_ell'))
    
    # Compute kappa auto power spectra
    all_cl, all_clerr = [], []
    for path_k in paths:
        kFourier = baseMap.loadDataFourier(path_k)
        lC, cl, sCl = baseMap.powerSpectrum(kFourier, theory=[], plot=False)
        all_cl.append(cl)
        all_clerr.append(sCl)
    
    # Compute kappa-galaxy cross-spectra
    clxs, clxerrs = [], []
    for path_k in paths:
        kFourier = baseMap.loadDataFourier(path_k)
        lC, clx, clxerr = baseMap.crossPowerSpectrum(kFourier, 
                                                      map_dict['galdensFourier'],
                                                      plot=False)
        clxs.append(clx)
        clxerrs.append(clxerr)
    
    # Apply corrections to cross-spectra (QE is beam-independent)
    for i in range(len(clxs)):
        clxs[i], clxerrs[i] = proc_clkg(lC, clxs[i], clxerrs[i], 
                                        B_ell=None,  # No beam correction for QE
                                        kcorr=corr_facs['kcorr'],
                                        unmask_frac=corr_facs['unmask_frac'])
    
    # Compute bispectrum if requested
    cl_bis, sCl_bis, clkg_bias = None, None, None
    if config_dict['compute_bis']:
        # Low-pass filter CIB map
        def bandpass(l):
            if (l > param_dict['lMax']) or (l < param_dict['lMin']):
                return 0.0
            else:
                return 1.0
        
        iVarCIBFourier = baseMap.filterFourierIsotropic(bandpass, 
                                                        dataFourier=map_dict['dataFourier'],
                                                        test=False)
        cib_lowpass = baseMap.inverseFourier(iVarCIBFourier)
        
        lC, cl_bis, sCl_bis = compute_skew_cl_I2G_simp(baseMap, cib_lowpass,
                                                        map_dict['galdensFourier'],
                                                        lMin=param_dict['lMin'],
                                                        lMax=param_dict['lMax'])
        
        # Apply beam correction to bispectrum
        if cl_fns['B_ell'] is not None:
            cl_bis /= cl_fns['B_ell'](lC)**2
            sCl_bis /= cl_fns['B_ell'](lC)**2
        
        # Apply corrections (vbeam includes modefrac, unmask_frac is separate)
        cl_bis, sCl_bis = proc_skewspec(lC, cl_bis, sCl_bis,
                                        B_ell=None,  # Already applied above
                                        vbeam=corr_facs['vbeam'],
                                        unmask_frac=corr_facs['unmask_frac'])
        
        # Compute bias estimate
        clkg_bias = cl_bis * param_dict['Apix'] / (2 * param_dict['c_i_shot'])
    
    results = {
        'lC': lC,
        'all_cl': all_cl,
        'all_clerr': all_clerr,
        'clxs': clxs,
        'clxerrs': clxerrs,
        'cl_bis': cl_bis,
        'sCl_bis': sCl_bis,
        'clkg_bias': clkg_bias
    }
    
    return results


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


# def proc_input_map(input_map, mask, galdens=False, apply_mask=True):
# def compute_map_ps(baseMap, input_map, mask_input, apply_mask=True):
# def calc_gal_fourier(baseMap, gal_densities, ifield, mask, plot=False):
# def kappa_gal_cross(baseMap, paths, mask, galdensFourier, plot=False):
# def compute_skew_cl_I2G_simp(baseMap, ciber_map, galdensFourier, lMax=5e4, lMin=1e4):
# def compute_skew_cl_I2G(baseMap, ciber_map, galdensFourier, fC0, fCtot, lMax=5e4, lMin=1e4):
# def calc_4pt_stats(baseMap, ld):


def calc_filters_and_corrections(clf, params, cfg):

    if cfg['pixel_fn_correct']:
        dtheta = params['pixel_size_arcsec'] / 206265.
        def P_ell_sq(ell):
            return np.exp(-((ell*dtheta)**2)/6)
    else:
        P_ell_sq = lambda ell: 1.

    def cib_unlensed_auto(ell):
        return params['c_i_shot']*np.ones_like(ell)

    if cfg['add_noise']:
        if cfg['grab_cib_sim']: # use empirical beam
            def obs_auto(ell):
                return cib_unlensed_auto(ell) + np.mean(params['clnoise'])/clf.bl(ell)**2
        elif params.get('psf_pix_fwhm') is not None:  # Gaussian beam
            def obs_auto(ell):
                return cib_unlensed_auto(ell) + np.mean(params['clnoise'])/gaussian_beam_window(ell, params['psf_pix_fwhm']*params['pixel_size_arcsec'])
        else:  # No beam (delta functions)
            def obs_auto(ell):
                return cib_unlensed_auto(ell) + np.mean(params['clnoise'])

    else:
        def obs_auto(ell):
            return cib_unlensed_auto(ell)*np.ones_like(ell)


    def cib_unlensed_auto_clus(ell):
        return cib_unlensed_auto(ell) - params['c_i_shot']*np.ones_like(ell)

    def W_ell(ell):
        return cib_unlensed_auto(ell)/obs_auto(ell)

    kcorr = compute_beam_correction_num(clf.bl, ell_min=params['lMin'], ell_max=params['lMax'], W_ell=W_ell)
    vbeam = compute_beam_correction_num(clf.bl, ell_min=params['lMin'], ell_max=params['lMax'])
    # Bandpass fraction from shear.tex eq. 93: f_band = π(ℓ_max²-ℓ_min²)/(4ℓ_Nyq²)
    modefrac = (np.pi / 4.0) * (params['lMax']**2 - params['lMin']**2) / params['l_nyquist']**2

    print('kcorr is ', kcorr)
    print('vbeam is ', vbeam)
    print('mode frac is ', modefrac)

    fns = {'obs_auto':obs_auto, 'cib_unlensed_auto':cib_unlensed_auto, 'cib_unlensed_auto_clus':cib_unlensed_auto_clus, 'W_ell':W_ell, 'P_ell_sq':P_ell_sq}

    facs = {'kcorr':kcorr, 'vbeam':vbeam, 'modefrac':modefrac}

    return fns, facs
    

def run_kappa_est(baseMap, ciber_unlensed_auto, ciber_obs_auto, params, dataFourier, dataFourier2=None, test=False, path=None, \
                 mode='qe_kappa_norm', cut_lxly=False, fB_ell=None, sigma=0., u=1.0, fUln=None):
    
    if mode=='qe_kappa_norm':
        resultFourier, norm_Fourier = baseMap.computeQuadEstKappaNorm(ciber_unlensed_auto, ciber_obs_auto,
                                                                     lMin=params['lMin'], lMax=params['lMax'],
                                                                     dataFourier=dataFourier, dataFourier2=dataFourier2,
                                                                     test=test, path=path, cut_lxly=cut_lxly, fB_ell=fB_ell)
        
    elif mode=='qe_kappa_shearonly':
        resultFourier, norm_Fourier = baseMap.computeQuadEstKappaShearNormCorr(ciber_unlensed_auto, ciber_obs_auto, 
                                                                lMin=params['lMin'], lMax=params['lMax'],
                                                                dataFourier=dataFourier, dataFourier2=dataFourier2,
                                                                test=test, path=path, corr=True, cut_lxly=cut_lxly, fB_ell=fB_ell)

    elif mode == 'qe_kappa_ps_hardened':
        resultFourier = baseMap.computeQuadEstKappaPointSourceHardenedNorm(
            ciber_unlensed_auto, ciber_obs_auto,
            lMin=params['lMin'], lMax=params['lMax'],
            dataFourier=dataFourier, dataFourier2=dataFourier2,
            test=test, path=path, cache=None, sigma=sigma, u=u)
        # hardened estimator has no single scalar normalization
        norm_Fourier = None

    elif mode == 'qe_kappa_ln_hardened':

        ell0 = 3000.
        u0 = fUln(ell0)
        fUln_norm = lambda l: fUln(l)/u0

        resultFourier = baseMap.computeQuadEstKappaLNHardenedNorm(
            ciber_unlensed_auto, ciber_obs_auto,
            lMin=params['lMin'], lMax=params['lMax'],
            dataFourier=dataFourier, dataFourier2=dataFourier2,
            test=test, path=path, cache=None, fUln=fUln_norm)
        # hardened estimator has no single scalar normalization
        norm_Fourier = None
        
    return resultFourier, norm_Fourier

    
def run_flatskyqe_ciber(tailstr, inst0=1, inst1=2, ifield_list=[4, 6, 7, 8], catname='WISE', addstr='unWISE_neo8', \
                       lMin=1000, lMax=2e5, calc_ciber_cross=True, save=False, colors=['b', 'r', 'magenta'], \
                       apply_FW=False, mode='qe_kappa_norm', test=False, calc_kappa=True, cut_lxly=False, \
                       single_band=False, bbox_to_anchor=[0.5, 1.2], plot=False, ylim=[1e-11, 3e-7], textypos=1e-7, \
                       compute_collapsed_tris=False, compute_bis=False, mag_lim=16.0):
    """
    Refactored CIBER flat-sky QE pipeline using dictionary-based approach.
    """
    
    lam_dict = dict({1:1.1, 2:1.8})
    Apix = 1.15e-9  # sr

    labels = [str(lam_dict[inst0])+' $\\mu$m']
    
    if not single_band:
        labels += str(lam_dict[inst1])+' $\\mu$m'

    if calc_ciber_cross:
        labels += str(lam_dict[inst0])+' $\\mu$m $\\times$ '+str(lam_dict[inst1])+' $\\mu$m'
        
    nlab = len(labels)

    if calc_ciber_cross or not single_band:
        use_union_mask = True
    else:
        use_union_mask = False

    gal_densities, _ = load_delta_g_maps(catname, inst0, addstr)
    
    all_clx, all_clxerr = [[[] for x in range(nlab)] for y in range(2)]
    all_clkk, all_clkkerr = [[[] for x in range(nlab)] for y in range(2)]  # Store per-field kappa auto-spectra
    all_tl, all_tlerr, all_tl_c02, all_tlerr_c02, all_clerr_bis_iig, all_cl_bis_iig = [[] for x in range(6)]
    all_norm_fourier = []
        
    for fieldidx, ifield in enumerate(ifield_list):
        
        # Initialize data and maps
        clf, ld, baseMap = initialize_clkk_prods(inst0, ifield, lMin=lMin, plot=plot, use_union_mask=use_union_mask, \
                                                lMax=lMax, calc_ciber_cross=calc_ciber_cross, mag_lim=mag_lim)

        if plot:
            plot_map(ld.ciber_map, title='ciber map')
    
        # Build parameter dictionary
        param_dict = build_param_dict(
            nX=ld.param_dict['nX'], nY=ld.param_dict['nY'],
            sizeX=ld.param_dict['sizeX'], sizeY=ld.param_dict['sizeY'],
            lMin=lMin, lMax=lMax, nBins=ld.param_dict['nBins'],
            ciber_inst=inst0, mean_cl_sky=ld.mean_cl_sky, Apix=Apix
        )
        
        # Build configuration dictionary
        config_dict = build_config_dict(
            apply_mask=True, cut_lxly=cut_lxly, mode=mode,
            calc_ciber_cross=calc_ciber_cross, single_band=single_band,
            apply_FW=apply_FW, compute_bis=compute_bis,
            pixel_fn_correct=False, add_noise=True, grab_cib_sim=False
        )
        
        # Build filter functions and correction factors
        cl_fns, corr_facs = build_cl_fns_and_corr_facs(
            clf, param_dict, config_dict,
            ld.ciber_unlensed_auto, ld.ciber_obs_auto, ld.mask
        )
    
        if apply_FW:
            print('applying Fourier weights')
            ld.dataFourier *= ld.rfft_fw
            if not single_band:
                ld.dataFourier2 *= ld.rfft_fw
            
        # Set up paths
        dirpath, figpath, paths = set_kappa_fpaths(inst0, inst1, ifield, tailstr, calc_ciber_cross=calc_ciber_cross)
        path_k1, path_k2, path_k1k2 = paths
        if single_band:
            paths = [paths[0]]
            
        # Compute collapsed triangles if requested
        if compute_collapsed_tris:
            lC, cl_4pt, clerr_4pt, tl_c02, tlerr_c02 = calc_4pt_stats(baseMap, ld)
            all_tl.append(cl_4pt)
            all_tlerr.append(clerr_4pt)
            all_tl_c02.append(tl_c02)
            all_tlerr_c02.append(tlerr_c02)
            
        # Load galaxy data
        galdens, galdensFourier = calc_gal_fourier(baseMap, gal_densities, ifield_list[fieldidx], ld.mask)
        
        # Build map dictionary
        map_dict = build_map_dict(ld, galdens, galdensFourier)

        # Compute all kappa products using unified function
        if calc_kappa:
            results = compute_ciber_kappa_products(
                baseMap, map_dict, cl_fns, param_dict, config_dict, corr_facs, paths
            )
            
            lC = results['lC']
            all_cl = results['all_cl']
            all_clerr = results['all_clerr']
            clxs = results['clxs']
            clxerrs = results['clxerrs']
            
            # Store per-field kappa auto-spectra
            for k in range(len(paths)):
                all_clkk[k].append(all_cl[k])
                all_clkkerr[k].append(all_clerr[k])
            
            # Plot kappa auto-spectra
            fig_k = plot_cl_kappas(lC, all_cl, all_clerr=all_clerr, colors=colors, labels=labels, ylim=[1e-11, 1e-5])
            fig_k.savefig(figpath+'/kappa_auto_CIBER_ifield'+str(ifield)+'_'+addstr+'_'+tailstr+'.pdf', bbox_inches='tight')
            
            # Store bispectrum results
            if compute_bis and results['cl_bis'] is not None:
                all_cl_bis_iig.append(results['cl_bis'])
                all_clerr_bis_iig.append(results['sCl_bis'])
            
            # Store cross-spectra
            for k in range(len(paths)):
                all_clx[k].append(clxs[k])
                all_clxerr[k].append(clxerrs[k])
            
            # Plot kappa-galaxy cross-spectra
            fig_kgx = plot_kappa_gal_cross(lC, clxs, clxerrs, inst0, inst1, colors=colors, labels=labels, ylim=[1e-12, 1e-5])
            fig_kgx.savefig(figpath+'/kappa_gal_cross_CIBER_'+catname+'_ifield'+str(ifield)+'_'+addstr+'_'+tailstr+'.pdf', bbox_inches='tight')

    
    # Compute field averages and plot
    # NOTE: Passing save=False to avoid redundant file saves - all data is in results_dict
    all_fieldav_clx, all_fieldav_clxerr, fpaths = proc_fieldav_clkg(lC, all_clx, all_clxerr, inst0=inst0, inst1=inst1, dirpath=dirpath, \
                                                                   single_band=single_band, calc_ciber_cross=calc_ciber_cross, save=False)

    # Process bispectrum results if computed
    if compute_bis and len(all_cl_bis_iig) > 0:
        all_dclkg_iig = [all_cl_bis_iig[fieldidx] * Apix / (2 * ld.mean_cl_sky) 
                        for fieldidx in range(len(ifield_list))]
        all_dclkg_err_iig = [all_clerr_bis_iig[fieldidx] * Apix / (2 * ld.mean_cl_sky) 
                            for fieldidx in range(len(ifield_list))]
        
        for fieldidx in range(len(ifield_list)):
            print('all_dclkg_iig:', all_dclkg_iig[fieldidx])
        
        textstr = 'CIBER '+str(lam_dict[inst0])+' $\\mu$m $\\times$ '+str(catname)
        # NOTE: Bispectrum data (all_cl_bis_iig, all_clerr_bis_iig) now saved in results_dict
        # skew_cl_fpath = save_skewcl_I2G_files(lC, all_cl_bis_iig, all_clerr_bis_iig, dirpath, inst0, inst1=inst1, catname=catname, addstr=addstr)

        skewcl_fig = plot_skewcl_I2G(ifield_list, lC, all_cl_bis_iig, all_clerr_bis_iig, textstr=textstr)
        skewcl_fig.savefig(figpath+'/ciber_skewcl_I2G_TM'+str(inst0)+'.pdf', bbox_inches='tight')

        clkg_bias_fig = plot_clkg_bias_from_skew(ifield_list, lC, all_dclkg_iig, all_dclkg_err_iig, textstr=textstr)
        clkg_bias_fig.savefig(figpath+'/ciber_delta_clkg_TM'+str(inst0)+'.pdf', bbox_inches='tight')

    
    if compute_collapsed_tris:
        fig_tl = plot_coll_tris_and_ratio(inst0, lC, all_tl, all_tlerr, all_tl_c02, all_tlerr_c02, ifield_list)
        fig_tl.savefig(figpath+'/ciber_coll_tris_and_ratio_'+tailstr+'.pdf', bbox_inches='tight')
        fig_tl.savefig(figpath+'/ciber_coll_tris_and_ratio_'+tailstr+'.png', bbox_inches='tight', dpi=300)
    
    
    if single_band:
        fig_kgall = plot_kappa_gal_all_fields_singleband(inst0, lC, all_clx, all_clxerr, all_fieldav_clx, all_fieldav_clxerr, \
                                       ifield_list=ifield_list, bbox_to_anchor=bbox_to_anchor, text_fs=16, textypos=textypos, ylim=ylim, \
                                                         figsize=(5, 4))
    else:
        fig_kgall = plot_kappa_gal_all_fields(lC, all_clx, all_clxerr, all_fieldav_clx, all_fieldav_clxerr, \
                                       inst0=inst0, inst1=inst1, ifield_list=ifield_list)

    
    fig_kgall.savefig(figpath+'/ciber_kappa_cross_'+catname+'_'+addstr+'_'+tailstr+'.pdf', bbox_inches='tight')
    fig_kgall.savefig(figpath+'/ciber_kappa_cross_'+catname+'_'+addstr+'_'+tailstr+'.png', bbox_inches='tight', dpi=300)
    
    # Build comprehensive results dictionary
    results_dict = {
        'lC': lC,
        'ifield_list': ifield_list,
        'inst0': inst0,
        'inst1': inst1,
        'catname': catname,
        'addstr': addstr,
        'lMin': lMin,
        'lMax': lMax,
        'mag_lim': mag_lim,
        
        # Per-field kappa auto-spectra (list of lists)
        'all_clkk': all_clkk,  # [n_bands][n_fields] structure
        'all_clkkerr': all_clkkerr,
        
        # Per-field kappa-galaxy cross-spectra (list of lists)
        'all_clkg': all_clx,  # [n_bands][n_fields] structure
        'all_clkgerr': all_clxerr,
        
        # Field-averaged kappa-galaxy cross-spectra
        'all_fieldav_clkg': all_fieldav_clx,
        'all_fieldav_clkgerr': all_fieldav_clxerr,
        
        # Bispectrum results (per field)
        'all_cl_bis_iig': np.array(all_cl_bis_iig) if len(all_cl_bis_iig) > 0 else None,
        'all_clerr_bis_iig': np.array(all_clerr_bis_iig) if len(all_clerr_bis_iig) > 0 else None,
        'all_dclkg_iig': np.array(all_dclkg_iig) if compute_bis and len(all_cl_bis_iig) > 0 else None,
        'all_dclkg_err_iig': np.array(all_dclkg_err_iig) if compute_bis and len(all_cl_bis_iig) > 0 else None,
        
        # Collapsed triangles (per field) if computed
        'all_tl': np.array(all_tl) if compute_collapsed_tris else None,
        'all_tlerr': np.array(all_tlerr) if compute_collapsed_tris else None,
        'all_tl_c02': np.array(all_tl_c02) if compute_collapsed_tris else None,
        'all_tlerr_c02': np.array(all_tlerr_c02) if compute_collapsed_tris else None,
        
        # Configuration info
        'single_band': single_band,
        'calc_ciber_cross': calc_ciber_cross,
        'compute_bis': compute_bis,
        'compute_collapsed_tris': compute_collapsed_tris,
        'cut_lxly': cut_lxly,
        'mode': mode,
        'apply_FW': apply_FW,
        
        # File paths
        'dirpath': dirpath,
        'figpath': figpath,
        'fieldav_fpaths': fpaths,
        
        # For backward compatibility
        'all_norm_fourier': all_norm_fourier
    }
    
    return results_dict


def save_ciber_results(results_dict, filepath):
    """
    Save CIBER lensing analysis results to NPZ file.
    
    Parameters:
    -----------
    results_dict : dict
        Results dictionary from run_flatskyqe_ciber
    filepath : str
        Path to save NPZ file
    """
    # Convert lists to arrays for better storage
    save_dict = {}
    for key, value in results_dict.items():
        if value is not None:
            if isinstance(value, list):
                # Handle nested lists (e.g., all_clx which is [n_bands][n_fields])
                if len(value) > 0 and isinstance(value[0], list):
                    # Convert nested lists to object array to preserve structure
                    save_dict[key] = np.array([np.array(v) for v in value], dtype=object)
                else:
                    save_dict[key] = np.array(value)
            else:
                save_dict[key] = value
    
    np.savez(filepath, **save_dict)
    print(f"Results saved to {filepath}")


def load_ciber_results(filepath):
    """
    Load CIBER lensing analysis results from NPZ file.
    
    Parameters:
    -----------
    filepath : str
        Path to NPZ file
        
    Returns:
    --------
    results_dict : dict
        Results dictionary with all measurements
    """
    data = np.load(filepath, allow_pickle=True)
    results_dict = {key: data[key] for key in data.files}
    print(f"Results loaded from {filepath}")
    print(f"Available keys: {list(results_dict.keys())}")
    return results_dict

# def save_skewcl_I2G_files(lC, clx_perfield, clxerr_perfield, dirpath, inst0, inst1=None, catname='unWISE', addstr=None):
    
# def save_kappa_g_cl_files(lC, clx_perfield, clxerr_perfield, field_av_clx, field_av_clxerr, field_weights, \
#                          dirpath, inst0, inst1=None, catname='unWISE', addstr=None):
    

    
def initialize_clkk_prods(ciber_inst, ifield, nX=1024, nY=1024, sizeX=2, sizeY=2, lMin=300., lMax=2.0e5, nBins=21, plot=False, \
                         use_union_mask=False, calc_ciber_cross=False, lMin_full=1000, mag_lim=16.0, \
                          apply_fourier_weights=True, apply_mask=True):
    
    inst1 = 2 if calc_ciber_cross else None

    # ell bins for power spectra
    lRange = (1., 2.*lMax)  # range for power spectra
    L = np.logspace(np.log10(lMin_full/2.), np.log10(2.*lMax), 1001, 10.)

    print('L:', L)
    param_dict = dict({'nX':nX, 'nY':nY, 'sizeX':sizeX, 'sizeY':sizeY, 'lMin':lMin, 'lMax':lMax, 'nBins':nBins})

    # lens data class
    ld = lens_data(param_dict=param_dict, L=L, lRange=lRange)
    
    # basic map object
    baseMap = FlatMap(nX=nX, nY=nY, sizeX=sizeX*np.pi/180., sizeY=sizeY*np.pi/180.)
    
    ld.load_maps(ciber_inst, mag_lim, ifield, inst1=inst1)
    ld.sig_clip()
    ld.mean_sub(apply_mask=apply_mask)

    if plot:
        plot_map(ld.mask, title='union mask', figsize=(5, 5))
        plot_map(ld.ciber_map, figsize=(5, 5), title='ciber map TM'+str(ciber_inst))
        if calc_ciber_cross:
            plot_map(ld.cross_map, figsize=(5, 5), title='proc regrid')
            
            
    ld.dataFourier = baseMap.fourier(ld.ciber_map)
    
    ld.dataFourier2 = None
    if calc_ciber_cross:
        ld.dataFourier2 = baseMap.fourier(ld.cross_map)

    # load Fourier weights
    if apply_fourier_weights:
        ld.load_fourier_weights(ciber_inst, ifield)
        
        if plot:
            plot_map(np.log10(ld.fourier_weights), title='log10 fourier weights')
            plot_map(np.log10(ld.rfft_fw), title='log10 RFFT fourier weights')
        
    lbobs, clauto_obs, clerr_obs = get_power_spec(ld.ciber_map, weights=ld.fourier_weights, nbins=26)

    if plot:
        plt.figure()
        plt.title('Observed power spectrum')
        plt.errorbar(lbobs, clauto_obs, yerr=clerr_obs, fmt='o', color='k')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('$\\ell$')
        plt.ylabel('$C_{\\ell}$')
        plt.show()
    
    
    clf = ciber_lens_forecast(ell_min=1, ell_max=1.1e5)
    clf.load_bl(ciber_inst, ifield, inplace=True, plot=False)
         
    # load auto power spectrum and interpolate to function
    ciber_cl = np.load('../data/input_recovered_ps/ciber_auto_'+ld.band_dict[ciber_inst]+'lt16.0_F25B.npz')
    lb, clauto = ciber_cl['lb'], ciber_cl['fieldav_cl']    
    ld.mean_cl_sky = np.mean(clauto[((lb > lMin)*(lb < lMax))])
    
    ld.load_mkk(ciber_inst, ifield, mag_lim=mag_lim)
    
    clauto *= clf.bl(lb)**2
    clauto_postmkk = np.dot(ld.mkk_clip.transpose(), clauto)
       
    ld.ciber_unlensed_auto = interp1d(lb, clauto_postmkk, kind='linear', bounds_error=False, fill_value=0.)
    
    # correct for masking fraction
    mask_frac = np.mean(ld.mask)
        
    ld.ciber_obs_auto = interp1d(lbobs, clauto_obs, kind='linear', bounds_error=False, fill_value=0.)
    ratio = ld.ciber_obs_auto(lb)/ld.ciber_unlensed_auto(lb)
    ld.clobs_from_auto = clauto*ratio
    ld.ciber_obs_auto = interp1d(lb, ld.clobs_from_auto, kind='linear', bounds_error=False, fill_value=0.)


    if plot:
        ld.plot_lensed_unlensed_auto()

    return clf, ld, baseMap


def init_clkk_prods_mock(ciber_inst, ifield, simidx, datestr='042725', nX=1024, nY=1024, sizeX=2, sizeY=2,\
                         lMin=300., lMax=2.0e5, lMin_full=1000, nBins=21, plot=False,\
                        scale_clkk=1.0, lensmode='unlensed', ifield_list = [4, 6, 7, 8], ell_max_forecast=1.1e5, 
                         sig_clip=False, mockstr='photnoiseonly_maskJ16', apply_mask=True):
    
    
    # basic map object
    baseMap = FlatMap(nX=nX, nY=nY, sizeX=sizeX*np.pi/180., sizeY=sizeY*np.pi/180.)
    
    fieldidx = ifield_list.index(ifield)
    
    clf = ciber_lens_forecast(ell_min=1, ell_max=ell_max_forecast)
    clf.load_bl(ciber_inst, ifield, inplace=True, plot=False)
    
    # ell bins for power spectra
    lRange = (1., 2.*lMax)  # range for power spectra
    L = np.logspace(np.log10(lMin_full/2.), np.log10(2.*lMax), 1001, 10.)
    param_dict = dict({'nX':nX, 'nY':nY, 'sizeX':sizeX, 'sizeY':sizeY, 'lMin':lMin, 'lMax':lMax, 'nBins':nBins})

    # mock_sim_fpath = '../data/lens_prods/mock_dat/'+datestr+'/TM'+str(ciber_inst)+'/mock_dat_'+mode+'_'+mockstr+'_simidx'+str(simidx)+'.npz'

    tmdir = '../data/lens_prods/mock_dat/'+datestr+'/TM'+str(ciber_inst)
    mock_sim_fpath = tmdir + '/mock_dat_'+lensmode+'_'+mockstr+'_simidx'+str(simidx)+'.npz'

    print('loading mock sims from ', mock_sim_fpath)

    mock_dat = np.load(mock_sim_fpath, allow_pickle=True)
    cib_map, kappa_map, total_signal, galcounts = [mock_dat[key][fieldidx] for key in ['cib_maps', 'kappa_maps', 'total_signal', 'galdens']]
    
    ld = lens_data(param_dict=param_dict, L=L, lRange=lRange)
    ld.mask = mock_dat['masks'][fieldidx]
    ld.ciber_map = total_signal
    ld.galcounts = galcounts   
    
    if plot:
        plot_map(ld.galcounts, figsize=(5, 5), title='gal counts in init_clkk_prods_mock')
        plot_map(gaussian_filter(cib_map, sigma=20), figsize=(5, 5), title='smoothed cib')

    if sig_clip:
        ld.sig_clip()

    ld.mean_sub(apply_mask=apply_mask)
    
    if plot:
        if apply_mask:
            print('applied mask..')
            plot_map(ld.mask, title='union mask', figsize=(5, 5))
        plot_map(ld.ciber_map, figsize=(5, 5), title='ciber map TM'+str(ciber_inst))
    
    ld.dataFourier = baseMap.fourier(ld.ciber_map)
    lbobs, clauto_obs, clerr_obs = get_power_spec(ld.ciber_map, weights=None, nbins=26)
    
    if plot:
        plot_cl(lbobs, clauto_obs, clerr_obs)
 
    cib_map_clipped = cib_map
    cib_map_clipped[cib_map_clipped != 0] -= np.mean(cib_map_clipped[cib_map_clipped != 0])
    
    ld.cibFourier = baseMap.fourier(cib_map_clipped)

    if plot:
        plot_map(cib_map_clipped, figsize=(5, 5))

    lb, clauto_sky, clerr_sky = get_power_spec(cib_map_clipped, weights=None, nbins=26)
    ld.mean_cl_sky = np.mean(clauto_sky[((lb > lMin)*(lb < lMax))])
    clauto_unlensed_true = clauto_sky
    
    ld.ciber_unlensed_auto = interp1d(lb, clauto_unlensed_true, kind='linear', bounds_error=False, fill_value=0.)  
    ld.ciber_obs_auto = interp1d(lb, clauto_obs, kind='linear', bounds_error=False, fill_value=0.)

    if plot:
        ld.plot_lensed_unlensed_auto()

    return clf, ld, baseMap, param_dict, kappa_map



class lens_data():
    ''' Container for data products used in QE'''
    
    ciber_basepath = '/Volumes/richext/workmac/ciber/ciber1/'
    
    lens_prod_dir = '../data/lens_prods/'
    
    band_dict = dict({1:'J', 2:'H'})

    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
            
        self.cross_map = None
        self.fourier_weights = None
            
    def load_maps(self, inst0, mag_lim, ifield, dat_type='observed', inst1=None):
        
        self.ciber_map = np.load(self.lens_prod_dir+'final_proc_maps_TM'+str(inst0)+'_maglim='+str(mag_lim)+'.npz')['masked_images'][ifield-4]
        self.mask = np.array((self.ciber_map != 0)).astype(int)
        
        if inst1 is not None:
            
            proc_basepath = self.lens_prod_dir+'proc_regrid_TM2_to_TM1/'
            proc_fpath = proc_basepath+'proc_regrid_TM2_to_TM1_ifield'+str(ifield)+'_Jlim_Vega_'+str(mag_lim)+'_Hlim_Vega_'+str(mag_lim-0.5)+'_ukdebias_111323_order=2_quadoff_grad_fcsub_order2.fits'
            self.cross_map = fits.open(proc_fpath)['proc_regrid_'+str(ifield)].data
            mask_regrid = np.array((self.ciber_cross_map != 0.)).astype(int)
            self.mask *= mask_regrid
            
    
    def sig_clip(self, sig=5, nitermax=1):
        
        sigclip1 = iter_sigma_clip_mask(self.ciber_map, sig=sig, nitermax=nitermax, mask=self.mask.astype(int))
        self.mask *= sigclip1

        if self.cross_map is not None:
            sigclip2 = iter_sigma_clip_mask(self.cross_map, sig=sig, nitermax=nitermax, mask=self.mask.astype(int))
            self.mask *= sigclip2


    def mean_sub(self, apply_mask=True):
        
        if apply_mask:
            self.ciber_map *= self.mask
        self.ciber_map[(self.ciber_map != 0)] -= np.mean(self.ciber_map[(self.ciber_map != 0)])

        if self.cross_map is not None:

            if apply_mask:
                self.cross_map *= self.mask

            self.cross_map[(self.cross_map != 0)] -= np.mean(self.cross_map[(self.cross_map != 0)])

    def load_fourier_weights(self, inst, ifield):
        
        noisemodl_fpath = self.lens_prod_dir+'observed_Jlt16.0_072424_quadoff_grad_fcsub_order2/noise_bias_fieldidx'+str(ifield-4)+'.npz'
        noisemodl_file = np.load(noisemodl_fpath)
        self.fourier_weights = noisemodl_file['fourier_weights_nofluc']
        self.rfft_fw = np.fft.fftshift(self.fourier_weights)[:,:513]
        
        
    def load_mkk(self, inst, ifield, mag_lim):

        # mkk for auto unlensed
        if mag_lim > 15.0:
            mkk_type, mkkdir = 'ffest_quadoff_fcsub_order2', 'mkk_ffest'
        else:
            mkk_type, mkkdir = 'mask_fcsub_order2_estimate', 'mkk'

        mask_tail = 'maglim_'+self.band_dict[inst]+'_Vega_'+str(mag_lim)+'_111323_ukdebias'
        mkk_mat = fits.open('../data/fluctuation_data/TM'+str(inst)+'/'+mkkdir+'/'+mask_tail+'/mkk_'+mkk_type+'_ifield'+str(ifield)+'_observed_'+mask_tail+'.fits')
        mkk = mkk_mat['Mkk_'+str(ifield)].data
        
        self.mkk_clip = mkk[2:-1, 2:-1]

        
    def plot_lensed_unlensed_auto(self):
        
        plt.figure(figsize=(5,4))
        plt.plot(self.L, self.ciber_unlensed_auto(self.L), label='Unlensed auto')
        plt.plot(self.L, self.ciber_obs_auto(self.L), label='Observed pseudo-$C_{\\ell}$')
        # plt.plot(self.L, self.ciber_unlensed_auto_true(self.L), label='True pseudo-$C_{\\ell}$ auto (sky)')

        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(1e2, 1e5)
        plt.legend()
        plt.xlabel('$\\ell$', fontsize=14)
        plt.ylabel('$C_{\\ell}$', fontsize=14)
        plt.grid(alpha=0.3)
        plt.show()
        
        plt.figure()
        plt.plot(self.L, self.ciber_unlensed_auto(self.L)/self.ciber_obs_auto(self.L))
        plt.yscale('log')
        plt.xscale('log')
        plt.ylabel('WF')
        plt.xlabel('$\\ell$')
        plt.show()