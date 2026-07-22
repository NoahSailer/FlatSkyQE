import numpy as np
import os
import config
import sys

import universe
import pn_2d
import flat_map
import weight

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

from scipy.interpolate import interp1d
import matplotlib
from scipy.ndimage import gaussian_filter1d


def fCtot_identity(l):
    return 1.

def compute_Neff(weights):
    N_eff = np.sum(weights)**2/np.sum(weights**2)

    return N_eff

def compute_multipole_bins(lMax, nbins=50):
    lRange = np.arange(1., 2.*lMax)  # range for power spectra
    lEdges = np.logspace(np.log10(np.min(lRange)), np.log10(np.max(lRange)), nbins+1, 10.)
    ell = baseMap.l.flatten()
    lCen, lEdges, binIndices = stats.binned_statistic(ell, ell, statistic='mean', bins=lEdges)
    
    return lRange, lEdges, ell, lCen, binIndices

def get_count_field(x_all, y_all, imdim=1024, smooth=False, smooth_sig=20, mean_sub=False):
    
    H, xedge, yedge = np.histogram2d(x_all, y_all, [np.arange(imdim+1)-0.5, np.arange(imdim+1)-0.5])
    
    if smooth:
        cf = gaussian_filter(H.transpose(), sigma=smooth_sig)
    else:
        cf = H
        
    if mean_sub:
        cf -= np.mean(cf)
    
    return cf


def iter_sigma_clip_mask(image, sig=5, nitermax=10, mask=None):
    # this version makes copy of the mask to be modified, rather than modifying the original
    # image assumed to be 2d
    iteridx = 0

    summask = image.shape[0]*image.shape[1]

    if mask is not None:
        running_mask = mask.copy()
    else:
        running_mask = np.ones_like(image)

    while iteridx < nitermax:

        new_mask = sigma_clip_maskonly(image, previous_mask=running_mask, sig=sig)

        if np.sum(running_mask*new_mask) < summask:
            running_mask *= new_mask
            summask = np.sum(running_mask)
        else:
            return running_mask

        iteridx += 1

    return running_mask

def sigma_clip_maskonly(vals, previous_mask=None, sig=5):
    
    valcopy = vals.copy()
    if previous_mask is not None:
        valcopy[previous_mask==0] = np.nan
        sigma_val = np.nanstd(valcopy)
    else:
        sigma_val = np.nanstd(valcopy)
    
    abs_dev = np.abs(vals-np.nanmedian(valcopy))
    mask = (abs_dev < sig*sigma_val).astype(int)

    return mask

def proc_input_map(input_map, mask, galdens=False, apply_mask=True):
    

    if not apply_mask:
        mask_use = np.ones_like(mask)
    else:
        mask_use = mask.copy()

    proc_map = input_map*mask_use
    
    # meanval = np.mean(proc_map[proc_map != 0])
    meanval = np.mean(input_map[mask_use != 0])
    
    proc_map[mask_use != 0] -= meanval
    
    if galdens:
        proc_map /= meanval
        
    return proc_map

def compute_map_ps(baseMap, input_map, mask_input, apply_mask=True):
    
    if apply_mask:
        mask = np.ones_like(mask_input)
    else:
        mask = mask_input
    input_map *= mask
    input_map[input_map != 0] -= np.mean(input_map[input_map != 0])
    
    fourier_map = baseMap.fourier(input_map)
    
    lC, cl, clerr = baseMap.powerSpectrum(fourier_map)
    
    return lC, cl, clerr


def calc_gal_fourier(baseMap, gal_densities, ifield, mask, plot=False):

    galdens = gal_densities['ifield'+str(ifield)].data.transpose()
    
    if plot:
        plot_map(galdens, title='original galdens')
    galdens *= mask
    
    if plot:
        plot_map(galdens, title='masked galdens')

    meandens = np.mean(galdens[mask != 0])
    print('mean density is ', meandens)

    galdens[mask != 0] -= meandens
    galdens[mask != 0] /= meandens
    
    if plot:
        plot_map(galdens, title='gal overdensity map')
        
    galdensFourier = baseMap.fourier(galdens)

    return galdens, galdensFourier

def compute_skew_cl_I2G_WF(baseMap, ciber_map, galdensFourier, W_ell, lMax=5e4, lMin=1e4):
    """Skew spectrum WITH Wiener filter weighting"""
    
    dataFourier = baseMap.fourier(ciber_map)
    
    # Apply Wiener filter
    def WF_filter(l):
        if (l < lMin) or (l > lMax):
            return 0.
        return W_ell(l)
    
    iVarDataFourier = baseMap.filterFourierIsotropic(WF_filter, dataFourier=dataFourier, test=False)
    iVarData = baseMap.inverseFourier(iVarDataFourier).real
    
    # Now square the Wiener-filtered map
    cb_sq = iVarData**2
    cb_sq[cb_sq != 0] -= np.mean(cb_sq[cb_sq != 0])
    
    cbsq_fourier = baseMap.fourier(cb_sq)
    lCen, Cl, sCl = baseMap.crossPowerSpectrum(cbsq_fourier, galdensFourier, plot=False)
    
    return lCen, Cl, sCl

def compute_skew_cl_I2G_simp(baseMap, ciber_map, galdensFourier, lMax=5e4, lMin=1e4, verbose=False, 
                              filter_mode='bandpass', W_ell=None):
    '''
    Skew spectrum with flexible filtering.
    
    Parameters:
    -----------
    filter_mode : str
        'bandpass' - Simple hard bandpass filter (default, original behavior)
        'wiener'   - Wiener filter using W_ell
    W_ell : callable, optional
        Wiener filter weight function C_unlensed/C_obs. Required if filter_mode='wiener'.
    '''
    
    if filter_mode == 'wiener':
        if W_ell is None:
            raise ValueError("W_ell must be provided when filter_mode='wiener'")
        
        # Apply Wiener filter before squaring
        dataFourier = baseMap.fourier(ciber_map)
        
        def WF_filter(l):
            if (l < lMin) or (l > lMax):
                return 0.
            return W_ell(l)
        
        iVarDataFourier = baseMap.filterFourierIsotropic(WF_filter, dataFourier=dataFourier, test=False)
        filtered_map = baseMap.inverseFourier(iVarDataFourier).real
        
        if verbose:
            print('Using Wiener-filtered map for bispectrum')
        
    else:  # filter_mode == 'bandpass'
        # Original behavior: use map as-is (already bandpass filtered externally)
        filtered_map = ciber_map
        if verbose:
            print('Using hard bandpass-filtered map for bispectrum')
    
    # Square the (filtered) map
    cb_sq = filtered_map**2
    if verbose:
        print('mean of squared map is ', np.mean(cb_sq))

    # cb_sq -= np.mean(cb_sq)
    cb_sq[cb_sq != 0] -= np.mean(cb_sq[cb_sq != 0])
    if verbose:
        print('mean of squared map now is ', np.mean(cb_sq))

    cbsq_fourier = baseMap.fourier(cb_sq)

    lCen, Cl, sCl = baseMap.crossPowerSpectrum(cbsq_fourier, galdensFourier, plot=False)
        
    return lCen, Cl, sCl

def compute_skew_cl_I2G(baseMap, ciber_map, galdensFourier, fC0, fCtot, lMax=5e4, lMin=1e4):
    
    ''' Skew spectrum '''

    dataFourier_cb = baseMap.fourier(ciber_map)
    
    def f2d_WF_simp(lx, ly):

        l2 = lx**2 + ly**2
        if l2==0:
            return 0.

        # if np.abs(lx) < lMin: # for read noise
        #     return 0.

        labs = np.sqrt(lx**2+ly**2)
        if (labs > lMax) or (labs < lMin):
            return 0.

        result = divide(fC0(labs), fCtot(labs))

        if not np.isfinite(result):
            result = 0.

        return result 
    
    iVarDataFourier = baseMap.filterFourier(f2d_WF_simp, dataFourier=dataFourier_cb, test=False)
    iVarData = baseMap.inverseFourier(iVarDataFourier).real
    
    cb_sq = iVarData**2
    cb_sq[(cb_sq != 0)] -= np.mean(cb_sq[(cb_sq != 0)])
    
    cbsq_fourier = baseMap.fourier(cb_sq)
    lCen, Cl, sCl = baseMap.crossPowerSpectrum(cbsq_fourier, galdensFourier, plot=False)
        
    return lCen, Cl, sCl


def calc_4pt_stats(baseMap, ld):
    ''' Collapsed trispectrum '''
    mean_l = np.mean(baseMap.l)
    print('mean ell:', mean_l)
    lC, cl_4pt, clerr_4pt = baseMap.collapsed4PtFunc(lMean=mean_l, dataFourier=ld.dataFourier)

    tl_c02 = cl_4pt/ld.ciber_unlensed_auto(lC)**2
    tlerr_c02 = clerr_4pt/ld.ciber_unlensed_auto(lC)**2

    tl_c02[np.isnan(tl_c02)] = 0.
    tlerr_c02[tl_c02==0] = 0.
    
    return lC, cl_4pt, clerr_4pt, tl_c02, tlerr_c02

def kappa_gal_cross(baseMap, paths, mask, galdensFourier, plot=False):
    
    clxs, clxerrs = [], []
    for k, path_k in enumerate(paths):
        
        print('path_k:', path_k)
        kFourier = baseMap.loadDataFourier(path_k)
        
        kappa_map = baseMap.inverseFourier(kFourier).real
        
        kappa_map *= mask
        
        kappa_map[kappa_map != 0] -= np.mean(kappa_map[kappa_map != 0])
        
        if plot:
            plot_map(kappa_map, figsize=(5, 5), title='$\\kappa$ estimate')
            
        print('mean of kappa map is ', np.mean(kappa_map[kappa_map!= 0]))
        print('std of kappa map is ', np.std(kappa_map[kappa_map != 0]))
        
        kFourier_masked = baseMap.fourier(kappa_map)
        
        lC, clx, clxerr = baseMap.crossPowerSpectrum(kFourier_masked, galdensFourier, plot=False)
        
        clxs.append(clx)
        clxerrs.append(clxerr)
        
    return lC, clxs, clxerrs

def compute_pixel_window_fn(baseMap, plot=False):


    W2D = baseMap.pixelWindow(baseMap.lx, baseMap.ly)  # 2D pixel window function
    # define reciprocal-lattice offsets
    N = 2
    dx = 7. * np.pi / (180. * 3600.)
    Gs = [(2*np.pi*n/dx, 2*np.pi*m/dx)
          for n in range(-N,N+1) for m in range(-N,N+1)]

    def T_of_ell(lx, ly, alpha):
        T = 0.0
        for (gx,gy) in Gs:
            lx2, ly2 = lx+gx, ly+gy
            W2 = baseMap.pixelWindow(lx2, ly2)**2
            l2 = np.sqrt(lx2**2+ly2**2)
            T += W2 * (l2/np.sqrt(lx*lx+ly*ly))**(-alpha)
        return T

    ell2D = np.sqrt(baseMap.lx**2 + baseMap.ly**2)  # 2D ell values

    W2D_Tell = T_of_ell(baseMap.lx, baseMap.ly, alpha)
    W1D_Tell = radial_bin(ell2D.flatten(), W2D_Tell.flatten(), lEdges)  # Obtain 1D window

    # Radial binning to create 1D pixel window function
    W1D = radial_bin(ell2D.flatten(), W2D.flatten(), lEdges)  # Obtain 1D window
    inv_W_pix = 1.0 / W1D  # Compute inverse pixel window function

    if plot:
        plt.figure(figsize=(5, 4))
        plt.title('pixel window function')
        plt.plot(lCen, W1D, label='basic window sinc')
        plt.plot(lCen, W1D_Tell, label='Full t_ell')
        plt.legend()

        plt.xscale('log')
        plt.yscale('log')
        plt.show()

    return W2D_Tell, W1D_Tell, W1D


def proc_skewspec(lC, skewcl, skewclerr, B_ell=None, vbeam=None, unmask_frac=None):

    if B_ell is not None:
        skewcl /= B_ell(lC)**2
        skewclerr /= B_ell(lC)**2

    if vbeam is not None:
        skewcl /= vbeam
        skewclerr /= vbeam

    if unmask_frac is not None:
        skewcl /= unmask_frac
        skewclerr /= unmask_frac

    return skewcl, skewclerr

def proc_clkg(lC, clkg, clkgerr, B_ell=None, kcorr=None, unmask_frac=None):

    if B_ell is not None:
        clkg /= B_ell(lC)**2
        clkgerr /= B_ell(lC)**2

    if kcorr is not None:
        clkg /= kcorr
        clkgerr /= kcorr

    if unmask_frac is not None:
        clkg /= unmask_frac
        clkgerr /= unmask_frac

    return clkg, clkgerr

def inv_var_weights(lb, all_dcl, plot=False):
    
    field_weights = 1/all_dcl**2
    
    nfield, n_ps_bin = all_dcl.shape[0], all_dcl.shape[1]
    
    for n in range(n_ps_bin):
        field_weights[:,n] /= np.sum(field_weights[:,n])
        
    if plot:
        plt.figure(figsize=(5, 4))
        
        for n in range(nfield):
            plt.plot(lb, field_weights[n], color='C'+str(n))
            
        plt.xlabel('$\\ell$', fontsize=12)
        plt.ylabel('Field weights', fontsize=12)
        plt.xscale('log')
        plt.show()
        
    return field_weights

def compute_weighted_cl(indiv_cl, field_weights):
    
    n_ps_bin = len(indiv_cl[0])
    field_average_cl, field_average_dcl = [np.zeros_like(indiv_cl[0]) for x in range(2)]
    
    for n in range(n_ps_bin):
        field_weights[:,n] /= np.sum(field_weights[:,n])
        field_average_cl[n] = np.average(indiv_cl[:,n], weights=field_weights[:,n])
        neff_indiv = compute_Neff(field_weights[:,n])

        psvar_indivbin = np.sum(field_weights[:,n]*(indiv_cl[:,n] - field_average_cl[n])**2)*neff_indiv/(neff_indiv-1.)
        field_average_dcl[n] = np.sqrt(psvar_indivbin/neff_indiv)
        
    return field_average_cl, field_average_dcl

def proc_fieldav_clkg(lC, all_clx, all_clxerr, inst0=1, inst1=2, save=True, dirpath=None, single_band=False, \
    calc_ciber_cross=True):
    
        
    all_fieldav_clx, all_fieldav_clxerr, fpaths = [[] for x in range(3)]
    
    inst_list = [inst0]

    if not single_band and inst1 is not None:
        inst_list.append(inst1)

    for idx, inst in enumerate(inst_list):
        
        field_weights = inv_var_weights(lC, np.array(all_clxerr[idx]))
        field_av_clx, field_av_clxerr = compute_weighted_cl(np.array(all_clx[idx]), field_weights)

        fpath_save = save_kappa_g_cl_files(lC, all_clx[idx], all_clxerr[idx], field_av_clx, field_av_clxerr, field_weights, \
                            dirpath, inst)
        
        fpaths.append(fpath_save)
        
        all_fieldav_clx.append(field_av_clx)
        all_fieldav_clxerr.append(field_av_clxerr)
        
    if calc_ciber_cross:
        # cross estimator
        field_weights = inv_var_weights(lC, np.array(all_clxerr[2]))
        field_av_clx, field_av_clxerr = compute_weighted_cl(np.array(all_clx[2]), field_weights)
        
        all_fieldav_clx.append(field_av_clx)
        all_fieldav_clxerr.append(field_av_clxerr)

        fpath_save = save_kappa_g_cl_files(lC, all_clx[2], all_clxerr[2], field_av_clx, field_av_clxerr, field_weights, \
                            dirpath, inst0, inst1=inst1)
        
        fpaths.append(fpath_save)

    
    return all_fieldav_clx, all_fieldav_clxerr, fpaths
