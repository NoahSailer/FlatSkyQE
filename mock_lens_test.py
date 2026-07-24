# from FlatSkyQE.diagnostic_qe_components import W_ell
from tabnanny import verbose

import universe
import pn_2d
import flat_map
import weight

from flat_map import *
from pn_2d import *
from universe import *
import halo_fit
from halo_fit import *
from weight import *

import cmb
from cmb import *

import config
import sys
import os
import matplotlib
import scipy
from scipy.interpolate import interp1d
import numpy as np

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(parent_dir)
from ciber.core.powerspec_pipeline import *
from ciber.core.ps_pipeline_go import *
from ciber.io.ciber_data_utils import *
from ciber.plotting.plotting_fns import plot_map
from kappa_auto_cross_fns import *
from bias_modl import *
from cibmockgen import *
from map_clus_utils import proc_input_map, compute_map_ps, proc_skewspec, proc_clkg, compute_skew_cl_I2G_simp
from forecast_cib_lens import ciber_lens_forecast
from bias_modl import effective_beam_skew_I2g_correct

class ScaledP2dAuto:
    def __init__(self, original_p2d, alpha):
        self.original_p2d = original_p2d
        self.alpha = alpha

    def eval(self, ell):
        return self.alpha * self.original_p2d.eval(ell)

    # If you need attributes like fPinterp or whatever, you can "proxy" them
    def __getattr__(self, name):
        return getattr(self.original_p2d, name)


 
# def proc_skewspec(lC, skewcl, skewclerr, B_ell=None, vbeam=None, unmask_frac=None):
# def proc_clkg(lC, clkg, clkgerr, B_ell=None, kcorr=None, unmask_frac=None):
# def compute_multipole_bins(lMax, nbins=50):


def plot_normalization(lC, N_L, N_L_err=None, lMin=None, lMax=None, figsize=(5, 4), show=True, ylim=[1e-11, 1e-6]):
    """
    Plot QE normalization N_L^κ (convergence normalization after L⁴ correction).
    
    Key point: N_L^κ = L⁴/4 × N_L^φ
    After L⁴ correction, N_L^κ should be nearly flat or mildly varying.
    """   

    fig = plt.figure(figsize=figsize)
 
    # Plot N_L^κ (convergence normalization - what actually matters!)
    if N_L_err is not None:
        plt.errorbar(lC, N_L, yerr=N_L_err, fmt='o-', 
                    label='$N_L^\\kappa$ (convergence normalization)', 
                    alpha=0.7, markersize=5, color='red', linewidth=2)
    else:
        plt.plot(lC, N_L, 'o-', label='$N_L^\\kappa$ (convergence normalization)', 
                alpha=0.7, markersize=5, color='red', linewidth=2)
    
    # Fit power law for N_L^κ
    valid_kappa = ~np.isnan(N_L) & (N_L > 0)
    if lMin:
        valid_kappa &= (lC > lMin)
    if lMax:
        valid_kappa &= (lC < lMax)
    
    if np.sum(valid_kappa) > 5:
        log_l = np.log10(lC[valid_kappa])
        log_N_kappa = np.log10(N_L[valid_kappa])
        slope_kappa, intercept_kappa = np.polyfit(log_l, log_N_kappa, 1)
        fit_N_kappa = 10**(intercept_kappa) * lC**slope_kappa
        plt.plot(lC, fit_N_kappa, '--', color='darkgray', linewidth=2.5, alpha=0.8,
                label=f'Power-law fit: $\\propto L^{{{slope_kappa:.2f}}}$')
                
    
    # Mark the lMax region
    if lMax:
        plt.axvline(lMax, color='red', linestyle='--', alpha=0.5, linewidth=2, label=f'lMax = {lMax:.0f}')
        # plt.axvline(0.8*lMax, color='orange', linestyle=':', alpha=0.5, linewidth=1.5, label=f'0.8×lMax')
    if lMin:
        plt.axvline(lMin, color='blue', linestyle='--', alpha=0.5, linewidth=2, label=f'lMin = {lMin:.0f}')
    
    plt.xlabel('$L$', fontsize=16)
    plt.ylabel('$N_L^\\kappa$', fontsize=16)
    plt.title('QE Convergence Normalization', fontsize=16)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(ylim)
    plt.xlim(1e2, 2e5)
    plt.legend(fontsize=11, loc=2)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    if show:
        plt.show()

    return fig



def plot_normalization_components(L_bins, N_L_fft, N_L_forward, components, lMin=None, lMax=None):
    """
    Compare FFT-based normalization with forward-modeled components.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Compare FFT vs forward model
    ax1.plot(L_bins, N_L_fft, 'o-', label='FFT method (from code)', color='blue', markersize=4)
    ax1.plot(L_bins, N_L_forward, 's--', label='Forward model', color='red', markersize=4, alpha=0.7)
    ax1.set_xlabel('$L$', fontsize=12)
    ax1.set_ylabel('$N_L$', fontsize=12)
    ax1.set_title('Normalization: FFT vs Forward Model', fontsize=13)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, which='both')
    
    if lMax:
        ax1.axvline(lMax, color='red', linestyle='--', alpha=0.3)
    
    # Plot 2: Individual components vs L
    ax2.plot(L_bins, components['F_avg'], 'o-', label='$F(\\ell,L-\\ell)$ avg', markersize=3)
    ax2.plot(L_bins, components['f_kappa_avg'], 's-', label='$f^\\kappa$ avg', markersize=3)
    ax2.plot(L_bins, components['beam_sq_avg'], '^-', label='$B(\\ell)B(L-\\ell)$ avg', markersize=3)
    ax2.set_xlabel('$L$', fontsize=12)
    ax2.set_ylabel('Component value', fontsize=12)
    ax2.set_title('Average Component Values', fontsize=13)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, which='both')
    
    # Plot 3: Contributing modes
    ax3.plot(L_bins, components['n_modes'], 'o-', color='purple', markersize=4)
    ax3.set_xlabel('$L$', fontsize=12)
    ax3.set_ylabel('Number of contributing (ℓ, L-ℓ) pairs', fontsize=12)
    ax3.set_title('Mode Count (Geometric Coupling)', fontsize=13)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3, which='both')
    
    if lMax:
        ax3.axvline(lMax, color='red', linestyle='--', alpha=0.3)
    
    # Plot 4: Ratio to isolate dominant effect
    # Normalize everything to first valid point
    idx_ref = np.argmax(N_L_forward > 0)
    
    N_L_norm = N_L_forward / N_L_forward[idx_ref]
    F_norm = components['F_avg'] / components['F_avg'][idx_ref]
    f_kappa_norm = components['f_kappa_avg'] / components['f_kappa_avg'][idx_ref]
    beam_norm = components['beam_sq_avg'] / components['beam_sq_avg'][idx_ref]
    n_modes_norm = components['n_modes'] / components['n_modes'][idx_ref]
    
    ax4.plot(L_bins, N_L_norm, 'o-', label='$N_L$ (total)', color='black', linewidth=2, markersize=5)
    ax4.plot(L_bins, F_norm, 's--', label='$F$ factor', alpha=0.7, markersize=3)
    ax4.plot(L_bins, f_kappa_norm, '^--', label='$f^\\kappa$ factor', alpha=0.7, markersize=3)
    ax4.plot(L_bins, beam_norm, 'v--', label='Beam factor', alpha=0.7, markersize=3)
    ax4.plot(L_bins, n_modes_norm, 'd--', label='Mode count', alpha=0.7, markersize=3)
    
    # Add power-law references
    L_ref = L_bins / L_bins[idx_ref]
    ax4.plot(L_bins, L_ref**(-1), ':', color='gray', alpha=0.5, label='$L^{-1}$')
    ax4.plot(L_bins, L_ref**(-2), ':', color='gray', alpha=0.5, label='$L^{-2}$')
    ax4.plot(L_bins, L_ref**(-3), ':', color='gray', alpha=0.5, label='$L^{-3}$')
    
    ax4.set_xlabel('$L$', fontsize=12)
    ax4.set_ylabel('Normalized to reference L', fontsize=12)
    ax4.set_title('Component Scaling (identifies dominant effect)', fontsize=13)
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.legend(fontsize=8, loc='best', ncol=2)
    ax4.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.show()
    
    # Print component scaling
    print("\n" + "="*60)
    print("COMPONENT SCALING ANALYSIS")
    print("="*60)
    for name, values in [('N_L total', N_L_norm), 
                         ('F factor', F_norm),
                         ('f_kappa factor', f_kappa_norm),
                         ('Beam factor', beam_norm),
                         ('Mode count', n_modes_norm)]:
        valid = values > 0
        if np.sum(valid) > 5:
            log_L = np.log10(L_bins[valid])
            log_val = np.log10(values[valid])
            slope, _ = np.polyfit(log_L, log_val, 1)
            print(f"{name:20s}: ∝ L^{slope:+.2f}")
    print("="*60)


def extend_cls_powerlaw(ell, cl, ell_max_extend=200_000, alpha=None):
    """
    Extend Cl(ell) beyond the given range as a power law.
    
    Parameters:
    - ell: 1D array of multipoles (e.g., ell = np.arange(10, 40000))
    - cl: 1D array of C_ell values
    - ell_max_extend: maximum ell to extrapolate to
    - alpha: if provided, extrapolate as Cl ~ ell^alpha; otherwise estimate from last two points

    Returns:
    - extended_ell: extended ell array
    - extended_cl: extended Cl array
    - cl_interp_fn: callable function returning extended Cl
    """

    # Estimate power-law slope from the last two bins if not provided
    if alpha is None:
        log_ell = np.log(ell[-2:])
        log_cl = np.log(cl[-2:])
        alpha = np.diff(log_cl) / np.diff(log_ell)
        alpha = alpha[0]

    print(f"Using power-law extrapolation with slope alpha = {alpha:.3f}")

    ell_extend = np.arange(ell[-1] + 1, ell_max_extend + 1)
    cl_extend = cl[-1] * (ell_extend / ell[-1])**alpha

    # Combine original and extrapolated data
    ell_full = np.concatenate([ell, ell_extend])
    cl_full = np.concatenate([cl, cl_extend])

    # Make sure to allow extrapolation if needed
    cl_interp_fn = interp1d(ell_full, cl_full, kind='linear', bounds_error=False, fill_value='extrapolate')

    return ell_full, cl_full, cl_interp_fn


# def sample_galaxy_positions(lognormal_field, n_galaxies, add_subpix_scatter=True):


def grab_nbar_tracer_cat(mock_cat, Adeg=4., nbar_targ=2e4):
    
    # order by magnitude
    
    N_target = int(np.floor(Adeg * nbar_targ))
    
    # Order by increasing magnitude (brightest first)
    sorted_indices = np.argsort(mock_cat[:, 3])
    
    # Select the top N_target galaxies
    sel = sorted_indices[:N_target]
    
    # Return the selected galaxies
    sel_tracer_cat = mock_cat[sel, :]
    
    return sel_tracer_cat

# def get_count_field(x_all, y_all, imdim=1024, smooth=False, smooth_sig=20, mean_sub=False):


class mock_lens_dat():
    ''' 
    With these mocks we can take a mock kappa field, use it to lens CIB emission, 
    apply the QE to estimate the lensing, and then calculate the cross-correlation with the initial kappa field.
    '''
    
    mock_data_dir = config.ciber_basepath+'data/ciber_mocks/'
    datestr = '112022'
    datestr_trilegal = '112022'
        
    ifield_list = [4, 6, 7, 8]

    
    def __init__(self, inst, baseMap, cbps=None):
        
        self.baseMap = baseMap
        self.inst = inst
        
        if cbps is None:
            cbps = CIBER_PS_pipeline()
        self.cbps = cbps
        
        self.set_up_filepaths(inst)

                
    def set_up_filepaths(self, inst):
        self.config_dict, self.pscb_dict, self.float_param_dict, fpath_dict = return_default_cbps_dicts()

        self.fpath_dict, list_of_dirpaths, self.base_path, self.trilegal_base_path = set_up_filepaths_cbps(fpath_dict, inst, 'test', '111323',\
                                                                                datestr_trilegal='112022', data_type='mock', \
                                                                               save_fpaths=True)
    
    def prep_lens(self, scale_clkk=1.0):
        
        u = UnivPlanck15()
        halofit = Halofit(u, save=False)
        w_cmblens = WeightLensSingle(u, z_source=1100., name="cmblens")
        self.p2d_cmblens = P2dAuto(u, halofit, w_cmblens, save=False)
        
        
    def gen_kappa(self, scale_clkk=1.0):
        
        ''' Generate kappa field, save power spectrum'''
        
        all_kappa, all_Cl, all_sCl = [[] for x in range(3)]

        # extend input lensing power spectrum to higher ell
        ell = np.arange(10, 40000)
        cl_kappa = self.p2d_cmblens.fPinterp(ell)
        ell_ext, cl_ext, cl_kappa_fn = extend_cls_powerlaw(ell, cl_kappa, ell_max_extend=200000)

        for fieldidx, ifield in enumerate(self.ifield_list):
            # kFourier = self.baseMap.genGRF(self.p2d_cmblens.fPinterp, test=False)
            kFourier = self.baseMap.genGRF(cl_kappa_fn, test=False)

            kappa = self.baseMap.inverseFourier(kFourier)
            print("plot kappa map")
            
            kappa *= np.sqrt(scale_clkk)
            fig = plot_map(kappa, title='kappa, scale_clkk='+str(scale_clkk), figsize=(5, 5), return_fig=True)
            
            kFourier = self.baseMap.fourier(kappa)
            lCen, Cl, sCl = self.baseMap.powerSpectrum(kFourier, theory=[self.p2d_cmblens.fPinterp], plot=False, save=False)

            all_kappa.append(kappa)
            all_Cl.append(Cl)
            all_sCl.append(sCl)
            
        return all_kappa, lCen, all_Cl, all_sCl
    
    def save_kappa_realiz(self, all_kappa, lCen, all_Cl, all_sCl):
        
        kappa_fpath = ''
        print('Saving kappa realizations and power spectra to ', kappa_fpath)
        np.savez(kappa_fpath, all_kappa=all_kappa, lCen=lCen, all_Cl=all_Cl, all_sCl=all_sCl)
        return kappa_fpath
    
    def load_kappa_realiz(self, kappa_fpath):
        
        kap = np.load(kappa_fpath)
        
        return kap['all_kappa'], kap['lCen'], kap['all_Cl'], kap['all_sCl']

    
    def apply_lensing(self, unlensed_map, kappa_map):

        print("Fourier transform of kappa map..")
        kFourier = self.baseMap.fourier(kappa_map)
        
        print("Lens the CIB map")
        lensed_map = self.baseMap.doLensing(unlensed_map, kappaFourier=kFourier)
        lensed_map_fourier = self.baseMap.fourier(lensed_map)
        print("plot lensed CIB map")
        print("check the power spectrum")
        lCen, Cl, sCl = self.baseMap.powerSpectrum(lensed_map_fourier, plot=False, save=False)
        
        return lensed_map, lCen, Cl, sCl
        
        
        
    def gen_mock_set(self, mask_tail='maglim_16.0_Vega_081323', datestr_mock='112022', cib_setidx=0, make_lens_map=False, save_kappa=False, \
                     do_lensing=False, add_isl=False, load_mask=False, with_noise=False, with_read_noise=False, plot=False, figsize=(5, 5), \
                    scale_clkk=1.0, datestr_lens_mock='050725', lens_mode='lensed', with_tracer=True, nbar_tracer=2e4):
        

        ''' Keep existing masks but use new CIB realizations. There may be a handful of bright galaxies but can clip '''


        basepath = '/Users/richardfeder/Documents/ciber/data/lens_prods/mock_dat/'
        cib_fpaths = [basepath+datestr_lens_mock+'/'+lens_mode+'_cib_mock_set'+str(cib_setidx)+'_TM'+str(self.inst)+'_ifield'+str(ifield)+'_nbar='+str(nbar_tracer)+'.npz' for ifield in self.ifield_list]

        print('cib fpaths:', cib_fpaths)
        # cib_fpaths = [self.mock_data_dir+datestr_lens_mock+'/'+lens_mode+'_cib_mock_set'+str(cib_setidx)+'_TM'+str(self.inst)+'_ifield'+str(ifield)+'.npz' for ifield in self.ifield_list]
        cib_maps = np.array([np.load(cib_fpath)['cib_map'] for cib_fpath in cib_fpaths])
        all_kappa = np.array([np.load(cib_fpath)['comb_kappa'] for cib_fpath in cib_fpaths])

        if with_tracer:
            tracer_cats = [np.load(cib_fpath)['tracer_cat'] for cib_fpath in cib_fpaths]
            print('first tracer catalog has shape', tracer_cats[0].shape)
            count_fields = np.array([get_count_field(tracer_cats[i][:,0], tracer_cats[i][:,1]) for i in range(len(tracer_cats))])

            meandens = np.array([np.mean(cf) for cf in count_fields])
            print('mean density is ', meandens)

        else:
            count_fields = None 
            tracer_cats = None

        make_lens_map = False
        # load CIB maps
        # cib_fpath = self.mock_data_dir+self.datestr+'/TM'+str(self.inst)+'/cib_realiz/'
        # cib_fpath += 'cib_with_tracer_with_dpoint_5field_set'+str(cib_setidx)+'_'+self.datestr+'_TM'+str(self.inst)+'.fits'
        
        # cib = fits.open(cib_fpath)
        # cib_maps = np.array([cib['cib_'+str(self.cbps.inst_to_band[self.inst])+'_'+str(ifield)].data for ifield in self.ifield_list])


        if load_mask:
        
            masks = np.zeros_like(cib_maps)
            if mask_tail is None:
                mask_tail = 'maglim_16.0_Vega_081323'
                
            for fieldidx, ifield in enumerate(self.ifield_list):
            
                mask_fpath = config.ciber_basepath+'data/ciber_mocks/'+datestr_mock+'/TM'+str(self.inst)+'/masks/'+mask_tail+'/joint_mask_ifield'+str(ifield)+'_inst'+str(self.inst)+'_simidx'+str(cib_setidx)+'_'+mask_tail+'.fits'
                
                masks[fieldidx] = fits.open(mask_fpath)[1].data
                
                if plot and fieldidx==0:
                    plot_map(masks[fieldidx], title='Mask', figsize=figsize)

                        
        total_signal = np.zeros_like(cib_maps)
        
        if make_lens_map:
            all_kappa, lCen, all_Cl, all_sCl = self.gen_kappa(scale_clkk=scale_clkk)
        
        if do_lensing:
            
            lensed_maps = np.zeros_like(cib_maps)
            all_Cl_lens, all_sCl_lens = [np.zeros((len(self.ifield_list), len(lCen))) for x in range(2)]
            
            for fieldidx, ifield in enumerate(self.ifield_list):
                
                lensed_map, _, Cl_lens, sCl_lens = self.apply_lensing(cib_maps[fieldidx], all_kappa[fieldidx])
                                
                lensed_maps[fieldidx] = lensed_map
                all_Cl_lens[fieldidx] = Cl_lens
                all_sCl_lens[fieldidx] = sCl_lens
                         
                if plot and fieldidx==0:

                    lb, clul, clerr_ul = get_power_spec(cib_maps[fieldidx]-np.mean(cib_maps[fieldidx]), weights=None, nbins=26)
                    lb, cl_lens, clerr_lens = get_power_spec(lensed_maps[fieldidx]-np.mean(lensed_maps[fieldidx]), weights=None, nbins=26)
                                    
            total_signal += lensed_maps
        else:
            total_signal += cib_maps
        
        # if not make_lens_map and not do_lensing:
        #     all_kappa = None
        
        # load ISL foreground  
        if add_isl:
            trilegal_fpath = self.mock_data_dir+self.datestr_trilegal+'/trilegal/mock_trilegal_simidx'+str(cib_setidx)+'_'+self.datestr_trilegal+'.fits'
               
            mock_trilegal = fits.open(trilegal_fpath)
            mock_trilegal_ims = np.array([mock_trilegal['trilegal_'+str(self.cbps.inst_to_band[self.inst])+'_'+str(ifield)].data for ifield in self.ifield_list])
            total_signal += mock_trilegal_ims

        if with_noise:

            if with_read_noise:
                noise_models = self.cbps.grab_noise_model_set(self.ifield_list, self.inst, noise_model_base_path=self.fpath_dict['read_noise_modl_base_path'], noise_modl_type=self.config_dict['noise_modl_type'])
            
            zl_levels = [self.cbps.zl_levels_ciber_fields[self.inst][self.cbps.ciber_field_dict[ifield]] for ifield in self.ifield_list]
            
            zl_perfield = np.array([generate_zl_realization(zl_levels[fieldidx], False, dimx=self.cbps.dimx, dimy=self.cbps.dimy) for fieldidx in range(len(self.ifield_list))])
            
            total_signal += zl_perfield
            
            for fieldidx, ifield in enumerate(self.ifield_list):
                
                field_nfr = self.cbps.field_nfrs[ifield]

                shot_sigma_sb = self.cbps.compute_shot_sigma_map(self.inst, image=total_signal[fieldidx], nfr=field_nfr)
                snmap = shot_sigma_sb*np.random.normal(0, 1, size=self.cbps.map_shape)
                
                total_signal[fieldidx] += snmap

                if with_read_noise:
                    rnmap, _ = self.cbps.noise_model_realization(self.inst, self.cbps.map_shape, noise_models[fieldidx], read_noise=True, photon_noise=False, chisq=False)

                    total_signal[fieldidx] += rnmap
                
                if plot and fieldidx==0:
                    if with_read_noise:
                        plot_map(rnmap+snmap, title='read+photon noise', figsize=figsize)
                    else:
                        plot_map(snmap, title='photon noise', figsize=figsize)

        if plot:
            plot_map(total_signal[0], title='total signal', figsize=figsize)
            plot_map(total_signal[0]*masks[0], title='masked signal', figsize=figsize)
        
        # if do_lensing

        dat_dict = {'total_signal':total_signal, 'cib_maps':cib_maps, 'all_kappa':all_kappa, 'masks':masks, 'count_fields':count_fields, 
                    'tracer_cats':tracer_cats}

        return dat_dict
        # return total_signal, cib_maps, all_kappa, masks, count_fields

        # return total_signal, cib_maps, all_kappa, masks


def mock_gen_wrapper(nsim, plot=True, with_noise=True, with_read_noise=True, do_lensing=True, make_lens_map=True, \
                     load_mask=True, add_isl=False, masktail=None, nX=1024, nY=1024, sizeX=2., sizeY=2., scale_clkk=1.0, inst_list=[1, 2], \
                    mock_test_dat_dir='../data/lens_prods/mock_dat', figsize_plot=(6, 6), \
                    save=False, datestr_lens_mock='042825', lens_mode='unlensed', mode='photnoiseonly_maskJ16_ISL', nbar_tracer=2e4):
    


    baseMap = FlatMap(nX=nX, nY=nY, sizeX=sizeX*np.pi/180., sizeY=sizeY*np.pi/180.)
    
    all_fpaths = []

    for inst in inst_list:
        lens_dat = mock_lens_dat(inst, baseMap)
        lens_dat.prep_lens()

        for simidx in range(nsim):
        
            # total_signal, cib_maps,\
            #     kappa_maps, masks, count_fields
            dat_dict = lens_dat.gen_mock_set(plot=plot,\
                                          with_noise=with_noise, do_lensing=do_lensing,\
                                          make_lens_map=make_lens_map, scale_clkk=scale_clkk, \
                                          load_mask=load_mask, add_isl=add_isl, \
                                         with_read_noise=with_read_noise, datestr_lens_mock=datestr_lens_mock, \
                                         lens_mode=lens_mode, cib_setidx=simidx, nbar_tracer=nbar_tracer)

            plot_map(dat_dict['cib_maps'][0]*dat_dict['masks'][0], figsize=figsize_plot, title='Masked CIB')
            plot_map(dat_dict['total_signal'][0]*dat_dict['masks'][0], figsize=figsize_plot, title='Total signal (masked)')

            tmdir = mock_test_dat_dir+'/'+datestr_lens_mock+'/TM'+str(inst)

            if not os.path.isdir(tmdir):
                print('making directory ', tmdir)
                os.makedirs(tmdir)
                
            fpath_save = tmdir+'/mock_dat_'+lens_mode+'_'+mode+'_simidx'+str(simidx)+'.npz'
#             fpath_save = tmdir+'/mock_dat_'+lens_mode+'_photnoiseonly_extl2e5_scaleclkk='+str(scale_clkk)+'_wnoise_maskJ16_ISL_simidx'+str(simidx)+'.npz'
            all_fpaths.append(fpath_save)

            if save:
                print('Saving to ', fpath_save)
                np.savez(fpath_save, inst=inst, total_signal=dat_dict['total_signal'], \
                        cib_maps=dat_dict['cib_maps'], kappa_maps=dat_dict['all_kappa'],\
                         masks=dat_dict['masks'], galdens=dat_dict['count_fields'], tracer_cats=dat_dict['tracer_cats'])

    return all_fpaths


def compute_mode_loss_factor(L_bins, lMin, lMax, n_grid=1024):
    """
    Computes the L-dependent geometric mode loss factor f_mode(L) for a sharp band-pass filter.
    This represents the fraction of available small-scale mode pairs that can form a triangle with a large-scale mode L.
    """
    # Create a 2D grid for the small-scale mode l, extending far enough to handle the shift
    l_max_grid = lMax + np.max(L_bins)
    l_vals = np.linspace(-l_max_grid, l_max_grid, n_grid)
    lx, ly = np.meshgrid(l_vals, l_vals)
    l_mag = np.sqrt(lx**2 + ly**2)

    # Define the filter annulus (Theta(l))
    filter_annulus = (l_mag >= lMin) & (l_mag < lMax)
    
    # The denominator is the total area of the annulus (total number of available modes for one leg)
    denominator = np.sum(filter_annulus)
    if denominator == 0:
        return np.zeros_like(L_bins)


def forward_model_normalization(L_bins, lMin, lMax, C_unlensed, C_obs, B_ell=None, 
                                n_ell=200, verbose=True):
    """
    Forward model the QE normalization integral from first principles:
    
    N_L^{-1} = ∫ d²ℓ F(ℓ,L-ℓ) f^κ(ℓ,L-ℓ) B(ℓ)B(L-ℓ)
    
    where:
    - F(ℓ,L-ℓ) = C_unlensed(ℓ) C_unlensed(|L-ℓ|) / [C_obs(ℓ) C_obs(|L-ℓ|)]
    - f^κ(ℓ,L-ℓ) = [ℓ·(L-ℓ)]² / |L-ℓ|²  (lensing response)
    - B(ℓ), B(L-ℓ) = beam functions
    
    Parameters:
    -----------
    L_bins : array
        Output L values to compute normalization at
    lMin, lMax : float
        Integration range for ℓ
    C_unlensed : function
        Unlensed power spectrum C(ℓ)
    C_obs : function  
        Observed power spectrum (includes beam and noise)
    B_ell : function, optional
        Beam function B(ℓ). If None, assumes B=1
    n_ell : int
        Number of ℓ bins for integration
    
    Returns:
    --------
    N_L : array
        Normalization values at each L
    components : dict
        Dictionary with individual components for diagnostics
    """
    
    if B_ell is None:
        B_ell = lambda ell: np.ones_like(ell)
    
    # Create ℓ grid for integration (annulus from lMin to lMax)
    ell_edges = np.logspace(np.log10(lMin), np.log10(lMax), n_ell+1)
    ell_centers = 0.5 * (ell_edges[:-1] + ell_edges[1:])
    d_ell = np.diff(ell_edges)
    
    N_L_inv = np.zeros(len(L_bins))
    
    # Store components for analysis
    F_avg = np.zeros(len(L_bins))
    f_kappa_avg = np.zeros(len(L_bins))
    beam_sq_avg = np.zeros(len(L_bins))
    n_contributing_modes = np.zeros(len(L_bins))
    
    for i, L in enumerate(L_bins):
        integrand_sum = 0.0
        n_modes = 0
        
        F_sum = 0.0
        f_kappa_sum = 0.0
        beam_sq_sum = 0.0
        
        # Integrate over ℓ in annulus [lMin, lMax]
        # Use isotropic approximation: integrate over |ℓ| and average over angles
        n_phi = 32  # Angular bins
        
        for j, ell in enumerate(ell_centers):
            # Integration weight: ℓ dℓ dφ → ell * d_ell[j] * 2π/n_phi
            weight_ell = ell * d_ell[j] * 2*np.pi / n_phi
            
            # Integrate over angle φ (direction of ℓ)
            for k in range(n_phi):
                phi = 2*np.pi * k / n_phi
                
                # ell vector
                ell_x = ell * np.cos(phi)
                ell_y = ell * np.sin(phi)
                
                # L-ell vector
                L_minus_ell_x = L - ell_x
                L_minus_ell_y = -ell_y
                L_minus_ell = np.sqrt(L_minus_ell_x**2 + L_minus_ell_y**2)
                
                # Check if |L-ell| is in valid range
                if L_minus_ell < lMin or L_minus_ell > lMax:
                    continue
                
                n_modes += 1
                
                # Compute components
                # F(ell,L-ell)
                F = C_unlensed(ell) * C_unlensed(L_minus_ell) / (C_obs(ell) * C_obs(L_minus_ell))
                
                # f^κ(ℓ,L-ℓ) = [ℓ·(L-ℓ)]² / |L-ℓ|²
                # ℓ·(L-ℓ) = ell_x*(L-ell_x) + ell_y*(-ell_y) = ell_x*L - ell_x² - ell_y²
                ell_dot_Lminusell = ell_x * L_minus_ell_x + ell_y * L_minus_ell_y
                f_kappa = (ell_dot_Lminusell)**2 / (L_minus_ell**2 + 1e-20)
                
                # Beam: B(ℓ) B(|L-ℓ|)
                beam_product = B_ell(ell) * B_ell(L_minus_ell)
                
                # Full integrand
                integrand = F * f_kappa * beam_product * weight_ell
                integrand_sum += integrand
                
                # Track components
                F_sum += F * weight_ell
                f_kappa_sum += f_kappa * weight_ell  
                beam_sq_sum += beam_product * weight_ell
        
        N_L_inv[i] = integrand_sum
        
        if n_modes > 0:
            # Average component values (for diagnostics)
            norm_factor = n_modes * d_ell[0] * 2*np.pi / n_phi
            F_avg[i] = F_sum / norm_factor if norm_factor > 0 else 0
            f_kappa_avg[i] = f_kappa_sum / norm_factor if norm_factor > 0 else 0
            beam_sq_avg[i] = beam_sq_sum / norm_factor if norm_factor > 0 else 0
            n_contributing_modes[i] = n_modes
    
    # Convert to normalization N_L
    N_L = 1.0 / (N_L_inv + 1e-30)
    N_L[N_L_inv == 0] = 0
    
    components = {
        'F_avg': F_avg,
        'f_kappa_avg': f_kappa_avg, 
        'beam_sq_avg': beam_sq_avg,
        'n_modes': n_contributing_modes,
        'N_L_inv': N_L_inv
    }
    
    if verbose:
        print("\nForward model normalization computed:")
        print(f"  L range: [{L_bins.min():.1f}, {L_bins.max():.1f}]")
        print(f"  N_L range: [{N_L[N_L>0].min():.3e}, {N_L.max():.3e}]")
        print(f"  Contributing modes: [{n_contributing_modes.min():.0f}, {n_contributing_modes.max():.0f}]")
    
    return N_L, components


def compute_mode_loss_factor(L_bins, lMin, lMax, n_grid=1024):
    """
    Computes the L-dependent geometric mode loss factor f_mode(L) for a sharp band-pass filter.
    This represents the fraction of available small-scale mode pairs that can form a triangle with a large-scale mode L.
    """
    # Create a 2D grid for the small-scale mode l, extending far enough to handle the shift
    l_max_grid = lMax + np.max(L_bins)
    l_vals = np.linspace(-l_max_grid, l_max_grid, n_grid)
    lx, ly = np.meshgrid(l_vals, l_vals)
    l_mag = np.sqrt(lx**2 + ly**2)

    # Define the filter annulus (Theta(l))
    filter_annulus = (l_mag >= lMin) & (l_mag < lMax)
    
    # The denominator is the total area of the annulus (total number of available modes for one leg)
    denominator = np.sum(filter_annulus)
    if denominator == 0:
        return np.zeros_like(L_bins)

    f_mode_L = np.zeros_like(L_bins, dtype=float)
    for i, L in enumerate(L_bins):
        # The second leg is |L - l|. We shift its annulus by (L, 0) without loss of generality.
        lx_shifted = lx - L
        l_mag_shifted = np.sqrt(lx_shifted**2 + ly**2)
        
        filter_annulus_shifted = (l_mag_shifted >= lMin) & (l_mag_shifted < lMax)
        
        # The numerator is the area of the intersection of the two annuli
        # (the number of mode pairs where BOTH legs are in the band)
        numerator = np.sum(filter_annulus & filter_annulus_shifted)
        
        f_mode_L[i] = numerator / denominator
        
    return f_mode_L


# def calc_filters_and_corrections(clf, params, c_i_shot):

def save_current_plot(fig, tag, sim_idx, save_intermediate_plots=False, intermediate_plot_dir=None):
    if not save_intermediate_plots:
        return
    fpath = os.path.join(intermediate_plot_dir, f"sim{sim_idx:03d}_{tag}.png")

    print('saving to ', fpath)
    fig.savefig(fpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved intermediate plot:", fpath)

def delta_fn_sources_test(nsim = 2, MAP_SIZE = 1024, 
                         N_CIB_PER_PIXEL = 0.2,  # Average 1 source every 20 pixels
                        N_G_PER_PIXEL = 0.2, 
                          Apix=1.15e-9, sizeX=2., sizeY = 2.,
                          lMin=10000, lMax = 80000, 
                         add_noise=False, sigma_noise_pix=25, psf_pix_fwhm=None, 
                         plot=False, alpha=0., apply_mask=False,
                         grab_cib_sim=False, datestr='042725', ciber_inst=1, 
                         lensmode='randomized', ifield=4, n_cib_sim=5, 
                         pixel_fn_correct=False, mockstr='JHlt16_nbar100000.0',
                         use_beam_in_norm=True, skew_filter_mode='bandpass', 
                         s_max=100.0, verbose=1,
                         save_intermediate_plots=False,
                         intermediate_plot_dir=None):

    def vprint(*args, level=1, **kwargs):
        if verbose >= level:
            print(*args, **kwargs)

    make_intermediate_plots = bool(save_intermediate_plots)
    if save_intermediate_plots:
        if intermediate_plot_dir is None:
            intermediate_plot_dir = os.path.join("res", "intermediate_plots")
        os.makedirs(intermediate_plot_dir, exist_ok=True)

    
    rng = np.random.default_rng(seed=123) # Use same seed for consistency

    baseMap = FlatMap(nX=MAP_SIZE, nY=MAP_SIZE, sizeX=sizeX*np.pi/180., sizeY=sizeY*np.pi/180.)
    param_dict = dict({'nX':MAP_SIZE, 'nY':MAP_SIZE, 'sizeX':sizeX, 'sizeY':sizeY, 'lMin':lMin, 'lMax':lMax, 'nBins':50, 
                         'ciber_inst':ciber_inst, 'psf_pix_fwhm':psf_pix_fwhm, 'sigma_noise_pix':sigma_noise_pix, 'Apix':Apix})

    config_dict = dict({'lensmode':lensmode, 'add_noise':add_noise, 
                        'grab_cib_sim':grab_cib_sim, 'pixel_fn_correct':pixel_fn_correct, 
                        'apply_mask':apply_mask, 'mode':'qe_kappa_norm', 'cut_lxly':False,
                        'skew_filter_mode':skew_filter_mode,
                        'verbose': verbose})  # 'bandpass' or 'wiener'

    clf = ciber_lens_forecast(ell_min=1, ell_max=2.*lMax)
    
    clgg, clII, cl_bis,\
        clkg, dclkg, cl_bis_WF, dclkg_WF, \
            N_L_kappa, N_L_err_kappa = [np.zeros((nsim, 50)) for _ in range(9)]
    
    param_dict['pixel_size_arcsec'] = 3600.*(sizeX/MAP_SIZE)
    vprint('pixel size:', param_dict['pixel_size_arcsec'])
    param_dict['l_nyquist'] = np.pi / (param_dict['pixel_size_arcsec'] / 206265.0)
    vprint('ell Nyquist:', param_dict['l_nyquist'])

    if param_dict['psf_pix_fwhm'] is not None:
        param_dict['sigma_b'] = psf_pix_fwhm_to_sigma_rad(param_dict['pixel_size_arcsec'], param_dict['psf_pix_fwhm'])
        B_ell_fn = lambda ell: np.exp(-(ell*param_dict['sigma_b'])**2/2)
        clf.bl = B_ell_fn  # Assign Gaussian beam to clf for use in corrections
    elif grab_cib_sim:
        clf.load_bl(ciber_inst, ifield, inplace=True, plot=False)
        B_ell_fn = clf.bl
    else:
        # No beam case
        B_ell_fn = None
        clf.bl = lambda ell: np.ones_like(ell)  # Unity beam

    for x in range(nsim):

        vprint(f"\n--- [sim {x+1}/{nsim}] starting ---", level=1)

        simidx = x % n_cib_sim
        if grab_cib_sim:
            tmdir = '../data/lens_prods/mock_dat/'+datestr+'/TM'+str(ciber_inst)
            mock_sim_fpath = tmdir+ '/mock_dat_'+lensmode+'_'+mockstr+'_simidx'+str(simidx)+'.npz'
        else:
            mock_sim_fpath = None

        vprint('[sim %d] loading from ' % x, mock_sim_fpath)
        # This map represents the true sky intensity I(x, y)
        cib_intensity_map, all_cib_fluxes, counts_map, mask = generate_cib_map(
            map_size=MAP_SIZE,
            n_cib_per_pixel=N_CIB_PER_PIXEL, n_gal_per_pixel=N_G_PER_PIXEL,
            seed=None, s_max=s_max,
        mock_sim_fpath=mock_sim_fpath, apply_mask=apply_mask)

        vprint(f'[sim {x}] generated CIB map: {len(all_cib_fluxes)} sources')
        vprint('mean mask is ', np.mean(mask))
        unmask_frac = np.mean(mask)

        cib_smooth = cib_intensity_map.copy()

        cib_intensity_map -= np.mean(cib_intensity_map)
        cibFourier = baseMap.fourier(cib_intensity_map)

        if psf_pix_fwhm is not None:
            if psf_pix_fwhm > 0:
                cib_smooth = gaussian_filter(cib_intensity_map, sigma=psf_pix_fwhm/2.355)
        
        if x==0 and make_intermediate_plots:
            fig = plot_map(cib_smooth, figsize=(5, 5), title='CIB map', show=False, return_fig=True)
            save_current_plot(fig, "cib_map", x, save_intermediate_plots=save_intermediate_plots, intermediate_plot_dir=intermediate_plot_dir)

        
        if grab_cib_sim:
            N_G_PER_PIXEL = len(all_cib_fluxes)/MAP_SIZE**2
            N_CIB_PER_PIXEL = len(all_cib_fluxes)/MAP_SIZE**2
            galaxy_fluxes = all_cib_fluxes.copy()
            
        else:
            n_cib_total = len(all_cib_fluxes)
            n_g_total = int( (N_G_PER_PIXEL / N_CIB_PER_PIXEL) * n_cib_total )
            # Randomly choose which CIB sources are also our tracer galaxies
            galaxy_indices = rng.choice(n_cib_total, size=n_g_total, replace=False)
            galaxy_fluxes = all_cib_fluxes[galaxy_indices]


        vprint(f'[sim {x}] computing analytic bias terms')
        # --- 4. Calculate the Analytic Bias ---
        analytic_bias, c_i_shot, c_i2_g_shot, galshot, trispec_noise_cib, trispec_noise_g = calculate_analytic_bias(
            n_cib_per_pixel=N_CIB_PER_PIXEL,
            n_g_per_pixel=N_G_PER_PIXEL,
            fluxes_cib=all_cib_fluxes,
            fluxes_g=galaxy_fluxes,
            pix_area=Apix
        )

        param_dict['c_i_shot'] = c_i_shot 
        param_dict['c_i2_g_shot'] = c_i2_g_shot
        param_dict['galshot'] = galshot
        param_dict['trispec_noise_cib'] = trispec_noise_cib
        param_dict['trispec_noise_g'] = trispec_noise_g
        param_dict['analytic_bias'] = analytic_bias

        vprint('cI shot, cI2 shot, galshot, analytic bias:', c_i_shot, c_i2_g_shot, galshot, analytic_bias)
        path_k = 'example_kappa.npz'
        # mask = np.ones_like(cib_intensity_map)    
        

        if add_noise:
            vprint(f'[sim {x}] adding noise, sigma_noise_pix={sigma_noise_pix}')
            noise = np.random.normal(0, sigma_noise_pix, cib_intensity_map.shape)
            if make_intermediate_plots:
                fig = plot_map(noise, figsize=(5, 5), title='noise', show=False, return_fig=True)
                save_current_plot(fig, "noise", x, save_intermediate_plots=save_intermediate_plots, intermediate_plot_dir=intermediate_plot_dir)
            
            lb, clnoise, clerrnose = get_power_spec(noise, nbins=26)
            vprint('mean nell:', np.mean(clnoise))
            obs_map = cib_smooth + noise
            param_dict['clnoise'] = clnoise
            
        else:
            obs_map = cib_smooth*np.ones_like(cib_smooth)

        if apply_mask:
            vprint(f'[sim {x}] applying mask')
            obs_map *= mask
            obs_map[obs_map != 0] -= np.mean(obs_map[obs_map != 0])
            if make_intermediate_plots:
                fig = plot_map(obs_map, figsize=(6, 6), title='masked map mean sub', show=False, return_fig=True)
                save_current_plot(fig, "masked_obs", x, save_intermediate_plots=save_intermediate_plots, intermediate_plot_dir=intermediate_plot_dir)

        vprint(f'[sim {x}] computing filters and corrections')
        fns, facs = calc_filters_and_corrections(clf, param_dict, config_dict)

        # Extract filter functions
        cib_unlensed_auto = fns['cib_unlensed_auto']
        obs_auto = fns['obs_auto']
        W_ell = fns['W_ell']

        # Apply mode fraction correction to vbeam
        facs['vbeam'] *= facs['modefrac']
        
        # Use kcorr for beam correction (not in normalization)

        vprint('kcorr is ', facs['kcorr'])
        vprint('vbeam is ', facs['vbeam'])
        vprint('mode frac is ', facs['modefrac'])

        vprint('obs map has mean:', np.mean(obs_map))
        dataFourier = baseMap.fourier(obs_map)

        # if pixel_fn_correct:
            # dataFourier /= baseMap.pixelWindow(baseMap.lx, baseMap.ly)

        ell_pl = np.logspace(1, 5, 100)
        
        if make_intermediate_plots:
            fig = plt.figure(figsize=(5, 4))
            plt.plot(ell_pl, cib_unlensed_auto(ell_pl), label='unlensed')
            plt.plot(ell_pl, obs_auto(ell_pl), label='obs')
            plt.plot(ell_pl, obs_auto(ell_pl)-cib_unlensed_auto(ell_pl), label='difference', linestyle='dashed', color='k')
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('$\\ell$', fontsize=14)
            plt.ylabel('$C_{\\ell}$', fontsize=14)
            plt.legend()
            save_current_plot(fig, "cl_components", x, save_intermediate_plots=save_intermediate_plots, intermediate_plot_dir=intermediate_plot_dir)
            plt.close()
            # if plot:
            #     plt.show()

        # Beam correction strategy: 
        # OLD WAY (use_beam_in_norm=False): b_ell_use=None, use external kcorr/vbeam corrections
        # NEW WAY (use_beam_in_norm=True): b_ell_use=B_ell_fn, beam properly applied in normalization
        if use_beam_in_norm:
            b_ell_use = B_ell_fn  # Pass beam to QE normalization (gives B(ell)×B(L-ell))
            kcorr_use = 1.0  # No additional beam correction needed
            vprint("Using beam in normalization (proper B(ell)×B(L-ell) correction)")
            vprint("  -> Setting kcorr=1.0 (no additional correction)")
        else:
            b_ell_use = None  # Use external kcorr/vbeam corrections
            kcorr_use = facs['kcorr']  # Apply beam correction in post-processing
            vprint("Using old method with external kcorr/vbeam corrections")
            vprint(f"  -> Using kcorr={kcorr_use:.6f}")
            
        # Organize into dictionaries for cleaner function calls
        map_dict = dict({'counts_map':counts_map, 'mask':mask, 'cib_intensity_map':cib_intensity_map, 
                        'obs_map':obs_map, 'cibFourier':cibFourier, 'kappa_true':cib_intensity_map})
        
        cl_fns = dict({'cib_unlensed_auto':cib_unlensed_auto, 'obs_auto':obs_auto, 
                      'W_ell':W_ell, 'B_ell':b_ell_use})
        
        corr_facs = dict({'kcorr':kcorr_use, 'vbeam':facs['vbeam'], 
                         'unmask_frac':unmask_frac, 'modefrac':facs['modefrac']})

        vprint(f'[sim {x}] computing lensing power spectrum quantities')
        psres = compute_lensing_ps_quantities_v2(baseMap, map_dict, cl_fns, param_dict,
                                                config_dict, corr_facs,
                                                save_intermediate_plots=save_intermediate_plots,
                                                intermediate_plot_dir=intermediate_plot_dir)
        vprint(f'[sim {x}] done computing power spectra')

        # Plot QE normalization N_L for diagnostics
        if make_intermediate_plots and psres['N_L'] is not None:
            fig = plot_normalization(psres['lC'], psres['N_L'], psres['N_L_err'],
                             lMin=param_dict['lMin'], lMax=param_dict['lMax'], show=False)
            save_current_plot(fig, "qe_normalization", x, save_intermediate_plots=save_intermediate_plots, intermediate_plot_dir=intermediate_plot_dir)

        clgg[x] = psres['clgg']
        clII[x] = psres['clII']    
        cl_bis[x] = psres['cl_bis']
        clkg[x] = psres['clkg']
        dclkg[x] = psres['dclkg']

        N_L_kappa[x] = psres['N_L']
        N_L_err_kappa[x] = psres['N_L_err']


        # clkg[x] /= P_ell_sq(psres['lC'])
        # dclkg[x] /= P_ell_sq(psres['lC'])
        # cl_bis[x] /= P_ell_sq(psres['lC'])

        
    res = {'clgg':clgg, 'clII':clII, 'cl_bis':cl_bis, 'clkg':clkg, 'dclkg':dclkg, 'lC':psres['lC'], 
          'analytic_bias':analytic_bias, 'c_i_shot':c_i_shot, 'c_i2_g_shot':c_i2_g_shot, 'galshot':galshot, 
          'N_L_kappa':N_L_kappa, 'N_L_err_kappa':N_L_err_kappa,
          'clkg_bias_predicted_2h': psres.get('clkg_bias_predicted_2h', None),
          'C_ell_II': psres.get('C_ell_II', None),
          'C_ell_gg': psres.get('C_ell_gg', None),
          'C_ell_Ig': psres.get('C_ell_Ig', None)}
    
    return res

def compute_normalization(baseMap, lC, norm_Fourier):
    l2d = baseMap.l
    l_flat = l2d.flatten()
    norm_phi_flat = norm_Fourier.flatten()

    # Compute bin edges from lC (assuming log spacing)
    if len(lC) > 1:
        # Reconstruct bin edges from bin centers
        log_lC = np.log10(lC)
        dlog = log_lC[1] - log_lC[0]
        lBins_norm = 10**(np.concatenate([
            [log_lC[0] - dlog/2],
            log_lC + dlog/2
        ]))
    else:
        # Fallback to single bin
        lBins_norm = np.array([lC[0] * 0.9, lC[0] * 1.1])

    # Bin the normalization values by L using same bins as power spectra
    # First extract N_L^φ from the 2D field
    N_L_phi = np.zeros(len(lC))
    N_L_phi_err = np.zeros(len(lC))
    
    for i in range(len(lC)):
        mask_bin = (l_flat >= lBins_norm[i]) & (l_flat < lBins_norm[i+1])
        if np.sum(mask_bin) > 0:
            vals = np.abs(norm_phi_flat[mask_bin])
            N_L_phi[i] = np.median(vals)
            N_L_phi_err[i] = np.std(vals) / np.sqrt(np.sum(mask_bin))
        else:
            N_L_phi[i] = np.nan
            N_L_phi_err[i] = np.nan
    
    # Convert to N_L^κ using: N_L^κ = (L²/2)² × N_L^φ = L⁴/4 × N_L^φ
    N_L = 0.25 * lC**4 * N_L_phi
    N_L_err = 0.25 * lC**4 * N_L_phi_err

    return N_L, N_L_err

def compute_lensing_ps_quantities_v2(baseMap, map_dict, cl_fns, param_dict, config_dict, corr_facs,
                                     save_intermediate_plots=False, intermediate_plot_dir=None):
    """
    Compute lensing power spectrum quantities using dictionary-based inputs.

    Parameters:
    -----------
    baseMap : FlatMap
        The flat map object
    map_dict : dict
        Dictionary containing maps: 'obs_map', 'counts_map', 'mask', 'cib_intensity_map',
        'kappa_true', 'cibFourier' (optional)
    cl_fns : dict
        Dictionary containing functions: 'cib_unlensed_auto', 'obs_auto', 'W_ell', 'B_ell'
    param_dict : dict
        Dictionary with parameters: 'lMin', 'lMax', 'Apix', etc.
    config_dict : dict
        Dictionary with configuration: 'apply_mask', 'cut_lxly', 'mode'
    corr_facs : dict
        Dictionary with correction factors: 'kcorr', 'vbeam', 'unmask_frac'
    save_intermediate_plots : bool
        Whether to save intermediate diagnostic plots
    intermediate_plot_dir : str
        Directory for saving plots
    """
    
    path_k = 'example_kappa.npz'
    
    print('obs map in compute_lensing_ps_quantities_v2 is ', np.mean(map_dict['obs_map']))
    dataFourier = baseMap.fourier(map_dict['obs_map'])

    # Extract beam function (if provided) for QE normalization
    fB_ell = cl_fns.get('B_ell', None)
    
    resultFourier, norm_Fourier = run_kappa_est(baseMap, cl_fns['cib_unlensed_auto'], 
                                               cl_fns['obs_auto'], 
                                               param_dict, dataFourier=dataFourier, test=False,
                                               path=path_k, cut_lxly=config_dict['cut_lxly'], 
                                               mode=config_dict['mode'], fB_ell=fB_ell)

    kFourier_est = baseMap.loadDataFourier(path_k)
    kappa_est_map_raw = baseMap.inverseFourier(kFourier_est).real

    # Get lC bins from power spectrum computation (this uses nBins parameter, typically 50)
    lC, clkk, clkkerr = compute_map_ps(baseMap, 
                                        map_dict['kappa_true'], 
                                        map_dict['mask'], 
                                        apply_mask=config_dict['apply_mask'])
    

    # Extract QE normalization N_L for diagnostics
    # IMPORTANT: We want the actual N_L values (radial profile), not the power spectrum!
    # norm_Fourier returned by computeQuadEstPhiNormalizationFFT is N_L^φ (for lensing potential)
    # But we need N_L^κ (for convergence): N_L^κ = (L^2/2)^2 × N_L^φ = L^4/4 × N_L^φ

    if norm_Fourier is not None:
        N_L, N_L_err = compute_normalization(baseMap, lC, norm_Fourier)        
    else:
        N_L, N_L_err = None, None

    # Process input maps
    kappa_est_map = proc_input_map(kappa_est_map_raw, map_dict['mask'], 
                                   galdens=False, apply_mask=config_dict['apply_mask'])
    kappa_true = proc_input_map(map_dict['kappa_true'], map_dict['mask'], 
                               galdens=False, apply_mask=config_dict['apply_mask'])
    galdens = proc_input_map(map_dict['counts_map'], map_dict['mask'], 
                            galdens=True, apply_mask=config_dict['apply_mask'])

    # Fourier transform processed maps
    kFourier_est_mask, kappa_true_Fourier, kFourier_gal_mask = [baseMap.fourier(mapex) 
                                                                 for mapex in [kappa_est_map, kappa_true, galdens]]

    # CRITICAL: Filter kappa estimate to valid L range
    # QE normalization is only valid for L < lMax (not 2*lMax)
    # Beyond this, mode coupling in normalization integral becomes unreliable
    L_max_valid = param_dict['lMax']
    f_L_filter = lambda l: (l <= L_max_valid)
    kFourier_est_filtered = baseMap.filterFourierIsotropic(f_L_filter, dataFourier=kFourier_est_mask, test=False)
    
    print(f"Filtering kappa estimate to L <= {L_max_valid}")

    # Compute power spectra
    lC, clkk, clkkerr = baseMap.powerSpectrum(kappa_true_Fourier)
    lC, clgg, clggerr = baseMap.powerSpectrum(kFourier_gal_mask)
    lC, clx, clxerr = baseMap.crossPowerSpectrum(kFourier_est_filtered, kappa_true_Fourier, plot=False)
    lC, clkg, clkgerr = baseMap.crossPowerSpectrum(kFourier_est_filtered, kFourier_gal_mask, plot=False)

    def f(l):
        # cut off the high ells from input map
        if (l < param_dict['lMin']) or (l > param_dict['lMax']):
            return 0.
        result = divide(cl_fns['cib_unlensed_auto'](l), cl_fns['obs_auto'](l))
        if not np.isfinite(result):
            result = 0.
        return result

    def bandpass(l):
        # cut off the high ells from input map
        if (l > param_dict['lMax']) or (l < param_dict['lMin']):
            return 0.0
        else:
            return 1.0

    # Compute CIB power spectrum
    if map_dict.get('cibFourier') is not None:
        print('already have cibFourier..')
        f_filt = lambda l: (l <= param_dict['lMax'])*(l >= param_dict['lMin'])
        # iVarDataFourier = baseMap.filterFourierIsotropic(f_filt, dataFourier=map_dict['cibFourier'], test=False)        
        # lC, clII, clIIerr = baseMap.powerSpectrum(iVarDataFourier)
        lC, clII, clIIerr = baseMap.powerSpectrum(map_dict['cibFourier'])

    else:        
        iVarDataFourier = baseMap.filterFourierIsotropic(f, dataFourier=dataFourier, test=False)        
        lC, clII, clIIerr = baseMap.powerSpectrum(iVarDataFourier)

    # Compute bispectrum based on filter mode
    # Two modes supported:
    #   'bandpass': Simple hard cutoff at [lMin, lMax], matches with W_ell=None in vbeam
    #   'wiener': Wiener-filtered before squaring, matches with W_ell² in vbeam
    skew_filter_mode = config_dict['skew_filter_mode']
    
    if skew_filter_mode == 'bandpass':
        print("Computing bispectrum with BANDPASS filter (hard cutoff)")
        # Pre-filter the map with hard bandpass
        iVarCIBFourier = baseMap.filterFourierIsotropic(bandpass, dataFourier=dataFourier, test=False)        
        cib_filtered = baseMap.inverseFourier(iVarCIBFourier)
        
        lC, cl_bis, sCl_bis = compute_skew_cl_I2G_simp(baseMap, 
                                                        cib_filtered,
                                                        kFourier_gal_mask,
                                                        lMin=param_dict['lMin'],
                                                        lMax=param_dict['lMax'],
                                                        filter_mode='bandpass',
                                                        W_ell=None)
        
        # DEBUG: Check filtered map and raw bispectrum properties  
        print(f"MOCK BISPECTRUM MAP DEBUG:")
        print(f"  Filtered map mean: {cib_filtered.mean():.6e}")
        print(f"  Filtered map std: {cib_filtered.std():.6e}")
        print(f"  Bandpass range: [{param_dict['lMin']}, {param_dict['lMax']}]")
        print(f"MOCK RAW BISPECTRUM DEBUG:")
        print(f"  cl_bis range before corrections: [{cl_bis.min():.6e}, {cl_bis.max():.6e}]")
        print(f"  sCl_bis range before corrections: [{sCl_bis.min():.6e}, {sCl_bis.max():.6e}]")
        print(f"  lC range: [{lC.min():.1f}, {lC.max():.1f}]")
    elif skew_filter_mode == 'wiener':
        print("Computing bispectrum with WIENER filter (C_unl/C_obs weighting)")
        # Use unfiltered map; Wiener filtering happens inside compute_skew_cl_I2G_simp
        lC, cl_bis, sCl_bis = compute_skew_cl_I2G_simp(baseMap, 
                                                        map_dict['obs_map'],
                                                        kFourier_gal_mask,
                                                        lMin=param_dict['lMin'],
                                                        lMax=param_dict['lMax'],
                                                        filter_mode='wiener',
                                                        W_ell=cl_fns.get('W_ell', None))
    else:
        raise ValueError(f"Unknown skew_filter_mode: {skew_filter_mode}")

    # Apply beam correction to CIB auto-power spectrum
    # The bispectrum has beam factor vbeam(L) = ⟨B(ℓ)B(|L-ℓ|)⟩
    # The auto-power has beam factor B(ℓ)²
    # These are DIFFERENT, so we must deconvolve both to get true (beam-free) quantities
    # for the bias formula to work: ΔC = C^{I²g,true} × A / (2 × C^{II,true})
    
    if config_dict['grab_cib_sim']:
        # Use CIBER beam - load it here
        print('Loading CIBER beam..')
        clf_temp = ciber_lens_forecast(ell_min=1, ell_max=2.*param_dict['lMax'])
        clf_temp.load_bl(param_dict['ciber_inst'], 4, inplace=True, plot=False)
        B_ell_for_bis = clf_temp.bl
        
        # Always deconvolve clII by B(ℓ)² to get true C^{II}
        clII /= clf_temp.bl(lC)**2
        clIIerr /= clf_temp.bl(lC)**2
        print("Deconvolving clII by B(ℓ)² to get true C^{II}")
            
    elif param_dict.get('psf_pix_fwhm') is not None:
        # Use Gaussian beam
        beam_vals = gaussian_beam_window(lC, param_dict['psf_pix_fwhm']*param_dict['pixel_size_arcsec'])
        B_ell_for_bis = lambda ell: gaussian_beam_window(ell, param_dict['psf_pix_fwhm']*param_dict['pixel_size_arcsec'])
        
        # Always deconvolve clII
        clII /= beam_vals**2
        clIIerr /= beam_vals**2
        print("Deconvolving clII by B(ell)² to get true C^{II}")
    else:
        B_ell_for_bis = None

    # Apply corrections to bispectrum and cross-spectrum
    # Compute L-dependent beam correction for bispectrum
    # CRITICAL: W_ell weighting in vbeam MUST match the filtering mode used in bispectrum!

    print('Computing L-dependent bispectrum beam correction...')

    if skew_filter_mode == 'bandpass':
        # Simple bandpass: compute mode-overlap integral over ell ∈ [lMin, lMax]
        # using corrected function that properly implements f_Θ(L)
        vbeam_L = effective_beam_skew_I2g_correct(
            lC, param_dict['lMin'], param_dict['lMax'],
            B_ell_fn=B_ell_for_bis, W_ell_fn=None
        )


    elif skew_filter_mode == 'wiener':
        # Wiener filter: both I(ell) and I(L-ell) are weighted by W(ell)
        # So the bispectrum has effective weight W(ell) × W(L-ell)
        # For the integral, if W is applied to both legs: use W² in single-leg integral
        print("  -> Using W_ell² (Wiener filter applied to both legs)")
        W_ell_fn = cl_fns.get('W_ell', None)
        if W_ell_fn is not None:
            def W_ell_sq(ell):
                return W_ell_fn(ell)**2
            vbeam_L = effective_beam_skew_I2g_correct(
                lC, param_dict['lMin'], param_dict['lMax'],
                flat_sky=True, B_ell_fn=B_ell_for_bis, W_ell_fn=W_ell_sq
            )
        else:
            print("  WARNING: W_ell not available, using no weighting")
            vbeam_L = effective_beam_skew_I2g_correct(
                lC, param_dict['lMin'], param_dict['lMax'],
                flat_sky=True, B_ell_fn=B_ell_for_bis, W_ell_fn=None
            )

    # Apply mode fraction to L-dependent array
    vbeam_L_corrected = vbeam_L * corr_facs['modefrac']
    # vbeam_L_corrected = vbeam_L

    print(f'L-dependent vbeam range: [{vbeam_L_corrected.min():.6f}, {vbeam_L_corrected.max():.6f}]')
    # print(f'Scalar vbeam for comparison: {corr_facs["vbeam"]:.6f}')
    print('vbeam before modefrac correction:', vbeam_L)
    print('modefrac in correction:', corr_facs['modefrac'])
    # Always divide bispectrum by vbeam(L) to get true C^{I²g}
    # This is independent of whether beam is in QE normalization or not
    # The bias formula requires TRUE (beam-free) spectra
    vbeam_use = vbeam_L_corrected
    print("Dividing bispectrum by vbeam(L) to get true C^{I²g}")
    
    # Apply corrections
    # unmask_frac is passed separately for clarity
    # NOTE: Bispectrum needs vbeam (L-dependent or scalar), not B_ell(L)!
    # cl_bis involves ∫ B(ell) B(L-ell) which is L-dependent
    cl_bis, sCl_bis = proc_skewspec(lC, cl_bis, sCl_bis, B_ell=None,  # Beam via vbeam_use
                                   vbeam=vbeam_L_corrected,  # 1.0 if beam in norm, else L-dependent
                                   unmask_frac=corr_facs['unmask_frac'])
    
    # Note: No beam correction applied to clkg since QE estimator is beam-independent
    clkg, clkgerr = proc_clkg(lC, clkg, clkgerr, B_ell=None,
                             kcorr=corr_facs['kcorr'], unmask_frac=corr_facs['unmask_frac'])

    # DEBUG: Plot L-dependent corrections to skew spectrum
    if save_intermediate_plots:
        fig, ax = plt.subplots(figsize=(5, 4))

        # Plot 1: vbeam_L before and after modefrac
        ax.loglog(lC, vbeam_L, label='vbeam_L (before modefrac)', marker='o', markersize=3, alpha=0.7)
        ax.axhline(corr_facs['modefrac'], color='r', linestyle='--', label=f'modefrac={corr_facs["modefrac"]:.6f}')
        ax.loglog(lC, vbeam_L_corrected, label='vbeam_L_corrected', marker='s', markersize=3, alpha=0.7)
        ax.set_xlabel('L')
        ax.set_ylabel('Correction factor')
        ax.legend()
        ax.set_title('L-dependent mode-overlap and modefrac correction')
        ax.grid(alpha=0.3)


        plt.tight_layout()
        save_current_plot(fig, "skew_corrections_diagnostic", 0, save_intermediate_plots=save_intermediate_plots,
                         intermediate_plot_dir=intermediate_plot_dir)

    # Create dedicated vbeam(L) figure showing mode-overlap fraction
    if save_intermediate_plots:
        fig = plt.figure(figsize=(10, 7))

        # Main panel: f_Θ(L) with reference annotations
        ax = plt.gca()
        ax.semilogx(lC, vbeam_L, label='$f_\\Theta(L)$ (mode-overlap fraction)',
                   marker='o', markersize=5, alpha=0.85, linewidth=2.5, color='darkblue')
        ax.axhline(1.0, color='k', linestyle='--', alpha=0.3, linewidth=1.5, label='No geometric constraint')
        ax.axvline(param_dict['lMin'], color='green', linestyle=':', alpha=0.5, linewidth=2, label=f'$\\ell_{{\\min}}$ = {param_dict["lMin"]:.0f}')
        ax.axvline(param_dict['lMax'], color='red', linestyle=':', alpha=0.5, linewidth=2, label=f'$\\ell_{{\\max}}$ = {param_dict["lMax"]:.0f}')

        ax.set_xlabel('Large-scale multipole $L$', fontsize=14, fontweight='bold')
        ax.set_ylabel('$f_\\Theta(L)$', fontsize=14, fontweight='bold')
        ax.set_title('Mode-overlap fraction for bispectrum (shear.tex eq. 96-105)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        ax.set_ylim([0, 1.05])
        ax.legend(fontsize=11, loc='best')

        # Add text box with explanation
        plt.tight_layout()
        save_current_plot(fig, "vbeam_L_mode_overlap", 0, save_intermediate_plots=save_intermediate_plots,
                         intermediate_plot_dir=intermediate_plot_dir)

    # Calculate shot noise level and bias using BEAM-CORRECTED spectra
    # Both clII and cl_bis are now beam-deconvolved
    clII_shot = np.mean(clII[(lC > param_dict['lMin']) * (lC < param_dict['lMax'])])
    print('clII shot:', clII_shot)

    # Bias formula: ΔC_L = C_L^{I²g} × A / (2 × C^{II})
    # Theory uses SCALAR bandpass-averaged C^{II}, NOT L-dependent spectrum
    # Both clII_shot and cl_bis are now beam-free, so ratio is correct
    clkg_bias = cl_bis * param_dict['Apix'] / (2 * clII_shot)

    # Apply mask correction to galaxy power spectrum
    if config_dict['apply_mask']:
        clgg /= corr_facs['unmask_frac']
        clggerr /= corr_facs['unmask_frac']

    psres = {'lC':lC, 'clkk':clkk, 'clkkerr':clkkerr, 'clgg':clgg, 'clggerr':clggerr, 
            'clII':clII, 'clIIerr':clIIerr, 'clx':clx, 'clxerr':clxerr, 
            'clkg':clkg, 'clkgerr':clkgerr, 'cl_bis':cl_bis, 'sCl_bis':sCl_bis, 
            'dclkg':clkg_bias, 'kappa_est_map':kappa_est_map, 'kappa_true':kappa_true, 'galdens':galdens,
            'N_L':N_L, 'N_L_err':N_L_err}

    return psres


def run_lens_recover(inst, nsim=5, scale_clkk=1.0, ifield_list=[4, 6, 7, 8], mock_dat_datestr='042825', \
                    lMin=10000, lMax=1e5, lensmode='unlensed', plot=False, cut_lxly=False, smooth_sigma=20, 
                    nbar_tracer=2e4, mockstr='photnoiseonly_maskJ16', plot_recov_true_maps=True, Apix = 1.15e-9, 
                    apply_mask=True):
    
    
    all_clx, all_clxerr,\
        all_clkk, all_clkkerr,\
            all_clkg, all_clkgerr, \
                all_clgg, all_clggerr, \
                    all_clskew, all_clskewerr, \
                        all_clII, all_clIIerr, \
                        all_clkgbias, all_clskew_WF,\
                             all_clkgbias_WF = [np.zeros((nsim, len(ifield_list), 50)) for x in range(15)]
    
    # Initialize arrays to store N_L normalization (will be filled on first iteration)
    all_N_L, all_N_L_err = [None, None]
    lC_norm_stored = None

    for simidx in range(nsim):
        
        print('simidx:', simidx)
                
        for fieldidx, ifield in enumerate(ifield_list):
    
            path_k = '../data/lens_prods/mock_dat/'+mock_dat_datestr+'/TM'+str(inst)+'/mock_kappa_est_'+lensmode+'_scaleclkk='+str(scale_clkk)+'_wnoise_maskJ16_ISL_ifield'+str(ifield)+'_simidx'+str(simidx)+'.txt'
            clf, ld, baseMap, \
                param_dict, kappa_true = init_clkk_prods_mock(inst, ifield, simidx, datestr=mock_dat_datestr,
                                                             lensmode=lensmode, apply_mask=apply_mask,
                                                             plot=plot, lMin=lMin, lMax=lMax, mockstr=mockstr)


            ells_plot = np.logspace(np.log10(300), np.log10(1e5), 50)
            pf_plot = ells_plot*(ells_plot+1)/(2*np.pi)
            plt.figure(figsize=(5, 4))
            plt.plot(ells_plot, pf_plot*ld.ciber_unlensed_auto(ells_plot), label='unlensed auto')
            plt.plot(ells_plot, pf_plot*ld.ciber_obs_auto(ells_plot), label='observed auto')
            plt.xscale('log')
            plt.yscale('log')
            plt.legend()
            plt.xlabel('$\\ell$', fontsize=14)
            plt.xlabel('$D_{\\ell}$', fontsize=14)
            plt.grid(alpha=0.3)
            plt.xlim(300, 1e5)
            plt.ylim(1e-3, 1e4)

            if plot:
                plt.show()
            else:
                plt.close()

            # Organize into dictionaries for v2 function
            map_dict = dict({'counts_map':ld.galcounts, 'mask':ld.mask, 'cib_intensity_map':ld.ciber_map,
                           'obs_map':ld.ciber_map, 'cibFourier':ld.cibFourier, 'kappa_true':kappa_true})
            
            cl_fns = dict({'cib_unlensed_auto':ld.ciber_unlensed_auto, 'obs_auto':ld.ciber_obs_auto,
                          'W_ell':None, 'B_ell':None})
            
            config_dict = dict({'lensmode':lensmode, 'apply_mask':apply_mask, 'mode':'qe_kappa_norm',
                              'cut_lxly':False})
            
            # Get correction factors - need to compute these from param_dict and config_dict
            fns, facs = calc_filters_and_corrections(clf, param_dict, config_dict)
            unmask_frac = np.mean(ld.mask) if apply_mask else 1.0
            
            corr_facs = dict({'kcorr':facs['kcorr'], 'vbeam':facs['vbeam'],
                            'unmask_frac':unmask_frac, 'modefrac':facs['modefrac']})

            psres = compute_lensing_ps_quantities_v2(baseMap, map_dict, cl_fns, param_dict,
                                                    config_dict, corr_facs,
                                                    save_intermediate_plots=False,
                                                    intermediate_plot_dir=None)

            all_clx[simidx, fieldidx] = psres['clx']
            all_clxerr[simidx, fieldidx] = psres['clxerr']
            
            all_clkk[simidx, fieldidx] = psres['clkk']
            all_clkkerr[simidx, fieldidx] = psres['clkkerr']
            
            all_clkg[simidx, fieldidx] = psres['clkg']
            all_clkgerr[simidx, fieldidx] = psres['clkgerr']

            all_clgg[simidx, fieldidx] = psres['clgg']
            all_clkgerr[simidx, fieldidx] = psres['clggerr']

            all_clskew[simidx, fieldidx] = psres['cl_bis']
            all_clskew_WF[simidx, fieldidx] = psres['cl_bis_WF']

            all_clskewerr[simidx, fieldidx] = psres['sCl_bis']

            all_clII[simidx, fieldidx] = psres['clII']
            all_clIIerr[simidx, fieldidx] = psres['clIIerr']

            all_clkgbias[simidx, fieldidx] = psres['dclkg']
            all_clkgbias_WF[simidx, fieldidx] = psres['dclkg_WF']

            # Store N_L normalization (same for all sims/fields, so just store once)
            if psres['N_L'] is not None and all_N_L is None:
                all_N_L = np.zeros((nsim, len(ifield_list), len(psres['N_L'])))
                all_N_L_err = np.zeros((nsim, len(ifield_list), len(psres['N_L'])))
            
            if psres['N_L'] is not None:
                all_N_L[simidx, fieldidx] = psres['N_L']
                all_N_L_err[simidx, fieldidx] = psres['N_L_err']

            # resultFourier, norm_Fourier = run_kappa_est(baseMap, ld.ciber_unlensed_auto, ld.ciber_obs_auto, \
            #                                             param_dict, dataFourier=ld.dataFourier, test=False,\
            #                                              path=path_k, cut_lxly=cut_lxly, mode='qe_kappa_norm')

            # print('loading kappa from ', path_k)
            # kFourier_est = baseMap.loadDataFourier(path_k)
            # kappa_est_map = baseMap.inverseFourier(kFourier_est).real
            
            # lC, clkk, clkkerr = compute_map_ps(baseMap, kappa_true, ld.mask, apply_mask=apply_mask)
            
            # # plot_map(ld.galcounts, figsize=(6, 6), title='gal counts')

            # kappa_est_map = proc_input_map(kappa_est_map, ld.mask, galdens=False, apply_mask=apply_mask)
            # kappa_true = proc_input_map(kappa_true, ld.mask, galdens=False, apply_mask=apply_mask)

            # print('sum of glaaxy counts is ', np.sum(ld.galcounts))
            # galdens = proc_input_map(ld.galcounts, ld.mask, galdens=True, apply_mask=apply_mask)
            

            if plot:
                fig_kappa = plot_map(psres['kappa_est_map'], figsize=(5, 5), title='kappa (estimated)', return_fig=True)
                fig_galdens = plot_map(psres['galdens'], figsize=(5, 5), title='galdens', return_fig=True)
                fig_ciber_map = plot_map(ld.ciber_map, figsize=(5, 5), title='ciber map', return_fig=True)

                save_current_plot(fig_kappa, "kappa_est_map_simidx"+str(simidx)+"_ifield"+str(fieldidx))
                save_current_plot(fig_galdens, "galdens_map_simidx"+str(simidx)+"_ifield"+str(fieldidx))
                save_current_plot(fig_ciber_map, "ciber_map_simidx"+str(simidx)+"_ifield"+str(fieldidx))

            # Plot QE normalization N_L for diagnostics
            if psres['N_L'] is not None:
                fig_nl = plot_normalization(psres['lC'], psres['N_L'], psres['N_L_err'], 
                                 lMin=param_dict['lMin'], lMax=param_dict['lMax'])
                save_current_plot(fig_nl, "N_L_normalization_simidx"+str(simidx)+"_ifield"+str(fieldidx))

            plot_clx_clkk_clkg(psres['lC'], psres['clx'], psres['clkk'], psres['clkg'], psres['clxerr'], psres['clkkerr'], psres['clkgerr'])

            
        lC = psres['lC']    
        plot_clx_clkk_clkg(lC, 
                           np.mean(all_clx[simidx], axis=0), 
                           np.mean(all_clkk[simidx], axis=0),
                           np.mean(all_clkg[simidx], axis=0),
                           np.mean(all_clxerr[simidx], axis=0)/2.,
                           np.mean(all_clkkerr[simidx], axis=0)/2., 
                           np.mean(all_clkgerr[simidx], axis=0)/2.,
                           clkk_lab='$C_{L}^{\\kappa_{CMB}}$ $\\times$ ($\\alpha=$'+str(scale_clkk)+')'
                          )
        
        # Plot averaged N_L normalization across fields
        # if all_N_L is not None:
        #     mean_N_L = np.mean(all_N_L[simidx], axis=0)
        #     mean_N_L_err = np.sqrt(np.mean(all_N_L_err[simidx]**2, axis=0)) / np.sqrt(len(ifield_list))
        #     plot_normalization(lC_norm_stored, mean_N_L, mean_N_L_err,
        #                      lMin=lMin, lMax=lMax)
        
#         plt.figure(figsize=(6, 5))
#         plt.title('Unlensed CIB + instrument noise, unmasked\nCIBER 1.1 $\\mu$m', fontsize=14)
#         plt.errorbar(lC, np.mean(all_clx[simidx], axis=0), yerr=np.mean(all_clxerr[simidx], axis=0)/2., fmt='o', color='k',label='Recovered $\\times$ input $\\kappa$')
#         plt.plot(lC, np.mean(all_clkk[simidx], axis=0), color='r', label='$C_{L}^{\\kappa_{CMB}}$ $\\times$ ($\\alpha=$'+str(scale_clkk)+')')
#         plt.errorbar(lC, np.mean(all_clkg[simidx], axis=0), yerr=np.mean(all_clkgerr[simidx], axis=0)/2., fmt='o', color='b',label='$C_{\\ell}^{\\hat{\\kappa} g}$')
#         plt.legend(fontsize=14)
#         plt.grid(alpha=0.3)
#         plt.xscale('log')
#         plt.yscale('log')
#         plt.xlabel('$L$', fontsize=14)
#         plt.ylabel('$C_{L}^{\\kappa}$', fontsize=14)
#         plt.ylim(1e-13, 1e-6)
#         plt.xlim(1e2, 1e5)
#         plt.show()
        
        simres = {'lC':lC, 'ifield_list':ifield_list,
                  'clx':all_clx[simidx], 'clxerr':all_clxerr[simidx],
                 'clkk':all_clkk[simidx], 'clkkerr':all_clkkerr[simidx],
                 'clkg':all_clkg[simidx], 'clkgerr':all_clkgerr[simidx],
                 'clgg':all_clgg[simidx], 'clggerr':all_clggerr[simidx], 
                 'clskew':all_clskew[simidx], 'clskewerr':all_clskewerr[simidx], 
                 'clII':all_clII[simidx], 'clIIerr':all_clIIerr[simidx], 
                 'clkgbias':all_clkgbias[simidx], 'clskew':all_clskew_WF[simidx],\
                  'clkgbias_WF':all_clkgbias_WF[simidx]}
        
        # Add N_L normalization if available
        if all_N_L is not None:
            simres['lC_norm'] = lC_norm_stored
            simres['N_L'] = all_N_L[simidx]
            simres['N_L_err'] = all_N_L_err[simidx]
        
        np.savez('../data/lens_prods/mock_dat/'+mock_dat_datestr+'/TM'+str(inst)+'/clk_recover_simidx'+str(simidx)+'_'+lensmode+'_'+mockstr+'.npz',
                 **simres)
 
    
    res = {'lC':lC, 'all_clx':all_clx, 'all_clxerr':all_clxerr, 'all_clkk':all_clkk, 'all_clkkerr':all_clkkerr, 'all_clkg':all_clkg, 
          'all_clkgerr':all_clkgerr, 'all_clgg':all_clgg, 'all_clggerr':all_clggerr, 'all_clskew':all_clskew, 'all_clskewerr':all_clskewerr, 
          'all_clII':all_clII, 'all_clIIerr':all_clIIerr, 'all_clkgbias':all_clkgbias, 
          'all_clkgbias_WF':all_clkgbias_WF, 'all_clskew_WF':all_clskew_WF}
    
    # Add N_L normalization to final results
    if all_N_L is not None:
        res['lC_norm'] = lC_norm_stored
        res['all_N_L'] = all_N_L
        res['all_N_L_err'] = all_N_L_err
    
    
    return res
