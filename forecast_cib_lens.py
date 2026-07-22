from scipy.interpolate import interp1d
import matplotlib
from scipy.ndimage import gaussian_filter1d
import numpy as np
import sys
import os
import config

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
# Add the parent directory to sys.path
sys.path.append(parent_dir)
from ciber.core.powerspec_pipeline import *
from ciber.io.ciber_data_utils import *

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
import bias_modl
from bias_modl import calculate_cib_poisson_terms


def pixel_window_fn(ell):
    
    # Constants
    arcsec_to_rad = np.pi / (180 * 3600)  # Conversion factor
    dX = 7. * arcsec_to_rad  # Pixel size in radians
    dY = 7. * arcsec_to_rad

    # Compute pixel window function
    W_pixel_x = np.sinc(ell * dX / (2 * np.pi))
    W_pixel_y = np.sinc(ell * dY / (2 * np.pi))
    W_pixel = W_pixel_x * W_pixel_y
    
    return W_pixel

def gaussian_smooth(data, sigma):
    smoothed_data = gaussian_filter1d(data, sigma)
    return smoothed_data


def forecast_bandpower_sensitivity(inst, lEdges=None, Adeg=16, ell_min=210, nBins=51):
    
    clf = ciber_lens_forecast(ell_min=ell_min, Adeg=Adeg)
    
    clf.load_bl(inst, 4, inplace=True, plot=False)

    clf.load_clk()
    clf.load_clg(inst, basepath='../data/lens_prods/ciber_gal_cross/')
    clf.load_clx(clkg_scale=0.5)
    clf.forecast_nlkappa(beam_correct=True)

    nbar = 2e4*(180/np.pi)**2 # in steradian -1

    print('nbar in steradian -1:', nbar)

    dclsq, fig = clf.forecast_kappa_gal_cross(nbar, figsize=(7, 4), plot=False)

    dcl = np.sqrt(dclsq)
    if lEdges is None:
        lEdges = np.logspace(np.log10(1.), np.log10(np.max(clf.lrange)), nBins, 10.)

    lcen = np.sqrt(lEdges[:-1]*lEdges[1:])

    dcl_bandpowers = np.zeros_like(lcen)

    for x in range(len(lEdges)-1):
        delta_ell = lEdges[x+1]-lEdges[x]

        dcl_bandpowers[x] = np.mean(dcl[(clf.lrange > lEdges[x])*(clf.lrange < lEdges[x+1])])/np.sqrt(delta_ell)


    return dcl_bandpowers, lcen, lEdges
    

def perform_clkg_forecast_wrapper(inst=1, ifield_use=4, Adeg=16., ell_min=210, \
                                 basepath='../data/lens_prods/ciber_gal_cross/', \
                                 nbar=2e4, figsize=(6, 4), lMin_snr=1000, lMax_snr=1e5, clkg_scale=0.5):
    
    ''' 
    CIBER lensing forecast
    
    nbar is assumed tracer density [deg^-2]
    
    '''
    
    clf = ciber_lens_forecast(ell_min=ell_min, Adeg=Adeg)
    
    clf.load_bl(inst, ifield_use, inplace=True, plot=False)
    
    clf.load_clk()
    clf.load_clg(inst, basepath=basepath)
    clf.load_clx(clkg_scale=clkg_scale)
    clf.forecast_nlkappa(beam_correct=True)

    fig_cls = clf.plot_all_cls(bbox_to_anchor=[1.0, 1.3])

    # nbar_sr = nbar*(180/np.pi)**2 # in steradian -1

    dclsq, fig = clf.forecast_kappa_gal_cross(nbar, figsize=figsize)
    
    dcl = np.sqrt(dclsq)    
    
    return clf, dcl


def compare_forecast_configurations(inst=1, ifield_use=4, ell_min=210, ell_max=1e5,
                                   basepath='../data/lens_prods/ciber_gal_cross/',
                                   Adeg_list=[10, 16, 25], 
                                   nbar_list=[1e4, 2e4, 5e4],
                                   L_min=300, L_max=50000, n_bins=20,
                                   figsize=(14, 5), ylim_kk=[1e-12, 1e-7], ylim_kg=[1e-12, 1e-7],
                                   show_signal=True, clkg_scale=0.5, textkg_xpos=300, textkg_ypos=2e-8, textkg=None, 
                                   bbox_to_anchor=[0.0, 1.0], 
                                   legend_fs=12, legend_ncol=1, wspace=0.3, 
                                   textkg_fs=14, cmap_name='Blues', nl_kappa_nongauss=None):
    """
    Compare bandpower sensitivities across multiple survey configurations.
    Creates two-panel figure: left panel shows N_L^kk components, right panel shows bandpower sensitivities.
    
    Parameters
    ----------
    inst : int
        CIBER instrument
    ifield_use : int
        Field index
    ell_min : float
        Minimum ell for forecast
    basepath : str
        Path to data products
    Adeg_list : list
        List of survey areas in deg^2
    nbar_list : list
        List of galaxy densities in deg^-2
    L_min, L_max, n_bins : float, float, int
        Bandpower configuration
    figsize : tuple
        Figure size
    ylim_kk : list
        Y-axis limits for left panel (N_L^kk)
    ylim_kg : list
        Y-axis limits for right panel (bandpower sensitivity)
    show_signal : bool
        Whether to show signal curve
    clkg_scale : float
        Scaling factor for input C_L^kg signal (default 0.5)
    nl_kappa_nongauss : float or None
        Optional constant (L-independent) non-Gaussian contribution to N_L^kappa.
        For example, from intensity field trispectrum: (1/4) * (<s^4>/<s^2>^2) / nbar.
        If None, only Gaussian noise is included.
        
    Returns
    -------
    fig : matplotlib figure
    results : dict
        Dictionary with all forecast results
    """
    
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    fig, (ax_kk, ax_kg) = plt.subplots(1, 2, figsize=figsize)
    
    # Generate colors using Blues colormap
    n_configs = len(Adeg_list) * len(nbar_list)
    cmap = cm.get_cmap(cmap_name)
    colors = cmap(np.linspace(0.3, 1.0, n_configs))
    
    results = []
    config_idx = 0
    
    # First, create one forecast object to get signal curve and compute components
    clf = ciber_lens_forecast(ell_min=ell_min, ell_max=ell_max, Adeg=Adeg_list[0])
    clf.load_bl(inst, ifield_use, inplace=True, plot=False)
    clf.load_clk()
    clf.load_clg(inst, basepath=basepath)
    clf.load_clx(clkg_scale=clkg_scale)
    clf.forecast_nlkappa(beam_correct=True, nl_kappa_nongauss=nl_kappa_nongauss)
    
    # Compute noise components for first configuration to show in left panel
    nbar_sr = nbar_list[0] * (180/np.pi)**2
    clg_sn_forecast = 1.0 / nbar_sr
    def n_ell_inv(l):
        return 1./(2*l+1)
    inverse_nl = n_ell_inv(clf.lrange) / clf.fsky
    
    term1 = clf.clx**2
    term2 = clf.clk*clf.clg_clus
    term3 = clf.nlk_tot*clf.clg_clus
    term4 = clf.clk*clg_sn_forecast
    term5 = clf.nlk_tot*clg_sn_forecast
    terms = [term1, term2, term3, term4, term5]
    labels_comp = ['$\\propto(C_L^{\\kappa g})^2$', '$\\propto C_L^{\\kappa}C_L^g$', '$\\propto N_L^{\\kappa}C_L^g$', 
                   '$\\propto C_L^{\\kappa}/\\bar{n}$', '$\\propto N_L^{\\kappa}/\\bar{n}$']
    
    # Plot noise components on left panel
    component_colors = ['purple', 'blue', 'green', 'orange', 'brown']
    for i, (term, label) in enumerate(zip(terms, labels_comp)):
        ax_kk.plot(clf.lrange, np.sqrt(inverse_nl*term), 
                  color=component_colors[i], linewidth=2, 
                  label=label, alpha=0.7, linestyle='-')
    
    # Add total uncertainty
    dclsq_total = inverse_nl * (term1 + term2 + term3 + term4 + term5)
    ax_kk.plot(clf.lrange, np.sqrt(dclsq_total), 
              color='k', linewidth=3, linestyle='--', 
              label='Total', alpha=0.9, zorder=100)
    
    # Plot signal curve on right panel once (interpolate over log-spaced grid)
    if show_signal:
        # Create log-spaced grid for smoother plotting
        lrange_plot = np.logspace(np.log10(clf.lrange.min()), np.log10(clf.lrange.max()), 20)
        clx_interp = interp1d(clf.lrange, clf.clx, kind='cubic', bounds_error=False, fill_value='extrapolate')
        clx_plot = clx_interp(lrange_plot)
        ax_kg.plot(lrange_plot, clx_plot, color='k', linewidth=2.5, 
                  label='Predicted $C_L^{\\kappa g}$', zorder=100)
    
    # Loop over configurations
    for Adeg in Adeg_list:
        for nbar in nbar_list:
            
            # Create forecast object
            clf = ciber_lens_forecast(ell_min=ell_min, ell_max=ell_max, Adeg=Adeg)
            clf.load_bl(inst, ifield_use, inplace=True, plot=False)
            clf.load_clk()
            clf.load_clg(inst, basepath=basepath)
            clf.load_clx(clkg_scale=clkg_scale)
            clf.forecast_nlkappa(beam_correct=True, nl_kappa_nongauss=nl_kappa_nongauss)
            
            # Compute bandpowers
            L_centers, sigma_bp, signal_bp, _ = clf.forecast_kappa_gal_cross_bandpowers(
                nbar, L_min=L_min, L_max=L_max, n_bins=n_bins, plot=False
            )
            
            # Store results
            results.append({
                'Adeg': Adeg,
                'nbar': nbar,
                'L_centers': L_centers,
                'sigma_bandpowers': sigma_bp,
                'signal_bandpowers': signal_bp,
                'fsky': clf.fsky,
                'nlk_gauss': clf.nlk_gauss,
                'nlk_tot': clf.nlk_tot
            })
            
            color = colors[config_idx % len(colors)]
            # Convert nbar from deg^-2 to arcmin^-2 (1 deg^2 = 3600 arcmin^2)
            nbar_arcmin = nbar / 3600.0
            label = f'$\\bar{{n}}={nbar_arcmin:.1f}$ arcmin$^{{-2}}$'
            
            # Right panel: Plot bandpower sensitivities
            L_edges = np.logspace(np.log10(L_min), np.log10(L_max), n_bins + 1)
            xerr = [L_centers - L_edges[:-1], L_edges[1:] - L_centers]
            
            ax_kg.errorbar(L_centers, sigma_bp, xerr=xerr, fmt='none',
                          color=color, capsize=3, elinewidth=2.5,
                          label=label, alpha=0.8, zorder=10+config_idx)
            
            config_idx += 1
    
    # Format left panel ($\sigma(C_L^{\kappa g})$ components)
    ax_kk.set_yscale('log')
    ax_kk.set_xscale('log')
    ax_kk.set_ylabel('$\\sigma(C_L^{\\kappa g})$', fontsize=14)
    ax_kk.set_xlabel('$L$', fontsize=14)
    ax_kk.set_ylim(ylim_kk)
    ax_kk.set_xlim([ell_min*0.8, L_max*1.2])
    ax_kk.grid(alpha=0.3)
    ax_kk.legend(fontsize=10, loc=1 ,ncol=2, framealpha=0.9)
    ax_kk.set_title('Per multipole uncertainties', fontsize=14)
    
    # Format right panel (bandpower sensitivity)
    ax_kg.set_yscale('log')
    ax_kg.set_xscale('log')
    ax_kg.set_ylabel('$\\sigma(C_L^{\\kappa g})$', fontsize=14)
    ax_kg.set_xlabel('$L$', fontsize=14)
    ax_kg.set_ylim(ylim_kg)
    ax_kg.set_xlim([L_min*0.8, L_max*1.2])
    ax_kg.grid(alpha=0.3)
    ax_kg.legend(fontsize=legend_fs, loc=2, bbox_to_anchor=bbox_to_anchor, framealpha=0.9, ncol=legend_ncol)
    if textkg is not None:
        ax_kg.text(textkg_xpos, textkg_ypos, textkg, fontsize=textkg_fs, color='k',
                   bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))


    # plt.tight_layout()
    plt.subplots_adjust(wspace=wspace)
    plt.show()
    
    return fig, results
    

class ciber_lens_forecast():
    
    
    def __init__(self, ell_min=300, ell_max=1e5, Adeg=20, nbar=None):
        
        
        self.cbps = CIBER_PS_pipeline()
        
        self.ell_min = ell_min
        self.ell_max = ell_max
        
        self.lrange = np.arange(self.ell_min, self.ell_max)
        self.Adeg = Adeg
        
        self.nbar = nbar # put in sr-1
        
        self.fsky = self.Adeg/41253.
        
        print('fsky = ', self.fsky)


    def load_bl(self, ciber_inst, ifield, inplace=True, plot=False):
        
        # load files
        
        # data_dir = config.ciber_basepath+'data/fluctuation_data/TM'+str(ciber_inst)+'/'
        
        data_dir = '/Users/richardfeder/Documents/ciber/data/fluctuation_data/TM'+str(ciber_inst)+'/'

        bls_fpath = data_dir+'/beam_correction/bl_est_postage_stamps_TM'+str(ciber_inst)+'_081121.npz'

        # bls_fpath = config.ciber_basepath+'data/fluctuation_data/TM'+str(ciber_inst)+'/beam_correction/bl_est_postage_stamps_TM'+str(ciber_inst)+'_081121.npz'
        
        beamdat = np.load(bls_fpath)
        print(beamdat.keys())
        
        blval = beamdat['B_ells_post'][ifield-4,:]
        
        lb = self.cbps.Mkk_obj.midbin_ell

        # blval[-1] = blval[-2]*np.exp(-(lb[-1]/lb[-2])**2/2.)
        
        bl = interp1d(lb, blval, bounds_error=False, fill_value=(1., 0.))
        
        lbindiv = np.arange(np.min(lb), np.max(lb))

        if plot:
            plt.figure(figsize=(4, 3))
            plt.scatter(lb, blval, color='b')
            plt.plot(lbindiv, bl(lbindiv), color='r')
            plt.yscale('log')
            plt.xscale('log')
            plt.show()
        
        if inplace:
            self.bl = bl
            
        else:
            return bl
        
        
    def load_clx(self, clkg_scale=0.5):
    
        # Load the new cl_kcmb_kgal.csv file
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        clx_fpath = os.path.join(parent_dir, 'data', 'cl_kcmb_kgal.csv')
        
        clx_data = np.loadtxt(clx_fpath, delimiter=',')
        lC, clx = clx_data[:, 0], clx_data[:, 1] * clkg_scale
        
        # Fit power law slope between ell=1000 and ell=10000 for extrapolation
        fit_mask = (lC >= 1000) & (lC <= 10000)
        log_ell_fit = np.log10(lC[fit_mask])
        log_cl_fit = np.log10(clx[fit_mask])
        
        # Linear fit in log space: log(Cl) = slope * log(ell) + intercept
        slope, intercept = np.polyfit(log_ell_fit, log_cl_fit, 1)
        
        # Create interpolation function with power law extrapolation
        def clx_func(ell):
            result = np.zeros_like(ell, dtype=float)
            
            # Interpolate where we have data
            within_range = (ell >= lC.min()) & (ell <= lC.max())
            if np.any(within_range):
                interp = interp1d(lC, clx, kind='linear', bounds_error=False)
                result[within_range] = interp(ell[within_range])
            
            # Extrapolate beyond data range using fitted power law
            beyond_range = ell > lC.max()
            if np.any(beyond_range):
                result[beyond_range] = 10**(slope * np.log10(ell[beyond_range]) + intercept)
            
            # Below range (shouldn't happen but just in case)
            below_range = ell < lC.min()
            if np.any(below_range):
                result[below_range] = 10**(slope * np.log10(ell[below_range]) + intercept)
                
            return result
        
        # Use coarser sampling to avoid visual artifacts at transition points
        lrange_coarse = self.lrange[::5]  # Sample every 5th element
        clx_coarse = clx_func(lrange_coarse)
        
        # Interpolate back to full resolution
        clx_smooth_interp = interp1d(lrange_coarse, clx_coarse, kind='cubic', bounds_error=False, fill_value='extrapolate')
        self.clx = gaussian_smooth(np.abs(clx_smooth_interp(self.lrange)), 30)
        
        print(f"Loaded C_L^kg with scale factor {clkg_scale:.2f}")
        print(f"Extrapolation power law: Cl ∝ ell^{slope:.3f} for ell > {lC.max():.0f}")
        
#         np.savez('output/fieldav_clx_WISE_unWISE_neo8_kappa_TM'+str(inst)+'.npz', lC=lC, clx=field_av_clx, clxerr=field_av_clxerr)
    
    
    def plot_all_cls(self, ylim=[1e-12, 1e-6], bbox_to_anchor=[0.0, 1.2], ncol=3):
        
        fig = plt.figure(figsize=(5, 4))
        plt.plot(self.lrange, self.clg_clus, label='$C_{\\ell}^{g}$')
        plt.axhline(self.clg_sn, label='$1/\\overline{n}$', color='k', linestyle='dashed')

        plt.plot(self.lrange, self.nlk_gauss, label='$N_L^{\\kappa, Gauss.}=\\frac{2\\pi}{\\ell_{max}^2-\\ell_{min}^2}$')

        plt.plot(self.lrange, self.clk, label='$C_{\\ell}^{\\kappa}$')
        plt.plot(self.lrange, self.clx, label='$C_{\\ell}^{\\kappa g}$')
        plt.legend(ncol=ncol, bbox_to_anchor=bbox_to_anchor)

        plt.yscale('log')
        plt.xscale('log')
        plt.ylabel('$C_{\\ell}$', fontsize=14)
        plt.xlabel('$\\ell$', fontsize=14)
        plt.ylim(ylim)
        plt.grid(alpha=0.3)
        plt.show()  
        return fig  
        
    def load_clg(self, ciber_inst, catname='WISE', addstr='unWISE_neo8', plot=False, basepath=None):
        
        # load autos from unWISE
        
            
        cgps_file = load_ciber_gal_ps(ciber_inst, catname, addstr=addstr, basepath=basepath)

        lb, all_cl_gal, all_clerr_gal, ifield_list_use = [cgps_file[key] for key in ['lb', 'all_cl_gal', 'all_clerr_gal', 'ifield_list_use']]  

        clg = np.mean(all_cl_gal, axis=0) # used for sample variance estimate
                
        # separate shot noise from clustering
        
        clg_sn = clg[-1]
        # clg_sn = np.mean(clg[lb > 1e5])
        
        clg_clus = clg - clg_sn
        
        # Create interpolation functions with ell^-2 extrapolation below minimum ell
        def clg_tot_func(ell):
            result = np.zeros_like(ell, dtype=float)
            within_range = (ell >= lb.min()) & (ell <= lb.max())
            below_range = ell < lb.min()
            
            if np.any(within_range):
                interp = interp1d(lb, clg, kind='linear', bounds_error=False)
                result[within_range] = interp(ell[within_range])
            
            # Extrapolate below minimum using ell^-2 scaling
            if np.any(below_range):
                clg_min = clg[0]  # C_ell at minimum ell in data
                ell_min = lb[0]
                result[below_range] = clg_min * (ell[below_range] / ell_min)**(-2)
            
            return result
        
        def clg_clus_func(ell):
            result = np.zeros_like(ell, dtype=float)
            within_range = (ell >= lb.min()) & (ell <= lb.max())
            below_range = ell < lb.min()
            
            if np.any(within_range):
                interp = interp1d(lb, clg_clus, kind='linear', bounds_error=False)
                result[within_range] = interp(ell[within_range])
            
            # Extrapolate below minimum using ell^-2 scaling
            if np.any(below_range):
                clg_clus_min = clg_clus[0]  # C_ell at minimum ell in data
                ell_min = lb[0]
                result[below_range] = clg_clus_min * (ell[below_range] / ell_min)**(-2)
            
            return result
                
        clg_tot_interp = clg_tot_func
        clg_interp = clg_clus_func

        self.clg_clus = clg_interp(self.lrange)
        self.clg_sn = clg_sn

        if plot:
        
            plt.figure(figsize=(5, 4))
            plt.plot(self.lrange, clg_tot_interp(self.lrange), label='Total $C_{\\ell}^g$')
            plt.plot(self.lrange, self.clg_clus, linestyle='dashed', color='r', label='clus')
            plt.axhline(self.clg_sn, color='k', linestyle='dashed', label='1/$\\overline{n}$')
            plt.legend()
            plt.yscale('log')
            plt.xscale('log')
            plt.ylabel('Clg')
            plt.show()        
        
        
    def load_clk(self, plot=False):
        
        ''' Load prediction for lensing power spectrum '''
        
        u = UnivPlanck15()
        halofit = Halofit(u, save=False)
        w_cmblens = WeightLensSingle(u, z_source=1100., name="cmblens")
        p2d_cmblens = P2dAuto(u, halofit, w_cmblens, save=False)

        clk = p2d_cmblens.fPinterp(self.lrange)
        
        # Let clk continue to decrease at high ell (removed plateau)
        

        if plot:
            plt.figure(figsize=(5, 4))
            plt.plot(self.lrange, clk)
            plt.yscale('log')
            plt.xscale('log')
            plt.ylabel('Clk')
            plt.show()
        
        self.clk = clk

    def forecast_nlkappa(self, include_gaussian=True, include_tris=False, beam_correct=True, 
                        nl_kappa_nongauss=None, plot=False):
        """
        Forecast N_L^kappa including Gaussian and optional non-Gaussian contributions.
        
        Parameters
        ----------
        include_gaussian : bool
            Include Gaussian reconstruction noise
        include_tris : bool
            Include trispectrum contribution (not yet implemented)
        beam_correct : bool
            Apply beam correction to Gaussian noise
        nl_kappa_nongauss : float or None
            Optional constant (L-independent) non-Gaussian contribution to N_L^kappa,
            e.g., from intensity field trispectrum: (1/4) * (<s^4>/<s^2>^2) * (1/nbar)
        plot : bool
            Whether to plot the result
        """
        
        self.nlk_tot = np.zeros_like(self.lrange)
        self.nlk_gauss, self.nlk_nongauss = None, None
        
        # gaussian piece
        if include_gaussian:
            
            self.nlk_gauss = np.ones_like(self.nlk_tot)
            self.nlk_gauss *= 2*np.pi/(self.ell_max**2 - self.ell_min**2)
#             nlk_gauss /= self.fsky
            
        # Add non-Gaussian contribution (constant in L)
        if nl_kappa_nongauss is not None:
            self.nlk_nongauss = nl_kappa_nongauss * np.ones_like(self.nlk_tot)
            if self.nlk_gauss is not None:
                ratio = nl_kappa_nongauss / np.mean(self.nlk_gauss)
                print(f"Adding non-Gaussian N_L^kappa: {nl_kappa_nongauss:.2e} (ratio to mean Gaussian noise: {ratio:.2f})")
            else:
                print(f"Adding non-Gaussian N_L^kappa contribution: {nl_kappa_nongauss:.2e}")
        
        if beam_correct:
            if self.bl is not None:
                
                blval = self.bl(self.lrange)
                
                self.nlk_gauss /= blval**2
                
        if self.nlk_gauss is not None:
            self.nlk_tot += self.nlk_gauss
            
        if self.nlk_nongauss is not None:
            self.nlk_tot += self.nlk_nongauss
            
        if plot:
            fig = plt.figure(figsize=(5, 4))
            if self.nlk_gauss is not None:
                plt.plot(self.lrange, self.nlk_gauss, label='$N_{\\ell}^{\\kappa}$ (Gaussian)')
            
            if self.nlk_nongauss is not None:
                plt.plot(self.lrange, self.nlk_nongauss, linestyle='--', 
                        label='$N_{\\ell}^{\\kappa}$ (Non-Gaussian)')
                
            if self.clk is not None:
                plt.plot(self.lrange, self.clk, color='k', label='Lensing power spectrum')
                
            plt.yscale('log')
            plt.xscale('log')
            plt.legend()
            
            plt.ylim(1e-11, 1e-4)
            plt.ylabel('$N_L^{\\kappa\\kappa}$', fontsize=14)
            plt.xlabel('$L$', fontsize=14)
            
            plt.grid(alpha=0.3)
            plt.show()


    def forecast_kappa_gal_cross(self, nbar, lab_fs=14, ylim=[1e-11, 1e-6], figsize=(5, 4), plot=True):

        # if nl_kappa is None:
            # _, _, nl_kappa = self.forecast_nlkappa()
        
        # Convert nbar from deg^-2 to sr^-1
        nbar_sr = nbar * (180/np.pi)**2
        
        # Shot noise is 1/nbar (overrides any loaded value)
        clg_sn_forecast = 1.0 / nbar_sr
        
        def n_ell_inv(l):
            
            return 1./(2*l+1)
        
        inverse_nl = n_ell_inv(self.lrange)   
        inverse_nl /= self.fsky
                
        term1 = self.clx**2
        term2 = self.clk*self.clg_clus
        term3 = self.nlk_tot*self.clg_clus
        term4 = self.clk*clg_sn_forecast
        term5 = self.nlk_tot*clg_sn_forecast
    
        terms = [term1, term2, term3, term4, term5]
        
        dclsq = inverse_nl*(term1+term2+term3+term4+term5)
        
        labels = ['$(C_L^{\\kappa g})^2$', '$C_L^{\\kappa}C_L^g$', '$N_L^{\\kappa}C_L^g$', \
                 '$C_L^{\\kappa} \\overline{n}^{-1}$', '$N_L^{\\kappa}\\overline{n}^{-1}$']
        
        if plot:
            fig = plt.figure(figsize=figsize)
            for x in range(len(terms)):
                plt.plot(self.lrange, np.sqrt(inverse_nl*terms[x]), label=labels[x])
            plt.plot(self.lrange, np.sqrt(dclsq), label='Total noise', color='k', linewidth=2, linestyle='dashed')
            
            # Plot the signal (model clkg)
            plt.plot(self.lrange, self.clx, label='$C_L^{\\kappa g}$ (signal)', color='red', linewidth=2.5, linestyle='-', zorder=10)
            
            plt.yscale('log')
            plt.xscale('log')
            plt.legend(ncol=2)
            
            plt.ylim(ylim)
            plt.ylabel('$N_L^{\\kappa g}$', fontsize=lab_fs)
            plt.xlabel('$L$', fontsize=lab_fs)
            
            plt.grid(alpha=0.3)
            plt.show()
        else:
            fig = None
                
        return dclsq, fig
    
    
    def forecast_kappa_gal_cross_bandpowers(self, nbar, L_min=300, L_max=50000, n_bins=20, 
                                            lab_fs=14, ylim=[1e-10, 1e-6], figsize=(6, 4), plot=True):
        """
        Compute and plot bandpower-integrated sensitivity for kappa-galaxy cross-correlation.
        
        Parameters
        ----------
        nbar : float
            Galaxy number density in deg^-2
        L_min : float
            Minimum L for bandpowers
        L_max : float
            Maximum L for bandpowers
        n_bins : int
            Number of logarithmically spaced bandpowers
        lab_fs : int
            Label fontsize
        ylim : list
            Y-axis limits
        figsize : tuple
            Figure size
        plot : bool
            Whether to plot
            
        Returns
        -------
        L_centers : array
            Bandpower centers
        sigma_bandpowers : array
            Bandpower uncertainties (noise)
        signal_bandpowers : array
            Bandpower signals
        fig : matplotlib figure or None
        """
        
        # First compute per-ell quantities
        dclsq, _ = self.forecast_kappa_gal_cross(nbar, plot=False)
        dcl = np.sqrt(dclsq)
        
        # Create log-spaced bandpower edges
        L_edges = np.logspace(np.log10(L_min), np.log10(L_max), n_bins + 1)
        L_centers = np.sqrt(L_edges[:-1] * L_edges[1:])
        
        # Integrate in quadrature within each bandpower using inverse-variance weighting
        # This optimally combines the per-L measurements
        sigma_bandpowers = np.zeros(n_bins)
        signal_bandpowers = np.zeros(n_bins)
        
        for i in range(n_bins):
            mask = (self.lrange >= L_edges[i]) & (self.lrange < L_edges[i+1])
            
            if np.sum(mask) > 0:
                # Inverse-variance weighting: sigma_bin^2 = 1 / sum(1/sigma_L^2)
                # This is optimal and scales correctly with number of modes at each L
                inv_var_sum = np.sum(1.0 / dcl[mask]**2)
                sigma_bandpowers[i] = 1.0 / np.sqrt(inv_var_sum)
                
                # Signal: inverse-variance weighted average
                weights = 1.0 / dcl[mask]**2
                signal_bandpowers[i] = np.sum(self.clx[mask] * weights) / np.sum(weights)
        
        if plot:
            fig = plt.figure(figsize=figsize)
            
            # Plot bandpower uncertainties
            xerr = [L_centers - L_edges[:-1], L_edges[1:] - L_centers]
            plt.errorbar(L_centers, sigma_bandpowers, xerr=xerr, fmt='o', 
                        color='k', capsize=3, markersize=6, linewidth=2,
                        label='Bandpower sensitivity', zorder=5)
            
            # Plot signal bandpowers
            plt.errorbar(L_centers, signal_bandpowers, xerr=xerr, fmt='s',
                        color='red', capsize=3, markersize=6, linewidth=2,
                        label='Signal (binned)', zorder=6)
            
            # Also show smooth signal curve for reference
            plt.plot(self.lrange, self.clx, color='red', linewidth=1, 
                    alpha=0.5, linestyle='--', label='Signal (full)', zorder=4)
            
            plt.yscale('log')
            plt.xscale('log')
            plt.legend(fontsize=lab_fs-2)
            
            plt.ylim(ylim)
            plt.xlim([L_min*0.8, L_max*1.2])
            plt.ylabel('$\\sigma(C_L^{\\kappa g})$', fontsize=lab_fs)
            plt.xlabel('$L$', fontsize=lab_fs)
            plt.title(f'{n_bins} bandpowers, $\\bar{{n}}={nbar:.0f}$ deg$^{{-2}}$', fontsize=lab_fs)
            
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()
        else:
            fig = None
        
        return L_centers, sigma_bandpowers, signal_bandpowers, fig


    def compute_bispectrum_bias(self, fluxes_cib, fluxes_g, n_cib_per_deg2, n_g_per_deg2, 
                                pix_size_arcsec=7.0):
        """
        Compute the primary bispectrum bias ΔC_L^{κg} using the Poisson limit formula.
        
        This bias arises from the self-lensing of CIB sources and is approximately 
        scale-independent in the Poisson limit.
        
        Parameters
        ----------
        fluxes_cib : array_like
            Flux distribution of all CIB sources [nW/m^2/sr or arbitrary units]
        fluxes_g : array_like
            Flux distribution of tracer galaxy subset [same units as fluxes_cib]
        n_cib_per_deg2 : float
            Number density of all CIB sources [deg^-2]
        n_g_per_deg2 : float
            Number density of tracer galaxies [deg^-2]
        pix_size_arcsec : float
            Pixel size in arcseconds (default 7.0 for CIBER)
            
        Returns
        -------
        result_dict : dict
            Dictionary containing:
            - 'delta_clkg': Primary bispectrum bias ΔC_L^{κg}
            - 'galshot': Galaxy shot noise 1/n_g
            - 'c_i_shot': CIB shot noise power
            - 'c_i2_g_shot': Galaxy flux moment term
            - 'mean_s2_cib': <s²>_cib
            - 'mean_s2_g': <s²>_g
            - 'mean_s4_cib': <s⁴>_cib (for trispectrum)
            - 'trispec_noise_cib': Trispectrum noise term (CIB)
            - 'trispec_noise_g': Trispectrum noise term (galaxies)
            
        Notes
        -----
        The bias formula is: ΔC_L^{κg} = (A_pix × <s²>_g) / (2 × C^I_shot)
        where C^I_shot = n_cib × <s²>_cib
        
        This correctly accounts for the flux-weighted ratio between all CIB sources
        and the tracer galaxy subset.
        """
        
        # Convert pixel size to area in deg^2 and steradians
        arcsec_per_deg = 3600.0
        pix_area_deg2 = (pix_size_arcsec / arcsec_per_deg)**2
        pix_area_sr = pix_area_deg2 * (np.pi / 180.)**2  # Convert deg^2 to sr
        
        # Convert densities from deg^-2 to sr^-1
        deg2_to_sr = (np.pi / 180.)**2
        n_cib_per_sr = n_cib_per_deg2 * deg2_to_sr
        n_g_per_sr = n_g_per_deg2 * deg2_to_sr
        
        # Convert number densities to per-pixel (using sr densities)
        n_cib_per_pixel = n_cib_per_sr * pix_area_sr
        n_g_per_pixel = n_g_per_sr * pix_area_sr
        
        # Use existing bias_modl function (pix_area must be in sr)
        result_dict = calculate_cib_poisson_terms(
            n_cib_per_pixel=n_cib_per_pixel,
            n_g_per_pixel=n_g_per_pixel,
            fluxes_cib=fluxes_cib,
            fluxes_g=fluxes_g,
            pix_area=pix_area_sr
        )
        
        # Store as class attributes for later use
        self.delta_clkg_bias = result_dict['delta_clkg']
        self.c_i_shot = result_dict['c_i_shot']
        self.galshot_bias = result_dict['galshot']
        
        return result_dict
    
    
    def forecast_bispectrum_bias_uncertainty(self, fluxes_cib, fluxes_g, 
                                            n_cib_per_deg2, n_g_per_deg2,
                                            pix_size_arcsec=7.0,
                                            L_min=300, L_max=50000, n_bins=20,
                                            figsize=(6, 4), plot=True):
        """
        Forecast the uncertainty on the primary bispectrum bias measurement.
        
        This computes ΔC_L^{κg} and then forecasts how well it can be measured
        given the noise properties of the kappa and galaxy maps.
        
        Parameters
        ----------
        fluxes_cib : array_like
            Flux distribution of all CIB sources
        fluxes_g : array_like  
            Flux distribution of tracer galaxy subset
        n_cib_per_deg2 : float
            Number density of all CIB sources [deg^-2]
        n_g_per_deg2 : float
            Number density of tracer galaxies [deg^-2]
        pix_size_arcsec : float
            Pixel size in arcseconds
        L_min, L_max : float
            Bandpower range
        n_bins : int
            Number of bandpowers
        figsize : tuple
            Figure size
        plot : bool
            Whether to plot results
            
        Returns
        -------
        bias_result : dict
            Dictionary with bias computation results
        L_centers : array
            Bandpower centers
        sigma_bias_bp : array
            Uncertainty on bias in each bandpower
        snr_bias_bp : array
            SNR on bias in each bandpower
        fig : matplotlib figure or None
        """
        
        # First compute the bias
        bias_result = self.compute_bispectrum_bias(
            fluxes_cib=fluxes_cib,
            fluxes_g=fluxes_g,
            n_cib_per_deg2=n_cib_per_deg2,
            n_g_per_deg2=n_g_per_deg2,
            pix_size_arcsec=pix_size_arcsec
        )
        
        delta_clkg = bias_result['delta_clkg']
        
        # Now compute the noise on measuring this bias
        # The noise structure is the same as for C_L^κg, but the signal is now delta_clkg
        
        # Use galaxy density for noise calculation
        nbar_sr = n_g_per_deg2 * (180/np.pi)**2
        clg_sn_forecast = 1.0 / nbar_sr
        
        def n_ell_inv(l):
            return 1./(2*l+1)
        
        inverse_nl = n_ell_inv(self.lrange) / self.fsky
        
        # Noise terms (same as standard cross-correlation)
        term1 = delta_clkg**2  # Signal squared (now constant)
        term2 = self.clk * self.clg_clus
        term3 = self.nlk_tot * self.clg_clus
        term4 = self.clk * clg_sn_forecast
        term5 = self.nlk_tot * clg_sn_forecast
        
        dclsq_bias = inverse_nl * (term1 + term2 + term3 + term4 + term5)
        dcl_bias = np.sqrt(dclsq_bias)
        
        # Compute bandpowers
        L_edges = np.logspace(np.log10(L_min), np.log10(L_max), n_bins + 1)
        L_centers = np.sqrt(L_edges[:-1] * L_edges[1:])
        
        sigma_bias_bp = np.zeros(n_bins)
        signal_bias_bp = np.zeros(n_bins)
        snr_bias_bp = np.zeros(n_bins)
        
        for i in range(n_bins):
            mask = (self.lrange >= L_edges[i]) & (self.lrange < L_edges[i+1])
            
            if np.sum(mask) > 0:
                # Inverse-variance weighting
                inv_var_sum = np.sum(1.0 / dcl_bias[mask]**2)
                sigma_bias_bp[i] = 1.0 / np.sqrt(inv_var_sum)
                signal_bias_bp[i] = delta_clkg  # Constant signal
                snr_bias_bp[i] = signal_bias_bp[i] / sigma_bias_bp[i]
        
        if plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            
            # Left panel: Signal and noise
            xerr = [L_centers - L_edges[:-1], L_edges[1:] - L_centers]
            
            ax1.errorbar(L_centers, sigma_bias_bp, xerr=xerr, fmt='o',
                        color='blue', capsize=3, markersize=6, linewidth=2,
                        label='$\\sigma(\\Delta C_L^{\\kappa g})$', zorder=5)
            
            ax1.axhline(delta_clkg, color='red', linewidth=2.5, linestyle='--',
                       label=f'$\\Delta C_L^{{\\kappa g}}$ = {delta_clkg:.2e}', zorder=10)
            
            ax1.set_yscale('log')
            ax1.set_xscale('log')
            ax1.legend(fontsize=10)
            ax1.set_ylabel('Bias amplitude', fontsize=12)
            ax1.set_xlabel('$L$', fontsize=12)
            ax1.set_xlim([L_min*0.8, L_max*1.2])
            ax1.grid(alpha=0.3)
            ax1.set_title('Bispectrum Bias', fontsize=12)
            
            # Right panel: SNR
            ax2.errorbar(L_centers, snr_bias_bp, xerr=xerr, fmt='s',
                        color='green', capsize=3, markersize=6, linewidth=2,
                        label='SNR per bandpower', zorder=5)
            
            ax2.axhline(1, color='k', linewidth=1, linestyle=':', alpha=0.5)
            ax2.set_xscale('log')
            ax2.legend(fontsize=10)
            ax2.set_ylabel('SNR', fontsize=12)
            ax2.set_xlabel('$L$', fontsize=12)
            ax2.set_xlim([L_min*0.8, L_max*1.2])
            ax2.grid(alpha=0.3)
            ax2.set_title(f'$\\bar{{n}}_g$ = {n_g_per_deg2:.1e} deg$^{{-2}}$', fontsize=12)
            
            plt.tight_layout()
            plt.show()
        else:
            fig = None
        
        return bias_result, L_centers, sigma_bias_bp, snr_bias_bp, fig


def test_flux_cut_impact(fluxes_cib, fluxes_g, n_cib_per_deg2, n_g_per_deg2,
                        flux_cut_fractions=[0.5, 0.7, 0.8, 0.9, 0.95, 1.0],
                        pix_size_arcsec=7.0, figsize=(12, 4)):
    """
    Test how limiting the maximum flux of galaxy tracers affects delta C_L^kg bias.
    
    This function applies percentile-based flux cuts to the tracer sample and shows
    how the bispectrum bias changes. Lower flux cuts remove the brightest sources,
    which can significantly reduce the bias.
    
    Parameters
    ----------
    fluxes_cib : array_like
        Full flux distribution of all CIB sources
    fluxes_g : array_like
        Full flux distribution of galaxy tracers (before cuts)
    n_cib_per_deg2 : float
        Number density of all CIB sources [deg^-2]
    n_g_per_deg2 : float
        Original number density of galaxy tracers [deg^-2]
    flux_cut_fractions : list
        Fractions of the tracer sample to keep (e.g., 0.9 = keep faintest 90%, 
        remove brightest 10%). Default tests several cuts from 50% to 100%.
    pix_size_arcsec : float
        Pixel size in arcseconds
    figsize : tuple
        Figure size
        
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'flux_cuts': Maximum flux values corresponding to each fraction
        - 'delta_clkg': Bias for each cut
        - 'n_g_eff': Effective galaxy density after each cut
        - 'mean_s2_g': <s²>_g for each cut
        - 'mean_s2_cib': <s²>_cib (constant)
        - 'reduction_factor': Delta_clkg / Delta_clkg(no cut)
    fig : matplotlib figure
        
    Notes
    -----
    A flux_cut_fraction of 0.9 means we keep the faintest 90% of sources
    and remove the brightest 10%. This reduces <s²>_g and thus the bias.
    """
    
    fluxes_g = np.asarray(fluxes_g)
    fluxes_cib = np.asarray(fluxes_cib)
    flux_cut_fractions = np.asarray(flux_cut_fractions)
    
    # Sort galaxy fluxes
    fluxes_g_sorted = np.sort(fluxes_g)
    
    # Storage for results
    n_cuts = len(flux_cut_fractions)
    flux_cuts = np.zeros(n_cuts)
    delta_clkg_vals = np.zeros(n_cuts)
    n_g_eff_vals = np.zeros(n_cuts)
    mean_s2_g_vals = np.zeros(n_cuts)
    mean_s2_cib_val = np.mean(fluxes_cib**2)
    c_i_shot_vals = np.zeros(n_cuts)
    
    # Pixel area
    arcsec_per_deg = 3600.0
    pix_area_deg2 = (pix_size_arcsec / arcsec_per_deg)**2
    pix_area_sr = pix_area_deg2 * (np.pi / 180.)**2  # Convert deg^2 to sr
    
    # Convert densities from deg^-2 to sr^-1
    deg2_to_sr = (np.pi / 180.)**2
    n_cib_per_sr = n_cib_per_deg2 * deg2_to_sr
    
    for i, frac in enumerate(flux_cut_fractions):
        # Determine flux cut (keep faintest frac of sources)
        idx_cut = int(frac * len(fluxes_g_sorted))
        if idx_cut >= len(fluxes_g_sorted):
            idx_cut = len(fluxes_g_sorted)
        
        fluxes_g_cut = fluxes_g_sorted[:idx_cut]
        flux_cuts[i] = fluxes_g_cut[-1] if len(fluxes_g_cut) > 0 else 0
        
        # Effective galaxy density after cut
        n_g_eff = n_g_per_deg2 * frac
        n_g_eff_vals[i] = n_g_eff
        n_g_eff_sr = n_g_eff * deg2_to_sr
        
        # Convert to per-pixel (using sr densities)
        n_cib_per_pixel = n_cib_per_sr * pix_area_sr
        n_g_per_pixel = n_g_eff_sr * pix_area_sr
        
        # Compute bias with cut sample
        if len(fluxes_g_cut) > 0:
            bias_result = calculate_cib_poisson_terms(
                n_cib_per_pixel=n_cib_per_pixel,
                n_g_per_pixel=n_g_per_pixel,
                fluxes_cib=fluxes_cib,
                fluxes_g=fluxes_g_cut,
                pix_area=pix_area_sr
            )
            delta_clkg_vals[i] = bias_result['delta_clkg']
            mean_s2_g_vals[i] = bias_result['mean_s2_g']
            c_i_shot_vals[i] = bias_result['c_i_shot']
        else:
            delta_clkg_vals[i] = 0
            mean_s2_g_vals[i] = 0
            c_i_shot_vals[i] = 0
    
    # Compute reduction factors relative to no cut
    reduction_factor = delta_clkg_vals / delta_clkg_vals[-1] if delta_clkg_vals[-1] != 0 else delta_clkg_vals
    
    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Panel 1: Delta C_L^kg vs flux cut
    ax = axes[0]
    ax.plot(flux_cut_fractions * 100, delta_clkg_vals, 'o-', 
            color='blue', linewidth=2.5, markersize=8)
    ax.set_xlabel('Tracer sample kept (%)', fontsize=12)
    ax.set_ylabel('$\\Delta C_L^{\\kappa g}$', fontsize=12)
    ax.set_title('Bias vs Flux Cut', fontsize=12)
    ax.grid(alpha=0.3)
    ax.axhline(delta_clkg_vals[-1], color='k', linestyle='--', alpha=0.5, label='No cut')
    ax.legend(fontsize=10)
    
    # Panel 2: Reduction factor
    ax = axes[1]
    ax.plot(flux_cut_fractions * 100, reduction_factor, 's-',
            color='red', linewidth=2.5, markersize=8)
    ax.set_xlabel('Tracer sample kept (%)', fontsize=12)
    ax.set_ylabel('Bias reduction factor', fontsize=12)
    ax.set_title('$\\Delta C_L^{\\kappa g}$ / $\\Delta C_L^{\\kappa g}$(no cut)', fontsize=12)
    ax.grid(alpha=0.3)
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.5)
    
    # Panel 3: <s²>_g vs cut
    ax = axes[2]
    ax.plot(flux_cut_fractions * 100, mean_s2_g_vals, '^-',
            color='green', linewidth=2.5, markersize=8)
    ax.set_xlabel('Tracer sample kept (%)', fontsize=12)
    ax.set_ylabel('$\\langle s^2 \\rangle_g$', fontsize=12)
    ax.set_title('Galaxy flux moment', fontsize=12)
    ax.grid(alpha=0.3)
    ax.axhline(mean_s2_g_vals[-1], color='k', linestyle='--', alpha=0.5, label='No cut')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\n" + "="*70)
    print("FLUX CUT IMPACT ON BISPECTRUM BIAS")
    print("="*70)
    print(f"{'Keep %':<10} {'Max Flux':<15} {'n_g [deg⁻²]':<15} {'ΔC_L^kg':<15} {'Reduction':<12}")
    print("-"*70)
    for i, frac in enumerate(flux_cut_fractions):
        print(f"{frac*100:<10.1f} {flux_cuts[i]:<15.3e} {n_g_eff_vals[i]:<15.2e} "
              f"{delta_clkg_vals[i]:<15.3e} {reduction_factor[i]:<12.3f}")
    print("="*70)
    print(f"CIB shot noise C^I: {c_i_shot_vals[0]:.3e}")
    print(f"<s²>_CIB: {mean_s2_cib_val:.3e}")
    print("="*70 + "\n")
    
    results = {
        'flux_cut_fractions': flux_cut_fractions,
        'flux_cuts': flux_cuts,
        'delta_clkg': delta_clkg_vals,
        'n_g_eff': n_g_eff_vals,
        'mean_s2_g': mean_s2_g_vals,
        'mean_s2_cib': mean_s2_cib_val,
        'reduction_factor': reduction_factor,
        'c_i_shot': c_i_shot_vals[0]
    }
    
    return results, fig


def generate_magnitude_based_flux_distribution(m_min=18, m_max=25, m0=18, 
                                               n_total=100000, flux_normalization=1.0,
                                               return_mags=False):
    """
    Generate flux distribution from magnitude distribution with dN/dm ∝ (m-m0)^2.
    
    This creates a realistic source distribution where fainter sources are much
    more numerous than bright sources.
    
    Parameters
    ----------
    m_min : float
        Minimum (brightest) magnitude (default 18)
    m_max : float
        Maximum (faintest) magnitude (default 25)
    m0 : float
        Reference magnitude for distribution (default 18)
    n_total : int
        Total number of sources to generate
    flux_normalization : float
        Multiplicative factor to convert from AB mag fluxes to physical units
        (e.g., nW/m²/sr). Default 1.0 keeps relative flux units.
        For physical units: F[nW/m²/sr] = 3631e9 * 10^(-0.4*m) where m is AB mag.
    return_mags : bool
        If True, also return the magnitude array
        
    Returns
    -------
    fluxes : array
        Flux values (units depend on flux_normalization)
    mags : array (optional)
        Magnitude values, returned only if return_mags=True
        
    Notes
    -----
    The magnitude distribution follows dN/dm ∝ (m-m0)^2, giving many more
    faint sources than bright ones. Fluxes are computed as F = 10^(-0.4*m),
    normalized so that m=0 corresponds to F=1.
    
    Typical source counts: ~100,000 sources/deg² with m < 25 
    (e.g., 400k sources over 4 deg²)
    
    For physical CIB flux units, use flux_normalization ≈ 3631e9 (AB mag to nW/m²/sr)
    or adjust based on your filter bandpass.
    """
    
    # Create magnitude grid for probability distribution
    m_grid = np.linspace(m_min, m_max, 1000)
    
    # Probability density: dN/dm ∝ (m-m0)^2
    prob_density = (m_grid - m0)**2
    prob_density /= np.trapz(prob_density, m_grid)  # Normalize
    
    # Cumulative distribution function
    cdf = np.cumsum(prob_density) * (m_grid[1] - m_grid[0])
    cdf /= cdf[-1]  # Ensure it goes to 1
    
    # Sample magnitudes using inverse transform sampling
    u = np.random.uniform(0, 1, n_total)
    mags = np.interp(u, cdf, m_grid)
    
    # Convert magnitudes to fluxes: F = flux_normalization * 10^(-0.4*m)
    fluxes = flux_normalization * 10**(-0.4 * mags)
    
    if return_mags:
        return fluxes, mags
    else:
        return fluxes


def compute_density_from_source_counts(n_sources_total, area_deg2, m_max):
    """
    Convert total source counts to surface density in deg^-2.
    
    Parameters
    ----------
    n_sources_total : float
        Total number of sources in the survey
    area_deg2 : float
        Survey area in deg^2
    m_max : float
        Maximum (faintest) magnitude of the sample
        
    Returns
    -------
    density_deg2 : float
        Surface density in deg^-2
        
    Example
    -------
    >>> # ~400k sources with m<25 over 4 deg^2
    >>> n_cib = compute_density_from_source_counts(400000, 4.0, 25)
    >>> print(f"CIB density: {n_cib:.2e} deg^-2")
    CIB density: 1.00e+05 deg^-2
    """
    density_deg2 = n_sources_total / area_deg2
    print(f"Surface density with m < {m_max}: {density_deg2:.2e} sources/deg²")
    print(f"  = {density_deg2 * (180/np.pi)**2:.2e} sources/sr")
    print(f"  Over {area_deg2} deg²: {n_sources_total:.2e} total sources")
    return density_deg2


def test_magnitude_cuts(fluxes_cib, fluxes_g, mags_g, n_cib_per_deg2, n_g_per_deg2,
                       mags_cib=None,
                       m_min_cuts=[18, 19, 20, 21, 22, 23],
                       pix_size_arcsec=7.0, figsize=(12, 4)):
    """
    Test how increasing minimum magnitude (removing bright sources) affects delta C_L^kg bias.
    
    Parameters
    ----------
    fluxes_cib : array_like
        Full flux distribution of all CIB sources
    fluxes_g : array_like
        Full flux distribution of galaxy tracers
    mags_g : array_like
        Magnitude values for galaxy tracers
    n_cib_per_deg2 : float
        Number density of all CIB sources [deg^-2]
    n_g_per_deg2 : float
        Original number density of galaxy tracers [deg^-2]
    mags_cib : array_like or None
        Magnitude values for CIB sources (required for proper C_ell^II calculation)
    m_min_cuts : list
        Minimum magnitude cuts to test (keeps sources with m >= m_min)
    pix_size_arcsec : float
        Pixel size in arcseconds
    figsize : tuple
        Figure size
        
    Returns
    -------
    results : dict
        Dictionary containing bias results for each magnitude cut
    """
    
    fluxes_g = np.asarray(fluxes_g)
    fluxes_cib = np.asarray(fluxes_cib)
    mags_g = np.asarray(mags_g)
    m_min_cuts = np.asarray(m_min_cuts)
    
    if mags_cib is None:
        raise ValueError("mags_cib is required to properly compute C_ell^II with magnitude cuts")
    mags_cib = np.asarray(mags_cib)
    
    # Storage for results
    n_cuts = len(m_min_cuts)
    delta_clkg_vals = np.zeros(n_cuts)
    n_g_eff_vals = np.zeros(n_cuts)
    n_cib_eff_vals = np.zeros(n_cuts)
    mean_s2_g_vals = np.zeros(n_cuts)
    mean_s2_cib_vals = np.zeros(n_cuts)
    c_i_shot_vals = np.zeros(n_cuts)
    cl_II_vals = np.zeros(n_cuts)  # CIB auto power
    cl_IIg_vals = np.zeros(n_cuts)  # CIB-galaxy cross power
    
    # Pixel area
    arcsec_per_deg = 3600.0
    pix_area_deg2 = (pix_size_arcsec / arcsec_per_deg)**2
    pix_area_sr = pix_area_deg2 * (np.pi / 180.)**2  # Convert deg^2 to sr
    
    # Conversion factor from deg^-2 to sr^-1
    deg2_to_sr = (np.pi / 180.)**2
    
    for i, m_min in enumerate(m_min_cuts):
        # Keep only sources fainter than (or equal to) m_min for both populations
        mask_g = mags_g >= m_min
        mask_cib = mags_cib >= m_min
        
        fluxes_g_cut = fluxes_g[mask_g]
        fluxes_cib_cut = fluxes_cib[mask_cib]
        
        # Effective densities after cuts
        fraction_kept_g = np.sum(mask_g) / len(mags_g)
        fraction_kept_cib = np.sum(mask_cib) / len(mags_cib)
        
        n_g_eff = n_g_per_deg2 * fraction_kept_g
        n_cib_eff = n_cib_per_deg2 * fraction_kept_cib
        
        n_g_eff_vals[i] = n_g_eff
        n_cib_eff_vals[i] = n_cib_eff
        
        # Convert densities to sr^-1
        n_cib_eff_sr = n_cib_eff * deg2_to_sr
        n_g_eff_sr = n_g_eff * deg2_to_sr
        
        # Convert to per-pixel (using sr densities)
        n_cib_per_pixel = n_cib_eff_sr * pix_area_sr
        n_g_per_pixel = n_g_eff_sr * pix_area_sr
        
        # Compute bias with cut samples
        if len(fluxes_g_cut) > 0 and n_g_per_pixel > 0 and len(fluxes_cib_cut) > 0:
            bias_result = calculate_cib_poisson_terms(
                n_cib_per_pixel=n_cib_per_pixel,
                n_g_per_pixel=n_g_per_pixel,
                fluxes_cib=fluxes_cib_cut,
                fluxes_g=fluxes_g_cut,
                pix_area=pix_area_sr
            )
            delta_clkg_vals[i] = bias_result['delta_clkg']
            mean_s2_g_vals[i] = bias_result['mean_s2_g']
            c_i_shot_vals[i] = bias_result['c_i_shot']
            mean_s2_cib_vals[i] = bias_result['mean_s2_cib']
            
            # Compute power spectra (Poisson limit, shot noise)
            # C_ell^II = n_CIB_eff * <s²>_CIB_cut (now depends on magnitude cut)
            cl_II_vals[i] = (n_cib_eff * (180/np.pi)**2) * mean_s2_cib_vals[i]
            
            # C_L^{IIg} = n_g_eff * <s²>_g (assuming galaxies are subset of CIB)
            cl_IIg_vals[i] = (n_g_eff * (180/np.pi)**2) * mean_s2_g_vals[i]
        else:
            delta_clkg_vals[i] = 0
            mean_s2_g_vals[i] = 0
            mean_s2_cib_vals[i] = 0
            c_i_shot_vals[i] = 0
            cl_II_vals[i] = 0
            cl_IIg_vals[i] = 0
    
    # Compute reduction factors relative to first cut (brightest sample)
    reduction_factor = delta_clkg_vals / delta_clkg_vals[0] if delta_clkg_vals[0] != 0 else delta_clkg_vals
    
    # Create plots - now with 4 panels
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Panel 1: Delta C_L^kg vs magnitude cut
    ax = axes[0]
    ax.plot(m_min_cuts, delta_clkg_vals, 'o-', 
            color='blue', linewidth=2.5, markersize=8)
    ax.set_xlabel('Minimum magnitude cut', fontsize=12)
    ax.set_ylabel('$\\Delta C_L^{\\kappa g}$', fontsize=12)
    ax.set_title('Bias vs Magnitude Cut', fontsize=12)
    ax.grid(alpha=0.3)
    
    # Panel 2: Reduction factor
    ax = axes[1]
    ax.plot(m_min_cuts, reduction_factor, 's-',
            color='red', linewidth=2.5, markersize=8)
    ax.set_xlabel('Minimum magnitude cut', fontsize=12)
    ax.set_ylabel('Bias reduction factor', fontsize=12)
    ax.set_title('Relative to m$_{\\rm min}$=' + f'{m_min_cuts[0]:.0f}', fontsize=12)
    ax.grid(alpha=0.3)
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.5)
    
    # Panel 3: <s²>_g vs cut
    ax = axes[2]
    ax.plot(m_min_cuts, mean_s2_g_vals, '^-',
            color='green', linewidth=2.5, markersize=8)
    ax.set_xlabel('Minimum magnitude cut', fontsize=12)
    ax.set_ylabel('$\\langle s^2 \\rangle_g$', fontsize=12)
    ax.set_title('Galaxy flux moment', fontsize=12)
    ax.grid(alpha=0.3)
    
    # Panel 4: Power spectra with twin y-axes
    ax = axes[3]
    ax2 = ax.twinx()
    
    # C_ell^II on left axis
    line1 = ax.plot(m_min_cuts, cl_II_vals, 'o-',
                    color='purple', linewidth=2.5, markersize=8,
                    label='$C_\\ell^{II}$ (CIB auto)')
    ax.set_xlabel('Minimum magnitude cut', fontsize=12)
    ax.set_ylabel('$C_\\ell^{II}$', fontsize=12, color='purple')
    ax.tick_params(axis='y', labelcolor='purple')
    
    # C_L^{IIg} on right axis
    line2 = ax2.plot(m_min_cuts, cl_IIg_vals, 's-',
                     color='darkorange', linewidth=2.5, markersize=8,
                     label='$C_L^{IIg}$ (CIB-gal cross)')
    ax2.set_ylabel('$C_L^{IIg}$', fontsize=12, color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange')
    
    ax.set_title('Power Spectra vs Cut', fontsize=12)
    ax.grid(alpha=0.3)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\\n" + "="*100)
    print("MAGNITUDE CUT IMPACT ON BISPECTRUM BIAS")
    print("="*100)
    print(f"{'m_min':<8} {'n_CIB [deg⁻²]':<14} {'n_g [deg⁻²]':<13} {'<s²>_CIB':<12} {'<s²>_g':<12} {'ΔC_L^kg':<12} {'C_ell^II':<12} {'C_L^IIg':<12} {'Reduction':<10}")
    print("-"*100)
    for i, m_min in enumerate(m_min_cuts):
        print(f"{m_min:<8.1f} {n_cib_eff_vals[i]:<14.2e} {n_g_eff_vals[i]:<13.2e} {mean_s2_cib_vals[i]:<12.3e} {mean_s2_g_vals[i]:<12.3e} "
              f"{delta_clkg_vals[i]:<12.3e} {cl_II_vals[i]:<12.3e} {cl_IIg_vals[i]:<12.3e} {reduction_factor[i]:<10.3f}")
    print("="*100)
    print(f"Note: C_ell^II and C_L^IIg now both reflect magnitude cuts on their respective populations")
    print("="*100 + "\\n")
    
    results = {
        'm_min_cuts': m_min_cuts,
        'delta_clkg': delta_clkg_vals,
        'n_g_eff': n_g_eff_vals,
        'n_cib_eff': n_cib_eff_vals,
        'mean_s2_g': mean_s2_g_vals,
        'mean_s2_cib': mean_s2_cib_vals,
        'reduction_factor': reduction_factor,
        'c_i_shot': c_i_shot_vals,
        'cl_II': cl_II_vals,
        'cl_IIg': cl_IIg_vals
    }
    
    return results


def test_flux_cut_with_magnitude_distribution(n_cib_per_deg2=1e5, n_g_per_deg2=2e4,
                                              m_min_cib=18, m_max_cib=25, m0_cib=18,
                                              m_min_g=18, m_max_g=23, m0_g=18,
                                              m_min_cuts=[18, 19, 20, 21, 22, 23],
                                              flux_normalization=1e7,
                                              pix_size_arcsec=7.0, figsize=(12, 4),
                                              seed=None):
    """
    Test flux cut impact using realistic magnitude-based flux distributions.
    
    Generates flux distributions from magnitude distributions with dN/dm ∝ (m-m0)^2,
    where fainter sources are much more numerous. CIB population extends to fainter
    magnitudes than the tracer galaxy sample.
    
    Parameters
    ----------
    n_cib_per_deg2 : float
        Number density of all CIB sources [deg^-2]
    n_g_per_deg2 : float
        Number density of galaxy tracers [deg^-2]
    m_min_cib, m_max_cib : float
        Magnitude range for full CIB population (e.g., 18-25)
    m0_cib : float
        Reference magnitude for CIB distribution
    m_min_g, m_max_g : float
        Magnitude range for galaxy tracers (e.g., 18-23, brighter than CIB)
    m0_g : float
        Reference magnitude for galaxy distribution
    m_min_cuts : list
        Minimum magnitude cuts to test (e.g., [18, 19, 20, 21, 22, 23])
        Each cut keeps only tracers with m >= m_min_cut (fainter than cut)
    flux_normalization : float
        Multiplicative factor to scale fluxes to physical units (default 1e7).
        Typical values: 1e7-1e9 for nW/m²/sr-like units.
        Adjust this to match your expected bias magnitude (~10^-9).
    pix_size_arcsec : float
        Pixel size in arcseconds
    figsize : tuple
        Figure size
    seed : int or None
        Random seed for reproducibility
        
    Returns
    -------
    results : dict
        Results from test_flux_cut_impact
    fig : matplotlib figure
    fluxes_cib : array
        Generated CIB flux distribution
    fluxes_g : array
        Generated galaxy tracer flux distribution
    mags_cib : array
        CIB magnitudes
    mags_g : array
        Galaxy tracer magnitudes
        
    Example
    -------
    >>> # Test with realistic galaxy populations
    >>> results, fig, f_cib, f_g, m_cib, m_g = test_flux_cut_with_magnitude_distribution(
    ...     n_cib_per_deg2=1e5,     # Dense CIB population
    ...     n_g_per_deg2=2e4,       # Sparser tracer sample
    ...     m_min_cib=18, m_max_cib=25,  # CIB to faint limits
    ...     m_min_g=18, m_max_g=23,      # Tracers only to m=23
    ...     m_min_cuts=[18, 19, 20, 21, 22, 23]  # Test magnitude cuts
    ... )
    >>> print(f"Bias at m>20 cut: {results['delta_clkg'][2]:.2e}")
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    # Print density information
    print("\n" + "="*70)
    print("SOURCE POPULATION SETUP")
    print("="*70)
    print(f"CIB population: {n_cib_per_deg2:.2e} sources/deg² with {m_min_cib} < m < {m_max_cib}")
    print(f"  = {n_cib_per_deg2 * (180/np.pi)**2:.2e} sources/sr")
    print(f"Galaxy tracers: {n_g_per_deg2:.2e} sources/deg² with {m_min_g} < m < {m_max_g}")
    print(f"  = {n_g_per_deg2 * (180/np.pi)**2:.2e} sources/sr")
    print(f"  Tracer fraction: {n_g_per_deg2/n_cib_per_deg2:.1%} of CIB")
    print(f"Flux normalization: {flux_normalization:.2e}")
    print(f"  (scales 10^(-0.4*m) to physical flux units)")
    print("="*70 + "\n")
    
    # Generate CIB flux distribution
    print(f"Generating {int(n_cib_per_deg2)} CIB sources with {m_min_cib} < m < {m_max_cib}")
    fluxes_cib, mags_cib = generate_magnitude_based_flux_distribution(
        m_min=m_min_cib, m_max=m_max_cib, m0=m0_cib,
        n_total=int(n_cib_per_deg2), flux_normalization=flux_normalization,
        return_mags=True
    )
    
    # Generate galaxy tracer flux distribution
    print(f"Generating {int(n_g_per_deg2):.0e} galaxy tracers with {m_min_g} < m < {m_max_g}")
    fluxes_g, mags_g = generate_magnitude_based_flux_distribution(
        m_min=m_min_g, m_max=m_max_g, m0=m0_g,
        n_total=int(n_g_per_deg2), flux_normalization=flux_normalization,
        return_mags=True
    )
    
    # Show magnitude distributions
    fig_hist = plt.figure(figsize=(5, 4))
    
    plt.hist(mags_cib, bins=np.linspace(m_min_cib, m_max_cib, 50), histtype='step', linewidth=2, label='CIB (all)', color='blue')
    plt.hist(mags_g, bins=np.linspace(m_min_cib, m_max_cib, 50), alpha=0.6, label='Tracers', color='red')
    plt.xlabel('Magnitude', fontsize=12)
    plt.ylabel('Counts', fontsize=12)
    plt.title('Magnitude Distributions', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    
    # ax2 = plt.subplot(1, 2, 2)
    # ax2.hist(np.log10(fluxes_cib), bins=np.linspace(np.log10(np.min(fluxes_cib)), np.log10(np.max(fluxes_cib)), 50), histtype='step', linewidth=2, label='CIB (all)', color='blue')
    # ax2.hist(np.log10(fluxes_g), bins=np.linspace(np.log10(np.min(fluxes_g)), np.log10(np.max(fluxes_g)), 50), alpha=0.6, label='Tracers', color='red')
    # ax2.set_xlabel('log$_{10}$(Flux)', fontsize=12)
    # ax2.set_ylabel('Counts', fontsize=12)
    # ax2.set_title('Flux Distributions', fontsize=12)
    # ax2.legend()
    # ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nCIB: <F> = {np.mean(fluxes_cib):.3e}, <F²> = {np.mean(fluxes_cib**2):.3e}")
    print(f"Tracers: <F> = {np.mean(fluxes_g):.3e}, <F²> = {np.mean(fluxes_g**2):.3e}")
    
    # Now test magnitude cuts
    print("\n" + "="*70)
    print("Testing magnitude cuts on tracer sample...")
    print("="*70)
    
    results = test_magnitude_cuts(
        fluxes_cib=fluxes_cib,
        fluxes_g=fluxes_g,
        mags_g=mags_g,
        mags_cib=mags_cib,
        n_cib_per_deg2=n_cib_per_deg2,
        n_g_per_deg2=n_g_per_deg2,
        m_min_cuts=m_min_cuts,
        pix_size_arcsec=pix_size_arcsec,
        figsize=figsize
    )
    
    return results, fluxes_cib, fluxes_g, mags_cib, mags_g


def plot_integrated_snr_vs_survey_params(inst=1, ifield=4, 
                                         nbar_list=[1e3, 5e3, 1e4, 2e4, 5e4, 1e5],
                                         Adeg_list=[5, 10, 20, 40, 80],
                                         L_ranges=[(300, 1e4), (300, 3e4), (300, 5e4)],
                                         ell_min=100, ell_max=1e5,
                                         figsize=(10, 5), catname='WISE', addstr='unWISE_neo8', 
                                         Adeg_fixed=100., nbar_fixed=2e4, 
                                         legend_fs=12, survey_labels=None, 
                                         bbox_to_anchor=[0.0, 1.3], 
                                         lab_fs=16, markersize=3, psf_fwhm=None, 
                                         ylim=[1e-1, 1e2], textypos=6, textypos2=10, text_fs=12, 
                                         hspace=0.3, nl_kappa_nongauss=None):
    """
    Compute and plot integrated SNR as a function of tracer density and sky area
    for different L ranges.
    
    Parameters
    ----------
    inst : int
        CIBER instrument (1 or 2)
    ifield : int
        Field number for beam loading
    nbar_list : list
        List of galaxy number densities to test [deg^-2]
    Adeg_list : list
        List of sky areas to test [deg^2]
    L_ranges : list of tuples
        List of (L_min, L_max) tuples for SNR integration
    ell_min, ell_max : float
        Range for lensing reconstruction
    figsize : tuple
        Figure size
    catname, addstr : str
        For loading galaxy clustering and cross-spectrum
    survey_labels : list or None
        Optional list of survey names to label each density point
        e.g., ['unWISE', 'Rubin', 'Roman']
    psf_fwhm : float or None
        If provided, use a Gaussian beam with this FWHM (in arcseconds).
        If None, load the CIBER beam for the specified instrument and field.
    nl_kappa_nongauss : float or None
        Optional constant (L-independent) non-Gaussian contribution to N_L^kappa.
        For example, from intensity field trispectrum: (1/4) * (<s^4>/<s^2>^2) / nbar.
        If None, only Gaussian noise is included.
        
    Returns
    -------
    fig : matplotlib figure
    snr_dict : dict
        Dictionary containing SNR arrays for each configuration
    """
    
    # Initialize forecast object with fiducial area
    clf = ciber_lens_forecast(ell_min=ell_min, ell_max=ell_max, Adeg=Adeg_list[0])
    
    # Load necessary spectra
    if psf_fwhm is None:
        # Use CIBER beam
        clf.load_bl(inst, ifield, inplace=True, plot=False)
        print(f"Using CIBER instrument {inst} beam (field {ifield})")
    else:
        # Generate Gaussian beam from FWHM
        # B(ℓ) = exp(-ℓ²σ²/2)
        # σ_rad = FWHM_arcsec / (2*sqrt(2*ln(2))) * (π/180/3600)
        from scipy.interpolate import interp1d
        
        fwhm_rad = psf_fwhm * np.pi / (180. * 3600.)  # Convert arcsec to radians
        sigma_rad = fwhm_rad / (2. * np.sqrt(2. * np.log(2.)))
        
        # Create beam function on clf.lrange
        bl_array = np.exp(-0.5 * (clf.lrange * sigma_rad)**2)
        
        # Store as interpolation function (like load_bl does)
        clf.bl = interp1d(clf.lrange, bl_array, bounds_error=False, fill_value=(1., 0.))
        print(f"Using Gaussian beam with FWHM = {psf_fwhm:.2f} arcsec (σ = {sigma_rad*180*3600/np.pi:.2f} arcsec)")
    
    clf.load_clk()
    clf.load_clg(inst, catname=catname, addstr=addstr)
    clf.load_clx(clkg_scale=0.5)
    clf.forecast_nlkappa(beam_correct=True, nl_kappa_nongauss=nl_kappa_nongauss, plot=False)
    
    # Store SNR results
    snr_vs_nbar = {f'L={Lr[0]:.0f}-{Lr[1]:.0e}': [] for Lr in L_ranges}
    snr_vs_area = {f'L={Lr[0]:.0f}-{Lr[1]:.0e}': [] for Lr in L_ranges}
    
    print("Computing SNR vs tracer density (Adeg={:.1f} deg^2)...".format(Adeg_fixed))
    for nbar in nbar_list:
        clf.Adeg = Adeg_fixed
        clf.fsky = clf.Adeg / 41253.
        
        # Get per-ell noise
        dclsq, _ = clf.forecast_kappa_gal_cross(nbar, plot=False)
        dcl = np.sqrt(dclsq)
        
        # Compute SNR for each L range
        for L_range in L_ranges:
            L_min, L_max = L_range
            mask = (clf.lrange >= L_min) & (clf.lrange <= L_max)
            
            # SNR = sqrt(sum((signal/noise)^2))
            snr_squared_sum = np.sum((clf.clx[mask] / dcl[mask])**2)
            snr = np.sqrt(snr_squared_sum)
            
            key = f'L={L_range[0]:.0f}-{L_range[1]:.0e}'
            snr_vs_nbar[key].append(snr)
    
    print("Computing SNR vs sky area (nbar={:.0f} deg^-2)...".format(nbar_fixed))
    for Adeg in Adeg_list:
        clf.Adeg = Adeg
        clf.fsky = clf.Adeg / 41253.
        
        # Get per-ell noise
        dclsq, _ = clf.forecast_kappa_gal_cross(nbar_fixed, plot=False)
        dcl = np.sqrt(dclsq)
        
        # Compute SNR for each L range
        for L_range in L_ranges:
            L_min, L_max = L_range
            mask = (clf.lrange >= L_min) & (clf.lrange <= L_max)
            
            # SNR = sqrt(sum((signal/noise)^2))
            snr_squared_sum = np.sum((clf.clx[mask] / dcl[mask])**2)
            snr = np.sqrt(snr_squared_sum)
            
            key = f'L={L_range[0]:.0f}-{L_range[1]:.0e}'
            snr_vs_area[key].append(snr)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=figsize)
    
    # Define colors for different L ranges
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    # Convert tracer density from deg^-2 to arcmin^-2 (1 deg^2 = 3600 arcmin^2)
    nbar_arcmin = [n / 3600. for n in nbar_list]
    
    # Plot 1: SNR vs tracer density
    for idx, L_range in enumerate(L_ranges):
        key = f'L={L_range[0]:.0f}-{L_range[1]:.0e}'
        label = str(L_range[0])+'$<L<$'+str(int(L_range[1]))

        ax1.plot(nbar_arcmin, snr_vs_nbar[key], color=colors[idx % len(colors)],
                linewidth=2, label=label, marker='o', markersize=markersize)
    
    ax1.set_xscale('log')
    # ax1.set_xlim(2, 60)
    # ax1.set_yscale('log')
    ax1.set_ylim(bottom=0, top=ylim[1])
    ax1.set_xlabel('Tracer density $\\bar{n}$ [arcmin$^{-2}$]', fontsize=14)
    ax1.set_ylabel('Integrated SNR', fontsize=lab_fs)
    # ax1.set_title(f'Sky area = {Adeg_fixed:.0f} deg$^2$', fontsize=14)
    ax1.text(3.0, 10, f'Sky area = {Adeg_fixed:.0f} deg$^2$', fontsize=16)

    # ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.tick_params(labelsize=12)
    ax1.legend(fontsize=legend_fs, ncol=2, loc=2, bbox_to_anchor=bbox_to_anchor)

    
    # Add custom tick labels if survey names are provided
    if survey_labels is not None:
        if len(survey_labels) == len(nbar_arcmin):
            # Create tick labels combining density and survey name
            # tick_labels = [f'{n:.1f}\n{survey}' if n < 1 else f'{n:.0f}\n({survey})' 
            #               for n, survey in zip(nbar_arcmin, survey_labels)]
            
            tick_labels = [f'{n:.1f}' if n < 1 else f'{n:.0f}' 
                          for n, survey in zip(nbar_arcmin, survey_labels)]
            ax1.set_xticks(nbar_arcmin)
            ax1.set_xticklabels(tick_labels, fontsize=11)

            for idx, (n, survey) in enumerate(zip(nbar_arcmin, survey_labels)):
                ax1.text(n*0.95, textypos, survey, fontsize=text_fs, rotation=90)

        else:
            print(f"Warning: survey_labels length ({len(survey_labels)}) does not match nbar_list length ({len(nbar_arcmin)})")


    nbar_fixed_arcmin = nbar_fixed / 3600.
    
    # Plot 2: SNR vs sky area
    for idx, L_range in enumerate(L_ranges):
        key = f'L={L_range[0]:.0f}-{L_range[1]:.0e}'
        # label = f'L=[{L_range[0]:.0f}, {L_range[1]:.0e}]'

        label = str(L_range[0])+'$<L<$'+str(int(L_range[1]))
        ax2.plot(Adeg_list, snr_vs_area[key], color=colors[idx % len(colors)],
                linewidth=2, label=label)
    
    ax2.set_xscale('log')
    ax2.set_xlabel('Sky area [deg$^2$]', fontsize=14)
    ax2.set_xlim(100, 30000)
    ax2.set_ylabel('Integrated SNR', fontsize=lab_fs)
    # ax2.set_yscale('log')
    ax2.set_ylim(bottom=0)
    # ax2.set_title(f'$\\bar{{n}}$ = {nbar_fixed_arcmin:.1f} arcmin$^{{-2}}$', fontsize=14)
    ax2.text(150, textypos2, f'$\\bar{{n}}$ = {nbar_fixed_arcmin:.1f} arcmin$^{{-2}}$', fontsize=16)

    ax2.grid(alpha=0.3)
    ax2.tick_params(labelsize=12)
    plt.subplots_adjust(hspace=hspace)

    # plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\n" + "="*70)
    print("Integrated SNR Summary")
    print("="*70)
    print(f"\nAt Adeg={Adeg_fixed:.0f} deg^2, nbar={nbar_fixed:.0f} deg^-2:")
    for L_range in L_ranges:
        key = f'L={L_range[0]:.0f}-{L_range[1]:.0e}'
        # Get SNR at closest point to fiducial values
        idx_nbar = np.argmin(np.abs(np.array(nbar_list) - nbar_fixed))
        snr_fid = snr_vs_nbar[key][idx_nbar]
        print(f"  L=[{L_range[0]:.0f}, {L_range[1]:.0e}]: SNR = {snr_fid:.1f}")
    
    snr_dict = {
        'nbar_list': nbar_list,
        'nbar_arcmin': nbar_arcmin,
        'Adeg_list': Adeg_list,
        'snr_vs_nbar': snr_vs_nbar,
        'snr_vs_area': snr_vs_area,
        'L_ranges': L_ranges
    }
    
    return fig, snr_dict


        