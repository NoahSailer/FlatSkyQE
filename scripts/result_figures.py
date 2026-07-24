import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from scripts.test_beam_exact_normalization import MAP_SIZE

# import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import from ciber base directory (/Documents/ciber/)



from prep_ciber_dat_lens import *
from forecast_cib_lens import *
from kappa_auto_cross_fns import *
from kappa_plotting_fns import *
from mock_lens_test import *
from lens_response import *
from forecast_cib_lens import plot_integrated_snr_vs_survey_params, ciber_lens_forecast


def init_clf_class(inst=1, ifield=8, ell_min=10., clkg_scale=0.5):

    clf = ciber_lens_forecast(ell_min=ell_min)
    clf.load_bl(inst, ifield)

    clf.load_clk()
    clf.load_clx(clkg_scale=clkg_scale)

    return clf    

def trispectrum_vs_mask_mag(ciber_inst=1, simidx=1, datestr='111125',
                             mag_mins=[17, 18, 19, 20, 21, 22, 23],
                               basedir ='../data/lens_prds/mock_dat/', 
                               Apix=1.15e-9, nbar=1e5, MAP_SIZE=1024):


    clf = init_clf_class(inst=ciber_inst)

    trispec_noise_vs_magcut = np.zeros((len(mag_mins)))

    tmdir = basedir+datestr+'/TM'+str(ciber_inst)
    mockstr = 'noiseless_JHlt16_nbar'+str(nbar)
    mock_sim_fpath = tmdir+ '/mock_dat_unlensed_'+mockstr+'_simidx'+str(simidx)+'.npz'

    cib_intensity_map, all_cib_fluxes, counts_map, mask = generate_cib_map(
                                                        map_size=MAP_SIZE,
                                                        n_cib_per_pixel=0.2, n_gal_per_pixel=0.2,
                                                        seed=0, s_max=None,
                                                    mock_sim_fpath=mock_sim_fpath, apply_mask=False)


    mock_dat = np.load(mock_sim_fpath, allow_pickle=True)
    tracer_cat = mock_dat['tracer_cats']
    mags = tracer_cat[:,3]

    for m, mag in enumerate(mag_mins):

        mag_mask = (mags > mag)*(mags < 25)

        all_cib_fluxes_use = all_cib_fluxes[mag_mask]

        print('all cib fluxes now has length', all_cib_fluxes_use.shape)

        N_G_PER_PIXEL = len(all_cib_fluxes_use)/MAP_SIZE**2
        N_CIB_PER_PIXEL = len(all_cib_fluxes_use)/MAP_SIZE**2
        galaxy_fluxes = all_cib_fluxes_use.copy()

        delta_c, c_i_shot,\
            c_i2_g_shot, galshot, \
                trispec_noise_cib, trispec_noise_g = calculate_analytic_bias(N_CIB_PER_PIXEL, N_G_PER_PIXEL,
                                                                            fluxes_cib=all_cib_fluxes_use,
                                                                            fluxes_g=galaxy_fluxes,
                                                                            pix_area=Apix
                                                                        )
    
        trispec_noise_vs_magcut[m] = trispec_noise_cib

    fig_tris_vs_magcut = plot_trispectrum_vs_magcut(clf, logLmin=1, logLmax=5, nL=100, mag_mins=mag_mins, trispec_noise_vs_magcut=trispec_noise_vs_magcut)


    return fig_tris_vs_magcut

def plot_trispectrum_vs_magcut(clf, logLmin=1, logLmax=5, nL=100, mag_mins=[17, 18, 19, 20, 21, 22, 23], trispec_noise_vs_magcut=None, 
                               figsize=(6, 4), lab_fs=16, tick_fs=12, cmap='jet'):

    Lspace = np.logspace(logLmin, logLmax, nL)

    cmap = plt.get_cmap(cmap)
    colors = cmap(np.linspace(0.2, 0.9, len(mag_mins)))

    fig = plt.figure(figsize=figsize)

    plt.plot(clf.lrange, 2.*clf.clx, color='k', linestyle='solid', linewidth=3, label='Signal')
    plt.plot(clf.lrange, clf.nlk_tot, color='k', linestyle='dashed', label='$N_L^{\\kappa}$ (G)', linewidth=2)

    if trispec_noise_vs_magcut is not None:
        for x in range(len(trispec_noise_vs_magcut)):
            if x==0:
                nglab = '$N_L^{\\kappa}$ (NG), $m>'+str(int(mag_mins[x]))+'$'
            else:
                nglab = '$m>'+str(int(mag_mins[x]))+'$'
            plt.plot(Lspace, trispec_noise_vs_magcut[x]/clf.bl(Lspace)**2, color=colors[x], label=nglab)
            
    # plt.plot(Lspace, clf.bl(Lspace))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(50, 1e5)
    plt.ylim(1e-11, 2e-7)
    plt.legend(fontsize=12,ncol=3, loc=2, bbox_to_anchor=[-0.02, 1.35])
    plt.grid(alpha=0.3)

    plt.tick_params(labelsize=tick_fs)
    plt.xlabel('$L$', fontsize=lab_fs)
    plt.ylabel('$C_L^{\\kappa \\kappa}$', fontsize=lab_fs)
    # plt.savefig('../figures/clkk_compare_gauss_nongauss_noise.pdf', bbox_inches='tight')
    plt.show()

    return fig


def snr_shear_vs_mag_forecasts(L_values = [300, 1000, 3000, 10000], 
                               ellmin=300, ellmax=100000, N_tris=1e-8, show_squeezed=False, 
                               inst=1, figsize=(5, 4.5), savefig=False, save_dir='/Documents/ciber/figures/', 
                               ylim=[1e4, 1e9], bbox_anchor=[-0.1, 1.3], legend_fs=11, shear_lmax=40000, 
                               Adeg_fixed=2000, nbar_fixed=1e5, survey_labels=None, textypos=11, textypos2=25, textypos3=17.5, text_fs=14, lab_fs=16):

    print('Loading CIBER F25 spectrum for TM'+str(inst)+'for SNR forecasts...')
    cell_interp, lb, cl = load_ciber_f25_spectrum(inst=inst, smooth=True, smooth_model='spline')

    nbar_list = [1e4, 2e4, 1e5, 2e5]
    Adeg_list = np.logspace(2, np.log10(3e4), 20)

    figs = plot_snr_shear_mag_fisher_threepanel(
        L_values=L_values,
        ellmin=ellmin, ellmax=ellmax,
        cell_interp=cell_interp,  # your interpolated spectrum
        N_tris=[N_tris],
        figsize=(9, 4),
        ylim_snr=[1e4, 1e10], 
        legend_fs=legend_fs,
        plot_xmin=1.2e4,
        shear_lmax=shear_lmax,
    )

    nbar_list = [1e4, 2e4, 1e5, 2e5]
    Adeg_list = np.logspace(2, np.log10(3e4), 20)

    # Create SNR plots
    fig_integrated_snr, snr_dict = plot_integrated_snr_vs_survey_params(
        nbar_list=nbar_list,
        Adeg_list=Adeg_list,
        L_ranges=[(100, 1000), (1000, 5000), (5000, 20000), (20000, 50000)],
        Adeg_fixed=Adeg_fixed,
        nbar_fixed=nbar_fixed,
        figsize=(5, 8),
        survey_labels=survey_labels,
        bbox_to_anchor=bbox_anchor,
        legend_fs=legend_fs,
        psf_fwhm=3.0,
        ylim=[0, 20],
        text_fs=text_fs, 
        textypos=textypos,
        textypos2=textypos2,
        textypos3=textypos3,
        nl_kappa_nongauss=N_tris,
        lab_fs=lab_fs,        
    )

    # if savefig:
    #     save_path = os.path.join(save_dir, 'snr_shear_vs_mag_forecast.pdf')
    #     print('Saving figure to: ', save_path)
    #     fig.savefig(save_path, bbox_inches='tight')



