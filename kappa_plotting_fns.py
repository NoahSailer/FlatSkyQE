import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def gen_suptitle(grab_cib_sim, add_noise, apply_mask, exact_beam=False, pixel_fn_correct=False, psf_pix_fwhm=None):
    if grab_cib_sim:
        suptitle = 'CIBER PSF'
    elif psf_pix_fwhm is not None:
        suptitle = 'Gaussian PSF FWHM='+str(psf_pix_fwhm)+' pix'
    else:
        suptitle = ''

    if add_noise:
        suptitle += ', with white noise'
    else:
        suptitle += ', noiseless'

    if apply_mask:
        suptitle+=', masked'
    else:
        suptitle+=', unmasked'

    if exact_beam:
        suptitle += ', exact beam corr.'

    return suptitle


def plot_input_recovered_kappa(
        res,
        bbox_to_anchor=None,
        ncol=1,
        figsize=(6, 5),
        markersize=5,
        legend_fs=10,
        xlim=[100, 1.0e5],
        ylim=[1e-11, 1e-5],
        loc=3,
        lMax=None,
        lMin=None,
        plot_sem=True,
        capsize=2.5,
    ):
    # Function body goes here


    clk_input = np.mean(res['clkg_kappa_input'], axis=0)
    kappa_amp = res['kappa_amplitude']
    lC = res['lC']
    refclkg = res['analytic_bias']

    y_clkg, sem_clkg = np.mean(res['clkg_kappa_input'], axis=0), np.std(res['clkg_kappa_input'], axis=0, ddof=1)/np.sqrt(res['clkg_kappa_input'].shape[0])
    y_dclkg, sem_dclkg = np.mean(res['dclkg_kappa_input'], axis=0), np.std(res['dclkg_kappa_input'], axis=0, ddof=1)/np.sqrt(res['dclkg_kappa_input'].shape[0])

    clkg_label = '$C_{L}^{\\hat{\\kappa}\\kappa_{input}}$'
    dclkg_label = '$C_{L}^{I^2\\kappa_{input}}$ (mocks)'

    print('y_dclkg:', y_dclkg)

    fig = plt.figure(figsize=figsize)

    if plot_sem:
        plt.errorbar(lC, y_clkg, yerr=sem_clkg, label=clkg_label, color='r', marker='x', markersize=markersize, capsize=capsize)
        plt.errorbar(lC, y_dclkg, yerr=sem_dclkg, label=dclkg_label, color='b', marker='^', markersize=markersize, capsize=capsize)
    else:
        plt.plot(lC, y_clkg, label=clkg_label, color='r', marker='x', markersize=markersize)
        plt.plot(lC, y_dclkg, label=dclkg_label, color='b', marker='^', markersize=markersize)
    # If lensing was applied, show the input C_L^kappa spectrum
    print('clk input is ', clk_input)
    plt.plot(lC, clk_input, label=f'Input $C_{{L}}^{{\\kappa}}$ (amp={kappa_amp})', color='purple', linewidth=2.5, linestyle='--', zorder=5)

    plt.axhline(refclkg, label='$\\Delta C_{L}^{\\kappa g}=\\Omega_{\\rm pix}C_{\\ell}^{I^2g}/2C_{\\ell}^{II}$', color='k', linestyle='dashed')
    if lMax is not None:
        plt.axvline(lMax, color='k', linestyle='solid')
    if lMin is not None:
        plt.axvline(lMin, color='k', linestyle='solid')

    plt.xlabel('$L$', fontsize=14)

    plt.xscale('log')
    plt.yscale('log')
    plt.legend(ncol=ncol, bbox_to_anchor=bbox_to_anchor, fontsize=legend_fs, loc=loc)
    plt.xlim(xlim)

    plt.ylim(ylim)

    return fig


def plot_recov_components(res, figsize=(9, 6), markersize=10, ylim=[1e-10, 1e-7], xlim=[300, 8e4],
                         ncol=1, bbox_to_anchor=[0.0, 1.3], legend_fs=10, loc=3, plot_ratio=False,
                         ylim_ratio=None, rat_min=0.05, rat_max=50, title_fs=16, suptitle=None, plot_sem=True,
                         capsize=2.5, lMax=None, lMin=None, show=True, ylogscale=True):
    
    
    if res['clII'].shape[0] < 2:
        print('Only one realization-- cannot plot error bars.')
        plot_sem = False

    if plot_ratio:
        # loc=2
        if ylim_ratio is None:
            ylim = [1e-2, 1e1]
        else:
            ylim = ylim_ratio

    ci2g_str = '$C_{L}^{I^2 g} = < s^2 > \\Omega_{\\rm pix}^{-1}$'

    y_clII, sem_clII = np.mean(res['clII'], axis=0), np.std(res['clII'], axis=0, ddof=1)/np.sqrt(res['clII'].shape[0])
    y_clbis, sem_clbis = np.mean(res['cl_bis'], axis=0), np.std(res['cl_bis'], axis=0, ddof=1)/np.sqrt(res['cl_bis'].shape[0])

    # Use kappa cross-correlation if lensing was applied, otherwise use galaxy cross
    if 'enable_lensing' in res and res['enable_lensing']:
        y_clkg, sem_clkg = np.mean(res['clkg_kappa_input'], axis=0), np.std(res['clkg_kappa_input'], axis=0, ddof=1)/np.sqrt(res['clkg_kappa_input'].shape[0])
        y_dclkg, sem_dclkg = np.mean(res['dclkg_kappa_input'], axis=0), np.std(res['dclkg_kappa_input'], axis=0, ddof=1)/np.sqrt(res['dclkg_kappa_input'].shape[0])
        using_kappa_input = True
    else:
        y_clkg, sem_clkg = np.mean(res['clkg'], axis=0), np.std(res['clkg'], axis=0, ddof=1)/np.sqrt(res['clkg'].shape[0])
        y_dclkg, sem_dclkg = np.mean(res['dclkg'], axis=0), np.std(res['dclkg'], axis=0, ddof=1)/np.sqrt(res['dclkg'].shape[0])
        using_kappa_input = False

    y_clgg, sem_clgg = np.mean(res['clgg'], axis=0), np.std(res['clgg'], axis=0, ddof=1)/np.sqrt(res['clgg'].shape[0])
    
    refclii, refclbis, refclkg, refclgg = res['c_i_shot'], res['c_i2_g_shot'], res['analytic_bias'], res['galshot']

    print('sem clbis:   ', sem_clbis)
    if plot_ratio:
        y_clII /= res['c_i_shot']
        y_clbis /= res['c_i2_g_shot']
        y_clkg /= res['analytic_bias']
        y_dclkg /= res['analytic_bias']
        y_clgg /= res['galshot']

        sem_clII /= res['c_i_shot']
        sem_clbis /= res['c_i2_g_shot']
        sem_clkg /= res['analytic_bias']
        sem_dclkg /= res['analytic_bias']
        sem_clgg /= res['galshot']

        refclii, refclbis, refclkg, refclgg = 1., 1., 1., 1.



    fig = plt.figure(figsize=figsize)
    
    if suptitle is not None:
        plt.suptitle(suptitle, fontsize=title_fs)

    lC = res['lC']

    plt.subplot(2,2,1)

    if plot_ratio:
        if plot_sem:
            if using_kappa_input:
                plt.errorbar(lC, y_clgg, yerr=sem_clgg, label='$C_{\\ell}^{II}$/theory', color='C1', marker='.', markersize=markersize, capsize=capsize)
            else:
                plt.errorbar(lC, y_clgg, yerr=sem_clgg, label='$C_{\\ell}^{gg}$/theory', color='C1', marker='.', markersize=markersize, capsize=capsize)
        else:
            if using_kappa_input:
                plt.plot(lC, y_clgg, label='$C_{\\ell}^{II}$/theory', color='C1', marker='.', markersize=markersize)
            else:
                plt.plot(lC, y_clgg, label='$C_{\\ell}^{gg}$/theory', color='C1', marker='.', markersize=markersize)
    else:
        if plot_sem:
            if using_kappa_input:
                plt.errorbar(lC, y_clgg, yerr=sem_clgg, label='$C_{\\ell}^{gg}$ (galaxy tracer)', color='C1', marker='.', markersize=markersize, capsize=capsize)
            else:
                plt.errorbar(lC, y_clgg, yerr=sem_clgg, label='$C_{\\ell}^{gg}$ (mocks)', color='C1', marker='.', markersize=markersize, capsize=capsize)
        else:
            if using_kappa_input:
                plt.plot(lC, y_clgg, label='$C_{\\ell}^{gg}$ (galaxy tracer)', color='C1', marker='.', markersize=markersize)
            else:
                plt.plot(lC, y_clgg, label='$C_{\\ell}^{gg}$ (mocks)', color='C1', marker='.', markersize=markersize)

    if using_kappa_input:
        plt.axhline(refclgg, label='$C_{\\ell}^{gg} = \\overline{n}_g$ (shot)', color='k', linestyle='dashed')
    else:
        plt.axhline(refclgg, label='$1/\\overline{n}_g$', color='k', linestyle='dashed')

    if lMax is not None:
        plt.axvline(lMax, color='k', linestyle='solid')
    if lMin is not None:
        plt.axvline(lMin, color='k', linestyle='solid')

    plt.xscale('log')
    if ylogscale:
        plt.yscale('log')
    plt.legend(ncol=ncol, bbox_to_anchor=bbox_to_anchor, fontsize=legend_fs, loc=loc)
    plt.xlim(xlim)



    if ylim is None:
        ylim_plot = [rat_min*res['galshot'], rat_max*res['galshot']]
    else:
        ylim_plot = ylim
    plt.ylim(ylim_plot)

    plt.ylabel('$C_{\\ell}$', fontsize=14)
    plt.subplot(2,2,2)

    if plot_ratio:
        if plot_sem:
            plt.errorbar(lC, y_clII, yerr=sem_clII, label='$C_{\ell}^{II}$/theory', color='C0', marker='x', markersize=markersize, capsize=capsize)
        else:
            plt.plot(lC, y_clII, label='$C_{\ell}^{II}$/theory', color='C0', marker='x', markersize=markersize)
    else:
        if plot_sem:
            plt.errorbar(lC, y_clII, yerr=sem_clII, label='$C_{\ell}^{II}$ (mocks)', color='C0', marker='x', markersize=markersize, capsize=capsize)
        else:
            plt.plot(lC, y_clII, label='$C_{\ell}^{II}$ (mocks)', color='C0', marker='x', markersize=markersize)

    plt.axhline(refclii, label='$C_{\ell}^{II} = \\overline{n}_g< s^2 >$', color='k', linestyle='dashed')
 
    if lMax is not None:
        plt.axvline(lMax, color='k', linestyle='solid')
    if lMin is not None:
        plt.axvline(lMin, color='k', linestyle='solid')

    plt.xscale('log')
    if ylogscale:
        plt.yscale('log')
    plt.legend(ncol=ncol, bbox_to_anchor=bbox_to_anchor, fontsize=legend_fs, loc=loc)
    plt.xlim(xlim)

    if ylim is None:
        ylim_plot = [rat_min*res['c_i_shot'], rat_max*res['c_i_shot']]
    else:
        ylim_plot = ylim
    plt.ylim(ylim_plot)

    plt.subplot(2,2,3)
    
    if plot_ratio:
        plt.errorbar(lC, y_clbis, yerr=sem_clbis, label='$C_{\\ell}^{I^2 g}$/theory', color='C2', marker='*', markersize=markersize, capsize=capsize)
        # plt.plot(lC, np.mean(res['cl_bis'], axis=0)/res['c_i2_g_shot'], label='$C_{\\ell}^{I^2 g}$/theory', color='C2', marker='*', markersize=markersize)
    else:
        if plot_sem:
            plt.errorbar(lC, y_clbis, yerr=sem_clbis, label='$C_{\\ell}^{I^2 g}$ (mocks)', color='C2', marker='*', markersize=markersize, capsize=capsize)
        else:
            plt.plot(lC, y_clbis, label='$C_{\\ell}^{I^2 g}$ (mocks)', color='C2', marker='*', markersize=markersize)

    plt.axhline(refclbis, label=ci2g_str, color='k', linestyle='dashed')

    plt.xscale('log')
    if ylogscale:
        plt.yscale('log')
    plt.legend(ncol=ncol, bbox_to_anchor=bbox_to_anchor, fontsize=legend_fs,  loc=loc)
    plt.xlim(xlim)

    if lMax is not None:
        plt.axvline(lMax, color='k', linestyle='solid')
    if lMin is not None:
        plt.axvline(lMin, color='k', linestyle='solid')

    if ylim is None:
        ylim_plot = [rat_min*res['c_i2_g_shot'], rat_max*res['c_i2_g_shot']]
    else:
        ylim_plot = ylim
    plt.ylim(ylim_plot)

    plt.ylabel('$C_{\\ell}$', fontsize=14)
    plt.xlabel('$\\ell$', fontsize=14)

    plt.subplot(2,2,4)

    if plot_ratio:
        # plt.plot(lC, np.mean(res['clkg'], axis=0)/res['analytic_bias'], label='$C_{L}^{\\hat{\\kappa} g}$/theory', color='r', marker='x', markersize=markersize)
        # plt.plot(lC, np.mean(res['dclkg'], axis=0)/res['analytic_bias'], label='$\\Delta C_{L}^{\\kappa g}$/theory', color='b', marker='^', markersize=markersize)

        plt.errorbar(lC, y_clkg, yerr=sem_clkg, label='$C_{L}^{\\hat{\\kappa} g}$/theory', color='r', marker='x', markersize=markersize, capsize=capsize)
        plt.errorbar(lC, y_dclkg, yerr=sem_dclkg, label='$\\Delta C_{L}^{\\kappa g}$/theory', color='b', marker='^', markersize=markersize, capsize=capsize)

    else:
        # plt.plot(lC, np.mean(res['clkg'], axis=0), label='$C_{L}^{\\hat{\\kappa} g}$', color='r', marker='x', markersize=markersize)
        # plt.plot(lC, np.mean(res['dclkg'], axis=0), label='$\\Delta C_{L}^{\\kappa g}$ (mocks)', color='b', marker='^', markersize=markersize)

        if using_kappa_input:
            clkg_label = '$C_{L}^{\\hat{\\kappa}\\kappa_{input}}$'
            dclkg_label = '$C_{L}^{I^2\\kappa_{input}}$ (mocks)'
        else:
            clkg_label = '$C_{L}^{\\hat{\\kappa} g}$'
            dclkg_label = '$\\Delta C_{L}^{\\kappa g}$ (mocks)'

        if plot_sem:
            plt.errorbar(lC, y_clkg, yerr=sem_clkg, label=clkg_label, color='r', marker='x', markersize=markersize, capsize=capsize)
            plt.errorbar(lC, y_dclkg, yerr=sem_dclkg, label=dclkg_label, color='b', marker='^', markersize=markersize, capsize=capsize)
        else:
            plt.plot(lC, y_clkg, label=clkg_label, color='r', marker='x', markersize=markersize)
            plt.plot(lC, y_dclkg, label=dclkg_label, color='b', marker='^', markersize=markersize)

    # If lensing was applied, show the input C_L^kappa spectrum
    if 'enable_lensing' in res and res['enable_lensing']:
        from lensing_utils import build_kappa_power_spectrum
        kappa_amp = res.get('kappa_amplitude', 1.0)
        f_kappa = build_kappa_power_spectrum(ell_min=1, ell_max=2.*lC[-1], clkg_scale=1.0)
        clk_input = f_kappa(lC) * kappa_amp
        print('clk input is ', clk_input)
        plt.plot(lC, clk_input, label=f'Input $C_{{L}}^{{\\kappa}}$ (amp={kappa_amp})', color='purple', linewidth=2.5, linestyle='--', zorder=5)

    plt.axhline(refclkg, label='$\\Delta C_{L}^{\\kappa g}=\\Omega_{\\rm pix}C_{\\ell}^{I^2g}/2C_{\\ell}^{II}$', color='k', linestyle='dashed')
    if lMax is not None:
        plt.axvline(lMax, color='k', linestyle='solid')
    if lMin is not None:
        plt.axvline(lMin, color='k', linestyle='solid')

    plt.xlabel('$L$', fontsize=14)

    plt.xscale('log')
    if ylogscale:
        plt.yscale('log')
    plt.legend(ncol=ncol, bbox_to_anchor=bbox_to_anchor, fontsize=legend_fs, loc=loc)
    # plt.xlim(300, 8e4)
    plt.xlim(xlim)

    # plt.ylim(ylim)
    if ylim is None:
        ylim_plot = [rat_min*res['analytic_bias'], rat_max*res['analytic_bias']]
    else:
        ylim_plot = ylim
    plt.ylim(ylim_plot)

#     plt.tight_layout()
    if show:
        plt.show()

    return fig


def plot_cl(lb, cl, clerr=None, figsize=(5, 4), title='Observed power spectrum', 
           return_fig=False, lab_fs=14):

    fig = plt.figure(figsize=figsize)
    plt.title('Observed power spectrum')
    plt.errorbar(lb, cl, yerr=clerr, fmt='o', color='k')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$\\ell$', fontsize=lab_fs)
    plt.ylabel('$C_{\\ell}$', fontsize=lab_fs)
    plt.show()
    
    if return_fig:
        return fig

def plot_clx_clkk_clkg(lC, clx, clkk, clkg, clxerr, clkkerr, clkgerr, 
                      ylim=[1e-12, 1e-6], xlim=[1e2, 1e5], lab_fs=14, 
                      figsize=(5, 4), return_fig=False, 
                      clx_lab='Recovered $\\times$ input $\\kappa$', 
                      clkk_lab='$C_{L}^{\\kappa}$', 
                      clkg_lab='$C_{\\ell}^{\\hat{\\kappa} g}$'):

    fig = plt.figure(figsize=figsize)
    plt.errorbar(lC, clx, yerr=clxerr, fmt='o', color='k', label=clx_lab)
    plt.errorbar(lC, clkk, yerr=clkkerr, fmt='o', color='r', label=clkk_lab)
    plt.errorbar(lC, clkg, yerr=clkgerr, fmt='o', color='b', label=clkg_lab)
    plt.legend(fontsize=12)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(alpha=0.3)
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.xlabel('L', fontsize=lab_fs)
    plt.ylabel('$C_L$', fontsize=lab_fs)
    plt.show()
    
    if return_fig:
        return fig

def plot_true_recovered_clkk(res, figsize=(8, 4), inst=1, lensmode='unlensed', mockstr='photnoiseonly_maskJ16', 
                            xlim=[1e2, 1e5], ylim_clkk=[1e-12, 1e-7], ylim_ratio=[1e-2, 1], 
                            lab_fs=16, gridalpha=0.5):
    
    fieldav_clkk_est = np.mean(res['all_clx'], axis=1)
    fieldav_clkkerr_est = np.std(res['all_clx'], axis=1)/np.sqrt(4)
    fieldav_clkk = np.mean(res['all_clkk'], axis=1)
    fieldav_response = fieldav_clkk_est/fieldav_clkk
    
    fig = plt.figure(figsize=figsize)
    plt.suptitle('CIBER '+str(lam_dict[inst])+' $\\mu$m, '+lensmode+', '+mockstr, fontsize=16)
    plt.subplot(1,2,1)
    for x in range(5):
        if x==0:
            truthlab = 'Input $C_L^{\\kappa \\kappa}$'
            recovlab = 'Recovered $C_L^{\\hat{\\kappa}\\kappa}$'
        else:
            truthlab, recovlab = None, None
        plt.plot(lC, fieldav_clkk[x], color='k', alpha=0.5, label=truthlab)
        plt.errorbar(lC, fieldav_clkk_est[x], yerr=fieldav_clkkerr_est[x], fmt='o', color='C'+str(x), label=recovlab)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('L', fontsize=lab_fs)
    plt.ylabel('$C_L$', fontsize=lab_fs)
    plt.grid(alpha=gridalpha)
    plt.xlim(xlim)
    plt.ylim(ylim_clkk)
    plt.legend()

    plt.subplot(1,2,2)

    mean_response = np.mean(fieldav_response, axis=0)

    for x in range(5):
        plt.plot(lC, fieldav_response[x], color='C'+str(x), alpha=0.3)
    plt.errorbar(lC, mean_response, yerr=np.abs(np.std(fieldav_response, axis=0)/np.sqrt(5)), fmt='o', color='k', capsize=3)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('L', fontsize=lab_fs)
    plt.ylabel('$C_L^{\\hat{\\kappa}\\kappa}/C_L^{\\kappa \\kappa}$', fontsize=lab_fs)
    plt.xlim(xlim)
    plt.ylim(ylim_ratio)
    plt.grid(alpha=gridalpha)
    plt.tight_layout()
    plt.show()
    
    return fig
    
def plot_skewcl_I2G(ifield_list, lC, all_cl_bis_iig, all_clerr_bis_iig, textstr=None,
                    figsize=(5, 4), capsize=2.5, lab_fs=12, text_fs=14, textxpos=1e3, textypos=1e-4, 
                   xlim=[100, 1e5], ylim=[1e-8, 1e-3], ylabel='$C_{\\ell}^{I^2 g}$'):
    
    lam_dict = dict({1:1.1, 2:1.8})

    fig = plt.figure(figsize=figsize)
    for fieldidx, ifield in enumerate(ifield_list):
        plt.errorbar(lC, all_cl_bis_iig[fieldidx], yerr=all_clerr_bis_iig[fieldidx], color='C'+str(ifield-4), fmt='o', capsize=capsize)

    plt.ylabel(ylabel, fontsize=lab_fs)
    plt.xlabel('$\\ell$', fontsize=lab_fs)
    
    if textstr is not None:
        plt.text(textxpos, textypos, textstr, fontsize=text_fs)

    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.grid(alpha=0.5)
    # plt.savefig(figpath+'/ciber_bispectrum_IsqG_TM'+str(inst0)+'.pdf', bbox_inches='tight')
    plt.show()
    
    return fig


def plot_clkg_bias_from_skew(ifield_list, lC, all_dclkg_iig, all_dclkg_err_iig, textstr=None,
                    figsize=(5, 4), capsize=2.5, lab_fs=12, text_fs=14, textxpos=1e3, textypos=1e-4, 
                   xlim=[100, 1e5], ylim=[1e-12, 1e-6], ylabel='$\\Delta C_{\\ell}^{\\kappa g}$'):
    
    fig = plt.figure(figsize=figsize)
    for fieldidx, ifield in enumerate(ifield_list):
        plt.errorbar(lC, all_dclkg_iig[fieldidx], yerr=all_dclkg_err_iig[fieldidx], color='C'+str(ifield-4), fmt='o', capsize=capsize)

    plt.ylabel(ylabel, fontsize=lab_fs)
    plt.xlabel('$\\ell$', fontsize=lab_fs)
    
    if textstr is not None:
        plt.text(textxpos, textypos, textstr, fontsize=text_fs)

    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.grid(alpha=0.5)
    # plt.savefig(figpath+'/ciber_bispectrum_IsqG_TM'+str(inst0)+'.pdf', bbox_inches='tight')
    plt.show()
    
    return fig


def plot_2h_bispectrum_prediction(lC, clkg_bias_predicted_2h, 
                                   C_ell_II=None, C_ell_gg=None, C_ell_Ig=None,
                                   all_C_ell_II_perfield=None, all_C_ell_gg_perfield=None,
                                   textstr=None, figsize=(12, 8), capsize=2.5, 
                                   lab_fs=14, text_fs=12, 
                                   xlim=[100, 1e5]):
    """
    Plot 2-halo bispectrum prediction and input power spectra.
    
    Parameters
    ----------
    lC : array
        Multipole bin centers
    clkg_bias_predicted_2h : array
        Predicted 2h bias to kappa-galaxy cross-spectrum
    C_ell_II : array, optional
        Field-averaged CIB auto-spectrum
    C_ell_gg : array, optional
        Field-averaged galaxy auto-spectrum
    C_ell_Ig : array, optional
        Field-averaged CIB-galaxy cross-spectrum
    all_C_ell_II_perfield : array, optional
        Per-field CIB auto-spectra [n_fields, n_bins]
    all_C_ell_gg_perfield : array, optional
        Per-field galaxy auto-spectra [n_fields, n_bins]
    textstr : str, optional
        Text to display on figure
    figsize : tuple
        Figure size
    capsize : float
        Error bar cap size
    lab_fs : int
        Label font size
    text_fs : int
        Text font size
    xlim : list
        x-axis limits
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    
    n_panels = 1  # At least the predicted bias
    if C_ell_II is not None or C_ell_gg is not None or C_ell_Ig is not None:
        n_panels = 2  # Add power spectra panel
    
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize)
    if n_panels == 1:
        axes = [axes]
    
    # Panel 1: Predicted 2h bias
    ax = axes[0]
    ax.plot(lC, np.abs(clkg_bias_predicted_2h), 'o-', color='C0', 
            markersize=6, lw=2, label='2h prediction')
    ax.axhline(0, color='k', ls=':', alpha=0.5)
    
    ax.set_ylabel('$|\\Delta C_{\\ell}^{\\kappa g}|$ (2h)', fontsize=lab_fs)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(xlim)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=lab_fs-2)
    
    if textstr is not None:
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, 
                fontsize=text_fs, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    if n_panels == 1:
        ax.set_xlabel('$\\ell$', fontsize=lab_fs)
    else:
        plt.setp(ax.get_xticklabels(), visible=False)
    
    # Panel 2: Input power spectra (if provided)
    if n_panels == 2:
        ax = axes[1]
        
        # Plot per-field power spectra if available
        if all_C_ell_gg_perfield is not None:
            for fieldidx in range(len(all_C_ell_gg_perfield)):
                ax.plot(lC, all_C_ell_gg_perfield[fieldidx], 
                       ls=':', color='C1', alpha=0.3, lw=1)
        if all_C_ell_II_perfield is not None:
            for fieldidx in range(len(all_C_ell_II_perfield)):
                ax.plot(lC, all_C_ell_II_perfield[fieldidx], 
                       ls=':', color='C2', alpha=0.3, lw=1)
        
        # Plot field-averaged power spectra
        if C_ell_gg is not None:
            ax.plot(lC, C_ell_gg, 'o-', color='C1', markersize=5, 
                   lw=2, label='$C_{\\ell}^{gg}$ (avg)')
        if C_ell_II is not None:
            ax.plot(lC, C_ell_II, 's-', color='C2', markersize=5, 
                   lw=2, label='$C_{\\ell}^{II}$ (avg)')
        if C_ell_Ig is not None:
            ax.plot(lC, np.abs(C_ell_Ig), '^-', color='C3', markersize=5, 
                   lw=2, label='$|C_{\\ell}^{Ig}|$ (avg)')
        
        ax.set_ylabel('$C_{\\ell}$ [Jy$^2$ sr$^{-2}$]', fontsize=lab_fs)
        ax.set_xlabel('$\\ell$', fontsize=lab_fs)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(xlim)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=lab_fs-2, loc='best')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def plot_pix_window_fn(lb, tl_pix, ylim=[1e-3, 2e0], figsize=(5, 4), \
                      xlim=[1e2, 1e5]):
    fig = plt.figure(figsize=figsize)
    plt.plot(lb, tl_pix)
    plt.ylabel('Pixel window function')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.show()
    return fig

def plot_cl_kappas(lCen, all_cl, all_clerr=None, colors=None, markersize=5, capsize=3, \
				  xlim=[100, 1e5], ylim=[1e-12, 1e-5], labels=None, return_fig=True, figsize=(5, 4), \
				  lab_fs=14):
	
	if colors is None:
		colors = ['b', 'r', 'magenta']
	
	fig = plt.figure(figsize=figsize)

	for x in range(len(all_cl)):
		
		if labels is not None:
			label = labels[x]
		else:
			label = None
			
		plt.errorbar(lCen, all_cl[x], yerr=all_clerr[x], fmt='o', color=colors[x], markersize=markersize, capsize=capsize, label=label)

	plt.xscale('log')
	plt.yscale('log')
	plt.ylabel('$C_{\\ell}^{\\kappa}$', fontsize=lab_fs)
	plt.xlabel('$\\ell$', fontsize=lab_fs)
	plt.grid(alpha=0.3)
	plt.legend()
	plt.xlim(xlim)
	plt.ylim(ylim)
	plt.show()
	
	return fig


def plot_kappa_gal_cross(lC, clxs, clxerrs, inst0, inst1, colors=None, labels=None, figsize=(5, 4), \
						markersize=5, capsize=3, xlim=[100, 1e5], ylim=[1e-12, 1e-4]):

	if colors is None:
		colors = ['b', 'r', 'magenta']


	fig = plt.figure(figsize=figsize)
	for k in range(len(clxs)):

		if labels is not None:
			label = labels[k]
		else:
			label = None
		plt.errorbar(lC, clxs[k], yerr=clxerrs[k], fmt='o', color=colors[k], markersize=markersize, capsize=capsize, label=label)

	plt.legend()
	plt.ylabel('$C_{\\ell}^{\\kappa g}$', fontsize=12)
	plt.grid(alpha=0.3)
	plt.xlim(xlim)
	plt.ylim(ylim)
	plt.xscale('log')
	plt.yscale('log')
	plt.show()

	return fig



def plot_kappa_gal_all_fields_singleband(inst, lC, all_clx, all_clxerr, all_fieldav_clx, all_fieldav_clxerr, ifield_list=[4, 6, 7, 8], lams=[1.1, 1.8], fieldnames = ['elat10', 'Bootes B', 'Bootes A', 'SWIRE'], \
                             textxpos=3e3, textypos=1e-7, ylim=[1e-11, 1e-6], catname='unWISE', text_fs=12, dirpath=None, \
                             xlim=[100, 1e5], bbox_to_anchor=[1.4, 1.3], figsize=(5, 4), alpha=0.2):
    
    
    pf = lC*(lC+1)/(2*np.pi)
    
    # plot the thing

    fig = plt.figure(figsize=figsize)

    plt.text(textxpos, textypos, '$\\kappa^{'+str(lams[inst-1])+'}\\times$ '+catname,fontsize=text_fs, color='k')
    for fieldidx, ifield in enumerate(ifield_list):
        plt.errorbar(lC, all_clx[0][fieldidx], yerr=all_clxerr[0][fieldidx], label=fieldnames[fieldidx], fmt='o', color='C'+str(ifield-4), markersize=3., alpha=0.3, capsize=2.5)

    plt.errorbar(lC, all_fieldav_clx[0], yerr=all_fieldav_clxerr[0], fmt='o', color='k', label='Field average', markersize=4.0, capsize=3.5)
    plt.plot(lC, all_fieldav_clxerr[0], color='r', linestyle='dashed', alpha=0.2, linewidth=1., label='1$\\sigma$ uncertainty')

    plt.legend(ncol=3, bbox_to_anchor=bbox_to_anchor)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.grid(alpha=0.3)

    plt.ylabel('$C_{\\ell}^{\\kappa g}$', fontsize=12)
    plt.xlabel('$\\ell$', fontsize=12)

    plt.show()
    
    return fig

def plot_kappa_gal_all_fields(lC, all_clx, all_clxerr, all_fieldav_clx, all_fieldav_clxerr, ifield_list=[4, 6, 7, 8], figsize=(9, 3), lams=[1.1, 1.8], fieldnames = ['elat10', 'Bootes B', 'Bootes A', 'SWIRE'], \
							 textypos=1e-7, ylim=[1e-11, 1e-6], catname='unWISE', text_fs=12, dirpath=None, \
							 inst0=1, inst1=2, xlim=[100, 1e5], bbox_to_anchor=[1.4, 1.3]):
	
	
	pf = lC*(lC+1)/(2*np.pi)
	
	# plot the thing

	fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(9, 3), sharey=True)

	for inst in [1, 2]:
		ax[inst-1].text(1e3, textypos, '$\\kappa^{'+str(lams[inst-1])+'}\\times$ '+catname,fontsize=text_fs, color='k')

		for fieldidx, ifield in enumerate(ifield_list):
			ax[inst-1].errorbar(lC, all_clx[inst-1][fieldidx], yerr=all_clxerr[inst-1][fieldidx], label=fieldnames[fieldidx], fmt='o', color='C'+str(ifield-4), markersize=3., alpha=0.3, capsize=2.5)

		ax[inst-1].errorbar(lC, all_fieldav_clx[inst-1], yerr=all_fieldav_clxerr[inst-1], fmt='o', color='k', label='Field average', markersize=3.5, capsize=3)
		ax[inst-1].plot(lC, all_fieldav_clxerr[inst-1], color='r', linestyle='dashed', alpha=0.5)

		if inst==2:
			ax[inst-1].legend(ncol=3, bbox_to_anchor=bbox_to_anchor)
		ax[inst-1].set_xscale('log')
		ax[inst-1].set_yscale('log')
		ax[inst-1].set_xlim(xlim)
		ax[inst-1].set_ylim(ylim)
		ax[inst-1].grid(alpha=0.3)

		if inst==1:
			ax[inst-1].set_ylabel('$\\ell^2 C_{\ell}^{\\kappa g}/2\\pi$', fontsize=12)
			ax[inst-1].set_ylabel('$C_{\ell}^{\\kappa g}$', fontsize=12)
		ax[inst-1].set_xlabel('$\\ell$', fontsize=12)

	ax[2].text(1e3, textypos, '$\\kappa^{1.1 \\times 1.8}\\times$ unWISE',fontsize=12, color='k')

	for fieldidx, ifield in enumerate(ifield_list):
		ax[2].errorbar(lC, all_clx[2][fieldidx], yerr=all_clxerr[2][fieldidx], label=fieldnames[fieldidx], fmt='o', color='C'+str(ifield-4), markersize=3., alpha=0.3, capsize=2.5)
	ax[2].errorbar(lC, all_fieldav_clx[2], yerr=all_fieldav_clxerr[2], fmt='o', color='k', label='Field average', markersize=3.5, capsize=3)
	
	ax[2].plot(lC, all_fieldav_clxerr[2], color='r', linestyle='dashed', alpha=0.5)

	
	ax[2].set_xscale('log')
	ax[2].set_yscale('log')
	ax[2].set_xlim(100, 1e5)
	ax[2].set_ylim(ylim)
	ax[2].set_xlabel('$\\ell$', fontsize=12)
	ax[2].grid(alpha=0.3)

	plt.subplots_adjust(hspace=0, wspace=0)
	plt.show()
	
	return fig


def plot_coll_tris_and_ratio(inst, lC, all_tl, all_tlerr, all_tl_c02, all_tlerr_c02, ifield_list, \
                             figsize=(7, 3.5), capsize=3, markersize=3, xlim=[300, 1e5]):
    
    lam_dict = dict({1:1.1, 2:1.8})

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    plt.suptitle('CIBER '+str(lam_dict[inst])+' $\\mu$m', fontsize=16)
    for x in range(len(all_tl_c02)):
        ax[0].errorbar(lC, all_tl[x], yerr=all_tlerr[x], fmt='o', color='C'+str(ifield_list[x]-4), capsize=capsize, markersize=markersize)
        ax[1].errorbar(lC, all_tl_c02[x], yerr=all_tlerr_c02[x], fmt='o', color='C'+str(ifield_list[x]-4), capsize=capsize, markersize=markersize)

    for y in [0, 1]:
        ax[y].set_xscale('log')
        ax[y].set_yscale('log')

        ax[y].set_xlabel('$\\ell$', fontsize=14)
        
        ax[y].set_xlim(xlim)
        ax[y].grid(alpha=0.3)

    ax[0].set_ylabel('$\\langle \\tau^0_{\\ell, L-\\ell, \\ell, -L-\\ell}\\rangle$', fontsize=14)

    ax[1].set_ylabel('$\\langle \\tau^0_{\\ell, L-\\ell, \\ell, -L-\\ell}\\rangle/(C^0)^2$', fontsize=14)
    ax[0].set_ylim(1e-22, 1e-16)
    ax[1].set_ylim(1e-14, 1e-7)

    plt.tight_layout()
    plt.show()
    
    return fig


def plot_clkg_with_bispectrum_bias(results_dict, forecast_cl_auto=None, igl_tot_frac=None, 
                                   lmax_snr=1e4, textxpos=3e3, textypos=1e-7, 
                                   bbox_to_anchor=[0.0, 1.4], xlim=[100, 1e5], ylim=[5e-12, 5e-7],
                                   text_fs=16, figsize=(5, 4), mag_lim=17.0, show_forecast=True, 
                                   legend_fs=12, capsize=2.5, markersize=4, 
                                   lMax=None, lMin=None, plot_dclkg_from_bispectrum=True, logyscale=True, 
                                   linestyle='dashed', show_fiducial_clkg=True, catname='WISE', addstr='unWISE_neo8'):
    """
    Plot kappa-galaxy cross-spectrum with bispectrum-derived bias from results dictionary.
    
    Parameters:
    -----------
    results_dict : dict
        Results dictionary from run_flatskyqe_ciber containing:
        - lC, ifield_list, inst0, inst1, catname
        - all_clx, all_clxerr (per-field measurements)
        - all_fieldav_clx, all_fieldav_clxerr (field-averaged)
        - all_dclkg_iig, all_dclkg_err_iig (bispectrum bias per field)
    forecast_cl_auto : dict, optional
        Dictionary with keys 'lC', 'clxerr_av_forecast' for forecast uncertainty
    igl_tot_frac : list or array, optional
        Fraction of IGL to total CIB signal for [TM1, TM2]. Default [0.5, 0.8]
    lmax_snr : float
        Maximum ell for SNR calculation
    textxpos, textypos : float
        Text position for labels
    bbox_to_anchor : list
        Legend position
    xlim, ylim : list
        Plot limits
    text_fs : int
        Text font size
    figsize : tuple
        Figure size
    mag_lim : float
        Magnitude limit for labels
    show_forecast : bool
        Whether to show forecast uncertainty curve
    plot_dclkg_from_bispectrum : bool
        Whether to plot the bispectrum-derived bias
    logyscale : bool
        Whether to use logarithmic scale for y-axis
    show_fiducial_clkg : bool
        Whether to show fiducial/theory clkg prediction
    catname, addstr : str
        For loading fiducial clkg if show_fiducial_clkg=True
    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    # import ciber_powerspec_pipeline as cbps
    from forecast_cib_lens import ciber_lens_forecast
    # from  import ciber_cl_forecast
    
    # cbps = CIBER_PS_Pipeline()

    ciber_field_dict = {4: 'elat10', 6: 'Bootes B', 7: 'Bootes A', 8: 'SWIRE'}
    # Set defaults
    if igl_tot_frac is None:
        igl_tot_frac = [0.5, 0.8]
    
    # Extract from results dict
    lC = results_dict['lC']
    ifield_list = results_dict['ifield_list']
    inst0 = results_dict['inst0']
    inst1 = results_dict.get('inst1', None)
    catname = results_dict['catname']
    single_band = results_dict.get('single_band', True)
    
    all_clkg  = results_dict['all_clkg']
    all_clkgerr = results_dict['all_clkgerr']
    all_fieldav_clkg = results_dict['all_fieldav_clkg']
    all_fieldav_clkgerr = results_dict['all_fieldav_clkgerr']
    
    # Bispectrum bias (if computed)
    all_dclkg_iig = results_dict.get('all_dclkg_iig', None)
    all_dclkg_err_iig = results_dict.get('all_dclkg_err_iig', None)
    
    bandstrs = ['J', 'H']
    lams = [1.1, 1.8]
    
    # Determine which instruments to plot
    if single_band:
        inst_list = [inst0]
        n_panels = 1
    else:
        inst_list = [inst0, inst1] if inst1 is not None else [inst0]
        n_panels = len(inst_list)
    
    fig = plt.figure(figsize=figsize)
    
    for panel_idx, inst in enumerate(inst_list):
        
        # Load fiducial clkg if requested
        if show_fiducial_clkg:
            # try:
            clf = ciber_lens_forecast()
            clf.load_clx(clkg_scale=0.5)
            clx_fid = clf.clx
            # except Exception as e:
            #     print(f"Warning: Could not load fiducial clkg for inst {inst}: {e}")
            #     show_fiducial_clkg = False
        
        # Get forecast if available
        if show_forecast and forecast_cl_auto is not None:
            clxerr_av_forecast = forecast_cl_auto.get('clxerr_av_forecast', None)
            lcen = forecast_cl_auto.get('lC', lC)
        else:
            clxerr_av_forecast = None
            lcen = None
        
        # Get per-field and field-averaged cross-spectra
        clkg_perf = all_clkg[panel_idx]
        clkgerr_perf = all_clkgerr[panel_idx]
        clkg_av = all_fieldav_clkg[panel_idx]
        clkgerr_av = all_fieldav_clkgerr[panel_idx]
        
        # Calculate SNR
        lbmask = (lC > 1000) * (lC < lmax_snr)
        snr = clkg_av / clkgerr_av
        tot_snr = np.sqrt(np.nansum(snr[lbmask]**2))
        print(f'For TM{inst}, total SNR between 1000 < ell < {lmax_snr}: {tot_snr:.2f}')
        
        plt.subplot(1, n_panels, panel_idx + 1)
        
        # Plot per-field measurements
        for fieldidx, ifield in enumerate(ifield_list):
            if len(ifield_list)==1:
                color='r'
                alpha=1.0
                label = 'QE ('+ciber_field_dict[ifield]+')'
            else:
                color='C'+str(ifield-4)
                alpha=0.3
                if fieldidx==0:
                    label = 'QE (per field)'
                else:
                    label = None


            plt.errorbar(lC, clkg_perf[fieldidx], yerr=clkgerr_perf[fieldidx], 
                        label=label, fmt='o', 
                        color=color, markersize=markersize, alpha=alpha, capsize=capsize)
        
        # Plot field average
        if len(ifield_list) > 1:
            plt.errorbar(lC, clkg_av, yerr=clkgerr_av, fmt='o', color='k', 
                        label='QE (field average)', markersize=1.2*markersize, capsize=3, capthick=1.5)
            plt.plot(lC, clkgerr_av, color='k', linestyle='dashed', alpha=0.5, 
                    linewidth=1., label='1$\\sigma$ uncertainty (data)')
            
        # Plot forecast if available
        if clxerr_av_forecast is not None and lcen is not None:
            plt.plot(lcen, clxerr_av_forecast, color='b', marker='.', 
                    linestyle='dashed', linewidth=1., 
                    label='1$\\sigma$ uncertainty\\n(forecast, Gauss.)')
        
        # Plot fiducial clkg if loaded
        if show_fiducial_clkg and 'clx_fid' in locals():
            plt.plot(clf.lrange, clx_fid, color='b', linestyle='solid', 
                    linewidth=2., alpha=0.7,
                    label='Fiducial $C_{\\ell}^{\\kappa g}$')
        
        # Plot bispectrum bias if available
        if all_dclkg_iig is not None and plot_dclkg_from_bispectrum:
            mean_dclkg = np.mean(all_dclkg_iig, axis=0)
            
            if len(ifield_list) > 1:
                std_dclkg = np.std(all_dclkg_iig, axis=0)/np.sqrt(len(ifield_list))
            else:
                std_dclkg = all_dclkg_err_iig[0]
            # plt.plot(lC, mean_dclkg, color='r', 
            #         label='$\\Delta C_{\\ell}^{\\kappa g}$ (bispectrum)')
            
            plt.errorbar(lC, mean_dclkg, yerr=std_dclkg, color='b', 
                    label='$\\Delta C_{\\ell}^{\\kappa g}$ (bispectrum)', capsize=3, fmt='o', 
                    markersize=2.0*markersize, alpha=0.7, marker='*', capthick=1.5)
        
        if lMax is not None:
            plt.axvline(lMax, color='k', linestyle=linestyle)
        if lMin is not None:
            plt.axvline(lMin, color='k', linestyle=linestyle)

        # Labels and formatting
        # if panel_idx == 0:
            # plt.legend(ncol=3, bbox_to_anchor=bbox_to_anchor, loc=2, fontsize=legend_fs)
        plt.legend(ncol=1, loc=1, fontsize=legend_fs)
        plt.ylabel('$C_{\\ell}^{\\kappa g}$', fontsize=14)
        
        plt.xscale('log')

        if logyscale:
            plt.yscale('log')

        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.grid(alpha=0.3)
        plt.xlabel('$L$', fontsize=16)
        plt.tick_params(labelsize=12)
        
        # Optional: add title with instrument info
        # plt.title(f'$\\kappa^{{{lams[inst-1]}}}\\times$ {catname}', fontsize=text_fs)
        # plt.title('$\\kappa^{'+str(lams[inst-1])+'}\\times$'+catname, fontsize=text_fs)
        plt.title('CIBER $\\kappa_{'+str(lams[inst-1])+' um} \\times $ '+str(catname)+' $\\delta_g$', fontsize=text_fs)
    plt.tight_layout()
    
    return fig