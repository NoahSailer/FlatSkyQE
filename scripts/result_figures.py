import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import from ciber base directory (/Documents/ciber/)



from prep_ciber_dat_lens import *
from forecast_cib_lens import *
from kappa_auto_cross_fns import *
from kappa_plotting_fns import *
from mock_lens_test import *
from lens_response import *
from forecast_cib_lens import plot_integrated_snr_vs_survey_params




def snr_shear_vs_mag_forecasts(L_values = [300, 1000, 3000, 10000], 
                               ellmin=300, ellmax=100000, N_tris=1e-8, show_squeezed=False, 
                               inst=1, figsize=(5, 4.5), savefig=False, save_dir='/Documents/ciber/figures/', 
                               ylim=[1e4, 1e9], bbox_anchor=[0.05, 1.35], legend_fs=11):

    print('Loading CIBER F25 spectrum for TM'+str(inst)+'for SNR forecasts...')
    cell_interp, lb, cl = load_ciber_f25_spectrum(inst=inst, smooth=True, smooth_model='spline')

    fig = plot_snr_shear_vs_mag(
        L_values=L_values,
        A=None, B=None, n=None,           # ignored when cell_interp is set
        ellmin=ellmin, ellmax=ellmax,
        N_tris=N_tris,
        show_squeezed=show_squeezed,
        cell_interp=cell_interp,
        exact_response=False,
        ylim=ylim,
        bbox_anchor=bbox_anchor,
        legend_fs=legend_fs,
        figsize=figsize, 
        show_mag_no_nltris = True
    )

    if savefig:
        save_path = os.path.join(save_dir, 'snr_shear_vs_mag_forecast.pdf')
        print('Saving figure to: ', save_path)
        fig.savefig(save_path, bbox_inches='tight')



