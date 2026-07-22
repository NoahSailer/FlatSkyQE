from scipy.interpolate import interp1d
import matplotlib
from scipy.ndimage import gaussian_filter1d
import numpy as np
import sys
import os
import config

import flat_map
import weight
from flat_map import *
from weight import *



# def compute_weighted_cl(indiv_cl, field_weights):
    
#     n_ps_bin = len(indiv_cl[0])
#     field_average_cl, field_average_dcl = [np.zeros_like(indiv_cl[0]) for x in range(2)]
    
#     for n in range(n_ps_bin):
#         field_weights[:,n] /= np.sum(field_weights[:,n])
#         field_average_cl[n] = np.average(indiv_cl[:,n], weights=field_weights[:,n])
#         neff_indiv = compute_Neff(field_weights[:,n])

#         psvar_indivbin = np.sum(field_weights[:,n]*(indiv_cl[:,n] - field_average_cl[n])**2)*neff_indiv/(neff_indiv-1.)
#         field_average_dcl[n] = np.sqrt(psvar_indivbin/neff_indiv)
        
#     return field_average_cl, field_average_dcl


# def inv_var_weights(lb, all_dcl, plot=False):
    
#     field_weights = 1/all_dcl**2
    
#     nfield, n_ps_bin = all_dcl.shape[0], all_dcl.shape[1]
    
#     for n in range(n_ps_bin):
#         field_weights[:,n] /= np.sum(field_weights[:,n])
        
        
#     if plot:
#         plt.figure(figsize=(5, 4))
        
#         for n in range(nfield):
#             plt.plot(lb, field_weights[n], color='C'+str(n))
            
#         plt.xlabel('$\\ell$', fontsize=12)
#         plt.ylabel('Field weights', fontsize=12)
#         plt.xscale('log')
#         plt.show()
        
#     return field_weights
    

# def compute_Neff(weights):
#     N_eff = np.sum(weights)**2/np.sum(weights**2)

#     return N_eff

# def fCtot_identity(l):
#     return 1.

# def iter_sigma_clip_mask(image, sig=5, nitermax=10, mask=None):
#     # this version makes copy of the mask to be modified, rather than modifying the original
#     # image assumed to be 2d
#     iteridx = 0

#     summask = image.shape[0]*image.shape[1]

#     if mask is not None:
#         running_mask = mask.copy()
#     else:
#         running_mask = np.ones_like(image)

#     while iteridx < nitermax:

#         new_mask = sigma_clip_maskonly(image, previous_mask=running_mask, sig=sig)

#         if np.sum(running_mask*new_mask) < summask:
#             running_mask *= new_mask
#             summask = np.sum(running_mask)
#         else:
#             return running_mask

#         iteridx += 1

#     return running_mask

# def sigma_clip_maskonly(vals, previous_mask=None, sig=5):
    
#     valcopy = vals.copy()
#     if previous_mask is not None:
#         valcopy[previous_mask==0] = np.nan
#         sigma_val = np.nanstd(valcopy)
#     else:
#         sigma_val = np.nanstd(valcopy)
    
#     abs_dev = np.abs(vals-np.nanmedian(valcopy))
#     mask = (abs_dev < sig*sigma_val).astype(int)

#     return mask

# def load_delta_g_maps(catname, inst, addstr, gal_basepath=None):

#     if gal_basepath is None:
#         gal_basepath = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/gal_density/'+catname+'/'

#     gal_fpath = gal_basepath+'gal_density_'+catname+'_TM'+str(inst)
#     if addstr is not None:
#         gal_fpath += '_'+addstr
#     gal_fpath +='.fits'
#     gal_densities = fits.open(gal_fpath)

#     noise_base_path = gal_basepath+'cross_noise/'

#     return gal_densities, noise_base_path
