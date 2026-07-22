import numpy as np
import os
from flat_map import *
from weight import *

def load_delta_g_maps(catname, inst, addstr, gal_basepath=None):

    if gal_basepath is None:
        gal_basepath = config.ciber_basepath+'data/fluctuation_data/TM'+str(inst)+'/gal_density/'+catname+'/'

    gal_fpath = gal_basepath+'gal_density_'+catname+'_TM'+str(inst)
    if addstr is not None:
        gal_fpath += '_'+addstr
    gal_fpath +='.fits'
    gal_densities = fits.open(gal_fpath)

    noise_base_path = gal_basepath+'cross_noise/'

    return gal_densities, noise_base_path

def set_kappa_fpaths(inst0, inst1, ifield, tailstr, output_basepath='./output/', calc_ciber_cross=True):
    
    # make result directory
    
    dirpath = output_basepath + tailstr
    
    figpath = dirpath+'/figs/'
    
    if not os.path.isdir(dirpath):
        print('making directory ', dirpath)
        os.makedirs(dirpath)
        
    if not os.path.isdir(figpath):
        print('making directory ', figpath)
        os.makedirs(figpath)
        
    path_k1 = dirpath+'/kappa_est_CIBER_TM'+str(inst0)+'_ifield_'+str(ifield)+'_'+tailstr+'.txt'
    path_k2 = dirpath+'/kappa_est_CIBER_TM'+str(inst1)+'_ifield_'+str(ifield)+'_'+tailstr+'.txt'
    
    paths = [path_k1, path_k2]
    
    if calc_ciber_cross:
        path_k1k2 = dirpath+'/kappa_est_CIBER_TM'+str(inst0)+'_TM'+str(inst1)+'_ifield'+str(ifield)+'_'+tailstr+'.txt'
        
    else:
        path_k1k2 = None
    
    paths.append(path_k1k2)
    
    return dirpath, figpath, paths

def save_skewcl_I2G_files(lC, clx_perfield, clxerr_perfield, dirpath, inst0, inst1=None, catname='unWISE', addstr=None):
    
    
    fpath = dirpath+'/skewcl_I2G_fieldav_CIBER_TM'+str(inst0)
    if inst1 is not None:
        fpath+= '_TM'+str(inst1)
    fpath += '_'+catname
    if addstr is not None:
        fpath+='_'+addstr  
    fpath += '.npz'
    
    print('Saving to ', fpath)

    np.savez(fpath, lC=lC, clx_perfield=clx_perfield, clxerr_perfield=clxerr_perfield)
    
    return fpath   


def save_kappa_g_cl_files(lC, clx_perfield, clxerr_perfield, field_av_clx, field_av_clxerr, field_weights, \
                         dirpath, inst0, inst1=None, catname='unWISE', addstr=None):
    
    
    fpath = dirpath+'/kappa_gal_fieldav_CIBER_TM'+str(inst0)
    if inst1 is not None:
        fpath+= '_TM'+str(inst1)
    fpath += '_'+catname
    if addstr is not None:
        fpath+='_'+addstr  
    fpath += '.npz'
    
    print('Saving to ', fpath)

    np.savez(fpath, lC=lC, clx_perfield=clx_perfield, clxerr_perfield=clxerr_perfield,\
             field_av_clx=field_av_clx, field_av_clxerr=field_av_clxerr, field_weights=field_weights)
    
    return fpath

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