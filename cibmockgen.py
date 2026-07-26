import flat_map
import weight

from flat_map import *
from universe import *
import halo_fit
from halo_fit import *
from weight import *

import cmb
from cmb import *

from lens_data_file_utils import *

import config
import sys
import os
import matplotlib
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
import numpy as np

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
# Add the parent directory to sys.path
sys.path.append(parent_dir)
from ciber.core.powerspec_pipeline import *
from ciber.core.ps_pipeline_go import *
from ciber.io.ciber_data_utils import *
from ciber.plotting.plotting_fns import plot_map
from ciber.theory.helgason_model import *
from ciber.mocks.lognormal import *


def genGRF_from_Cl(self, fCl, seed=None):
    rng = np.random.default_rng(seed)
    A = self.sizeX * self.sizeY

    # full complex array shape for rfft grid
    shape = self.l.shape
    z = rng.normal(size=shape) + 1j * rng.normal(size=shape)

    # enforce pure-real modes on self-conjugate lines (ky=0 and Nyquist if present)
    z[:, 0] = rng.normal(size=self.nX) + 0j
    if self.nY % 2 == 0:
        z[:, -1] = rng.normal(size=self.nX) + 0j

    Cl = np.array(fCl(self.l))
    Cl = np.nan_to_num(Cl, nan=0.0, posinf=0.0, neginf=0.0)
    Cl[self.l == 0] = 0.0

    kfour = np.sqrt(0.5 * A * Cl) * z
    kfour[:, 0] *= np.sqrt(2.0)
    if self.nY % 2 == 0:
        kfour[:, -1] *= np.sqrt(2.0)

    return kfour

def generate_cib_map(map_size=1024, n_cib_per_pixel=0.01, n_gal_per_pixel=0.01,
                     s_min=1.0, s_max=100.0, alpha=2.5, seed=42, mock_sim_fpath=False, fieldidx=0, ciber_inst=1, 
                     apply_mask=False, dx=None, dy=None, mu=None):
    """
    Generates a 2D CIB intensity map with Poisson-distributed point sources.

    Args:
        map_size (int): The side length of the square map in pixels.
        n_cib_per_pixel (float): The average number of CIB sources per pixel.
        s_min (float): The minimum flux of the sources.
        s_max (float): The maximum flux of the sources.
        alpha (float): The negative power-law index for the flux distribution P(s) ~ s^(-alpha).
        seed (int): A seed for the random number generator for reproducibility.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The 2D intensity map (map_size x map_size).
            - np.ndarray: An array of fluxes for all generated sources.
    """
    rng = np.random.default_rng(seed)

    mask = np.ones((map_size, map_size))
    if mock_sim_fpath is not None:
        mock_dat = np.load(mock_sim_fpath, allow_pickle=True)
        cib_map, kappa_map, total_signal,\
             counts_map, tracer_cat = [mock_dat[key][fieldidx] for key in ['cib_maps', 'kappa_maps', 'total_signal', 'galdens', 'tracer_cats']]
        if apply_mask:
            mask = mock_dat['masks'][fieldidx]

        print('tracer cat has shape ', tracer_cat.shape)
        mags = tracer_cat[:,3]

        cmock = ciber_mock()

        fluxes = np.array(cmock.mag_2_nu_Inu(mags, band=ciber_inst-1).value)

    
    else:

        # Total number of sources to place in the map
        total_pixels = map_size * map_size
        n_sources = int(n_cib_per_pixel * total_pixels)
        
        print(f"Generating a {map_size}x{map_size} map with {n_sources} sources...")
        
        # Create an empty map
        cib_map, counts_map = [np.zeros((map_size, map_size)) for _ in range(2)]
        
        # Generate random pixel coordinates for the sources
        x_coords = rng.integers(0, map_size, size=n_sources)
        y_coords = rng.integers(0, map_size, size=n_sources)
        
        # Generate source fluxes from a power-law distribution using inverse transform sampling
        # P(s) propto s^(-alpha) for s_min <= s <= s_max
        if alpha == 1.0:
            # Special case to avoid division by zero
            log_s = rng.uniform(np.log(s_min), np.log(s_max), size=n_sources)
            fluxes = np.exp(log_s)
        else:
            y = rng.uniform(0, 1, size=n_sources)
            s_pow = (s_max**(1 - alpha) - s_min**(1 - alpha)) * y + s_min**(1 - alpha)
            fluxes = s_pow**(1 / (1 - alpha))

        # Apply lensing: perturb positions and magnify fluxes if provided
        if dx is not None and dy is not None and mu is not None:
            # Perturb source positions by deflection fields
            x_coords_lens = x_coords + dx[y_coords, x_coords]
            y_coords_lens = y_coords + dy[y_coords, x_coords]

            # Clip perturbed positions to map boundaries
            x_coords_lens = np.clip(x_coords_lens, 0, map_size - 1).astype(int)
            y_coords_lens = np.clip(y_coords_lens, 0, map_size - 1).astype(int)

            # Magnify fluxes by magnification map evaluated at LENSED positions
            fluxes_lens = fluxes * mu[y_coords_lens, x_coords_lens]

            # fluxes_lens = fluxes * mu[y_coords, x_coords]


            x_coords = x_coords_lens
            y_coords = y_coords_lens
            fluxes = fluxes_lens

        # Place fluxes into the map at the source locations.
        # Using np.add.at for safe addition in case multiple sources fall in one pixel.

        ratio_gal_cib = int(n_cib_per_pixel/n_gal_per_pixel)

        np.add.at(cib_map, (y_coords, x_coords), fluxes)

        np.add.at(counts_map, (y_coords[::ratio_gal_cib], x_coords[::ratio_gal_cib]), 1.)

        
    return cib_map, fluxes, counts_map, mask


def make_cib_mock_simp(inst, ifield, mock_cat, base_fluc_path='../data'):

    cmock = ciber_mock()
    
    full_tracer_cat = np.array([mock_cat[:,0], mock_cat[:,1], mock_cat[:,2], mock_cat[:,2]])

    bright_src_map = cmock.make_srcmap_temp_bank(ifield, inst, full_tracer_cat.transpose(), flux_idx=-1, load_precomp_tempbank=True, \
                                                    tempbank_dirpath='../data/subpixel_psfs_TM'+str(inst)+'/')
    
    
    return bright_src_map


def make_cib_mock(inst, ifield, mock_cat, base_fluc_path='../data'):

    cmock = ciber_mock()
    
    I_arr_full = cmock.mag_2_nu_Inu(mock_cat[:,3], inst-1)

    full_tracer_cat = np.array([mock_cat[:,0], mock_cat[:,1], mock_cat[:,3], I_arr_full])

    bright_src_map = cmock.make_srcmap_temp_bank(ifield, inst, full_tracer_cat.transpose(), flux_idx=-1, load_precomp_tempbank=True, \
                                                    tempbank_dirpath='../data/subpixel_psfs_TM'+str(inst)+'/')
    
    
    return bright_src_map



def positions_from_counts(counts_map, cat_len=None, add_subpix_scatter=False):

    ''' Given a counts map, generate source catalog positions consistent with those counts. 
    Not doing any subpixel position assignment or anything like that.''' 

    thetax, thetay = [], []
    for i in np.arange(np.max(counts_map)):
        pos = np.where(counts_map > i)
        thetax.extend(pos[0].astype(float))
        thetay.extend(pos[1].astype(float))

    if cat_len is not None:
        idxs = np.random.choice(np.arange(len(thetax)), cat_len)
        thetax = np.array(thetax)[idxs]
        thetay = np.array(thetay)[idxs]

    if add_subpix_scatter:
        thetax += np.random.uniform(-0.5, 0.5, len(thetax))
        thetay += np.random.uniform(-0.5, 0.5, len(thetay))

    return np.array(thetax), np.array(thetay)



def counts_from_overdensity(overdensity_field, Ntot = 200000):
    
    ''' This function takes in a collection of number density fields and
    for each generates a poisson realization to get galaxy counts in each cell of the field.
    When a Poisson realization is taken and a specific number of galaxies is desired, a small 
    correction is needed to either remove or add sources. Sources are added/removed with uniform probabilities.

    Inputs:
        overdensity_fields (np.array): array of overdensity fields, which are usually obtained with gaussian_random_field_2d() and then exponentiated.
        
        Ntot (int, default=200000): number of source positions to sample from density field
    
    Output:
        count_maps (np.array): array of counts maps. This has the same shape as overdensity_fields

    '''
    
    # compute the mean number density per cell 
    N_mean = float(Ntot) / float(overdensity_field.shape[0]*overdensity_field.shape[1])
    
    # calculate the expected number of galaxies per cell
    expectation_ngal = (overdensity_field+1.)*N_mean 
    # generate Poisson realization of 2D field
    count_map = np.random.poisson(expectation_ngal)
    
    dcount = int(np.sum(count_map)-Ntot)
    
    if dcount > 0:
        nonzero_x, nonzero_y = np.nonzero(count_map)
        rand_idxs = np.random.choice(np.arange(len(nonzero_x)), dcount, replace=True)
        count_map[nonzero_x[rand_idxs], nonzero_y[rand_idxs]] -= 1
    elif dcount < 0:
        randx, randy = np.random.choice(np.arange(count_map.shape[-1]), size=(2, np.abs(dcount)))
        for j in range(np.abs(dcount)):
            count_map[randx[j], randy[j]] += 1

    return count_map

def lensing_kernel_weights(z_s, z_bins):
    chi_s = cosmo.comoving_distance(z_s).value  # Mpc
    weights = []

    for z in z_bins:
        chi = cosmo.comoving_distance(z).value
        a = 1.0 / (1 + z)
        if chi >= chi_s:
            weights.append(0.0)
        else:
            w = (chi / a) * (chi_s - chi) / chi_s
            weights.append(w)
    
    prefac = (3/2) * (cosmo.H0.value / 3e5)**2 * cosmo.Om0  # in units of 1/Mpc^2
    return prefac * np.array(weights)  


def combine_kappa_fields(grf_list, z_bins, z_s=2.0):
    
    weights = lensing_kernel_weights(z_s, z_bins)  # shape [n_slices]

    weights /= np.sum(weights)
    print('weights:', weights)
    kappa_map = np.zeros_like(grf_list[0])
    for delta_i, w_i in zip(grf_list, weights):
        kappa_map += w_i * delta_i
    
    return kappa_map

def sample_galaxy_positions(lognormal_field, n_galaxies, add_subpix_scatter=True):
    """
    Sample galaxy positions from a log-normal field.

    Parameters:
    - lognormal_field (2D array): Log-normal random field.
    - n_galaxies (int): Number of galaxies to sample.

    Returns:
    - positions (list of tuples): List of (x, y) positions.
    """
    # Normalize the field to create a PDF
    
    n_galaxies = int(n_galaxies)
    
    pdf = lognormal_field / np.sum(lognormal_field)

    # Flatten the PDF and create a cumulative distribution function (CDF)
    cdf = np.cumsum(pdf.ravel())

    # Generate random samples
    random_values = np.random.rand(n_galaxies)
    indices = np.searchsorted(cdf, random_values)

    # Convert flat indices to 2D positions
    thetay, thetax = np.unravel_index(indices, lognormal_field.shape)
    
    thetay = thetay.astype(float)
    thetax = thetax.astype(float)
    
    if add_subpix_scatter:
        
        thetax += np.random.uniform(-0.5, 0.5, len(thetax))
        thetay += np.random.uniform(-0.5, 0.5, len(thetay))

    return thetax, thetay


class galaxy_clus_gen():
    
    lf = Luminosity_Function()
    cosmo = FlatLambdaCDM(H0=70, Om0=0.28)

    
    Mabs_min=-30.0
    Mabs_max=-15.
    Npix_side = 1024
    
#     limber_basepath = config.ciber_basepath+'data/ciber_mocks/limber_cl_vs_redshift/'

    limber_basepath = '../data/limber_cl_vs_redshift/'
    def __init__(self, band='J', ell_min=90., ell_max=1e5, zmin=0.0, zmax=2.0, m_min=13., m_max=28.):
        
        for attr, valu in locals().items():
            setattr(self, attr, valu)
        
        self.Adeg = (180./self.ell_min)**2
    

    def draw_mags_given_zs(self, Mabs, gal_zs, ngal_per_z, pdfs, zs):
        '''we are going to order these by absolute magnitude, which makes things easier when abundance matching to 
        halo mass'''
        gal_app_mags = np.array([])
        gal_abs_mags = np.array([])
        for i in range(len(ngal_per_z)):
            absolute_mags = np.random.choice(Mabs, ngal_per_z[i], p=pdfs[i])
            apparent_mags = apparent_mag_from_absolute(absolute_mags, zs[i])
            gal_app_mags = np.append(gal_app_mags, apparent_mags)
            gal_abs_mags = np.append(gal_abs_mags, absolute_mags)
                        
        arr = np.array([gal_zs, gal_app_mags, gal_abs_mags]).transpose()
        cat = arr[np.argsort(arr[:,2])] # sort apparent and absolute mags by absolute mags

        return cat

    def draw_redshifts(self, Nsrc, zmin, zmax, Mabs):
        ''' Given some redshift range and observing band, this function uses the luminosity function from Helgason to sample Nsrc redshifts.'''
        zfine = np.linspace(zmin, zmax, 20)[:-1]
        dndz = []
        for zed in zfine:
            dndz.append(np.sum(self.lf.schechter_lf_dm(Mabs, zed, self.band)*(10**(-3) * self.lf.schechter_units)*(np.max(Mabs)-np.min(Mabs))/len(Mabs)).value)
        dndz = np.array(dndz)/np.sum(dndz)    
        zeds = np.random.choice(zfine, Nsrc, p=dndz)
        return zeds, zfine

    def get_schechter_m_given_zs(self, zs, Mabs):
        ''' this function computes collection of apparent magnitude PDFs evaluated at different redshifts as determined by the Schechter luminosity function 
        from Helgason'''
        pdfs = []
        for z in zs:
            pdf = self.lf.schechter_lf_dm(Mabs, z, self.band)
            pdfs.append(pdf/np.sum(pdf))
        return pdfs
    
    
    def gen_kappa_ln(self, cl, ell_sampled, fac=2.):
        
        kappa_grf = self.gen_grf(cl, ell_sampled, fac=fac)
        
        nx, ny = kappa_grf.shape[0], kappa_grf.shape[1]

        cropped_grf = kappa_grf[int(0.5*(nx-self.Npix_side)):int(0.5*(nx+self.Npix_side)), int(0.5*(ny-self.Npix_side)):int(0.5*(ny+self.Npix_side))]
        
        # convergence kappa doesn't have subtraction by 1, but lognormal overdensity subtracts by 1
        kappa_ln = np.exp(cropped_grf-0.5*np.var(cropped_grf))
        
        return kappa_ln
    
    def gen_grf(self, cl, ell_sampled, fac=2.):

        ''' 
        Generate independent 2D gaussian random fields given some input angular power spectrum C_ell.
        To account for the impact of larger spatial modes than the desired FOV, this function generates a field "fac" times as large as specified, 
        after which one can crop for the desired central region. This also means that the angular power spectrum input should have 
        sufficient multipole coverage "fac" times larger than the FOV. This function works by computing a spline interpolation of the input 
        power spectrum, which is then used in lognormal calculations specified by Carron et al. 2014 (arxiv:1406.6072v2). 

        Inputs:
            nfield (int): number of desired GRF samples
            cl (np.array): input angular power spectrum
            ell_sampled (np.array): multipole bins corresponding to input angular power spectrum
            fac (int, default=2): to account for larger than FOV modes, this makes this function increase the angular coverage of the 
                GRF in each dimension by a desired factor. fac=2 generally works sufficiently well.

        Outputs:
            gfield (array of np.arrays): the output gaussian random fields, which are not cropped from upscaled size determined by fac. 
                These fields should each have mean zero

        '''
        
        up_size = int(self.Npix_side*fac)
        steradperpixel = ((np.pi/self.ell_min)/up_size)**2
        surface_area = (np.pi/self.ell_min)**2

        cl_g, spline_cl_g = hankel_spline_lognormal_cl(ell_sampled, cl)
        ells = make_ell_grid(up_size, ell_min=self.ell_min)
        amplitude = 10**spline_cl_g(np.log10(ells))
        amplitude /= surface_area
        amplitude[0,0] = 0.
        
        grfshape = (up_size, up_size)

        noise = np.random.normal(size = grfshape) + 1j * np.random.normal(size = grfshape)
        gfield = np.fft.ifft2(noise * amplitude).real
        gfield /= steradperpixel

        # up to this point, the gaussian random fields have mean zero
        return gfield

    
    def specify_sims(self, ng_bins, Mabs_nbin=100, n_bin_Mapp=200, lam_obs=None, mode=None):
                
        zrange_grf = np.linspace(self.zmin, self.zmax, ng_bins+1) # ng_bins+1 since we take midpoints
        midzs = 0.5*(zrange_grf[:-1]+zrange_grf[1:])
        dzs = zrange_grf[1:]-zrange_grf[:-1] 
        
        Mabs = np.linspace(self.Mabs_min, self.Mabs_max, Mabs_nbin)
        Mapps = np.linspace(self.m_min, self.m_max, n_bin_Mapp)
        number_counts = np.array(self.lf.number_counts(zrange_grf, Mapps, self.band, dzs=dzs, lam_obs=lam_obs, mode=mode)[1]).astype(int)

        return zrange_grf, midzs, dzs, Mabs, Mapps, number_counts
        
    
    def generate_galaxy_catalog(self, ng_bins=8, size=1024, plot=False, lam_obs=None, mode=None, calc_weighted_kappa=False, \
                               z_source=2.0, randomize_counts=False):
        
        ''' This function puts together other functions in the galaxy_catalog() class as full pipeline to generate galaxy catalog realizations,
        given some angular power spectrum and Helgason model'''


        zrange_grf, midzs, dzs, Mabs, Mapps, number_counts = self.specify_sims(ng_bins)
        print('number counts has shape ', number_counts.shape, 'while Mapps has nbins=', len(Mapps))

        ntot_perz = np.sum(number_counts, axis=1)*self.Adeg
                    
        thetax, thetay, gal_z, mags, gal_app_mag, all_finezs = [[] for y in range(6)]

        all_kappa = np.zeros((len(midzs), self.Npix_side, self.Npix_side))
        
        # loop over redshift
        for i, z in enumerate(midzs):
            
            z0, z1 = zrange_grf[i], zrange_grf[i+1]
            # assume we already have the limber cl files
            clfile = np.load(self.limber_basepath+'limber_cls_zmin='+str(self.zmin)+'_zmax='+str(self.zmax)+'_zbin'+str(i)+'.npz')
            ells, cl = clfile['lb_limber'], clfile['integral_cl']
            
            kappa_ln = self.gen_kappa_ln(cl, ells)
            
            all_kappa[i] = kappa_ln
            
            gal_overdensity = kappa_ln - 1. 
    
            if plot:
                plot_map(kappa_ln, title='kappa zidx='+str(i), figsize=(6, 6))
                plot_map(gal_overdensity, figsize=(6, 6), title='galaxy overdensity')
    
            counts = counts_from_overdensity(gal_overdensity, Ntot=ntot_perz[i])
            tx, ty = positions_from_counts(counts, add_subpix_scatter=True)

            if randomize_counts:

                tx = np.random.uniform(np.min(tx), np.max(tx), len(tx))
                ty = np.random.uniform(np.min(ty), np.max(ty), len(ty))

                # tx = tx[np.random.permutation(len(tx))]
                # ty = ty[np.random.permutation(len(ty))]


            thetax.extend(tx)
            thetay.extend(ty)
            
            zeds, zfine = self.draw_redshifts(len(tx), zrange_grf[i], zrange_grf[i+1], Mabs)
            gal_z.extend(zeds)
            
            all_finezs.extend(zfine)
            
            # draw apparent magnitudes based on Helgason number counts N(m)
            
            mapp_pdf = self.Adeg*number_counts[i].astype(float)/float(ntot_perz[i])
            
            mapp_pdf /= np.sum(mapp_pdf)
            mag_draw = np.random.choice(Mapps, size=len(tx), p=mapp_pdf)
            mags.extend(mag_draw)

             
        print('min max gal z:', np.min(gal_z), np.max(gal_z))
        print('all galaxies:', np.sum(ntot_perz))
        if len(mags) > len(thetax):
            print('OHHHHHHHH')
            idx_choice = np.sort(np.random.choice(np.arange(len(mags_list[cat])), len(thetax_list[cat]), replace=False))
            mags_list[cat] = np.array(mags_list[cat])[idx_choice]

        mock_cat = np.array([thetax, thetay, gal_z, mags]).transpose()
        
        if calc_weighted_kappa:
            comb_kappa = combine_kappa_fields(all_kappa, midzs, z_s=z_source)
        else:
            comb_kappa = None
        
                
        return mock_cat, all_kappa, comb_kappa, zrange_grf


