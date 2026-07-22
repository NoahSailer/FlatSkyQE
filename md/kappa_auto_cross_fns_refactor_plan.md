# Refactoring Plan: run_flatskyqe_ciber

## Current Issues
1. Many scattered parameters passed individually
2. Inconsistent correction application (beam, mask, etc.)
3. `lens_data` class mixes data storage with processing logic
4. Difficult to track what corrections are applied where

## Proposed Structure (matching delta_fn_sources_test_v2)

### 1. Extend param_dict in initialize_clkk_prods
```python
param_dict = dict({
    'nX': nX, 'nY': nY, 'sizeX': sizeX, 'sizeY': sizeY,
    'lMin': lMin, 'lMax': lMax, 'nBins': nBins,
    'ciber_inst': ciber_inst,
    'pixel_size_arcsec': 3600.*(sizeX/nX),
    'l_nyquist': np.pi / ((3600.*(sizeX/nX)) / 206265.0),
    'c_i_shot': ld.mean_cl_sky,  # from loaded auto-spectrum
    'Apix': 1.15e-9
})
```

### 2. Create config_dict
```python
config_dict = dict({
    'apply_mask': apply_mask,
    'cut_lxly': cut_lxly,
    'mode': mode,
    'calc_ciber_cross': calc_ciber_cross,
    'single_band': single_band,
    'apply_FW': apply_FW,
    'compute_bis': compute_bis,
    'pixel_fn_correct': False,  # or as parameter
    'add_noise': True,  # real data always has noise
    'grab_cib_sim': False  # real data, not simulation
})
```

### 3. Use calc_filters_and_corrections
Instead of manually creating interpolated functions in `initialize_clkk_prods`, call:
```python
# After loading ld.ciber_unlensed_auto and setting up params
fns, facs = calc_filters_and_corrections(clf, param_dict, config_dict)

# Extract what we need
cl_fns = dict({
    'cib_unlensed_auto': fns['cib_unlensed_auto'],
    'obs_auto': fns['obs_auto'],
    'W_ell': fns['W_ell'],
    'B_ell': clf.bl
})

# Apply mode fraction correction
facs['vbeam'] *= facs['modefrac']

# Get unmask_frac
unmask_frac = np.mean(ld.mask)

corr_facs = dict({
    'kcorr': facs['kcorr'],
    'vbeam': facs['vbeam'],
    'unmask_frac': unmask_frac
})
```

### 4. Create map_dict
```python
map_dict = dict({
    'ciber_map': ld.ciber_map,
    'cross_map': ld.cross_map,
    'mask': ld.mask,
    'dataFourier': ld.dataFourier,
    'dataFourier2': ld.dataFourier2,
    'galdens': galdens,
    'galdensFourier': galdensFourier
})
```

### 5. Create unified processing function
```python
def compute_ciber_kappa_products(baseMap, map_dict, cl_fns, param_dict, 
                                 config_dict, corr_facs, paths):
    """
    Unified function to compute kappa estimates and cross-spectra for CIBER data.
    Analogous to compute_lensing_ps_quantities_v2 but for real data pipeline.
    
    Parameters:
    -----------
    baseMap : FlatMap
    map_dict : dict with 'ciber_map', 'mask', 'dataFourier', 'galdens', etc.
    cl_fns : dict with 'cib_unlensed_auto', 'obs_auto', 'W_ell', 'B_ell'
    param_dict : dict with 'lMin', 'lMax', 'Apix', etc.
    config_dict : dict with 'apply_mask', 'cut_lxly', 'mode', etc.
    corr_facs : dict with 'kcorr', 'vbeam', 'unmask_frac'
    paths : list of output paths for kappa estimates
    
    Returns:
    --------
    results : dict with all computed power spectra and corrections
    """
    
    # Run kappa estimation
    for k, path_k in enumerate(paths):
        dataFourier_use = map_dict['dataFourier'] if k == 0 else map_dict['dataFourier2']
        run_kappa_est(baseMap, cl_fns['cib_unlensed_auto'], cl_fns['obs_auto'],
                     param_dict, dataFourier=dataFourier_use, test=False,
                     path=path_k, cut_lxly=config_dict['cut_lxly'], 
                     mode=config_dict['mode'])
    
    # Compute power spectra
    all_cl, all_clerr = [], []
    for path_k in paths:
        kFourier = baseMap.loadDataFourier(path_k)
        lC, cl, sCl = baseMap.powerSpectrum(kFourier, theory=[], plot=False)
        all_cl.append(cl)
        all_clerr.append(sCl)
    
    # Compute kappa-galaxy cross-spectra
    lC, clxs, clxerrs = [], [], []
    for path_k in paths:
        kFourier = baseMap.loadDataFourier(path_k)
        lC, clx, clxerr = baseMap.crossPowerSpectrum(kFourier, 
                                                      map_dict['galdensFourier'],
                                                      plot=False)
        clxs.append(clx)
        clxerrs.append(clxerr)
    
    # Apply corrections to cross-spectra
    for i in range(len(clxs)):
        clxs[i], clxerrs[i] = proc_clkg(lC, clxs[i], clxerrs[i], 
                                        B_ell=None,  # QE is beam-independent
                                        kcorr=corr_facs['kcorr'],
                                        unmask_frac=corr_facs['unmask_frac'])
    
    # Compute bispectrum if requested
    if config_dict['compute_bis']:
        # Low-pass filter CIB map
        def bandpass(l):
            return 1.0 if (param_dict['lMin'] <= l <= param_dict['lMax']) else 0.0
        
        iVarCIBFourier = baseMap.filterFourierIsotropic(bandpass, 
                                                        dataFourier=map_dict['dataFourier'],
                                                        test=False)
        cib_lowpass = baseMap.inverseFourier(iVarCIBFourier)
        
        lC, cl_bis, sCl_bis = compute_skew_cl_I2G_simp(baseMap, cib_lowpass,
                                                        map_dict['galdensFourier'],
                                                        lMin=param_dict['lMin'],
                                                        lMax=param_dict['lMax'])
        
        # Apply corrections
        cl_bis, sCl_bis = proc_skewspec(lC, cl_bis, sCl_bis,
                                        B_ell=cl_fns['B_ell'],
                                        vbeam=corr_facs['vbeam'],
                                        unmask_frac=corr_facs['unmask_frac'])
        
        # Compute bias
        # Need to get clII_shot from somewhere or pass it in
        clkg_bias = cl_bis * param_dict['Apix'] / (2 * param_dict['c_i_shot'])
    else:
        cl_bis, sCl_bis, clkg_bias = None, None, None
    
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
```

### 6. Refactor initialize_clkk_prods
Key changes needed:
- Remove direct creation of `ld.ciber_unlensed_auto` and `ld.ciber_obs_auto`
- Store raw data in dictionaries instead
- Use `calc_filters_and_corrections` to create filter functions
- Return dictionaries instead of objects where possible

### 7. Simplify run_flatskyqe_ciber
Main loop becomes:
```python
for fieldidx, ifield in enumerate(ifield_list):
    # Initialize
    clf, ld, baseMap = initialize_clkk_prods(...)
    
    # Build dictionaries
    param_dict = build_param_dict(ld, ...)
    config_dict = build_config_dict(...)
    
    # Get filters and corrections
    fns, facs = calc_filters_and_corrections(clf, param_dict, config_dict)
    cl_fns, corr_facs = organize_corrections(fns, facs, clf, ld.mask)
    
    # Load galaxy data
    galdens, galdensFourier = calc_gal_fourier(baseMap, gal_densities, 
                                               ifield_list[fieldidx], ld.mask)
    
    # Build map dictionary
    map_dict = build_map_dict(ld, galdens, galdensFourier)
    
    # Compute all products
    results = compute_ciber_kappa_products(baseMap, map_dict, cl_fns,
                                          param_dict, config_dict, 
                                          corr_facs, paths)
    
    # Store results
    all_clx[...] = results['clxs']
    ...
```

## Benefits
1. **Consistency**: Same pattern as mock pipeline
2. **Clarity**: Easy to see what corrections are applied
3. **Maintainability**: Adding new corrections is straightforward
4. **Debugging**: Dictionary structure makes it easy to inspect intermediate values
5. **Testing**: Can test components independently with mock dictionaries

## Implementation Order
1. Start with helper functions (build_param_dict, build_config_dict, etc.)
2. Refactor initialize_clkk_prods to return dictionaries
3. Create compute_ciber_kappa_products
4. Update run_flatskyqe_ciber to use new structure
5. Test with one field first, then expand
