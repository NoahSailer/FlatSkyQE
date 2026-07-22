# Using Beam-Exact Normalization in Your Pipeline

## Quick Start

The beam-exact normalization is now integrated into `flat_map.py`. Here's how to use it:

### 1. Basic Usage

```python
from FlatSkyQE.flat_map import FlatMap

# Your existing setup
baseMap = FlatMap(nX=1024, nY=1024, sizeX=..., sizeY=...)

# Define spectra (CRITICAL: fC0 must be unlensed, no beam!)
fC0 = lambda ell: c_i_shot  # Raw unlensed CIB
fCtot = lambda ell: B_ell(ell)**2 * fC0(ell) + N_ell  # Observed
B_ell = clf.bl  # Your beam function

# Run QE with beam-exact normalization
kappa_fourier, norm = baseMap.computeQuadEstKappaNorm(
    fC0, fCtot,
    lMin=1e4, lMax=1e5,
    dataFourier=your_data_fourier,
    fB_ell=B_ell,           # Beam function
    use_beam_exact=True,    # Use explicit B(ℓ)B(L-ℓ)
    nEll_beam=100          # Number of ℓ bins
)
```

### 2. Integration with Your Existing Pipeline

In `calc_filters_and_corrections`, you're already doing the right thing:

```python
def calc_filters_and_corrections(clf, params, cfg):
    # ✅ CORRECT: No beam in fC0
    def cib_unlensed_auto(ell):
        return params['c_i_shot'] * np.ones_like(ell)
    
    # ✅ CORRECT: Beam in observed spectrum
    def obs_auto(ell):
        return cib_unlensed_auto(ell) * clf.bl(ell)**2 + np.mean(params['clnoise'])
    
    # Now you DON'T need to compute kcorr/vbeam if using beam-exact!
    # The beam is handled correctly in the normalization
```

When calling the QE:

```python
# In your analysis code
resultFourier, norm_Fourier = run_kappa_est(
    baseMap, 
    fns['cib_unlensed_auto'],  # No beam
    fns['obs_auto'],            # Has beam
    param_dict,
    dataFourier=ld.dataFourier,
    fB_ell=clf.bl,              # Pass beam here
    use_beam_exact=True,        # Enable beam-exact method
    nEll_beam=100               # Integration bins
)
```

### 3. Testing Your Implementation

Run the test script:
```bash
cd /Users/richardfeder/Documents/ciber/FlatSkyQE
python test_beam_exact_normalization.py
```

This will:
- Compute beam-exact normalization
- Compare with old FFT method
- Show you the difference in a plot
- Run a full QE test

### 4. Performance Tuning

**For different map sizes:**
- 512×512: `nEll_beam=100`, takes ~5-10 minutes
- 1024×1024: `nEll_beam=80`, takes ~20-30 minutes  
- 2048×2048: `nEll_beam=50`, takes ~1-2 hours

**Speed vs. accuracy:**
- `nEll_beam=30`: Fast, good for testing
- `nEll_beam=100`: Accurate, recommended for production
- `nEll_beam=200`: Very accurate, slow

### 5. Comparison with Old Method

```python
# Old way (INCORRECT for beams)
resultFourier_old, _ = baseMap.computeQuadEstKappaNorm(
    fC0, fCtot,
    fB_ell=B_ell,
    use_beam_exact=False  # Uses FFT approximation
)

# New way (CORRECT)
resultFourier_new, _ = baseMap.computeQuadEstKappaNorm(
    fC0, fCtot,
    fB_ell=B_ell,
    use_beam_exact=True   # Uses explicit B(ℓ)B(L-ℓ)
)

# Compare power spectra
clkg_old = compute_cross_spectrum(resultFourier_old, ...)
clkg_new = compute_cross_spectrum(resultFourier_new, ...)
```

### 6. What Changed

**Before (with external corrections):**
```python
# Compute QE
resultFourier, norm = baseMap.computeQuadEstKappaNorm(fC0, fCtot, fB_ell=None)

# Apply external corrections
clkg /= facs['kcorr']
cl_bis *= facs['vbeam']
```

**After (with beam-exact):**
```python
# Compute QE with beam-exact normalization
resultFourier, norm = baseMap.computeQuadEstKappaNorm(
    fC0, fCtot, 
    fB_ell=clf.bl, 
    use_beam_exact=True
)

# No external corrections needed! Beam is handled correctly
clkg = compute_cross_spectrum(resultFourier, ...)
```

### 7. Updating `compute_lensing_ps_quantities_v2`

If you want to use beam-exact in your high-level functions, update the call:

```python
def compute_lensing_ps_quantities_v2(baseMap, map_dict, cl_fns, param_dict, 
                                     config_dict, corr_facs):
    
    # Extract beam function
    B_ell = cl_fns.get('B_ell', None)
    
    # Run kappa estimation with beam-exact if beam provided
    resultFourier, norm_Fourier = run_kappa_est(
        baseMap, 
        cl_fns['cib_unlensed_auto'],
        cl_fns['obs_auto'],
        param_dict,
        dataFourier=baseMap.fourier(map_dict['obs_map']),
        fB_ell=B_ell,
        use_beam_exact=True,  # NEW: use exact method
        nEll_beam=100,
        mode=config_dict['mode']
    )
    
    # ... rest of function
```

### 8. Validation Checklist

Before trusting your results:

- [ ] Verify `fC0` has NO beam (should be raw unlensed spectrum)
- [ ] Verify `fCtot` HAS beam (should be B²×C_unlensed + N)
- [ ] Run test script and check that ratio plots look reasonable
- [ ] Compare κ-g cross-spectrum with theory predictions
- [ ] Check that bispectrum bias matches expected values
- [ ] Compare beam-exact vs FFT results to see the difference

### 9. Expected Improvements

With beam-exact normalization, you should see:

- **Better agreement with theory** for κ-g cross-spectrum
- **Correct bias estimates** for bispectrum (no more manual vbeam corrections)
- **Consistent results** across different multipole ranges
- **No need for empirical kcorr/vbeam factors**

### 10. Troubleshooting

**"Normalization is taking too long"**
- Reduce `nEll_beam` (try 50 instead of 100)
- Use smaller map for testing (512×512)
- Check that lMin, lMax are reasonable

**"Results don't match theory"**
- Double-check that fC0 has NO beam
- Verify fCtot includes B²
- Make sure you're passing the right beam function
- Compare with FFT method to see the difference

**"Normalization has zeros/NaNs"**
- Check that fCtot(ell) != 0 for ell in [lMin, lMax]
- Verify beam function is well-defined
- Check for numerical issues in power spectra

### 11. Contact

If you encounter issues or need help:
- Check the plots from test script
- Compare with BEAM_EXACT_IMPLEMENTATION_GUIDE.md
- Review your collaborator's comments about B(ℓ)×B(L-ℓ)
