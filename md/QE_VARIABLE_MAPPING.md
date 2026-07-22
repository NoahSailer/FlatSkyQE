# QE Variable Mapping: Math ↔ Code

## Summary of Your Colleague's Point

**The key issue**: When you include the beam in the QE normalization, you must be careful that `fC0` is the **raw unlensed spectrum without beam**, not a beam-deconvolved spectrum.

## Complete Mathematical → Code Mapping

### Equation (1): The Full Quadratic Estimator

$$\hat{\kappa}_L = \frac{\int_\ell F_{\ell, L-\ell} I^{obs}_{\ell} I^{obs}_{L-\ell}}{\int_\ell F_{\ell, L-\ell} f^{\kappa}_{\ell, L-\ell} PSF_{\ell} PSF_{L-\ell}}$$

### Component Mapping

| Math Symbol | Code Variable | Current Value | Should Be |
|-------------|---------------|---------------|-----------|
| **NUMERATOR** |
| $I^{obs}_\ell$ | `dataFourier` | Fourier transform of observed map | ✓ Correct |
| | | (beam-smoothed CIB + noise) | |
| **DENOMINATOR** |
| $PSF_\ell$ | `fB_ell` | `clf.bl(ell)` or Gaussian | ✓ Correct |
| $C^{II}_\ell$ (raw) | `fC0(ell)` | `c_i_shot` (constant) | ✓ Correct! |
| | | **NO BEAM** | |
| $C^{obs}_\ell$ | `fCtot(ell)` | See below ⬇️ | Need to check |

### The Observed Spectrum Definition

**Physical model:**
$$I^{obs}_\ell = PSF_\ell \times I^{lensed}_\ell + \text{noise}_\ell$$

**Power spectrum:**
$$C^{obs}_\ell = PSF^2_\ell \times C^{II}_\ell + N_\ell$$

**In your code** (`calc_filters_and_corrections` in `kappa_auto_cross_fns.py`):

```python
def cib_unlensed_auto(ell):
    return params['c_i_shot'] * np.ones_like(ell)  # This is C^{II}_ell (raw, no beam) ✓

def obs_auto(ell):  # This is C^{obs}_ell
    if add_noise and grab_cib_sim:
        return cib_unlensed_auto(ell) + np.mean(clnoise)/clf.bl(ell)**2
    else:
        return cib_unlensed_auto(ell)
```

## The Problem!

### What's Wrong with Current `obs_auto`

Looking at line 399 of `kappa_auto_cross_fns.py`:

```python
def obs_auto(ell):
    return cib_unlensed_auto(ell) + np.mean(params['clnoise'])/clf.bl(ell)**2
```

This says: $C^{obs}_\ell = C^{II}_\ell + N_\ell / PSF^2_\ell$

But it **should be**:
$$C^{obs}_\ell = PSF^2_\ell \times C^{II}_\ell + N_\ell$$

### Why This Matters

When you have beam in the normalization:
- **Numerator**: Uses $I^{obs}$ which includes beam naturally ✓
- **Denominator**: Uses `fC0` (unlensed, no beam) × `fB_ell`^2 ✓

But if your `fCtot` doesn't properly account for the beam-convolved signal, the **filter function** $F_{\ell, L-\ell}$ will be wrong!

The filter is built from:
$$F_{\ell, L-\ell} \propto \frac{C^{II}_\ell C^{II}_{L-\ell}}{C^{obs}_\ell C^{obs}_{L-\ell}}$$

## The Solution

### Option 1: Fix `obs_auto` Definition (RECOMMENDED)

In your **noiseless** case, you currently have:
```python
def obs_auto(ell):
    return cib_unlensed_auto(ell)*np.ones_like(ell)
```

This should be:
```python
def obs_auto(ell):
    return cib_unlensed_auto(ell) * clf.bl(ell)**2  # Include beam!
```

For the **noisy** case:
```python
def obs_auto(ell):  
    if grab_cib_sim:
        return cib_unlensed_auto(ell) * clf.bl(ell)**2 + np.mean(params['clnoise'])
    else:
        beam_gauss = gaussian_beam_window(ell, params['psf_pix_fwhm']*params['pixel_size_arcsec'])
        return cib_unlensed_auto(ell) * beam_gauss**2 + np.mean(params['clnoise'])
```

### Why Your kcorr=1.0 Gave Wrong Normalization

When you set `kcorr=1.0`, you were correct that the beam is now in the QE normalization.

But the **overall normalization was still wrong** because:
1. Your `obs_auto` didn't include the beam-convolved signal power
2. This made the filter $F$ wrong
3. The normalization integral computed with wrong $F$ gave wrong answer

## Verification Steps

After fixing `obs_auto`:

1. **Check the filter weights look reasonable:**
   ```python
   ell_test = np.logspace(3, 5, 100)
   W_ell_test = cib_unlensed_auto(ell_test) / obs_auto(ell_test)
   
   plt.plot(ell_test, W_ell_test)
   plt.xscale('log')
   plt.xlabel('ell')
   plt.ylabel('W(ell) = C_unl / C_obs')
   plt.title('Should decrease with ell (signal dominated at low ell)')
   plt.show()
   ```

2. **Verify obs_auto has correct shape:**
   ```python
   plt.figure()
   plt.loglog(ell_test, cib_unlensed_auto(ell_test), label='Unlensed (no beam)')
   plt.loglog(ell_test, obs_auto(ell_test), label='Observed (with beam)')
   plt.loglog(ell_test, clf.bl(ell_test)**2, label='Beam^2')
   plt.legend()
   plt.show()
   ```
   
   The observed should be **higher** than unlensed at low ell (beam suppression less important) and converge at high ell.

3. **Test normalization:**
   - Compare results with `kcorr=1.0` (beam in normalization)
   - Should now match analytic predictions

## Summary Table

| Quantity | Symbol | Code | Should Include Beam? | Current Status |
|----------|--------|------|---------------------|----------------|
| Data map | $I^{obs}$ | `obs_map` | Yes (physical) | ✓ Correct |
| Data Fourier | $I^{obs}_\ell$ | `dataFourier` | Yes (physical) | ✓ Correct |
| Unlensed CIB | $C^{II}_\ell$ | `fC0` = `cib_unlensed_auto` | **NO** | ✓ Correct |
| Observed PS | $C^{obs}_\ell$ | `fCtot` = `obs_auto` | Yes ($PSF^2 C^{II} + N$) | ❌ **WRONG** |
| Beam | $PSF_\ell$ | `fB_ell` | N/A | ✓ Correct |
| QE norm. | Denominator | `computeQuadEstPhiNormalizationFFT` | Yes (via `fB_ell`) | ✓ Correct |

## The Fix Needed

**File**: `kappa_auto_cross_fns.py`, function `calc_filters_and_corrections`

**Change lines 399-408** from:
```python
if cfg['add_noise']:
    if cfg['grab_cib_sim']: # use empirical beam
        def obs_auto(ell):
            return cib_unlensed_auto(ell) + np.mean(params['clnoise'])/clf.bl(ell)**2
    else:
        def obs_auto(ell): # Gaussian beam
            return cib_unlensed_auto(ell) + np.mean(params['clnoise'])/gaussian_beam_window(ell, params['psf_pix_fwhm']*params['pixel_size_arcsec'])
else:
    def obs_auto(ell):
        return cib_unlensed_auto(ell)*np.ones_like(ell)
```

**To**:
```python
if cfg['add_noise']:
    if cfg['grab_cib_sim']: # use empirical beam
        def obs_auto(ell):
            return cib_unlensed_auto(ell) * clf.bl(ell)**2 + np.mean(params['clnoise'])
    else:
        def obs_auto(ell): # Gaussian beam
            beam_sq = gaussian_beam_window(ell, params['psf_pix_fwhm']*params['pixel_size_arcsec'])**2
            return cib_unlensed_auto(ell) * beam_sq + np.mean(params['clnoise'])
else:
    if cfg['grab_cib_sim']:
        def obs_auto(ell):
            return cib_unlensed_auto(ell) * clf.bl(ell)**2
    elif cfg.get('psf_pix_fwhm') is not None:
        def obs_auto(ell):
            beam_sq = gaussian_beam_window(ell, params['psf_pix_fwhm']*params['pixel_size_arcsec'])**2
            return cib_unlensed_auto(ell) * beam_sq
    else:
        def obs_auto(ell):
            return cib_unlensed_auto(ell)
```

This ensures $C^{obs}_\ell = PSF^2_\ell \times C^{II}_\ell + N_\ell$ as required by the physics!
