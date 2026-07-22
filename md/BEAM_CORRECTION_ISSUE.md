# Beam Correction Issue in QE Normalization

## The Problem

Your collaborator has identified that **you cannot correctly capture B_ℓ × B_{L-ℓ} by modifying fC0** in the quadratic estimator weights.

### Current (Incorrect) Approach

In `calc_filters_and_corrections()` ([kappa_auto_cross_fns.py](kappa_auto_cross_fns.py#L483-L568)):
- `cib_unlensed_auto(ell)` = raw unlensed spectrum (correct - no beam)
- `obs_auto(ell)` = includes PSF²: `cib_unlensed_auto(ell) * clf.bl(ell)**2` (correct for observed data)
- `W_ell(ell)` = `cib_unlensed_auto(ell)/obs_auto(ell)` (weights - correct)

In `computeQuadEstPhiNormalizationFFT()` ([flat_map.py](flat_map.py#L1780)):
- Line 1891: Applies `fB_ell(l)**2` to the C term
- Line 1965: Applies `fB_ell(l)` to the WF term

**Problem:** This applies beam corrections BEFORE the convolution operation. When you compute terms like:
```python
term1x = inverseFourier(lx² * C(ℓ)) * iVar
term1xFourier = Fourier(term1x) * lx²
```

You're not getting the correct B(ℓ) × B(L-ℓ) for each ℓ, L-ℓ pair. Instead, you get factors like B²(ℓ) and B²(L-ℓ) that don't combine properly.

## The Correct Approach

### Equation from Your Collaborator

The full estimator should be:
```
κ̂_L = (∫_ℓ F_{ℓ,L-ℓ} I^obs_ℓ I^obs_{L-ℓ}) / (∫_ℓ F_{ℓ,L-ℓ} f^κ_{ℓ,L-ℓ} PSF_ℓ PSF_{L-ℓ})
```

Where:
- `f^κ_{ℓ,L-ℓ} = 2 L · (ℓ C^{II}_ℓ + (L-ℓ) C^{II}_{L-ℓ}) / L²`
- `C^{II}_ℓ` = **raw unlensed spectrum** (no beam)
- `I^obs_ℓ = PSF_ℓ * I^lensed_ℓ + noise`

### Key Points

1. **fC0 should NOT include beam convolution** - it should be the raw unlensed C^{II}
2. **The beam appears only in the normalization denominator** as PSF_ℓ × PSF_{L-ℓ}
3. **The observed data already includes the beam** (correctly)

## Required Changes

### 1. In `calc_filters_and_corrections()`

**Current (Correct):**
```python
def cib_unlensed_auto(ell):
    return params['c_i_shot']*np.ones_like(ell)  # ✅ No beam - correct!
```

**Keep this as is** - fC0 should be the raw unlensed spectrum.

### 2. In `computeQuadEstPhiNormalizationFFT()`

The beam must be applied **differently** in the normalization. The current approach of multiplying by `fB_ell(l)**2` before convolution is wrong.

**Option A: Modify the algorithm structure** (Complex)
You'd need to explicitly compute B(ℓ) × B(L-ℓ) for each mode pair, which requires restructuring the FFT-based convolution approach. This is mathematically correct but computationally expensive.

**Option B: Apply post-hoc correction** (Simpler, but approximate)
Compute an effective beam correction factor that approximates the ratio:
```python
correction_factor = ∫_ℓ F * f^κ * B_ℓ * B_{L-ℓ} / ∫_ℓ F * f^κ
```

This is what your current `kcorr` and `vbeam` aim to do, but applied **outside** the normalization rather than inside it.

### 3. In `run_kappa_est()`

Currently passing `fB_ell` to the normalization:
```python
resultFourier, norm_Fourier = baseMap.computeQuadEstKappaNorm(
    ciber_unlensed_auto, ciber_obs_auto,
    lMin=params['lMin'], lMax=params['lMax'],
    dataFourier=dataFourier, dataFourier2=dataFourier2,
    test=test, path=path, cut_lxly=cut_lxly, fB_ell=fB_ell)  # ← PROBLEMATIC
```

**Should either:**
1. Remove `fB_ell` from normalization and rely on post-hoc corrections (current kcorr/vbeam)
2. Implement proper B(ℓ) × B(L-ℓ) computation in the normalization (requires algorithm rewrite)

## Your Collaborator's Recommendation

> "you'll have to modify computeQuadEstPhiNormalizationFFT in flat_map.py to optionally include the beam"

But NOT by multiplying fC0 - instead, the PSF factors need to appear explicitly in the normalization integral after the mode coupling.

## Current Workaround

Your current approach with **kcorr** and **vbeam** computed externally may be the practical solution:

1. Compute normalization **without** beam in `fB_ell` parameter
2. Apply post-hoc corrections using:
   - `kcorr` for κ-g cross-spectrum
   - `vbeam(L)` for bispectrum (L-dependent)
   - These approximate the proper beam corrections

## Questions to Resolve

1. **What exactly is "beam" in your case?**
   - Is it `clf.bl` (CIBER empirical beam)?
   - Is it the Gaussian PSF approximation?
   - Clarify this with your collaborator

2. **What is B^{QE}_L in your figures?**
   - Is this the beam correction applied to the estimator?
   - Is this vbeam(L)?

3. **Accuracy requirements**
   - Is the post-hoc correction (kcorr/vbeam) sufficient?
   - Or do you need to implement the full B(ℓ) × B(L-ℓ) in normalization?

## Recommended Next Steps

1. **Remove fB_ell from normalization calls** - set it to `None`
2. **Verify fC0 = raw unlensed spectrum** (already correct)
3. **Rely on external kcorr/vbeam corrections** computed in `calc_filters_and_corrections()`
4. **Test**: Compare results with/without beam corrections
5. **Discuss with collaborator**: Is external correction acceptable or do you need algorithm rewrite?

## Code Locations

- `calc_filters_and_corrections()`: [kappa_auto_cross_fns.py:483-568](kappa_auto_cross_fns.py#L483-L568)
- `computeQuadEstPhiNormalizationFFT()`: [flat_map.py:1780-2050](flat_map.py#L1780-L2050)
- `run_kappa_est()`: [kappa_auto_cross_fns.py:571](kappa_auto_cross_fns.py#L571)
- Beam correction factors: [kappa_auto_cross_fns.py:532-549](kappa_auto_cross_fns.py#L532-L549)
