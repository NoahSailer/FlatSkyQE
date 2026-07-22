# kcorr Double-Correction Fix

## Problem

When `use_beam_in_norm=True`, the recovered C_ell^kg was **overestimated** relative to theory. This was because beam corrections were being applied **twice**:

1. **In the QE normalization**: B(ℓ)×B(L-ℓ) properly included in the denominator
2. **In post-processing**: `kcorr` (which includes beam) was still being applied via `proc_clkg()`

This resulted in the kappa estimates being under-corrected (normalization was too large), which made the cross-spectrum too high.

## Solution

When `use_beam_in_norm=True`, set `kcorr=1.0` to avoid double-counting the beam correction.

### In delta_fn_sources_test() (mock_lens_test.py, line ~598)

```python
if use_beam_in_norm:
    b_ell_use = B_ell_fn  # Pass beam to QE normalization (gives B(ℓ)×B(L-ℓ))
    kcorr_use = 1.0  # No additional beam correction needed
    print("Using beam in normalization (proper B(ℓ)×B(L-ℓ) correction)")
    print("  -> Setting kcorr=1.0 (no additional correction)")
else:
    b_ell_use = None  # Use external kcorr/vbeam corrections
    kcorr_use = facs['kcorr']  # Apply beam correction in post-processing
    print("Using old method with external kcorr/vbeam corrections")
```

Then pass `kcorr_use` to `corr_facs` dictionary instead of `facs['kcorr']`.

## Why This Happens

The `kcorr` factor (defined in kappa_auto_cross_fns.py) is computed as:

```python
kcorr = modefrac / np.mean(B_ell(ell_range)**2)
```

This includes a 1/B² beam correction factor. When the beam is already in the QE normalization:

- QE produces kappa estimates that are already "beam-corrected" (no beam suppression)
- Dividing by `kcorr` (which has 1/B²) applies an **extra** beam boost
- Result: C^kg is overestimated

## Verification

With this fix:
- `use_beam_in_norm=False` → uses old method: no beam in norm, apply `kcorr` in post-processing
- `use_beam_in_norm=True` → uses new method: beam in norm (B(ℓ)×B(L-ℓ)), `kcorr=1.0` in post-processing

Both methods should now give consistent results matching theory!

## Related to vbeam

Note: `vbeam` correction for the **bispectrum** is still applied regardless of `use_beam_in_norm`, because:
- The bispectrum involves ∫ B(ℓ) B(L-ℓ) which is L-dependent
- This is a separate geometric factor from the QE beam correction
- See `compute_bispectrum_beam_correction_L_dependent()` in kappa_auto_cross_fns.py
