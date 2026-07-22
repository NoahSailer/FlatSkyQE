# Beam Correction Fix - Proper B(ℓ)×B(L-ℓ) Implementation

## Problem
The original FFT normalization was applying B²(ℓ) to the CFourier map before the convolution. This doesn't capture the correct mode coupling B(ℓ)×B(L-ℓ) in the QE normalization denominator.

## Solution
Apply B(ℓ) to **both** the CFourier and iVarFourier maps before the inverse FFT. When these are convolved in real space and transformed back, the result at wavevector L automatically includes the correct B(ℓ)×B(L-ℓ) product from the coupled modes.

## Changes Made

### 1. flat_map.py - iVarFourier (line ~1855)
**Before:**
```python
iVarFourier = np.array(list(map(f, self.l.flatten())))
iVarFourier = iVarFourier.reshape(self.l.shape)
iVar = self.inverseFourier(dataFourier=iVarFourier)
```

**After:**
```python
iVarFourier = np.array(list(map(f, self.l.flatten())))
iVarFourier = iVarFourier.reshape(self.l.shape)

# Apply beam correction to iVarFourier if provided
# This ensures B(ℓ) × B(L-ℓ) in the convolution
if fB_ell is not None:
    def fBeam(l):
        return fB_ell(l)
    beamFourier = np.array(list(map(fBeam, self.l.flatten())))
    beamFourier = beamFourier.reshape(self.l.shape)
    iVarFourier *= beamFourier

iVar = self.inverseFourier(dataFourier=iVarFourier)
```

### 2. flat_map.py - CFourier (lines ~1878, ~1905)
**Before:**
```python
if fB_ell is not None:
    result *= fB_ell(l)**2
```

**After:**
```python
if fB_ell is not None:
    result *= fB_ell(l)
```

## Why This Works

The FFT-based algorithm computes:
```
term1x = IFFT(CFourier) * iVar
term1xFourier = FFT(term1x)
```

This is a **convolution** in Fourier space. At output wavevector L, it couples modes ℓ and L-ℓ.

With the fix:
- CFourier(ℓ) = [C0²(ℓ)/Ctot(ℓ)] × B(ℓ)
- iVarFourier(ℓ) = [1/Ctot(ℓ)] × B(ℓ)

After convolution, the result at L contains contributions from all pairs (ℓ, L-ℓ), each weighted by B(ℓ) × B(L-ℓ), which is exactly what we need!

## Usage

No changes needed to function calls. The beam correction is automatically applied properly when you provide `fB_ell`:

```python
# Old way - no beam in normalization (uses external kcorr/vbeam)
res_old = delta_fn_sources_test(..., use_beam_in_norm=False)

# New way - beam properly handled in normalization
res_new = delta_fn_sources_test(..., use_beam_in_norm=True)
```

When `use_beam_in_norm=True`, the beam function is passed to the QE normalization and properly accounts for B(ℓ)×B(L-ℓ).

## Verification

The key test is that with this fix, the normalization should give:
```
N_L = 1 / ∫_ℓ F_{ℓ,L-ℓ} f^κ_{ℓ,L-ℓ} B(ℓ) B(|L-ℓ|)
```

where each mode pair (ℓ, L-ℓ) is weighted by the product of their beam values, not by B²(ℓ).
