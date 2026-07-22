# How to Implement Explicit B(ℓ)B(L-ℓ) Beam Correction

## The Problem

The FFT-based normalization computes convolutions in real space:
```python
term = inverseFourier(ℓ² × C(ℓ)) × iVar
result = Fourier(term) × ℓ²
```

When you multiply C(ℓ) by B²(ℓ) before this operation, you don't get the correct B(ℓ) × B(L-ℓ) for mode pairs. You need explicit mode-pair computation.

## Solution Overview

I've created `beam_corrected_normalization.py` with three approaches:

### 1. **Full Direct Sum** (`computeQuadEstPhiNormalizationFFT_with_beam`)
- Loops over all pixel pairs in Fourier space
- Explicitly computes B(ℓ) × B(|L-ℓ|) for each mode pair
- **Accurate but slow** (~N⁴ complexity for N×N maps)
- Use for small maps or verification

### 2. **Binned Integration** (`computeQuadEstPhiNormalizationFFT_with_beam_fast`)
- Bins ℓ into radial annuli
- Integrates over ℓ magnitude and angle
- **Much faster** (~N² × n_bins × n_angles)
- Good approximation if spectra are smooth
- **Recommended for production**

### 3. **Comparison Function** (`compare_normalizations`)
- Compares old vs. new approach
- Plots ratios to show the difference
- Use this to validate

## Integration Steps

### Step 1: Test the New Function

First, verify it works with your data:

```python
from beam_corrected_normalization import (
    computeQuadEstPhiNormalizationFFT_with_beam_fast,
    compare_normalizations
)

# Your existing setup
baseMap = FlatMap(nX=1024, nY=1024, sizeX=..., sizeY=...)
fC0 = lambda ell: c_i_shot  # Raw unlensed (no beam!)
fCtot = lambda ell: fC0(ell) * B_ell(ell)**2 + N_ell  # Observed
B_ell = clf.bl  # Your beam function

# Compare approaches
norm_old, norm_new = compare_normalizations(
    baseMap, fC0, fCtot, B_ell, lMin=1e4, lMax=1e5
)
```

This will show you:
- 2D maps of both normalizations
- 1D profiles
- Ratio between them

### Step 2: Modify `flat_map.py`

Add the new method to your `FlatMap` class:

```python
# In flat_map.py, after line ~2075

def computeQuadEstPhiNormalizationFFT_BeamExact(self, fC0, fCtot, fB_ell,
                                                lMin=1., lMax=1.e5, 
                                                nEll=100, test=False):
    """
    Compute normalization with explicit B(ℓ) × B(L-ℓ) beam correction.
    
    Uses binned integration over ℓ for efficiency while maintaining
    the correct beam mode-coupling structure.
    
    Parameters:
    -----------
    fC0 : function
        UNLENSED power spectrum (no beam!)
    fCtot : function
        Observed power spectrum (includes beam and noise)
    fB_ell : function
        Beam/PSF function B(ℓ)
    nEll : int
        Number of ℓ bins for integration (default 100)
    """
    print(f"Computing beam-exact normalization (nEll={nEll})")
    
    # Create ℓ bins
    ell_bins = np.logspace(np.log10(max(lMin, 1)), np.log10(lMax), nEll)
    ell_centers = 0.5 * (ell_bins[:-1] + ell_bins[1:])
    dell = np.diff(ell_bins)
    
    normFourier = np.zeros(self.lx.shape, dtype=float)
    
    for iLx in range(self.nX):
        for iLy in range(self.nY):
            
            Lx, Ly = self.lx[iLx, iLy], self.ly[iLx, iLy]
            L = np.sqrt(Lx**2 + Ly**2)
            
            if L < lMin or L > 2*lMax:
                continue
            
            # Integrate over ℓ (magnitude and angle)
            sum_L = 0.0
            
            for ell in ell_centers:
                if ell < lMin or ell > lMax:
                    continue
                
                # Angular integration (Trapezoidal rule)
                n_phi = 32
                for i_phi in range(n_phi):
                    phi = 2 * np.pi * i_phi / n_phi
                    
                    ellx = ell * np.cos(phi)
                    elly = ell * np.sin(phi)
                    
                    Lminusellx = Lx - ellx
                    Lminuselly = Ly - elly
                    Lminusell = np.sqrt(Lminusellx**2 + Lminuselly**2)
                    
                    if Lminusell < lMin or Lminusell > lMax:
                        continue
                    
                    # Get spectra at these ℓ values
                    C0_ell = fC0(ell)
                    Ctot_ell = fCtot(ell)
                    C0_Lminusell = fC0(Lminusell)
                    Ctot_Lminusell = fCtot(Lminusell)
                    
                    if Ctot_ell == 0 or Ctot_Lminusell == 0:
                        continue
                    
                    # Geometric dot products
                    ell_dot_L = ellx * Lx + elly * Ly
                    Lminusell_dot_L = Lminusellx * Lx + Lminuselly * Ly
                    
                    # Filter weight F
                    F = (ell_dot_L * C0_ell / Ctot_ell + 
                         Lminusell_dot_L * C0_Lminusell / Ctot_Lminusell)
                    
                    # Response f^κ
                    if L > 0:
                        f_kappa = 2 * (ell_dot_L * C0_ell + 
                                      Lminusell_dot_L * C0_Lminusell) / L**2
                    else:
                        f_kappa = 0
                    
                    # CRITICAL: Explicit beam factors
                    B_ell = fB_ell(ell)
                    B_Lminusell = fB_ell(Lminusell)
                    
                    # Jacobian and integration measure
                    dphi = 2 * np.pi / n_phi
                    idx = np.searchsorted(ell_bins, ell)
                    d_ell = dell[min(idx, len(dell)-1)]
                    
                    # Add contribution
                    sum_L += F * f_kappa * B_ell * B_Lminusell * ell * dphi * d_ell
            
            normFourier[iLx, iLy] = sum_L
        
        if iLx % (self.nX // 10) == 0:
            print(f"  {100*iLx/self.nX:.0f}%")
    
    # Invert
    normFourier[normFourier != 0] = 1.0 / normFourier[normFourier != 0]
    normFourier[np.isnan(normFourier) | np.isinf(normFourier)] = 0.0
    
    if test:
        plt.figure(figsize=(8, 4))
        plt.subplot(121)
        plt.imshow(np.log10(np.abs(normFourier) + 1e-20))
        plt.colorbar()
        plt.title('Beam-exact N_L')
        plt.subplot(122)
        mask = self.l.flatten() > 0
        plt.loglog(self.l.flatten()[mask], 
                   np.abs(normFourier.flatten()[mask]), 'b.', alpha=0.1)
        plt.xlabel('L')
        plt.ylabel('|N_L|')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    return normFourier
```

### Step 3: Update `computeQuadEstKappaNorm`

Modify to use the new method when beam is provided:

```python
# In flat_map.py, ~line 2165

def computeQuadEstKappaNorm(self, fC0, fCtot, lMin=1., lMax=1.e5, 
                           dataFourier=None, dataFourier2=None, path=None, 
                           test=False, cache=None, fourier_weights=None, 
                           cut_lxly=False, fB_ell=None, 
                           use_beam_exact=True):  # NEW PARAMETER
    """
    [existing docstring]
    
    use_beam_exact : bool
        If True and fB_ell is provided, uses explicit B(ℓ)B(L-ℓ) computation.
        If False, uses FFT method (faster but less accurate for beams).
    """
    
    # Non-normalized QE for phi
    resultFourier = self.quadEstPhiNonNorm(
        fC0, fCtot, lMin=lMin, lMax=lMax, dataFourier=dataFourier, 
        dataFourier2=dataFourier2, test=test, fourier_weights=fourier_weights, 
        cut_lxly=cut_lxly
    )
    
    # Convert phi -> kappa
    resultFourier = self.kappaFromPhi(resultFourier)
    
    # Compute normalization
    if fB_ell is not None and use_beam_exact:
        # Use explicit B(ℓ)B(L-ℓ) method
        print("Using beam-exact normalization")
        normalizationFourier = self.computeQuadEstPhiNormalizationFFT_BeamExact(
            fC0, fCtot, fB_ell, lMin=lMin, lMax=lMax, nEll=100, test=test
        )
    else:
        # Use standard FFT method
        print("Using FFT normalization")
        normalizationFourier = self.computeQuadEstPhiNormalizationFFT(
            fC0, fCtot, lMin=lMin, lMax=lMax, test=test, cache=cache, 
            fourier_weights=fourier_weights, cut_lxly=cut_lxly, fB_ell=fB_ell
        )
    
    # Normalize
    resultFourier *= normalizationFourier
    
    if path is not None:
        self.saveDataFourier(resultFourier, path)
    
    return resultFourier, normalizationFourier
```

### Step 4: Update Your Pipeline

In `kappa_auto_cross_fns.py`, modify `run_kappa_est`:

```python
def run_kappa_est(baseMap, ciber_unlensed_auto, ciber_obs_auto, params, 
                 dataFourier, dataFourier2=None, test=False, path=None, 
                 mode='qe_kappa_norm', cut_lxly=False, fB_ell=None,
                 use_beam_exact=True):  # NEW PARAMETER
    """
    use_beam_exact : bool
        If True, uses explicit B(ℓ)B(L-ℓ) computation (slower but correct).
        If False, uses FFT approximation (faster).
    """
    
    if mode == 'qe_kappa_norm':
        resultFourier, norm_Fourier = baseMap.computeQuadEstKappaNorm(
            ciber_unlensed_auto, ciber_obs_auto,
            lMin=params['lMin'], lMax=params['lMax'],
            dataFourier=dataFourier, dataFourier2=dataFourier2,
            test=test, path=path, cut_lxly=cut_lxly, 
            fB_ell=fB_ell, use_beam_exact=use_beam_exact
        )
    # ... rest of function
```

### Step 5: Update `calc_filters_and_corrections`

Ensure fC0 has NO beam:

```python
def calc_filters_and_corrections(clf, params, cfg):
    # ... existing code ...
    
    # CRITICAL: fC0 should be raw unlensed (no beam!)
    def cib_unlensed_auto(ell):
        return params['c_i_shot'] * np.ones_like(ell)  # ✅ Correct
    
    # fCtot includes beam (this is correct)
    if cfg['grab_cib_sim']:
        def obs_auto(ell):
            return cib_unlensed_auto(ell) * clf.bl(ell)**2 + np.mean(params['clnoise'])
    
    # ... rest of function ...
```

## Usage Example

```python
# In your analysis script
from FlatSkyQE.flat_map import FlatMap
from FlatSkyQE.kappa_auto_cross_fns import calc_filters_and_corrections

# Setup
baseMap = FlatMap(nX=1024, nY=1024, sizeX=2*np.pi/180, sizeY=2*np.pi/180)
clf = ciber_lens_forecast()
clf.load_bl(inst, ifield)

# Get filters (fC0 has no beam, obs_auto has beam)
fns, facs = calc_filters_and_corrections(clf, param_dict, config_dict)

# Run QE with beam-exact normalization
resultFourier, norm_Fourier = baseMap.computeQuadEstKappaNorm(
    fC0=fns['cib_unlensed_auto'],  # No beam!
    fCtot=fns['obs_auto'],          # Includes beam
    lMin=param_dict['lMin'],
    lMax=param_dict['lMax'],
    dataFourier=dataFourier,
    fB_ell=clf.bl,                  # Pass beam here
    use_beam_exact=True,            # Use explicit B(ℓ)B(L-ℓ)
    test=False
)

# Now you don't need external kcorr/vbeam corrections!
# The beam is properly accounted for in the normalization
```

## Performance Notes

- **Small maps** (512×512): Direct sum feasible (~minutes)
- **Large maps** (2048×2048): Use binned version with nEll=50-100
- **Optimization**: Can parallelize over Lx using multiprocessing

## Verification

Compare with your current results:
```python
# Old way
resultFourier_old = run_old_way_with_kcorr_vbeam(...)
clkg_old, bias_old = compute_spectra(resultFourier_old, ...)

# New way  
resultFourier_new = run_with_beam_exact(...)
clkg_new, bias_new = compute_spectra(resultFourier_new, ...)

# Compare
plt.figure()
plt.plot(lC, clkg_old, label='Old (kcorr/vbeam)')
plt.plot(lC, clkg_new, label='New (beam-exact)')
plt.legend()
plt.show()
```

## Next Steps

1. Test `beam_corrected_normalization.py` standalone
2. Integrate into `flat_map.py` as new method
3. Update pipeline to call with `use_beam_exact=True`
4. Compare results with theory
5. If agreement is good, can remove kcorr/vbeam external corrections
