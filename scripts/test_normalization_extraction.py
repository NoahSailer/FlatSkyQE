#!/usr/bin/env python
"""
Test script to verify N_L extraction and plotting works correctly.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/richardfeder/Documents/ciber/FlatSkyQE')

from mock_lens_test import plot_normalization

# Create synthetic N_L data to test plotting
lC_norm = np.logspace(np.log10(1000), np.log10(100000), 50)
# Simulate N_L that decreases at high L (mimicking potential issue)
N_L = 1e-6 * (lC_norm / 10000)**(-0.5)
N_L_err = 0.1 * N_L

# Add a roll-off at high L to simulate the issue
high_L_mask = lC_norm > 50000
N_L[high_L_mask] *= np.exp(-((lC_norm[high_L_mask] - 50000) / 20000)**2)

lMin = 10000
lMax = 60000

print("Testing plot_normalization function...")
print(f"lC_norm range: [{lC_norm.min():.1f}, {lC_norm.max():.1f}]")
print(f"N_L range: [{N_L.min():.3e}, {N_L.max():.3e}]")

# Test the plotting function
plot_normalization(lC_norm, N_L, N_L_err, lMin=lMin, lMax=lMax)

print("\nPlot generated successfully!")
print("This demonstrates what the N_L diagnostic will look like.")
print("If N_L rolls off at high L (near lMax), it indicates the normalization")
print("integral is becoming unreliable due to mode coupling constraints.")
