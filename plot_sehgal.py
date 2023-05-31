from __future__ import division
from builtins import map
from past.utils import old_div
#from importlib import reload
import sys

import universe
reload(universe)
from universe import *

import halo_fit
reload(halo_fit)
from halo_fit import *

import weight
reload(weight)
from weight import *

import pn_2d
reload(pn_2d)
from pn_2d import *

import cmb
reload(cmb)
from cmb import *

import flat_map
reload(flat_map)
from flat_map import *

##################################################################################
##################################################################################
print("Map properties")

nX = 1200
nY = 1200
size = 20.  # degrees, determined by the Sehgal cutouts
baseMap = FlatMap(nX=nX, nY=nY, sizeX=size*np.pi/180., sizeY=size*np.pi/180.)


# multipoles to include in the lensing reconstruction
lMin = 30.; lMax = 4.e3

# ell bins for power spectra
nBins = 21  # number of bins
lRange = (1., 2.*lMax)  # range for power spectra

##################################################################################
print("CMB experiment properties")

# Adjust the lMin and lMax to the assumptions of the analysis
# CMB S3 specs
cmb = CMB(beam=1.4, noise=7., lMin=lMin, lMaxT=lMax, lMaxP=lMax, atm=False, fg=True)
# Total power spectrum, for the lens reconstruction
forCtotal = lambda l: cmb.ftotalTT(l) 
# reinterpolate: gain factor 10 in speed
L = np.logspace(np.log10(lMin/2.), np.log10(2.*lMax), 1001, 10.)
F = np.array(list(map(forCtotal, L)))
cmb.fCtotal = interp1d(L, F, kind='linear', bounds_error=False, fill_value=0.)


##################################################################################
print("CMB lensing power spectrum")

u = UnivPlanck15()
halofit = Halofit(u, save=False)
w_cmblens = WeightLensSingle(u, z_source=1100., name="cmblens")
p2d_cmblens = P2dAuto(u, halofit, w_cmblens, save=False)


##################################################################################


path = "sehgal_radiops_148_large_cutout_3.txt"

#path = "sehgal_tsz_148_large_cutout_3.txt"

radiops = np.genfromtxt(path)

path2 = "sehgal_lsstgold_large_cutout_3.txt"
g = np.genfromtxt(path2)
gFourier = baseMap.fourier(g)

# rescale to match Dunkley+13 power spectra

# Fourier transform
radiopsFourier = baseMap.fourier(radiops)
# use the flux cut from Dunkley+13, to match power spectra
fluxCut = 0.015  # in Jy (15 mJy)
maskPatchRadius = 5. * np.pi/(180.*60.)   # in rad   
maskRadiops = baseMap.pointSourceMaskMatchedFilterIsotropic(cmb.fCtotal, fluxCut, fprof=None, dataFourier=radiopsFourier, maskPatchRadius=maskPatchRadius, test=False)
# remove the mean
#radiops -= np.mean(radiops.flatten())
# convert from Jy/sr to muK
conversion = 1.e6 * 1.e-26 / cmb.dBdT(148.e9, cmb.Tcmb) #cmb.nu1
radiops *= conversion
# mask the map
#radiops *= maskRadiops
# fourier transform
radiops *= 1.1
radiopsFourier = baseMap.fourier(radiops)

path3 = "sehgal_kcmb_large_cutout_3.txt"
kCmb = np.genfromtxt(path3)
kCmbFourier = baseMap.fourier(kCmb)

#baseMap.plot(radiops,save=False)
lCen, Cl, sCl = baseMap.powerSpectrum(radiopsFourier, theory=[cmb.fradioPoisson], plot=True, save=False, dataLabel=r'$\text{fg}\times \text{fg}$', theoryLabel="radio PS")

#baseMap.plot(kCmb,save=False)
#lCen, Cl, sCl = baseMap.powerSpectrum(kCmbFourier, theory=[p2d_cmblens.fPinterp], plot=True, save=False, dataLabel=r'$\kappa\times\kappa$', theoryLabel=r'$C^\kappa$')

#baseMap.plot(g,save=False)
#lCen, Cl, sCl = baseMap.powerSpectrum(gFourier, theory=None, plot=True, save=False, dataLabel=r'$\text{g}\times \text{g}$')


where = np.where(radiops.flatten() != 0.)

x1 = np.mean(radiops.flatten()[where]**2.)**2.
x2 = np.mean(radiops.flatten()[where]**4.)

print(x1/x2)
print(len(where[0]))

import sys
sys.exit()


#################################################################################
print("Calculate noises and response")

print("- standard QE")
fNqCmb_fft = baseMap.forecastN0Kappa(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False)
print("- shear E-mode estimator")
fNsCmb_fft = baseMap.forecastN0KappaShear(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, corr=True, test=False)
print("- magnification estimator")
fNdCmb_fft = baseMap.forecastN0KappaDilation(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, corr=True, test=False)
print("- bias hardened estimator")
fNqBHCmb_fft = baseMap.forecastN0KappaBH(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False)


#################################################################################
pathQ = "./output/pQ.txt"
baseMap.computeQuadEstKappaNorm(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=radiopsFourier, test=False, path=pathQ)
pQFourier = baseMap.loadDataFourier(pathQ)

pathSCmb = "./output/sCmb.txt"
baseMap.computeQuadEstKappaShearNormCorr(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=radiopsFourier, test=False, path=pathSCmb)
sCmbFourier = baseMap.loadDataFourier(pathSCmb)

pathDCmb = "./output/dCmb.txt"
baseMap.computeQuadEstKappaDilationNormCorr(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=radiopsFourier, test=False, path=pathDCmb)
dCmbFourier = baseMap.loadDataFourier(pathDCmb)

pathQBH = "./output/pQBH.txt"
baseMap.computeQuadEstKappaPointSourceHardenedNorm(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=radiopsFourier, test=False, path=pathQBH)
pQBHFourier = baseMap.loadDataFourier(pathQBH)


##############################################################################
print("Auto-power: standard QE")
lCenQ, ClQ, sClQ = baseMap.powerSpectrum(pQFourier,theory=[fNqCmb_fft], plot=False, save=False)

print("Auto-power: shear E-mode")
lCenS, ClS, sClS = baseMap.powerSpectrum(sCmbFourier,theory=[fNsCmb_fft], plot=False, save=False)

print("Auto-power: magnification")
lCenD, ClD, sClD = baseMap.powerSpectrum(dCmbFourier,theory=[fNdCmb_fft], plot=False, save=False)

print("Auto-power: bias hardened")
lCenQBH, ClQBH, sClQBH = baseMap.powerSpectrum(pQBHFourier,theory=[fNqBHCmb_fft], plot=False, save=False)


################################################################################
print("Cross-power: standard QE x k_true")
lCenQx, ClQx, sClQx = baseMap.crossPowerSpectrum(pQFourier, kCmbFourier, theory=[p2d_cmblens.fPinterp], plot=False, save=False)

print("Cross-power: shear E-mode x k_true")
lCenSx, ClSx, sClSx = baseMap.crossPowerSpectrum(sCmbFourier, kCmbFourier, theory=[p2d_cmblens.fPinterp], plot=False, save=False)

print("Cross-power: magnification x k_true")
lCenDx, ClDx, sClDx = baseMap.crossPowerSpectrum(dCmbFourier, kCmbFourier, theory=[p2d_cmblens.fPinterp], plot=False, save=False)

print("Cross-power: bias hardened x k_true")
lCenQBHx, ClQBHx, sClQBHx = baseMap.crossPowerSpectrum(pQBHFourier, kCmbFourier, theory=[p2d_cmblens.fPinterp], plot=False, save=False)


################################################################################
print("Cross-power: standard QE x g")
lCenQx1, ClQx1, sClQx1 = baseMap.crossPowerSpectrum(pQFourier, gFourier, theory=[p2d_cmblens.fPinterp], plot=False, save=False)

print("Cross-power: shear E-mode x g")
lCenSx1, ClSx1, sClSx1 = baseMap.crossPowerSpectrum(sCmbFourier, gFourier, theory=[p2d_cmblens.fPinterp], plot=False, save=False)

print("Cross-power: magnification x g")
lCenDx1, ClDx1, sClDx1 = baseMap.crossPowerSpectrum(dCmbFourier, gFourier, theory=[p2d_cmblens.fPinterp], plot=False, save=False)

print("Cross-power: bias hardened x g")
lCenQBHx1, ClQBHx1, sClQBHx1 = baseMap.crossPowerSpectrum(pQBHFourier, gFourier, theory=[p2d_cmblens.fPinterp], plot=False, save=False)


#################################################################################
print("Cross-power: k_true x g")
lCen, Cl, sCl = baseMap.crossPowerSpectrum(kCmbFourier, gFourier, theory=[p2d_cmblens.fPinterp], plot=False, save=False)


#################################################################################
print("Plotting the bias in cross correlation of CMB with galaxies")

fig=plt.figure(0,figsize=(6,6))
ax=fig.add_subplot(111)
#
plt.plot(lCen, np.abs(ClSx1/Cl), c='b',marker='.',label='Shear E')
plt.plot(lCen, np.abs(ClDx1/Cl), c='g',marker='.',label='Mag')
plt.plot(lCen, np.abs(ClQBHx1/Cl), c='purple',marker='.',label='BH')
plt.plot(lCen, np.abs(ClQx1/Cl), c='r',lw=3,marker='o',label='QE')
#
ax.set_xscale('log', nonposx='clip')
ax.set_yscale('log', nonposy='clip')
ax.set_xlabel(r'$\ell$')
ax.set_xlim(1.e1, 4.e3)
ax.legend(loc=0,title=r'$\frac{\hat{\kappa}[\text{fg}] \times g }{\kappa\times g}$')
plt.show()


#################################################################################
print("Plotting the primary bias")

fig=plt.figure(0,figsize=(6,6))
ax=fig.add_subplot(111)
#
plt.plot(lCenSx, np.abs(ClSx/p2d_cmblens.fPinterp(lCenS)), c='b',marker='.',label='Shear E')
plt.plot(lCenDx, np.abs(ClDx/p2d_cmblens.fPinterp(lCenD)), c='g',marker='.',label='Mag')
plt.plot(lCenQBHx, np.abs(ClQBHx/p2d_cmblens.fPinterp(lCenQBH)), c='purple',marker='.',label='BH')
plt.plot(lCenQx, np.abs(ClQx/p2d_cmblens.fPinterp(lCenQ)), c='r',marker='o',lw=3,label='QE')
#
ax.set_xscale('log', nonposx='clip')
ax.set_yscale('log', nonposy='clip')
ax.set_xlabel(r'$\ell$')
ax.set_xlim(1.e1, 4.e3)
ax.legend(loc=0,title=r'$\frac{\hat{\kappa}[\text{fg}] \times \kappa }{C^\kappa }$')
plt.show()


#################################################################################
print("Plotting the non-Gaussian noise contribution from source trispectrum")

fig=plt.figure(0,figsize=(6,6))
ax=fig.add_subplot(111)
#
plt.plot(lCenS, ClS/(fNsCmb_fft(lCenS)+p2d_cmblens.fPinterp(lCenS)), c='b', marker='.',label='Shear E')
plt.plot(lCenD, ClD/(fNdCmb_fft(lCenD)+p2d_cmblens.fPinterp(lCenD)), c='g', marker='.',label='Mag')
plt.plot(lCenQBH, ClQBH/(fNqBHCmb_fft(lCenQBH)+p2d_cmblens.fPinterp(lCenQBH)), c='purple', marker='.',label='BH')
plt.plot(lCenQ, ClQ/(fNqCmb_fft(lCenQ)+p2d_cmblens.fPinterp(lCenQ)), c='r',lw=3, marker='o',label='QE')
#
ax.set_xscale('log', nonposx='clip')
ax.set_yscale('log', nonposy='clip')
ax.set_xlabel(r'$\ell$')
ax.set_xlim(1.e1, 4.e3)
ax.legend(loc=0,title=r'$\frac{\hat{\kappa}[\text{fg}] \times \hat{\kappa}[\text{fg}]}{C^\kappa + N^\kappa}$')
plt.show()


#################################################################################
print("Plotting the non-Gaussian bias from source trispectrum")

fig=plt.figure(0,figsize=(6,6))
ax=fig.add_subplot(111)
#
plt.plot(lCenS, ClS/p2d_cmblens.fPinterp(lCenS), c='b', marker='.',label='Shear E')
plt.plot(lCenD, ClD/p2d_cmblens.fPinterp(lCenD), c='g', marker='.',label='Mag')
plt.plot(lCenQBH, ClQBH/p2d_cmblens.fPinterp(lCenQBH), c='purple', marker='.',label='BH')
plt.plot(lCenQ, ClQ/p2d_cmblens.fPinterp(lCenQ), c='r',lw=3, marker='o',label='QE')
#
ax.set_xscale('log', nonposx='clip')
ax.set_yscale('log', nonposy='clip')
ax.set_xlabel(r'$\ell$')
ax.set_xlim(1.e1, 4.e3)
ax.legend(loc=0,title=r'$\frac{\hat{\kappa}[\text{fg}] \times \hat{\kappa}[\text{fg}]}{C^\kappa}$')
plt.show()	
