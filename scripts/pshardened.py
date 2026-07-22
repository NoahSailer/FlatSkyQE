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
#print("Map properties")

# number of pixels for the flat map
nX = 400 #1200
nY = 400 #1200

# map dimensions in degrees
sizeX = 10.
sizeY = 10.

# basic map object
baseMap = FlatMap(nX=nX, nY=nY, sizeX=sizeX*np.pi/180., sizeY=sizeY*np.pi/180.)

# multipoles to include in the lensing reconstruction
lMin = 30.; lMax = 3.5e3

# ell bins for power spectra
nBins = 21  # number of bins
lRange = (1., 2.*lMax)  # range for power spectra


##################################################################################
##################################################################################
print("CMB experiment properties")

# Adjust the lMin and lMax to the assumptions of the analysis
# CMB S3 specs
cmb = CMB(beam=1.4, noise=7., lMin=lMin, lMaxT=lMax, lMaxP=lMax, atm=False, fg=True)
# Total power spectrum, for the lens reconstruction
forCtotal = lambda l: cmb.ftotalTT(l) #cmb.flensedTT(l) #+ cmb.fdetectorNoise(l)
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
clusterRadius = 2.2 * np.pi/(60. * 180.) # radians


l = np.genfromtxt('l.txt')
p = np.genfromtxt('profile.txt')
profile = interp1d(l,p,kind='linear')

#################################################################################
print("Calculate noises and response")

fNqCmb_fft = baseMap.forecastN0Kappa(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False)
fNs_fft = baseMap.forecastN0S(cmb.fCtotal, lMin=lMin, lMax=lMax, test=False)
fNqBHCmb_fft = baseMap.forecastN0KappaBH(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False)
fNqBHgCmb_fft = baseMap.forecastN0KappaBH(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False, sigma = clusterRadius)
fNqBH2Cmb_fft = baseMap.forecastN0KappaBH2(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False, sigma = clusterRadius)
fNqBHPCmb_fft = baseMap.forecastN0KappaBH2(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False, u = profile)
response = baseMap.response(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False, cache=None)
determinant = baseMap.determinant(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False, cache=None)


#################################################################################
print("Checking the noises")

tFourier = baseMap.genGRF(cmb.fCtotal, test=False)
t = baseMap.inverseFourier(tFourier)
#print("plot the GRF")
#baseMap.plot(t, save=False)
#print "check the power spectrum"
#lCen, Cl, sCl = baseMap.powerSpectrum(tFourier, theory=[cmb.fCtotal], plot=True, save=False, dataLabel=r'$t\times t$', theoryLabel=r'$C^\text{tot}_\ell$')

pathQ = "./output/pQ.txt"
baseMap.computeQuadEstKappaNorm(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=tFourier, test=False, path=pathQ)
pQFourier = baseMap.loadDataFourier(pathQ)

pathS = "./output/ps.txt"
baseMap.computeQuadEstSNorm(cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=tFourier, test=False, path=pathS)
pSFourier = baseMap.loadDataFourier(pathS)

pathQBH = "./output/pQBH.txt"
baseMap.computeQuadEstKappaPointSourceHardenedNorm(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=tFourier, test=False, path=pathQBH)
pQBHFourier = baseMap.loadDataFourier(pathQBH)

pathQBHg = "./output/pQBHg.txt"
baseMap.computeQuadEstKappaPointSourceHardenedNorm(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=tFourier, test=False, path=pathQBHg, sigma=clusterRadius)
pQBHgFourier = baseMap.loadDataFourier(pathQBHg)

pathQBH2 = "./output/pQBH2.txt"
baseMap.computeQuadEstBiasHardened2Norm(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=tFourier, test=False, path=pathQBH2, sigma=clusterRadius)
pQBH2Fourier = baseMap.loadDataFourier(pathQBH2)

pathQBHP = "./output/pQBHP.txt"
baseMap.computeQuadEstBiasHardened2Norm(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=tFourier, test=False, path=pathQBHP, u=profile)
pQBHPFourier = baseMap.loadDataFourier(pathQBHP)


print("Auto-power: kappa_rec")
lCen, Cl, sCl = baseMap.powerSpectrum(pQFourier,theory=[fNqCmb_fft], plot=True, save=True, dataLabel=r'$\hat{\kappa}\times\hat{\kappa}$', ylabel=r'$N^{\kappa}_L$', name="auto_QE")

print("Auto-power: S_rec")
lCen, Cl, sCl = baseMap.powerSpectrum(pSFourier,theory=[fNs_fft], plot=True, save=True, dataLabel=r'$\hat{s}\times\hat{s}$',ylabel=r'$N^{s}_L$',name="auto_S")

print("Auto-power: kappa^BH_rec")
lCen, Cl, sCl = baseMap.powerSpectrum(pQBHFourier,theory=[fNqBHCmb_fft], plot=True, save=True, dataLabel=r'$\hat{\kappa}^\text{PSH}\times\hat{\kappa}^\text{PSH}$', ylabel=r'$N^{\kappa^\text{PSH}}_L$',name="auto_BH")

print("Auto-power: kappa^BHg_rec")
lCen, Cl, sCl = baseMap.powerSpectrum(pQBHgFourier,theory=[fNqBHgCmb_fft], plot=True, save=True, dataLabel=r'$\hat{\kappa}^\text{BHG}\times\hat{\kappa}^\text{BHG}$', ylabel=r'$N^{\kappa^\text{BHG}}_L$',name="auto_BHg")

print("Auto-power: kappa^BHP_rec")
lCen, Cl, sCl = baseMap.powerSpectrum(pQBHPFourier,theory=[fNqBHPCmb_fft], plot=True, save=True, dataLabel=r'$\hat{\kappa}^\text{BHP}\times\hat{\kappa}^\text{BHP}$', ylabel=r'$N^{\kappa^\text{BHP}}_L$',name="auto_BHP")



##################################################################################
print("Generate GRF unlensed CMB map (debeamed)")

cmb0Fourier = baseMap.genGRF(cmb.funlensedTT, test=False)
cmb0 = baseMap.inverseFourier(cmb0Fourier)
#print("plot unlensed CMB map")
#baseMap.plot(cmb0)
#print("check the power spectrum")
#lCen, Cl, sCl = baseMap.powerSpectrum(cmb0Fourier, theory=[cmb.funlensedTT], plot=True, save=False)


##################################################################################
print("Generate GRF kappa map")

kCmbFourier = baseMap.genGRF(p2d_cmblens.fPinterp, test=False)
kCmb = baseMap.inverseFourier(kCmbFourier)
#print("plot kappa map")
#baseMap.plot(kCmb)
#print("check the power spectrum")
#lCen, Cl, sCl = baseMap.powerSpectrum(kCmbFourier, theory=[p2d_cmblens.fPinterp], plot=True, save=False)


##################################################################################
print("Lens the CMB map")

lensedCmb = baseMap.doLensing(cmb0, kappaFourier=kCmbFourier)
lensedCmbFourier = baseMap.fourier(lensedCmb)
#print("plot lensed CMB map")
#baseMap.plot(lensedCmb, save=False)
#print "check the power spectrum"
#lCen, Cl, sCl = baseMap.powerSpectrum(lensedCmbFourier, theory=[cmb.flensedTT], plot=True, save=False, dataLabel=r'$T\times T$', theoryLabel=r'$C_\ell$')


##################################################################################
print("Checking cross correlation on lensed CMB map")

pathQ = "./output/pQ.txt"
baseMap.computeQuadEstKappaNorm(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=lensedCmbFourier, test=False, path=pathQ)
pQFourier = baseMap.loadDataFourier(pathQ)

pathS = "./output/ps.txt"
baseMap.computeQuadEstSNorm(cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=lensedCmbFourier, test=False, path=pathS)
pSFourier = baseMap.loadDataFourier(pathS)

pathQBH = "./output/pQBH.txt"
baseMap.computeQuadEstKappaPointSourceHardenedNorm(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=lensedCmbFourier, test=False, path=pathQBH)
pQBHFourier = baseMap.loadDataFourier(pathQBH)

pathQBH2 = "./output/pQBH2.txt"
baseMap.computeQuadEstBiasHardened2Norm(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=lensedCmbFourier, test=False, path=pathQBH2, sigma=clusterRadius)
pQBH2Fourier = baseMap.loadDataFourier(pathQBH2)

pathQBHP = "./output/pQBHP.txt"
baseMap.computeQuadEstBiasHardened2Norm(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=lensedCmbFourier, test=False, path=pathQBHP, u=profile)
pQBHPFourier = baseMap.loadDataFourier(pathQBHP)


print("Cross-power: kappa_rec")
lCen, Cl, sCl = baseMap.crossPowerSpectrum(pQFourier, kCmbFourier, theory=[p2d_cmblens.fPinterp], plot=True, save=True, dataLabel=r'$\hat{\kappa} \times \kappa$', ylabel=r'$C^\kappa_L$', name="QE_Tlensedxtrue")

print("Cross-power: S_rec")
lCen, Cl, sCl = baseMap.crossPowerSpectrumFiltered(pSFourier, kCmbFourier, fNs_fft, p2d_cmblens.fPinterp, theory=[response], plot=True, save=True, dataLabel=r'$(\hat{s}/N^s) \times (\kappa/C^\kappa)$', ylabel=r'$\mathcal{R}_L$', name="S_Tlensedxtrue")

print("Cross-power: kappa^BH_rec")
lCen, Cl, sCl = baseMap.crossPowerSpectrum(pQBHFourier, kCmbFourier, theory=[p2d_cmblens.fPinterp], plot=True, save=True, dataLabel=r'$\hat{\kappa}^\text{PSH} \times \kappa$', ylabel=r'$C^\kappa_L$', name="BH_Tlensedxtrue")

print("Cross-power: kappa^BH2_rec")
lCen, Cl, sCl = baseMap.crossPowerSpectrum(pQBH2Fourier, kCmbFourier, theory=[p2d_cmblens.fPinterp], plot=True, save=True, dataLabel=r'$\hat{\kappa}^\text{BH2} \times \kappa$', ylabel=r'$C^\kappa_L$', name="BH2_Tlensedxtrue")

print("Cross-power: kappa^BHP_rec")
lCen, Cl, sCl = baseMap.crossPowerSpectrum(pQBHPFourier, kCmbFourier, theory=[p2d_cmblens.fPinterp], plot=True, save=True, dataLabel=r'$\hat{\kappa}^\text{BHP} \times \kappa$', ylabel=r'$C^\kappa_L$', name="BHP_Tlensedxtrue")


##################################################################################
print("Generate Poisson point source map")

nbar = 5.e3 
scaledPoisson = baseMap.genPoissonWhiteNoise(nbar=nbar, fradioPoisson=cmb.fradioPoisson, norm=False, test=False)*0.
scaledPoisson[50,50] = -10.
scaledPoisson[100,100] = 20.
scaledPoissonFourier = baseMap.fourier(-scaledPoisson)

poissonTheory = lambda l: cmb.fradioPoisson(l)

#print "plot scaled poisson map"
#baseMap.plot(scaledPoisson, save=False)
#print "check the power spectrum"
#lCen, Cl, sCl = baseMap.powerSpectrum(scaledPoissonFourier, theory=[poissonTheory], plot=True, save=False, dataLabel=r'$\tilde{S}\times \tilde{S}$', theoryLabel=r"$C^{\Tilde{S}}_\ell$")


##################################################################################
print("Checking cross correlation on tilde{S} map")

pathQ = "./output/pQ.txt"
baseMap.computeQuadEstKappaNorm(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=scaledPoissonFourier, test=False, path=pathQ)
pQFourier = baseMap.loadDataFourier(pathQ)

pathS = "./output/ps.txt"
baseMap.computeQuadEstSNorm(cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=scaledPoissonFourier, test=False, path=pathS)
pSFourier = baseMap.loadDataFourier(pathS)

pathQBH = "./output/pQBH.txt"
baseMap.computeQuadEstKappaPointSourceHardenedNorm(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=scaledPoissonFourier, test=False, path=pathQBH)
pQBHFourier = baseMap.loadDataFourier(pathQBH)


baseMap.plot(baseMap.inverseFourier(scaledPoissonFourier))
baseMap.plot(baseMap.inverseFourier(pSFourier))

print(baseMap.inverseFourier(scaledPoissonFourier)[50,50])
print(baseMap.inverseFourier(scaledPoissonFourier)[100,100])
print(baseMap.inverseFourier(pSFourier)[50,50])
print(baseMap.inverseFourier(pSFourier)[100,100])
import sys
sys.exit()

# number of objects per pixel
Ngal_perpix = nbar * baseMap.dX * baseMap.dY
# number of objects per map
Ngal = Ngal_perpix * baseMap.nX * baseMap.nY
# area of single pixel
singleA = (baseMap.fSky * 4.*np.pi) / (baseMap.nX*baseMap.nY)
# rms flux of sources
sRMS = np.sqrt(np.sum(scaledPoisson**2.)/Ngal) * singleA

sTrueFourier = scaledPoissonFourier * sRMS
sTrueTheory = lambda l: poissonTheory(l) * sRMS**2.

print("Cross-power: kappa_rec")
lCen, Cl, sCl = baseMap.crossPowerSpectrumFiltered(pQFourier, sTrueFourier, fNqCmb_fft, sTrueTheory, theory=[response], plot=True, save=True, dataLabel=r'$(\hat{\kappa}/N^\kappa) \times (s/C^s)$', ylabel=r'$\mathcal{R}_L$', name="QE_Sxtrue")

print("Cross-power: S_rec")
lCen, Cl, sCl = baseMap.crossPowerSpectrum(pSFourier, sTrueFourier, theory=[sTrueTheory], plot=True, save=True, dataLabel=r'$\hat{s} \times s$', ylabel=r'$C^s_L$', name="S_Sxtrue")

print("Cross-power: kappa^BH_rec")
lCen, Cl, sCl = baseMap.crossPowerSpectrumFiltered(pQBHFourier, sTrueFourier, fNqBHCmb_fft, sTrueTheory, theory=[response], plot=True, save=True, dataLabel=r'$(\hat{\kappa}^\text{PSH}/N^{\kappa^\text{PSH}}) \times (s/C^s)$', ylabel=r'$\mathcal{R}_L$', name="BH_Sxtrue")




