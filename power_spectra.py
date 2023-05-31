from __future__ import division
from builtins import map
from multiprocessing import Pool
from past.utils import old_div
from importlib import reload
import sys

import universe
reload(universe)
from universe import *

import halo_fit
reload(halo_fit)
from halo_fit import *

import cmb_ilc
from cmb_ilc import *

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

#########################################################################################################################

# multipoles to include in the lensing reconstruction
lMin = 30.; lMax = 3.5e3

# ell bins for power spectra
nBins = 21  # number of bins
lRange = (1., 2.*lMax)  # range for power spectra


##################################################################################
print("CMB lensing power spectrum")

u = UnivPlanck15()
halofit = Halofit(u, save=False)
w_cmblens = WeightLensSingle(u, z_source=1100., name="cmblens")
w_lsstgold = WeightTracerLSSTGold(u)
p2d_cmblens = P2dAuto(u, halofit, w_cmblens, save=False)
p2d_lsstgold = P2dAuto(u, halofit, w_lsstgold, fPnoise=lambda l:1./w_lsstgold.ngal, nProc=3, save=True)
p2d_lsstgoldcmblens = P2dCross(u, halofit, w_lsstgold, w_cmblens, nProc=3, save=True)

cmb = CMB(beam=1.4, noise=6.3, lMin=lMin/2, lMaxT=2*lMax, lMaxP=lMax, atm=True, fg=True)
forCtotal = lambda l: cmb.flensedTT(l)
L = np.logspace(np.log10(lMin/2.), np.log10(2.*lMax), 1001, 10.)
F = np.array(list(map(forCtotal, L)))
cmb.fCtotal = interp1d(L, F, kind='linear', bounds_error=False, fill_value=0.)

fNqCmb_fft = baseMap.forecastN0Kappa(cmb.flensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False)


def measure_spectrum():
   cmb0Fourier = baseMap.genGRF(cmb.funlensedTT, test=False)
   cmb0 = baseMap.inverseFourier(cmb0Fourier)
   kCmbFourier = baseMap.genGRF(p2d_cmblens.fPinterp, test=False)
   lensedCmb = baseMap.doLensing(cmb0, kappaFourier=kCmbFourier)
   lensedCmbFourier = baseMap.fourier(lensedCmb)
   kCmbQEFourier = baseMap.computeQuadEstKappaNorm(cmb.flensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=lensedCmbFourier, test=False)
   lCen, Cl, sCl = baseMap.crossPowerSpectrum(kCmbQEFourier, kCmbFourier, plot=False, save=False, nBins=nBins)
   return lCen, Cl
  
  
def mean_spectrum(lmin,lmax):
   ells = np.linspace(lmin,lmax,2000)
   Ckk = p2d_cmblens.fPinterp(ells)
   return np.sum((2*ells+1)*Ckk)/np.sum((2*ells+1))
   
def mean_noise(lmin,lmax):
   ells = np.linspace(lmin,lmax,2000)
   Nkk = fNqCmb_fft(ells)
   return np.sum((2*ells+1)*Nkk)/np.sum((2*ells+1))

lCen,_ = measure_spectrum()
ell = baseMap.l
lEdges = np.logspace(np.log10(1.), np.log10(np.max(ell)), nBins, 10.)

result = []

M = 50

for i in range(M):
   print(i)
   result.append(measure_spectrum()[1])
  
result = np.array(result)

mean = np.mean(result,axis=0)
std = np.std(result,axis=0)
thy = np.array([mean_spectrum(lEdges[i],lEdges[i+1]) for i in range(nBins-1)])
thyN = np.array([mean_noise(lEdges[i],lEdges[i+1]) for i in range(nBins-1)])

np.savetxt('mean_power.txt',mean)
np.savetxt('std_power.txt',std)
np.savetxt('theory_power.txt',thy)
np.savetxt('noise_power.txt',thyN)
np.savetxt('lCen.txt',lCen)
np.savetxt('lEdges.txt',lEdges)
np.savetxt('theory_dumb.txt',p2d_cmblens.fPinterp(lCen))
