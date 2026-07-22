from __future__ import division
from builtins import map
from past.utils import old_div
#from importlib import reload
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
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
lMax = float(sys.argv[1])

# ell bins for power spectra
nBins = 21  # number of bins
lRange = (1., 2.*lMax)  # range for power spectra

##################################################################################
print("CMB experiment properties")

# Adjust the lMin and lMax to the assumptions of the analysis
# CMB S3 specs
cmb = CMB(beam=1.4, noise=6., lMin=lMin, lMaxT=lMax, lMaxP=lMax, atm=False, fg=True)
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
w_lsstgold = WeightTracerLSSTGold(u)
p2d_cmblens = P2dAuto(u, halofit, w_cmblens, save=False)
p2d_lsstgold = P2dAuto(u, halofit, w_lsstgold, fPnoise=lambda l:1./w_lsstgold.ngal, nProc=3, save=True)
p2d_lsstgoldcmblens = P2dCross(u, halofit, w_lsstgold, w_cmblens, nProc=3, save=True)
clusterRadius = 2.2 * np.pi/(180. * 60.) # radians


#################################################################################
print("Calculate noises and response")

print("- standard QE")
fNqCmb_fft = baseMap.forecastN0Kappa(cmb.flensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False)
print("- shear E-mode estimator")
fNsCmb_fft = baseMap.forecastN0KappaShear(cmb.flensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, corr=True, test=False)
print("- magnification estimator")
fNdCmb_fft = baseMap.forecastN0KappaDilation(cmb.flensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, corr=True, test=False)
print("- bias hardened estimator")
fNqBHCmb_fft = baseMap.forecastN0KappaBH(cmb.flensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False)
print("- bias hardened estimator with Gaussian profile")
fNqBHgCmb_fft = baseMap.forecastN0KappaBH2(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False, cache=None, sigma=clusterRadius)


#################################################################################
cQ = 'r'
cS = 'g'
cD = 'g'
cBH = 'b'
cBHg = 'orange'

cKsz = 'r'
cTsz = 'b'
cCib = 'g'
cRadiops = 'y'
cAll = 'cyan'


#################################################################################
# rescale the error bars with area,
# because of the weird octant thing in the Sehgal sims

octantArea = 4.*np.pi * (180./np.pi)**2   # deg 2
octantArea /= 8.
ourArea = 81. * 20.**2  # deg 2
uncertaintyFactor = np.sqrt(ourArea / octantArea)


def plot_data(lCen, dataAvg, dataStd):

   ClkCmb = p2d_cmblens.fPinterp(lCen)
   NqCmb = fNqCmb_fft(lCen)   
   nBins = len(lCen) + 1
   #lRange = (1., 2.*lMax)
   #lEdges = np.logspace(np.log10(lRange[0]), np.log10(lRange[-1]), nBins, 10.)
   ell = baseMap.l.flatten()
   lEdges = np.logspace(np.log10(1.), np.log10(np.max(ell)), nBins, 10.)
   Nmodes = lEdges[1:]**2. - lEdges[:-1]**2.   

   ##################################################################################
   # Mean values 
   i=0
   # Radio point sources
   #
   # Trispectrum
   ClqCmbRadiops = dataAvg[int(0+20*i)]
   ClsCmbRadiops = dataAvg[int(1+20*i)]
   CldCmbRadiops = dataAvg[int(2+20*i)]
   ClBHCmbRadiops = dataAvg[int(3+20*i)]
   ClBHgCmbRadiops = dataAvg[int(4+20*i)]
   # Primary
   ClqCmbRadiopsKappa = dataAvg[int(5+20*i)]
   ClsCmbRadiopsKappa = dataAvg[int(6+20*i)]
   CldCmbRadiopsKappa = dataAvg[int(7+20*i)]
   ClBHCmbRadiopsKappa = dataAvg[int(8+20*i)]
   ClBHgCmbRadiopsKappa = dataAvg[int(9+20*i)]
   # k_rec x LSST
   ClqCmbRadiopsLsstgold = dataAvg[int(10+20*i)]
   ClsCmbRadiopsLsstgold = dataAvg[int(11+20*i)]
   CldCmbRadiopsLsstgold = dataAvg[int(12+20*i)]
   ClBHCmbRadiopsLsstgold = dataAvg[int(13+20*i)]
   ClBHgCmbRadiopsLsstgold = dataAvg[int(14+20*i)]
   # Secondary
   ClqCmbRadiopsSec = dataAvg[int(15+20*i)]
   CldCmbRadiopsSec = dataAvg[int(16+20*i)]
   ClsCmbRadiopsSec = dataAvg[int(17+20*i)]
   ClBHCmbRadiopsSec = dataAvg[int(18+20*i)]
   ClBHgCmbRadiopsSec = dataAvg[int(19+20*i)]

   i=1
   # CIB
   #
   # Trispectrum
   ClqCmbCib = dataAvg[int(0+20*i)]
   ClsCmbCib = dataAvg[int(1+20*i)]
   CldCmbCib = dataAvg[int(2+20*i)]
   ClBHCmbCib = dataAvg[int(3+20*i)]
   ClBHgCmbCib = dataAvg[int(4+20*i)]
   # Primary
   ClqCmbCibKappa = dataAvg[int(5+20*i)]
   ClsCmbCibKappa = dataAvg[int(6+20*i)]
   CldCmbCibKappa = dataAvg[int(7+20*i)]
   ClBHCmbCibKappa = dataAvg[int(8+20*i)]
   ClBHgCmbCibKappa = dataAvg[int(9+20*i)]
   # k_rec x LSST
   ClqCmbCibLsstgold = dataAvg[int(10+20*i)]
   ClsCmbCibLsstgold = dataAvg[int(11+20*i)]
   CldCmbCibLsstgold = dataAvg[int(12+20*i)]
   ClBHCmbCibLsstgold = dataAvg[int(13+20*i)]
   ClBHgCmbCibLsstgold = dataAvg[int(14+20*i)]
   # Secondary
   ClqCmbCibSec = dataAvg[int(15+20*i)]
   CldCmbCibSec = dataAvg[int(16+20*i)]
   ClsCmbCibSec = dataAvg[int(17+20*i)]
   ClBHCmbCibSec = dataAvg[int(18+20*i)]
   ClBHgCmbCibSec = dataAvg[int(19+20*i)]

   i=2
   # tSZ
   #
   # Trispectrum
   ClqCmbTsz = dataAvg[int(0+20*i)]
   ClsCmbTsz = dataAvg[int(1+20*i)]
   CldCmbTsz = dataAvg[int(2+20*i)]
   ClBHCmbTsz = dataAvg[int(3+20*i)]
   ClBHgCmbTsz = dataAvg[int(4+20*i)]
   # Primary
   ClqCmbTszKappa = dataAvg[int(5+20*i)]
   ClsCmbTszKappa = dataAvg[int(6+20*i)]
   CldCmbTszKappa = dataAvg[int(7+20*i)]
   ClBHCmbTszKappa = dataAvg[int(8+20*i)]
   ClBHgCmbTszKappa = dataAvg[int(9+20*i)]
   # k_rec x LSST
   ClqCmbTszLsstgold = dataAvg[int(10+20*i)]
   ClsCmbTszLsstgold = dataAvg[int(11+20*i)]
   CldCmbTszLsstgold = dataAvg[int(12+20*i)]
   ClBHCmbTszLsstgold = dataAvg[int(13+20*i)]
   ClBHgCmbTszLsstgold = dataAvg[int(14+20*i)]
   # Secondary
   ClqCmbTszSec = dataAvg[int(15+20*i)]
   CldCmbTszSec = dataAvg[int(16+20*i)]
   ClsCmbTszSec = dataAvg[int(17+20*i)]
   ClBHCmbTszSec = dataAvg[int(18+20*i)]
   ClBHgCmbTszSec = dataAvg[int(19+20*i)]

   i=3
   # kSZ
   # 
   # Trispectrum
   ClqCmbKsz = dataAvg[int(0+20*i)]
   ClsCmbKsz = dataAvg[int(1+20*i)]
   CldCmbKsz = dataAvg[int(2+20*i)]
   ClBHCmbKsz = dataAvg[int(3+20*i)]
   ClBHgCmbKsz = dataAvg[int(4+20*i)]
   # Primary
   ClqCmbKszKappa = dataAvg[int(5+20*i)]
   ClsCmbKszKappa = dataAvg[int(6+20*i)]
   CldCmbKszKappa = dataAvg[int(7+20*i)]
   ClBHCmbKszKappa = dataAvg[int(8+20*i)]
   ClBHgCmbKszKappa = dataAvg[int(9+20*i)]
   # k_rec x LSST 
   ClqCmbKszLsstgold = dataAvg[int(10+20*i)]
   ClsCmbKszLsstgold = dataAvg[int(11+20*i)]
   CldCmbKszLsstgold = dataAvg[int(12+20*i)]
   ClBHCmbKszLsstgold = dataAvg[int(13+20*i)]
   ClBHgCmbKszLsstgold = dataAvg[int(14+20*i)]
   # Secondary
   ClqCmbKszSec = dataAvg[int(15+20*i)]
   CldCmbKszSec = dataAvg[int(16+20*i)]
   ClsCmbKszSec = dataAvg[int(17+20*i)]
   ClBHCmbKszSec = dataAvg[int(18+20*i)]
   ClBHgCmbKszSec = dataAvg[int(19+20*i)]

   i=4
   # Dust
   # 
   # Trispectrum
   ClqCmbDust = dataAvg[int(0+20*i)]
   ClsCmbDust = dataAvg[int(1+20*i)]
   CldCmbDust = dataAvg[int(2+20*i)]
   ClBHCmbDust = dataAvg[int(3+20*i)]
   ClBHgCmbDust = dataAvg[int(4+20*i)]
   # Primary
   ClqCmbDustKappa = dataAvg[int(5+20*i)]
   ClsCmbDustKappa = dataAvg[int(6+20*i)]
   CldCmbDustKappa = dataAvg[int(7+20*i)]
   ClBHCmbDustKappa = dataAvg[int(8+20*i)]
   ClBHgCmbDustKappa = dataAvg[int(9+20*i)]
   # k_rec x LSST 
   ClqCmbDustLsstgold = dataAvg[int(10+20*i)]
   ClsCmbDustLsstgold = dataAvg[int(11+20*i)]
   CldCmbDustLsstgold = dataAvg[int(12+20*i)]
   ClBHCmbDustLsstgold = dataAvg[int(13+20*i)]
   ClBHgCmbDustLsstgold = dataAvg[int(14+20*i)]
   # Secondary
   ClqCmbDustSec = dataAvg[int(15+20*i)]
   CldCmbDustSec = dataAvg[int(16+20*i)]
   ClsCmbDustSec = dataAvg[int(17+20*i)]
   ClBHCmbDustSec = dataAvg[int(18+20*i)]
   ClBHgCmbDustSec = dataAvg[int(19+20*i)]

   i=5
   # FreeFree
   # 
   # Trispectrum
   ClqCmbFreeFree = dataAvg[int(0+20*i)]
   ClsCmbFreeFree = dataAvg[int(1+20*i)]
   CldCmbFreeFree = dataAvg[int(2+20*i)]
   ClBHCmbFreeFree = dataAvg[int(3+20*i)]
   ClBHgCmbFreeFree = dataAvg[int(4+20*i)]
   # Primary
   ClqCmbFreeFreeKappa = dataAvg[int(5+20*i)]
   ClsCmbFreeFreeKappa = dataAvg[int(6+20*i)]
   CldCmbFreeFreeKappa = dataAvg[int(7+20*i)]
   ClBHCmbFreeFreeKappa = dataAvg[int(8+20*i)]
   ClBHgCmbFreeFreeKappa = dataAvg[int(9+20*i)]
   # k_rec x LSST 
   ClqCmbFreeFreeLsstgold = dataAvg[int(10+20*i)]
   ClsCmbFreeFreeLsstgold = dataAvg[int(11+20*i)]
   CldCmbFreeFreeLsstgold = dataAvg[int(12+20*i)]
   ClBHCmbFreeFreeLsstgold = dataAvg[int(13+20*i)]
   ClBHgCmbFreeFreeLsstgold = dataAvg[int(14+20*i)]
   # Secondary
   ClqCmbFreeFreeSec = dataAvg[int(15+20*i)]
   CldCmbFreeFreeSec = dataAvg[int(16+20*i)]
   ClsCmbFreeFreeSec = dataAvg[int(17+20*i)]
   ClBHCmbFreeFreeSec = dataAvg[int(18+20*i)]
   ClBHgCmbFreeFreeSec = dataAvg[int(19+20*i)]

   i=6
   # Sync
   # 
   # Trispectrum
   ClqCmbSync = dataAvg[int(0+20*i)]
   ClsCmbSync = dataAvg[int(1+20*i)]
   CldCmbSync = dataAvg[int(2+20*i)]
   ClBHCmbSync = dataAvg[int(3+20*i)]
   ClBHgCmbSync = dataAvg[int(4+20*i)]
   # Primary
   ClqCmbSyncKappa = dataAvg[int(5+20*i)]
   ClsCmbSyncKappa = dataAvg[int(6+20*i)]
   CldCmbSyncKappa = dataAvg[int(7+20*i)]
   ClBHCmbSyncKappa = dataAvg[int(8+20*i)]
   ClBHgCmbSyncKappa = dataAvg[int(9+20*i)]
   # k_rec x LSST 
   ClqCmbSyncLsstgold = dataAvg[int(10+20*i)]
   ClsCmbSyncLsstgold = dataAvg[int(11+20*i)]
   CldCmbSyncLsstgold = dataAvg[int(12+20*i)]
   ClBHCmbSyncLsstgold = dataAvg[int(13+20*i)]
   ClBHgCmbSyncLsstgold = dataAvg[int(14+20*i)]
   # Secondary
   ClqCmbSyncSec = dataAvg[int(15+20*i)]
   CldCmbSyncSec = dataAvg[int(16+20*i)]
   ClsCmbSyncSec = dataAvg[int(17+20*i)]
   ClBHCmbSyncSec = dataAvg[int(18+20*i)]
   ClBHgCmbSyncSec = dataAvg[int(19+20*i)]

   i=7
   # All extragalactic foregrounds
   # 
   # Trispectrum
   ClqCmbAll = dataAvg[int(0+20*i)]
   ClsCmbAll = dataAvg[int(1+20*i)]
   CldCmbAll = dataAvg[int(2+20*i)]
   ClBHCmbAll = dataAvg[int(3+20*i)]
   ClBHgCmbAll = dataAvg[int(4+20*i)]
   # Primary
   ClqCmbAllKappa = dataAvg[int(5+20*i)]
   ClsCmbAllKappa = dataAvg[int(6+20*i)]
   CldCmbAllKappa = dataAvg[int(7+20*i)]
   ClBHCmbAllKappa = dataAvg[int(8+20*i)]
   ClBHgCmbAllKappa = dataAvg[int(9+20*i)]
   # k_rec x LSST 
   ClqCmbAllLsstgold = dataAvg[int(10+20*i)]
   ClsCmbAllLsstgold = dataAvg[int(11+20*i)]
   CldCmbAllLsstgold = dataAvg[int(12+20*i)]
   ClBHCmbAllLsstgold = dataAvg[int(13+20*i)]
   ClBHgCmbAllLsstgold = dataAvg[int(14+20*i)]
   # Secondary
   ClqCmbAllSec = dataAvg[int(15+20*i)]
   CldCmbAllSec = dataAvg[int(16+20*i)]
   ClsCmbAllSec = dataAvg[int(17+20*i)]
   ClBHCmbAllSec = dataAvg[int(18+20*i)]
   ClBHgCmbAllSec = dataAvg[int(19+20*i)]
   

   ##################################################################################
   # Statistical errors   

   i=0
   # Radio point sources
   #
   # Trispectrum
   sClqCmbRadiops = dataStd[int(0+20*i)]
   sClsCmbRadiops = dataStd[int(1+20*i)]
   sCldCmbRadiops = dataStd[int(2+20*i)]
   sClBHCmbRadiops = dataStd[int(3+20*i)]
   sClBHgCmbRadiops = dataStd[int(4+20*i)]
   # Primary
   sClqCmbRadiopsKappa = dataStd[int(5+20*i)]
   sClsCmbRadiopsKappa = dataStd[int(6+20*i)]
   sCldCmbRadiopsKappa = dataStd[int(7+20*i)]
   sClBHCmbRadiopsKappa = dataStd[int(8+20*i)]
   sClBHgCmbRadiopsKappa = dataStd[int(9+20*i)]
   # k_rec x LSST
   sClqCmbRadiopsLsstgold = dataStd[int(10+20*i)]
   sClsCmbRadiopsLsstgold = dataStd[int(11+20*i)]
   sCldCmbRadiopsLsstgold = dataStd[int(12+20*i)]
   sClBHCmbRadiopsLsstgold = dataStd[int(13+20*i)]
   sClBHgCmbRadiopsLsstgold = dataStd[int(14+20*i)]
   # Secondary
   sClqCmbRadiopsSec = dataStd[int(15+20*i)]
   sCldCmbRadiopsSec = dataStd[int(16+20*i)]
   sClsCmbRadiopsSec = dataStd[int(17+20*i)]
   sClBHCmbRadiopsSec = dataStd[int(18+20*i)]
   sClBHgCmbRadiopsSec = dataStd[int(19+20*i)]

   i=1
   # CIB
   #
   # Trispectrum
   sClqCmbCib = dataStd[int(0+20*i)]
   sClsCmbCib = dataStd[int(1+20*i)]
   sCldCmbCib = dataStd[int(2+20*i)]
   sClBHCmbCib = dataStd[int(3+20*i)]
   sClBHgCmbCib = dataStd[int(4+20*i)]
   # Primary
   sClqCmbCibKappa = dataStd[int(5+20*i)]
   sClsCmbCibKappa = dataStd[int(6+20*i)]
   sCldCmbCibKappa = dataStd[int(7+20*i)]
   sClBHCmbCibKappa = dataStd[int(8+20*i)]
   sClBHgCmbCibKappa = dataStd[int(9+20*i)]
   # k_rec x LSST
   sClqCmbCibLsstgold = dataStd[int(10+20*i)]
   sClsCmbCibLsstgold = dataStd[int(11+20*i)]
   sCldCmbCibLsstgold = dataStd[int(12+20*i)]
   sClBHCmbCibLsstgold = dataStd[int(13+20*i)]
   sClBHgCmbCibLsstgold = dataStd[int(14+20*i)]
   # Secondary
   sClqCmbCibSec = dataStd[int(15+20*i)]
   sCldCmbCibSec = dataStd[int(16+20*i)]
   sClsCmbCibSec = dataStd[int(17+20*i)]
   sClBHCmbCibSec = dataStd[int(18+20*i)]
   sClBHgCmbCibSec = dataStd[int(19+20*i)]

   i=2
   # tSZ
   #
   # Trispectrum
   sClqCmbTsz = dataStd[int(0+20*i)]
   sClsCmbTsz = dataStd[int(1+20*i)]
   sCldCmbTsz = dataStd[int(2+20*i)]
   sClBHCmbTsz = dataStd[int(3+20*i)]
   sClBHgCmbTsz = dataStd[int(4+20*i)]
   # Primary
   sClqCmbTszKappa = dataStd[int(5+20*i)]
   sClsCmbTszKappa = dataStd[int(6+20*i)]
   sCldCmbTszKappa = dataStd[int(7+20*i)]
   sClBHCmbTszKappa = dataStd[int(8+20*i)]
   sClBHgCmbTszKappa = dataStd[int(9+20*i)]
   # k_rec x LSST
   sClqCmbTszLsstgold = dataStd[int(10+20*i)]
   sClsCmbTszLsstgold = dataStd[int(11+20*i)]
   sCldCmbTszLsstgold = dataStd[int(12+20*i)]
   sClBHCmbTszLsstgold = dataStd[int(13+20*i)]
   sClBHgCmbTszLsstgold = dataStd[int(14+20*i)]
   # Secondary
   sClqCmbTszSec = dataStd[int(15+20*i)]
   sCldCmbTszSec = dataStd[int(16+20*i)]
   sClsCmbTszSec = dataStd[int(17+20*i)]
   sClBHCmbTszSec = dataStd[int(18+20*i)]
   sClBHgCmbTszSec = dataStd[int(19+20*i)]

   i=3
   # kSZ
   # 
   # Trispectrum
   sClqCmbKsz = dataStd[int(0+20*i)]
   sClsCmbKsz = dataStd[int(1+20*i)]
   sCldCmbKsz = dataStd[int(2+20*i)]
   sClBHCmbKsz = dataStd[int(3+20*i)]
   sClBHgCmbKsz = dataStd[int(4+20*i)]
   # Primary
   sClqCmbKszKappa = dataStd[int(5+20*i)]
   sClsCmbKszKappa = dataStd[int(6+20*i)]
   sCldCmbKszKappa = dataStd[int(7+20*i)]
   sClBHCmbKszKappa = dataStd[int(8+20*i)]
   sClBHgCmbKszKappa = dataStd[int(9+20*i)]
   # k_rec x LSST 
   sClqCmbKszLsstgold = dataStd[int(10+20*i)]
   sClsCmbKszLsstgold = dataStd[int(11+20*i)]
   sCldCmbKszLsstgold = dataStd[int(12+20*i)]
   sClBHCmbKszLsstgold = dataStd[int(13+20*i)]
   sClBHgCmbKszLsstgold = dataStd[int(14+20*i)]
   # Secondary
   sClqCmbKszSec = dataStd[int(15+20*i)]
   sCldCmbKszSec = dataStd[int(16+20*i)]
   sClsCmbKszSec = dataStd[int(17+20*i)]
   sClBHCmbKszSec = dataStd[int(18+20*i)]
   sClBHgCmbKszSec = dataStd[int(19+20*i)]

   i=4
   # Dust
   # 
   # Trispectrum
   sClqCmbDust = dataStd[int(0+20*i)]
   sClsCmbDust = dataStd[int(1+20*i)]
   sCldCmbDust = dataStd[int(2+20*i)]
   sClBHCmbDust = dataStd[int(3+20*i)]
   sClBHgCmbDust = dataStd[int(4+20*i)]
   # Primary
   sClqCmbDustKappa = dataStd[int(5+20*i)]
   sClsCmbDustKappa = dataStd[int(6+20*i)]
   sCldCmbDustKappa = dataStd[int(7+20*i)]
   sClBHCmbDustKappa = dataStd[int(8+20*i)]
   sClBHgCmbDustKappa = dataStd[int(9+20*i)]
   # k_rec x LSST 
   sClqCmbDustLsstgold = dataStd[int(10+20*i)]
   sClsCmbDustLsstgold = dataStd[int(11+20*i)]
   sCldCmbDustLsstgold = dataStd[int(12+20*i)]
   sClBHCmbDustLsstgold = dataStd[int(13+20*i)]
   sClBHgCmbDustLsstgold = dataStd[int(14+20*i)]
   # Secondary
   sClqCmbDustSec = dataStd[int(15+20*i)]
   sCldCmbDustSec = dataStd[int(16+20*i)]
   sClsCmbDustSec = dataStd[int(17+20*i)]
   sClBHCmbDustSec = dataStd[int(18+20*i)]
   sClBHgCmbDustSec = dataStd[int(19+20*i)]

   i=5
   # FreeFree
   # 
   # Trispectrum
   sClqCmbFreeFree = dataStd[int(0+20*i)]
   sClsCmbFreeFree = dataStd[int(1+20*i)]
   sCldCmbFreeFree = dataStd[int(2+20*i)]
   sClBHCmbFreeFree = dataStd[int(3+20*i)]
   sClBHgCmbFreeFree = dataStd[int(4+20*i)]
   # Primary
   sClqCmbFreeFreeKappa = dataStd[int(5+20*i)]
   sClsCmbFreeFreeKappa = dataStd[int(6+20*i)]
   sCldCmbFreeFreeKappa = dataStd[int(7+20*i)]
   sClBHCmbFreeFreeKappa = dataStd[int(8+20*i)]
   sClBHgCmbFreeFreeKappa = dataStd[int(9+20*i)]
   # k_rec x LSST 
   sClqCmbFreeFreeLsstgold = dataStd[int(10+20*i)]
   sClsCmbFreeFreeLsstgold = dataStd[int(11+20*i)]
   sCldCmbFreeFreeLsstgold = dataStd[int(12+20*i)]
   sClBHCmbFreeFreeLsstgold = dataStd[int(13+20*i)]
   sClBHgCmbFreeFreeLsstgold = dataStd[int(14+20*i)]
   # Secondary
   sClqCmbFreeFreeSec = dataStd[int(15+20*i)]
   sCldCmbFreeFreeSec = dataStd[int(16+20*i)]
   sClsCmbFreeFreeSec = dataStd[int(17+20*i)]
   sClBHCmbFreeFreeSec = dataStd[int(18+20*i)]
   sClBHgCmbFreeFreeSec = dataStd[int(19+20*i)]

   i=6
   # Sync
   # 
   # Trispectrum
   sClqCmbSync = dataStd[int(0+20*i)]
   sClsCmbSync = dataStd[int(1+20*i)]
   sCldCmbSync = dataStd[int(2+20*i)]
   sClBHCmbSync = dataStd[int(3+20*i)]
   sClBHgCmbSync = dataStd[int(4+20*i)]
   # Primary
   sClqCmbSyncKappa = dataStd[int(5+20*i)]
   sClsCmbSyncKappa = dataStd[int(6+20*i)]
   sCldCmbSyncKappa = dataStd[int(7+20*i)]
   sClBHCmbSyncKappa = dataStd[int(8+20*i)]
   sClBHgCmbSyncKappa = dataStd[int(9+20*i)]
   # k_rec x LSST 
   sClqCmbSyncLsstgold = dataStd[int(10+20*i)]
   sClsCmbSyncLsstgold = dataStd[int(11+20*i)]
   sCldCmbSyncLsstgold = dataStd[int(12+20*i)]
   sClBHCmbSyncLsstgold = dataStd[int(13+20*i)]
   sClBHgCmbSyncLsstgold = dataStd[int(14+20*i)]
   # Secondary
   sClqCmbSyncSec = dataStd[int(15+20*i)]
   sCldCmbSyncSec = dataStd[int(16+20*i)]
   sClsCmbSyncSec = dataStd[int(17+20*i)]
   sClBHCmbSyncSec = dataStd[int(18+20*i)]
   sClBHgCmbSyncSec = dataStd[int(19+20*i)]

   i=7
   # All extragalactic foregrounds
   # 
   # Trispectrum
   sClqCmbAll = dataStd[int(0+20*i)]
   sClsCmbAll = dataStd[int(1+20*i)]
   sCldCmbAll = dataStd[int(2+20*i)]
   sClBHCmbAll = dataStd[int(3+20*i)]
   sClBHgCmbAll = dataStd[int(4+20*i)]
   # Primary
   sClqCmbAllKappa = dataStd[int(5+20*i)]
   sClsCmbAllKappa = dataStd[int(6+20*i)]
   sCldCmbAllKappa = dataStd[int(7+20*i)]
   sClBHCmbAllKappa = dataStd[int(8+20*i)]
   sClBHgCmbAllKappa = dataStd[int(9+20*i)]
   # k_rec x LSST 
   sClqCmbAllLsstgold = dataStd[int(10+20*i)]
   sClsCmbAllLsstgold = dataStd[int(11+20*i)]
   sCldCmbAllLsstgold = dataStd[int(12+20*i)]
   sClBHCmbAllLsstgold = dataStd[int(13+20*i)]
   sClBHgCmbAllLsstgold = dataStd[int(14+20*i)]
   # Secondary
   sClqCmbAllSec = dataStd[int(15+20*i)]
   sCldCmbAllSec = dataStd[int(16+20*i)]
   sClsCmbAllSec = dataStd[int(17+20*i)]
   sClBHCmbAllSec = dataStd[int(18+20*i)]
   sClBHgCmbAllSec = dataStd[int(19+20*i)]



   #################################################################################
   #################################################################################
   # Primary contraction

   fig=plt.figure(0)
   gs = gridspec.GridSpec(5, 1)
   gs.update(hspace=0.1)

   # CIB
   ax0=plt.subplot(gs[0])
   # ClkCmb
   ax0.axhline(0., color='k')
   # statistical uncertainty on lensing amplitude
   sigma = np.sqrt(2. * (ClkCmb + NqCmb)**2 / Nmodes)
   sigma0 = np.array(list(zip(sigma/ClkCmb, sigma/ClkCmb))).flatten()
   l0 = np.array(zip(lEdges[:-1],lEdges[1:])).flatten()
   ax0.fill_between(l0, -sigma0, sigma0, edgecolor='k', facecolor='gray', alpha=0.4)
   # Dilation
   #ax0.plot(lCen, 2. * CldCmbCibKappa/ClkCmb, color=cD, ls='-')
   #Up = CldCmbCibKappa + sCldCmbCibKappa
   #Down = CldCmbCibKappa - sCldCmbCibKappa
   #ax0.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cD, alpha=0.6)
   # QE
   ax0.plot(lCen, 2. * ClqCmbCibKappa/ClkCmb, color=cQ, ls='-')
   Up = ClqCmbCibKappa + sClqCmbCibKappa
   Down = ClqCmbCibKappa - sClqCmbCibKappa
   ax0.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cQ, alpha=0.6)
   # Bias hardened
   ax0.plot(lCen, 2. * ClBHCmbCibKappa/ClkCmb, color=cBH, ls='-')
   Up = ClBHCmbCibKappa + sClBHCmbCibKappa
   Down = ClBHCmbCibKappa - sClBHCmbCibKappa
   ax0.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cBH, alpha=0.6)
   # Bias hardened with Gaussian profile
   ax0.plot(lCen, 2. * ClBHgCmbCibKappa/ClkCmb, color=cBHg, ls='-')
   Up = ClBHgCmbCibKappa + sClBHgCmbCibKappa
   Down = ClBHgCmbCibKappa - sClBHgCmbCibKappa
   ax0.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cBHg, alpha=0.6)
   # Shear
   ax0.plot(lCen, 2. * ClsCmbCibKappa/ClkCmb, color=cS, ls='-')
   Up = ClsCmbCibKappa + sClsCmbCibKappa
   Down = ClsCmbCibKappa - sClsCmbCibKappa
   ax0.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cS, alpha=0.6)
   #
#   ax0.set_ylim((-0.2,0.1))
   ax0.set_ylim((-0.1, 0.25))
   ax0.set_ylabel(r'CIB')
   #
   ax0.plot([], [], color=cS, ls='-', label=r'Shear')
   ax0.plot([], [], color=cBH, ls='-', label=r'BH1')
   ax0.plot([], [], color=cBHg, ls='-', label=r'BH2')
   ax0.plot([], [], color=cQ, ls='-', label=r'QE')
   #ax0.plot([], [], color=cD, ls='-', label='Mag')
   #
   plt.setp(ax0.get_xticklabels(), visible=False)
   ax0.legend(loc='upper center', ncol=4, mode=None, handlelength=.5, fontsize='x-small', borderpad=0.1, borderaxespad=0.1, frameon=False)
   ax0.set_title(r'Relative Bias on $C_L^{\kappa_\text{CMB}}$: Primary')

   # tSZ
   ax1=plt.subplot(gs[1], sharex=ax0)
   # ClkCmb
   ax1.axhline(0., color='k')
   # statistical uncertainty on lensing amplitude
   sigma = np.sqrt(2. * (ClkCmb + NqCmb)**2 / Nmodes)
   sigma0 = np.array(list(zip(sigma/ClkCmb, sigma/ClkCmb))).flatten()
   l0 = np.array(zip(lEdges[:-1],lEdges[1:])).flatten()
   ax1.fill_between(l0, -sigma0, sigma0, edgecolor='k', facecolor='gray', alpha=0.4)
   # Dilation
   #ax1.plot(lCen, 2. * CldCmbTszKappa/ClkCmb, color=cD, ls='-', label=r'Dilation')
   #Up = CldCmbTszKappa + sCldCmbTszKappa
   #Down = CldCmbTszKappa - sCldCmbTszKappa
   #ax1.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cD, alpha=0.6)
   # QE
   ax1.plot(lCen, 2. * ClqCmbTszKappa/ClkCmb, color=cQ, ls='-', label=r'QE')
   Up = ClqCmbTszKappa + sClqCmbTszKappa
   Down = ClqCmbTszKappa - sClqCmbTszKappa
   ax1.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cQ, alpha=0.6)
   # Bias hardened
   ax1.plot(lCen, 2. * ClBHCmbTszKappa/ClkCmb, color=cBH, ls='-', label=r'BH1')
   Up = ClBHCmbTszKappa + sClBHCmbTszKappa
   Down = ClBHCmbTszKappa - sClBHCmbTszKappa
   ax1.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cBH, alpha=0.6)
   # Bias hardened with Gaussian profile
   ax1.plot(lCen, 2. * ClBHgCmbTszKappa/ClkCmb, color=cBHg, ls='-', label=r'BH2')
   Up = ClBHgCmbTszKappa + sClBHgCmbTszKappa
   Down = ClBHgCmbTszKappa - sClBHgCmbTszKappa
   ax1.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cBHg, alpha=0.6)
   # Shear
   ax1.plot(lCen, 2. * ClsCmbTszKappa/ClkCmb, color=cS, ls='-', label=r'S')
   Up = ClsCmbTszKappa + sClsCmbTszKappa
   Down = ClsCmbTszKappa - sClsCmbTszKappa
   ax1.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cS, alpha=0.6)
   #
   ax1.set_ylim((-0.1,0.1))
   ax1.set_ylabel(r'tSZ')
   #
   plt.setp(ax1.get_xticklabels(), visible=False)
   # remove last tick label for the second subplot
   yticks = ax1.yaxis.get_major_ticks()
   yticks[-1].label1.set_visible(False)

   # kSZ
   ax2=plt.subplot(gs[2], sharex=ax1)
   # ClkCmb
   ax2.axhline(0., color='k')
   # statistical uncertainty on lensing amplitude
   sigma = np.sqrt(2. * (ClkCmb + NqCmb)**2 / Nmodes)
   sigma0 = np.array(list(zip(sigma/ClkCmb, sigma/ClkCmb))).flatten()
   l0 = np.array(zip(lEdges[:-1],lEdges[1:])).flatten()
   ax2.fill_between(l0, -sigma0, sigma0, edgecolor='k', facecolor='gray', alpha=0.4)
   # Dilation
   #ax2.plot(lCen, 2. * CldCmbKszKappa/ClkCmb, color=cD, ls='-', label=r'Magnification')
   #Up = CldCmbKszKappa + sCldCmbKszKappa
   #Down = CldCmbKszKappa - sCldCmbKszKappa
   #ax2.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cD, alpha=0.6)
   # QE
   ax2.plot(lCen, 2. * ClqCmbKszKappa/ClkCmb, color=cQ, ls='-', label=r'QE')
   Up = ClqCmbKszKappa + sClqCmbKszKappa
   Down = ClqCmbKszKappa - sClqCmbKszKappa
   ax2.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cQ, alpha=0.6)
   # Bias hardened
   ax2.plot(lCen, 2. * ClBHCmbKszKappa/ClkCmb, color=cBH, ls='-', label=r'BH1')
   Up = ClBHCmbKszKappa + sClBHCmbKszKappa
   Down = ClBHCmbKszKappa - sClBHCmbKszKappa
   ax2.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cBH, alpha=0.6)
   # Bias hardened with Gaussian profile
   ax2.plot(lCen, 2. * ClBHgCmbKszKappa/ClkCmb, color=cBHg, ls='-', label=r'BH2')
   Up = ClBHgCmbKszKappa + sClBHgCmbKszKappa
   Down = ClBHgCmbKszKappa - sClBHgCmbKszKappa
   ax2.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cBHg, alpha=0.6)
   # Shear
   ax2.plot(lCen, 2. * ClsCmbKszKappa/ClkCmb, color=cS, ls='-', label=r'Shear')
   Up = ClsCmbKszKappa + sClsCmbKszKappa
   Down = ClsCmbKszKappa - sClsCmbKszKappa
   ax2.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cS, alpha=0.6)
   #
   ax2.set_ylim((-0.025,0.025))
   ax2.set_ylabel(r'kSZ')
   #
   plt.setp(ax2.get_xticklabels(), visible=False)
   # remove last tick label for the second subplot
   yticks = ax2.yaxis.get_major_ticks()
   yticks[-1].label1.set_visible(False)


   # Radio point sources
   ax3=plt.subplot(gs[3], sharex=ax2)
   # ClkCmb
   ax3.axhline(0., color='k')
   # statistical uncertainty on lensing amplitude
   sigma = np.sqrt(2. * (ClkCmb + NqCmb)**2 / Nmodes)
   sigma0 = np.array(list(zip(sigma/ClkCmb, sigma/ClkCmb))).flatten()
   l0 = np.array(zip(lEdges[:-1],lEdges[1:])).flatten()
   ax3.fill_between(l0, -sigma0, sigma0, edgecolor='k', facecolor='gray', alpha=0.4)
   # Dilation
   #ax3.plot(lCen, 2. * CldCmbRadiopsKappa/ClkCmb, color=cD, ls='-', label=r'Magnification')
   #Up = CldCmbRadiopsKappa + sCldCmbRadiopsKappa
   #Down = CldCmbRadiopsKappa - sCldCmbRadiopsKappa
   #ax3.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cD, alpha=0.6)
   # QE
   ax3.plot(lCen, 2. * ClqCmbRadiopsKappa/ClkCmb, color=cQ, ls='-', label=r'QE')
   Up = ClqCmbRadiopsKappa + sClqCmbRadiopsKappa
   Down = ClqCmbRadiopsKappa - sClqCmbRadiopsKappa
   ax3.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cQ, alpha=0.6)
   # Bias hardened
   ax3.plot(lCen, 2. * ClBHCmbRadiopsKappa/ClkCmb, color=cBH, ls='-', label=r'BH1')
   Up = ClBHCmbRadiopsKappa + sClBHCmbRadiopsKappa
   Down = ClBHCmbRadiopsKappa - sClBHCmbRadiopsKappa
   ax3.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cBH, alpha=0.6)
   # Bias hardened with Gaussian profile
   ax3.plot(lCen, 2. * ClBHgCmbRadiopsKappa/ClkCmb, color=cBH, ls='-', label=r'BH2')
   Up = ClBHgCmbRadiopsKappa + sClBHgCmbRadiopsKappa
   Down = ClBHgCmbRadiopsKappa - sClBHgCmbRadiopsKappa
   ax3.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cBHg, alpha=0.6)
   # Shear
   ax3.plot(lCen, 2. * ClsCmbRadiopsKappa/ClkCmb, color=cS, ls='-', label=r'Shear')
   Up = ClsCmbRadiopsKappa + sClsCmbRadiopsKappa
   Down = ClsCmbRadiopsKappa - sClsCmbRadiopsKappa
   ax3.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cS, alpha=0.6)
   #
   ax3.set_ylim((-0.025,0.025))
   ax3.set_ylabel(r'Radio')
   #
   plt.setp(ax3.get_xticklabels(), visible=False)
   # remove last tick label for the second subplot
   yticks = ax3.yaxis.get_major_ticks()
   yticks[-1].label1.set_visible(False)


   # all extragalactic foregrounds
   ax4=plt.subplot(gs[4], sharex=ax3)
   # ClkCmb
   ax4.axhline(0., color='k')
   # statistical uncertainty on lensing amplitude
   sigma = np.sqrt(2. * (ClkCmb + NqCmb)**2 / Nmodes)
   sigma0 = np.array(list(zip(sigma/ClkCmb, sigma/ClkCmb))).flatten()
   l0 = np.array(zip(lEdges[:-1],lEdges[1:])).flatten()
   ax4.fill_between(l0, -sigma0, sigma0, edgecolor='k', facecolor='gray', alpha=0.4)
   # Dilation
   #ax2.plot(lCen, 2. * CldCmbAllKappa/ClkCmb, color=cD, ls='-', label=r'Magnification')
   #Up = CldCmbAllKappa + sCldCmbAllKappa
   #Down = CldCmbAllKappa - sCldCmbAllKappa
   #ax2.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cD, alpha=0.6)
   # QE
   ax4.plot(lCen, 2. * ClqCmbAllKappa/ClkCmb, color=cQ, ls='-', label=r'QE')
   Up = ClqCmbAllKappa + sClqCmbAllKappa
   Down = ClqCmbAllKappa - sClqCmbAllKappa
   ax4.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cQ, alpha=0.6)
   # Bias hardened
   ax4.plot(lCen, 2. * ClBHCmbAllKappa/ClkCmb, color=cBH, ls='-', label=r'BH1')
   Up = ClBHCmbAllKappa + sClBHCmbAllKappa
   Down = ClBHCmbAllKappa - sClBHCmbAllKappa
   ax4.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cBH, alpha=0.6)
   # Bias hardened with Gaussian profile
   ax4.plot(lCen, 2. * ClBHgCmbAllKappa/ClkCmb, color=cBHg, ls='-', label=r'BH2')
   Up = ClBHgCmbAllKappa + sClBHgCmbAllKappa
   Down = ClBHgCmbAllKappa - sClBHgCmbAllKappa
   ax4.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cBHg, alpha=0.6)
   # Shear
   ax4.plot(lCen, 2. * ClsCmbAllKappa/ClkCmb, color=cS, ls='-', label=r'Shear')
   Up = ClsCmbAllKappa + sClsCmbAllKappa
   Down = ClsCmbAllKappa - sClsCmbAllKappa
   ax4.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cS, alpha=0.6)
   #
   ax4.set_ylim((-0.05,0.025))
   ax4.set_ylabel(r'All')
   #
   # remove last tick label for the second subplot
   yticks = ax4.yaxis.get_major_ticks()
   yticks[-1].label1.set_visible(False)
   
   
   ax4.set_xscale('log', nonposx='clip')
   ax4.set_xlim((30., 1.5e3))
   ax4.set_xlabel(r'$L$')

   path = "output/primary_lmax_"+str(int(lMax))+".pdf"
   fig.savefig(path, bbox_inches='tight')
   fig.clf()



   ##################################################################################
   ##################################################################################
   # Secondary contraction

   fig=plt.figure(0)
   gs = gridspec.GridSpec(5, 1)
   gs.update(hspace=0.1)

   # CIB
   ax0=plt.subplot(gs[0])
   # ClkCmb
   ax0.axhline(0., color='k')
   # statistical uncertainty on lensing amplitude
   sigma = np.sqrt(2. * (ClkCmb + NqCmb)**2 / Nmodes)
   sigma0 = np.array(list(zip(sigma/ClkCmb, sigma/ClkCmb))).flatten()
   l0 = np.array(zip(lEdges[:-1],lEdges[1:])).flatten()
   ax0.fill_between(l0, -sigma0, sigma0, edgecolor='k', facecolor='gray', alpha=0.4)
   # Dilation
   #ax0.plot(lCen, 2. * CldCmbCibSec/ClkCmb, color=cD, ls='-')
   #Up = CldCmbCibSec + sCldCmbCibSec
   #Down = CldCmbCibSec - sCldCmbCibSec
   #ax0.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cD, alpha=0.6)
   # QE
   ax0.plot(lCen, 2. * ClqCmbCibSec/ClkCmb, color=cQ, ls='-')
   Up = ClqCmbCibSec + sClqCmbCibSec
   Down = ClqCmbCibSec - sClqCmbCibSec
   ax0.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cQ, alpha=0.6)
   # Bias hardened
   ax0.plot(lCen, 2. * ClBHCmbCibSec/ClkCmb, color=cBH, ls='-')
   Up = ClBHCmbCibSec + sClBHCmbCibSec
   Down = ClBHCmbCibSec - sClBHCmbCibSec
   ax0.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cBH, alpha=0.6)
   # Bias hardened with Gaussian profile
   ax0.plot(lCen, 2. * ClBHgCmbCibSec/ClkCmb, color=cBHg, ls='-')
   Up = ClBHgCmbCibSec + sClBHgCmbCibSec
   Down = ClBHgCmbCibSec - sClBHgCmbCibSec
   ax0.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cBHg, alpha=0.6)
   # Shear
   ax0.plot(lCen, 2. * ClsCmbCibSec/ClkCmb, color=cS, ls='-')
   Up = ClsCmbCibSec + sClsCmbCibSec
   Down = ClsCmbCibSec - sClsCmbCibSec
   ax0.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cS, alpha=0.6)
   #
#   ax0.set_ylim((-0.2,0.1))
   ax0.set_ylim((-0.05, 0.05))
   ax0.set_ylabel(r'CIB')
   #
   ax0.plot([], [], color=cS, ls='-', label=r'Shear')
   ax0.plot([], [], color=cBH, ls='-', label=r'BH1')
   ax0.plot([], [], color=cBHg, ls='-', label=r'BH2')
   ax0.plot([], [], color=cQ, ls='-', label=r'QE')
   #ax0.plot([], [], color=cD, ls='-', label='Mag')
   #
   plt.setp(ax0.get_xticklabels(), visible=False)
   ax0.legend(loc='upper center', ncol=4, mode=None, handlelength=.5, fontsize='x-small', borderpad=0.1, borderaxespad=0.1, frameon=False)
   ax0.set_title(r'Relative Bias on $C_L^{\kappa_\text{CMB}}$: Secondary')

   # tSZ
   ax1=plt.subplot(gs[1], sharex=ax0)
   # ClkCmb
   ax1.axhline(0., color='k')
   # statistical uncertainty on lensing amplitude
   sigma = np.sqrt(2. * (ClkCmb + NqCmb)**2 / Nmodes)
   sigma0 = np.array(list(zip(sigma/ClkCmb, sigma/ClkCmb))).flatten()
   l0 = np.array(zip(lEdges[:-1],lEdges[1:])).flatten()
   ax1.fill_between(l0, -sigma0, sigma0, edgecolor='k', facecolor='gray', alpha=0.4)
   # Dilation
   #ax1.plot(lCen, 2. * CldCmbTszSec/ClkCmb, color=cD, ls='-', label=r'Dilation')
   #Up = CldCmbTszSec + sCldCmbTszSec
   #Down = CldCmbTszSec - sCldCmbTszSec
   #ax1.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cD, alpha=0.6)
   # QE
   ax1.plot(lCen, 2. * ClqCmbTszSec/ClkCmb, color=cQ, ls='-', label=r'QE')
   Up = ClqCmbTszSec + sClqCmbTszSec
   Down = ClqCmbTszSec - sClqCmbTszSec
   ax1.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cQ, alpha=0.6)
   # Bias hardened
   ax1.plot(lCen, 2. * ClBHCmbTszSec/ClkCmb, color=cBH, ls='-', label=r'BH1')
   Up = ClBHCmbTszSec + sClBHCmbTszSec
   Down = ClBHCmbTszSec - sClBHCmbTszSec
   ax1.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cBH, alpha=0.6)
   # Bias hardened with Gaussian profile
   ax1.plot(lCen, 2. * ClBHgCmbTszSec/ClkCmb, color=cBHg, ls='-', label=r'BH2')
   Up = ClBHgCmbTszSec + sClBHgCmbTszSec
   Down = ClBHgCmbTszSec - sClBHgCmbTszSec
   ax1.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cBHg, alpha=0.6)
   # Shear
   ax1.plot(lCen, 2. * ClsCmbTszSec/ClkCmb, color=cS, ls='-', label=r'S')
   Up = ClsCmbTszSec + sClsCmbTszSec
   Down = ClsCmbTszSec - sClsCmbTszSec
   ax1.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cS, alpha=0.6)
   #
   ax1.set_ylim((-0.05,0.05))
   ax1.set_ylabel(r'tSZ')
   #
   plt.setp(ax1.get_xticklabels(), visible=False)
   # remove last tick label for the second subplot
   yticks = ax1.yaxis.get_major_ticks()
   yticks[-1].label1.set_visible(False)

   # kSZ
   ax2=plt.subplot(gs[2], sharex=ax1)
   # ClkCmb
   ax2.axhline(0., color='k')
   # statistical uncertainty on lensing amplitude
   sigma = np.sqrt(2. * (ClkCmb + NqCmb)**2 / Nmodes)
   sigma0 = np.array(list(zip(sigma/ClkCmb, sigma/ClkCmb))).flatten()
   l0 = np.array(zip(lEdges[:-1],lEdges[1:])).flatten()
   ax2.fill_between(l0, -sigma0, sigma0, edgecolor='k', facecolor='gray', alpha=0.4)
   # Dilation
   #ax2.plot(lCen, 2. * CldCmbKszSec/ClkCmb, color=cD, ls='-', label=r'Magnification')
   #Up = CldCmbKszSec + sCldCmbKszSec
   #Down = CldCmbKszSec - sCldCmbKszSec
   #ax2.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cD, alpha=0.6)
   # QE
   ax2.plot(lCen, 2. * ClqCmbKszSec/ClkCmb, color=cQ, ls='-', label=r'QE')
   Up = ClqCmbKszSec + sClqCmbKszSec
   Down = ClqCmbKszSec - sClqCmbKszSec
   ax2.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cQ, alpha=0.6)
   # Bias hardened
   ax2.plot(lCen, 2. * ClBHCmbKszSec/ClkCmb, color=cBH, ls='-', label=r'BH1')
   Up = ClBHCmbKszSec + sClBHCmbKszSec
   Down = ClBHCmbKszSec - sClBHCmbKszSec
   ax2.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cBH, alpha=0.6)
   # Bias hardened with Gaussian profile
   ax2.plot(lCen, 2. * ClBHgCmbKszSec/ClkCmb, color=cBHg, ls='-', label=r'BH2')
   Up = ClBHgCmbKszSec + sClBHgCmbKszSec
   Down = ClBHgCmbKszSec - sClBHgCmbKszSec
   ax2.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cBHg, alpha=0.6)
   # Shear
   ax2.plot(lCen, 2. * ClsCmbKszSec/ClkCmb, color=cS, ls='-', label=r'Shear')
   Up = ClsCmbKszSec + sClsCmbKszSec
   Down = ClsCmbKszSec - sClsCmbKszSec
   ax2.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cS, alpha=0.6)
   #
   ax2.set_ylim((-0.05,0.05))
   ax2.set_ylabel(r'kSZ')
   #
   plt.setp(ax2.get_xticklabels(), visible=False)
   # remove last tick label for the second subplot
   yticks = ax2.yaxis.get_major_ticks()
   yticks[-1].label1.set_visible(False)


   # Radio point sources
   ax3=plt.subplot(gs[3], sharex=ax2)
   # ClkCmb
   ax3.axhline(0., color='k')
   # statistical uncertainty on lensing amplitude
   sigma = np.sqrt(2. * (ClkCmb + NqCmb)**2 / Nmodes)
   sigma0 = np.array(list(zip(sigma/ClkCmb, sigma/ClkCmb))).flatten()
   l0 = np.array(zip(lEdges[:-1],lEdges[1:])).flatten()
   ax3.fill_between(l0, -sigma0, sigma0, edgecolor='k', facecolor='gray', alpha=0.4)
   # Dilation
   #ax3.plot(lCen, 2. * CldCmbRadiopsSec/ClkCmb, color=cD, ls='-', label=r'Magnification')
   #Up = CldCmbRadiopsSec + sCldCmbRadiopsSec
   #Down = CldCmbRadiopsSec - sCldCmbRadiopsSec
   #ax3.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cD, alpha=0.6)
   # QE
   ax3.plot(lCen, 2. * ClqCmbRadiopsSec/ClkCmb, color=cQ, ls='-', label=r'QE')
   Up = ClqCmbRadiopsSec + sClqCmbRadiopsSec
   Down = ClqCmbRadiopsSec - sClqCmbRadiopsSec
   ax3.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cQ, alpha=0.6)
   # Bias hardened
   ax3.plot(lCen, 2. * ClBHCmbRadiopsSec/ClkCmb, color=cBH, ls='-', label=r'BH1')
   Up = ClBHCmbRadiopsSec + sClBHCmbRadiopsSec
   Down = ClBHCmbRadiopsSec - sClBHCmbRadiopsSec
   ax3.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cBH, alpha=0.6)
   # Bias hardened with Gaussian profile
   ax3.plot(lCen, 2. * ClBHgCmbRadiopsSec/ClkCmb, color=cBHg, ls='-', label=r'BH2')
   Up = ClBHgCmbRadiopsSec + sClBHgCmbRadiopsSec
   Down = ClBHgCmbRadiopsSec - sClBHgCmbRadiopsSec
   ax3.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cBHg, alpha=0.6)
   # Shear
   ax3.plot(lCen, 2. * ClsCmbRadiopsSec/ClkCmb, color=cS, ls='-', label=r'Shear')
   Up = ClsCmbRadiopsSec + sClsCmbRadiopsSec
   Down = ClsCmbRadiopsSec - sClsCmbRadiopsSec
   ax3.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cS, alpha=0.6)
   #
   ax3.set_ylim((-0.05,0.05))
   ax3.set_ylabel(r'Radio')
   #
   plt.setp(ax3.get_xticklabels(), visible=False)
   # remove last tick label for the second subplot
   yticks = ax3.yaxis.get_major_ticks()
   yticks[-1].label1.set_visible(False)


   # all extragalactic foregrounds
   ax4=plt.subplot(gs[4], sharex=ax3)
   # ClkCmb
   ax4.axhline(0., color='k')
   # statistical uncertainty on lensing amplitude
   sigma = np.sqrt(2. * (ClkCmb + NqCmb)**2 / Nmodes)
   sigma0 = np.array(list(zip(sigma/ClkCmb, sigma/ClkCmb))).flatten()
   l0 = np.array(zip(lEdges[:-1],lEdges[1:])).flatten()
   ax4.fill_between(l0, -sigma0, sigma0, edgecolor='k', facecolor='gray', alpha=0.4)
   # Dilation
   #ax2.plot(lCen, 2. * CldCmbAllSec/ClkCmb, color=cD, ls='-', label=r'Magnification')
   #Up = CldCmbAllSec + sCldCmbAllSec
   #Down = CldCmbAllSec - sCldCmbAllSec
   #ax2.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cD, alpha=0.6)
   # QE
   ax4.plot(lCen, 2. * ClqCmbAllSec/ClkCmb, color=cQ, ls='-', label=r'QE')
   Up = ClqCmbAllSec + sClqCmbAllSec
   Down = ClqCmbAllSec - sClqCmbAllSec
   ax4.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cQ, alpha=0.6)
   # Bias hardened
   ax4.plot(lCen, 2. * ClBHCmbAllSec/ClkCmb, color=cBH, ls='-', label=r'BH1')
   Up = ClBHCmbAllSec + sClBHCmbAllSec
   Down = ClBHCmbAllSec - sClBHCmbAllSec
   ax4.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cBH, alpha=0.6)
   # Bias hardened with Gaussian profile
   ax4.plot(lCen, 2. * ClBHgCmbAllSec/ClkCmb, color=cBHg, ls='-', label=r'BH1')
   Up = ClBHgCmbAllSec + sClBHgCmbAllSec
   Down = ClBHgCmbAllSec - sClBHgCmbAllSec
   ax4.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cBHg, alpha=0.6)
   # Shear
   ax4.plot(lCen, 2. * ClsCmbAllSec/ClkCmb, color=cS, ls='-', label=r'Shear')
   Up = ClsCmbAllSec + sClsCmbAllSec
   Down = ClsCmbAllSec - sClsCmbAllSec
   ax4.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cS, alpha=0.6)
   #
   ax4.set_ylim((-0.06,0.05))
   ax4.set_ylabel(r'All')
   #
   # remove last tick label for the second subplot
   yticks = ax4.yaxis.get_major_ticks()
   yticks[-1].label1.set_visible(False)
   
   
   ax4.set_xscale('log', nonposx='clip')
   ax4.set_xlim((30., 1.5e3))
   ax4.set_xlabel(r'$L$')

   path = "output/secondary_lmax_"+str(int(lMax))+".pdf"
   fig.savefig(path, bbox_inches='tight')
   fig.clf()


   ##################################################################################
   ##################################################################################
   # Trispectrum: mock N0 subtraction

   fig=plt.figure(0)
   gs = gridspec.GridSpec(5, 1)
   gs.update(hspace=0.1)

   # CIB
   ax0=plt.subplot(gs[0])
   # ClkCmb
   ax0.axhline(0., color='k')
   # statistical uncertainty on lensing amplitude
   sigma = np.sqrt(2. * (ClkCmb + NqCmb)**2 / Nmodes)
   sigma0 = np.array(list(zip(sigma/ClkCmb, sigma/ClkCmb))).flatten()
   l0 = np.array(zip(lEdges[:-1],lEdges[1:])).flatten()
   ax0.fill_between(l0, -sigma0, sigma0, edgecolor='k', facecolor='gray', alpha=0.4)
   # Dilation
   #ax0.plot(lCen,  CldCmbCib/ClkCmb, color=cD, ls='-')
   #Up = CldCmbCib + sCldCmbCib
   #Down = CldCmbCib - sCldCmbCib
   #ax0.fill_between(lCen, Down/ClkCmb, Up/ClkCmb, edgecolor='', facecolor=cD, alpha=0.6)
   # QE
   ax0.plot(lCen, ClqCmbCib/ClkCmb, color=cQ, ls='-')
   Up = ClqCmbCib + sClqCmbCib
   Down = ClqCmbCib - sClqCmbCib
   ax0.fill_between(lCen, Down/ClkCmb, Up/ClkCmb, edgecolor='', facecolor=cQ, alpha=0.6)
   # Bias hardened
   ax0.plot(lCen, ClBHCmbCib/ClkCmb, color=cBH, ls='-')
   Up = ClBHCmbCib + sClBHCmbCib
   Down = ClBHCmbCib - sClBHCmbCib
   ax0.fill_between(lCen, Down/ClkCmb, Up/ClkCmb, edgecolor='', facecolor=cBH, alpha=0.6)
   # Bias hardened with Gaussian profile
   ax0.plot(lCen, ClBHgCmbCib/ClkCmb, color=cBHg, ls='-')
   Up = ClBHgCmbCib + sClBHgCmbCib
   Down = ClBHgCmbCib - sClBHgCmbCib
   ax0.fill_between(lCen, Down/ClkCmb, Up/ClkCmb, edgecolor='', facecolor=cBHg, alpha=0.6)
   # Shear
   ax0.plot(lCen, ClsCmbCib/ClkCmb, color=cS, ls='-')
   Up = ClsCmbCib + sClsCmbCib
   Down = ClsCmbCib - sClsCmbCib
   ax0.fill_between(lCen, Down/ClkCmb, Up/ClkCmb, edgecolor='', facecolor=cS, alpha=0.6)
   #
#   ax0.set_ylim((-0.2,0.1))
   ax0.set_ylim((-0.05, 0.15))
   ax0.set_ylabel(r'CIB')
   #
   ax0.plot([], [], color=cS, ls='-', label=r'Shear')
   ax0.plot([], [], color=cBH, ls='-', label=r'BH1')
   ax0.plot([], [], color=cBHg, ls='-', label=r'BH2')
   ax0.plot([], [], color=cQ, ls='-', label=r'QE')
   #ax0.plot([], [], color=cD, ls='-', label='Mag')
   #
   plt.setp(ax0.get_xticklabels(), visible=False)
   ax0.legend(loc='upper center', ncol=4, mode=None, handlelength=.5, fontsize='x-small', borderpad=0.1, borderaxespad=0.1, frameon=False)
   ax0.set_title(r'Relative Bias on $C_L^{\kappa_\text{CMB}}$: Trispectrum')

   # tSZ
   ax1=plt.subplot(gs[1], sharex=ax0)
   # ClkCmb
   ax1.axhline(0., color='k')
   # statistical uncertainty on lensing amplitude
   sigma = np.sqrt(2. * (ClkCmb + NqCmb)**2 / Nmodes)
   sigma0 = np.array(list(zip(sigma/ClkCmb, sigma/ClkCmb))).flatten()
   l0 = np.array(zip(lEdges[:-1],lEdges[1:])).flatten()
   ax1.fill_between(l0, -sigma0, sigma0, edgecolor='k', facecolor='gray', alpha=0.4)
   # Dilation
   #ax1.plot(lCen, CldCmbTsz/ClkCmb, color=cD, ls='-', label=r'Dilation')
   #Up = CldCmbTsz + sCldCmbTsz
   #Down = CldCmbTsz - sCldCmbTsz
   #ax1.fill_between(lCen, Down/ClkCmb, Up/ClkCmb, edgecolor='', facecolor=cD, alpha=0.6)
   # QE
   ax1.plot(lCen, ClqCmbTsz/ClkCmb, color=cQ, ls='-', label=r'QE')
   Up = ClqCmbTsz + sClqCmbTsz
   Down = ClqCmbTsz - sClqCmbTsz
   ax1.fill_between(lCen, Down/ClkCmb, Up/ClkCmb, edgecolor='', facecolor=cQ, alpha=0.6)
   # Bias hardened
   ax1.plot(lCen, ClBHCmbTsz/ClkCmb, color=cBH, ls='-', label=r'BH1')
   Up = ClBHCmbTsz + sClBHCmbTsz
   Down = ClBHCmbTsz - sClBHCmbTsz
   ax1.fill_between(lCen, Down/ClkCmb, Up/ClkCmb, edgecolor='', facecolor=cBH, alpha=0.6)
   # Bias hardened with Gaussian profile
   ax1.plot(lCen, ClBHgCmbTsz/ClkCmb, color=cBHg, ls='-', label=r'BH2')
   Up = ClBHgCmbTsz + sClBHgCmbTsz
   Down = ClBHgCmbTsz - sClBHgCmbTsz
   ax1.fill_between(lCen, Down/ClkCmb, Up/ClkCmb, edgecolor='', facecolor=cBHg, alpha=0.6)
   # Shear
   ax1.plot(lCen, ClsCmbTsz/ClkCmb, color=cS, ls='-', label=r'S')
   Up = ClsCmbTsz + sClsCmbTsz
   Down = ClsCmbTsz - sClsCmbTsz
   ax1.fill_between(lCen, Down/ClkCmb, Up/ClkCmb, edgecolor='', facecolor=cS, alpha=0.6)
   #
   ax1.set_ylim((-0.05,0.15))
   ax1.set_ylabel(r'tSZ')
   #
   plt.setp(ax1.get_xticklabels(), visible=False)
   # remove last tick label for the second subplot
   yticks = ax1.yaxis.get_major_ticks()
   yticks[-1].label1.set_visible(False)

   # kSZ
   ax2=plt.subplot(gs[2], sharex=ax1)
   # ClkCmb
   ax2.axhline(0., color='k')
   # statistical uncertainty on lensing amplitude
   sigma = np.sqrt(2. * (ClkCmb + NqCmb)**2 / Nmodes)
   sigma0 = np.array(list(zip(sigma/ClkCmb, sigma/ClkCmb))).flatten()
   l0 = np.array(zip(lEdges[:-1],lEdges[1:])).flatten()
   ax2.fill_between(l0, -sigma0, sigma0, edgecolor='k', facecolor='gray', alpha=0.4)
   # Dilation
   #ax2.plot(lCen, CldCmbKsz/ClkCmb, color=cD, ls='-', label=r'Magnification')
   #Up = CldCmbKsz + sCldCmbKsz
   #Down = CldCmbKsz - sCldCmbKsz
   #ax2.fill_between(lCen,  Down/ClkCmb, Up/ClkCmb, edgecolor='', facecolor=cD, alpha=0.6)
   # QE
   ax2.plot(lCen,  ClqCmbKsz/ClkCmb, color=cQ, ls='-', label=r'QE')
   Up = ClqCmbKsz + sClqCmbKsz
   Down = ClqCmbKsz - sClqCmbKsz
   ax2.fill_between(lCen, Down/ClkCmb, Up/ClkCmb, edgecolor='', facecolor=cQ, alpha=0.6)
   # Bias hardened
   ax2.plot(lCen, ClBHCmbKsz/ClkCmb, color=cBH, ls='-', label=r'BH1')
   Up = ClBHCmbKsz + sClBHCmbKsz
   Down = ClBHCmbKsz - sClBHCmbKsz
   ax2.fill_between(lCen, Down/ClkCmb, Up/ClkCmb, edgecolor='', facecolor=cBH, alpha=0.6)
   # Bias hardened with Gaussian profile
   ax2.plot(lCen, ClBHgCmbKsz/ClkCmb, color=cBHg, ls='-', label=r'BH2')
   Up = ClBHgCmbKsz + sClBHgCmbKsz
   Down = ClBHgCmbKsz - sClBHgCmbKsz
   ax2.fill_between(lCen, Down/ClkCmb, Up/ClkCmb, edgecolor='', facecolor=cBHg, alpha=0.6)
   # Shear
   ax2.plot(lCen, ClsCmbKsz/ClkCmb, color=cS, ls='-', label=r'Shear')
   Up = ClsCmbKsz + sClsCmbKsz
   Down = ClsCmbKsz - sClsCmbKsz
   ax2.fill_between(lCen, Down/ClkCmb, Up/ClkCmb, edgecolor='', facecolor=cS, alpha=0.6)
   #
   ax2.set_ylim((-0.025,0.025))
   ax2.set_ylabel(r'kSZ')
   #
   plt.setp(ax2.get_xticklabels(), visible=False)
   # remove last tick label for the second subplot
   yticks = ax2.yaxis.get_major_ticks()
   yticks[-1].label1.set_visible(False)


   # Radio point sources
   ax3=plt.subplot(gs[3], sharex=ax2)
   # ClkCmb
   ax3.axhline(0., color='k')
   # statistical uncertainty on lensing amplitude
   sigma = np.sqrt(2. * (ClkCmb + NqCmb)**2 / Nmodes)
   sigma0 = np.array(list(zip(sigma/ClkCmb, sigma/ClkCmb))).flatten()
   l0 = np.array(zip(lEdges[:-1],lEdges[1:])).flatten()
   ax3.fill_between(l0, -sigma0, sigma0, edgecolor='k', facecolor='gray', alpha=0.4)
   # Dilation
   #ax3.plot(lCen, CldCmbRadiops/ClkCmb, color=cD, ls='-', label=r'Magnification')
   #Up = CldCmbRadiops + sCldCmbRadiops
   #Down = CldCmbRadiops - sCldCmbRadiops
   #ax3.fill_between(lCen, Down/ClkCmb, Up/ClkCmb, edgecolor='', facecolor=cD, alpha=0.6)
   # QE
   ax3.plot(lCen, ClqCmbRadiops/ClkCmb, color=cQ, ls='-', label=r'QE')
   Up = ClqCmbRadiops + sClqCmbRadiops
   Down = ClqCmbRadiops - sClqCmbRadiops
   ax3.fill_between(lCen, Down/ClkCmb, Up/ClkCmb, edgecolor='', facecolor=cQ, alpha=0.6)
   # Bias hardened
   ax3.plot(lCen, ClBHCmbRadiops/ClkCmb, color=cBH, ls='-', label=r'BH1')
   Up = ClBHCmbRadiops + sClBHCmbRadiops
   Down = ClBHCmbRadiops - sClBHCmbRadiops
   ax3.fill_between(lCen, Down/ClkCmb,  Up/ClkCmb, edgecolor='', facecolor=cBH, alpha=0.6)
   # Bias hardened with Gaussian profile
   ax3.plot(lCen, ClBHgCmbRadiops/ClkCmb, color=cBHg, ls='-', label=r'BH2')
   Up = ClBHgCmbRadiops + sClBHgCmbRadiops
   Down = ClBHgCmbRadiops - sClBHgCmbRadiops
   ax3.fill_between(lCen, Down/ClkCmb,  Up/ClkCmb, edgecolor='', facecolor=cBHg, alpha=0.6)
   # Shear
   ax3.plot(lCen, ClsCmbRadiops/ClkCmb, color=cS, ls='-', label=r'Shear')
   Up = ClsCmbRadiops + sClsCmbRadiops
   Down = ClsCmbRadiops - sClsCmbRadiops
   ax3.fill_between(lCen, Down/ClkCmb, Up/ClkCmb, edgecolor='', facecolor=cS, alpha=0.6)
   #
   ax3.set_ylim((-0.025,0.025))
   ax3.set_ylabel(r'Radio')
   #
   plt.setp(ax3.get_xticklabels(), visible=False)
   # remove last tick label for the second subplot
   yticks = ax3.yaxis.get_major_ticks()
   yticks[-1].label1.set_visible(False)


   # all extragalactic foregrounds
   ax4=plt.subplot(gs[4], sharex=ax3)
   # ClkCmb
   ax4.axhline(0., color='k')
   # statistical uncertainty on lensing amplitude
   sigma = np.sqrt(2. * (ClkCmb + NqCmb)**2 / Nmodes)
   sigma0 = np.array(list(zip(sigma/ClkCmb, sigma/ClkCmb))).flatten()
   l0 = np.array(zip(lEdges[:-1],lEdges[1:])).flatten()
   ax4.fill_between(l0, -sigma0, sigma0, edgecolor='k', facecolor='gray', alpha=0.4)
   # Dilation
   #ax2.plot(lCen, CldCmbAll/ClkCmb, color=cD, ls='-', label=r'Magnification')
   #Up = CldCmbAll + sCldCmbAll
   #Down = CldCmbAll - sCldCmbAll
   #ax2.fill_between(lCen, Down/ClkCmb, Up/ClkCmb, edgecolor='', facecolor=cD, alpha=0.6)
   # QE
   ax4.plot(lCen, ClqCmbAll/ClkCmb, color=cQ, ls='-', label=r'QE')
   Up = ClqCmbAll + sClqCmbAll
   Down = ClqCmbAll - sClqCmbAll
   ax4.fill_between(lCen, Down/ClkCmb, Up/ClkCmb, edgecolor='', facecolor=cQ, alpha=0.6)
   # Bias hardened
   ax4.plot(lCen, ClBHCmbAll/ClkCmb, color=cBH, ls='-', label=r'BH1')
   Up = ClBHCmbAll + sClBHCmbAll
   Down = ClBHCmbAll - sClBHCmbAll
   ax4.fill_between(lCen, Down/ClkCmb, Up/ClkCmb, edgecolor='', facecolor=cBH, alpha=0.6)
   # Bias hardened with Gaussian profile
   ax4.plot(lCen, ClBHgCmbAll/ClkCmb, color=cBHg, ls='-', label=r'BH2')
   Up = ClBHgCmbAll + sClBHgCmbAll
   Down = ClBHgCmbAll - sClBHgCmbAll
   ax4.fill_between(lCen, Down/ClkCmb, Up/ClkCmb, edgecolor='', facecolor=cBHg, alpha=0.6)
   # Shear
   ax4.plot(lCen, ClsCmbAll/ClkCmb, color=cS, ls='-', label=r'Shear')
   Up = ClsCmbAll + sClsCmbAll
   Down = ClsCmbAll - sClsCmbAll
   ax4.fill_between(lCen, Down/ClkCmb, Up/ClkCmb, edgecolor='', facecolor=cS, alpha=0.6)
   #
   ax4.set_ylim((-0.05,0.15))
   ax4.set_ylabel(r'All')
   #
   # remove last tick label for the second subplot
   yticks = ax4.yaxis.get_major_ticks()
   yticks[-1].label1.set_visible(False)
   
   
   ax4.set_xscale('log', nonposx='clip')
   ax4.set_xlim((30., 1.5e3))
   ax4.set_xlabel(r'$L$')

   path = "output/trispectrum_lmax_"+str(int(lMax))+".pdf"
   fig.savefig(path, bbox_inches='tight')
   fig.clf()


   ##################################################################################
   ##################################################################################
   # Cross-correlation with LSST gold

   fig=plt.figure(0)
   gs = gridspec.GridSpec(5, 1, height_ratios = [2.2,1.8,1.2,1.2, 2.5])
   gs.update(hspace=0.1)

   # CIB
   ax0=plt.subplot(gs[0])
   # ClkCmb
   ax0.axhline(0., color='k')
   # statistical uncertainty on lensing amplitude
   Ckg = p2d_lsstgoldcmblens.fPtotinterp(lCen)
   Cgg = p2d_lsstgold.fPtotinterp(lCen)
   sigma = np.sqrt(((ClkCmb + NqCmb)*Cgg + (Ckg)**2) / Nmodes)
   sigma0 = np.array(list(zip(sigma/Ckg, sigma/Ckg))).flatten()
   l0 = np.array(zip(lEdges[:-1],lEdges[1:])).flatten()
   ax0.fill_between(l0, -sigma0, sigma0, edgecolor='k', facecolor='gray', alpha=0.4)
   # Dilation
   #ax0.plot(lCen, CldCmbCibLsstgold/Ckg, color=cD, ls='-')
   #Up = CldCmbCibLsstgold + sCldCmbCibLsstgold
   #Down = CldCmbCibLsstgold - sCldCmbCibLsstgold
   #ax0.fill_between(lCen, Down/Ckg, Up/Ckg, edgecolor='', facecolor=cD, alpha=0.6)
   # QE
   ax0.plot(lCen, ClqCmbCibLsstgold/Ckg, color=cQ, ls='-')
   Up = ClqCmbCibLsstgold + sClqCmbCibLsstgold
   Down = ClqCmbCibLsstgold - sClqCmbCibLsstgold
   ax0.fill_between(lCen, Down/Ckg, Up/Ckg, edgecolor='', facecolor=cQ, alpha=0.6)
   # Bias hardened
   ax0.plot(lCen, ClBHCmbCibLsstgold/Ckg, color=cBH, ls='-')
   Up = ClBHCmbCibLsstgold + sClBHCmbCibLsstgold
   Down = ClBHCmbCibLsstgold - sClBHCmbCibLsstgold
   ax0.fill_between(lCen, Down/Ckg, Up/Ckg, edgecolor='', facecolor=cBH, alpha=0.6)
   # Bias hardened with Gaussian profile
   ax0.plot(lCen, ClBHgCmbCibLsstgold/Ckg, color=cBHg, ls='-')
   Up = ClBHgCmbCibLsstgold + sClBHgCmbCibLsstgold
   Down = ClBHgCmbCibLsstgold - sClBHgCmbCibLsstgold
   ax0.fill_between(lCen, Down/Ckg, Up/Ckg, edgecolor='', facecolor=cBHg, alpha=0.6)
   # Shear
   ax0.plot(lCen, ClsCmbCibLsstgold/Ckg, color=cS, ls='-')
   Up = ClsCmbCibLsstgold + sClsCmbCibLsstgold
   Down = ClsCmbCibLsstgold - sClsCmbCibLsstgold
   ax0.fill_between(lCen, Down/Ckg, Up/Ckg, edgecolor='', facecolor=cS, alpha=0.6)
   #
   ax0.plot([], [], color=cS, ls='-', label=r'Shear')
   ax0.plot([], [], color=cBH, ls='-', label=r'BH1')
   ax0.plot([], [], color=cBHg, ls='-', label=r'BH2')
   ax0.plot([], [], color=cQ, ls='-', label=r'QE')
   #ax0.plot([], [], color=cD, ls='-', label='Mag.')
   #
#   ax0.set_ylim((-0.25, 0.15))
   ax0.set_ylim((-0.12, 0.05))
   ax0.set_ylabel(r'CIB')
   #
   plt.setp(ax0.get_xticklabels(), visible=False)
   ax0.legend(loc='upper center', ncol=4, mode=None, handlelength=.5, fontsize='x-small', borderpad=0.1, borderaxespad=0.1, frameon=False)
   ax0.set_title(r'Relative Bias on $C_L^{\kappa_\text{CMB} \times \text{LSST} }$')

   # tSZ
   ax1=plt.subplot(gs[1], sharex=ax0)
   # ClkCmb
   ax1.axhline(0., color='k')
   # statistical uncertainty on lensing amplitude
   Ckg = p2d_lsstgoldcmblens.fPtotinterp(lCen)
   Cgg = p2d_lsstgold.fPtotinterp(lCen)
   sigma = np.sqrt(((ClkCmb + NqCmb)*Cgg + (Ckg)**2) / Nmodes)
   sigma0 = np.array(list(zip(sigma/Ckg, sigma/Ckg))).flatten()
   l0 = np.array(zip(lEdges[:-1],lEdges[1:])).flatten()
   ax1.fill_between(l0, -sigma0, sigma0, edgecolor='k', facecolor='gray', alpha=0.4)
   # Dilation
   #ax1.plot(lCen, CldCmbTszLsstgold/Ckg, color=cD, ls='-', label=r'Dilation')
   #Up = CldCmbTszLsstgold + sCldCmbTszLsstgold
   #Down = CldCmbTszLsstgold - sCldCmbTszLsstgold
   #ax1.fill_between(lCen, Down/Ckg, Up/Ckg, edgecolor='', facecolor=cD, alpha=0.6)
   # QE
   ax1.plot(lCen, ClqCmbTszLsstgold/Ckg, color=cQ, ls='-', label=r'QE')
   Up = ClqCmbTszLsstgold + sClqCmbTszLsstgold
   Down = ClqCmbTszLsstgold - sClqCmbTszLsstgold
   ax1.fill_between(lCen, Down/Ckg, Up/Ckg, edgecolor='', facecolor=cQ, alpha=0.6)
   # Bias hardened
   ax1.plot(lCen, ClBHCmbTszLsstgold/Ckg, color=cBH, ls='-', label=r'BH1')
   Up = ClBHCmbTszLsstgold + sClBHCmbTszLsstgold
   Down = ClBHCmbTszLsstgold - sClBHCmbTszLsstgold
   ax1.fill_between(lCen, Down/Ckg, Up/Ckg, edgecolor='', facecolor=cBH, alpha=0.6)
   # Bias hardened with Gaussian profile
   ax1.plot(lCen, ClBHgCmbTszLsstgold/Ckg, color=cBHg, ls='-', label=r'BH2')
   Up = ClBHgCmbTszLsstgold + sClBHgCmbTszLsstgold
   Down = ClBHgCmbTszLsstgold - sClBHgCmbTszLsstgold
   ax1.fill_between(lCen, Down/Ckg, Up/Ckg, edgecolor='', facecolor=cBHg, alpha=0.6)
   # Shear
   ax1.plot(lCen, ClsCmbTszLsstgold/Ckg, color=cS, ls='-', label=r'S')
   Up = ClsCmbTszLsstgold + sClsCmbTszLsstgold
   Down = ClsCmbTszLsstgold - sClsCmbTszLsstgold
   ax1.fill_between(lCen, Down/Ckg, Up/Ckg, edgecolor='', facecolor=cS, alpha=0.6)
   #
   ax1.set_ylim((-0.07, 0.02))
   ax1.set_ylabel(r'tSZ')
   #
   plt.setp(ax1.get_xticklabels(), visible=False)
   # remove last tick label for the second subplot
   yticks = ax1.yaxis.get_major_ticks()
   yticks[-1].label1.set_visible(False)

   # kSZ
   ax2=plt.subplot(gs[2], sharex=ax1)
   # ClkCmb
   ax2.axhline(0., color='k')
   # statistical uncertainty on lensing amplitude
   Ckg = p2d_lsstgoldcmblens.fPtotinterp(lCen)
   Cgg = p2d_lsstgold.fPtotinterp(lCen)
   sigma = np.sqrt(((ClkCmb + NqCmb)*Cgg + (Ckg)**2) / Nmodes)
   sigma0 = np.array(list(zip(sigma/Ckg, sigma/Ckg))).flatten()
   l0 = np.array(zip(lEdges[:-1],lEdges[1:])).flatten()
   ax2.fill_between(l0, -sigma0, sigma0, edgecolor='k', facecolor='gray', alpha=0.4)
   # Dilation
   #ax2.plot(lCen, CldCmbKszLsstgold/Ckg, color=cD, ls='-', label=r'Magnification')
   #Up = CldCmbKszLsstgold + sCldCmbKszLsstgold
   #Down = CldCmbKszLsstgold - sCldCmbKszLsstgold
   #ax2.fill_between(lCen, Down/Ckg, Up/Ckg, edgecolor='', facecolor=cD, alpha=0.6)
   # QE
   ax2.plot(lCen, ClqCmbKszLsstgold/Ckg, color=cQ, ls='-', label=r'QE')
   Up = ClqCmbKszLsstgold + sClqCmbKszLsstgold
   Down = ClqCmbKszLsstgold - sClqCmbKszLsstgold
   ax2.fill_between(lCen, Down/Ckg, Up/Ckg, edgecolor='', facecolor=cQ, alpha=0.6)
   # Bias hardened
   ax2.plot(lCen, ClBHCmbKszLsstgold/Ckg, color=cBH, ls='-', label=r'BH1')
   Up = ClBHCmbKszLsstgold + sClBHCmbKszLsstgold
   Down = ClBHCmbKszLsstgold - sClBHCmbKszLsstgold
   ax2.fill_between(lCen, Down/Ckg, Up/Ckg, edgecolor='', facecolor=cBH, alpha=0.6)
   # Bias hardened with Gaussian profile
   ax2.plot(lCen, ClBHgCmbKszLsstgold/Ckg, color=cBHg, ls='-', label=r'BH2')
   Up = ClBHgCmbKszLsstgold + sClBHgCmbKszLsstgold
   Down = ClBHgCmbKszLsstgold - sClBHgCmbKszLsstgold
   ax2.fill_between(lCen, Down/Ckg, Up/Ckg, edgecolor='', facecolor=cBHg, alpha=0.6)
   # Shear
   ax2.plot(lCen, ClsCmbKszLsstgold/Ckg, color=cS, ls='-', label=r'Shear')
   Up = ClsCmbKszLsstgold + sClsCmbKszLsstgold
   Down = ClsCmbKszLsstgold - sClsCmbKszLsstgold
   ax2.fill_between(lCen, Down/Ckg, Up/Ckg, edgecolor='', facecolor=cS, alpha=0.6)
   #
   ax2.set_ylim((-0.01, 0.01))
   ax2.set_ylabel(r'kSZ')
   #
   plt.setp(ax2.get_xticklabels(), visible=False)
   # remove last tick label for the second subplot
   yticks = ax2.yaxis.get_major_ticks()
   yticks[-1].label1.set_visible(False)


   # Radio point sources
   ax3=plt.subplot(gs[3], sharex=ax2)
   # ClkCmb
   ax3.axhline(0., color='k')
   # statistical uncertainty on lensing amplitude
   Ckg = p2d_lsstgoldcmblens.fPtotinterp(lCen)
   Cgg = p2d_lsstgold.fPtotinterp(lCen)
   sigma = np.sqrt(((ClkCmb + NqCmb)*Cgg + (Ckg)**2) / Nmodes)
   sigma0 = np.array(list(zip(sigma/Ckg, sigma/Ckg))).flatten()
   l0 = np.array(zip(lEdges[:-1],lEdges[1:])).flatten()
   ax3.fill_between(l0, -sigma0, sigma0, edgecolor='k', facecolor='gray', alpha=0.4)
   # Dilation
   #ax3.plot(lCen, CldCmbRadiopsLsstgold/Ckg, color=cD, ls='-', label=r'Magnification')
   #Up = CldCmbRadiopsLsstgold + sCldCmbRadiopsLsstgold
   #Down = CldCmbRadiopsLsstgold - sCldCmbRadiopsLsstgold
   #ax3.fill_between(lCen, Down/Ckg, Up/Ckg, edgecolor='', facecolor=cD, alpha=0.6)
   # QE
   ax3.plot(lCen, ClqCmbRadiopsLsstgold/Ckg, color=cQ, ls='-', label=r'QE')
   Up = ClqCmbRadiopsLsstgold + sClqCmbRadiopsLsstgold
   Down = ClqCmbRadiopsLsstgold - sClqCmbRadiopsLsstgold
   ax3.fill_between(lCen, Down/Ckg, Up/Ckg, edgecolor='', facecolor=cQ, alpha=0.6)
   # Bias hardened
   ax3.plot(lCen, ClBHCmbRadiopsLsstgold/Ckg, color=cBH, ls='-', label=r'BH1')
   Up = ClBHCmbRadiopsLsstgold + sClBHCmbRadiopsLsstgold
   Down = ClBHCmbRadiopsLsstgold - sClBHCmbRadiopsLsstgold
   ax3.fill_between(lCen, Down/Ckg, Up/Ckg, edgecolor='', facecolor=cBH, alpha=0.6)
   # Bias hardened with Gaussian profile
   ax3.plot(lCen, ClBHgCmbRadiopsLsstgold/Ckg, color=cBHg, ls='-', label=r'BH2')
   Up = ClBHgCmbRadiopsLsstgold + sClBHgCmbRadiopsLsstgold
   Down = ClBHgCmbRadiopsLsstgold - sClBHgCmbRadiopsLsstgold
   ax3.fill_between(lCen, Down/Ckg, Up/Ckg, edgecolor='', facecolor=cBHg, alpha=0.6)
   # Shear
   ax3.plot(lCen, ClsCmbRadiopsLsstgold/Ckg, color=cS, ls='-', label=r'Shear')
   Up = ClsCmbRadiopsLsstgold + sClsCmbRadiopsLsstgold
   Down = ClsCmbRadiopsLsstgold - sClsCmbRadiopsLsstgold
   ax3.fill_between(lCen, Down/Ckg, Up/Ckg, edgecolor='', facecolor=cS, alpha=0.6)
   #
   ax3.set_ylim((-0.01, 0.01))
   ax3.set_ylabel(r'Radio')
   #
   plt.setp(ax3.get_xticklabels(), visible=False)
   # remove last tick label for the second subplot
   yticks = ax3.yaxis.get_major_ticks()
   yticks[-1].label1.set_visible(False)


   # sum
   #ax4=plt.subplot(gs[4], sharex=ax3)
   # ClkCmb
   #ax4.axhline(0., color='k')
   # statistical uncertainty on lensing amplitude
   #sigma = np.sqrt(((ClkCmb + NqCmb)*Cgg + (Ckg)**2) / Nmodes)
   #sigma0 = np.array(list(zip(sigma/Ckg, sigma/Ckg))).flatten()
   #l0 = np.array(zip(lEdges[:-1],lEdges[1:])).flatten()
   #ax4.fill_between(l0, -sigma0, sigma0, edgecolor='k', facecolor='gray', alpha=0.4)
   # QE
   #ax4.plot(lCen, ClqCmbSumLsstgold/Ckg, color=cQ, ls='-', label=r'QE')
   #Up = ClqCmbSumLsstgold + sClqCmbSumLsstgold
   #Down = ClqCmbSumLsstgold - sClqCmbSumLsstgold
   #ax4.fill_between(lCen, Down/Ckg, Up/Ckg, edgecolor='', facecolor=cQ, alpha=0.6)
   # Bias hardened
   #ax4.plot(lCen, ClBHCmbSumLsstgold/Ckg, color=cBH, ls='-', label=r'BH1')
   #Up = ClBHCmbSumLsstgold + sClBHCmbSumLsstgold
   #Down = ClBHCmbSumLsstgold - sClBHCmbSumLsstgold
   #ax4.fill_between(lCen, Down/Ckg, Up/Ckg, edgecolor='', facecolor=cBH, alpha=0.6)
   # Shear
   #ax4.plot(lCen, ClsCmbSumLsstgold/Ckg, color=cS, ls='-', label=r'Shear')
   #Up = ClsCmbSumLsstgold + sClsCmbSumLsstgold
   #Down = ClsCmbSumLsstgold - sClsCmbSumLsstgold
   #ax4.fill_between(lCen, Down/Ckg, Up/Ckg, edgecolor='', facecolor=cS, alpha=0.6)
   #
   #ax4.set_ylim((-0.4, 0.1))
   #ax4.set_ylabel(r'Sum')
   #
   #plt.setp(ax4.get_xticklabels(), visible=False)
   ## remove last tick label for the second subplot
   #yticks = ax4.yaxis.get_major_ticks()
   #yticks[-1].label1.set_visible(False)


   # all extragalactic foregrounds
   ax4=plt.subplot(gs[4], sharex=ax3)
   # ClkCmb
   ax4.axhline(0., color='k')
   # statistical uncertainty on lensing amplitude
   sigma = np.sqrt(((ClkCmb + NqCmb)*Cgg + (Ckg)**2) / Nmodes)
   sigma0 = np.array(list(zip(sigma/Ckg, sigma/Ckg))).flatten()
   l0 = np.array(zip(lEdges[:-1],lEdges[1:])).flatten()
   ax4.fill_between(l0, -sigma0, sigma0, edgecolor='k', facecolor='gray', alpha=0.4)
   # QE
   ax4.plot(lCen, ClqCmbAllLsstgold/Ckg, color=cQ, ls='-', label=r'QE')
   Up = ClqCmbAllLsstgold + sClqCmbAllLsstgold
   Down = ClqCmbAllLsstgold - sClqCmbAllLsstgold
   ax4.fill_between(lCen, Down/Ckg, Up/Ckg, edgecolor='', facecolor=cQ, alpha=0.6)
   # Bias hardened
   ax4.plot(lCen, ClBHCmbAllLsstgold/Ckg, color=cBH, ls='-', label=r'BH1')
   Up = ClBHCmbAllLsstgold + sClBHCmbAllLsstgold
   Down = ClBHCmbAllLsstgold - sClBHCmbAllLsstgold
   ax4.fill_between(lCen, Down/Ckg, Up/Ckg, edgecolor='', facecolor=cBH, alpha=0.6)
   # Bias hardened
   ax4.plot(lCen, ClBHgCmbAllLsstgold/Ckg, color=cBHg, ls='-', label=r'BH2')
   Up = ClBHgCmbAllLsstgold + sClBHgCmbAllLsstgold
   Down = ClBHgCmbAllLsstgold - sClBHgCmbAllLsstgold
   ax4.fill_between(lCen, Down/Ckg, Up/Ckg, edgecolor='', facecolor=cBHg, alpha=0.6)
   # Shear
   ax4.plot(lCen, ClsCmbAllLsstgold/Ckg, color=cS, ls='-', label=r'Shear')
   Up = ClsCmbAllLsstgold + sClsCmbAllLsstgold
   Down = ClsCmbAllLsstgold - sClsCmbAllLsstgold
   ax4.fill_between(lCen, Down/Ckg, Up/Ckg, edgecolor='', facecolor=cS, alpha=0.6)
   #
   ax4.set_ylim((-0.15, 0.05))
   ax4.set_ylabel(r'All')
   #
   # remove last tick label for the second subplot
   yticks = ax4.yaxis.get_major_ticks()
   yticks[-1].label1.set_visible(False)

   ax4.set_xscale('log', nonposx='clip')
   ax4.set_xlim((30., 1.5e3))
   ax4.set_xlabel(r'$L$')

   path = "output/crossLSST_lmax_"+str(int(lMax))+".pdf"
   plt.show()
   #fig.savefig(path, bbox_inches='tight')
   fig.clf()


   
##################################################################################
# Do some averaging 
nPatch = 81
outputFolder = "bias_output"
datalist = [np.genfromtxt(outputFolder+"/data_lmaxT_"+str(int(lMax))+"_"+str(i)+".txt") for i in range(nPatch)]
datalist = np.array(datalist)
dataAvg = np.mean(datalist,axis=0)
dataStd = np.std(datalist,axis=0) * uncertaintyFactor /np.sqrt(nPatch)


##################################################################################
# Plot biases
path = outputFolder+"/lCen_lmaxT_"+str(int(lMax))+".txt"
lCen = np.genfromtxt(path)
plot_data(lCen, dataAvg, dataStd)


##################################################################################
##################################################################################
# Calculate SNR and Full = Prim + Sec + trispectrum

LMin = 20.
LMax = 1.e3

llMax = np.array([2000.,2500.,3000.,3500., 4000.])

snrQCmb = np.zeros(len(llMax))
snrSCmb = np.zeros(len(llMax))
snrBHCmb = np.zeros(len(llMax))
snrBHgCmb = np.zeros(len(llMax))
sHybridCmb = np.zeros(len(llMax))
#
sQCmbLsstgold = np.zeros(len(llMax))
sSCmbLsstgold = np.zeros(len(llMax))
sBHCmbLsstgold = np.zeros(len(llMax))
sBHgCmbLsstgold = np.zeros(len(llMax))
sHybridCmbLsstgold = np.zeros(len(llMax))
snrGCCmb = np.zeros(len(llMax))
snrGCCmbLsstgold = np.zeros(len(llMax))

bSCmbCibLsstgold = np.zeros((len(llMax), nPatch))
#
bSCmbTszLsstgold = np.zeros((len(llMax), nPatch))
#
bSCmbKszLsstgold = np.zeros((len(llMax), nPatch))
#
bSCmbRadiopsLsstgold = np.zeros((len(llMax), nPatch))
#
bSCmbAllLsstgold = np.zeros((len(llMax), nPatch))
#
bQCmbCibLsstgold = np.zeros((len(llMax), nPatch))
#
bQCmbTszLsstgold = np.zeros((len(llMax), nPatch))
#
bQCmbKszLsstgold = np.zeros((len(llMax), nPatch))
#
bQCmbRadiopsLsstgold = np.zeros((len(llMax), nPatch))
#
bQCmbAllLsstgold = np.zeros((len(llMax), nPatch))
#
bBHCmbCibLsstgold = np.zeros((len(llMax), nPatch))
#
bBHCmbTszLsstgold = np.zeros((len(llMax), nPatch))
#
bBHCmbKszLsstgold = np.zeros((len(llMax), nPatch))
#
bBHCmbRadiopsLsstgold = np.zeros((len(llMax), nPatch))
#
bBHCmbAllLsstgold = np.zeros((len(llMax), nPatch))
#
bBHgCmbCibLsstgold = np.zeros((len(llMax), nPatch))
#
bBHgCmbTszLsstgold = np.zeros((len(llMax), nPatch))
#
bBHgCmbKszLsstgold = np.zeros((len(llMax), nPatch))
#
bBHgCmbRadiopsLsstgold = np.zeros((len(llMax), nPatch))
#
bBHgCmbAllLsstgold = np.zeros((len(llMax), nPatch))
#
bSCmbCibFull = np.zeros((len(llMax), nPatch))
#
bSCmbTszFull = np.zeros((len(llMax), nPatch))
#
bSCmbKszFull = np.zeros((len(llMax), nPatch))
#
bSCmbRadiopsFull = np.zeros((len(llMax), nPatch))
#
bSCmbAllFull = np.zeros((len(llMax), nPatch))
#
bQCmbCibFull = np.zeros((len(llMax), nPatch))
#
bQCmbTszFull = np.zeros((len(llMax), nPatch))
#
bQCmbKszFull = np.zeros((len(llMax), nPatch))
#
bQCmbRadiopsFull = np.zeros((len(llMax), nPatch))
#
bQCmbAllFull = np.zeros((len(llMax), nPatch))
#
bBHCmbCibFull = np.zeros((len(llMax), nPatch))
#
bBHCmbTszFull = np.zeros((len(llMax), nPatch))
#
bBHCmbKszFull = np.zeros((len(llMax), nPatch))
#
bBHCmbRadiopsFull = np.zeros((len(llMax), nPatch))
#
bBHCmbAllFull = np.zeros((len(llMax), nPatch))
#
bBHgCmbCibFull = np.zeros((len(llMax), nPatch))
#
bBHgCmbTszFull = np.zeros((len(llMax), nPatch))
#
bBHgCmbKszFull = np.zeros((len(llMax), nPatch))
#
bBHgCmbRadiopsFull = np.zeros((len(llMax), nPatch))
#
bBHgCmbAllFull = np.zeros((len(llMax), nPatch))
#
bHybridCmbAllFull = np.zeros((len(llMax), nPatch))
#
bHybridCmbAllLsstgold = np.zeros((len(llMax), nPatch))


for ilMax in range(len(llMax)):

   lMax = llMax[ilMax]

   # Get data
   nPatch = 81
   outputFolder = "bias_output"
   datalist = [np.genfromtxt(outputFolder+"/data_lmaxT_"+str(int(lMax))+"_"+str(i)+".txt") for i in  range(nPatch)]
   datalist = np.array(datalist)

   cmb = CMB(beam=1.4, noise=6., lMin=lMin, lMaxT=lMax, lMaxP=lMax, atm=False, fg=True)
   L = np.logspace(np.log10(lMin/2.), np.log10(2.*lMax), 1001, 10.)
   F = np.array(list(map(cmb.ftotalTT, L)))
   cmb.fCtotal = interp1d(L, F, kind='linear', bounds_error=False, fill_value=0.)

   fNqCmb_fft = baseMap.forecastN0Kappa(cmb.flensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False)
   fNsCmb_fft = baseMap.forecastN0KappaShear(cmb.flensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, corr=True, test=False)
   fNqBHCmb_fft = baseMap.forecastN0KappaBH(cmb.flensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False)
   fNqBHgCmb_fft = baseMap.forecastN0KappaBH2(cmb.flensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False, cache=None, sigma=clusterRadius)

   path = outputFolder+"/lCen_lmaxT_"+str(int(lMax))+".txt"
   lCen = np.genfromtxt(path)
   #lRange = (1., 2.*lMax)
   #lEdges = np.logspace(np.log10(lRange[0]), np.log10(lRange[-1]), nBins, 10.)
   ell = baseMap.l.flatten()
   lEdges = np.logspace(np.log10(1.), np.log10(np.max(ell)), nBins, 10.)
   Nmodes = lEdges[1:]**2. - lEdges[:-1]**2.   
   I = np.where((lCen>=LMin)*(lCen<=LMax))

   ClkCmb = p2d_cmblens.fPinterp(lCen)
   NqCmb = fNqCmb_fft(lCen)  
   NsCmb = fNsCmb_fft(lCen)
   NbhCmb = fNqBHCmb_fft(lCen)
   NbhgCmb = fNqBHgCmb_fft(lCen)
   Ckg = p2d_lsstgoldcmblens.fPtotinterp(lCen)
   Cgg = p2d_lsstgold.fPtotinterp(lCen)

   ##################################################################################
   # Spectra info

   i=0
   # Radio point sources
   #
   # Trispectrum
   ClqCmbRadiops = datalist[:,int(0+20*i)]
   ClsCmbRadiops = datalist[:,int(1+20*i)]
   CldCmbRadiops = datalist[:,int(2+20*i)]
   ClBHCmbRadiops = datalist[:,int(3+20*i)]
   ClBHgCmbRadiops = datalist[:,int(4+20*i)]
   # Primary
   ClqCmbRadiopsKappa = datalist[:,int(5+20*i)]
   ClsCmbRadiopsKappa = datalist[:,int(6+20*i)]
   CldCmbRadiopsKappa = datalist[:,int(7+20*i)]
   ClBHCmbRadiopsKappa = datalist[:,int(8+20*i)]
   ClBHgCmbRadiopsKappa = datalist[:,int(9+20*i)]
   # k_rec x LSST
   ClqCmbRadiopsLsstgold = datalist[:,int(10+20*i)]
   ClsCmbRadiopsLsstgold = datalist[:,int(11+20*i)]
   CldCmbRadiopsLsstgold = datalist[:,int(12+20*i)]
   ClBHCmbRadiopsLsstgold = datalist[:,int(13+20*i)]
   ClBHgCmbRadiopsLsstgold = datalist[:,int(14+20*i)]
   # Secondary
   ClqCmbRadiopsSec = datalist[:,int(15+20*i)]
   CldCmbRadiopsSec = datalist[:,int(16+20*i)]
   ClsCmbRadiopsSec = datalist[:,int(17+20*i)]
   ClBHCmbRadiopsSec = datalist[:,int(18+20*i)]
   ClBHgCmbRadiopsSec = datalist[:,int(19+20*i)]

   i=1
   # CIB
   #
   # Trispectrum
   ClqCmbCib = datalist[:,int(0+20*i)]
   ClsCmbCib = datalist[:,int(1+20*i)]
   CldCmbCib = datalist[:,int(2+20*i)]
   ClBHCmbCib = datalist[:,int(3+20*i)]
   ClBHgCmbCib = datalist[:,int(4+20*i)]
   # Primary
   ClqCmbCibKappa = datalist[:,int(5+20*i)]
   ClsCmbCibKappa = datalist[:,int(6+20*i)]
   CldCmbCibKappa = datalist[:,int(7+20*i)]
   ClBHCmbCibKappa = datalist[:,int(8+20*i)]
   ClBHgCmbCibKappa = datalist[:,int(9+20*i)]
   # k_rec x LSST
   ClqCmbCibLsstgold = datalist[:,int(10+20*i)]
   ClsCmbCibLsstgold = datalist[:,int(11+20*i)]
   CldCmbCibLsstgold = datalist[:,int(12+20*i)]
   ClBHCmbCibLsstgold = datalist[:,int(13+20*i)]
   ClBHgCmbCibLsstgold = datalist[:,int(14+20*i)]
   # Secondary
   ClqCmbCibSec = datalist[:,int(15+20*i)]
   CldCmbCibSec = datalist[:,int(16+20*i)]
   ClsCmbCibSec = datalist[:,int(17+20*i)]
   ClBHCmbCibSec = datalist[:,int(18+20*i)]
   ClBHgCmbCibSec = datalist[:,int(19+20*i)]

   i=2
   # tSZ
   #
   # Trispectrum
   ClqCmbTsz = datalist[:,int(0+20*i)]
   ClsCmbTsz = datalist[:,int(1+20*i)]
   CldCmbTsz = datalist[:,int(2+20*i)]
   ClBHCmbTsz = datalist[:,int(3+20*i)]
   ClBHgCmbTsz = datalist[:,int(4+20*i)]
   # Primary
   ClqCmbTszKappa = datalist[:,int(5+20*i)]
   ClsCmbTszKappa = datalist[:,int(6+20*i)]
   CldCmbTszKappa = datalist[:,int(7+20*i)]
   ClBHCmbTszKappa = datalist[:,int(8+20*i)]
   ClBHgCmbTszKappa = datalist[:,int(9+20*i)]
   # k_rec x LSST
   ClqCmbTszLsstgold = datalist[:,int(10+20*i)]
   ClsCmbTszLsstgold = datalist[:,int(11+20*i)]
   CldCmbTszLsstgold = datalist[:,int(12+20*i)]
   ClBHCmbTszLsstgold = datalist[:,int(13+20*i)]
   ClBHgCmbTszLsstgold = datalist[:,int(14+20*i)]
   # Secondary
   ClqCmbTszSec = datalist[:,int(15+20*i)]
   CldCmbTszSec = datalist[:,int(16+20*i)]
   ClsCmbTszSec = datalist[:,int(17+20*i)]
   ClBHCmbTszSec = datalist[:,int(18+20*i)]
   ClBHgCmbTszSec = datalist[:,int(19+20*i)]

   i=3
   # kSZ
   # 
   # Trispectrum
   ClqCmbKsz = datalist[:,int(0+20*i)]
   ClsCmbKsz = datalist[:,int(1+20*i)]
   CldCmbKsz = datalist[:,int(2+20*i)]
   ClBHCmbKsz = datalist[:,int(3+20*i)]
   ClBHgCmbKsz = datalist[:,int(4+20*i)]
   # Primary
   ClqCmbKszKappa = datalist[:,int(5+20*i)]
   ClsCmbKszKappa = datalist[:,int(6+20*i)]
   CldCmbKszKappa = datalist[:,int(7+20*i)]
   ClBHCmbKszKappa = datalist[:,int(8+20*i)]
   ClBHgCmbKszKappa = datalist[:,int(9+20*i)]
   # k_rec x LSST 
   ClqCmbKszLsstgold = datalist[:,int(10+20*i)]
   ClsCmbKszLsstgold = datalist[:,int(11+20*i)]
   CldCmbKszLsstgold = datalist[:,int(12+20*i)]
   ClBHCmbKszLsstgold = datalist[:,int(13+20*i)]
   ClBHgCmbKszLsstgold = datalist[:,int(14+20*i)]
   # Secondary
   ClqCmbKszSec = datalist[:,int(15+20*i)]
   CldCmbKszSec = datalist[:,int(16+20*i)]
   ClsCmbKszSec = datalist[:,int(17+20*i)]
   ClBHCmbKszSec = datalist[:,int(18+20*i)]
   ClBHgCmbKszSec = datalist[:,int(19+20*i)]

   i=4
   # Dust
   # 
   # Trispectrum
   ClqCmbDust = datalist[:,int(0+20*i)]
   ClsCmbDust = datalist[:,int(1+20*i)]
   CldCmbDust = datalist[:,int(2+20*i)]
   ClBHCmbDust = datalist[:,int(3+20*i)]
   ClBHgCmbDust = datalist[:,int(4+20*i)]
   # Primary
   ClqCmbDustKappa = datalist[:,int(5+20*i)]
   ClsCmbDustKappa = datalist[:,int(6+20*i)]
   CldCmbDustKappa = datalist[:,int(7+20*i)]
   ClBHCmbDustKappa = datalist[:,int(8+20*i)]
   ClBHgCmbDustKappa = datalist[:,int(9+20*i)]
   # k_rec x LSST 
   ClqCmbDustLsstgold = datalist[:,int(10+20*i)]
   ClsCmbDustLsstgold = datalist[:,int(11+20*i)]
   CldCmbDustLsstgold = datalist[:,int(12+20*i)]
   ClBHCmbDustLsstgold = datalist[:,int(13+20*i)]
   ClBHgCmbDustLsstgold = datalist[:,int(14+20*i)]
   # Secondary
   ClqCmbDustSec = datalist[:,int(15+20*i)]
   CldCmbDustSec = datalist[:,int(16+20*i)]
   ClsCmbDustSec = datalist[:,int(17+20*i)]
   ClBHCmbDustSec = datalist[:,int(18+20*i)]
   ClBHgCmbDustSec = datalist[:,int(19+20*i)]

   i=5
   # FreeFree
   # 
   # Trispectrum
   ClqCmbFreeFree = datalist[:,int(0+20*i)]
   ClsCmbFreeFree = datalist[:,int(1+20*i)]
   CldCmbFreeFree = datalist[:,int(2+20*i)]
   ClBHCmbFreeFree = datalist[:,int(3+20*i)]
   ClBHgCmbFreeFree = datalist[:,int(4+20*i)]
   # Primary
   ClqCmbFreeFreeKappa = datalist[:,int(5+20*i)]
   ClsCmbFreeFreeKappa = datalist[:,int(6+20*i)]
   CldCmbFreeFreeKappa = datalist[:,int(7+20*i)]
   ClBHCmbFreeFreeKappa = datalist[:,int(8+20*i)]
   ClBHgCmbFreeFreeKappa = datalist[:,int(9+20*i)]
   # k_rec x LSST 
   ClqCmbFreeFreeLsstgold = datalist[:,int(10+20*i)]
   ClsCmbFreeFreeLsstgold = datalist[:,int(11+20*i)]
   CldCmbFreeFreeLsstgold = datalist[:,int(12+20*i)]
   ClBHCmbFreeFreeLsstgold = datalist[:,int(13+20*i)]
   ClBHgCmbFreeFreeLsstgold = datalist[:,int(14+20*i)]
   # Secondary
   ClqCmbFreeFreeSec = datalist[:,int(15+20*i)]
   CldCmbFreeFreeSec = datalist[:,int(16+20*i)]
   ClsCmbFreeFreeSec = datalist[:,int(17+20*i)]
   ClBHCmbFreeFreeSec = datalist[:,int(18+20*i)]
   ClBHgCmbFreeFreeSec = datalist[:,int(19+20*i)]

   i=6
   # Sync
   # 
   # Trispectrum
   ClqCmbSync = datalist[:,int(0+20*i)]
   ClsCmbSync = datalist[:,int(1+20*i)]
   CldCmbSync = datalist[:,int(2+20*i)]
   ClBHCmbSync = datalist[:,int(3+20*i)]
   ClBHgCmbSync = datalist[:,int(4+20*i)]
   # Primary
   ClqCmbSyncKappa = datalist[:,int(5+20*i)]
   ClsCmbSyncKappa = datalist[:,int(6+20*i)]
   CldCmbSyncKappa = datalist[:,int(7+20*i)]
   ClBHCmbSyncKappa = datalist[:,int(8+20*i)]
   ClBHgCmbSyncKappa = datalist[:,int(9+20*i)]
   # k_rec x LSST 
   ClqCmbSyncLsstgold = datalist[:,int(10+20*i)]
   ClsCmbSyncLsstgold = datalist[:,int(11+20*i)]
   CldCmbSyncLsstgold = datalist[:,int(12+20*i)]
   ClBHCmbSyncLsstgold = datalist[:,int(13+20*i)]
   ClBHgCmbSyncLsstgold = datalist[:,int(14+20*i)]
   # Secondary
   ClqCmbSyncSec = datalist[:,int(15+20*i)]
   CldCmbSyncSec = datalist[:,int(16+20*i)]
   ClsCmbSyncSec = datalist[:,int(17+20*i)]
   ClBHCmbSyncSec = datalist[:,int(18+20*i)]
   ClBHgCmbSyncSec = datalist[:,int(19+20*i)]

   i=7
   # All extragalactic foregrounds
   # 
   # Trispectrum
   ClqCmbAll = datalist[:,int(0+20*i)]
   ClsCmbAll = datalist[:,int(1+20*i)]
   CldCmbAll = datalist[:,int(2+20*i)]
   ClBHCmbAll = datalist[:,int(3+20*i)]
   ClBHgCmbAll = datalist[:,int(4+20*i)]
   # Primary
   ClqCmbAllKappa = datalist[:,int(5+20*i)]
   ClsCmbAllKappa = datalist[:,int(6+20*i)]
   CldCmbAllKappa = datalist[:,int(7+20*i)]
   ClBHCmbAllKappa = datalist[:,int(8+20*i)]
   ClBHgCmbAllKappa = datalist[:,int(9+20*i)]
   # k_rec x LSST 
   ClqCmbAllLsstgold = datalist[:,int(10+20*i)]
   ClsCmbAllLsstgold = datalist[:,int(11+20*i)]
   CldCmbAllLsstgold = datalist[:,int(12+20*i)]
   ClBHCmbAllLsstgold = datalist[:,int(13+20*i)]
   ClBHgCmbAllLsstgold = datalist[:,int(14+20*i)]
   # Secondary
   ClqCmbAllSec = datalist[:,int(15+20*i)]
   CldCmbAllSec = datalist[:,int(16+20*i)]
   ClsCmbAllSec = datalist[:,int(17+20*i)]
   ClBHCmbAllSec = datalist[:,int(18+20*i)]
   ClBHgCmbAllSec = datalist[:,int(19+20*i)]
   

   # QE
   s2 =  2. * (ClkCmb + NqCmb)**2. / Nmodes 
   norm = np.sum(ClkCmb[I]**2. / s2[I])
   #
   snrQCmb[ilMax] = np.sqrt(norm)
   #
   #
   bQCmbCibFull[ilMax] = np.sum((2.*ClqCmbCibKappa[:,I[0]]+2.*ClqCmbCibSec[:,I[0]]+ClqCmbCib[:,I[0]]) * ClkCmb[I] / s2[I], axis=1) / norm
   #
   bQCmbTszFull[ilMax] = np.sum((2.*ClqCmbTszKappa[:,I[0]]+2.*ClqCmbTszSec[:,I[0]]+ClqCmbTsz[:,I[0]]) * ClkCmb[I] / s2[I], axis=1) / norm
   #
   bQCmbKszFull[ilMax] = np.sum((2.*ClqCmbKszKappa[:,I[0]]+2.*ClqCmbKszSec[:,I[0]]+ClqCmbKsz[:,I[0]]) * ClkCmb[I] / s2[I], axis=1) / norm
   #
   bQCmbRadiopsFull[ilMax] = np.sum((2.*ClqCmbRadiopsKappa[:,I[0]]+2.*ClqCmbRadiopsSec[:,I[0]]+ClqCmbRadiops[:,I[0]]) * ClkCmb[I] / s2[I], axis=1) / norm
   #
   bQCmbAllFull[ilMax] = np.sum((2.*ClqCmbAllKappa[:,I[0]]+2.*ClqCmbAllSec[:,I[0]]+ClqCmbAll[:,I[0]]) * ClkCmb[I] / s2[I], axis=1) / norm
   #
   #
   s2 = ((ClkCmb + NqCmb)*Cgg + (Ckg)**2) / Nmodes
   norm = np.sum(Ckg[I]**2 / s2[I])
   #
   sQCmbLsstgold[ilMax] = 1./np.sqrt(norm)
   #
   bQCmbCibLsstgold[ilMax] = np.sum(ClqCmbCibLsstgold[:,I[0]] * Ckg[I] / s2[I], axis=1) / norm
   #
   bQCmbTszLsstgold[ilMax] = np.sum(ClqCmbTszLsstgold[:,I[0]] * Ckg[I] / s2[I], axis=1) / norm
   #
   bQCmbKszLsstgold[ilMax] = np.sum(ClqCmbKszLsstgold[:,I[0]] * Ckg[I] / s2[I], axis=1) / norm
   #
   bQCmbRadiopsLsstgold[ilMax] = np.sum(ClqCmbRadiopsLsstgold[:,I[0]] * Ckg[I] / s2[I], axis=1) / norm
   #
   bQCmbAllLsstgold[ilMax] = np.sum(ClqCmbAllLsstgold[:,I[0]] * Ckg[I] / s2[I], axis=1) / norm
   

   # Shear
   s2 =  2. * (ClkCmb + NsCmb)**2 / Nmodes 
   norm = np.sum(ClkCmb[I]**2 / s2[I])
   #
   snrSCmb[ilMax] = np.sqrt(norm)
   #
   bSCmbCibFull[ilMax] = np.sum((2.*ClsCmbCibKappa[:,I[0]]+2.*ClsCmbCibSec[:,I[0]]+ClsCmbCib[:,I[0]]) * ClkCmb[I] / s2[I], axis=1) / norm
   #
   bSCmbTszFull[ilMax] = np.sum((2.*ClsCmbTszKappa[:,I[0]]+2.*ClsCmbTszSec[:,I[0]]+ClsCmbTsz[:,I[0]]) * ClkCmb[I] / s2[I], axis=1) / norm
   #
   bSCmbKszFull[ilMax] = np.sum((2.*ClsCmbKszKappa[:,I[0]]+2.*ClsCmbKszSec[:,I[0]]+ClsCmbKsz[:,I[0]]) * ClkCmb[I] / s2[I], axis=1) / norm
   #
   bSCmbRadiopsFull[ilMax] = np.sum((2.*ClsCmbRadiopsKappa[:,I[0]]+2.*ClsCmbRadiopsSec[:,I[0]]+ClsCmbRadiops[:,I[0]]) * ClkCmb[I] / s2[I], axis=1) / norm
   #
   bSCmbAllFull[ilMax] = np.sum((2.*ClsCmbAllKappa[:,I[0]]+2.*ClsCmbAllSec[:,I[0]]+ClsCmbAll[:,I[0]]) * ClkCmb[I] / s2[I], axis=1) / norm
   #
   #
   s2 = ((ClkCmb + NsCmb)*Cgg + (Ckg)**2) / Nmodes
   norm = np.sum(Ckg[I]**2 / s2[I])
   #
   sSCmbLsstgold[ilMax] = 1./np.sqrt(norm)
   #
   bSCmbCibLsstgold[ilMax] = np.sum(ClsCmbCibLsstgold[:,I[0]] * Ckg[I] / s2[I], axis=1) / norm
   #
   bSCmbTszLsstgold[ilMax] = np.sum(ClsCmbTszLsstgold[:,I[0]] * Ckg[I] / s2[I], axis=1) / norm
   #
   bSCmbKszLsstgold[ilMax] = np.sum(ClsCmbKszLsstgold[:,I[0]] * Ckg[I] / s2[I], axis=1) / norm
   #
   bSCmbRadiopsLsstgold[ilMax] = np.sum(ClsCmbRadiopsLsstgold[:,I[0]] * Ckg[I] / s2[I], axis=1) / norm
   #
   bSCmbAllLsstgold[ilMax] = np.sum(ClsCmbAllLsstgold[:,I[0]] * Ckg[I] / s2[I], axis=1) / norm


   # BH
   s2 =  2. * (ClkCmb + NbhCmb)**2 / Nmodes 
   norm = np.sum(ClkCmb[I]**2 / s2[I])
   #
   snrBHCmb[ilMax] = np.sqrt(norm)
   #
   bBHCmbCibFull[ilMax] = np.sum((2.*ClBHCmbCibKappa[:,I[0]]+2.*ClBHCmbCibSec[:,I[0]]+ClBHCmbCib[:,I[0]]) * ClkCmb[I] / s2[I], axis=1) / norm
   #
   bBHCmbTszFull[ilMax] = np.sum((2.*ClBHCmbTszKappa[:,I[0]]+2.*ClBHCmbTszSec[:,I[0]]+ClBHCmbTsz[:,I[0]]) * ClkCmb[I] / s2[I], axis=1) / norm
   #
   bBHCmbKszFull[ilMax] = np.sum((2.*ClBHCmbKszKappa[:,I[0]]+2.*ClBHCmbKszSec[:,I[0]]+ClBHCmbKsz[:,I[0]]) * ClkCmb[I] / s2[I], axis=1) / norm
   #
   bBHCmbRadiopsFull[ilMax] = np.sum((2.*ClBHCmbRadiopsKappa[:,I[0]]+2.*ClBHCmbRadiopsSec[:,I[0]]+ClBHCmbRadiops[:,I[0]]) * ClkCmb[I] / s2[I], axis=1) / norm
   #
   bBHCmbAllFull[ilMax] = np.sum((2.*ClBHCmbAllKappa[:,I[0]]+2.*ClBHCmbAllSec[:,I[0]]+ClBHCmbAll[:,I[0]]) * ClkCmb[I] / s2[I], axis=1) / norm
   #
   #
   s2 = ((ClkCmb + NbhCmb)*Cgg + (Ckg)**2) / Nmodes
   norm = np.sum(Ckg[I]**2 / s2[I])
   #
   sBHCmbLsstgold[ilMax] = 1./np.sqrt(norm)
   #
   bBHCmbCibLsstgold[ilMax] = np.sum(ClBHCmbCibLsstgold[:,I[0]] * Ckg[I] / s2[I], axis=1) / norm
   #
   bBHCmbTszLsstgold[ilMax] = np.sum(ClBHCmbTszLsstgold[:,I[0]] * Ckg[I] / s2[I], axis=1) / norm
   #
   bBHCmbKszLsstgold[ilMax] = np.sum(ClBHCmbKszLsstgold[:,I[0]] * Ckg[I] / s2[I], axis=1) / norm
   #
   bBHCmbRadiopsLsstgold[ilMax] = np.sum(ClBHCmbRadiopsLsstgold[:,I[0]] * Ckg[I] / s2[I], axis=1) / norm
   #
   bBHCmbAllLsstgold[ilMax] = np.sum(ClBHCmbAllLsstgold[:,I[0]] * Ckg[I] / s2[I], axis=1) / norm


   # BH with gaussian profile
   s2 =  2. * (ClkCmb + NbhgCmb)**2 / Nmodes 
   norm = np.sum(ClkCmb[I]**2 / s2[I])
   #
   snrBHgCmb[ilMax] = np.sqrt(norm)
   #
   bBHgCmbCibFull[ilMax] = np.sum((2.*ClBHgCmbCibKappa[:,I[0]]+2.*ClBHgCmbCibSec[:,I[0]]+ClBHgCmbCib[:,I[0]]) * ClkCmb[I] / s2[I], axis=1) / norm
   #
   bBHgCmbTszFull[ilMax] = np.sum((2.*ClBHgCmbTszKappa[:,I[0]]+2.*ClBHgCmbTszSec[:,I[0]]+ClBHgCmbTsz[:,I[0]]) * ClkCmb[I] / s2[I], axis=1) / norm
   #
   bBHgCmbKszFull[ilMax] = np.sum((2.*ClBHgCmbKszKappa[:,I[0]]+2.*ClBHgCmbKszSec[:,I[0]]+ClBHgCmbKsz[:,I[0]]) * ClkCmb[I] / s2[I], axis=1) / norm
   #
   bBHgCmbRadiopsFull[ilMax] = np.sum((2.*ClBHgCmbRadiopsKappa[:,I[0]]+2.*ClBHgCmbRadiopsSec[:,I[0]]+ClBHgCmbRadiops[:,I[0]]) * ClkCmb[I] / s2[I], axis=1) / norm
   #
   bBHgCmbAllFull[ilMax] = np.sum((2.*ClBHgCmbAllKappa[:,I[0]]+2.*ClBHgCmbAllSec[:,I[0]]+ClBHgCmbAll[:,I[0]]) * ClkCmb[I] / s2[I], axis=1) / norm
   #
   #
   s2 = ((ClkCmb + NbhgCmb)*Cgg + (Ckg)**2) / Nmodes
   norm = np.sum(Ckg[I]**2 / s2[I])
   #
   sBHgCmbLsstgold[ilMax] = 1./np.sqrt(norm)
   #
   bBHgCmbCibLsstgold[ilMax] = np.sum(ClBHgCmbCibLsstgold[:,I[0]] * Ckg[I] / s2[I], axis=1) / norm
   #
   bBHgCmbTszLsstgold[ilMax] = np.sum(ClBHgCmbTszLsstgold[:,I[0]] * Ckg[I] / s2[I], axis=1) / norm
   #
   bBHgCmbKszLsstgold[ilMax] = np.sum(ClBHgCmbKszLsstgold[:,I[0]] * Ckg[I] / s2[I], axis=1) / norm
   #
   bBHgCmbRadiopsLsstgold[ilMax] = np.sum(ClBHgCmbRadiopsLsstgold[:,I[0]] * Ckg[I] / s2[I], axis=1) / norm
   #
   bBHgCmbAllLsstgold[ilMax] = np.sum(ClBHgCmbAllLsstgold[:,I[0]] * Ckg[I] / s2[I], axis=1) / norm


   # Hybrid: QE(<=2000) + shear(2000-lmax)
   if llMax[ilMax]<=2000.:
      sHybridCmb[ilMax] = 1./snrQCmb[ilMax]
      bHybridCmbAllFull[ilMax] = bQCmbAllFull[ilMax]
      #
      sHybridCmbLsstgold[ilMax] = sQCmbLsstgold[ilMax]
      bHybridCmbAllLsstgold[ilMax] = bQCmbAllLsstgold[ilMax]
   else:
      # compute combined noise
      fNqCmbLowl = baseMap.forecastN0Kappa(cmb.flensedTT, cmb.fCtotal, lMin=lMin, lMax=2000., test=False)
      fNsCmbHighl = baseMap.forecastN0KappaShear(cmb.flensedTT, cmb.fCtotal, lMin=2000., lMax=llMax[ilMax], corr=True, test=False)
      Nhybrid = 1./(1./fNqCmbLowl(lCen) + 1./fNsCmbHighl(lCen))
      # Auto
      s2 = 2. * (ClkCmb + Nhybrid)**2 / Nmodes
      norm = np.sum(ClkCmb[I]**2 / s2[I])
      #
      sHybridCmb[ilMax] = 1./np.sqrt(norm)
      # Get bias for QE at lMax = 2000.
      datalist = np.array([np.genfromtxt(outputFolder+"/data_lmaxT_"+str(int(2000.))+"_"+str(i)+".txt") for i in  range(nPatch)])
      ClqCmbAll = datalist[:,int(0+20*7)]
      ClqCmbAllKappa = datalist[:,int(5+20*7)]
      ClqCmbAllSec = datalist[:,int(15+20*7)]
      bQCmbAll2000 = 2.*ClqCmbAllKappa[:,I[0]]+2.*ClqCmbAllSec[:,I[0]]+ClqCmbAll[:,I[0]]
      #
      bSCmbAllilMax = 2.*ClsCmbAllKappa[:,I[0]]+2.*ClsCmbAllSec[:,I[0]]+ClsCmbAll[:,I[0]]
      bHybridCmbAll = (Nhybrid[I]**2.) * (bQCmbAll2000/(fNqCmbLowl(lCen)[I])**2. + bSCmbAllilMax/(fNsCmbHighl(lCen)[I])**2.)
      bHybridCmbAllFull[ilMax] = np.sum(bHybridCmbAll * ClkCmb[I] / s2[I], axis=1) / norm
      # Cross
      s2 = ((ClkCmb + Nhybrid)*Cgg + (Ckg)**2) / Nmodes
      norm = np.sum(Ckg[I]**2 / s2[I])
      #
      sHybridCmbLsstgold[ilMax] = 1./np.sqrt(norm)
      # Get bias for QE at lMax = 2000.
      ClqCmbAllLsstgold = datalist[:,int(10+20*7)]
      bQCmbAllLsstgold2000 = ClqCmbAllLsstgold[:,I[0]] 
      #
      bSCmbAllLsstgoldilMax = ClsCmbAllLsstgold[:,I[0]]
      bHybridCmbAllLsstgoldt = Nhybrid[I] * (bQCmbAllLsstgold2000/(fNqCmbLowl(lCen)[I]) + bSCmbAllLsstgoldilMax/(fNsCmbHighl(lCen)[I]))
      bHybridCmbAllLsstgold[ilMax] = np.sum(bHybridCmbAllLsstgoldt * Ckg[I] / s2[I], axis=1) / norm


##################################################################################
##################################################################################
# Average over patches

# QE
sBQCmbCibFull = np.std(bQCmbCibFull, axis=1) * uncertaintyFactor / np.sqrt(nPatch)
bQCmbCibFull = np.mean(bQCmbCibFull, axis=1)
#
sBQCmbTszFull = np.std(bQCmbTszFull, axis=1) * uncertaintyFactor / np.sqrt(nPatch)
bQCmbTszFull = np.mean(bQCmbTszFull, axis=1)
#
sBQCmbKszFull = np.std(bQCmbKszFull, axis=1) * uncertaintyFactor / np.sqrt(nPatch)
bQCmbKszFull = np.mean(bQCmbKszFull, axis=1)
#
sBQCmbRadiopsFull = np.std(bQCmbRadiopsFull, axis=1) * uncertaintyFactor / np.sqrt(nPatch)
bQCmbRadiopsFull = np.mean(bQCmbRadiopsFull, axis=1)
#
sBQCmbAllFull = np.std(bQCmbAllFull, axis=1) * uncertaintyFactor / np.sqrt(nPatch)
bQCmbAllFull = np.mean(bQCmbAllFull, axis=1)
#
#
sBQCmbCibLsstgold = np.std(bQCmbCibLsstgold, axis=1) * uncertaintyFactor / np.sqrt(nPatch)
bQCmbCibLsstgold = np.mean(bQCmbCibLsstgold, axis=1)
#
sBQCmbTszLsstgold = np.std(bQCmbTszLsstgold, axis=1) * uncertaintyFactor / np.sqrt(nPatch)
bQCmbTszLsstgold = np.mean(bQCmbTszLsstgold, axis=1)
#
sBQCmbKszLsstgold = np.std(bQCmbKszLsstgold, axis=1) * uncertaintyFactor / np.sqrt(nPatch)
bQCmbKszLsstgold = np.mean(bQCmbKszLsstgold, axis=1)
#
sBQCmbRadiopsLsstgold = np.std(bQCmbRadiopsLsstgold, axis=1) / np.sqrt(nPatch)
bQCmbRadiopsLsstgold = np.mean(bQCmbRadiopsLsstgold, axis=1)
#
sBQCmbAllLsstgold = np.std(bQCmbAllLsstgold, axis=1) / np.sqrt(nPatch)
bQCmbAllLsstgold = np.mean(bQCmbAllLsstgold, axis=1)


# Shear
sBSCmbCibFull = np.std(bSCmbCibFull, axis=1) * uncertaintyFactor / np.sqrt(nPatch)
bSCmbCibFull = np.mean(bSCmbCibFull, axis=1)
#
sBSCmbTszFull = np.std(bSCmbTszFull, axis=1) * uncertaintyFactor / np.sqrt(nPatch)
bSCmbTszFull = np.mean(bSCmbTszFull, axis=1)
#
sBSCmbKszFull = np.std(bSCmbKszFull, axis=1) * uncertaintyFactor / np.sqrt(nPatch)
bSCmbKszFull = np.mean(bSCmbKszFull, axis=1)
#
sBSCmbRadiopsFull = np.std(bSCmbRadiopsFull, axis=1) * uncertaintyFactor / np.sqrt(nPatch)
bSCmbRadiopsFull = np.mean(bSCmbRadiopsFull, axis=1) 
#
sBSCmbAllFull = np.std(bSCmbAllFull, axis=1) * uncertaintyFactor / np.sqrt(nPatch)
bSCmbAllFull = np.mean(bSCmbAllFull, axis=1)
#
#
sBSCmbCibLsstgold = np.std(bSCmbCibLsstgold, axis=1) * uncertaintyFactor / np.sqrt(nPatch)
bSCmbCibLsstgold = np.mean(bSCmbCibLsstgold, axis=1)
#
sBSCmbTszLsstgold = np.std(bSCmbTszLsstgold, axis=1) * uncertaintyFactor / np.sqrt(nPatch)
bSCmbTszLsstgold = np.mean(bSCmbTszLsstgold, axis=1)
#
sBSCmbKszLsstgold = np.std(bSCmbKszLsstgold, axis=1) * uncertaintyFactor / np.sqrt(nPatch)
bSCmbKszLsstgold = np.mean(bSCmbKszLsstgold, axis=1)
#
sBSCmbRadiopsLsstgold = np.std(bSCmbRadiopsLsstgold, axis=1) * uncertaintyFactor / np.sqrt(nPatch)
bSCmbRadiopsLsstgold = np.mean(bSCmbRadiopsLsstgold, axis=1)
#
sBSCmbAllLsstgold = np.std(bSCmbAllLsstgold, axis=1) * uncertaintyFactor / np.sqrt(nPatch)
bSCmbAllLsstgold = np.mean(bSCmbAllLsstgold, axis=1)


# BH
sBbHCmbCibFull = np.std(bBHCmbCibFull, axis=1) * uncertaintyFactor / np.sqrt(nPatch)
bBHCmbCibFull = np.mean(bBHCmbCibFull, axis=1)
#
sBbHCmbTszFull = np.std(bBHCmbTszFull, axis=1) * uncertaintyFactor / np.sqrt(nPatch)
bBHCmbTszFull = np.mean(bBHCmbTszFull, axis=1)
#
sBbHCmbKszFull = np.std(bBHCmbKszFull, axis=1) * uncertaintyFactor / np.sqrt(nPatch) 
bBHCmbKszFull = np.mean(bBHCmbKszFull, axis=1)
#
sBbHCmbRadiopsFull = np.std(bBHCmbRadiopsFull, axis=1) * uncertaintyFactor / np.sqrt(nPatch) 
bBHCmbRadiopsFull = np.mean(bBHCmbRadiopsFull, axis=1)
#
sBbHCmbAllFull = np.std(bBHCmbAllFull, axis=1) * uncertaintyFactor / np.sqrt(nPatch)
bBHCmbAllFull = np.mean(bBHCmbAllFull, axis=1)
#
#
sBbHCmbCibLsstgold = np.std(bBHCmbCibLsstgold, axis=1) * uncertaintyFactor / np.sqrt(nPatch)
bBHCmbCibLsstgold = np.mean(bBHCmbCibLsstgold, axis=1)
#
sBbHCmbTszLsstgold = np.std(bBHCmbTszLsstgold, axis=1) * uncertaintyFactor / np.sqrt(nPatch)
bBHCmbTszLsstgold = np.mean(bBHCmbTszLsstgold, axis=1)
#
sBbHCmbKszLsstgold = np.std(bBHCmbKszLsstgold, axis=1) * uncertaintyFactor / np.sqrt(nPatch)
bBHCmbKszLsstgold = np.mean(bBHCmbKszLsstgold, axis=1)
#
sBbHCmbRadiopsLsstgold = np.std(bBHCmbRadiopsLsstgold, axis=1) * uncertaintyFactor / np.sqrt(nPatch)
bBHCmbRadiopsLsstgold = np.mean(bBHCmbRadiopsLsstgold, axis=1)
#
sBbHCmbAllLsstgold = np.std(bBHCmbAllLsstgold, axis=1) * uncertaintyFactor / np.sqrt(nPatch)
bBHCmbAllLsstgold = np.mean(bBHCmbAllLsstgold, axis=1)


# BH with Gaussian profile
sBbHgCmbCibFull = np.std(bBHgCmbCibFull, axis=1) * uncertaintyFactor / np.sqrt(nPatch)
bBHgCmbCibFull = np.mean(bBHgCmbCibFull, axis=1)
#
sBbHgCmbTszFull = np.std(bBHgCmbTszFull, axis=1) * uncertaintyFactor / np.sqrt(nPatch)
bBHgCmbTszFull = np.mean(bBHgCmbTszFull, axis=1)
#
sBbHgCmbKszFull = np.std(bBHgCmbKszFull, axis=1) * uncertaintyFactor / np.sqrt(nPatch) 
bBHgCmbKszFull = np.mean(bBHgCmbKszFull, axis=1)
#
sBbHgCmbRadiopsFull = np.std(bBHgCmbRadiopsFull, axis=1) * uncertaintyFactor / np.sqrt(nPatch) 
bBHgCmbRadiopsFull = np.mean(bBHgCmbRadiopsFull, axis=1)
#
sBbHgCmbAllFull = np.std(bBHgCmbAllFull, axis=1) * uncertaintyFactor / np.sqrt(nPatch)
bBHgCmbAllFull = np.mean(bBHgCmbAllFull, axis=1)
#
#
sBbHgCmbCibLsstgold = np.std(bBHgCmbCibLsstgold, axis=1) * uncertaintyFactor / np.sqrt(nPatch)
bBHgCmbCibLsstgold = np.mean(bBHgCmbCibLsstgold, axis=1)
#
sBbHgCmbTszLsstgold = np.std(bBHgCmbTszLsstgold, axis=1) * uncertaintyFactor / np.sqrt(nPatch)
bBHgCmbTszLsstgold = np.mean(bBHgCmbTszLsstgold, axis=1)
#
sBbHgCmbKszLsstgold = np.std(bBHgCmbKszLsstgold, axis=1) * uncertaintyFactor / np.sqrt(nPatch)
bBHgCmbKszLsstgold = np.mean(bBHgCmbKszLsstgold, axis=1)
#
sBbHgCmbRadiopsLsstgold = np.std(bBHgCmbRadiopsLsstgold, axis=1) * uncertaintyFactor / np.sqrt(nPatch)
bBHgCmbRadiopsLsstgold = np.mean(bBHgCmbRadiopsLsstgold, axis=1)
#
sBbHgCmbAllLsstgold = np.std(bBHgCmbAllLsstgold, axis=1) * uncertaintyFactor / np.sqrt(nPatch)
bBHgCmbAllLsstgold = np.mean(bBHgCmbAllLsstgold, axis=1)


# Hybrid
bHybridCmbAllFull = np.mean(bHybridCmbAllFull, axis=1)
bHybridCmbAllLsstgold = np.mean(bHybridCmbAllLsstgold, axis=1)



##################################################################################
##################################################################################
# Summary for each estimator: Cross with LSST gold

fig=plt.figure(0)
gs = gridspec.GridSpec(4, 1)
gs.update(hspace=0.)


# Bias hardened
ax0=plt.subplot(gs[0])
#
ax0.axhline(0., c='k', lw=1)
ax0.fill_between(llMax, -sBHCmbLsstgold, sBHCmbLsstgold, edgecolor='', facecolor='gray', alpha=0.6)
# CIB
ax0.plot(llMax, bBHCmbCibLsstgold, cCib, label='CIB')
Up = bBHCmbCibLsstgold + sBbHCmbCibLsstgold
Down = bBHCmbCibLsstgold - sBbHCmbCibLsstgold
ax0.fill_between(llMax, Down, Up, edgecolor='', facecolor=cCib, alpha=0.4)
# tSZ
ax0.plot(llMax, bBHCmbTszLsstgold, cTsz, label='tSZ')
Up = bBHCmbTszLsstgold + sBbHCmbTszLsstgold
Down = bBHCmbTszLsstgold - sBbHCmbTszLsstgold
ax0.fill_between(llMax, Down, Up, edgecolor='', facecolor=cTsz, alpha=0.4)
# kSZ
ax0.plot(llMax, bBHCmbKszLsstgold, cKsz, label='kSZ')
Up = bBHCmbKszLsstgold + sBbHCmbKszLsstgold
Down = bBHCmbKszLsstgold - sBbHCmbKszLsstgold
ax0.fill_between(llMax, Down, Up, edgecolor='', facecolor=cKsz, alpha=0.4)
# radio point sources
ax0.plot(llMax, bBHCmbRadiopsLsstgold, cRadiops, label='Radiops')
Up = bBHCmbRadiopsLsstgold + sBbHCmbRadiopsLsstgold
Down = bBHCmbRadiopsLsstgold - sBbHCmbRadiopsLsstgold
ax0.fill_between(llMax, Down, Up, edgecolor='', facecolor=cRadiops, alpha=0.4)
# all
ax0.plot(llMax, bBHCmbAllLsstgold, cAll, label='All')
Up = bBHCmbAllLsstgold + sBbHCmbAllLsstgold
Down = bBHCmbAllLsstgold - sBbHCmbAllLsstgold
ax0.fill_between(llMax, Down, Up, edgecolor='', facecolor=cAll, alpha=0.4)
#
ax0.set_ylim((-0.025, 0.025))
ax0.set_ylabel('BH1', fontsize='x-small')
#
plt.setp(ax0.get_xticklabels(), visible=False)
ax0.legend(loc=2, ncol=4, mode=None, handlelength=.5, fontsize='x-small', borderpad=0.1, borderaxespad=0.1, frameon=False)
ax0.set_title(r'Relative Bias on $C_L^{\kappa \times \text{LSST}}$ amplitude')


# Bias hardened with Gaussian profile
ax1=plt.subplot(gs[1], sharex=ax0)
#
ax1.axhline(0., c='k', lw=1)
ax1.fill_between(llMax, -sBHgCmbLsstgold, sBHgCmbLsstgold, edgecolor='', facecolor='gray', alpha=0.6)
# CIB
ax1.plot(llMax, bBHgCmbCibLsstgold, cCib, label='CIB')
Up = bBHgCmbCibLsstgold + sBbHgCmbCibLsstgold
Down = bBHgCmbCibLsstgold - sBbHgCmbCibLsstgold
ax1.fill_between(llMax, Down, Up, edgecolor='', facecolor=cCib, alpha=0.4)
# tSZ
ax1.plot(llMax, bBHgCmbTszLsstgold, cTsz, label='tSZ')
Up = bBHgCmbTszLsstgold + sBbHgCmbTszLsstgold
Down = bBHgCmbTszLsstgold - sBbHgCmbTszLsstgold
ax1.fill_between(llMax, Down, Up, edgecolor='', facecolor=cTsz, alpha=0.4)
# kSZ
ax1.plot(llMax, bBHgCmbKszLsstgold, cKsz, label='kSZ')
Up = bBHgCmbKszLsstgold + sBbHgCmbKszLsstgold
Down = bBHgCmbKszLsstgold - sBbHgCmbKszLsstgold
ax1.fill_between(llMax, Down, Up, edgecolor='', facecolor=cKsz, alpha=0.4)
# radio point sources
ax1.plot(llMax, bBHgCmbRadiopsLsstgold, cRadiops, label='Radiops')
Up = bBHgCmbRadiopsLsstgold + sBbHgCmbRadiopsLsstgold
Down = bBHgCmbRadiopsLsstgold - sBbHgCmbRadiopsLsstgold
ax1.fill_between(llMax, Down, Up, edgecolor='', facecolor=cRadiops, alpha=0.4)
# all
ax1.plot(llMax, bBHgCmbAllLsstgold, cAll, label='All')
Up = bBHgCmbAllLsstgold + sBbHgCmbAllLsstgold
Down = bBHgCmbAllLsstgold - sBbHgCmbAllLsstgold
ax1.fill_between(llMax, Down, Up, edgecolor='', facecolor=cAll, alpha=0.4)
#
ax1.set_ylim((-0.025, 0.025))
ax1.set_ylabel('BH2', fontsize='x-small')
#
plt.setp(ax1.get_xticklabels(), visible=False)
# remove last tick label for the second subplot
yticks = ax1.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)


# Shear
ax2=plt.subplot(gs[2], sharex=ax1)
#
ax2.axhline(0., c='k', lw=1)
ax2.fill_between(llMax, -sSCmbLsstgold, sSCmbLsstgold, edgecolor='', facecolor='gray', alpha=0.6)
# CIB
ax2.plot(llMax, bSCmbCibLsstgold, cCib, label='CIB')
Up = bSCmbCibLsstgold + sBSCmbCibLsstgold
Down = bSCmbCibLsstgold - sBSCmbCibLsstgold
ax2.fill_between(llMax, Down, Up, edgecolor='', facecolor=cCib, alpha=0.4)
# tSZ
ax2.plot(llMax, bSCmbTszLsstgold, cTsz, label='tSZ')
Up = bSCmbTszLsstgold + sBSCmbTszLsstgold
Down = bSCmbTszLsstgold - sBSCmbTszLsstgold
ax2.fill_between(llMax, Down, Up, edgecolor='', facecolor=cTsz, alpha=0.4)
# kSZ
ax2.plot(llMax, bSCmbKszLsstgold, cKsz, label='kSZ')
Up = bSCmbKszLsstgold + sBSCmbKszLsstgold
Down = bSCmbKszLsstgold - sBSCmbKszLsstgold
ax2.fill_between(llMax, Down, Up, edgecolor='', facecolor=cKsz, alpha=0.4)
# radio point sources
ax2.plot(llMax, bSCmbRadiopsLsstgold, cRadiops, label='Radiops')
Up = bSCmbRadiopsLsstgold + sBSCmbRadiopsLsstgold
Down = bSCmbRadiopsLsstgold - sBSCmbRadiopsLsstgold
ax2.fill_between(llMax, Down, Up, edgecolor='', facecolor=cRadiops, alpha=0.4)
# all
ax2.plot(llMax, bSCmbAllLsstgold, cAll, label='All')
Up = bSCmbAllLsstgold + sBSCmbAllLsstgold
Down = bSCmbAllLsstgold - sBSCmbAllLsstgold
ax2.fill_between(llMax, Down, Up, edgecolor='', facecolor=cAll, alpha=0.4)
#
ax2.set_ylim((-0.025, 0.025))
ax2.set_ylabel(r'Shear', fontsize='x-small')
#
plt.setp(ax2.get_xticklabels(), visible=False)
# remove last tick label for the second subplot
yticks = ax2.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)


# QE
ax3=plt.subplot(gs[3], sharex=ax2)
#
ax3.axhline(0., c='k', lw=1)
ax3.fill_between(llMax, -sQCmbLsstgold, sQCmbLsstgold, edgecolor='', facecolor='gray', alpha=0.6)
# CIB
ax3.plot(llMax, bQCmbCibLsstgold, cCib, label='CIB')
Up = bQCmbCibLsstgold + sBQCmbCibLsstgold
Down = bQCmbCibLsstgold - sBQCmbCibLsstgold
ax3.fill_between(llMax, Down, Up, edgecolor='', facecolor=cCib, alpha=0.4)
# tSZ
ax3.plot(llMax, bQCmbTszLsstgold, cTsz, label='tSZ')
Up = bQCmbTszLsstgold + sBQCmbTszLsstgold
Down = bQCmbTszLsstgold - sBQCmbTszLsstgold
ax3.fill_between(llMax, Down, Up, edgecolor='', facecolor=cTsz, alpha=0.4)
# kSZ
ax3.plot(llMax, bQCmbKszLsstgold, cKsz, label='kSZ')
Up = bQCmbKszLsstgold + sBQCmbKszLsstgold
Down = bQCmbKszLsstgold - sBQCmbKszLsstgold
ax3.fill_between(llMax, Down, Up, edgecolor='', facecolor=cKsz, alpha=0.4)
# radio point sources
ax3.plot(llMax, bQCmbRadiopsLsstgold, cRadiops, label='Radiops')
Up = bQCmbRadiopsLsstgold + sBQCmbRadiopsLsstgold
Down = bQCmbRadiopsLsstgold - sBQCmbRadiopsLsstgold
ax3.fill_between(llMax, Down, Up, edgecolor='', facecolor=cRadiops, alpha=0.4)
# all
ax3.plot(llMax, bQCmbAllLsstgold, cAll, label='All')
Up = bQCmbAllLsstgold + sBQCmbAllLsstgold
Down = bQCmbAllLsstgold - sBQCmbAllLsstgold
ax3.fill_between(llMax, Down, Up, edgecolor='', facecolor=cAll, alpha=0.4)
#
ax3.set_ylim((-0.025,0.025))
ax3.set_ylabel(r'QE', fontsize='x-small')
#
# remove last tick label for the second subplot
yticks = ax3.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)
#
ax3.set_xlim((llMax[0], llMax[-1]))
ax3.set_xlabel(r'$\ell_{\text{max, T}}$')


path = "output/summary_bias_Lsstgold.pdf"
fig.savefig(path, bbox_inches='tight')
fig.clf()


##################################################################################
##################################################################################
# Summary for each estimator: bias on k_CMB amplitude

fig=plt.figure(0)
gs = gridspec.GridSpec(4, 1)
gs.update(hspace=0.)

sSCmb = 1./snrSCmb
sQCmb = 1./snrQCmb
sBHCmb = 1./snrBHCmb
sBHgCmb = 1./snrBHgCmb


# Bias hardened
ax0=plt.subplot(gs[0])
#
ax0.axhline(0., c='k', lw=1)
ax0.fill_between(llMax, -sBHCmb, sBHCmb, edgecolor='', facecolor='gray', alpha=0.6)
# CIB
ax0.plot(llMax, bBHCmbCibFull, cCib, label='CIB')
Up = bBHCmbCibFull + sBbHCmbCibFull
Down = bBHCmbCibFull - sBbHCmbCibFull
ax0.fill_between(llMax, Down, Up, edgecolor='', facecolor=cCib, alpha=0.4)
# tSZ
ax0.plot(llMax, bBHCmbTszFull, cTsz, label='tSZ')
Up = bBHCmbTszFull + sBbHCmbTszFull
Down = bBHCmbTszFull - sBbHCmbTszFull
ax0.fill_between(llMax, Down, Up, edgecolor='', facecolor=cTsz, alpha=0.4)
# kSZ
ax0.plot(llMax, bBHCmbKszFull, cKsz, label='kSZ')
Up = bBHCmbKszFull + sBbHCmbKszFull
Down = bBHCmbKszFull - sBbHCmbKszFull
ax0.fill_between(llMax, Down, Up, edgecolor='', facecolor=cKsz, alpha=0.4)
# radio point sources
ax0.plot(llMax, bBHCmbRadiopsFull, cRadiops, label='Radiops')
Up = bBHCmbRadiopsFull + sBbHCmbRadiopsFull
Down = bBHCmbRadiopsFull - sBbHCmbRadiopsFull
ax0.fill_between(llMax, Down, Up, edgecolor='', facecolor=cRadiops, alpha=0.4)
# all
ax0.plot(llMax, bBHCmbAllFull, cAll, label='All')
Up = bBHCmbAllFull + sBbHCmbAllFull
Down = bBHCmbAllFull - sBbHCmbAllFull
ax0.fill_between(llMax, Down, Up, edgecolor='', facecolor=cAll, alpha=0.4)
#
ax0.set_ylim((-0.025, 0.025))
ax0.set_ylabel('BH1', fontsize='x-small')
#
plt.setp(ax0.get_xticklabels(), visible=False)
ax0.legend(loc=2, ncol=4, mode=None, handlelength=.5, fontsize='x-small', borderpad=0.1, borderaxespad=0.1, frameon=False)
ax0.set_title(r'Relative Bias on $C_L^{\kappa}$ amplitude')


# Bias hardened with Gaussian profile
ax1=plt.subplot(gs[1], sharex=ax0)
#
ax1.axhline(0., c='k', lw=1)
ax1.fill_between(llMax, -sBHgCmb, sBHgCmb, edgecolor='', facecolor='gray', alpha=0.6)
# CIB
ax1.plot(llMax, bBHgCmbCibFull, cCib, label='CIB')
Up = bBHgCmbCibFull + sBbHgCmbCibFull
Down = bBHgCmbCibFull - sBbHgCmbCibFull
ax1.fill_between(llMax, Down, Up, edgecolor='', facecolor=cCib, alpha=0.4)
# tSZ
ax1.plot(llMax, bBHgCmbTszFull, cTsz, label='tSZ')
Up = bBHgCmbTszFull + sBbHgCmbTszFull
Down = bBHgCmbTszFull - sBbHgCmbTszFull
ax1.fill_between(llMax, Down, Up, edgecolor='', facecolor=cTsz, alpha=0.4)
# kSZ
ax1.plot(llMax, bBHgCmbKszFull, cKsz, label='kSZ')
Up = bBHgCmbKszFull + sBbHgCmbKszFull
Down = bBHgCmbKszFull - sBbHgCmbKszFull
ax1.fill_between(llMax, Down, Up, edgecolor='', facecolor=cKsz, alpha=0.4)
# radio point sources
ax1.plot(llMax, bBHgCmbRadiopsFull, cRadiops, label='Radiops')
Up = bBHgCmbRadiopsFull + sBbHgCmbRadiopsFull
Down = bBHgCmbRadiopsFull - sBbHgCmbRadiopsFull
ax1.fill_between(llMax, Down, Up, edgecolor='', facecolor=cRadiops, alpha=0.4)
# all
ax1.plot(llMax, bBHgCmbAllFull, cAll, label='All')
Up = bBHgCmbAllFull + sBbHgCmbAllFull
Down = bBHgCmbAllFull - sBbHgCmbAllFull
ax1.fill_between(llMax, Down, Up, edgecolor='', facecolor=cAll, alpha=0.4)
#
ax1.set_ylim((-0.025, 0.025))
ax1.set_ylabel('BH2', fontsize='x-small')
#
plt.setp(ax1.get_xticklabels(), visible=False)
# remove last tick label for the second subplot
yticks = ax1.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)


# Shear
ax2=plt.subplot(gs[2], sharex=ax1)
#
ax2.axhline(0., c='k', lw=1)
ax2.fill_between(llMax, -sSCmb, sSCmb, edgecolor='', facecolor='gray', alpha=0.6)
# CIB
ax2.plot(llMax, bSCmbCibFull, cCib, label='CIB')
Up = bSCmbCibFull + sBSCmbCibFull
Down = bSCmbCibFull - sBSCmbCibFull
ax2.fill_between(llMax, Down, Up, edgecolor='', facecolor=cCib, alpha=0.4)
# tSZ
ax2.plot(llMax, bSCmbTszFull, cTsz, label='tSZ')
Up = bSCmbTszFull + sBSCmbTszFull
Down = bSCmbTszFull - sBSCmbTszFull
ax2.fill_between(llMax, Down, Up, edgecolor='', facecolor=cTsz, alpha=0.4)
# kSZ
ax2.plot(llMax, bSCmbKszFull, cKsz, label='kSZ')
Up = bSCmbKszFull + sBSCmbKszFull
Down = bSCmbKszFull - sBSCmbKszFull
ax2.fill_between(llMax, Down, Up, edgecolor='', facecolor=cKsz, alpha=0.4)
# radio point sources
ax2.plot(llMax, bSCmbRadiopsFull, cRadiops, label='Radiops')
Up = bSCmbRadiopsFull + sBSCmbRadiopsFull
Down = bSCmbRadiopsFull - sBSCmbRadiopsFull
ax2.fill_between(llMax, Down, Up, edgecolor='', facecolor=cRadiops, alpha=0.4)
# all
ax2.plot(llMax, bSCmbAllFull, cAll, label='All')
Up = bSCmbAllFull + sBSCmbAllFull
Down = bSCmbAllFull - sBSCmbAllFull
ax2.fill_between(llMax, Down, Up, edgecolor='', facecolor=cAll, alpha=0.4)
#
ax2.set_ylim((-0.025, 0.025))
ax2.set_ylabel(r'Shear', fontsize='x-small')
#
plt.setp(ax2.get_xticklabels(), visible=False)
# remove last tick label for the second subplot
yticks = ax2.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)


# QE
ax3=plt.subplot(gs[3], sharex=ax2)
#
ax3.axhline(0., c='k', lw=1)
ax3.fill_between(llMax, -sQCmb, sQCmb, edgecolor='', facecolor='gray', alpha=0.6)
# CIB
ax3.plot(llMax, bQCmbCibFull, cCib, label='CIB')
Up = bQCmbCibFull + sBQCmbCibFull
Down = bQCmbCibFull - sBQCmbCibFull
ax3.fill_between(llMax, Down, Up, edgecolor='', facecolor=cCib, alpha=0.4)
# tSZ
ax3.plot(llMax, bQCmbTszFull, cTsz, label='tSZ')
Up = bQCmbTszFull + sBQCmbTszFull
Down = bQCmbTszFull - sBQCmbTszFull
ax3.fill_between(llMax, Down, Up, edgecolor='', facecolor=cTsz, alpha=0.4)
# kSZ
ax3.plot(llMax, bQCmbKszFull, cKsz, label='kSZ')
Up = bQCmbKszFull + sBQCmbKszFull
Down = bQCmbKszFull - sBQCmbKszFull
ax3.fill_between(llMax, Down, Up, edgecolor='', facecolor=cKsz, alpha=0.4)
# radio point sources
ax3.plot(llMax, bQCmbRadiopsFull, cRadiops, label='Radiops')
Up = bQCmbRadiopsFull + sBQCmbRadiopsFull
Down = bQCmbRadiopsFull - sBQCmbRadiopsFull
ax3.fill_between(llMax, Down, Up, edgecolor='', facecolor=cRadiops, alpha=0.4)
# all
ax3.plot(llMax, bQCmbAllFull, cAll, label='All')
Up = bQCmbAllFull + sBQCmbAllFull
Down = bQCmbAllFull - sBQCmbAllFull
ax3.fill_between(llMax, Down, Up, edgecolor='', facecolor=cAll, alpha=0.4)
#
ax3.set_ylim((-0.025,0.025))
ax3.set_ylabel(r'QE', fontsize='x-small')
# remove last tick label for the second subplot
yticks = ax3.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)
#
ax3.set_xlim((llMax[0], llMax[-1]))
ax3.set_xlabel(r'$\ell_{\text{max, T}}$')

path = "output/summary_bias_full.pdf"
fig.savefig(path, bbox_inches='tight')
fig.clf()


##################################################################################
##################################################################################
# Interpolating to make color plots

llMaxc = np.arange(llMax[0], llMax[-1]+1., 1.)

# QE
# Auto
f = interp1d(llMax, bQCmbAllFull)
bQCmbAllFull = f(llMaxc)
f = interp1d(llMax, snrQCmb)
snrQCmb = f(llMaxc)
# Cross
f = interp1d(llMax, bQCmbAllLsstgold)
bQCmbAllLsstgold = f(llMaxc)
f = interp1d(llMax, 1./sQCmbLsstgold)
snrQCmbLsstgold = f(llMaxc)

# Shear
# Auto
f = interp1d(llMax, bSCmbAllFull)
bSCmbAllFull = f(llMaxc)
f = interp1d(llMax, snrSCmb)
snrSCmb = f(llMaxc)
# Cross
f = interp1d(llMax, bSCmbAllLsstgold)
bSCmbAllLsstgold = f(llMaxc)
f = interp1d(llMax, 1./sSCmbLsstgold)
snrSCmbLsstgold = f(llMaxc)

# Hybrid
# Auto
f = interp1d(llMax, bHybridCmbAllFull)
bHybridCmbAllFull = f(llMaxc)
f = interp1d(llMax, 1./sHybridCmb)
snrHybridCmb = f(llMaxc)
# Cross
f = interp1d(llMax, bHybridCmbAllLsstgold)
bHybridCmbAllLsstgold = f(llMaxc)
f = interp1d(llMax, 1./sHybridCmbLsstgold)
snrHybridCmbLsstgold = f(llMaxc)

# Bias-hardened
# Auto
f = interp1d(llMax, bBHCmbAllFull)
bBHCmbAllFull = f(llMaxc)
f = interp1d(llMax, snrBHCmb)
snrBHCmb = f(llMaxc)
# Cross
f = interp1d(llMax, bBHCmbAllLsstgold)
bBHCmbAllLsstgold = f(llMaxc)
f = interp1d(llMax, 1./sBHCmbLsstgold)
snrBHCmbLsstgold = f(llMaxc)

# Bias-hardened with Gaussian profile
# Auto
f = interp1d(llMax, bBHgCmbAllFull)
bBHgCmbAllFull = f(llMaxc)
f = interp1d(llMax, snrBHgCmb)
snrBHgCmb = f(llMaxc)
# Cross
f = interp1d(llMax, bBHgCmbAllLsstgold)
bBHgCmbAllLsstgold = f(llMaxc)
f = interp1d(llMax, 1./sBHgCmbLsstgold)
snrBHgCmbLsstgold = f(llMaxc)


##################################################################################
##################################################################################
# Plotting SNR for the C^kappa amplitude

fig=plt.figure(0)
ax=fig.add_subplot(111)

# Bias hardened
points = np.array([llMaxc, snrBHCmb]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
biasInSigmaUnits = np.abs(bBHCmbAllFull * snrBHCmb)
OneSigmaBias = np.where(np.abs(biasInSigmaUnits) >= 1.0)[0][0]
norm = plt.Normalize(0, 2.)
lc = LineCollection(segments, cmap='rainbow', norm=norm)
lc.set_array(biasInSigmaUnits)
lc.set_linewidth(3)
ax.add_collection(lc)
ax.scatter(llMaxc[OneSigmaBias], snrBHCmb[OneSigmaBias], c='k', marker='d', label='BH1',zorder=2)

# Bias hardened with Gaussian profile
points = np.array([llMaxc, snrBHgCmb]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
biasInSigmaUnits = np.abs(bBHgCmbAllFull * snrBHgCmb)
OneSigmaBias = np.where(np.abs(biasInSigmaUnits) >= 1.0)[0][0]
norm = plt.Normalize(0, 2.)
lc = LineCollection(segments, cmap='rainbow', norm=norm)
lc.set_array(biasInSigmaUnits)
lc.set_linewidth(3)
ax.add_collection(lc)
ax.scatter(llMaxc[OneSigmaBias], snrBHgCmb[OneSigmaBias], c='k', marker='*', label='BH2',zorder=2)

# hybrid estimator: QE(<2000) + shear(>2000)
points = np.array([llMaxc, snrHybridCmb]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
biasInSigmaUnits = np.abs(bHybridCmbAllFull * snrHybridCmb)
lc = LineCollection(segments, cmap='rainbow', norm=norm)
lc.set_array(biasInSigmaUnits)
lc.set_linewidth(3)
ax.add_collection(lc)
ax.scatter(llMaxc[-1], snrHybridCmb[-1], c='k', marker='s', label='Hybrid',zorder=2)

# Shear
points = np.array([llMaxc, snrSCmb]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
biasInSigmaUnits = np.abs(bSCmbAllFull * snrSCmb)
lc = LineCollection(segments, cmap='rainbow', norm=norm)
# Set the values used for colormapping
lc.set_array(biasInSigmaUnits)
lc.set_linewidth(3)
ax.add_collection(lc)
ax.scatter(llMaxc[-1], snrSCmb[-1], c='k', marker='^', label='Shear',zorder=2)

# QE
points = np.array([llMaxc, snrQCmb]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
biasInSigmaUnits = np.abs(bQCmbAllFull * snrQCmb)
OneSigmaBias = np.where(np.abs(biasInSigmaUnits) >= 1.0)[0][0]
lc = LineCollection(segments, cmap='rainbow', norm=norm)
lc.set_array(biasInSigmaUnits)
lc.set_linewidth(3)
line = ax.add_collection(lc)
ax.scatter(llMaxc[OneSigmaBias], snrQCmb[OneSigmaBias], c='k', marker='o', label='QE',zorder=2)

# GC
#ax.plot(llMax,snrGCCmb,c='k',lw=2,label='GC',zorder=1)

#colorbar
cb = fig.colorbar(line,ax=ax)
cb.set_label(r'Relative Bias [$\sigma$]')

# relevant tick marks
xticks = np.array([2000, 2500, 3000, 3500, 4000])
ax.set_xticks(xticks)
#
plt.ylim(25,160)
plt.xlim(llMax[0]-100.,llMax[-1]+100.)
ax.grid(alpha=0.3)
ax.set_xlabel(r'$\ell_\text{max,T}$')
ax.set_ylabel(r'SNR of $C^{\kappa}_L$ amplitude')
handles, labels = plt.gca().get_legend_handles_labels()
order = [1,2,3,4,0]
order = [0,1,2,3,4]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize='x-small', labelspacing=0.2,loc=2)
#ax.set_title(r'SNR of $C^{\kappa}_L$ amplitude')
#
path = "/home/noah/Documents/Berkeley/LensQuEst-1/output/compare_snr_lmax.pdf"
fig.savefig(path, bbox_inches='tight')
fig.clf()


##################################################################################
##################################################################################
# Plotting SNR for the C^kappa x LSST amplitude


fig=plt.figure(0)
ax=fig.add_subplot(111)

# Bias hardened
points = np.array([llMaxc, snrBHCmbLsstgold]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
biasInSigmaUnits = np.abs(bBHCmbAllLsstgold * snrBHCmbLsstgold)
OneSigmaBias = np.where(np.abs(biasInSigmaUnits) >= 1.0)[0][0]
norm = plt.Normalize(0, 2.)
lc = LineCollection(segments, cmap='rainbow', norm=norm)
lc.set_array(biasInSigmaUnits)
lc.set_linewidth(3)
ax.add_collection(lc)
ax.scatter(llMaxc[OneSigmaBias], snrBHCmbLsstgold[OneSigmaBias], c='k', marker='d', label='BH1',zorder=2)

# Bias hardened with Gaussian profile
points = np.array([llMaxc, snrBHgCmbLsstgold]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
biasInSigmaUnits = np.abs(bBHgCmbAllLsstgold * snrBHgCmbLsstgold)
norm = plt.Normalize(0, 2.)
lc = LineCollection(segments, cmap='rainbow', norm=norm)
lc.set_array(biasInSigmaUnits)
lc.set_linewidth(3)
ax.add_collection(lc)
ax.scatter(llMaxc[-1], snrBHgCmbLsstgold[-1], c='k', marker='*', label='BH2',zorder=2)

# hybrid estimator: QE(<2000) + shear(>2000)
points = np.array([llMaxc, snrHybridCmbLsstgold]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
biasInSigmaUnits = np.abs(bHybridCmbAllLsstgold * snrHybridCmbLsstgold)
lc = LineCollection(segments, cmap='rainbow', norm=norm)
lc.set_array(biasInSigmaUnits)
lc.set_linewidth(3)
ax.add_collection(lc)
ax.scatter(llMaxc[-1], snrHybridCmbLsstgold[-1], c='k', marker='s', label='Hybrid',zorder=2)

# Shear
points = np.array([llMaxc, snrSCmbLsstgold]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
biasInSigmaUnits = np.abs(bSCmbAllLsstgold * snrSCmbLsstgold)
OneSigmaBias = np.where(np.abs(biasInSigmaUnits) >= 1.0)[0][0]
lc = LineCollection(segments, cmap='rainbow', norm=norm)
# Set the values used for colormapping
lc.set_array(biasInSigmaUnits)
lc.set_linewidth(3)
ax.add_collection(lc)
ax.scatter(llMaxc[OneSigmaBias], snrSCmbLsstgold[OneSigmaBias], c='k', marker='^', label='Shear',zorder=2)

# QE
points = np.array([llMaxc, snrQCmbLsstgold]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
biasInSigmaUnits = np.abs(bQCmbAllLsstgold * snrQCmbLsstgold)
OneSigmaBias = np.where(np.abs(biasInSigmaUnits) >= 1.0)[0][0]
lc = LineCollection(segments, cmap='rainbow', norm=norm)
lc.set_array(biasInSigmaUnits)
lc.set_linewidth(3)
line = ax.add_collection(lc)
ax.scatter(llMaxc[OneSigmaBias], snrQCmbLsstgold[OneSigmaBias], c='k', marker='o', label='QE',zorder=2)

# GC
#ax.plot(llMax,snrGCCmbLsstgold,c='k',lw=2,label='GC',zorder=1)

#colorbar
cb = fig.colorbar(line,ax=ax)
cb.set_label(r'Relative Bias [$\sigma$]')

# relevant tick marks
xticks = np.array([2000, 2500, 3000, 3500, 4000])
ax.set_xticks(xticks)
#
plt.ylim(90,260)
plt.xlim(llMax[0]-100.,llMax[-1]+100.)
ax.grid(alpha=0.3)
ax.set_xlabel(r'$\ell_\text{max,T}$')
ax.set_ylabel(r'SNR of $C^{\kappa\times\text{ LSST }}_L$ amplitude')
handles, labels = plt.gca().get_legend_handles_labels()
order = [1,2,3,4,0]
order = [0,1,2,3,4]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize='x-small', labelspacing=0.2,loc=2)
#ax.set_title(r'SNR of $C^{\kappa\times\text{ LSST }}_L$ amplitude')
#
path = "/home/noah/Documents/Berkeley/LensQuEst-1/output/compare_snr_cross_lmax.pdf"
fig.savefig(path, bbox_inches='tight')
fig.clf()
