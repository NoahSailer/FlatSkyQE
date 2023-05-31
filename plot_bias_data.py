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

lll = np.genfromtxt('l.txt')
profile = np.genfromtxt('profile.txt')
uTsz = interp1d(lll,profile,kind='linear') 

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
print("- bias hardened estimator with profile")
fNqBHgCmb_fft = baseMap.forecastN0KappaBH(cmb.flensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False, cache=None, u=uTsz)
print("- bias hardened estimator point sources + profile")
fNqBH2Cmb_fft = baseMap.forecastN0KappaBH2(cmb.flensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False, cache=None, u=uTsz)


#################################################################################
cQ = 'r'
cS = 'g'
cD = 'g'
cBH = 'b'
cBHg = 'purple'
cBH2 = 'orange'

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
   ClqCmbRadiops = dataAvg[int(0+24*i)]
   ClsCmbRadiops = dataAvg[int(1+24*i)]
   CldCmbRadiops = dataAvg[int(2+24*i)]
   ClBHCmbRadiops = dataAvg[int(3+24*i)]
   ClBHgCmbRadiops = dataAvg[int(4+24*i)]
   ClBH2CmbRadiops = dataAvg[int(5+24*i)]
   # Primary
   ClqCmbRadiopsKappa = dataAvg[int(6+24*i)]
   ClsCmbRadiopsKappa = dataAvg[int(7+24*i)]
   CldCmbRadiopsKappa = dataAvg[int(8+24*i)]
   ClBHCmbRadiopsKappa = dataAvg[int(9+24*i)]
   ClBHgCmbRadiopsKappa = dataAvg[int(10+24*i)]
   ClBH2CmbRadiopsKappa = dataAvg[int(11+24*i)]
   # k_rec x LSST
   ClqCmbRadiopsLsstgold = dataAvg[int(12+24*i)]
   ClsCmbRadiopsLsstgold = dataAvg[int(13+24*i)]
   CldCmbRadiopsLsstgold = dataAvg[int(14+24*i)]
   ClBHCmbRadiopsLsstgold = dataAvg[int(15+24*i)]
   ClBHgCmbRadiopsLsstgold = dataAvg[int(16+24*i)]
   ClBH2CmbRadiopsLsstgold = dataAvg[int(17+24*i)]
   # Secondary
   ClqCmbRadiopsSec = dataAvg[int(18+24*i)]
   CldCmbRadiopsSec = dataAvg[int(19+24*i)]
   ClsCmbRadiopsSec = dataAvg[int(20+24*i)]
   ClBHCmbRadiopsSec = dataAvg[int(21+24*i)]
   ClBHgCmbRadiopsSec = dataAvg[int(22+24*i)]
   ClBH2CmbRadiopsSec = dataAvg[int(23+24*i)]

   i=1
   # CIB
   #
   # Trispectrum
   ClqCmbCib = dataAvg[int(0+24*i)]
   ClsCmbCib = dataAvg[int(1+24*i)]
   CldCmbCib = dataAvg[int(2+24*i)]
   ClBHCmbCib = dataAvg[int(3+24*i)]
   ClBHgCmbCib = dataAvg[int(4+24*i)]
   ClBH2CmbCib = dataAvg[int(5+24*i)]
   # Primary
   ClqCmbCibKappa = dataAvg[int(6+24*i)]
   ClsCmbCibKappa = dataAvg[int(7+24*i)]
   CldCmbCibKappa = dataAvg[int(8+24*i)]
   ClBHCmbCibKappa = dataAvg[int(9+24*i)]
   ClBHgCmbCibKappa = dataAvg[int(10+24*i)]
   ClBH2CmbCibKappa = dataAvg[int(11+24*i)]
   # k_rec x LSST
   ClqCmbCibLsstgold = dataAvg[int(12+24*i)]
   ClsCmbCibLsstgold = dataAvg[int(13+24*i)]
   CldCmbCibLsstgold = dataAvg[int(14+24*i)]
   ClBHCmbCibLsstgold = dataAvg[int(15+24*i)]
   ClBHgCmbCibLsstgold = dataAvg[int(16+24*i)]
   ClBH2CmbCibLsstgold = dataAvg[int(17+24*i)]
   # Secondary
   ClqCmbCibSec = dataAvg[int(18+24*i)]
   CldCmbCibSec = dataAvg[int(19+24*i)]
   ClsCmbCibSec = dataAvg[int(20+24*i)]
   ClBHCmbCibSec = dataAvg[int(21+24*i)]
   ClBHgCmbCibSec = dataAvg[int(22+24*i)]
   ClBH2CmbCibSec = dataAvg[int(23+24*i)]

   i=2
   # tSZ
   #
   # Trispectrum
   ClqCmbTsz = dataAvg[int(0+24*i)]
   ClsCmbTsz = dataAvg[int(1+24*i)]
   CldCmbTsz = dataAvg[int(2+24*i)]
   ClBHCmbTsz = dataAvg[int(3+24*i)]
   ClBHgCmbTsz = dataAvg[int(4+24*i)]
   ClBH2CmbTsz = dataAvg[int(5+24*i)]
   # Primary
   ClqCmbTszKappa = dataAvg[int(6+24*i)]
   ClsCmbTszKappa = dataAvg[int(7+24*i)]
   CldCmbTszKappa = dataAvg[int(8+24*i)]
   ClBHCmbTszKappa = dataAvg[int(9+24*i)]
   ClBHgCmbTszKappa = dataAvg[int(10+24*i)]
   ClBH2CmbTszKappa = dataAvg[int(11+24*i)]
   # k_rec x LSST
   ClqCmbTszLsstgold = dataAvg[int(12+24*i)]
   ClsCmbTszLsstgold = dataAvg[int(13+24*i)]
   CldCmbTszLsstgold = dataAvg[int(14+24*i)]
   ClBHCmbTszLsstgold = dataAvg[int(15+24*i)]
   ClBHgCmbTszLsstgold = dataAvg[int(16+24*i)]
   ClBH2CmbTszLsstgold = dataAvg[int(17+24*i)]
   # Secondary
   ClqCmbTszSec = dataAvg[int(18+24*i)]
   CldCmbTszSec = dataAvg[int(19+24*i)]
   ClsCmbTszSec = dataAvg[int(20+24*i)]
   ClBHCmbTszSec = dataAvg[int(21+24*i)]
   ClBHgCmbTszSec = dataAvg[int(22+24*i)]
   ClBH2CmbTszSec = dataAvg[int(23+24*i)]

   i=3
   # kSZ
   # 
   # Trispectrum
   ClqCmbKsz = dataAvg[int(0+24*i)]
   ClsCmbKsz = dataAvg[int(1+24*i)]
   CldCmbKsz = dataAvg[int(2+24*i)]
   ClBHCmbKsz = dataAvg[int(3+24*i)]
   ClBHgCmbKsz = dataAvg[int(4+24*i)]
   ClBH2CmbKsz = dataAvg[int(5+24*i)]
   # Primary
   ClqCmbKszKappa = dataAvg[int(6+24*i)]
   ClsCmbKszKappa = dataAvg[int(7+24*i)]
   CldCmbKszKappa = dataAvg[int(8+24*i)]
   ClBHCmbKszKappa = dataAvg[int(9+24*i)]
   ClBHgCmbKszKappa = dataAvg[int(10+24*i)]
   ClBH2CmbKszKappa = dataAvg[int(11+24*i)]
   # k_rec x LSST 
   ClqCmbKszLsstgold = dataAvg[int(12+24*i)]
   ClsCmbKszLsstgold = dataAvg[int(13+24*i)]
   CldCmbKszLsstgold = dataAvg[int(14+24*i)]
   ClBHCmbKszLsstgold = dataAvg[int(15+24*i)]
   ClBHgCmbKszLsstgold = dataAvg[int(16+24*i)]
   ClBH2CmbKszLsstgold = dataAvg[int(17+24*i)]
   # Secondary
   ClqCmbKszSec = dataAvg[int(18+24*i)]
   CldCmbKszSec = dataAvg[int(19+24*i)]
   ClsCmbKszSec = dataAvg[int(20+24*i)]
   ClBHCmbKszSec = dataAvg[int(21+24*i)]
   ClBHgCmbKszSec = dataAvg[int(22+24*i)]
   ClBH2CmbKszSec = dataAvg[int(23+24*i)]

   i=4
   # Dust
   # 
   # Trispectrum
   ClqCmbDust = dataAvg[int(0+24*i)]
   ClsCmbDust = dataAvg[int(1+24*i)]
   CldCmbDust = dataAvg[int(2+24*i)]
   ClBHCmbDust = dataAvg[int(3+24*i)]
   ClBHgCmbDust = dataAvg[int(4+24*i)]
   ClBH2CmbDust = dataAvg[int(5+24*i)]
   # Primary
   ClqCmbDustKappa = dataAvg[int(6+24*i)]
   ClsCmbDustKappa = dataAvg[int(7+24*i)]
   CldCmbDustKappa = dataAvg[int(8+24*i)]
   ClBHCmbDustKappa = dataAvg[int(9+24*i)]
   ClBHgCmbDustKappa = dataAvg[int(10+24*i)]
   ClBH2CmbDustKappa = dataAvg[int(11+24*i)]
   # k_rec x LSST 
   ClqCmbDustLsstgold = dataAvg[int(12+24*i)]
   ClsCmbDustLsstgold = dataAvg[int(13+24*i)]
   CldCmbDustLsstgold = dataAvg[int(14+24*i)]
   ClBHCmbDustLsstgold = dataAvg[int(15+24*i)]
   ClBHgCmbDustLsstgold = dataAvg[int(16+24*i)]
   ClBH2CmbDustLsstgold = dataAvg[int(17+24*i)]
   # Secondary
   ClqCmbDustSec = dataAvg[int(18+24*i)]
   CldCmbDustSec = dataAvg[int(19+24*i)]
   ClsCmbDustSec = dataAvg[int(20+24*i)]
   ClBHCmbDustSec = dataAvg[int(21+24*i)]
   ClBHgCmbDustSec = dataAvg[int(22+24*i)]
   ClBH2CmbDustSec = dataAvg[int(23+24*i)]

   i=5
   # FreeFree
   # 
   # Trispectrum
   ClqCmbFreeFree = dataAvg[int(0+24*i)]
   ClsCmbFreeFree = dataAvg[int(1+24*i)]
   CldCmbFreeFree = dataAvg[int(2+24*i)]
   ClBHCmbFreeFree = dataAvg[int(3+24*i)]
   ClBHgCmbFreeFree = dataAvg[int(4+24*i)]
   ClBH2CmbFreeFree = dataAvg[int(5+24*i)]
   # Primary
   ClqCmbFreeFreeKappa = dataAvg[int(6+24*i)]
   ClsCmbFreeFreeKappa = dataAvg[int(7+24*i)]
   CldCmbFreeFreeKappa = dataAvg[int(8+24*i)]
   ClBHCmbFreeFreeKappa = dataAvg[int(9+24*i)]
   ClBHgCmbFreeFreeKappa = dataAvg[int(10+24*i)]
   ClBH2CmbFreeFreeKappa = dataAvg[int(11+24*i)]
   # k_rec x LSST 
   ClqCmbFreeFreeLsstgold = dataAvg[int(12+24*i)]
   ClsCmbFreeFreeLsstgold = dataAvg[int(13+24*i)]
   CldCmbFreeFreeLsstgold = dataAvg[int(14+24*i)]
   ClBHCmbFreeFreeLsstgold = dataAvg[int(15+24*i)]
   ClBHgCmbFreeFreeLsstgold = dataAvg[int(16+24*i)]
   ClBH2CmbFreeFreeLsstgold = dataAvg[int(17+24*i)]
   # Secondary
   ClqCmbFreeFreeSec = dataAvg[int(18+24*i)]
   CldCmbFreeFreeSec = dataAvg[int(19+24*i)]
   ClsCmbFreeFreeSec = dataAvg[int(20+24*i)]
   ClBHCmbFreeFreeSec = dataAvg[int(21+24*i)]
   ClBHgCmbFreeFreeSec = dataAvg[int(22+24*i)]
   ClBH2CmbFreeFreeSec = dataAvg[int(23+24*i)]

   i=6
   # Sync
   # 
   # Trispectrum
   ClqCmbSync = dataAvg[int(0+24*i)]
   ClsCmbSync = dataAvg[int(1+24*i)]
   CldCmbSync = dataAvg[int(2+24*i)]
   ClBHCmbSync = dataAvg[int(3+24*i)]
   ClBHgCmbSync = dataAvg[int(4+24*i)]
   ClBH2CmbSync = dataAvg[int(5+24*i)]
   # Primary
   ClqCmbSyncKappa = dataAvg[int(6+24*i)]
   ClsCmbSyncKappa = dataAvg[int(7+24*i)]
   CldCmbSyncKappa = dataAvg[int(8+24*i)]
   ClBHCmbSyncKappa = dataAvg[int(9+24*i)]
   ClBHgCmbSyncKappa = dataAvg[int(10+24*i)]
   ClBH2CmbSyncKappa = dataAvg[int(11+24*i)]
   # k_rec x LSST 
   ClqCmbSyncLsstgold = dataAvg[int(12+24*i)]
   ClsCmbSyncLsstgold = dataAvg[int(13+24*i)]
   CldCmbSyncLsstgold = dataAvg[int(14+24*i)]
   ClBHCmbSyncLsstgold = dataAvg[int(15+24*i)]
   ClBHgCmbSyncLsstgold = dataAvg[int(16+24*i)]
   ClBH2CmbSyncLsstgold = dataAvg[int(17+24*i)]
   # Secondary
   ClqCmbSyncSec = dataAvg[int(18+24*i)]
   CldCmbSyncSec = dataAvg[int(19+24*i)]
   ClsCmbSyncSec = dataAvg[int(20+24*i)]
   ClBHCmbSyncSec = dataAvg[int(21+24*i)]
   ClBHgCmbSyncSec = dataAvg[int(22+24*i)]
   ClBH2CmbSyncSec = dataAvg[int(23+24*i)]

   i=7
   # All extragalactic foregrounds
   # 
   # Trispectrum
   ClqCmbAll = dataAvg[int(0+24*i)]
   ClsCmbAll = dataAvg[int(1+24*i)]
   CldCmbAll = dataAvg[int(2+24*i)]
   ClBHCmbAll = dataAvg[int(3+24*i)]
   ClBHgCmbAll = dataAvg[int(4+24*i)]
   ClBH2CmbAll = dataAvg[int(5+24*i)]
   # Primary
   ClqCmbAllKappa = dataAvg[int(6+24*i)]
   ClsCmbAllKappa = dataAvg[int(7+24*i)]
   CldCmbAllKappa = dataAvg[int(8+24*i)]
   ClBHCmbAllKappa = dataAvg[int(9+24*i)]
   ClBHgCmbAllKappa = dataAvg[int(10+24*i)]
   ClBH2CmbAllKappa = dataAvg[int(11+24*i)]
   # k_rec x LSST 
   ClqCmbAllLsstgold = dataAvg[int(12+24*i)]
   ClsCmbAllLsstgold = dataAvg[int(13+24*i)]
   CldCmbAllLsstgold = dataAvg[int(14+24*i)]
   ClBHCmbAllLsstgold = dataAvg[int(15+24*i)]
   ClBHgCmbAllLsstgold = dataAvg[int(16+24*i)]
   ClBH2CmbAllLsstgold = dataAvg[int(17+24*i)]
   # Secondary
   ClqCmbAllSec = dataAvg[int(18+24*i)]
   CldCmbAllSec = dataAvg[int(19+24*i)]
   ClsCmbAllSec = dataAvg[int(20+24*i)]
   ClBHCmbAllSec = dataAvg[int(21+24*i)]
   ClBHgCmbAllSec = dataAvg[int(22+24*i)]
   ClBH2CmbAllSec = dataAvg[int(22+24*i)]
   
   
   bias_cross_list = np.array([
   ClqCmbCibLsstgold,     # QE
   ClsCmbCibLsstgold,     # Shear
   ClBHCmbCibLsstgold,    # PSH
   ClBHgCmbCibLsstgold,   # PH
   ClBH2CmbCibLsstgold,   # PPH
   ClqCmbTszLsstgold,
   ClsCmbTszLsstgold,
   ClBHCmbTszLsstgold, 
   ClBHgCmbTszLsstgold, 
   ClBH2CmbTszLsstgold, 
   ClqCmbKszLsstgold,
   ClsCmbKszLsstgold,
   ClBHCmbKszLsstgold, 
   ClBHgCmbKszLsstgold, 
   ClBH2CmbKszLsstgold, 
   ClqCmbRadiopsLsstgold,
   ClsCmbRadiopsLsstgold,
   ClBHCmbRadiopsLsstgold, 
   ClBHgCmbRadiopsLsstgold, 
   ClBH2CmbRadiopsLsstgold
   ])
    
   labels_cross = np.array([
   'QE: CIB',
   'Shear: CIB',
   'PSH: CIB',
   'PH: CIB',
   'PPH: CIB', 
   'QE: tSZ',
   'Shear: tSZ',
   'PSH: tSZ',
   'PH: tSZ',
   'PPH: tSZ',
   'QE: kSZ',
   'Shear: kSZ',
   'PSH: kSZ',
   'PH: kSZ',
   'PPH: kSZ',
   'QE: Radio PS',
   'Shear: Radio PS',
   'PSH: Radio PS',
   'PH: Radio PS',
   'PPH: Radio PS'
   ])



   bias_list = np.array([
   lCen,
   Nmodes,
   ClqCmbCibKappa,
   ClBHCmbCibKappa,
   ClBHgCmbCibKappa,
   ClBH2CmbCibKappa,
   ClsCmbCibKappa,
   ClqCmbTszKappa,
   ClBHCmbTszKappa,
   ClBHgCmbTszKappa,
   ClBH2CmbTszKappa,
   ClsCmbTszKappa,
   ClqCmbKszKappa,
   ClBHCmbKszKappa,
   ClBHgCmbKszKappa,
   ClBH2CmbKszKappa,
   ClsCmbKszKappa,
   ClqCmbRadiopsKappa,
   ClBHCmbRadiopsKappa,
   ClBHgCmbRadiopsKappa,
   ClBH2CmbRadiopsKappa,
   ClsCmbRadiopsKappa,
   ClqCmbAllKappa,
   ClBHCmbAllKappa,
   ClBHgCmbAllKappa,
   ClBH2CmbAllKappa,
   ClsCmbAllKappa,
   ClqCmbCibSec,
   ClBHCmbCibSec,
   ClBHgCmbCibSec,
   ClBH2CmbCibSec,
   ClsCmbCibSec,
   ClqCmbTszSec,
   ClBHCmbTszSec,
   ClBHgCmbTszSec,
   ClBH2CmbTszSec,
   ClsCmbTszSec,
   ClqCmbKszSec,
   ClBHCmbKszSec,
   ClBHgCmbKszSec,
   ClBH2CmbKszSec,
   ClsCmbKszSec,
   ClqCmbRadiopsSec,
   ClBHCmbRadiopsSec,
   ClBHgCmbRadiopsSec,
   ClBH2CmbRadiopsSec,
   ClsCmbRadiopsSec,
   ClqCmbAllSec,
   ClBHCmbAllSec,
   ClBHgCmbAllSec,
   ClBH2CmbAllSec,
   ClsCmbAllSec,
   ClqCmbCib,
   ClBHCmbCib,
   ClBHgCmbCib,
   ClBH2CmbCib,
   ClsCmbCib,
   ClqCmbTsz,
   ClBHCmbTsz,
   ClBHgCmbTsz,
   ClBH2CmbTsz,
   ClsCmbTsz,
   ClqCmbKsz,
   ClBHCmbKsz,
   ClBHgCmbKsz,
   ClBH2CmbKsz,
   ClsCmbKsz,
   ClqCmbRadiops,
   ClBHCmbRadiops,
   ClBHgCmbRadiops,
   ClBH2CmbRadiops,
   ClsCmbRadiops,
   ClqCmbAll,
   ClBHCmbAll,
   ClBHgCmbAll,
   ClBH2CmbAll,
   ClsCmbAll])

   headers = np.array([
   'lCen',
   'Nmodes',
   'QE primary, CIB',
   'PSH primary, CIB',
   'PH primary, CIB',
   'PPH primary, CIB',
   'Shear primary, CIB',
   'QE primary, tSZ',
   'PSH primary, tSZ',
   'PH primary, tSZ',
   'PPH primary, tSZ',
   'Shear primary, tSZ',
   'QE primary, kSZ',
   'PSH primary, kSZ',
   'PH primary, kSZ',
   'PPH primary, kSZ',
   'Shear primary, kSZ',
   'QE primary, Radio PS',
   'PSH primary, Radio PS',
   'PH primary, Radio PS',
   'PPH primary, Radio PS',
   'Shear primary, Radio PS',
   'QE primary, All',
   'PSH primary, All',
   'PH primary, All',
   'PPH primary, All',
   'Shear primary, All',
   'QE secondary, CIB',
   'PSH secondary, CIB',
   'PH secondary, CIB',
   'PPH secondary, CIB',
   'Shear secondary, CIB',
   'QE secondary, tSZ',
   'PSH secondary, tSZ',
   'PH secondary, tSZ',
   'PPH secondary, tSZ',
   'Shear secondary, tSZ',
   'QE secondary, kSZ',
   'PSH secondary, kSZ',
   'PH secondary, kSZ',
   'PPH secondary, kSZ',
   'Shear secondary, kSZ',
   'QE secondary, Radio PS',
   'PSH secondary, Radio PS',
   'PH secondary, Radio PS',
   'PPH secondary, Radio PS',
   'Shear secondary, Radio PS',
   'QE secondary, All',
   'PSH secondary, All',
   'PH secondary, All',
   'PPH secondary, All',
   'Shear secondary, All',
   'QE trispectrum, CIB',
   'PSH trispectrum, CIB',
   'PH trispectrum, CIB',
   'PPH trispectrum, CIB',
   'Shear trispectrum, CIB',
   'QE trispectrum, tSZ',
   'PSH trispectrum, tSZ',
   'PH trispectrum, tSZ',
   'PPH trispectrum, tSZ',
   'Shear trispectrum, tSZ',
   'QE trispectrum, kSZ',
   'PSH trispectrum, kSZ',
   'PH trispectrum, kSZ',
   'PPH trispectrum, kSZ',
   'Shear trispectrum, kSZ',
   'QE trispectrum, Radio PS',
   'PSH trispectrum, Radio PS',
   'PH trispectrum, Radio PS',
   'PPH trispectrum, Radio PS',
   'Shear trispectrum, Radio PS',
   'QE trispectrum, All',
   'PSH trispectrum, All',
   'PH trispectrum, All',
   'PPH trispectrum, All',
   'Shear trispectrum, All'])

   import pandas as pd
   res = {headers[i]:bias_list[i] for i in range(len(headers))}
   res2 = {labels_cross[i]:bias_cross_list[i] for i in range(len(labels_cross))}
   df = pd.DataFrame(res)
   df2 = pd.DataFrame(res2)
   df.to_csv("output/Ckk_biases_lmaxT_"+str(int(lMax))+".csv")
   df2.to_csv("output/Ckg_biases_lmaxT_"+str(int(lMax))+".csv")

   import sys
   sys.exit()
   

   ##################################################################################
   # Statistical errors   

   i=0
   # Radio point sources
   #
   # Trispectrum
   sClqCmbRadiops = dataStd[int(0+24*i)]
   sClsCmbRadiops = dataStd[int(1+24*i)]
   sCldCmbRadiops = dataStd[int(2+24*i)]
   sClBHCmbRadiops = dataStd[int(3+24*i)]
   sClBHgCmbRadiops = dataStd[int(4+24*i)]
   sClBH2CmbRadiops = dataStd[int(5+24*i)]
   # Primary
   sClqCmbRadiopsKappa = dataStd[int(6+24*i)]
   sClsCmbRadiopsKappa = dataStd[int(7+24*i)]
   sCldCmbRadiopsKappa = dataStd[int(8+24*i)]
   sClBHCmbRadiopsKappa = dataStd[int(9+24*i)]
   sClBHgCmbRadiopsKappa = dataStd[int(10+24*i)]
   sClBH2CmbRadiopsKappa = dataStd[int(11+24*i)]
   # k_rec x LSST
   sClqCmbRadiopsLsstgold = dataStd[int(12+24*i)]
   sClsCmbRadiopsLsstgold = dataStd[int(13+24*i)]
   sCldCmbRadiopsLsstgold = dataStd[int(14+24*i)]
   sClBHCmbRadiopsLsstgold = dataStd[int(15+24*i)]
   sClBHgCmbRadiopsLsstgold = dataStd[int(16+24*i)]
   sClBH2CmbRadiopsLsstgold = dataStd[int(17+24*i)]
   # Secondary
   sClqCmbRadiopsSec = dataStd[int(18+24*i)]
   sCldCmbRadiopsSec = dataStd[int(19+24*i)]
   sClsCmbRadiopsSec = dataStd[int(20+24*i)]
   sClBHCmbRadiopsSec = dataStd[int(21+24*i)]
   sClBHgCmbRadiopsSec = dataStd[int(22+24*i)]
   sClBH2CmbRadiopsSec = dataStd[int(23+24*i)]

   i=1
   # CIB
   #
   # Trispectrum
   sClqCmbCib = dataStd[int(0+24*i)]
   sClsCmbCib = dataStd[int(1+24*i)]
   sCldCmbCib = dataStd[int(2+24*i)]
   sClBHCmbCib = dataStd[int(3+24*i)]
   sClBHgCmbCib = dataStd[int(4+24*i)]
   sClBH2CmbCib = dataStd[int(5+24*i)]
   # Primary
   sClqCmbCibKappa = dataStd[int(6+24*i)]
   sClsCmbCibKappa = dataStd[int(7+24*i)]
   sCldCmbCibKappa = dataStd[int(8+24*i)]
   sClBHCmbCibKappa = dataStd[int(9+24*i)]
   sClBHgCmbCibKappa = dataStd[int(10+24*i)]
   sClBH2CmbCibKappa = dataStd[int(11+24*i)]
   # k_rec x LSST
   sClqCmbCibLsstgold = dataStd[int(12+24*i)]
   sClsCmbCibLsstgold = dataStd[int(13+24*i)]
   sCldCmbCibLsstgold = dataStd[int(14+24*i)]
   sClBHCmbCibLsstgold = dataStd[int(15+24*i)]
   sClBHgCmbCibLsstgold = dataStd[int(16+24*i)]
   sClBH2CmbCibLsstgold = dataStd[int(17+24*i)]
   # Secondary
   sClqCmbCibSec = dataStd[int(18+24*i)]
   sCldCmbCibSec = dataStd[int(19+24*i)]
   sClsCmbCibSec = dataStd[int(20+24*i)]
   sClBHCmbCibSec = dataStd[int(21+24*i)]
   sClBHgCmbCibSec = dataStd[int(22+24*i)]
   sClBH2CmbCibSec = dataStd[int(23+24*i)]

   i=2
   # tSZ
   #
   # Trispectrum
   sClqCmbTsz = dataStd[int(0+24*i)]
   sClsCmbTsz = dataStd[int(1+24*i)]
   sCldCmbTsz = dataStd[int(2+24*i)]
   sClBHCmbTsz = dataStd[int(3+24*i)]
   sClBHgCmbTsz = dataStd[int(4+24*i)]
   sClBH2CmbTsz = dataStd[int(5+24*i)]
   # Primary
   sClqCmbTszKappa = dataStd[int(6+24*i)]
   sClsCmbTszKappa = dataStd[int(7+24*i)]
   sCldCmbTszKappa = dataStd[int(8+24*i)]
   sClBHCmbTszKappa = dataStd[int(9+24*i)]
   sClBHgCmbTszKappa = dataStd[int(10+24*i)]
   sClBH2CmbTszKappa = dataStd[int(11+24*i)]
   # k_rec x LSST
   sClqCmbTszLsstgold = dataStd[int(12+24*i)]
   sClsCmbTszLsstgold = dataStd[int(13+24*i)]
   sCldCmbTszLsstgold = dataStd[int(14+24*i)]
   sClBHCmbTszLsstgold = dataStd[int(15+24*i)]
   sClBHgCmbTszLsstgold = dataStd[int(16+24*i)]
   sClBH2CmbTszLsstgold = dataStd[int(17+24*i)]
   # Secondary
   sClqCmbTszSec = dataStd[int(18+24*i)]
   sCldCmbTszSec = dataStd[int(19+24*i)]
   sClsCmbTszSec = dataStd[int(20+24*i)]
   sClBHCmbTszSec = dataStd[int(21+24*i)]
   sClBHgCmbTszSec = dataStd[int(22+24*i)]
   sClBH2CmbTszSec = dataStd[int(23+24*i)]

   i=3
   # kSZ
   # 
   # Trispectrum
   sClqCmbKsz = dataStd[int(0+24*i)]
   sClsCmbKsz = dataStd[int(1+24*i)]
   sCldCmbKsz = dataStd[int(2+24*i)]
   sClBHCmbKsz = dataStd[int(3+24*i)]
   sClBHgCmbKsz = dataStd[int(4+24*i)]
   sClBH2CmbKsz = dataStd[int(5+24*i)]
   # Primary
   sClqCmbKszKappa = dataStd[int(6+24*i)]
   sClsCmbKszKappa = dataStd[int(7+24*i)]
   sCldCmbKszKappa = dataStd[int(8+24*i)]
   sClBHCmbKszKappa = dataStd[int(9+24*i)]
   sClBHgCmbKszKappa = dataStd[int(10+24*i)]
   sClBH2CmbKszKappa = dataStd[int(11+24*i)]
   # k_rec x LSST 
   sClqCmbKszLsstgold = dataStd[int(12+24*i)]
   sClsCmbKszLsstgold = dataStd[int(13+24*i)]
   sCldCmbKszLsstgold = dataStd[int(14+24*i)]
   sClBHCmbKszLsstgold = dataStd[int(15+24*i)]
   sClBHgCmbKszLsstgold = dataStd[int(16+24*i)]
   sClBH2CmbKszLsstgold = dataStd[int(17+24*i)]
   # Secondary
   sClqCmbKszSec = dataStd[int(18+24*i)]
   sCldCmbKszSec = dataStd[int(19+24*i)]
   sClsCmbKszSec = dataStd[int(20+24*i)]
   sClBHCmbKszSec = dataStd[int(21+24*i)]
   sClBHgCmbKszSec = dataStd[int(22+24*i)]
   sClBH2CmbKszSec = dataStd[int(23+24*i)]

   i=4
   # Dust
   # 
   # Trispectrum
   sClqCmbDust = dataStd[int(0+24*i)]
   sClsCmbDust = dataStd[int(1+24*i)]
   sCldCmbDust = dataStd[int(2+24*i)]
   sClBHCmbDust = dataStd[int(3+24*i)]
   sClBHgCmbDust = dataStd[int(4+24*i)]
   sClBH2CmbDust = dataStd[int(5+24*i)]
   # Primary
   sClqCmbDustKappa = dataStd[int(6+24*i)]
   sClsCmbDustKappa = dataStd[int(7+24*i)]
   sCldCmbDustKappa = dataStd[int(8+24*i)]
   sClBHCmbDustKappa = dataStd[int(9+24*i)]
   sClBHgCmbDustKappa = dataStd[int(10+24*i)]
   sClBH2CmbDustKappa = dataStd[int(11+24*i)]
   # k_rec x LSST 
   sClqCmbDustLsstgold = dataStd[int(12+24*i)]
   sClsCmbDustLsstgold = dataStd[int(13+24*i)]
   sCldCmbDustLsstgold = dataStd[int(14+24*i)]
   sClBHCmbDustLsstgold = dataStd[int(15+24*i)]
   sClBHgCmbDustLsstgold = dataStd[int(16+24*i)]
   sClBH2CmbDustLsstgold = dataStd[int(17+24*i)]
   # Secondary
   sClqCmbDustSec = dataStd[int(18+24*i)]
   sCldCmbDustSec = dataStd[int(19+24*i)]
   sClsCmbDustSec = dataStd[int(20+24*i)]
   sClBHCmbDustSec = dataStd[int(21+24*i)]
   sClBHgCmbDustSec = dataStd[int(22+24*i)]
   sClBH2CmbDustSec = dataStd[int(23+24*i)]

   i=5
   # FreeFree
   # 
   # Trispectrum
   sClqCmbFreeFree = dataStd[int(0+24*i)]
   sClsCmbFreeFree = dataStd[int(1+24*i)]
   sCldCmbFreeFree = dataStd[int(2+24*i)]
   sClBHCmbFreeFree = dataStd[int(3+24*i)]
   sClBHgCmbFreeFree = dataStd[int(4+24*i)]
   sClBH2CmbFreeFree = dataStd[int(5+24*i)]
   # Primary
   sClqCmbFreeFreeKappa = dataStd[int(6+24*i)]
   sClsCmbFreeFreeKappa = dataStd[int(7+24*i)]
   sCldCmbFreeFreeKappa = dataStd[int(8+24*i)]
   sClBHCmbFreeFreeKappa = dataStd[int(9+24*i)]
   sClBHgCmbFreeFreeKappa = dataStd[int(10+24*i)]
   sClBH2CmbFreeFreeKappa = dataStd[int(11+24*i)]
   # k_rec x LSST 
   sClqCmbFreeFreeLsstgold = dataStd[int(12+24*i)]
   sClsCmbFreeFreeLsstgold = dataStd[int(13+24*i)]
   sCldCmbFreeFreeLsstgold = dataStd[int(14+24*i)]
   sClBHCmbFreeFreeLsstgold = dataStd[int(15+24*i)]
   sClBHgCmbFreeFreeLsstgold = dataStd[int(16+24*i)]
   sClBH2CmbFreeFreeLsstgold = dataStd[int(17+24*i)]
   # Secondary
   sClqCmbFreeFreeSec = dataStd[int(18+24*i)]
   sCldCmbFreeFreeSec = dataStd[int(19+24*i)]
   sClsCmbFreeFreeSec = dataStd[int(20+24*i)]
   sClBHCmbFreeFreeSec = dataStd[int(21+24*i)]
   sClBHgCmbFreeFreeSec = dataStd[int(22+24*i)]
   sClBH2CmbFreeFreeSec = dataStd[int(23+24*i)]

   i=6
   # Sync
   # 
   # Trispectrum
   sClqCmbSync = dataStd[int(0+24*i)]
   sClsCmbSync = dataStd[int(1+24*i)]
   sCldCmbSync = dataStd[int(2+24*i)]
   sClBHCmbSync = dataStd[int(3+24*i)]
   sClBHgCmbSync = dataStd[int(4+24*i)]
   sClBH2CmbSync = dataStd[int(5+24*i)]
   # Primary
   sClqCmbSyncKappa = dataStd[int(6+24*i)]
   sClsCmbSyncKappa = dataStd[int(7+24*i)]
   sCldCmbSyncKappa = dataStd[int(8+24*i)]
   sClBHCmbSyncKappa = dataStd[int(9+24*i)]
   sClBHgCmbSyncKappa = dataStd[int(10+24*i)]
   sClBH2CmbSyncKappa = dataStd[int(11+24*i)]
   # k_rec x LSST 
   sClqCmbSynClsstgold = dataStd[int(12+24*i)]
   sClsCmbSynClsstgold = dataStd[int(13+24*i)]
   sCldCmbSynClsstgold = dataStd[int(14+24*i)]
   sClBHCmbSynClsstgold = dataStd[int(15+24*i)]
   sClBHgCmbSynClsstgold = dataStd[int(16+24*i)]
   sClBH2CmbSynClsstgold = dataStd[int(17+24*i)]
   # Secondary
   sClqCmbSyncSec = dataStd[int(18+24*i)]
   sCldCmbSyncSec = dataStd[int(19+24*i)]
   sClsCmbSyncSec = dataStd[int(20+24*i)]
   sClBHCmbSyncSec = dataStd[int(21+24*i)]
   sClBHgCmbSyncSec = dataStd[int(22+24*i)]
   sClBH2CmbSyncSec = dataStd[int(23+24*i)]

   i=7
   # All extragalactic foregrounds
   # 
   # Trispectrum
   sClqCmbAll = dataStd[int(0+24*i)]
   sClsCmbAll = dataStd[int(1+24*i)]
   sCldCmbAll = dataStd[int(2+24*i)]
   sClBHCmbAll = dataStd[int(3+24*i)]
   sClBHgCmbAll = dataStd[int(4+24*i)]
   sClBH2CmbAll = dataStd[int(5+24*i)]
   # Primary
   sClqCmbAllKappa = dataStd[int(6+24*i)]
   sClsCmbAllKappa = dataStd[int(7+24*i)]
   sCldCmbAllKappa = dataStd[int(8+24*i)]
   sClBHCmbAllKappa = dataStd[int(9+24*i)]
   sClBHgCmbAllKappa = dataStd[int(10+24*i)]
   sClBH2CmbAllKappa = dataStd[int(11+24*i)]
   # k_rec x LSST 
   sClqCmbAllLsstgold = dataStd[int(12+24*i)]
   sClsCmbAllLsstgold = dataStd[int(13+24*i)]
   sCldCmbAllLsstgold = dataStd[int(14+24*i)]
   sClBHCmbAllLsstgold = dataStd[int(15+24*i)]
   sClBHgCmbAllLsstgold = dataStd[int(16+24*i)]
   sClBH2CmbAllLsstgold = dataStd[int(17+24*i)]
   # Secondary
   sClqCmbAllSec = dataStd[int(18+24*i)]
   sCldCmbAllSec = dataStd[int(19+24*i)]
   sClsCmbAllSec = dataStd[int(20+24*i)]
   sClBHCmbAllSec = dataStd[int(21+24*i)]
   sClBHgCmbAllSec = dataStd[int(22+24*i)]
   sClBH2CmbAllSec = dataStd[int(22+24*i)]


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
   # Bias hardened with profile
   ax0.plot(lCen, 2. * ClBHgCmbCibKappa/ClkCmb, color=cBHg, ls='-')
   Up = ClBHgCmbCibKappa + sClBHgCmbCibKappa
   Down = ClBHgCmbCibKappa - sClBHgCmbCibKappa
   ax0.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cBHg, alpha=0.6)
   # Bias hardened point sources + profile
   ax0.plot(lCen, 2. * ClBH2CmbCibKappa/ClkCmb, color=cBH2, ls='-')
   Up = ClBH2CmbCibKappa + sClBH2CmbCibKappa
   Down = ClBH2CmbCibKappa - sClBH2CmbCibKappa
   ax0.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cBH2, alpha=0.6)
   # Shear
   ax0.plot(lCen, 2. * ClsCmbCibKappa/ClkCmb, color=cS, ls='-')
   Up = ClsCmbCibKappa + sClsCmbCibKappa
   Down = ClsCmbCibKappa - sClsCmbCibKappa
   ax0.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cS, alpha=0.6)
   #
#   ax0.set_ylim((-0.2,0.1))
   ax0.set_ylim((-0.1, 0.15))
   ax0.set_ylabel(r'CIB')
   #
   ax0.plot([], [], color=cS, ls='-', label=r'Shear')
   ax0.plot([], [], color=cBH, ls='-', label=r'PSH')
   ax0.plot([], [], color=cBHg, ls='-', label=r'PH')
   ax0.plot([], [], color=cBH2, ls='-', label=r'PPH')
   ax0.plot([], [], color=cQ, ls='-', label=r'QE')
   #ax0.plot([], [], color=cD, ls='-', label='Mag')
   #
   plt.setp(ax0.get_xticklabels(), visible=False)
   ax0.legend(loc='upper center', ncol=5, mode=None, handlelength=.5, fontsize='x-small', borderpad=0.1, borderaxespad=0.1, frameon=False, columnspacing=.8)
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
   ax1.plot(lCen, 2. * ClBHCmbTszKappa/ClkCmb, color=cBH, ls='-', label=r'PSH')
   Up = ClBHCmbTszKappa + sClBHCmbTszKappa
   Down = ClBHCmbTszKappa - sClBHCmbTszKappa
   ax1.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cBH, alpha=0.6)
   # Bias hardened with profile
   ax1.plot(lCen, 2. * ClBHgCmbTszKappa/ClkCmb, color=cBHg, ls='-', label=r'PH')
   Up = ClBHgCmbTszKappa + sClBHgCmbTszKappa
   Down = ClBHgCmbTszKappa - sClBHgCmbTszKappa
   ax1.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cBHg, alpha=0.6)
   # Bias hardened point sources + profile
   ax1.plot(lCen, 2. * ClBH2CmbTszKappa/ClkCmb, color=cBH2, ls='-', label=r'PPH')
   Up = ClBH2CmbTszKappa + sClBH2CmbTszKappa
   Down = ClBH2CmbTszKappa - sClBH2CmbTszKappa
   ax1.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cBH2, alpha=0.6)
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
   ax2.plot(lCen, 2. * ClBHCmbKszKappa/ClkCmb, color=cBH, ls='-', label=r'PSH')
   Up = ClBHCmbKszKappa + sClBHCmbKszKappa
   Down = ClBHCmbKszKappa - sClBHCmbKszKappa
   ax2.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cBH, alpha=0.6)
   # Bias hardened with profile
   ax2.plot(lCen, 2. * ClBHgCmbKszKappa/ClkCmb, color=cBHg, ls='-', label=r'PH')
   Up = ClBHgCmbKszKappa + sClBHgCmbKszKappa
   Down = ClBHgCmbKszKappa - sClBHgCmbKszKappa
   ax2.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cBHg, alpha=0.6)
   # Bias hardened point sources + profile
   ax2.plot(lCen, 2. * ClBH2CmbKszKappa/ClkCmb, color=cBH2, ls='-', label=r'PPH')
   Up = ClBH2CmbKszKappa + sClBH2CmbKszKappa
   Down = ClBH2CmbKszKappa - sClBH2CmbKszKappa
   ax2.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cBH2, alpha=0.6)
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
   ax3.plot(lCen, 2. * ClBHCmbRadiopsKappa/ClkCmb, color=cBH, ls='-', label=r'PSH')
   Up = ClBHCmbRadiopsKappa + sClBHCmbRadiopsKappa
   Down = ClBHCmbRadiopsKappa - sClBHCmbRadiopsKappa
   ax3.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cBH, alpha=0.6)
   # Bias hardened with profile
   ax3.plot(lCen, 2. * ClBHgCmbRadiopsKappa/ClkCmb, color=cBHg, ls='-', label=r'PH')
   Up = ClBHgCmbRadiopsKappa + sClBHgCmbRadiopsKappa
   Down = ClBHgCmbRadiopsKappa - sClBHgCmbRadiopsKappa
   ax3.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cBHg, alpha=0.6)
   # Bias hardened point sources + profile
   ax3.plot(lCen, 2. * ClBH2CmbRadiopsKappa/ClkCmb, color=cBH2, ls='-', label=r'PPH')
   Up = ClBH2CmbRadiopsKappa + sClBH2CmbRadiopsKappa
   Down = ClBH2CmbRadiopsKappa - sClBH2CmbRadiopsKappa
   ax3.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cBH2, alpha=0.6)
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
   ax4.plot(lCen, 2. * ClBHCmbAllKappa/ClkCmb, color=cBH, ls='-', label=r'PSH')
   Up = ClBHCmbAllKappa + sClBHCmbAllKappa
   Down = ClBHCmbAllKappa - sClBHCmbAllKappa
   ax4.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cBH, alpha=0.6)
   # Bias hardened with profile
   ax4.plot(lCen, 2. * ClBHgCmbAllKappa/ClkCmb, color=cBHg, ls='-', label=r'PH')
   Up = ClBHgCmbAllKappa + sClBHgCmbAllKappa
   Down = ClBHgCmbAllKappa - sClBHgCmbAllKappa
   ax4.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cBHg, alpha=0.6)
   # Bias hardened with Gaussian profile
   ax4.plot(lCen, 2. * ClBH2CmbAllKappa/ClkCmb, color=cBH2, ls='-', label=r'PPH')
   Up = ClBH2CmbAllKappa + sClBH2CmbAllKappa
   Down = ClBH2CmbAllKappa - sClBH2CmbAllKappa
   ax4.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cBH2, alpha=0.6)
   # Shear
   ax4.plot(lCen, 2. * ClsCmbAllKappa/ClkCmb, color=cS, ls='-', label=r'Shear')
   Up = ClsCmbAllKappa + sClsCmbAllKappa
   Down = ClsCmbAllKappa - sClsCmbAllKappa
   ax4.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cS, alpha=0.6)
   #
   ax4.set_ylim((-0.15,0.05))
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
   # Bias hardened with profile
   ax0.plot(lCen, 2. * ClBHgCmbCibSec/ClkCmb, color=cBHg, ls='-')
   Up = ClBHgCmbCibSec + sClBHgCmbCibSec
   Down = ClBHgCmbCibSec - sClBHgCmbCibSec
   ax0.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cBHg, alpha=0.6)
   # Bias hardened point sources + profile
   ax0.plot(lCen, 2. * ClBH2CmbCibSec/ClkCmb, color=cBH2, ls='-')
   Up = ClBH2CmbCibSec + sClBH2CmbCibSec
   Down = ClBH2CmbCibSec - sClBH2CmbCibSec
   ax0.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cBH2, alpha=0.6)
   # Shear
   ax0.plot(lCen, 2. * ClsCmbCibSec/ClkCmb, color=cS, ls='-')
   Up = ClsCmbCibSec + sClsCmbCibSec
   Down = ClsCmbCibSec - sClsCmbCibSec
   ax0.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cS, alpha=0.6)
   #
#   ax0.set_ylim((-0.2,0.1))
   ax0.set_ylim((-0.05, 0.1))
   ax0.set_ylabel(r'CIB')
   #
   ax0.plot([], [], color=cS, ls='-', label=r'Shear')
   ax0.plot([], [], color=cBH, ls='-', label=r'PSH')
   ax0.plot([], [], color=cBHg, ls='-', label=r'PH')
   ax0.plot([], [], color=cBH2, ls='-', label=r'PPH')
   ax0.plot([], [], color=cQ, ls='-', label=r'QE')
   #ax0.plot([], [], color=cD, ls='-', label='Mag')
   #
   plt.setp(ax0.get_xticklabels(), visible=False)
   ax0.legend(loc='upper center', ncol=5, mode=None, handlelength=.5, fontsize='x-small', borderpad=0.1, borderaxespad=0.1, frameon=False, columnspacing=.8)
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
   ax1.plot(lCen, 2. * ClBHCmbTszSec/ClkCmb, color=cBH, ls='-', label=r'PSH')
   Up = ClBHCmbTszSec + sClBHCmbTszSec
   Down = ClBHCmbTszSec - sClBHCmbTszSec
   ax1.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cBH, alpha=0.6)
   # Bias hardened with profile
   ax1.plot(lCen, 2. * ClBHgCmbTszSec/ClkCmb, color=cBHg, ls='-', label=r'PH')
   Up = ClBHgCmbTszSec + sClBHgCmbTszSec
   Down = ClBHgCmbTszSec - sClBHgCmbTszSec
   ax1.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cBHg, alpha=0.6)
   # Bias hardened point sources + profile
   ax1.plot(lCen, 2. * ClBH2CmbTszSec/ClkCmb, color=cBH2, ls='-', label=r'PPH')
   Up = ClBH2CmbTszSec + sClBH2CmbTszSec
   Down = ClBH2CmbTszSec - sClBH2CmbTszSec
   ax1.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cBH2, alpha=0.6)
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
   ax2.plot(lCen, 2. * ClBHCmbKszSec/ClkCmb, color=cBH, ls='-', label=r'PSH')
   Up = ClBHCmbKszSec + sClBHCmbKszSec
   Down = ClBHCmbKszSec - sClBHCmbKszSec
   ax2.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cBH, alpha=0.6)
   # Bias hardened with profile
   ax2.plot(lCen, 2. * ClBHgCmbKszSec/ClkCmb, color=cBHg, ls='-', label=r'PH')
   Up = ClBHgCmbKszSec + sClBHgCmbKszSec
   Down = ClBHgCmbKszSec - sClBHgCmbKszSec
   ax2.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cBHg, alpha=0.6)
   # Bias hardened poinit sources + profile
   ax2.plot(lCen, 2. * ClBH2CmbKszSec/ClkCmb, color=cBH2, ls='-', label=r'PPH')
   Up = ClBH2CmbKszSec + sClBH2CmbKszSec
   Down = ClBH2CmbKszSec - sClBH2CmbKszSec
   ax2.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cBH2, alpha=0.6)
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
   ax3.plot(lCen, 2. * ClBHCmbRadiopsSec/ClkCmb, color=cBH, ls='-', label=r'PSH')
   Up = ClBHCmbRadiopsSec + sClBHCmbRadiopsSec
   Down = ClBHCmbRadiopsSec - sClBHCmbRadiopsSec
   ax3.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cBH, alpha=0.6)
   # Bias hardened with profile
   ax3.plot(lCen, 2. * ClBHgCmbRadiopsSec/ClkCmb, color=cBHg, ls='-', label=r'PH')
   Up = ClBHgCmbRadiopsSec + sClBHgCmbRadiopsSec
   Down = ClBHgCmbRadiopsSec - sClBHgCmbRadiopsSec
   ax3.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cBHg, alpha=0.6)
   # Bias hardened point sources + profile
   ax3.plot(lCen, 2. * ClBH2CmbRadiopsSec/ClkCmb, color=cBH2, ls='-', label=r'PPH')
   Up = ClBH2CmbRadiopsSec + sClBH2CmbRadiopsSec
   Down = ClBH2CmbRadiopsSec - sClBH2CmbRadiopsSec
   ax3.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cBH2, alpha=0.6)
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
   ax4.plot(lCen, 2. * ClBHCmbAllSec/ClkCmb, color=cBH, ls='-', label=r'PSH')
   Up = ClBHCmbAllSec + sClBHCmbAllSec
   Down = ClBHCmbAllSec - sClBHCmbAllSec
   ax4.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cBH, alpha=0.6)
   # Bias hardened with profile
   ax4.plot(lCen, 2. * ClBHgCmbAllSec/ClkCmb, color=cBHg, ls='-', label=r'PH')
   Up = ClBHgCmbAllSec + sClBHgCmbAllSec
   Down = ClBHgCmbAllSec - sClBHgCmbAllSec
   ax4.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cBHg, alpha=0.6)
   # Bias hardened point sources + profile
   ax4.plot(lCen, 2. * ClBH2CmbAllSec/ClkCmb, color=cBH2, ls='-', label=r'PPH')
   Up = ClBH2CmbAllSec + sClBH2CmbAllSec
   Down = ClBH2CmbAllSec - sClBH2CmbAllSec
   ax4.fill_between(lCen, 2. * Down/ClkCmb, 2. * Up/ClkCmb, edgecolor='', facecolor=cBH2, alpha=0.6)
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
   # Bias hardened with profile
   ax0.plot(lCen, ClBHgCmbCib/ClkCmb, color=cBHg, ls='-')
   Up = ClBHgCmbCib + sClBHgCmbCib
   Down = ClBHgCmbCib - sClBHgCmbCib
   ax0.fill_between(lCen, Down/ClkCmb, Up/ClkCmb, edgecolor='', facecolor=cBHg, alpha=0.6)
   # Bias hardened point sources + profile
   ax0.plot(lCen, ClBH2CmbCib/ClkCmb, color=cBH2, ls='-')
   Up = ClBH2CmbCib + sClBH2CmbCib
   Down = ClBH2CmbCib - sClBH2CmbCib
   ax0.fill_between(lCen, Down/ClkCmb, Up/ClkCmb, edgecolor='', facecolor=cBH2, alpha=0.6)
   # Shear
   ax0.plot(lCen, ClsCmbCib/ClkCmb, color=cS, ls='-')
   Up = ClsCmbCib + sClsCmbCib
   Down = ClsCmbCib - sClsCmbCib
   ax0.fill_between(lCen, Down/ClkCmb, Up/ClkCmb, edgecolor='', facecolor=cS, alpha=0.6)
   #
#   ax0.set_ylim((-0.2,0.1))
   ax0.set_ylim((-0.05, 0.1))
   ax0.set_ylabel(r'CIB')
   #
   ax0.plot([], [], color=cS, ls='-', label=r'Shear')
   ax0.plot([], [], color=cBH, ls='-', label=r'PSH')
   ax0.plot([], [], color=cBHg, ls='-', label=r'PH')
   ax0.plot([], [], color=cBH2, ls='-', label=r'PPH')
   ax0.plot([], [], color=cQ, ls='-', label=r'QE')
   #ax0.plot([], [], color=cD, ls='-', label='Mag')
   #
   plt.setp(ax0.get_xticklabels(), visible=False)
   ax0.legend(loc='upper center', ncol=5, mode=None, handlelength=.5, fontsize='x-small', borderpad=0.1, borderaxespad=0.1, frameon=False, columnspacing=.8)
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
   ax1.plot(lCen, ClBHCmbTsz/ClkCmb, color=cBH, ls='-', label=r'PSH')
   Up = ClBHCmbTsz + sClBHCmbTsz
   Down = ClBHCmbTsz - sClBHCmbTsz
   ax1.fill_between(lCen, Down/ClkCmb, Up/ClkCmb, edgecolor='', facecolor=cBH, alpha=0.6)
   # Bias hardened with profile
   ax1.plot(lCen, ClBHgCmbTsz/ClkCmb, color=cBHg, ls='-', label=r'PH')
   Up = ClBHgCmbTsz + sClBHgCmbTsz
   Down = ClBHgCmbTsz - sClBHgCmbTsz
   ax1.fill_between(lCen, Down/ClkCmb, Up/ClkCmb, edgecolor='', facecolor=cBHg, alpha=0.6)
   # Bias hardened point sources + profile
   ax1.plot(lCen, ClBH2CmbTsz/ClkCmb, color=cBH2, ls='-', label=r'PPH')
   Up = ClBH2CmbTsz + sClBH2CmbTsz
   Down = ClBH2CmbTsz - sClBH2CmbTsz
   ax1.fill_between(lCen, Down/ClkCmb, Up/ClkCmb, edgecolor='', facecolor=cBH2, alpha=0.6)
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
   ax2.plot(lCen, ClBHCmbKsz/ClkCmb, color=cBH, ls='-', label=r'PSH')
   Up = ClBHCmbKsz + sClBHCmbKsz
   Down = ClBHCmbKsz - sClBHCmbKsz
   ax2.fill_between(lCen, Down/ClkCmb, Up/ClkCmb, edgecolor='', facecolor=cBH, alpha=0.6)
   # Bias hardened with profile
   ax2.plot(lCen, ClBHgCmbKsz/ClkCmb, color=cBHg, ls='-', label=r'PH')
   Up = ClBHgCmbKsz + sClBHgCmbKsz
   Down = ClBHgCmbKsz - sClBHgCmbKsz
   ax2.fill_between(lCen, Down/ClkCmb, Up/ClkCmb, edgecolor='', facecolor=cBHg, alpha=0.6)
   # Bias hardened point sources + profile
   ax2.plot(lCen, ClBH2CmbKsz/ClkCmb, color=cBH2, ls='-', label=r'PPH')
   Up = ClBH2CmbKsz + sClBH2CmbKsz
   Down = ClBH2CmbKsz - sClBH2CmbKsz
   ax2.fill_between(lCen, Down/ClkCmb, Up/ClkCmb, edgecolor='', facecolor=cBH2, alpha=0.6)
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
   ax3.plot(lCen, ClBHCmbRadiops/ClkCmb, color=cBH, ls='-', label=r'PSH')
   Up = ClBHCmbRadiops + sClBHCmbRadiops
   Down = ClBHCmbRadiops - sClBHCmbRadiops
   ax3.fill_between(lCen, Down/ClkCmb,  Up/ClkCmb, edgecolor='', facecolor=cBH, alpha=0.6)
   # Bias hardened with profile
   ax3.plot(lCen, ClBHgCmbRadiops/ClkCmb, color=cBHg, ls='-', label=r'PH')
   Up = ClBHgCmbRadiops + sClBHgCmbRadiops
   Down = ClBHgCmbRadiops - sClBHgCmbRadiops
   ax3.fill_between(lCen, Down/ClkCmb,  Up/ClkCmb, edgecolor='', facecolor=cBHg, alpha=0.6)
   # Bias hardened point sources + profile
   ax3.plot(lCen, ClBH2CmbRadiops/ClkCmb, color=cBH2, ls='-', label=r'PPH')
   Up = ClBH2CmbRadiops + sClBH2CmbRadiops
   Down = ClBH2CmbRadiops - sClBH2CmbRadiops
   ax3.fill_between(lCen, Down/ClkCmb,  Up/ClkCmb, edgecolor='', facecolor=cBH2, alpha=0.6)
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
   ax4.plot(lCen, ClBHCmbAll/ClkCmb, color=cBH, ls='-', label=r'PSH')
   Up = ClBHCmbAll + sClBHCmbAll
   Down = ClBHCmbAll - sClBHCmbAll
   ax4.fill_between(lCen, Down/ClkCmb, Up/ClkCmb, edgecolor='', facecolor=cBH, alpha=0.6)
   # Bias hardened with profile
   ax4.plot(lCen, ClBHgCmbAll/ClkCmb, color=cBHg, ls='-', label=r'PH')
   Up = ClBHgCmbAll + sClBHgCmbAll
   Down = ClBHgCmbAll - sClBHgCmbAll
   ax4.fill_between(lCen, Down/ClkCmb, Up/ClkCmb, edgecolor='', facecolor=cBHg, alpha=0.6)
   # Bias hardened point sources + profile
   ax4.plot(lCen, ClBH2CmbAll/ClkCmb, color=cBH2, ls='-', label=r'PPH')
   Up = ClBH2CmbAll + sClBH2CmbAll
   Down = ClBH2CmbAll - sClBH2CmbAll
   ax4.fill_between(lCen, Down/ClkCmb, Up/ClkCmb, edgecolor='', facecolor=cBH2, alpha=0.6)
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
   # Bias hardened with profile
   ax0.plot(lCen, ClBHgCmbCibLsstgold/Ckg, color=cBHg, ls='-')
   Up = ClBHgCmbCibLsstgold + sClBHgCmbCibLsstgold
   Down = ClBHgCmbCibLsstgold - sClBHgCmbCibLsstgold
   ax0.fill_between(lCen, Down/Ckg, Up/Ckg, edgecolor='', facecolor=cBHg, alpha=0.6)
   # Bias hardened point sources + profile
   ax0.plot(lCen, ClBH2CmbCibLsstgold/Ckg, color=cBH2, ls='-')
   Up = ClBH2CmbCibLsstgold + sClBH2CmbCibLsstgold
   Down = ClBH2CmbCibLsstgold - sClBH2CmbCibLsstgold
   ax0.fill_between(lCen, Down/Ckg, Up/Ckg, edgecolor='', facecolor=cBH2, alpha=0.6)
   # Shear
   ax0.plot(lCen, ClsCmbCibLsstgold/Ckg, color=cS, ls='-')
   Up = ClsCmbCibLsstgold + sClsCmbCibLsstgold
   Down = ClsCmbCibLsstgold - sClsCmbCibLsstgold
   ax0.fill_between(lCen, Down/Ckg, Up/Ckg, edgecolor='', facecolor=cS, alpha=0.6)
   #
   ax0.plot([], [], color=cS, ls='-', label=r'Shear')
   ax0.plot([], [], color=cBH, ls='-', label=r'PSH')
   ax0.plot([], [], color=cBHg, ls='-', label=r'PH')
   ax0.plot([], [], color=cBH2, ls='-', label=r'PPH')
   ax0.plot([], [], color=cQ, ls='-', label=r'QE')
   #ax0.plot([], [], color=cD, ls='-', label='Mag.')
   #
#   ax0.set_ylim((-0.25, 0.15))
   ax0.set_ylim((-0.12, 0.1))
   ax0.set_ylabel(r'CIB')
   #
   plt.setp(ax0.get_xticklabels(), visible=False)
   ax0.legend(loc='upper center', ncol=5, mode=None, handlelength=.5, fontsize='x-small', borderpad=0.1, borderaxespad=0.1, frameon=False)
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
   ax1.plot(lCen, ClBHCmbTszLsstgold/Ckg, color=cBH, ls='-', label=r'PSH')
   Up = ClBHCmbTszLsstgold + sClBHCmbTszLsstgold
   Down = ClBHCmbTszLsstgold - sClBHCmbTszLsstgold
   ax1.fill_between(lCen, Down/Ckg, Up/Ckg, edgecolor='', facecolor=cBH, alpha=0.6)
   # Bias hardened with profile
   ax1.plot(lCen, ClBHgCmbTszLsstgold/Ckg, color=cBHg, ls='-', label=r'PH')
   Up = ClBHgCmbTszLsstgold + sClBHgCmbTszLsstgold
   Down = ClBHgCmbTszLsstgold - sClBHgCmbTszLsstgold
   ax1.fill_between(lCen, Down/Ckg, Up/Ckg, edgecolor='', facecolor=cBHg, alpha=0.6)
   # Bias hardened point sources + profile
   ax1.plot(lCen, ClBH2CmbTszLsstgold/Ckg, color=cBH2, ls='-', label=r'PPH')
   Up = ClBH2CmbTszLsstgold + sClBH2CmbTszLsstgold
   Down = ClBH2CmbTszLsstgold - sClBH2CmbTszLsstgold
   ax1.fill_between(lCen, Down/Ckg, Up/Ckg, edgecolor='', facecolor=cBH2, alpha=0.6)
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
   ax2.plot(lCen, ClBHCmbKszLsstgold/Ckg, color=cBH, ls='-', label=r'PSH')
   Up = ClBHCmbKszLsstgold + sClBHCmbKszLsstgold
   Down = ClBHCmbKszLsstgold - sClBHCmbKszLsstgold
   ax2.fill_between(lCen, Down/Ckg, Up/Ckg, edgecolor='', facecolor=cBH, alpha=0.6)
   # Bias hardened with profile
   ax2.plot(lCen, ClBHgCmbKszLsstgold/Ckg, color=cBHg, ls='-', label=r'PH')
   Up = ClBHgCmbKszLsstgold + sClBHgCmbKszLsstgold
   Down = ClBHgCmbKszLsstgold - sClBHgCmbKszLsstgold
   ax2.fill_between(lCen, Down/Ckg, Up/Ckg, edgecolor='', facecolor=cBHg, alpha=0.6)
   # Bias hardened point sources + profile
   ax2.plot(lCen, ClBH2CmbKszLsstgold/Ckg, color=cBH2, ls='-', label=r'PPH')
   Up = ClBH2CmbKszLsstgold + sClBH2CmbKszLsstgold
   Down = ClBH2CmbKszLsstgold - sClBH2CmbKszLsstgold
   ax2.fill_between(lCen, Down/Ckg, Up/Ckg, edgecolor='', facecolor=cBH2, alpha=0.6)
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
   ax3.plot(lCen, ClBHCmbRadiopsLsstgold/Ckg, color=cBH, ls='-', label=r'PSH')
   Up = ClBHCmbRadiopsLsstgold + sClBHCmbRadiopsLsstgold
   Down = ClBHCmbRadiopsLsstgold - sClBHCmbRadiopsLsstgold
   ax3.fill_between(lCen, Down/Ckg, Up/Ckg, edgecolor='', facecolor=cBH, alpha=0.6)
   # Bias hardened with profile
   ax3.plot(lCen, ClBHgCmbRadiopsLsstgold/Ckg, color=cBHg, ls='-', label=r'PH')
   Up = ClBHgCmbRadiopsLsstgold + sClBHgCmbRadiopsLsstgold
   Down = ClBHgCmbRadiopsLsstgold - sClBHgCmbRadiopsLsstgold
   ax3.fill_between(lCen, Down/Ckg, Up/Ckg, edgecolor='', facecolor=cBHg, alpha=0.6)
   # Bias hardened point sources + profile
   ax3.plot(lCen, ClBH2CmbRadiopsLsstgold/Ckg, color=cBH2, ls='-', label=r'PPH')
   Up = ClBH2CmbRadiopsLsstgold + sClBH2CmbRadiopsLsstgold
   Down = ClBH2CmbRadiopsLsstgold - sClBH2CmbRadiopsLsstgold
   ax3.fill_between(lCen, Down/Ckg, Up/Ckg, edgecolor='', facecolor=cBH2, alpha=0.6)
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
   #ax4.plot(lCen, ClBHCmbSumLsstgold/Ckg, color=cBH, ls='-', label=r'PSH')
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
   ax4.plot(lCen, ClBHCmbAllLsstgold/Ckg, color=cBH, ls='-', label=r'PSH')
   Up = ClBHCmbAllLsstgold + sClBHCmbAllLsstgold
   Down = ClBHCmbAllLsstgold - sClBHCmbAllLsstgold
   ax4.fill_between(lCen, Down/Ckg, Up/Ckg, edgecolor='', facecolor=cBH, alpha=0.6)
   # Bias hardened with profile
   ax4.plot(lCen, ClBHgCmbAllLsstgold/Ckg, color=cBHg, ls='-', label=r'PH')
   Up = ClBHgCmbAllLsstgold + sClBHgCmbAllLsstgold
   Down = ClBHgCmbAllLsstgold - sClBHgCmbAllLsstgold
   ax4.fill_between(lCen, Down/Ckg, Up/Ckg, edgecolor='', facecolor=cBHg, alpha=0.6)
   # Bias hardened point sources + profile
   ax4.plot(lCen, ClBH2CmbAllLsstgold/Ckg, color=cBH2, ls='-', label=r'PPH')
   Up = ClBH2CmbAllLsstgold + sClBH2CmbAllLsstgold
   Down = ClBH2CmbAllLsstgold - sClBH2CmbAllLsstgold
   ax4.fill_between(lCen, Down/Ckg, Up/Ckg, edgecolor='', facecolor=cBH2, alpha=0.6)
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
   #plt.show()
   #fig.savefig(path, bbox_inches='tight')
   fig.clf()

   # Basic info
   print('max(ell)',np.max(ell))
   print('nBins',nBins)


   
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
snrBH2Cmb = np.zeros(len(llMax))
sHybridCmb = np.zeros(len(llMax))
#
sQCmbLsstgold = np.zeros(len(llMax))
sSCmbLsstgold = np.zeros(len(llMax))
sBHCmbLsstgold = np.zeros(len(llMax))
sBHgCmbLsstgold = np.zeros(len(llMax))
sBH2CmbLsstgold = np.zeros(len(llMax))
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
bBH2CmbCibLsstgold = np.zeros((len(llMax), nPatch))
#
bBH2CmbTszLsstgold = np.zeros((len(llMax), nPatch))
#
bBH2CmbKszLsstgold = np.zeros((len(llMax), nPatch))
#
bBH2CmbRadiopsLsstgold = np.zeros((len(llMax), nPatch))
#
bBH2CmbAllLsstgold = np.zeros((len(llMax), nPatch))
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
bBH2CmbCibFull = np.zeros((len(llMax), nPatch))
#
bBH2CmbTszFull = np.zeros((len(llMax), nPatch))
#
bBH2CmbKszFull = np.zeros((len(llMax), nPatch))
#
bBH2CmbRadiopsFull = np.zeros((len(llMax), nPatch))
#
bBH2CmbAllFull = np.zeros((len(llMax), nPatch))
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
   fNqBHgCmb_fft = baseMap.forecastN0KappaBH(cmb.flensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False, cache=None, u=uTsz)
   fNqBH2Cmb_fft = baseMap.forecastN0KappaBH2(cmb.flensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False, cache=None, u=uTsz)

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
   Nbh2Cmb = fNqBH2Cmb_fft(lCen)
   Ckg = p2d_lsstgoldcmblens.fPtotinterp(lCen)
   Cgg = p2d_lsstgold.fPtotinterp(lCen)

   ##################################################################################
   # Spectra info

   i=0
   # Radio point sources
   #
   # Trispectrum
   ClqCmbRadiops = datalist[:,int(0+24*i)]
   ClsCmbRadiops = datalist[:,int(1+24*i)]
   CldCmbRadiops = datalist[:,int(2+24*i)]
   ClBHCmbRadiops = datalist[:,int(3+24*i)]
   ClBHgCmbRadiops = datalist[:,int(4+24*i)]
   ClBH2CmbRadiops = datalist[:,int(5+24*i)]
   # Primary
   ClqCmbRadiopsKappa = datalist[:,int(6+24*i)]
   ClsCmbRadiopsKappa = datalist[:,int(7+24*i)]
   CldCmbRadiopsKappa = datalist[:,int(8+24*i)]
   ClBHCmbRadiopsKappa = datalist[:,int(9+24*i)]
   ClBHgCmbRadiopsKappa = datalist[:,int(10+24*i)]
   ClBH2CmbRadiopsKappa = datalist[:,int(11+24*i)]
   # k_rec x LSST
   ClqCmbRadiopsLsstgold = datalist[:,int(12+24*i)]
   ClsCmbRadiopsLsstgold = datalist[:,int(13+24*i)]
   CldCmbRadiopsLsstgold = datalist[:,int(14+24*i)]
   ClBHCmbRadiopsLsstgold = datalist[:,int(15+24*i)]
   ClBHgCmbRadiopsLsstgold = datalist[:,int(16+24*i)]
   ClBH2CmbRadiopsLsstgold = datalist[:,int(17+24*i)]
   # Secondary
   ClqCmbRadiopsSec = datalist[:,int(18+24*i)]
   CldCmbRadiopsSec = datalist[:,int(19+24*i)]
   ClsCmbRadiopsSec = datalist[:,int(20+24*i)]
   ClBHCmbRadiopsSec = datalist[:,int(21+24*i)]
   ClBHgCmbRadiopsSec = datalist[:,int(22+24*i)]
   ClBH2CmbRadiopsSec = datalist[:,int(23+24*i)]

   i=1
   # CIB
   #
   # Trispectrum
   ClqCmbCib = datalist[:,int(0+24*i)]
   ClsCmbCib = datalist[:,int(1+24*i)]
   CldCmbCib = datalist[:,int(2+24*i)]
   ClBHCmbCib = datalist[:,int(3+24*i)]
   ClBHgCmbCib = datalist[:,int(4+24*i)]
   ClBH2CmbCib = datalist[:,int(5+24*i)]
   # Primary
   ClqCmbCibKappa = datalist[:,int(6+24*i)]
   ClsCmbCibKappa = datalist[:,int(7+24*i)]
   CldCmbCibKappa = datalist[:,int(8+24*i)]
   ClBHCmbCibKappa = datalist[:,int(9+24*i)]
   ClBHgCmbCibKappa = datalist[:,int(10+24*i)]
   ClBH2CmbCibKappa = datalist[:,int(11+24*i)]
   # k_rec x LSST
   ClqCmbCibLsstgold = datalist[:,int(12+24*i)]
   ClsCmbCibLsstgold = datalist[:,int(13+24*i)]
   CldCmbCibLsstgold = datalist[:,int(14+24*i)]
   ClBHCmbCibLsstgold = datalist[:,int(15+24*i)]
   ClBHgCmbCibLsstgold = datalist[:,int(16+24*i)]
   ClBH2CmbCibLsstgold = datalist[:,int(17+24*i)]
   # Secondary
   ClqCmbCibSec = datalist[:,int(18+24*i)]
   CldCmbCibSec = datalist[:,int(19+24*i)]
   ClsCmbCibSec = datalist[:,int(20+24*i)]
   ClBHCmbCibSec = datalist[:,int(21+24*i)]
   ClBHgCmbCibSec = datalist[:,int(22+24*i)]
   ClBH2CmbCibSec = datalist[:,int(23+24*i)]

   i=2
   # tSZ
   #
   # Trispectrum
   ClqCmbTsz = datalist[:,int(0+24*i)]
   ClsCmbTsz = datalist[:,int(1+24*i)]
   CldCmbTsz = datalist[:,int(2+24*i)]
   ClBHCmbTsz = datalist[:,int(3+24*i)]
   ClBHgCmbTsz = datalist[:,int(4+24*i)]
   ClBH2CmbTsz = datalist[:,int(5+24*i)]
   # Primary
   ClqCmbTszKappa = datalist[:,int(6+24*i)]
   ClsCmbTszKappa = datalist[:,int(7+24*i)]
   CldCmbTszKappa = datalist[:,int(8+24*i)]
   ClBHCmbTszKappa = datalist[:,int(9+24*i)]
   ClBHgCmbTszKappa = datalist[:,int(10+24*i)]
   ClBH2CmbTszKappa = datalist[:,int(11+24*i)]
   # k_rec x LSST
   ClqCmbTszLsstgold = datalist[:,int(12+24*i)]
   ClsCmbTszLsstgold = datalist[:,int(13+24*i)]
   CldCmbTszLsstgold = datalist[:,int(14+24*i)]
   ClBHCmbTszLsstgold = datalist[:,int(15+24*i)]
   ClBHgCmbTszLsstgold = datalist[:,int(16+24*i)]
   ClBH2CmbTszLsstgold = datalist[:,int(17+24*i)]
   # Secondary
   ClqCmbTszSec = datalist[:,int(18+24*i)]
   CldCmbTszSec = datalist[:,int(19+24*i)]
   ClsCmbTszSec = datalist[:,int(20+24*i)]
   ClBHCmbTszSec = datalist[:,int(21+24*i)]
   ClBHgCmbTszSec = datalist[:,int(22+24*i)]
   ClBH2CmbTszSec = datalist[:,int(23+24*i)]

   i=3
   # kSZ
   # 
   # Trispectrum
   ClqCmbKsz = datalist[:,int(0+24*i)]
   ClsCmbKsz = datalist[:,int(1+24*i)]
   CldCmbKsz = datalist[:,int(2+24*i)]
   ClBHCmbKsz = datalist[:,int(3+24*i)]
   ClBHgCmbKsz = datalist[:,int(4+24*i)]
   ClBH2CmbKsz = datalist[:,int(5+24*i)]
   # Primary
   ClqCmbKszKappa = datalist[:,int(6+24*i)]
   ClsCmbKszKappa = datalist[:,int(7+24*i)]
   CldCmbKszKappa = datalist[:,int(8+24*i)]
   ClBHCmbKszKappa = datalist[:,int(9+24*i)]
   ClBHgCmbKszKappa = datalist[:,int(10+24*i)]
   ClBH2CmbKszKappa = datalist[:,int(11+24*i)]
   # k_rec x LSST 
   ClqCmbKszLsstgold = datalist[:,int(12+24*i)]
   ClsCmbKszLsstgold = datalist[:,int(13+24*i)]
   CldCmbKszLsstgold = datalist[:,int(14+24*i)]
   ClBHCmbKszLsstgold = datalist[:,int(15+24*i)]
   ClBHgCmbKszLsstgold = datalist[:,int(16+24*i)]
   ClBH2CmbKszLsstgold = datalist[:,int(17+24*i)]
   # Secondary
   ClqCmbKszSec = datalist[:,int(18+24*i)]
   CldCmbKszSec = datalist[:,int(19+24*i)]
   ClsCmbKszSec = datalist[:,int(20+24*i)]
   ClBHCmbKszSec = datalist[:,int(21+24*i)]
   ClBHgCmbKszSec = datalist[:,int(22+24*i)]
   ClBH2CmbKszSec = datalist[:,int(23+24*i)]

   i=4
   # Dust
   # 
   # Trispectrum
   ClqCmbDust = datalist[:,int(0+24*i)]
   ClsCmbDust = datalist[:,int(1+24*i)]
   CldCmbDust = datalist[:,int(2+24*i)]
   ClBHCmbDust = datalist[:,int(3+24*i)]
   ClBHgCmbDust = datalist[:,int(4+24*i)]
   ClBH2CmbDust = datalist[:,int(5+24*i)]
   # Primary
   ClqCmbDustKappa = datalist[:,int(6+24*i)]
   ClsCmbDustKappa = datalist[:,int(7+24*i)]
   CldCmbDustKappa = datalist[:,int(8+24*i)]
   ClBHCmbDustKappa = datalist[:,int(9+24*i)]
   ClBHgCmbDustKappa = datalist[:,int(10+24*i)]
   ClBH2CmbDustKappa = datalist[:,int(11+24*i)]
   # k_rec x LSST 
   ClqCmbDustLsstgold = datalist[:,int(12+24*i)]
   ClsCmbDustLsstgold = datalist[:,int(13+24*i)]
   CldCmbDustLsstgold = datalist[:,int(14+24*i)]
   ClBHCmbDustLsstgold = datalist[:,int(15+24*i)]
   ClBHgCmbDustLsstgold = datalist[:,int(16+24*i)]
   ClBH2CmbDustLsstgold = datalist[:,int(17+24*i)]
   # Secondary
   ClqCmbDustSec = datalist[:,int(18+24*i)]
   CldCmbDustSec = datalist[:,int(19+24*i)]
   ClsCmbDustSec = datalist[:,int(20+24*i)]
   ClBHCmbDustSec = datalist[:,int(21+24*i)]
   ClBHgCmbDustSec = datalist[:,int(22+24*i)]
   ClBH2CmbDustSec = datalist[:,int(23+24*i)]

   i=5
   # FreeFree
   # 
   # Trispectrum
   ClqCmbFreeFree = datalist[:,int(0+24*i)]
   ClsCmbFreeFree = datalist[:,int(1+24*i)]
   CldCmbFreeFree = datalist[:,int(2+24*i)]
   ClBHCmbFreeFree = datalist[:,int(3+24*i)]
   ClBHgCmbFreeFree = datalist[:,int(4+24*i)]
   ClBH2CmbFreeFree = datalist[:,int(5+24*i)]
   # Primary
   ClqCmbFreeFreeKappa = datalist[:,int(6+24*i)]
   ClsCmbFreeFreeKappa = datalist[:,int(7+24*i)]
   CldCmbFreeFreeKappa = datalist[:,int(8+24*i)]
   ClBHCmbFreeFreeKappa = datalist[:,int(9+24*i)]
   ClBHgCmbFreeFreeKappa = datalist[:,int(10+24*i)]
   ClBH2CmbFreeFreeKappa = datalist[:,int(11+24*i)]
   # k_rec x LSST 
   ClqCmbFreeFreeLsstgold = datalist[:,int(12+24*i)]
   ClsCmbFreeFreeLsstgold = datalist[:,int(13+24*i)]
   CldCmbFreeFreeLsstgold = datalist[:,int(14+24*i)]
   ClBHCmbFreeFreeLsstgold = datalist[:,int(15+24*i)]
   ClBHgCmbFreeFreeLsstgold = datalist[:,int(16+24*i)]
   ClBH2CmbFreeFreeLsstgold = datalist[:,int(17+24*i)]
   # Secondary
   ClqCmbFreeFreeSec = datalist[:,int(18+24*i)]
   CldCmbFreeFreeSec = datalist[:,int(19+24*i)]
   ClsCmbFreeFreeSec = datalist[:,int(20+24*i)]
   ClBHCmbFreeFreeSec = datalist[:,int(21+24*i)]
   ClBHgCmbFreeFreeSec = datalist[:,int(22+24*i)]
   ClBH2CmbFreeFreeSec = datalist[:,int(23+24*i)]

   i=6
   # Sync
   # 
   # Trispectrum
   ClqCmbSync = datalist[:,int(0+24*i)]
   ClsCmbSync = datalist[:,int(1+24*i)]
   CldCmbSync = datalist[:,int(2+24*i)]
   ClBHCmbSync = datalist[:,int(3+24*i)]
   ClBHgCmbSync = datalist[:,int(4+24*i)]
   ClBH2CmbSync = datalist[:,int(5+24*i)]
   # Primary
   ClqCmbSyncKappa = datalist[:,int(6+24*i)]
   ClsCmbSyncKappa = datalist[:,int(7+24*i)]
   CldCmbSyncKappa = datalist[:,int(8+24*i)]
   ClBHCmbSyncKappa = datalist[:,int(9+24*i)]
   ClBHgCmbSyncKappa = datalist[:,int(10+24*i)]
   ClBH2CmbSyncKappa = datalist[:,int(11+24*i)]
   # k_rec x LSST 
   ClqCmbSyncLsstgold = datalist[:,int(12+24*i)]
   ClsCmbSyncLsstgold = datalist[:,int(13+24*i)]
   CldCmbSyncLsstgold = datalist[:,int(14+24*i)]
   ClBHCmbSyncLsstgold = datalist[:,int(15+24*i)]
   ClBHgCmbSyncLsstgold = datalist[:,int(16+24*i)]
   ClBH2CmbSyncLsstgold = datalist[:,int(17+24*i)]
   # Secondary
   ClqCmbSyncSec = datalist[:,int(18+24*i)]
   CldCmbSyncSec = datalist[:,int(19+24*i)]
   ClsCmbSyncSec = datalist[:,int(20+24*i)]
   ClBHCmbSyncSec = datalist[:,int(21+24*i)]
   ClBHgCmbSyncSec = datalist[:,int(22+24*i)]
   ClBH2CmbSyncSec = datalist[:,int(23+24*i)]

   i=7
   # All extragalactic foregrounds
   # 
   # Trispectrum
   ClqCmbAll = datalist[:,int(0+24*i)]
   ClsCmbAll = datalist[:,int(1+24*i)]
   CldCmbAll = datalist[:,int(2+24*i)]
   ClBHCmbAll = datalist[:,int(3+24*i)]
   ClBHgCmbAll = datalist[:,int(4+24*i)]
   ClBH2CmbAll = datalist[:,int(5+24*i)]
   # Primary
   ClqCmbAllKappa = datalist[:,int(6+24*i)]
   ClsCmbAllKappa = datalist[:,int(7+24*i)]
   CldCmbAllKappa = datalist[:,int(8+24*i)]
   ClBHCmbAllKappa = datalist[:,int(9+24*i)]
   ClBHgCmbAllKappa = datalist[:,int(10+24*i)]
   ClBH2CmbAllKappa = datalist[:,int(11+24*i)]
   # k_rec x LSST 
   ClqCmbAllLsstgold = datalist[:,int(12+24*i)]
   ClsCmbAllLsstgold = datalist[:,int(13+24*i)]
   CldCmbAllLsstgold = datalist[:,int(14+24*i)]
   ClBHCmbAllLsstgold = datalist[:,int(15+24*i)]
   ClBHgCmbAllLsstgold = datalist[:,int(16+24*i)]
   ClBH2CmbAllLsstgold = datalist[:,int(17+24*i)]
   # Secondary
   ClqCmbAllSec = datalist[:,int(18+24*i)]
   CldCmbAllSec = datalist[:,int(19+24*i)]
   ClsCmbAllSec = datalist[:,int(20+24*i)]
   ClBHCmbAllSec = datalist[:,int(21+24*i)]
   ClBHgCmbAllSec = datalist[:,int(22+24*i)]
   ClBH2CmbAllSec = datalist[:,int(22+24*i)]

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


   # BH with profile
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


   # BH point sources + profile
   s2 =  2. * (ClkCmb + Nbh2Cmb)**2 / Nmodes 
   norm = np.sum(ClkCmb[I]**2 / s2[I])
   #
   snrBH2Cmb[ilMax] = np.sqrt(norm)
   #
   bBH2CmbCibFull[ilMax] = np.sum((2.*ClBH2CmbCibKappa[:,I[0]]+2.*ClBH2CmbCibSec[:,I[0]]+ClBH2CmbCib[:,I[0]]) * ClkCmb[I] / s2[I], axis=1) / norm
   #
   bBH2CmbTszFull[ilMax] = np.sum((2.*ClBH2CmbTszKappa[:,I[0]]+2.*ClBH2CmbTszSec[:,I[0]]+ClBH2CmbTsz[:,I[0]]) * ClkCmb[I] / s2[I], axis=1) / norm
   #
   bBH2CmbKszFull[ilMax] = np.sum((2.*ClBH2CmbKszKappa[:,I[0]]+2.*ClBH2CmbKszSec[:,I[0]]+ClBH2CmbKsz[:,I[0]]) * ClkCmb[I] / s2[I], axis=1) / norm
   #
   bBH2CmbRadiopsFull[ilMax] = np.sum((2.*ClBH2CmbRadiopsKappa[:,I[0]]+2.*ClBH2CmbRadiopsSec[:,I[0]]+ClBH2CmbRadiops[:,I[0]]) * ClkCmb[I] / s2[I], axis=1) / norm
   #
   bBH2CmbAllFull[ilMax] = np.sum((2.*ClBH2CmbAllKappa[:,I[0]]+2.*ClBH2CmbAllSec[:,I[0]]+ClBH2CmbAll[:,I[0]]) * ClkCmb[I] / s2[I], axis=1) / norm
   #
   #
   s2 = ((ClkCmb + Nbh2Cmb)*Cgg + (Ckg)**2) / Nmodes
   norm = np.sum(Ckg[I]**2 / s2[I])
   #
   sBH2CmbLsstgold[ilMax] = 1./np.sqrt(norm)
   #
   bBH2CmbCibLsstgold[ilMax] = np.sum(ClBH2CmbCibLsstgold[:,I[0]] * Ckg[I] / s2[I], axis=1) / norm
   #
   bBH2CmbTszLsstgold[ilMax] = np.sum(ClBH2CmbTszLsstgold[:,I[0]] * Ckg[I] / s2[I], axis=1) / norm
   #
   bBH2CmbKszLsstgold[ilMax] = np.sum(ClBH2CmbKszLsstgold[:,I[0]] * Ckg[I] / s2[I], axis=1) / norm
   #
   bBH2CmbRadiopsLsstgold[ilMax] = np.sum(ClBH2CmbRadiopsLsstgold[:,I[0]] * Ckg[I] / s2[I], axis=1) / norm
   #
   bBH2CmbAllLsstgold[ilMax] = np.sum(ClBH2CmbAllLsstgold[:,I[0]] * Ckg[I] / s2[I], axis=1) / norm


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
      ClqCmbAll = datalist[:,int(0+24*7)]
      ClqCmbAllKappa = datalist[:,int(6+24*7)]
      ClqCmbAllSec = datalist[:,int(18+24*7)]
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
      ClqCmbAllLsstgold = datalist[:,int(12+24*7)]
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


# BH with profile
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


# BH point sources + profile
sBbH2CmbCibFull = np.std(bBH2CmbCibFull, axis=1) * uncertaintyFactor / np.sqrt(nPatch)
bBH2CmbCibFull = np.mean(bBH2CmbCibFull, axis=1)
#
sBbH2CmbTszFull = np.std(bBH2CmbTszFull, axis=1) * uncertaintyFactor / np.sqrt(nPatch)
bBH2CmbTszFull = np.mean(bBH2CmbTszFull, axis=1)
#
sBbH2CmbKszFull = np.std(bBH2CmbKszFull, axis=1) * uncertaintyFactor / np.sqrt(nPatch) 
bBH2CmbKszFull = np.mean(bBH2CmbKszFull, axis=1)
#
sBbH2CmbRadiopsFull = np.std(bBH2CmbRadiopsFull, axis=1) * uncertaintyFactor / np.sqrt(nPatch) 
bBH2CmbRadiopsFull = np.mean(bBH2CmbRadiopsFull, axis=1)
#
sBbH2CmbAllFull = np.std(bBH2CmbAllFull, axis=1) * uncertaintyFactor / np.sqrt(nPatch)
bBH2CmbAllFull = np.mean(bBH2CmbAllFull, axis=1)
#
#
sBbH2CmbCibLsstgold = np.std(bBH2CmbCibLsstgold, axis=1) * uncertaintyFactor / np.sqrt(nPatch)
bBH2CmbCibLsstgold = np.mean(bBH2CmbCibLsstgold, axis=1)
#
sBbH2CmbTszLsstgold = np.std(bBH2CmbTszLsstgold, axis=1) * uncertaintyFactor / np.sqrt(nPatch)
bBH2CmbTszLsstgold = np.mean(bBH2CmbTszLsstgold, axis=1)
#
sBbH2CmbKszLsstgold = np.std(bBH2CmbKszLsstgold, axis=1) * uncertaintyFactor / np.sqrt(nPatch)
bBH2CmbKszLsstgold = np.mean(bBH2CmbKszLsstgold, axis=1)
#
sBbH2CmbRadiopsLsstgold = np.std(bBH2CmbRadiopsLsstgold, axis=1) * uncertaintyFactor / np.sqrt(nPatch)
bBH2CmbRadiopsLsstgold = np.mean(bBH2CmbRadiopsLsstgold, axis=1)
#
sBbH2CmbAllLsstgold = np.std(bBH2CmbAllLsstgold, axis=1) * uncertaintyFactor / np.sqrt(nPatch)
bBH2CmbAllLsstgold = np.mean(bBH2CmbAllLsstgold, axis=1)


# Hybrid
bHybridCmbAllFull = np.mean(bHybridCmbAllFull, axis=1)
bHybridCmbAllLsstgold = np.mean(bHybridCmbAllLsstgold, axis=1)



##################################################################################
##################################################################################
# Summary for each estimator: Cross with LSST gold

fig=plt.figure(0)
gs = gridspec.GridSpec(5, 1)
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
ax0.set_ylim((-0.025, 0.05))
ax0.set_ylabel('PSH', fontsize='x-small')
#
plt.setp(ax0.get_xticklabels(), visible=False)
ax0.legend(loc='upper center', ncol=5, mode=None, handlelength=.5, fontsize='x-small', borderpad=0.1, borderaxespad=0.1, frameon=False, columnspacing=.8)
ax0.set_title(r'Relative Bias on $C_L^{\kappa \times \text{LSST}}$ amplitude')


# Bias hardened with profile
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
ax1.set_ylabel('PH', fontsize='x-small')
#
plt.setp(ax1.get_xticklabels(), visible=False)
# remove last tick label for the second subplot
yticks = ax1.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)


# Bias hardened point sources + profile
ax2=plt.subplot(gs[2], sharex=ax1)
#
ax2.axhline(0., c='k', lw=1)
ax2.fill_between(llMax, -sBH2CmbLsstgold, sBH2CmbLsstgold, edgecolor='', facecolor='gray', alpha=0.6)
# CIB
ax2.plot(llMax, bBH2CmbCibLsstgold, cCib, label='CIB')
Up = bBH2CmbCibLsstgold + sBbH2CmbCibLsstgold
Down = bBH2CmbCibLsstgold - sBbH2CmbCibLsstgold
ax2.fill_between(llMax, Down, Up, edgecolor='', facecolor=cCib, alpha=0.4)
# tSZ
ax2.plot(llMax, bBH2CmbTszLsstgold, cTsz, label='tSZ')
Up = bBH2CmbTszLsstgold + sBbH2CmbTszLsstgold
Down = bBH2CmbTszLsstgold - sBbH2CmbTszLsstgold
ax2.fill_between(llMax, Down, Up, edgecolor='', facecolor=cTsz, alpha=0.4)
# kSZ
ax2.plot(llMax, bBH2CmbKszLsstgold, cKsz, label='kSZ')
Up = bBH2CmbKszLsstgold + sBbH2CmbKszLsstgold
Down = bBH2CmbKszLsstgold - sBbH2CmbKszLsstgold
ax2.fill_between(llMax, Down, Up, edgecolor='', facecolor=cKsz, alpha=0.4)
# radio point sources
ax2.plot(llMax, bBH2CmbRadiopsLsstgold, cRadiops, label='Radiops')
Up = bBH2CmbRadiopsLsstgold + sBbH2CmbRadiopsLsstgold
Down = bBH2CmbRadiopsLsstgold - sBbH2CmbRadiopsLsstgold
ax2.fill_between(llMax, Down, Up, edgecolor='', facecolor=cRadiops, alpha=0.4)
# all
ax2.plot(llMax, bBH2CmbAllLsstgold, cAll, label='All')
Up = bBH2CmbAllLsstgold + sBbH2CmbAllLsstgold
Down = bBH2CmbAllLsstgold - sBbH2CmbAllLsstgold
ax2.fill_between(llMax, Down, Up, edgecolor='', facecolor=cAll, alpha=0.4)
#
ax2.set_ylim((-0.025, 0.025))
ax2.set_ylabel('PPH', fontsize='x-small')
#
plt.setp(ax2.get_xticklabels(), visible=False)
# remove last tick label for the second subplot
yticks = ax2.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)


# Shear
ax3=plt.subplot(gs[3], sharex=ax2)
#
ax3.axhline(0., c='k', lw=1)
ax3.fill_between(llMax, -sSCmbLsstgold, sSCmbLsstgold, edgecolor='', facecolor='gray', alpha=0.6)
# CIB
ax3.plot(llMax, bSCmbCibLsstgold, cCib, label='CIB')
Up = bSCmbCibLsstgold + sBSCmbCibLsstgold
Down = bSCmbCibLsstgold - sBSCmbCibLsstgold
ax3.fill_between(llMax, Down, Up, edgecolor='', facecolor=cCib, alpha=0.4)
# tSZ
ax3.plot(llMax, bSCmbTszLsstgold, cTsz, label='tSZ')
Up = bSCmbTszLsstgold + sBSCmbTszLsstgold
Down = bSCmbTszLsstgold - sBSCmbTszLsstgold
ax3.fill_between(llMax, Down, Up, edgecolor='', facecolor=cTsz, alpha=0.4)
# kSZ
ax3.plot(llMax, bSCmbKszLsstgold, cKsz, label='kSZ')
Up = bSCmbKszLsstgold + sBSCmbKszLsstgold
Down = bSCmbKszLsstgold - sBSCmbKszLsstgold
ax3.fill_between(llMax, Down, Up, edgecolor='', facecolor=cKsz, alpha=0.4)
# radio point sources
ax3.plot(llMax, bSCmbRadiopsLsstgold, cRadiops, label='Radiops')
Up = bSCmbRadiopsLsstgold + sBSCmbRadiopsLsstgold
Down = bSCmbRadiopsLsstgold - sBSCmbRadiopsLsstgold
ax3.fill_between(llMax, Down, Up, edgecolor='', facecolor=cRadiops, alpha=0.4)
# all
ax3.plot(llMax, bSCmbAllLsstgold, cAll, label='All')
Up = bSCmbAllLsstgold + sBSCmbAllLsstgold
Down = bSCmbAllLsstgold - sBSCmbAllLsstgold
ax3.fill_between(llMax, Down, Up, edgecolor='', facecolor=cAll, alpha=0.4)
#
ax3.set_ylim((-0.025, 0.025))
ax3.set_ylabel(r'Shear', fontsize='x-small')
#
plt.setp(ax3.get_xticklabels(), visible=False)
# remove last tick label for the second subplot
yticks = ax3.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)


# QE
ax4=plt.subplot(gs[4], sharex=ax3)
#
ax4.axhline(0., c='k', lw=1)
ax4.fill_between(llMax, -sQCmbLsstgold, sQCmbLsstgold, edgecolor='', facecolor='gray', alpha=0.6)
# CIB
ax4.plot(llMax, bQCmbCibLsstgold, cCib, label='CIB')
Up = bQCmbCibLsstgold + sBQCmbCibLsstgold
Down = bQCmbCibLsstgold - sBQCmbCibLsstgold
ax4.fill_between(llMax, Down, Up, edgecolor='', facecolor=cCib, alpha=0.4)
# tSZ
ax4.plot(llMax, bQCmbTszLsstgold, cTsz, label='tSZ')
Up = bQCmbTszLsstgold + sBQCmbTszLsstgold
Down = bQCmbTszLsstgold - sBQCmbTszLsstgold
ax4.fill_between(llMax, Down, Up, edgecolor='', facecolor=cTsz, alpha=0.4)
# kSZ
ax4.plot(llMax, bQCmbKszLsstgold, cKsz, label='kSZ')
Up = bQCmbKszLsstgold + sBQCmbKszLsstgold
Down = bQCmbKszLsstgold - sBQCmbKszLsstgold
ax4.fill_between(llMax, Down, Up, edgecolor='', facecolor=cKsz, alpha=0.4)
# radio point sources
ax4.plot(llMax, bQCmbRadiopsLsstgold, cRadiops, label='Radiops')
Up = bQCmbRadiopsLsstgold + sBQCmbRadiopsLsstgold
Down = bQCmbRadiopsLsstgold - sBQCmbRadiopsLsstgold
ax4.fill_between(llMax, Down, Up, edgecolor='', facecolor=cRadiops, alpha=0.4)
# all
ax4.plot(llMax, bQCmbAllLsstgold, cAll, label='All')
Up = bQCmbAllLsstgold + sBQCmbAllLsstgold
Down = bQCmbAllLsstgold - sBQCmbAllLsstgold
ax4.fill_between(llMax, Down, Up, edgecolor='', facecolor=cAll, alpha=0.4)
#
ax4.set_ylim((-0.025,0.025))
ax4.set_ylabel(r'QE', fontsize='x-small')
#
# remove last tick label for the second subplot
yticks = ax4.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)
#
ax4.set_xlim((llMax[0], llMax[-1]))
ax4.set_xlabel(r'$\ell_{\text{max, T}}$')


path = "output/summary_bias_Lsstgold.pdf"
fig.savefig(path, bbox_inches='tight')
fig.clf()


##################################################################################
##################################################################################
# Summary for each estimator: bias on k_CMB amplitude

fig=plt.figure(0)
gs = gridspec.GridSpec(5, 1)
gs.update(hspace=0.)

sSCmb = 1./snrSCmb
sQCmb = 1./snrQCmb
sBHCmb = 1./snrBHCmb
sBHgCmb = 1./snrBHgCmb
sBH2Cmb = 1./snrBH2Cmb


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
ax0.set_ylabel('PSH', fontsize='x-small')
#
plt.setp(ax0.get_xticklabels(), visible=False)
ax0.legend(loc='upper center', ncol=5, mode=None, handlelength=.5, fontsize='x-small', borderpad=0.1, borderaxespad=0.1, frameon=False, columnspacing=.8)
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
ax1.set_ylabel('PH', fontsize='x-small')
#
plt.setp(ax1.get_xticklabels(), visible=False)
# remove last tick label for the second subplot
yticks = ax1.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)


# Bias hardened point sources + profile
ax2=plt.subplot(gs[2], sharex=ax1)
#
ax2.axhline(0., c='k', lw=1)
ax2.fill_between(llMax, -sBH2Cmb, sBH2Cmb, edgecolor='', facecolor='gray', alpha=0.6)
# CIB
ax2.plot(llMax, bBH2CmbCibFull, cCib, label='CIB')
Up = bBH2CmbCibFull + sBbH2CmbCibFull
Down = bBH2CmbCibFull - sBbH2CmbCibFull
ax2.fill_between(llMax, Down, Up, edgecolor='', facecolor=cCib, alpha=0.4)
# tSZ
ax2.plot(llMax, bBH2CmbTszFull, cTsz, label='tSZ')
Up = bBH2CmbTszFull + sBbH2CmbTszFull
Down = bBH2CmbTszFull - sBbH2CmbTszFull
ax2.fill_between(llMax, Down, Up, edgecolor='', facecolor=cTsz, alpha=0.4)
# kSZ
ax2.plot(llMax, bBH2CmbKszFull, cKsz, label='kSZ')
Up = bBH2CmbKszFull + sBbH2CmbKszFull
Down = bBH2CmbKszFull - sBbH2CmbKszFull
ax2.fill_between(llMax, Down, Up, edgecolor='', facecolor=cKsz, alpha=0.4)
# radio point sources
ax2.plot(llMax, bBH2CmbRadiopsFull, cRadiops, label='Radiops')
Up = bBH2CmbRadiopsFull + sBbH2CmbRadiopsFull
Down = bBH2CmbRadiopsFull - sBbH2CmbRadiopsFull
ax2.fill_between(llMax, Down, Up, edgecolor='', facecolor=cRadiops, alpha=0.4)
# all
ax2.plot(llMax, bBH2CmbAllFull, cAll, label='All')
Up = bBH2CmbAllFull + sBbH2CmbAllFull
Down = bBH2CmbAllFull - sBbH2CmbAllFull
ax2.fill_between(llMax, Down, Up, edgecolor='', facecolor=cAll, alpha=0.4)
#
ax2.set_ylim((-0.025, 0.025))
ax2.set_ylabel('PPH', fontsize='x-small')
#
plt.setp(ax2.get_xticklabels(), visible=False)
# remove last tick label for the second subplot
yticks = ax2.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)


# Shear
ax3=plt.subplot(gs[3], sharex=ax2)
#
ax3.axhline(0., c='k', lw=1)
ax3.fill_between(llMax, -sSCmb, sSCmb, edgecolor='', facecolor='gray', alpha=0.6)
# CIB
ax3.plot(llMax, bSCmbCibFull, cCib, label='CIB')
Up = bSCmbCibFull + sBSCmbCibFull
Down = bSCmbCibFull - sBSCmbCibFull
ax3.fill_between(llMax, Down, Up, edgecolor='', facecolor=cCib, alpha=0.4)
# tSZ
ax3.plot(llMax, bSCmbTszFull, cTsz, label='tSZ')
Up = bSCmbTszFull + sBSCmbTszFull
Down = bSCmbTszFull - sBSCmbTszFull
ax3.fill_between(llMax, Down, Up, edgecolor='', facecolor=cTsz, alpha=0.4)
# kSZ
ax3.plot(llMax, bSCmbKszFull, cKsz, label='kSZ')
Up = bSCmbKszFull + sBSCmbKszFull
Down = bSCmbKszFull - sBSCmbKszFull
ax3.fill_between(llMax, Down, Up, edgecolor='', facecolor=cKsz, alpha=0.4)
# radio point sources
ax3.plot(llMax, bSCmbRadiopsFull, cRadiops, label='Radiops')
Up = bSCmbRadiopsFull + sBSCmbRadiopsFull
Down = bSCmbRadiopsFull - sBSCmbRadiopsFull
ax3.fill_between(llMax, Down, Up, edgecolor='', facecolor=cRadiops, alpha=0.4)
# all
ax3.plot(llMax, bSCmbAllFull, cAll, label='All')
Up = bSCmbAllFull + sBSCmbAllFull
Down = bSCmbAllFull - sBSCmbAllFull
ax3.fill_between(llMax, Down, Up, edgecolor='', facecolor=cAll, alpha=0.4)
#
ax3.set_ylim((-0.025, 0.025))
ax3.set_ylabel(r'Shear', fontsize='x-small')
#
plt.setp(ax3.get_xticklabels(), visible=False)
# remove last tick label for the second subplot
yticks = ax3.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)


# QE
ax4=plt.subplot(gs[4], sharex=ax3)
#
ax4.axhline(0., c='k', lw=1)
ax4.fill_between(llMax, -sQCmb, sQCmb, edgecolor='', facecolor='gray', alpha=0.6)
# CIB
ax4.plot(llMax, bQCmbCibFull, cCib, label='CIB')
Up = bQCmbCibFull + sBQCmbCibFull
Down = bQCmbCibFull - sBQCmbCibFull
ax4.fill_between(llMax, Down, Up, edgecolor='', facecolor=cCib, alpha=0.4)
# tSZ
ax4.plot(llMax, bQCmbTszFull, cTsz, label='tSZ')
Up = bQCmbTszFull + sBQCmbTszFull
Down = bQCmbTszFull - sBQCmbTszFull
ax4.fill_between(llMax, Down, Up, edgecolor='', facecolor=cTsz, alpha=0.4)
# kSZ
ax4.plot(llMax, bQCmbKszFull, cKsz, label='kSZ')
Up = bQCmbKszFull + sBQCmbKszFull
Down = bQCmbKszFull - sBQCmbKszFull
ax4.fill_between(llMax, Down, Up, edgecolor='', facecolor=cKsz, alpha=0.4)
# radio point sources
ax4.plot(llMax, bQCmbRadiopsFull, cRadiops, label='Radiops')
Up = bQCmbRadiopsFull + sBQCmbRadiopsFull
Down = bQCmbRadiopsFull - sBQCmbRadiopsFull
ax4.fill_between(llMax, Down, Up, edgecolor='', facecolor=cRadiops, alpha=0.4)
# all
ax4.plot(llMax, bQCmbAllFull, cAll, label='All')
Up = bQCmbAllFull + sBQCmbAllFull
Down = bQCmbAllFull - sBQCmbAllFull
ax4.fill_between(llMax, Down, Up, edgecolor='', facecolor=cAll, alpha=0.4)
#
ax4.set_ylim((-0.025,0.025))
ax4.set_ylabel(r'QE', fontsize='x-small')
# remove last tick label for the second subplot
yticks = ax4.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)
#
ax4.set_xlim((llMax[0], llMax[-1]))
ax4.set_xlabel(r'$\ell_{\text{max, T}}$')

path = "output/summary_bias_full.pdf"
fig.savefig(path, bbox_inches='tight')
fig.clf()


import sys
sys.exit()

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

# Bias-hardened with profile
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

# Bias-hardened point sources + profile
# Auto
f = interp1d(llMax, bBH2CmbAllFull)
bBH2CmbAllFull = f(llMaxc)
f = interp1d(llMax, snrBH2Cmb)
snrBH2Cmb = f(llMaxc)
# Cross
f = interp1d(llMax, bBH2CmbAllLsstgold)
bBH2CmbAllLsstgold = f(llMaxc)
f = interp1d(llMax, 1./sBH2CmbLsstgold)
snrBH2CmbLsstgold = f(llMaxc)


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
ax.scatter(llMaxc[OneSigmaBias], snrBHCmb[OneSigmaBias], c='k', marker='d', label='PSH',zorder=2)
print(llMaxc[OneSigmaBias], snrBHCmb[OneSigmaBias])

# Bias hardened with profile
points = np.array([llMaxc, snrBHgCmb]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
biasInSigmaUnits = np.abs(bBHgCmbAllFull * snrBHgCmb)
OneSigmaBias = np.where(np.abs(biasInSigmaUnits) >= 1.0)[0][0]
norm = plt.Normalize(0, 2.)
lc = LineCollection(segments, cmap='rainbow', norm=norm)
lc.set_array(biasInSigmaUnits)
lc.set_linewidth(3)
ax.add_collection(lc)
ax.scatter(llMaxc[OneSigmaBias], snrBHgCmb[OneSigmaBias], c='k', marker='*', label='PH',zorder=2)
print(llMaxc[OneSigmaBias], snrBHgCmb[OneSigmaBias])

# Bias hardened point sources + profile
points = np.array([llMaxc, snrBH2Cmb]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
biasInSigmaUnits = np.abs(bBH2CmbAllFull * snrBH2Cmb)
OneSigmaBias = np.where(np.abs(biasInSigmaUnits) >= 1.0)[0][0]
norm = plt.Normalize(0, 2.)
lc = LineCollection(segments, cmap='rainbow', norm=norm)
lc.set_array(biasInSigmaUnits)
lc.set_linewidth(3)
ax.add_collection(lc)
ax.scatter(llMaxc[OneSigmaBias], snrBH2Cmb[OneSigmaBias], c='k', marker='P', label='PPH',zorder=2)
print(llMaxc[OneSigmaBias], snrBH2Cmb[OneSigmaBias])

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
print(llMaxc[-1], snrSCmb[-1])

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
print(llMaxc[OneSigmaBias], snrQCmb[OneSigmaBias])

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
order = [0,1,2,3,4,5]
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
ax.scatter(llMaxc[OneSigmaBias], snrBHCmbLsstgold[OneSigmaBias], c='k', marker='d', label='PSH',zorder=2)
print(llMaxc[OneSigmaBias], snrBHCmbLsstgold[OneSigmaBias])

# Bias hardened with profile
points = np.array([llMaxc, snrBHgCmbLsstgold]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
biasInSigmaUnits = np.abs(bBHgCmbAllLsstgold * snrBHgCmbLsstgold)
OneSigmaBias = np.where(np.abs(biasInSigmaUnits) >= 1.0)[0][0]
norm = plt.Normalize(0, 2.)
lc = LineCollection(segments, cmap='rainbow', norm=norm)
lc.set_array(biasInSigmaUnits)
lc.set_linewidth(3)
ax.add_collection(lc)
ax.scatter(llMaxc[OneSigmaBias], snrBHgCmbLsstgold[OneSigmaBias], c='k', marker='*', label='PH',zorder=2)
print(llMaxc[OneSigmaBias], snrBHgCmbLsstgold[OneSigmaBias])

# Bias hardened point sources + profile
points = np.array([llMaxc, snrBH2CmbLsstgold]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
biasInSigmaUnits = np.abs(bBH2CmbAllLsstgold * snrBH2CmbLsstgold)
norm = plt.Normalize(0, 2.)
lc = LineCollection(segments, cmap='rainbow', norm=norm)
lc.set_array(biasInSigmaUnits)
lc.set_linewidth(3)
ax.add_collection(lc)
ax.scatter(llMaxc[-1], snrBH2CmbLsstgold[-1], c='k', marker='P', label='PPH',zorder=2)
print(llMaxc[-1], snrBH2CmbLsstgold[-1])

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
print(llMaxc[OneSigmaBias], snrSCmbLsstgold[OneSigmaBias])

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
print(llMaxc[OneSigmaBias], snrQCmbLsstgold[OneSigmaBias])

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
order = [0,1,2,3,4,5]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize='x-small', labelspacing=0.2,loc=2)
#ax.set_title(r'SNR of $C^{\kappa\times\text{ LSST }}_L$ amplitude')
#
path = "/home/noah/Documents/Berkeley/LensQuEst-1/output/compare_snr_cross_lmax.pdf"
fig.savefig(path, bbox_inches='tight')
fig.clf()
