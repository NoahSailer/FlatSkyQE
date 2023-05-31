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

l = np.genfromtxt('l.txt')
profile = np.genfromtxt('profile.txt')
uTsz = interp1d(l,profile,kind='linear')

##################################################################################
##################################################################################
# Calculate SNR and Full = Prim + Sec + trispectrum

path = "bias_output/lCen_lmaxT_2500.txt"
lCen = np.genfromtxt(path)

LMin = 20.
LMax = 1.e3

llMax = np.array([2000.,2500.,3000.,3500., 4000.])

snrQCmb = np.zeros(len(llMax))
snrBHCmb = np.zeros(len(llMax))
sQCmbLsstgold = np.zeros(len(llMax))
sBHCmbLsstgold = np.zeros(len(llMax))

snrBHCmb22 = np.zeros(len(llMax))
snrBH2Cmb22 = np.zeros(len(llMax))
snrBHCmb14 = np.zeros(len(llMax))
snrBH2Cmb14 = np.zeros(len(llMax))
snrTSZ = np.zeros(len(llMax))

sBHCmb22Lsstgold = np.zeros(len(llMax))
sBH2Cmb22Lsstgold = np.zeros(len(llMax))
sBHCmb14Lsstgold = np.zeros(len(llMax))
sBH2Cmb14Lsstgold = np.zeros(len(llMax))
sTSZLsstgold = np.zeros(len(llMax))

for ilMax in range(len(llMax)):

   lMax = llMax[ilMax]


   cmb = CMB(beam=1.4, noise=6., lMin=lMin, lMaxT=lMax, lMaxP=lMax, atm=False, fg=True)
   L = np.logspace(np.log10(lMin/2.), np.log10(2.*lMax), 1001, 10.)
   F = np.array(list(map(cmb.ftotalTT, L)))
   cmb.fCtotal = interp1d(L, F, kind='linear', bounds_error=False, fill_value=0.)

   fNqCmb_fft = baseMap.forecastN0Kappa(cmb.flensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False)
   fNqBHCmb_fft = baseMap.forecastN0KappaBH(cmb.flensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False)
   clusterRadius = 2.2 * np.pi/(180. * 60.) # radians
   fNqBHCmb22_fft = baseMap.forecastN0KappaBH(cmb.flensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False, cache=None, sigma=clusterRadius)
   fNqBH2Cmb22_fft = baseMap.forecastN0KappaBH2(cmb.flensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False, cache=None, sigma=clusterRadius)
   clusterRadius = 1.4 * np.pi/(180. * 60.) # radians
   fNqBHCmb14_fft = baseMap.forecastN0KappaBH(cmb.flensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False, cache=None, sigma=clusterRadius)
   fNqBH2Cmb14_fft = baseMap.forecastN0KappaBH2(cmb.flensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False, cache=None, sigma=clusterRadius)
   fTsz2_fft = baseMap.forecastN0KappaBH2(cmb.flensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False, cache=None, u = uTsz)   

   ell = baseMap.l.flatten()
   lEdges = np.logspace(np.log10(1.), np.log10(np.max(ell)), nBins, 10.)
   Nmodes = lEdges[1:]**2. - lEdges[:-1]**2.   
   I = np.where((lCen>=LMin)*(lCen<=LMax))

   ClkCmb = p2d_cmblens.fPinterp(lCen)
   Ckg = p2d_lsstgoldcmblens.fPtotinterp(lCen)
   Cgg = p2d_lsstgold.fPtotinterp(lCen)

   NqCmb = fNqCmb_fft(lCen)  
   NbhCmb = fNqBHCmb_fft(lCen)

   NbhCmb22 = fNqBHCmb22_fft(lCen)
   Nbh2Cmb22 = fNqBH2Cmb22_fft(lCen)
   NbhCmb14 = fNqBHCmb14_fft(lCen)
   Nbh2Cmb14 = fNqBH2Cmb14_fft(lCen)
   Ntsz = fTsz2_fft(lCen)  

   # QE
   s2 =  2. * (ClkCmb + NqCmb)**2. / Nmodes 
   norm = np.sum(ClkCmb[I]**2. / s2[I])
   #
   snrQCmb[ilMax] = np.sqrt(norm)
   #
   #
   s2 = ((ClkCmb + NqCmb)*Cgg + (Ckg)**2) / Nmodes
   norm = np.sum(Ckg[I]**2 / s2[I])
   #
   sQCmbLsstgold[ilMax] = np.sqrt(norm)

   # BH 0
   s2 =  2. * (ClkCmb + NbhCmb)**2. / Nmodes 
   norm = np.sum(ClkCmb[I]**2. / s2[I])
   #
   snrBHCmb[ilMax] = np.sqrt(norm)
   #
   #
   s2 = ((ClkCmb + NbhCmb)*Cgg + (Ckg)**2) / Nmodes
   norm = np.sum(Ckg[I]**2 / s2[I])
   #
   sBHCmbLsstgold[ilMax] = np.sqrt(norm)
   
   # BH 2.2
   s2 =  2. * (ClkCmb + NbhCmb22)**2 / Nmodes 
   norm = np.sum(ClkCmb[I]**2 / s2[I])
   #
   snrBHCmb22[ilMax] = np.sqrt(norm)
   #
   #
   s2 = ((ClkCmb + NbhCmb22)*Cgg + (Ckg)**2) / Nmodes
   norm = np.sum(Ckg[I]**2 / s2[I])
   #
   sBHCmb22Lsstgold[ilMax] = np.sqrt(norm)

   # BH 1.4
   s2 =  2. * (ClkCmb + NbhCmb14)**2 / Nmodes 
   norm = np.sum(ClkCmb[I]**2 / s2[I])
   #
   snrBHCmb14[ilMax] = np.sqrt(norm)
   #
   #
   s2 = ((ClkCmb + NbhCmb14)*Cgg + (Ckg)**2) / Nmodes
   norm = np.sum(Ckg[I]**2 / s2[I])
   #
   sBHCmb14Lsstgold[ilMax] = np.sqrt(norm)

   # BH2 2.2
   s2 =  2. * (ClkCmb + Nbh2Cmb22)**2 / Nmodes 
   norm = np.sum(ClkCmb[I]**2 / s2[I])
   #
   snrBH2Cmb22[ilMax] = np.sqrt(norm)
   #
   #
   s2 = ((ClkCmb + Nbh2Cmb22)*Cgg + (Ckg)**2) / Nmodes
   norm = np.sum(Ckg[I]**2 / s2[I])
   #
   sBH2Cmb22Lsstgold[ilMax] = np.sqrt(norm)

   # BH2 1.4
   s2 =  2. * (ClkCmb + Nbh2Cmb14)**2 / Nmodes 
   norm = np.sum(ClkCmb[I]**2 / s2[I])
   #
   snrBH2Cmb14[ilMax] = np.sqrt(norm)
   #
   #
   s2 = ((ClkCmb + Nbh2Cmb14)*Cgg + (Ckg)**2) / Nmodes
   norm = np.sum(Ckg[I]**2 / s2[I])
   #
   sBH2Cmb14Lsstgold[ilMax] = np.sqrt(norm)

   # tsz profile
   s2 =  2. * (ClkCmb + Ntsz)**2 / Nmodes 
   norm = np.sum(ClkCmb[I]**2 / s2[I])
   #
   snrTSZ[ilMax] = np.sqrt(norm)
   #
   #
   s2 = ((ClkCmb + Ntsz)*Cgg + (Ckg)**2) / Nmodes
   norm = np.sum(Ckg[I]**2 / s2[I])
   #
   sTSZLsstgold[ilMax] = np.sqrt(norm)


##################################################################################
##################################################################################
# Plotting SNR for the C^kappa amplitude

plt.plot(llMax,snrQCmb,c='k',label='QE')
plt.plot(llMax,snrBHCmb,c='r',label='BH1')
plt.plot(llMax,snrBHCmb22,c='g',label=r'BHG$(\sigma=2.2)$')
plt.plot(llMax,snrBH2Cmb22,c='g',linestyle='--',label=r'BH2$(\sigma=2.2)$')
plt.plot(llMax,snrBHCmb14,c='b',label=r'BHG$(\sigma=1.4)$')
plt.plot(llMax,snrBH2Cmb14,c='b',linestyle='--',label=r'BH2$(\sigma=1.4)$')
plt.plot(llMax,snrTSZ,c='purple',label='tsz')

plt.legend(fontsize='x-small', labelspacing=0.2,loc=2)
plt.xlabel(r'$\ell_{\text{max},T}$')
plt.ylabel(r'SNR of $C^{\kappa}_L$ amplitude')
path = "/home/noah/Documents/Berkeley/LensQuEst-1/output/14vs22_snr_lmax.pdf"
plt.savefig(path, bbox_inches='tight')
plt.clf()


##################################################################################
##################################################################################
# Plotting SNR for the C^kappa x LSST amplitude

plt.plot(llMax,sQCmbLsstgold,c='k',label='QE')
plt.plot(llMax,sBHCmbLsstgold,c='r',label='BH1')
plt.plot(llMax,sBHCmb22Lsstgold,c='g',label=r'BHG$(\sigma=2.2)$')
plt.plot(llMax,sBH2Cmb22Lsstgold,c='g',linestyle='--',label=r'BH2$(\sigma=2.2)$')
plt.plot(llMax,sBHCmb14Lsstgold,c='b',label=r'BHG$(\sigma=1.4)$')
plt.plot(llMax,sBH2Cmb14Lsstgold,c='b',linestyle='--',label=r'BH2$(\sigma=1.4)$')
plt.plot(llMax,sTSZLsstgold,c='purple',label='tsz')

plt.legend(fontsize='x-small', labelspacing=0.2,loc=2)
plt.xlabel(r'$\ell_{\text{max},T}$')
plt.ylabel(r'SNR of $C^{\kappa\times \text{LSST}}_L$ amplitude')
path = "/home/noah/Documents/Berkeley/LensQuEst-1/output/14vs22_snr_cross_lmax.pdf"
plt.savefig(path, bbox_inches='tight')
plt.clf()
