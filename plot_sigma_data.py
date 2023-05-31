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


#################################################################################
c10 = 'r'
c12 = 'orange'
c15 = 'yellow'
c17 = 'green'
c20 = 'blue'
c22 = 'indigo'
c25 = 'violet'
c30 = 'brown'
c35 = 'black'



#################################################################################
# rescale the error bars with area,
# because of the weird octant thing in the Sehgal sims

octantArea = 4.*np.pi * (180./np.pi)**2   # deg 2
octantArea /= 8.
ourArea = 81. * 20.**2  # deg 2
uncertaintyFactor = np.sqrt(ourArea / octantArea)


##################################################################################
##################################################################################
# Calculate SNR and Full = Prim + Sec + trispectrum

LMin = 20.
LMax = 1.e3

llMax = np.array([2500.,3000.,3500., 4000.])

snrQ10Cmb = np.zeros(len(llMax))
snrQ12Cmb = np.zeros(len(llMax))
snrQ15Cmb = np.zeros(len(llMax))
snrQ17Cmb = np.zeros(len(llMax))
snrQ20Cmb = np.zeros(len(llMax))
snrQ22Cmb = np.zeros(len(llMax))
snrQ25Cmb = np.zeros(len(llMax))
snrQ30Cmb = np.zeros(len(llMax))
snrQ35Cmb = np.zeros(len(llMax))
#
sQ10CmbLsstgold = np.zeros(len(llMax))
sQ12CmbLsstgold = np.zeros(len(llMax))
sQ15CmbLsstgold = np.zeros(len(llMax))
sQ17CmbLsstgold = np.zeros(len(llMax))
sQ20CmbLsstgold = np.zeros(len(llMax))
sQ22CmbLsstgold = np.zeros(len(llMax))
sQ25CmbLsstgold = np.zeros(len(llMax))
sQ30CmbLsstgold = np.zeros(len(llMax))
sQ35CmbLsstgold = np.zeros(len(llMax))

bQ10CmbTszFull = np.array([None]*len(llMax))
bQ12CmbTszFull = np.array([None]*len(llMax))
bQ15CmbTszFull = np.array([None]*len(llMax))
bQ17CmbTszFull = np.array([None]*len(llMax))
bQ20CmbTszFull = np.array([None]*len(llMax))
bQ22CmbTszFull = np.array([None]*len(llMax))
bQ25CmbTszFull = np.array([None]*len(llMax))
bQ30CmbTszFull = np.array([None]*len(llMax))
bQ35CmbTszFull = np.array([None]*len(llMax))

bQ10CmbTszLsstgold = np.array([None]*len(llMax))
bQ12CmbTszLsstgold = np.array([None]*len(llMax))
bQ15CmbTszLsstgold = np.array([None]*len(llMax))
bQ17CmbTszLsstgold = np.array([None]*len(llMax))
bQ20CmbTszLsstgold = np.array([None]*len(llMax))
bQ22CmbTszLsstgold = np.array([None]*len(llMax))
bQ25CmbTszLsstgold = np.array([None]*len(llMax))
bQ30CmbTszLsstgold = np.array([None]*len(llMax))
bQ35CmbTszLsstgold = np.array([None]*len(llMax))

for ilMax in range(len(llMax)):

   lMax = llMax[ilMax]

   # Get data
   nPatch = 80
   outputFolder = "sigma_output"


   datalist10 = []
   datalist12 = []
   datalist15 = []
   datalist17 = []
   datalist20 = []
   datalist22 = []
   datalist25 = []
   datalist30 = []
   datalist35 = []


   for i in range(nPatch):
      try:
         tmpfile = np.genfromtxt(outputFolder+"/data_lmaxT_"+str(int(lMax))+"_radius_10_"+str(i)+".txt") 
         datalist10.append(tmpfile)
      except:
         print('')

      try:
         tmpfile = np.genfromtxt(outputFolder+"/data_lmaxT_"+str(int(lMax))+"_radius_12_"+str(i)+".txt") 
         datalist12.append(tmpfile)
      except:
         print('')

      try:
         tmpfile = np.genfromtxt(outputFolder+"/data_lmaxT_"+str(int(lMax))+"_radius_15_"+str(i)+".txt") 
         datalist15.append(tmpfile)
      except:
         print('')

      try:
         tmpfile = np.genfromtxt(outputFolder+"/data_lmaxT_"+str(int(lMax))+"_radius_17_"+str(i)+".txt") 
         datalist17.append(tmpfile)
      except:
         print('')
     
      try:
         tmpfile = np.genfromtxt(outputFolder+"/data_lmaxT_"+str(int(lMax))+"_radius_20_"+str(i)+".txt")
         datalist20.append(tmpfile)
      except:
         print('')

      try:
         tmpfile = np.genfromtxt(outputFolder+"/data_lmaxT_"+str(int(lMax))+"_radius_22_"+str(i)+".txt")
         datalist22.append(tmpfile)
      except:
         print('')

      try:
         tmpfile = np.genfromtxt(outputFolder+"/data_lmaxT_"+str(int(lMax))+"_radius_25_"+str(i)+".txt")
         datalist25.append(tmpfile)
      except:
         print('')

      try:
         tmpfile = np.genfromtxt(outputFolder+"/data_lmaxT_"+str(int(lMax))+"_radius_30_"+str(i)+".txt") 
         datalist30.append(tmpfile)
      except:
         print('')
      
      try:
         tmpfile = np.genfromtxt(outputFolder+"/data_lmaxT_"+str(int(lMax))+"_radius_35_"+str(i)+".txt")
         datalist35.append(tmpfile)
      except:
         print('')

   datalist10 = np.array(datalist10)
   datalist12 = np.array(datalist12)
   datalist15 = np.array(datalist15)
   datalist17 = np.array(datalist17)
   datalist20 = np.array(datalist20)
   datalist22 = np.array(datalist22)
   datalist25 = np.array(datalist25)
   datalist30 = np.array(datalist30)
   datalist35 = np.array(datalist35)


   cmb = CMB(beam=1.4, noise=6., lMin=lMin, lMaxT=lMax, lMaxP=lMax, atm=False, fg=True)
   L = np.logspace(np.log10(lMin/2.), np.log10(2.*lMax), 1001, 10.)
   F = np.array(list(map(cmb.ftotalTT, L)))
   cmb.fCtotal = interp1d(L, F, kind='linear', bounds_error=False, fill_value=0.)


   clusterRadius = 1.0 * np.pi/(180. * 60.) # radians
   fNq10Cmb_fft = baseMap.forecastN0KappaBH(cmb.flensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False, sigma=clusterRadius)
   clusterRadius = 1.25 * np.pi/(180. * 60.) # radians
   fNq12Cmb_fft = baseMap.forecastN0KappaBH(cmb.flensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False, sigma=clusterRadius)
   clusterRadius = 1.5 * np.pi/(180. * 60.) # radians
   fNq15Cmb_fft = baseMap.forecastN0KappaBH(cmb.flensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False, sigma=clusterRadius)
   clusterRadius = 1.75 * np.pi/(180. * 60.) # radians
   fNq17Cmb_fft = baseMap.forecastN0KappaBH(cmb.flensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False, sigma=clusterRadius)
   clusterRadius = 2. * np.pi/(180. * 60.) # radians
   fNq20Cmb_fft = baseMap.forecastN0KappaBH(cmb.flensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False, sigma=clusterRadius)
   clusterRadius = 2.25 * np.pi/(180. * 60.) # radians
   fNq22Cmb_fft = baseMap.forecastN0KappaBH(cmb.flensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False, sigma=clusterRadius)
   clusterRadius = 2.5 * np.pi/(180. * 60.) # radians
   fNq25Cmb_fft = baseMap.forecastN0KappaBH(cmb.flensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False, sigma=clusterRadius)
   clusterRadius = 3. * np.pi/(180. * 60.) # radians
   fNq30Cmb_fft = baseMap.forecastN0KappaBH(cmb.flensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False, sigma=clusterRadius)
   clusterRadius = 3.5 * np.pi/(180. * 60.) # radians
   fNq35Cmb_fft = baseMap.forecastN0KappaBH(cmb.flensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False, sigma=clusterRadius)

   path = outputFolder+"/lCen_lmaxT_"+str(int(lMax))+".txt"
   lCen = np.genfromtxt(path)
   ell = baseMap.l.flatten()
   lEdges = np.logspace(np.log10(1.), np.log10(np.max(ell)), nBins, 10.)
   Nmodes = lEdges[1:]**2. - lEdges[:-1]**2.   
   I = np.where((lCen>=LMin)*(lCen<=LMax))

   ClkCmb = p2d_cmblens.fPinterp(lCen)
   Nq10Cmb = fNq10Cmb_fft(lCen)
   Nq12Cmb = fNq12Cmb_fft(lCen)
   Nq15Cmb = fNq15Cmb_fft(lCen)
   Nq17Cmb = fNq17Cmb_fft(lCen)
   Nq20Cmb = fNq20Cmb_fft(lCen) 
   Nq22Cmb = fNq22Cmb_fft(lCen) 
   Nq25Cmb = fNq25Cmb_fft(lCen) 
   Nq30Cmb = fNq30Cmb_fft(lCen) 
   Nq35Cmb = fNq35Cmb_fft(lCen)   
   Ckg = p2d_lsstgoldcmblens.fPtotinterp(lCen)
   Cgg = p2d_lsstgold.fPtotinterp(lCen)

   ##################################################################################
   # Spectra info

   # tSZ
   # Trispectrum
   Clq10CmbTsz = datalist10[:,0]
   Clq12CmbTsz = datalist12[:,0]
   Clq15CmbTsz = datalist15[:,0]
   Clq17CmbTsz = datalist17[:,0]
   Clq20CmbTsz = datalist20[:,0]
   Clq22CmbTsz = datalist22[:,0]
   Clq25CmbTsz = datalist25[:,0]
   Clq30CmbTsz = datalist30[:,0]
   Clq35CmbTsz = datalist35[:,0]
   # Primary
   Clq10CmbTszKappa = datalist10[:,1]
   Clq12CmbTszKappa = datalist12[:,1]
   Clq15CmbTszKappa = datalist15[:,1]
   Clq17CmbTszKappa = datalist17[:,1]
   Clq20CmbTszKappa = datalist20[:,1]
   Clq22CmbTszKappa = datalist22[:,1]
   Clq25CmbTszKappa = datalist25[:,1]
   Clq30CmbTszKappa = datalist30[:,1]
   Clq35CmbTszKappa = datalist35[:,1]
   # k_rec x LSST
   Clq10CmbTszLsstgold = datalist10[:,2]
   Clq12CmbTszLsstgold = datalist12[:,2]
   Clq15CmbTszLsstgold = datalist15[:,2]
   Clq17CmbTszLsstgold = datalist17[:,2]
   Clq20CmbTszLsstgold = datalist20[:,2]
   Clq22CmbTszLsstgold = datalist22[:,2]
   Clq25CmbTszLsstgold = datalist25[:,2]
   Clq30CmbTszLsstgold = datalist30[:,2]
   Clq35CmbTszLsstgold = datalist35[:,2]
   # Secondary
   Clq10CmbTszSec = datalist10[:,3]
   Clq12CmbTszSec = datalist12[:,3]
   Clq15CmbTszSec = datalist15[:,3]
   Clq17CmbTszSec = datalist17[:,3]
   Clq20CmbTszSec = datalist20[:,3]
   Clq22CmbTszSec = datalist22[:,3]
   Clq25CmbTszSec = datalist25[:,3]
   Clq30CmbTszSec = datalist30[:,3]
   Clq35CmbTszSec = datalist35[:,3]

   # 10
   s2 =  2. * (ClkCmb + Nq10Cmb)**2. / Nmodes 
   norm = np.sum(ClkCmb[I]**2. / s2[I])
   #
   snrQ10Cmb[ilMax] = np.sqrt(norm)
   #
   bQ10CmbTszFull[ilMax] = np.sum((2.*Clq10CmbTszKappa[:,I[0]]+2.*Clq10CmbTszSec[:,I[0]]+Clq10CmbTsz[:,I[0]]) * ClkCmb[I] / s2[I], axis=1) / norm
   #
   s2 = ((ClkCmb + Nq10Cmb)*Cgg + (Ckg)**2) / Nmodes
   norm = np.sum(Ckg[I]**2 / s2[I])
   #
   sQ10CmbLsstgold[ilMax] = 1./np.sqrt(norm)
   #
   bQ10CmbTszLsstgold[ilMax] = np.sum(Clq10CmbTszLsstgold[:,I[0]] * Ckg[I] / s2[I], axis=1) / norm

   # 12
   s2 =  2. * (ClkCmb + Nq12Cmb)**2. / Nmodes 
   norm = np.sum(ClkCmb[I]**2. / s2[I])
   #
   snrQ12Cmb[ilMax] = np.sqrt(norm)
   #
   bQ12CmbTszFull[ilMax] = np.sum((2.*Clq12CmbTszKappa[:,I[0]]+2.*Clq12CmbTszSec[:,I[0]]+Clq12CmbTsz[:,I[0]]) * ClkCmb[I] / s2[I], axis=1) / norm
   #
   s2 = ((ClkCmb + Nq12Cmb)*Cgg + (Ckg)**2) / Nmodes
   norm = np.sum(Ckg[I]**2 / s2[I])
   #
   sQ12CmbLsstgold[ilMax] = 1./np.sqrt(norm)
   #
   bQ12CmbTszLsstgold[ilMax] = np.sum(Clq12CmbTszLsstgold[:,I[0]] * Ckg[I] / s2[I], axis=1) / norm

   # 15
   s2 =  2. * (ClkCmb + Nq15Cmb)**2. / Nmodes 
   norm = np.sum(ClkCmb[I]**2. / s2[I])
   #
   snrQ15Cmb[ilMax] = np.sqrt(norm)
   #
   bQ15CmbTszFull[ilMax] = np.sum((2.*Clq15CmbTszKappa[:,I[0]]+2.*Clq15CmbTszSec[:,I[0]]+Clq15CmbTsz[:,I[0]]) * ClkCmb[I] / s2[I], axis=1) / norm
   #
   s2 = ((ClkCmb + Nq15Cmb)*Cgg + (Ckg)**2) / Nmodes
   norm = np.sum(Ckg[I]**2 / s2[I])
   #
   sQ15CmbLsstgold[ilMax] = 1./np.sqrt(norm)
   #
   bQ15CmbTszLsstgold[ilMax] = np.sum(Clq15CmbTszLsstgold[:,I[0]] * Ckg[I] / s2[I], axis=1) / norm

   # 17
   s2 =  2. * (ClkCmb + Nq17Cmb)**2. / Nmodes 
   norm = np.sum(ClkCmb[I]**2. / s2[I])
   #
   snrQ17Cmb[ilMax] = np.sqrt(norm)
   #
   bQ17CmbTszFull[ilMax] = np.sum((2.*Clq17CmbTszKappa[:,I[0]]+2.*Clq17CmbTszSec[:,I[0]]+Clq17CmbTsz[:,I[0]]) * ClkCmb[I] / s2[I], axis=1) / norm
   #
   s2 = ((ClkCmb + Nq17Cmb)*Cgg + (Ckg)**2) / Nmodes
   norm = np.sum(Ckg[I]**2 / s2[I])
   #
   sQ17CmbLsstgold[ilMax] = 1./np.sqrt(norm)
   #
   bQ17CmbTszLsstgold[ilMax] = np.sum(Clq17CmbTszLsstgold[:,I[0]] * Ckg[I] / s2[I], axis=1) / norm

   # 20
   s2 =  2. * (ClkCmb + Nq20Cmb)**2. / Nmodes 
   norm = np.sum(ClkCmb[I]**2. / s2[I])
   #
   snrQ20Cmb[ilMax] = np.sqrt(norm)
   #
   bQ20CmbTszFull[ilMax] = np.sum((2.*Clq20CmbTszKappa[:,I[0]]+2.*Clq20CmbTszSec[:,I[0]]+Clq20CmbTsz[:,I[0]]) * ClkCmb[I] / s2[I], axis=1) / norm
   #
   s2 = ((ClkCmb + Nq20Cmb)*Cgg + (Ckg)**2) / Nmodes
   norm = np.sum(Ckg[I]**2 / s2[I])
   #
   sQ20CmbLsstgold[ilMax] = 1./np.sqrt(norm)
   #
   bQ20CmbTszLsstgold[ilMax] = np.sum(Clq20CmbTszLsstgold[:,I[0]] * Ckg[I] / s2[I], axis=1) / norm

   # 22
   s2 =  2. * (ClkCmb + Nq22Cmb)**2. / Nmodes 
   norm = np.sum(ClkCmb[I]**2. / s2[I])
   #
   snrQ22Cmb[ilMax] = np.sqrt(norm)
   #
   bQ22CmbTszFull[ilMax] = np.sum((2.*Clq22CmbTszKappa[:,I[0]]+2.*Clq22CmbTszSec[:,I[0]]+Clq22CmbTsz[:,I[0]]) * ClkCmb[I] / s2[I], axis=1) / norm
   #
   s2 = ((ClkCmb + Nq22Cmb)*Cgg + (Ckg)**2) / Nmodes
   norm = np.sum(Ckg[I]**2 / s2[I])
   #
   sQ22CmbLsstgold[ilMax] = 1./np.sqrt(norm)
   #
   bQ22CmbTszLsstgold[ilMax] = np.sum(Clq22CmbTszLsstgold[:,I[0]] * Ckg[I] / s2[I], axis=1) / norm

   # 25
   s2 =  2. * (ClkCmb + Nq25Cmb)**2. / Nmodes 
   norm = np.sum(ClkCmb[I]**2. / s2[I])
   #
   snrQ25Cmb[ilMax] = np.sqrt(norm)
   #
   bQ25CmbTszFull[ilMax] = np.sum((2.*Clq25CmbTszKappa[:,I[0]]+2.*Clq25CmbTszSec[:,I[0]]+Clq25CmbTsz[:,I[0]]) * ClkCmb[I] / s2[I], axis=1) / norm
   #
   s2 = ((ClkCmb + Nq25Cmb)*Cgg + (Ckg)**2) / Nmodes
   norm = np.sum(Ckg[I]**2 / s2[I])
   #
   sQ25CmbLsstgold[ilMax] = 1./np.sqrt(norm)
   #
   bQ25CmbTszLsstgold[ilMax] = np.sum(Clq25CmbTszLsstgold[:,I[0]] * Ckg[I] / s2[I], axis=1) / norm

   # 30
   s2 =  2. * (ClkCmb + Nq30Cmb)**2. / Nmodes 
   norm = np.sum(ClkCmb[I]**2. / s2[I])
   #
   snrQ30Cmb[ilMax] = np.sqrt(norm)
   #
   bQ30CmbTszFull[ilMax] = np.sum((2.*Clq30CmbTszKappa[:,I[0]]+2.*Clq30CmbTszSec[:,I[0]]+Clq30CmbTsz[:,I[0]]) * ClkCmb[I] / s2[I], axis=1) / norm
   #
   s2 = ((ClkCmb + Nq30Cmb)*Cgg + (Ckg)**2) / Nmodes
   norm = np.sum(Ckg[I]**2 / s2[I])
   #
   sQ30CmbLsstgold[ilMax] = 1./np.sqrt(norm)
   #
   bQ30CmbTszLsstgold[ilMax] = np.sum(Clq30CmbTszLsstgold[:,I[0]] * Ckg[I] / s2[I], axis=1) / norm

   # 35
   s2 =  2. * (ClkCmb + Nq35Cmb)**2. / Nmodes 
   norm = np.sum(ClkCmb[I]**2. / s2[I])
   #
   snrQ35Cmb[ilMax] = np.sqrt(norm)
   #
   bQ35CmbTszFull[ilMax] = np.sum((2.*Clq35CmbTszKappa[:,I[0]]+2.*Clq35CmbTszSec[:,I[0]]+Clq35CmbTsz[:,I[0]]) * ClkCmb[I] / s2[I], axis=1) / norm
   #
   s2 = ((ClkCmb + Nq35Cmb)*Cgg + (Ckg)**2) / Nmodes
   norm = np.sum(Ckg[I]**2 / s2[I])
   #
   sQ35CmbLsstgold[ilMax] = 1./np.sqrt(norm)
   #
   bQ35CmbTszLsstgold[ilMax] = np.sum(Clq35CmbTszLsstgold[:,I[0]] * Ckg[I] / s2[I], axis=1) / norm
  

##################################################################################
##################################################################################
# Average over patches

# 10
sBQ10CmbTszFull = np.array([np.std(bQ10CmbTszFull[i]) * uncertaintyFactor / np.sqrt(len(bQ10CmbTszFull[i])) for i in range(len(bQ10CmbTszFull))])
bQ10CmbTszFull = np.array([np.mean(bQ10CmbTszFull[i]) for i in range(len(bQ10CmbTszFull))])
#
sBQ10CmbTszLsstgold = np.array([np.std(bQ10CmbTszLsstgold[i]) * uncertaintyFactor / np.sqrt(len(bQ10CmbTszLsstgold[i])) for i in range(len(bQ10CmbTszLsstgold))])
bQ10CmbTszLsstgold = np.array([np.mean(bQ10CmbTszLsstgold[i]) for i in range(len(bQ10CmbTszLsstgold))])

# 12
sBQ12CmbTszFull = np.array([np.std(bQ12CmbTszFull[i]) * uncertaintyFactor / np.sqrt(len(bQ12CmbTszFull[i])) for i in range(len(bQ12CmbTszFull))])
bQ12CmbTszFull = np.array([np.mean(bQ12CmbTszFull[i]) for i in range(len(bQ12CmbTszFull))])
#
sBQ12CmbTszLsstgold = np.array([np.std(bQ12CmbTszLsstgold[i]) * uncertaintyFactor / np.sqrt(len(bQ12CmbTszLsstgold[i])) for i in range(len(bQ12CmbTszLsstgold))])
bQ12CmbTszLsstgold = np.array([np.mean(bQ12CmbTszLsstgold[i]) for i in range(len(bQ12CmbTszLsstgold))])

# 15
sBQ15CmbTszFull = np.array([np.std(bQ15CmbTszFull[i]) * uncertaintyFactor / np.sqrt(len(bQ15CmbTszFull[i])) for i in range(len(bQ15CmbTszFull))])
bQ15CmbTszFull = np.array([np.mean(bQ15CmbTszFull[i]) for i in range(len(bQ15CmbTszFull))])
#
sBQ15CmbTszLsstgold = np.array([np.std(bQ15CmbTszLsstgold[i]) * uncertaintyFactor / np.sqrt(len(bQ15CmbTszLsstgold[i])) for i in range(len(bQ15CmbTszLsstgold))])
bQ15CmbTszLsstgold = np.array([np.mean(bQ15CmbTszLsstgold[i]) for i in range(len(bQ15CmbTszLsstgold))])

# 17
sBQ17CmbTszFull = np.array([np.std(bQ17CmbTszFull[i]) * uncertaintyFactor / np.sqrt(len(bQ17CmbTszFull[i])) for i in range(len(bQ17CmbTszFull))])
bQ17CmbTszFull = np.array([np.mean(bQ17CmbTszFull[i]) for i in range(len(bQ17CmbTszFull))])
#
sBQ17CmbTszLsstgold = np.array([np.std(bQ17CmbTszLsstgold[i]) * uncertaintyFactor / np.sqrt(len(bQ17CmbTszLsstgold[i])) for i in range(len(bQ17CmbTszLsstgold))])
bQ17CmbTszLsstgold = np.array([np.mean(bQ17CmbTszLsstgold[i]) for i in range(len(bQ17CmbTszLsstgold))])

# 20
sBQ20CmbTszFull = np.array([np.std(bQ20CmbTszFull[i]) * uncertaintyFactor / np.sqrt(len(bQ20CmbTszFull[i])) for i in range(len(bQ20CmbTszFull))])
bQ20CmbTszFull = np.array([np.mean(bQ20CmbTszFull[i]) for i in range(len(bQ20CmbTszFull))])
#
sBQ20CmbTszLsstgold = np.array([np.std(bQ20CmbTszLsstgold[i]) * uncertaintyFactor / np.sqrt(len(bQ20CmbTszLsstgold[i])) for i in range(len(bQ20CmbTszLsstgold))])
bQ20CmbTszLsstgold = np.array([np.mean(bQ20CmbTszLsstgold[i]) for i in range(len(bQ20CmbTszLsstgold))])

# 22
sBQ22CmbTszFull = np.array([np.std(bQ22CmbTszFull[i]) * uncertaintyFactor / np.sqrt(len(bQ22CmbTszFull[i])) for i in range(len(bQ22CmbTszFull))])
bQ22CmbTszFull = np.array([np.mean(bQ22CmbTszFull[i]) for i in range(len(bQ22CmbTszFull))])
#
sBQ22CmbTszLsstgold = np.array([np.std(bQ22CmbTszLsstgold[i]) * uncertaintyFactor / np.sqrt(len(bQ22CmbTszLsstgold[i])) for i in range(len(bQ22CmbTszLsstgold))])
bQ22CmbTszLsstgold = np.array([np.mean(bQ22CmbTszLsstgold[i]) for i in range(len(bQ22CmbTszLsstgold))])

# 25
sBQ25CmbTszFull = np.array([np.std(bQ25CmbTszFull[i]) * uncertaintyFactor / np.sqrt(len(bQ25CmbTszFull[i])) for i in range(len(bQ25CmbTszFull))])
bQ25CmbTszFull = np.array([np.mean(bQ25CmbTszFull[i]) for i in range(len(bQ25CmbTszFull))])
#
sBQ25CmbTszLsstgold = np.array([np.std(bQ25CmbTszLsstgold[i]) * uncertaintyFactor / np.sqrt(len(bQ25CmbTszLsstgold[i])) for i in range(len(bQ25CmbTszLsstgold))])
bQ25CmbTszLsstgold = np.array([np.mean(bQ25CmbTszLsstgold[i]) for i in range(len(bQ25CmbTszLsstgold))])

# 30
sBQ30CmbTszFull = np.array([np.std(bQ30CmbTszFull[i]) * uncertaintyFactor / np.sqrt(len(bQ30CmbTszFull[i])) for i in range(len(bQ30CmbTszFull))])
bQ30CmbTszFull = np.array([np.mean(bQ30CmbTszFull[i]) for i in range(len(bQ30CmbTszFull))])
#
sBQ30CmbTszLsstgold = np.array([np.std(bQ30CmbTszLsstgold[i]) * uncertaintyFactor / np.sqrt(len(bQ30CmbTszLsstgold[i])) for i in range(len(bQ30CmbTszLsstgold))])
bQ30CmbTszLsstgold = np.array([np.mean(bQ30CmbTszLsstgold[i]) for i in range(len(bQ30CmbTszLsstgold))])

# 35
sBQ35CmbTszFull = np.array([np.std(bQ35CmbTszFull[i]) * uncertaintyFactor / np.sqrt(len(bQ35CmbTszFull[i])) for i in range(len(bQ35CmbTszFull))])
bQ35CmbTszFull = np.array([np.mean(bQ35CmbTszFull[i]) for i in range(len(bQ35CmbTszFull))])
#
sBQ35CmbTszLsstgold = np.array([np.std(bQ35CmbTszLsstgold[i]) * uncertaintyFactor / np.sqrt(len(bQ35CmbTszLsstgold[i])) for i in range(len(bQ35CmbTszLsstgold))])
bQ35CmbTszLsstgold = np.array([np.mean(bQ35CmbTszLsstgold[i]) for i in range(len(bQ35CmbTszLsstgold))])



##################################################################################
##################################################################################
# Summary for each estimator: Cross with LSST gold

plt.axhline(0., c='k', lw=1, linestyle='--')
#plt.fill_between(llMax, -sQ15CmbLsstgold, sQ15CmbLsstgold, edgecolor='', facecolor='gray', alpha=0.6)
plt.plot(llMax,sQ10CmbLsstgold,lw=0.5,linestyle='--',c=c10)
plt.plot(llMax,-sQ10CmbLsstgold,lw=0.5,linestyle='--',c=c10)

plt.plot(llMax,sQ12CmbLsstgold,lw=0.5,linestyle='--',c=c12)
plt.plot(llMax,-sQ12CmbLsstgold,lw=0.5,linestyle='--',c=c12)

plt.plot(llMax,sQ15CmbLsstgold,lw=0.5,linestyle='--',c=c15)
plt.plot(llMax,-sQ15CmbLsstgold,lw=0.5,linestyle='--',c=c15)

plt.plot(llMax,sQ17CmbLsstgold,lw=0.5,linestyle='--',c=c17)
plt.plot(llMax,-sQ17CmbLsstgold,lw=0.5,linestyle='--',c=c17)

plt.plot(llMax,sQ20CmbLsstgold,lw=0.5,linestyle='--',c=c20)
plt.plot(llMax,-sQ20CmbLsstgold,lw=0.5,linestyle='--',c=c20)

plt.plot(llMax,sQ22CmbLsstgold,lw=0.5,linestyle='--',c=c22)
plt.plot(llMax,-sQ22CmbLsstgold,lw=0.5,linestyle='--',c=c22)

plt.plot(llMax,sQ25CmbLsstgold,lw=0.5,linestyle='--',c=c25)
plt.plot(llMax,-sQ25CmbLsstgold,lw=0.5,linestyle='--',c=c25)

plt.plot(llMax,sQ30CmbLsstgold,lw=0.5,linestyle='--',c=c30)
plt.plot(llMax,-sQ30CmbLsstgold,lw=0.5,linestyle='--',c=c30)

plt.plot(llMax,sQ35CmbLsstgold,lw=0.5,linestyle='--',c=c35)
plt.plot(llMax,-sQ35CmbLsstgold,lw=0.5,linestyle='--',c=c35)

# 10
plt.plot(llMax, bQ10CmbTszLsstgold, c10, label='1.0\'')
Up = bQ10CmbTszLsstgold + sBQ10CmbTszLsstgold
Down = bQ10CmbTszLsstgold - sBQ10CmbTszLsstgold
plt.fill_between(llMax, Down, Up, edgecolor='', facecolor=c10, alpha=0.4)
# 12
plt.plot(llMax, bQ12CmbTszLsstgold, c12, label='1.25\'')
Up = bQ12CmbTszLsstgold + sBQ12CmbTszLsstgold
Down = bQ12CmbTszLsstgold - sBQ12CmbTszLsstgold
plt.fill_between(llMax, Down, Up, edgecolor='', facecolor=c12, alpha=0.4)
# 15
plt.plot(llMax, bQ15CmbTszLsstgold, c15, label='1.5\'')
Up = bQ15CmbTszLsstgold + sBQ15CmbTszLsstgold
Down = bQ15CmbTszLsstgold - sBQ15CmbTszLsstgold
plt.fill_between(llMax, Down, Up, edgecolor='', facecolor=c15, alpha=0.4)
# 17
plt.plot(llMax, bQ17CmbTszLsstgold, c17, label='1.75\'')
Up = bQ17CmbTszLsstgold + sBQ17CmbTszLsstgold
Down = bQ17CmbTszLsstgold - sBQ17CmbTszLsstgold
plt.fill_between(llMax, Down, Up, edgecolor='', facecolor=c17, alpha=0.4)
# 20
plt.plot(llMax, bQ20CmbTszLsstgold, c20, label='2.0\'')
Up = bQ20CmbTszLsstgold + sBQ20CmbTszLsstgold
Down = bQ20CmbTszLsstgold - sBQ20CmbTszLsstgold
plt.fill_between(llMax, Down, Up, edgecolor='', facecolor=c20, alpha=0.4)
# 22
plt.plot(llMax, bQ22CmbTszLsstgold, c22, label='2.25\'')
Up = bQ22CmbTszLsstgold + sBQ22CmbTszLsstgold
Down = bQ22CmbTszLsstgold - sBQ22CmbTszLsstgold
plt.fill_between(llMax, Down, Up, edgecolor='', facecolor=c22, alpha=0.4)
# 25
plt.plot(llMax, bQ25CmbTszLsstgold, c25, label='2.5\'')
Up = bQ25CmbTszLsstgold + sBQ25CmbTszLsstgold
Down = bQ25CmbTszLsstgold - sBQ25CmbTszLsstgold
plt.fill_between(llMax, Down, Up, edgecolor='', facecolor=c25, alpha=0.4)
# 30
plt.plot(llMax, bQ30CmbTszLsstgold, c30, label='3.0\'')
Up = bQ30CmbTszLsstgold + sBQ30CmbTszLsstgold
Down = bQ30CmbTszLsstgold - sBQ30CmbTszLsstgold
plt.fill_between(llMax, Down, Up, edgecolor='', facecolor=c30, alpha=0.4)
# 35
plt.plot(llMax, bQ35CmbTszLsstgold, c35, label='3.5\'')
Up = bQ35CmbTszLsstgold + sBQ35CmbTszLsstgold
Down = bQ35CmbTszLsstgold - sBQ35CmbTszLsstgold
plt.fill_between(llMax, Down, Up, edgecolor='', facecolor=c35, alpha=0.4)
#
plt.ylim((-0.025,0.025))
#
#
plt.xlim((llMax[0], llMax[-1]))
plt.xlabel(r'$\ell_{\text{max, T}}$')
plt.title(r'Relative bias on $C^{\kappa\times\text{LSST}}_L$'+'\n amplitude from tSZ')
plt.legend(loc=2, ncol=5, mode=None, handlelength=.5, fontsize='x-small', borderpad=0.1, borderaxespad=0.1, frameon=False)

path = "output/summary_sigma_Lsstgold.pdf"
plt.savefig(path, bbox_inches='tight')
plt.clf()


##################################################################################
##################################################################################
# Summary for each estimator: bias on k_CMB amplitude

sQ10Cmb = 1./snrQ10Cmb
sQ12Cmb = 1./snrQ12Cmb
sQ15Cmb = 1./snrQ15Cmb
sQ17Cmb = 1./snrQ17Cmb
sQ20Cmb = 1./snrQ20Cmb
sQ22Cmb = 1./snrQ22Cmb
sQ25Cmb = 1./snrQ25Cmb
sQ30Cmb = 1./snrQ30Cmb
sQ35Cmb = 1./snrQ35Cmb

plt.axhline(0., c='k', lw=1, linestyle='--')
#plt.fill_between(llMax, -sQ15Cmb, sQ15Cmb, edgecolor='', facecolor='gray', alpha=0.6)
plt.plot(llMax,sQ10Cmb,lw=0.5,linestyle='--',c=c10)
plt.plot(llMax,-sQ10Cmb,lw=0.5,linestyle='--',c=c10)

plt.plot(llMax,sQ12Cmb,lw=0.5,linestyle='--',c=c12)
plt.plot(llMax,-sQ12Cmb,lw=0.5,linestyle='--',c=c12)

plt.plot(llMax,sQ15Cmb,lw=0.5,linestyle='--',c=c15)
plt.plot(llMax,-sQ15Cmb,lw=0.5,linestyle='--',c=c15)

plt.plot(llMax,sQ17Cmb,lw=0.5,linestyle='--',c=c17)
plt.plot(llMax,-sQ17Cmb,lw=0.5,linestyle='--',c=c17)

plt.plot(llMax,sQ20Cmb,lw=0.5,linestyle='--',c=c20)
plt.plot(llMax,-sQ20Cmb,lw=0.5,linestyle='--',c=c20)

plt.plot(llMax,sQ22Cmb,lw=0.5,linestyle='--',c=c22)
plt.plot(llMax,-sQ22Cmb,lw=0.5,linestyle='--',c=c22)

plt.plot(llMax,sQ25Cmb,lw=0.5,linestyle='--',c=c25)
plt.plot(llMax,-sQ25Cmb,lw=0.5,linestyle='--',c=c25)

plt.plot(llMax,sQ30Cmb,lw=0.5,linestyle='--',c=c30)
plt.plot(llMax,-sQ30Cmb,lw=0.5,linestyle='--',c=c30)

plt.plot(llMax,sQ35Cmb,lw=0.5,linestyle='--',c=c35)
plt.plot(llMax,-sQ35Cmb,lw=0.5,linestyle='--',c=c35)

# 10
plt.plot(llMax, bQ10CmbTszFull, c10, label='1.0\'')
Up = bQ10CmbTszFull + sBQ10CmbTszFull
Down = bQ10CmbTszFull - sBQ10CmbTszFull
plt.fill_between(llMax, Down, Up, edgecolor='', facecolor=c10, alpha=0.4)
# 12
plt.plot(llMax, bQ12CmbTszFull, c12, label='1.25\'')
Up = bQ12CmbTszFull + sBQ12CmbTszFull
Down = bQ12CmbTszFull - sBQ12CmbTszFull
plt.fill_between(llMax, Down, Up, edgecolor='', facecolor=c12, alpha=0.4)
# 15
plt.plot(llMax, bQ15CmbTszFull, c15, label='1.5\'')
Up = bQ15CmbTszFull + sBQ15CmbTszFull
Down = bQ15CmbTszFull - sBQ15CmbTszFull
plt.fill_between(llMax, Down, Up, edgecolor='', facecolor=c15, alpha=0.4)
# 17
plt.plot(llMax, bQ17CmbTszFull, c17, label='1.75\'')
Up = bQ17CmbTszFull + sBQ17CmbTszFull
Down = bQ17CmbTszFull - sBQ17CmbTszFull
plt.fill_between(llMax, Down, Up, edgecolor='', facecolor=c17, alpha=0.4)
# 20
plt.plot(llMax, bQ20CmbTszFull, c20, label='2.0\'')
Up = bQ20CmbTszFull + sBQ20CmbTszFull
Down = bQ20CmbTszFull - sBQ20CmbTszFull
plt.fill_between(llMax, Down, Up, edgecolor='', facecolor=c20, alpha=0.4)
# 22
plt.plot(llMax, bQ22CmbTszFull, c22, label='2.25\'')
Up = bQ22CmbTszFull + sBQ22CmbTszFull
Down = bQ22CmbTszFull - sBQ22CmbTszFull
plt.fill_between(llMax, Down, Up, edgecolor='', facecolor=c22, alpha=0.4)
# 25
plt.plot(llMax, bQ25CmbTszFull, c25, label='2.5\'')
Up = bQ25CmbTszFull + sBQ25CmbTszFull
Down = bQ25CmbTszFull - sBQ25CmbTszFull
plt.fill_between(llMax, Down, Up, edgecolor='', facecolor=c25, alpha=0.4)
# 30
plt.plot(llMax, bQ30CmbTszFull, c30, label='3.0\'')
Up = bQ30CmbTszFull + sBQ30CmbTszFull
Down = bQ30CmbTszFull - sBQ30CmbTszFull
plt.fill_between(llMax, Down, Up, edgecolor='', facecolor=c30, alpha=0.4)
# 35
plt.plot(llMax, bQ35CmbTszFull, c35, label='3.5\'')
Up = bQ35CmbTszFull + sBQ35CmbTszFull
Down = bQ35CmbTszFull - sBQ35CmbTszFull
plt.fill_between(llMax, Down, Up, edgecolor='', facecolor=c35, alpha=0.4)
#
plt.ylim((-0.025,0.025))
plt.xlim((llMax[0], llMax[-1]))
plt.xlabel(r'$\ell_{\text{max, T}}$')
plt.title(r'Relative bias on $C^{\kappa}_L$'+'\n amplitude from tSZ')
plt.legend(loc=2, ncol=5, mode=None, handlelength=.5, fontsize='x-small', borderpad=0.1, borderaxespad=0.1, frameon=False)

path = "output/summary_sigma_full.pdf"
plt.savefig(path, bbox_inches='tight')
plt.clf()
