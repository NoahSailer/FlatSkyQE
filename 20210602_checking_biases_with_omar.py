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
lMax4mask = 4.5e3

lMax = float(sys.argv[1])
patchMin,patchMax = int(sys.argv[2]),int(sys.argv[3])
t = float(sys.argv[4])

# ell bins for power spectra
nBins = 21  # number of bins
lRange = (1., 2.*lMax)  # range for power spectra

##################################################################################
print("ILC weights")

Nu = np.array([27.e9,39.e9,93.e9,145.e9,225.e9,280.e9]) # [Hz]
Beam = np.array([7.4,5.1,2.2,1.4,1.0,0.9])
Noise = np.array([52.,27.,5.8,6.3,15.,37.])
cmbIlc = CMBILC(Nu, Beam, Noise, atm=True,lMaxT =2*lMax,lMaxP = lMax)
#cmbIlc4mask = CMBILC(Nu,Beam,Noise,atm=True,lMaxT =2*4500,lMaxP=4500)

# create CMB object for masking at 150 GHz
cmb4mask = CMB(beam=1.4, noise=6.3, lMin=lMin/2, lMaxT=2*lMax4mask, lMaxP=lMax4mask, atm=True, fg=True)
L = np.logspace(np.log10(lMin/2),np.log10(2*lMax4mask),5001)
forCtot = np.array([cmb4mask.ftotalTT(l) for l in L])
cmb4mask.fCtotal = interp1d(L, forCtot, kind='linear', bounds_error=False, fill_value=np.inf)

# interpolate to increase speed
print('Interpolating individual TiTj Ctots')
for i in range(len(Nu)):
   for j in range(len(Nu)):
      L = np.logspace(np.log10(lMin/2),np.log10(2*lMax),5001)
      forCtot = np.array([cmbIlc.cmb[i,j].ftotalTT(l) for l in L])
      cmbIlc.cmb[i,j].ftotalTT = interp1d(L, forCtot, kind='linear', bounds_error=False, fill_value=np.inf)
      #
      #L = np.logspace(np.log10(lMin/2),np.log10(2*4500),5001)
      #forCtot = np.array([cmbIlc4mask.cmb[i,j].ftotalTT(l) for l in L])
      #cmbIlc4mask.cmb[i,j].ftotalTT = interp1d(L, forCtot, kind='linear', bounds_error=False, fill_value=np.inf)
print('Finished interpolating')

print('Interpolating the ILC weights')
def wILC(ell): return cmbIlc.weightsIlcCmb(ell) + t*(cmbIlc.weightsDeprojTszCIB(ell)-cmbIlc.weightsIlcCmb(ell))
#def wILC(ell): return cmbIlc.weightsIlcCmb(ell) + t*(cmbIlc.weightsDeprojCIB(ell)-cmbIlc.weightsIlcCmb(ell))
#def wILC(ell): return cmbIlc.weightsIlcCmb(ell) + t*(cmbIlc.weightsDeprojTsz(ell)-cmbIlc.weightsIlcCmb(ell))
tmp_ells = np.logspace(np.log10(lMin/2),np.log10(2*lMax),5001)
tmp = np.array([wILC(l) for l in tmp_ells]).T
wILC = np.array([interp1d(tmp_ells,tmp[i],bounds_error=False,fill_value=0) for i in range(len(Nu))])
print('Finished interpolating')

fTsz = np.array([cmbIlc.cmb[0,0].tszFreqDpdceT(Nu[i])/cmbIlc.cmb[0,0].tszFreqDpdceT(150.e9) for i in range(len(Nu))])
fKsz = np.array([cmbIlc.cmb[0,0].kszFreqDpdceT(Nu[i])/cmbIlc.cmb[0,0].kszFreqDpdceT(150.e9) for i in range(len(Nu))])
fCIB = np.array([cmbIlc.cmb[0,0].cibPoissonFreqDpdceT(Nu[i])/cmbIlc.cmb[0,0].cibPoissonFreqDpdceT(150.e9) for i in range(len(Nu))])
fRadioPS = np.array([cmbIlc.cmb[0,0].radioPoissonFreqDpdceT(Nu[i])/cmbIlc.cmb[0,0].radioPoissonFreqDpdceT(150.e9) for i in range(len(Nu))])

#fTsz_int = np.array([cmbIlc.cmb[0,0].tszFreqDpdceInt(Nu[i])/cmbIlc.cmb[0,0].tszFreqDpdceInt(148.e9) for i in range(len(Nu))])
#fKsz_int = np.array([cmbIlc.cmb[0,0].kszFreqDpdceInt(Nu[i])/cmbIlc.cmb[0,0].kszFreqDpdceInt(148.e9) for i in range(len(Nu))])
#fCIB_int = np.array([cmbIlc.cmb[0,0].cibPoissonFreqDpdceInt(Nu[i])/cmbIlc.cmb[0,0].cibPoissonFreqDpdceInt(148.e9) for i in range(len(Nu))])
#fRadioPS_int = np.array([cmbIlc.cmb[0,0].radioPoissonFreqDpdceInt(Nu[i])/cmbIlc.cmb[0,0].radioPoissonFreqDpdceInt(148.e9) for i in range(len(Nu))])

def tszFilter(l): return np.dot(fTsz,np.array([wILC[i](l) for i in range(len(Nu))]))
def kszFilter(l): return np.dot(fKsz,np.array([wILC[i](l) for i in range(len(Nu))]))
def cibFilter(l): return np.dot(fCIB,np.array([wILC[i](l) for i in range(len(Nu))]))
def radiopsFilter(l): return np.dot(fRadioPS,np.array([wILC[i](l) for i in range(len(Nu))]))


##################################################################################
# calculate the map noise
print('Interpolating the noise of the ILC map')
def ctot(l):
   result = 0
   for i in range(len(Nu)):
      for j in range(len(Nu)):
         result += cmbIlc.cmb[i,j].ftotalTT(l)*wILC[i](l)*wILC[j](l)
   return result
# interpolate to make things fast
print("CMB experiment properties")
cmb = CMB(beam=1.4, noise=6.3, lMin=lMin/2, lMaxT=2*lMax, lMaxP=lMax, atm=True, fg=True)
L = np.logspace(np.log10(lMin/2),np.log10(2*lMax),5001)
forCtot = np.array([ctot(l) for l in L])
cmb.fCtotal = interp1d(L, forCtot, kind='linear', bounds_error=False, fill_value=np.inf)
print('Finished interpolating')

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

# load tSZ profile (sqrt of power spectrum, measured from sims)
l = np.genfromtxt('l.txt')
profile = np.genfromtxt('profile.txt')
uTsz = interp1d(l,profile,kind='linear')


##################################################################################
##################################################################################


def computeCorrelations(fgFourier=None,fgGFourier=None,kCmbFourier=None,gFourier=None,cmb0Fourier=None,cmb1Fourier=None,tot=None):

   if tot == None: tot = cmb.fCtotal
   #################################################################################
   # Standard QE
   pQFourier = baseMap.computeQuadEstKappaNorm(cmb.flensedTT, tot, lMin=lMin, lMax=lMax, dataFourier=fgFourier, test=False)
   #
   pQGFourier = baseMap.computeQuadEstKappaNorm(cmb.flensedTT, tot, lMin=lMin, lMax=lMax, dataFourier=fgGFourier, test=False)
   #
   pQ0Fourier = baseMap.computeQuadEstKappaNorm(cmb.flensedTT, tot, lMin=lMin, lMax=lMax, dataFourier=fgFourier, dataFourier2=cmb0Fourier, test=False)
   pQ0Fourier += baseMap.computeQuadEstKappaNorm(cmb.flensedTT, tot, lMin=lMin, lMax=lMax, dataFourier=cmb0Fourier, dataFourier2=fgFourier, test=False)
   #
   pQ1Fourier = baseMap.computeQuadEstKappaNorm(cmb.flensedTT, tot, lMin=lMin, lMax=lMax, dataFourier=fgFourier, dataFourier2=cmb1Fourier, test=False)
   pQ1Fourier += baseMap.computeQuadEstKappaNorm(cmb.flensedTT, tot, lMin=lMin, lMax=lMax, dataFourier=cmb1Fourier, dataFourier2=fgFourier, test=False)

   pQFullFourier = baseMap.computeQuadEstKappaNorm(cmb.flensedTT, tot, lMin=lMin, lMax=lMax, dataFourier=cmb0Fourier+cmb1Fourier+fgFourier, test=False)

   #################################################################################
   # Shear E
   sCmbFourier = baseMap.computeQuadEstKappaShearNormCorr(cmb.flensedTT, tot, lMin=lMin, lMax=lMax, dataFourier=fgFourier, test=False)
   #
   sCmbGFourier = baseMap.computeQuadEstKappaShearNormCorr(cmb.flensedTT, tot, lMin=lMin, lMax=lMax, dataFourier=fgGFourier, test=False)
   #
   sCmb0Fourier = baseMap.computeQuadEstKappaShearNormCorr(cmb.flensedTT, tot, lMin=lMin, lMax=lMax, dataFourier=fgFourier, dataFourier2=cmb0Fourier, test=False)
   sCmb0Fourier += baseMap.computeQuadEstKappaShearNormCorr(cmb.flensedTT, tot, lMin=lMin, lMax=lMax, dataFourier=cmb0Fourier, dataFourier2=fgFourier, test=False)
   #
   sCmb1Fourier = baseMap.computeQuadEstKappaShearNormCorr(cmb.flensedTT, tot, lMin=lMin, lMax=lMax, dataFourier=fgFourier, dataFourier2=cmb1Fourier, test=False)
   sCmb1Fourier += baseMap.computeQuadEstKappaShearNormCorr(cmb.flensedTT, tot, lMin=lMin, lMax=lMax, dataFourier=cmb1Fourier, dataFourier2=fgFourier, test=False)

   #################################################################################
   # Bias hardened

   pQBHFourier = baseMap.computeQuadEstKappaPointSourceHardenedNorm(cmb.flensedTT, tot, lMin=lMin, lMax=lMax, dataFourier=fgFourier, test=False)
   #
   pQBHGFourier = baseMap.computeQuadEstKappaPointSourceHardenedNorm(cmb.flensedTT, tot, lMin=lMin, lMax=lMax, dataFourier=fgGFourier, test=False)
   #
   pQBH0Fourier = baseMap.computeQuadEstKappaPointSourceHardenedNorm(cmb.flensedTT, tot, lMin=lMin, lMax=lMax, dataFourier=fgFourier, dataFourier2=cmb0Fourier, test=False)
   pQBH0Fourier += baseMap.computeQuadEstKappaPointSourceHardenedNorm(cmb.flensedTT, tot, lMin=lMin, lMax=lMax, dataFourier=cmb0Fourier, dataFourier2=fgFourier, test=False)
   #
   pQBH1Fourier = baseMap.computeQuadEstKappaPointSourceHardenedNorm(cmb.flensedTT, tot, lMin=lMin, lMax=lMax, dataFourier=fgFourier, dataFourier2=cmb1Fourier, test=False)
   pQBH1Fourier += baseMap.computeQuadEstKappaPointSourceHardenedNorm(cmb.flensedTT, tot, lMin=lMin, lMax=lMax, dataFourier=cmb1Fourier, dataFourier2=fgFourier, test=False)


   #################################################################################
   # Bias hardened with tsz profile

   pQBHPFourier = baseMap.computeQuadEstKappaPointSourceHardenedNorm(cmb.flensedTT, tot, lMin=lMin, lMax=lMax, dataFourier=fgFourier, test=False, u=uTsz)
   #
   pQBHPGFourier = baseMap.computeQuadEstKappaPointSourceHardenedNorm(cmb.flensedTT, tot, lMin=lMin, lMax=lMax, dataFourier=fgGFourier, test=False, u=uTsz)
   #
   pQBHP0Fourier = baseMap.computeQuadEstKappaPointSourceHardenedNorm(cmb.flensedTT, tot, lMin=lMin, lMax=lMax, dataFourier=fgFourier, dataFourier2=cmb0Fourier, test=False, u=uTsz)
   pQBHP0Fourier += baseMap.computeQuadEstKappaPointSourceHardenedNorm(cmb.flensedTT, tot, lMin=lMin, lMax=lMax, dataFourier=cmb0Fourier, dataFourier2=fgFourier, test=False, u=uTsz)
   #
   pQBHP1Fourier = baseMap.computeQuadEstKappaPointSourceHardenedNorm(cmb.flensedTT, tot, lMin=lMin, lMax=lMax, dataFourier=fgFourier, dataFourier2=cmb1Fourier, test=False, u=uTsz)
   pQBHP1Fourier += baseMap.computeQuadEstKappaPointSourceHardenedNorm(cmb.flensedTT, tot, lMin=lMin, lMax=lMax, dataFourier=cmb1Fourier, dataFourier2=fgFourier, test=False, u=uTsz)

   ################################################################################
   print("Trispectrum: standard QE")
   lCen, ClQ, sCl = baseMap.powerSpectrum(pQFourier, plot=False, save=False, nBins=nBins)
   lCen, ClQG, sCl = baseMap.powerSpectrum(pQGFourier, plot=False, save=False, nBins=nBins)
   ClQ -= ClQG

   print("Trispectrum: shear E-mode")
   lCen, ClS, sCl = baseMap.powerSpectrum(sCmbFourier, plot=False, save=False, nBins=nBins)
   lCen, ClSG, sCl = baseMap.powerSpectrum(sCmbGFourier, plot=False, save=False, nBins=nBins)
   ClS -= ClSG

   print("Trispectrum: bias hardened")
   lCen, ClQBH, sCl = baseMap.powerSpectrum(pQBHFourier, plot=False, save=False, nBins=nBins)
   lCen, ClQBHG, sCl = baseMap.powerSpectrum(pQBHGFourier, plot=False, save=False, nBins=nBins)
   ClQBH -= ClQBHG

   print("Trispectrum: bias hardened with tsz profile")
   lCen, ClQBHP, sCl = baseMap.powerSpectrum(pQBHPFourier, plot=False, save=False, nBins=nBins)
   lCen, ClQBHPG, sCl = baseMap.powerSpectrum(pQBHPGFourier, plot=False, save=False, nBins=nBins)
   ClQBHP -= ClQBHPG

   ################################################################################
   print("Primary: standard QE x k_true")
   lCen, ClQx, sCl = baseMap.crossPowerSpectrum(pQFourier, kCmbFourier, plot=False, save=False, nBins=nBins)

   print("Primary: shear E-mode x k_true")
   lCen, ClSx, sCl = baseMap.crossPowerSpectrum(sCmbFourier, kCmbFourier, plot=False, save=False, nBins=nBins)

   print("Primary: bias hardened x k_true")
   lCen, ClQBHx, sCl = baseMap.crossPowerSpectrum(pQBHFourier, kCmbFourier, plot=False, save=False, nBins=nBins)

   print("Primary: bias hardened with tsz profile x k_true")
   lCen, ClQBHPx, sCl = baseMap.crossPowerSpectrum(pQBHPFourier, kCmbFourier, plot=False, save=False, nBins=nBins)

   ################################################################################
   print("Cross-power: standard QE x g")
   lCen, ClQx1, sCl = baseMap.crossPowerSpectrum(pQFourier, gFourier, plot=False, save=False, nBins=nBins)

   print("Cross-power: shear E-mode x g")
   lCen, ClSx1, sCl = baseMap.crossPowerSpectrum(sCmbFourier, gFourier, plot=False, save=False, nBins=nBins)

   print("Cross-power: bias hardened x g")
   lCen, ClQBHx1, sCl = baseMap.crossPowerSpectrum(pQBHFourier, gFourier, plot=False, save=False, nBins=nBins)

   print("Cross-power: bias hardened with tsz profile x g")
   lCen, ClQBHPx1, sCl = baseMap.crossPowerSpectrum(pQBHPFourier, gFourier, plot=False, save=False, nBins=nBins)

   #################################################################################
   print("Secondary: standard QE")
   lCen, ClQsec, sCl = baseMap.crossPowerSpectrum(pQ0Fourier, pQ1Fourier, plot=False, save=False, nBins=nBins)

   print("Secondary: shear E")
   lCen, ClSsec, sCl = baseMap.crossPowerSpectrum(sCmb0Fourier, sCmb1Fourier, plot=False, save=False, nBins=nBins)

   print("Secondary: bias hardened")
   lCen, ClQBHsec, sCl = baseMap.crossPowerSpectrum(pQBH0Fourier, pQBH1Fourier, plot=False, save=False, nBins=nBins)

   print("Secondary: bias hardened with tsz profile")
   lCen, ClQBHPsec, sCl = baseMap.crossPowerSpectrum(pQBHP0Fourier, pQBHP1Fourier, plot=False, save=False, nBins=nBins)

   #################################################################################
   data = np.array([ClQ,ClS,ClQBH,ClQBHP,ClQx,ClSx,ClQBHx,ClQBHPx,ClQx1,ClSx1,ClQBHx1,ClQBHPx1,ClQsec,ClSsec,ClQBHsec,ClQBHPsec])
   return lCen,data


##################################################################################
##################################################################################
# function to analyze one patch

maskRadiops = None

def analyzePatch(iPatch):

   patch = str(iPatch)
   print("Analyzing patch", patch)
   
   ##################################################################################
   # read foreground maps, and halo map

   # read Sehgal maps [Jy/sr]
   path = "flat_maps_large/sehgal_cib_148/sehgal_cib_148_large_cutout_"+str(iPatch)+".txt"
   cib148 = np.genfromtxt(path)
   path = "flat_maps_large/sehgal_tsz_148/sehgal_tsz_148_large_cutout_"+str(iPatch)+".txt"
   tsz = np.genfromtxt(path)
   path = "flat_maps_large/sehgal_ksz_148/sehgal_ksz_148_large_cutout_"+str(iPatch)+".txt"
   ksz = np.genfromtxt(path)
   path = "flat_maps_large/sehgal_radiops_148/sehgal_radiops_148_large_cutout_"+str(iPatch)+".txt"
   radiops = np.genfromtxt(path)
   path = "flat_maps_large/sehgal_kcmb/sehgal_kcmb_large_cutout_"+str(iPatch)+".txt"
   kCmb = np.genfromtxt(path)
   kCmbFourier = baseMap.fourier(kCmb)
   path = "flat_maps_large/sehgal_lsstgold/sehgal_lsstgold_large_cutout_"+str(iPatch)+".txt"
   lsstgold = np.genfromtxt(path)
   lsstgoldFourier = baseMap.fourier(lsstgold)

   # rescale to match Dunkley+13 power spectra
   # I rescale before masking, because otherwise the CIB mask masks too much
   # this numbers are chosen such that after a Dunkley mask (15mJy, 5' patches),
   # the power matches Dunkley
   cib148 *= 0.38 # 0.35 * np.sqrt(1.2)
   tsz *= 0.7  #0.82 # 0.68 * np.sqrt(1.45)
   ksz *= 0.82 # 0.8 * np.sqrt(1.05)
   radiops *= 1.1
    
   # sum of foregrounds
   tszCib = cib148 + tsz
   fTot = cib148 + tsz + ksz + radiops
    
   # Fourier transform
   cib148Fourier = baseMap.fourier(cib148)
   tszFourier = baseMap.fourier(tsz)
   kszFourier = baseMap.fourier(ksz)
   radiopsFourier = baseMap.fourier(radiops)
   tszCibFourier = baseMap.fourier(tszCib)
   fTotFourier = baseMap.fourier(fTot)

   fluxCut = 0.005  # in Jy 
   maskPatchRadius = 3. * np.pi/(180.*60.)   # in rad    
   maskTot = baseMap.pointSourceMaskMatchedFilterIsotropic(cmb4mask.fCtotal, fluxCut, fprof=None, dataFourier=fTotFourier, maskPatchRadius=maskPatchRadius, test=False)
                
   #def fcut(i):
      # gives the flux-cut for the i'th frequency channel
   #   conversion = 1.e6 * 1.e-26 / cmb.dBdT(Nu[i], cmb.Tcmb)
   #   return 5*cmbIlc4mask.cmb[i,i].fsigmaMatchedFilter() / conversion   
                     
   #tot4mask = lambda i: cib148Fourier*fCIB_int[i]+tszFourier*fTsz_int[i]+kszFourier*fKsz_int[i]+radiopsFourier*fRadioPS_int[i]
   #mask = lambda i: baseMap.pointSourceMaskMatchedFilterIsotropic(cmbIlc4mask.cmb[i,i].ftotalTT, fcut(i), fprof=None, dataFourier=tot4mask(i), maskPatchRadius=maskPatchRadius, test=False)
   #maskTot = mask(0)*mask(1)*mask(2)*mask(3)*mask(4)*mask(5)
   #maskTot = mask(3)
     
   # remove the mean
   cib148 -= np.mean(cib148.flatten())
   tsz -= np.mean(tsz.flatten())
   ksz -= np.mean(ksz.flatten())
   radiops -= np.mean(radiops.flatten())
   tszCib -= np.mean(tszCib.flatten())
   fTot -= np.mean(fTot.flatten())
    
   # convert from Jy/sr to muK
   # consistent with Sehgal's conversion: https://lambda.gsfc.nasa.gov/toolbox/tb_sim_info.cfm
   conversion = 1.e6 * 1.e-26 / cmb.dBdT(cmb.nu1, cmb.Tcmb) #
   cib148 *= conversion
   tsz *= conversion
   ksz *= conversion
   radiops *= conversion
   tszCib *= conversion
   fTot *= conversion

   # mask all the maps with the Dunkley mask, to match
   # mask each map with its own mask
   cib148 *= maskTot
   tsz *= maskTot
   ksz *= maskTot
   radiops *= maskTot
   fTot *= maskTot
   tszCib *= maskTot
    
   # Fourier transform
   cib148Fourier = baseMap.fourier(cib148)
   tszFourier = baseMap.fourier(tsz)
   kszFourier = baseMap.fourier(ksz)
   radiopsFourier = baseMap.fourier(radiops)
   tszCibFourier = baseMap.fourier(tszCib)
   fTotFourier = baseMap.fourier(fTot)

   # Foregrounds with ILC weights
   cib148FilteredFourier = baseMap.filterFourierIsotropic(cibFilter, dataFourier=cib148Fourier, test=False)
   tszFilteredFourier = baseMap.filterFourierIsotropic(tszFilter, dataFourier=tszFourier, test=False)
   kszFilteredFourier = baseMap.filterFourierIsotropic(kszFilter, dataFourier=kszFourier, test=False)
   radiopsFilteredFourier = baseMap.filterFourierIsotropic(radiopsFilter, dataFourier=radiopsFourier, test=False)
   tszCibFilteredFourier = cib148FilteredFourier + tszFilteredFourier
   fTotFilteredFourier = cib148FilteredFourier + tszFilteredFourier + kszFilteredFourier + radiopsFilteredFourier
   #
   cib148Filtered = baseMap.inverseFourier(cib148FilteredFourier)
   tszFiltered = baseMap.inverseFourier(tszFilteredFourier)
   kszFiltered = baseMap.inverseFourier(kszFilteredFourier)
   radiopsFiltered = baseMap.inverseFourier(radiopsFilteredFourier)
   tszCibFiltered = baseMap.inverseFourier(tszCibFilteredFourier)
   fTotFiltered = baseMap.inverseFourier(fTotFilteredFourier) 

   ##################################################################################
   # Generate lensed and unlensed CMB map

   cmb0Fourier = baseMap.genGRF(cmb.funlensedTT, test=False)
   cmb0 = baseMap.inverseFourier(cmb0Fourier)
   cmb1 = baseMap.doLensingTaylor(cmb0, kappaFourier=kCmbFourier, order=1)
   cmb1 -= cmb0
   cmb1Fourier = baseMap.fourier(cmb1)
   del cmb0, cmb1

   ##################################################################################
   # Generating mocks for source trispectra

   # randomize the phases
   f = lambda lx,ly: np.exp(1j*np.random.uniform(0., 2.*np.pi))
   #
   cib148GFourier = baseMap.filterFourier(f, dataFourier=cib148Fourier)
   tszGFourier = baseMap.filterFourier(f, dataFourier=tszFourier)
   kszGFourier = baseMap.filterFourier(f, dataFourier=kszFourier)
   radiopsGFourier = baseMap.filterFourier(f, dataFourier=radiopsFourier)
   tszCibGFourier = baseMap.filterFourier(f, dataFourier=tszCibFourier)
   fTotGFourier = baseMap.filterFourier(f,dataFourier=fTotFourier)
   #
   cib148FilteredGFourier = baseMap.filterFourier(f, dataFourier=cib148FilteredFourier)
   tszFilteredGFourier = baseMap.filterFourier(f, dataFourier=tszFilteredFourier)
   kszFilteredGFourier = baseMap.filterFourier(f, dataFourier=kszFilteredFourier)
   radiopsFilteredGFourier = baseMap.filterFourier(f, dataFourier=radiopsFilteredFourier)
   tszCibFilteredGFourier = baseMap.filterFourier(f, dataFourier=tszCibFilteredFourier)
   fTotFilteredGFourier = baseMap.filterFourier(f,dataFourier=fTotFilteredFourier)

   # convert to real space
   cib148G = baseMap.inverseFourier(cib148GFourier)
   tszG = baseMap.inverseFourier(tszGFourier)
   kszG = baseMap.inverseFourier(kszGFourier)
   radiopsG = baseMap.inverseFourier(radiopsGFourier)
   tszCibG = baseMap.inverseFourier(tszCibGFourier)
   fTotG = baseMap.inverseFourier(fTotGFourier)
   #
   cib148FilteredG = baseMap.inverseFourier(cib148FilteredGFourier)
   tszFilteredG = baseMap.inverseFourier(tszFilteredGFourier)
   kszFilteredG = baseMap.inverseFourier(kszFilteredGFourier)
   radiopsFilteredG = baseMap.inverseFourier(radiopsFilteredGFourier)
   tszCibFilteredG = baseMap.inverseFourier(tszCibFilteredGFourier)
   fTotFilteredG = baseMap.inverseFourier(fTotFilteredGFourier)

   # apply mask
   cib148G *= maskTot
   tszG *= maskTot
   kszG *= maskTot
   radiopsG *= maskTot
   tszCibG *= maskTot
   fTotG *= maskTot
   #
   cib148FilteredG *= maskTot
   tszFilteredG *= maskTot
   kszFilteredG *= maskTot
   radiopsFilteredG *= maskTot
   tszCibFilteredG *= maskTot
   fTotFilteredG *= maskTot

   # convert back to Fourier space
   cib148GFourier = baseMap.fourier(cib148G)
   tszGFourier = baseMap.fourier(tszG)
   kszGFourier = baseMap.fourier(kszG)
   radiopsGFourier = baseMap.fourier(radiopsG)
   tszCibGFourier = baseMap.fourier(tszCibG)
   fTotGFourier = baseMap.fourier(fTotG)
   # 
   cib148FilteredGFourier = baseMap.fourier(cib148FilteredG)
   tszFilteredGFourier = baseMap.fourier(tszFilteredG)
   kszFilteredGFourier = baseMap.fourier(kszFilteredG)
   radiopsFilteredGFourier = baseMap.fourier(radiopsFilteredG)
   tszCibFilteredGFourier = baseMap.fourier(tszCibFilteredG)
   fTotFilteredGFourier = baseMap.fourier(fTotFilteredG)
   del cib148G, tszG, kszG, radiopsG, fTotG
   del cib148FilteredG, tszFilteredG, kszFilteredG, radiopsFilteredG, fTotFilteredG

   ##################################################################################
   # Compute the biases for all foregrounds, at 148 GHz

   lCen,dataRadioPS = computeCorrelations(fgFourier=radiopsFourier,fgGFourier=radiopsGFourier,kCmbFourier=kCmbFourier,gFourier=lsstgoldFourier,cmb0Fourier=cmb0Fourier,cmb1Fourier=cmb1Fourier)
   print('########################################## radio PS',iPatch)
   #
   lCen,dataCIB = computeCorrelations(fgFourier=cib148Fourier,fgGFourier=cib148GFourier,kCmbFourier=kCmbFourier,gFourier=lsstgoldFourier,cmb0Fourier=cmb0Fourier,cmb1Fourier=cmb1Fourier)
   print('########################################## CIB',iPatch)
   #
   lCen,datatSZ = computeCorrelations(fgFourier=tszFourier,fgGFourier=tszGFourier,kCmbFourier=kCmbFourier,gFourier=lsstgoldFourier,cmb0Fourier=cmb0Fourier,cmb1Fourier=cmb1Fourier)
   print('########################################## tSZ',iPatch)
   #
   lCen,datakSZ = computeCorrelations(fgFourier=kszFourier,fgGFourier=kszGFourier,kCmbFourier=kCmbFourier,gFourier=lsstgoldFourier,cmb0Fourier=cmb0Fourier,cmb1Fourier=cmb1Fourier)
   print('########################################## kSZ',iPatch)
   #
   lCen,dataTot = computeCorrelations(fgFourier=fTotFourier,fgGFourier=fTotGFourier,kCmbFourier=kCmbFourier,gFourier=lsstgoldFourier,cmb0Fourier=cmb0Fourier,cmb1Fourier=cmb1Fourier)
   print('########################################## all',iPatch)

   data = np.concatenate((dataRadioPS,
                          dataCIB,
                          datatSZ,
                          datakSZ,
                          dataTot))

   outputFolder = "20210602"
   path = outputFolder+"/data_lmaxT_"+str(int(lMax))+"_"+str(iPatch)+".txt"
   np.savetxt(path,data)    
   np.savetxt(outputFolder+"/lCen_lmaxT_"+str(int(lMax))+".txt",lCen)


nProc = patchMax - patchMin
tStart = time()
pool = Pool(nProc)
pool.map(analyzePatch, range(patchMin,patchMax))
tStop = time()
print("\n\nFinished the parallel map calculations!")
print("Took "+str((tStop-tStart)/60.)+" min")
print("It worked!")
