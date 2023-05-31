from __future__ import print_function
from __future__ import division
from builtins import map
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

import pandas as pd

import cmb_ilc
reload(cmb_ilc)
from cmb_ilc import *


##################################################################################

# Specifications
Nu = np.array([27.e9,39.e9,93.e9,145.e9,225.e9,280.e9]) # [Hz]
Beam = np.array([7.4,5.1,2.2,1.4,1.0,0.9])
Noise = np.array([71., 36., 8., 10., 22., 54.])  # [muK*arcmin]



##################################################################################
# init

cmbIlc = CMBILC(Nu, Beam, Noise)


##################################################################################
##################################################################################
lMin = 10.
lMax = 3500.
LMin = 20.
LMax = 1.e3

nBands = 6
nBins = 21

data = pd.read_csv('Ckk_biases_lmaxT_3500.csv')

lCen = np.array(data['lCen'])
I = np.where((lCen>=LMin)*(lCen<=LMax))
lEdges = np.logspace(np.log10(1.), np.log10(np.max(15273.50)), nBins, 10.) # Edges of the bins
Nmodes = np.array(data['Nmodes'])

Nu = np.array([27.e9,39.e9,93.e9,145.e9,225.e9,280.e9]) # [Hz]
Beam = np.array([7.4,5.1,2.2,1.4,1.0,0.9])
Noise = np.array([71., 36., 8., 10., 22., 54.])  # [muK*arcmin]

ILC_weights_lmaxT_3500 = cmbIlc.weightsIlcCmb(3000.)

##################################################################################
##################################################################################
print("CMB experiment properties")

cmb = []

for i in range(nBands):
   cmb.append(CMB(beam=Beam[i], noise=Noise[i], nu1=Nu[i], nu2=Nu[i], lMin=lMin, lMaxT=lMax, lMaxP=lMax, atm=False, fg=True))
   forCtotal = lambda l: cmb[i].ftotalTT(l)
   # reinterpolate: gain factor 10 in speed
   L = np.logspace(np.log10(lMin/2.), np.log10(2.*lMax), 1001, 10.)
   F = np.array(list(map(forCtotal, L)))
   cmb[i].fCtotal = interp1d(L, F, kind='linear', bounds_error=False, fill_value=0.)

cmb = np.array(cmb)


##################################################################################
##################################################################################
print("Map properties")

nX = 400
nY = 400
size = 10.  # degrees, determined by the Sehgal cutouts
baseMap = FlatMap(nX=nX, nY=nY, sizeX=size*np.pi/180., sizeY=size*np.pi/180.)


##################################################################################
##################################################################################
print("CMB lensing power spectrum")

u = UnivPlanck15()
halofit = Halofit(u, save=False)
w_cmblens = WeightLensSingle(u, z_source=1100., name="cmblens")
w_lsstgold = WeightTracerLSSTGold(u)
p2d_cmblens = P2dAuto(u, halofit, w_cmblens, save=False)
ClkCmb = p2d_cmblens.fPinterp(lCen)


L = np.logspace(np.log10(lMin/2.), np.log10(2.*lMax), 1001, 10.)
weightsIlcCmb = np.array([cmbIlc.weightsIlcCmb(l) for l in L])



# plot ILC weights
for i in range(6): plt.semilogx(L,weightsIlcCmb[:,i],label=str(Nu[i]/1.e9),lw=2)
plt.legend(loc='upper right',ncol=3,title='Frequency [GHz]',fontsize='x-small')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$w^\text{ILC}_\ell$')
plt.savefig('output/ILC_weights.pdf',bbox_inches='tight')
plt.xlim(40,4000)
plt.clf()

# plot ILC deproj weights
ILC_deproj = np.array([cmbIlc.weightsDeprojTszCIB(l)-cmbIlc.weightsIlcCmb(l) for l in L])    
for i in range(6): plt.semilogx(L,ILC_deproj[:,i],label=str(Nu[i]/1.e9),lw=2)
plt.legend(loc='upper right',ncol=3,title='Frequency [GHz]',fontsize='x-small')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$w^\text{deproj}_\ell-w^\text{ILC}_\ell$')
plt.savefig('output/ILC_deproj_weights.pdf',bbox_inches='tight')
plt.xlim(40,4000)
plt.clf()

import sys
sys.exit()

##################################################################################
##################################################################################

def ctot(epsilon,l,lindex):
   # here weights really mean epsilon
   tSZ = cmb[0].ftSZ(l)*0.
   kSZ = cmb[0].ftSZ(l)*0.
   radioPS = cmb[0].ftSZ(l)*0.
   CIB = cmb[0].ftSZ(l)*0.
   detectornoise = cmb[0].ftSZ(l)*0.
   tszCIB = cmb[0].ftSZ(l)*0.

   for i in range(nBands):
      detectornoise += cmb[i].fdetectorNoise(l)*(weightsIlcCmb[lindex][i]+epsilon[i])**2.

      for j in range(nBands):
         tSZ += np.sqrt(cmb[i].ftSZ(l)*cmb[j].ftSZ(l))*(weightsIlcCmb[lindex][i]+epsilon[i])*(weightsIlcCmb[lindex][j]+epsilon[j])

         kSZ += np.sqrt(cmb[i].fkSZ(l)*cmb[j].fkSZ(l))*(weightsIlcCmb[lindex][i]+epsilon[i])*(weightsIlcCmb[lindex][j]+epsilon[j])

         radioPS += np.sqrt(cmb[i].fradioPoisson(l)*cmb[j].fradioPoisson(l))*(weightsIlcCmb[lindex][i]+epsilon[i])*(weightsIlcCmb[lindex][j]+epsilon[j])

         CIB += np.sqrt(cmb[i].fCIB(l)*cmb[j].fCIB(l))*(weightsIlcCmb[lindex][i]+epsilon[i])*(weightsIlcCmb[lindex][j]+epsilon[j])

         #include the cross later if you have time

   ctot = tSZ+kSZ+radioPS+CIB+detectornoise+cmb[0].flensedTT(l)
   return ctot


def NqCmb(epsilon): 
   #fCtotal =  lambda l: ctot(weights,l)
   fCtotal = np.array([ctot(epsilon,l,lindex) for lindex,l in enumerate(L)])
   interp_ctot = interp1d(L, fCtotal, kind='linear', bounds_error=False, fill_value=0.)
   return baseMap.forecastN0Kappa(cmb[0].funlensedTT, interp_ctot, lMin=lMin, lMax=lMax, test=False)(lCen)

def NqCmb_constant_weights(epsilon): 
   #fCtotal =  lambda l: ctot(weights,l)
   fCtotal = np.array([ctot(epsilon,l,np.where(L>3000.)[0][0]) for lindex,l in enumerate(L)])
   interp_ctot = interp1d(L, fCtotal, kind='linear', bounds_error=False, fill_value=0.)
   return baseMap.forecastN0Kappa(cmb[0].funlensedTT, interp_ctot, lMin=lMin, lMax=lMax, test=False)(lCen)
   
def noise_Alens(epsilon):
   s2 =  2. * (ClkCmb + NqCmb(epsilon))**2. / Nmodes 
   norm = np.sum(ClkCmb[I]**2. / s2[I])
   return 1./norm

def bias_Alens(weights):
   s2 =  2. * (ClkCmb + NqCmb(weights-ILC_weights_lmaxT_3500))**2. / Nmodes 
   norm = np.sum(ClkCmb[I]**2. / s2[I])
   #
   Cib_scalefactor = sum([cmb[0].cibClusteredFreqDpdceT(Nu[i])*weights[i]/cmb[0].cibClusteredFreqDpdceT(148.e9) for i in range(nBands)])
   Tsz_scalefactor = sum([cmb[0].tszFreqDpdceT(Nu[i])*weights[i]/cmb[0].tszFreqDpdceT(148.e9) for i in range(nBands)])
   Ksz_scalefactor = sum([cmb[0].kszFreqDpdceT(Nu[i])*weights[i]/cmb[0].kszFreqDpdceT(148.e9) for i in range(nBands)])
   Radiops_scalefactor = sum([cmb[0].radioPoissonFreqDpdceT(Nu[i])*weights[i]/cmb[0].radioPoissonFreqDpdceT(148.e9) for i in range(nBands)])
   #
   Cib = np.sum((2.*np.array(data['QE primary, CIB'])[I]*Cib_scalefactor**2.+2.*np.array(data['QE secondary, CIB'])[I]*Cib_scalefactor**2.+np.array(data['QE trispectrum, CIB'])[I]*Cib_scalefactor**4. ) * ClkCmb[I] / s2[I]) / norm
   #
   Tsz = np.sum((2.*np.array(data['QE primary, tSZ'])[I]*Tsz_scalefactor**2.+2.*np.array(data['QE secondary, tSZ'])[I]*Tsz_scalefactor**2.+np.array(data['QE trispectrum, tSZ'])[I]*Tsz_scalefactor**4. ) * ClkCmb[I] / s2[I]) / norm
   #
   Ksz = np.sum((2.*np.array(data['QE primary, kSZ'])[I]*Ksz_scalefactor**2.+2.*np.array(data['QE secondary, kSZ'])[I]*Ksz_scalefactor**2.+np.array(data['QE trispectrum, kSZ'])[I]*Ksz_scalefactor**4. ) * ClkCmb[I] / s2[I]) / norm
   #
   Radiops = np.sum((2.*np.array(data['QE primary, Radio PS'])[I]*Radiops_scalefactor**2.+2.*np.array(data['QE secondary, Radio PS'])[I]*Radiops_scalefactor**2.+np.array(data['QE trispectrum, Radio PS'])[I]*Radiops_scalefactor**4. ) * ClkCmb[I] / s2[I]) / norm

   return Cib**2. + Tsz**2. + Ksz**2. + Radiops**2.

def epsilon(x): return np.array([0.,0.,x,-x,0.,0.])

colors = ['red','orange','yellow','green','blue','indigo']

for i,x in enumerate(np.linspace(0.,3.,len(colors))):
   plt.loglog(lCen,NqCmb(epsilon(x)),label=str(round(x,1)),lw=2,c=colors[i])
   plt.loglog(lCen,NqCmb_constant_weights(epsilon(x)),lw=2,ls='--',c=colors[i])

plt.legend(loc='upper left',ncol=2,fontsize='x-small',title=r'$x,\epsilon_\nu = (0,0,x,-x,0,0)$')
plt.xlabel(r'$L$')
plt.ylabel(r'$N^\kappa_L[w^\text{ILC}+\epsilon]$')
plt.xlim(10,7000)
plt.title('Solid (dashed) curves are calculated with\nscale dependent (independent) ILC weights')
plt.savefig('output/noise.pdf',bbox_inches='tight')
plt.clf()
   

