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


L = np.logspace(np.log10(lMin/2.), np.log10(2.*lMax), 101, 10.)
weightsIlcCmb = np.array([cmbIlc.weightsIlcCmb(l) for l in L])

lindex3000 = np.where(L>=3000.)[0][0]
##################################################################################
##################################################################################

def ctot(weights,l,lindex):
   # here weights really mean epsilon
   tSZ = cmb[0].ftSZ(l)*0.
   kSZ = cmb[0].ftSZ(l)*0.
   radioPS = cmb[0].ftSZ(l)*0.
   CIB = cmb[0].ftSZ(l)*0.
   detectornoise = cmb[0].ftSZ(l)*0.
   tszCIB = cmb[0].ftSZ(l)*0.

   for i in range(nBands):
      detectornoise += cmb[i].fdetectorNoise(l)*(weightsIlcCmb[lindex][i]+weights[i])**2.

      for j in range(nBands):
         tSZ += np.sqrt(cmb[i].ftSZ(l)*cmb[j].ftSZ(l))*(weightsIlcCmb[lindex][i]+weights[i])*(weightsIlcCmb[lindex][j]+weights[j])

         kSZ += np.sqrt(cmb[i].fkSZ(l)*cmb[j].fkSZ(l))*(weightsIlcCmb[lindex][i]+weights[i])*(weightsIlcCmb[lindex][j]+weights[j])

         radioPS += np.sqrt(cmb[i].fradioPoisson(l)*cmb[j].fradioPoisson(l))*(weightsIlcCmb[lindex][i]+weights[i])*(weightsIlcCmb[lindex][j]+weights[j])

         CIB += np.sqrt(cmb[i].fCIB(l)*cmb[j].fCIB(l))*(weightsIlcCmb[lindex][i]+weights[i])*(weightsIlcCmb[lindex][j]+weights[j])

         #include the cross later if you have time

   ctot = tSZ+kSZ+radioPS+CIB+detectornoise+cmb[0].flensedTT(l)
   return ctot


def tSZ_power(weights,l,lindex):
   # here weights really mean epsilon
   tSZ = cmb[0].ftSZ(l)*0.

   for i in range(nBands):
      for j in range(nBands):
         tSZ += np.sqrt(cmb[i].ftSZ(l)*cmb[j].ftSZ(l))*(weightsIlcCmb[lindex][i]+weights[i])*(weightsIlcCmb[lindex][j]+weights[j])

         #include the cross later if you have time

   ctot = tSZ
   return ctot

def CtSZ(epsilon): return np.array([tSZ_power(epsilon,l,lindex) for lindex,l in enumerate(L)])

def CIB_power(weights,l,lindex):
   # here weights really mean epsilon
   CIB = cmb[0].fCIB(l)*0.

   for i in range(nBands):
      for j in range(nBands):
         CIB += np.sqrt(cmb[i].fCIB(l)*cmb[j].fCIB(l))*(weightsIlcCmb[lindex][i]+weights[i])*(weightsIlcCmb[lindex][j]+weights[j])
   return CIB

def CCIB(epsilon): return np.array([CIB_power(epsilon,l,lindex) for lindex,l in enumerate(L)])


def NqCmb(epsilon): 
   #fCtotal =  lambda l: ctot(weights,l)
   fCtotal = np.array([ctot(epsilon,l,lindex) for lindex,l in enumerate(L)])
   interp_ctot = interp1d(L, fCtotal, kind='linear', bounds_error=False, fill_value=0.)
   return baseMap.forecastN0Kappa(cmb[0].funlensedTT, interp_ctot, lMin=lMin, lMax=lMax, test=False)(lCen)
   
def noise_Alens(weights):
   s2 =  2. * (ClkCmb + NqCmb(weights))**2. / Nmodes 
   norm = np.sum(ClkCmb[I]**2. / s2[I])
   return 1./norm


##################################################################################
##################################################################################
fb = np.logspace(-4,1,300)
#epsilons = np.genfromtxt('modified_ILC.txt')
#epsilons = np.genfromtxt('modified_ILC_sum_squared.txt')[:,:-1]
eps = np.genfromtxt('modified_ILC_sum_squared.txt')
epsilons = np.zeros((len(eps),nBands))
for i in range(len(eps)):
   for j in range(nBands):
      if j == 0: epsilons[i,j]=0.
      elif j < nBands - 1: epsilons[i,j] = eps[i,j-1]
      elif j == nBands - 1: epsilons[i,j] = -sum(eps[i,:])

epsilonT = np.transpose(epsilons)

ftSZ = np.array([cmbIlc.cmb[0,0].tszFreqDpdceT(cmbIlc.Nu[i]) for i in range(cmbIlc.nNu)])
fCIB = np.array([cmbIlc.cmb[0,0].cibPoissonFreqDpdceT(cmbIlc.Nu[i]) for i in range(cmbIlc.nNu)])

#print(np.dot(epsilons[-1]+ILC_weights_lmaxT_3500,ftSZ)/ftSZ[0])
#print(np.dot(epsilons[-1]+ILC_weights_lmaxT_3500,fCIB)/fCIB[0])

#print(np.dot(cmbIlc.weightsDeprojTszCIB(3000),ftSZ)/ftSZ[0])
#print(np.dot(cmbIlc.weightsDeprojTszCIB(3000),fCIB)/fCIB[0])


##################################################################################
##################################################################################
plt.semilogx(fb,epsilonT[0],label='27',color='k')
plt.semilogx(fb,epsilonT[1],label='39',color='b')
plt.semilogx(fb,epsilonT[2],label='93',color='g')
plt.semilogx(fb,epsilonT[3],label='145',color='r')
plt.semilogx(fb,epsilonT[4],label='225',color='orange')
plt.semilogx(fb,epsilonT[5],label='280',color='pink')

ILC_deproj = cmbIlc.weightsDeprojTszCIB(3000) - cmbIlc.weightsIlcCmb(3000)
plt.axhline(ILC_deproj[0],color='k',ls='--',lw=0.7)
plt.axhline(ILC_deproj[1],color='b',ls='--',lw=0.7)
plt.axhline(ILC_deproj[2],color='g',ls='--',lw=0.7)
plt.axhline(ILC_deproj[3],color='r',ls='--',lw=0.7)
plt.axhline(ILC_deproj[4],color='orange',ls='--',lw=0.7)
plt.axhline(ILC_deproj[5],color='pink',ls='--',lw=0.7)


#plt.xlim(fb[0],fb[-2])
plt.legend(loc='lower left',fontsize='x-small',title=r'$\nu$ [GHz]',ncol=3)
plt.ylabel(r'$\epsilon_\nu$')
plt.xlabel(r'$f_b$')
plt.savefig('output/modified_ILC_weights.pdf',bbox_inches='tight')
plt.clf()

import sys
sys.exit()


plt.semilogx(L,CtSZ(np.zeros(6)))
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C^\text{tSZ}_\ell$')
plt.savefig('output/ctsz_ILC.pdf',bbox_inches='tight')
plt.clf()


ilc = cmbIlc.weightsIlcCmb(3000)
plt.semilogy([],[],label='27')
plt.semilogy(fb,np.abs(epsilonT[1]/ilc[1]),label='39')
plt.semilogy(fb,np.abs(epsilonT[2]/ilc[2]),label='93')
plt.semilogy(fb,np.abs(epsilonT[3]/ilc[3]),label='145')
plt.semilogy(fb,np.abs(epsilonT[4]/ilc[4]),label='225')
plt.semilogy(fb,np.abs(epsilonT[5]/ilc[5]),label='280')
plt.xlim(fb[0],fb[-2])
plt.legend(loc='lower right',fontsize='x-small',title=r'$\nu$ [GHz]',ncol=3)
plt.ylabel(r'$|\epsilon_\nu/w^\text{ILC}_\nu|$')
plt.xlabel(r'$f_b$')
plt.xlim(0.1,10.)
plt.savefig('output/modified_ILC_weights_fractional.pdf',bbox_inches='tight')
plt.clf()


plt.semilogy(fb,[CtSZ(epsilons[i])[lindex3000] for i in range(len(fb))],label='tSZ')
plt.semilogy(fb,[CCIB(epsilons[i])[lindex3000] for i in range(len(fb))],label='CIB')
plt.xlabel(r'$f_b$')
plt.ylabel(r'$C_{L=3000}$')
plt.legend(loc=0,fontsize='x-small',frameon=False)
plt.savefig('output/ctsz_vs_fb.pdf',bbox_inches='tight')
plt.clf()


##################################################################################
##################################################################################

def plot_tSZ_bias(weights):
   Tsz_scalefactor = sum([cmb[0].tszFreqDpdceT(Nu[i])*weights[i]/cmb[0].tszFreqDpdceT(148.e9) for i in range(nBands)])
   Tsz = 2.*np.array(data['QE primary, tSZ'])[I]*Tsz_scalefactor**2.+2.*np.array(data['QE secondary, tSZ'])[I]*Tsz_scalefactor**2.+np.array(data['QE trispectrum, tSZ'])[I]*Tsz_scalefactor**4.
   plt.plot(lCen[I],Tsz)
   plt.xlabel(r'$L$')
   plt.ylabel(r'$\text{bias}(C^\kappa_L)_\text{tSZ}$')
   plt.savefig('output/tSZ_bias.pdf',bbox_inches='tight')
   plt.clf()
   
plot_tSZ_bias(ILC_weights_lmaxT_3500)


def calculate_biases(weights):
   Cib_scalefactor = sum([cmb[0].cibClusteredFreqDpdceT(Nu[i])*weights[i]/cmb[0].cibClusteredFreqDpdceT(148.e9) for i in range(nBands)])
   Tsz_scalefactor = sum([cmb[0].tszFreqDpdceT(Nu[i])*weights[i]/cmb[0].tszFreqDpdceT(148.e9) for i in range(nBands)])
   Ksz_scalefactor = sum([cmb[0].kszFreqDpdceT(Nu[i])*weights[i]/cmb[0].kszFreqDpdceT(148.e9) for i in range(nBands)])
   Radiops_scalefactor = sum([cmb[0].radioPoissonFreqDpdceT(Nu[i])*weights[i]/cmb[0].radioPoissonFreqDpdceT(148.e9) for i in range(nBands)])
   s2 =  2. * (ClkCmb + NqCmb(weights-ILC_weights_lmaxT_3500))**2. / Nmodes 
   norm = np.sum(ClkCmb[I]**2. / s2[I])
   Cib = np.sum((2.*np.array(data['QE primary, CIB'])[I]*Cib_scalefactor**2.+2.*np.array(data['QE secondary, CIB'])[I]*Cib_scalefactor**2.+np.array(data['QE trispectrum, CIB'])[I]*Cib_scalefactor**4. ) * ClkCmb[I] / s2[I]) / norm
   Tsz = np.sum((2.*np.array(data['QE primary, tSZ'])[I]*Tsz_scalefactor**2.+2.*np.array(data['QE secondary, tSZ'])[I]*Tsz_scalefactor**2.+np.array(data['QE trispectrum, tSZ'])[I]*Tsz_scalefactor**4. ) * ClkCmb[I] / s2[I]) / norm
   Ksz = np.sum((2.*np.array(data['QE primary, kSZ'])[I]*Ksz_scalefactor**2.+2.*np.array(data['QE secondary, kSZ'])[I]*Ksz_scalefactor**2.+np.array(data['QE trispectrum, kSZ'])[I]*Ksz_scalefactor**4. ) * ClkCmb[I] / s2[I]) / norm
   Radiops = np.sum((2.*np.array(data['QE primary, Radio PS'])[I]*Radiops_scalefactor**2.+2.*np.array(data['QE secondary, Radio PS'])[I]*Radiops_scalefactor**2.+np.array(data['QE trispectrum, Radio PS'])[I]*Radiops_scalefactor**4. ) * ClkCmb[I] / s2[I]) / norm
   return [Cib,Tsz,Ksz,Radiops,np.sqrt(1./norm)]


Cib_bias = []
Tsz_bias = []
Ksz_bias = []
Radiops_bias = []
noise = []

for epsilon in epsilons:
   solutions = calculate_biases(ILC_weights_lmaxT_3500+epsilon)
   Cib_bias.append(solutions[0])
   Tsz_bias.append(solutions[1])
   Ksz_bias.append(solutions[2])
   Radiops_bias.append(solutions[3])
   noise.append(solutions[4])

print('done!')

plt.semilogx(fb,Cib_bias,c='b',label='CIB')
plt.semilogx(fb,Tsz_bias,c='g',label='tSZ')
plt.semilogx(fb,Ksz_bias,c='r',label='kSZ')
plt.semilogx(fb,Radiops_bias,c='orange',label='Radio PS')
plt.ylabel(r'bias($A_{\text{lens}}$)')
plt.xlabel(r'$f_b$')
plt.legend(loc='lower right',fontsize='x-small')
plt.savefig('output/bias_vs_fb.pdf',bbox_inches='tight')
plt.clf()

plt.semilogx(fb,noise,c='k')
plt.ylabel(r'$\sigma_{A_{\text{lens}}}(f_b)$')
plt.xlabel(r'$f_b$')
plt.savefig('output/noise_vs_fb.pdf',bbox_inches='tight')
plt.clf()
