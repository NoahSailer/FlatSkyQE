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

import scipy


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
lMax = 3000.

LMin = 20.
LMax = 1.e3

nBands = 6
nBins = 21

data = pd.read_csv('Ckk_biases_lmaxT_'+str(int(lMax))+'.csv')

lCen = np.array(data['lCen'])
I = np.where((lCen>=LMin)*(lCen<=LMax))
lEdges = np.logspace(np.log10(1.), np.log10(np.max(15273.50)), nBins, 10.) # Edges of the bins
Nmodes = np.array(data['Nmodes'])

Nu = np.array([27.e9,39.e9,93.e9,145.e9,225.e9,280.e9]) # [Hz]
Beam = np.array([7.4,5.1,2.2,1.4,1.0,0.9])
Noise = np.array([71., 36., 8., 10., 22., 54.])  # [muK*arcmin]

ILC_weights_lmaxT_3000 = cmbIlc.weightsIlcCmb(3000.)

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


L = np.logspace(np.log10(lMin/2.), np.log10(2.*lMax), 501, 10.)
weightsIlcCmb = np.array([cmbIlc.weightsIlcCmb(l) for l in L])

weightsIlcCmb_jointDeproj = np.array([cmbIlc.weightsDeprojTszCIB(l) for l in L])

weightsIlcCmb_tszdeproj = np.array([cmbIlc.weightsDeprojTsz(l) for l in L])


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
   
   
def ctot_deproj(l,lindex):
   # here weights really mean epsilon
   tSZ = cmb[0].ftSZ(l)*0.
   kSZ = cmb[0].ftSZ(l)*0.
   radioPS = cmb[0].ftSZ(l)*0.
   CIB = cmb[0].ftSZ(l)*0.
   detectornoise = cmb[0].ftSZ(l)*0.
   tszCIB = cmb[0].ftSZ(l)*0.

   for i in range(nBands):
      detectornoise += cmb[i].fdetectorNoise(l)*weightsIlcCmb_jointDeproj[lindex][i]**2.

      for j in range(nBands):
         tSZ += np.sqrt(cmb[i].ftSZ(l)*cmb[j].ftSZ(l))*weightsIlcCmb_jointDeproj[lindex][i]*weightsIlcCmb_jointDeproj[lindex][j]

         kSZ += np.sqrt(cmb[i].fkSZ(l)*cmb[j].fkSZ(l))*weightsIlcCmb_jointDeproj[lindex][i]*weightsIlcCmb_jointDeproj[lindex][j]

         radioPS += np.sqrt(cmb[i].fradioPoisson(l)*cmb[j].fradioPoisson(l))*weightsIlcCmb_jointDeproj[lindex][i]*weightsIlcCmb_jointDeproj[lindex][j]

         CIB += np.sqrt(cmb[i].fCIB(l)*cmb[j].fCIB(l))*weightsIlcCmb_jointDeproj[lindex][i]*weightsIlcCmb_jointDeproj[lindex][j]

         #include the cross later if you have time

   ctot = tSZ+kSZ+radioPS+CIB+detectornoise+cmb[0].flensedTT(l)
   return ctot
   
   
def ctot_tszdeproj(l,lindex):
   # here weights really mean epsilon
   tSZ = cmb[0].ftSZ(l)*0.
   kSZ = cmb[0].ftSZ(l)*0.
   radioPS = cmb[0].ftSZ(l)*0.
   CIB = cmb[0].ftSZ(l)*0.
   detectornoise = cmb[0].ftSZ(l)*0.
   tszCIB = cmb[0].ftSZ(l)*0.

   for i in range(nBands):
      detectornoise += cmb[i].fdetectorNoise(l)*weightsIlcCmb_tszdeproj[lindex][i]**2.

      for j in range(nBands):
         tSZ += np.sqrt(cmb[i].ftSZ(l)*cmb[j].ftSZ(l))*weightsIlcCmb_tszdeproj[lindex][i]*weightsIlcCmb_tszdeproj[lindex][j]

         kSZ += np.sqrt(cmb[i].fkSZ(l)*cmb[j].fkSZ(l))*weightsIlcCmb_tszdeproj[lindex][i]*weightsIlcCmb_tszdeproj[lindex][j]

         radioPS += np.sqrt(cmb[i].fradioPoisson(l)*cmb[j].fradioPoisson(l))*weightsIlcCmb_tszdeproj[lindex][i]*weightsIlcCmb_tszdeproj[lindex][j]

         CIB += np.sqrt(cmb[i].fCIB(l)*cmb[j].fCIB(l))*weightsIlcCmb_tszdeproj[lindex][i]*weightsIlcCmb_tszdeproj[lindex][j]

         #include the cross later if you have time

   ctot = tSZ+kSZ+radioPS+CIB+detectornoise+cmb[0].flensedTT(l)
   return ctot
   
   
   

fCtotal = np.array([ctot(np.zeros(6),l,lindex) for lindex,l in enumerate(L)])
fCtotal_deproj = np.array([ctot_deproj(l,lindex) for lindex,l in enumerate(L)])
fCtotal_tszdeproj = np.array([ctot_tszdeproj(l,lindex) for lindex,l in enumerate(L)])
signal = np.array([cmb[0].flensedTT(l) for l in L])
noise = fCtotal - signal
noise_deproj = fCtotal_deproj - signal
noise_tszdeproj = fCtotal_tszdeproj - signal

plt.semilogy(L,signal*L*(L+1)/(2.*np.pi),color='k')
plt.semilogy(L,noise*L*(L+1.)/(2.*np.pi),color='C1',label='standard ILC')
plt.semilogy(L,noise_deproj*L*(L+1.)/(2.*np.pi),color='C2',label='joint deprojection')
plt.semilogy(L,noise_tszdeproj*L*(L+1.)/(2.*np.pi),color='C3',label='tSZ deprojection')
plt.xlim(0,6000)
plt.ylim(0.1,7e3)
plt.xlabel(r'Multipole $\ell$')
plt.ylabel(r'$\ell(\ell+1)C_\ell/(2\pi)$ [$\mu$K^2]')
plt.legend(loc=0,frameon=False)
plt.tight_layout()
plt.savefig('output/ILC_noises.png',bbox_inches='tight')
plt.clf()

plt.figure(figsize=(8,3))
plt.plot(L,noise_deproj/noise,c='C2',label=r'$N^\text{joint deproj}_\ell/N^\text{ILC}_\ell$')
plt.plot(L,noise_tszdeproj/noise,c='C3',label=r'$N^\text{tSZ deproj}_\ell/N^\text{ILC}_\ell$')
plt.xlabel(r'Multipole $\ell$')
plt.tight_layout()
plt.legend(loc=0,frameon=False)
plt.savefig('output/ratio_ILC_noises.png',bbox_inches='tight')
plt.clf()   
   
   
##########################################################################################################################
## Standard QE

def NqCmb(epsilon):  
   #fCtotal =  lambda l: ctot(weights,l)
   fCtotal = np.array([ctot(epsilon,l,lindex) for lindex,l in enumerate(L)])
   interp_ctot = interp1d(L, fCtotal, kind='linear', bounds_error=False, fill_value=0.)
   return baseMap.forecastN0Kappa(cmb[0].flensedTT, interp_ctot, lMin=lMin, lMax=lMax, test=False)(lCen)

def fun_e(i):
   result = np.zeros(6)
   result[i] = 0.01
   return result

def fun_e2(i,j):
   result = np.zeros(6)
   result[i] = 0.01
   result[j] = 0.01
   if i == j: result[i] = 0.02
   return result

print('calculating ILC noise')
n0 = NqCmb(np.zeros(6))

normal_points = np.array([None]*6)
matrix_points = np.array([[None]*6]*6)

print('calculating normal points')
for i in range(6): normal_points[i] = NqCmb(fun_e(i))

print('calculating off-diagonal points')
for i in range(6):
   for j in range(i+1):
      matrix_points[i,j] = NqCmb(fun_e2(i,j))

hessian = np.array([[None]*6]*6)
der = np.array([None]*6)
for i in range(6):
   for j in range(i+1):    
      hessian[i,j] = matrix_points[i,j] - normal_points[i] - normal_points[j] + n0
      hessian[i,j] /= 0.01**2.

for i in range(6): 
   der[i] = normal_points[i]-n0
   der[i] /= 0.01

def NqCmb_interp(epsilon):
   result = n0.copy()
   for i in range(6): 
      result += 0.5 * hessian[i,i] * epsilon[i]**2.
      result += der[i] * epsilon[i]
   for i in range(6):
      for j in range(i):
         result += hessian[i,j] * epsilon[i] * epsilon[j]
   return result

plt.clf()
deltas = np.array([0.01,0.05,0.1,0.2,0.3])
for d in deltas: 
   ep = fun_e(2)*d/0.01
   plt.plot(lCen,NqCmb_interp(ep)/NqCmb(ep),label=str(d))

plt.legend(loc=0,fontsize='x-small',title=r'$\delta$')
plt.xlabel('L')
plt.ylabel(r'$N^\text{interp}_L(\delta\hat{e}_\nu)/N_L(\delta \epsilon_\nu)$')
plt.savefig('output/noise_interp_ratio_QE.png',bbox_inches='tight')
plt.clf()


def loss(partial_eps,fb=0.5):
   epsilon = np.array([0.]+list(partial_eps)+[-sum(partial_eps)])

   s2 =  2. * (ClkCmb + NqCmb_interp(epsilon))**2. / Nmodes 
   norm = np.sum(ClkCmb[I]**2. / s2[I])
   noise =  1./norm

   weights = ILC_weights_lmaxT_3000+epsilon
   #
   Cib_scalefactor = sum([cmb[0].cibPoissonFreqDpdceT(Nu[i])*weights[i]/cmb[0].cibPoissonFreqDpdceT(148.e9) for i in range(nBands)])
   Tsz_scalefactor = sum([cmb[0].tszFreqDpdceT(Nu[i])*weights[i]/cmb[0].tszFreqDpdceT(148.e9) for i in range(nBands)])
   Ksz_scalefactor = sum([cmb[0].kszFreqDpdceT(Nu[i])*weights[i]/cmb[0].kszFreqDpdceT(148.e9) for i in range(nBands)])
   Radiops_scalefactor = sum([cmb[0].radioPoissonFreqDpdceT(Nu[i])*weights[i]/cmb[0].radioPoissonFreqDpdceT(148.e9) for i in range(nBands)])
   #
   Cib = np.sum(np.abs(2.*np.array(data['QE primary, CIB'])[I]*Cib_scalefactor**2.+2.*np.array(data['QE secondary, CIB'])[I]*Cib_scalefactor**2.+np.array(data['QE trispectrum, CIB'])[I]*Cib_scalefactor**4. ) * ClkCmb[I] / s2[I]) / norm
   #
   Tsz = np.sum(np.abs(2.*np.array(data['QE primary, tSZ'])[I]*Tsz_scalefactor**2.+2.*np.array(data['QE secondary, tSZ'])[I]*Tsz_scalefactor**2.+np.array(data['QE trispectrum, tSZ'])[I]*Tsz_scalefactor**4. ) * ClkCmb[I] / s2[I]) / norm
   #
   Ksz = np.sum(np.abs(2.*np.array(data['QE primary, kSZ'])[I]*Ksz_scalefactor**2.+2.*np.array(data['QE secondary, kSZ'])[I]*Ksz_scalefactor**2.+np.array(data['QE trispectrum, kSZ'])[I]*Ksz_scalefactor**4. ) * ClkCmb[I] / s2[I]) / norm
   #
   Radiops = np.sum(np.abs(2.*np.array(data['QE primary, Radio PS'])[I]*Radiops_scalefactor**2.+2.*np.array(data['QE secondary, Radio PS'])[I]*Radiops_scalefactor**2.+np.array(data['QE trispectrum, Radio PS'])[I]*Radiops_scalefactor**4. ) * ClkCmb[I] / s2[I]) / norm

   bias = Cib**2. + Tsz**2. + Ksz**2. + Radiops**2.
   return np.log(noise + fb*bias)*1.e5
   

print('optimizing ILC weights')


fbs = np.logspace(-4,1,300)

result = []
for i,fb in enumerate(fbs):
   print('#########################################################################################')
   print(i)
   if len(result) == 0.: epsilon = scipy.optimize.minimize(loss,np.array([0.]*(nBands-2)),args=(fb),method='SLSQP') 
   else: epsilon = scipy.optimize.minimize(loss,result[-1],args=(fb),method='SLSQP')
   result.append(epsilon.x)
   
epsilons = np.zeros((len(result),nBands))
for i in range(len(result)):
   for j in range(nBands):
      if j == 0: epsilons[i,j]=0.
      elif j < nBands - 1: epsilons[i,j] = np.array(result)[i,j-1]
      elif j == nBands - 1: epsilons[i,j] = -sum(np.array(result)[i,:])
epsilonT = np.transpose(epsilons)


plt.figure(figsize=(8,5))
Nu = np.array([27.e9,39.e9,93.e9,145.e9,225.e9,280.e9])
from headers import*
ILC_deproj = cmbIlc.weightsDeprojTszCIB(3000) - cmbIlc.weightsIlcCmb(3000)
plt.plot(fbs,np.ones(fbs.shape)*ILC_deproj[0],color='k',ls='--')
plt.plot(fbs,np.ones(fbs.shape)*ILC_deproj[1],color='b',ls='--')
plt.plot(fbs,np.ones(fbs.shape)*ILC_deproj[2],color='g',ls='--')
plt.plot(fbs,np.ones(fbs.shape)*ILC_deproj[3],color='r',ls='--')
plt.plot(fbs,np.ones(fbs.shape)*ILC_deproj[4],color='orange',ls='--')
plt.plot(fbs,np.ones(fbs.shape)*ILC_deproj[5],color='pink',ls='--')

plt.plot(fbs,epsilonT[0],color='k',lw=2,label=r'$\nu = 27$ GHz')
plt.plot(fbs,epsilonT[1],color='b',lw=2,label='39 GHz')
plt.plot(fbs,epsilonT[2],color='g',lw=2,label='93 GHz')
plt.plot(fbs,epsilonT[3],color='r',lw=2,label='145 GHz')
plt.plot(fbs,epsilonT[4],color='orange',lw=2,label='225 GHz')
plt.plot(fbs,epsilonT[5],color='pink',lw=2,label='280 GHz')

plt.ylabel(r'$\epsilon_\nu$')
plt.legend(loc=(0,1),frameon=False,fontsize='x-small',ncol=3,handlelength=1)
plt.xlabel(r'$f_b$')
plt.xlim(0,5)
plt.savefig('output/epsilons_varying_loss_functions_SLSQP_QE.png',bbox_inches='tight')
plt.clf()

def bias_to_alens(partial_eps,fb=0.5):
   epsilon = np.array([0.]+list(partial_eps)+[-sum(partial_eps)])

   s2 =  2. * (ClkCmb + NqCmb_interp(epsilon))**2. / Nmodes 
   norm = np.sum(ClkCmb[I]**2. / s2[I])
   noise =  1./norm

   weights = ILC_weights_lmaxT_3000+epsilon
   #
   Cib_scalefactor = sum([cmb[0].cibPoissonFreqDpdceT(Nu[i])*weights[i]/cmb[0].cibPoissonFreqDpdceT(148.e9) for i in range(nBands)])
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

   return np.array([Cib,Tsz,Ksz,Radiops,np.sqrt(noise)])
      
    
plt.axhline(0,-10,10,color='k')  
plt.fill_between(fbs,[-bias_to_alens(peps)[4] for peps in result],\
[bias_to_alens(peps)[4] for peps in result],color='gray',alpha=0.5)
plt.plot(fbs,[bias_to_alens(peps)[0] for peps in result],label='CIB',lw=2)
plt.plot(fbs,[bias_to_alens(peps)[1] for peps in result],label='tSZ',lw=2)
plt.plot(fbs,[bias_to_alens(peps)[2] for peps in result],label='kSZ',lw=2)
plt.plot(fbs,[bias_to_alens(peps)[3] for peps in result],label='Radio PS',lw=2)
plt.legend(loc=0,frameon=True,framealpha=1,fontsize='x-small',ncol=2,\
           title=r'$\text{bias}(A_\text{lens})_s$')
plt.xlabel(r'$f_b$')
plt.savefig('output/prim_bias_and_noise_vs_fb_QE.png',bbox_inches='tight')
plt.clf()




fig,ax = plt.subplots(figsize=(8,5))

plt.axhline(0,-10,10,color='k')  
ax.plot([bias_to_alens(peps)[4] for peps in result],[bias_to_alens(peps)[0] for peps in result],label='CIB',lw=4)
ax.plot([bias_to_alens(peps)[4] for peps in result],[bias_to_alens(peps)[1] for peps in result],label='tSZ',lw=4)
ax.plot([bias_to_alens(peps)[4] for peps in result],[bias_to_alens(peps)[2] for peps in result],label='kSZ',lw=4)
ax.plot([bias_to_alens(peps)[4] for peps in result],[bias_to_alens(peps)[3] for peps in result],label='Radio PS',lw=4)

ax.legend(loc=0,frameon=True,framealpha=1,fontsize='x-small',ncol=2)
ax.set_xlim(0.008,0.012)
ax.set_xlabel(r'$\sigma(A_\text{lens})$')
ax.set_ylabel(r'$\text{bias}(A_\text{lens})$')

fbVnoise_interp = interp1d([bias_to_alens(peps)[4] for peps in result],fbs, kind='linear', bounds_error=False, fill_value=0.)

secax = ax.twiny()

new_tick_locations = np.array([.008, .009, .01, .0108,0.0112])

secax.set_xlim(ax.get_xlim())
secax.set_xticks(new_tick_locations)
secax.set_xticklabels(np.round(fbVnoise_interp(new_tick_locations),2))
secax.set_xlabel(r'$f_b$')

plt.savefig('output/bias_vs_noise_QE.png',bbox_inches='tight')
plt.clf()



import sys
sys.exit()


##########################################################################################################################
## Shear

def NqCmb(epsilon):  
   fCtotal = np.array([ctot(epsilon,l,lindex) for lindex,l in enumerate(L)])
   interp_ctot = interp1d(L, fCtotal, kind='linear', bounds_error=False, fill_value=0.)
   return baseMap.forecastN0KappaShear(cmb[0].flensedTT, interp_ctot, lMin=lMin, lMax=lMax, test=False)(lCen)

def fun_e(i):
   result = np.zeros(6)
   result[i] = 0.01
   return result

def fun_e2(i,j):
   result = np.zeros(6)
   result[i] = 0.01
   result[j] = 0.01
   if i == j: result[i] = 0.02
   return result

print('calculating ILC noise')
n0 = NqCmb(np.zeros(6))

normal_points = np.array([None]*6)
matrix_points = np.array([[None]*6]*6)

print('calculating normal points')
for i in range(6): normal_points[i] = NqCmb(fun_e(i))

print('calculating off-diagonal points')
for i in range(6):
   for j in range(i+1):
      matrix_points[i,j] = NqCmb(fun_e2(i,j))

hessian = np.array([[None]*6]*6)
der = np.array([None]*6)
for i in range(6):
   for j in range(i+1):    
      hessian[i,j] = matrix_points[i,j] - normal_points[i] - normal_points[j] + n0
      hessian[i,j] /= 0.01**2.

for i in range(6): 
   der[i] = normal_points[i]-n0
   der[i] /= 0.01

def NqCmb_interp(epsilon):
   result = n0.copy()
   for i in range(6): 
      result += 0.5 * hessian[i,i] * epsilon[i]**2.
      result += der[i] * epsilon[i]
   for i in range(6):
      for j in range(i):
         result += hessian[i,j] * epsilon[i] * epsilon[j]
   return result

plt.clf()
deltas = np.array([0.01,0.05,0.1,0.2,0.3])
for d in deltas: 
   ep = fun_e(2)*d/0.01
   plt.plot(lCen,NqCmb_interp(ep)/NqCmb(ep),label=str(d))

plt.legend(loc=0,fontsize='x-small',title=r'$\delta$')
plt.xlabel('L')
plt.ylabel(r'$N^\text{interp}_L(\delta\hat{e}_\nu)/N_L(\delta \epsilon_\nu)$')
plt.savefig('output/noise_interp_ratio_Shear.png',bbox_inches='tight')
plt.clf()


def loss(partial_eps,fb=0.5):
   epsilon = np.array([0.]+list(partial_eps)+[-sum(partial_eps)])

   s2 =  2. * (ClkCmb + NqCmb_interp(epsilon))**2. / Nmodes 
   norm = np.sum(ClkCmb[I]**2. / s2[I])
   noise =  1./norm

   weights = ILC_weights_lmaxT_3000+epsilon
   #
   Cib_scalefactor = sum([cmb[0].cibPoissonFreqDpdceT(Nu[i])*weights[i]/cmb[0].cibPoissonFreqDpdceT(148.e9) for i in range(nBands)])
   Tsz_scalefactor = sum([cmb[0].tszFreqDpdceT(Nu[i])*weights[i]/cmb[0].tszFreqDpdceT(148.e9) for i in range(nBands)])
   Ksz_scalefactor = sum([cmb[0].kszFreqDpdceT(Nu[i])*weights[i]/cmb[0].kszFreqDpdceT(148.e9) for i in range(nBands)])
   Radiops_scalefactor = sum([cmb[0].radioPoissonFreqDpdceT(Nu[i])*weights[i]/cmb[0].radioPoissonFreqDpdceT(148.e9) for i in range(nBands)])
   #
   Cib = np.sum(np.abs(2.*np.array(data['Shear primary, CIB'])[I]*Cib_scalefactor**2.+2.*np.array(data['Shear secondary, CIB'])[I]*Cib_scalefactor**2.+np.array(data['Shear trispectrum, CIB'])[I]*Cib_scalefactor**4. ) * ClkCmb[I] / s2[I]) / norm
   #
   Tsz = np.sum(np.abs(2.*np.array(data['Shear primary, tSZ'])[I]*Tsz_scalefactor**2.+2.*np.array(data['Shear secondary, tSZ'])[I]*Tsz_scalefactor**2.+np.array(data['Shear trispectrum, tSZ'])[I]*Tsz_scalefactor**4. ) * ClkCmb[I] / s2[I]) / norm
   #
   Ksz = np.sum(np.abs(2.*np.array(data['Shear primary, kSZ'])[I]*Ksz_scalefactor**2.+2.*np.array(data['Shear secondary, kSZ'])[I]*Ksz_scalefactor**2.+np.array(data['Shear trispectrum, kSZ'])[I]*Ksz_scalefactor**4. ) * ClkCmb[I] / s2[I]) / norm
   #
   Radiops = np.sum(np.abs(2.*np.array(data['Shear primary, Radio PS'])[I]*Radiops_scalefactor**2.+2.*np.array(data['Shear secondary, Radio PS'])[I]*Radiops_scalefactor**2.+np.array(data['Shear trispectrum, Radio PS'])[I]*Radiops_scalefactor**4. ) * ClkCmb[I] / s2[I]) / norm

   bias = Cib**2. + Tsz**2. + Ksz**2. + Radiops**2.
   return np.log(noise + fb*bias)*1.e5
   

print('optimizing ILC weights')

result = []
for i,fb in enumerate(fbs):
   print('#########################################################################################')
   print(i)
   if len(result) == 0.: epsilon = scipy.optimize.minimize(loss,np.array([0.]*(nBands-2)),args=(fb),method='SLSQP') 
   else: epsilon = scipy.optimize.minimize(loss,result[-1],args=(fb),method='SLSQP')
   result.append(epsilon.x)
   
epsilons = np.zeros((len(result),nBands))
for i in range(len(result)):
   for j in range(nBands):
      if j == 0: epsilons[i,j]=0.
      elif j < nBands - 1: epsilons[i,j] = np.array(result)[i,j-1]
      elif j == nBands - 1: epsilons[i,j] = -sum(np.array(result)[i,:])
epsilonT = np.transpose(epsilons)


Nu = np.array([27.e9,39.e9,93.e9,145.e9,225.e9,280.e9])
from headers import*
ILC_deproj = cmbIlc.weightsDeprojTszCIB(3000) - cmbIlc.weightsIlcCmb(3000)
plt.plot(fbs,np.ones(fbs.shape)*ILC_deproj[0],color='k',ls='--')
plt.plot(fbs,np.ones(fbs.shape)*ILC_deproj[1],color='b',ls='--')
plt.plot(fbs,np.ones(fbs.shape)*ILC_deproj[2],color='g',ls='--')
plt.plot(fbs,np.ones(fbs.shape)*ILC_deproj[3],color='r',ls='--')
plt.plot(fbs,np.ones(fbs.shape)*ILC_deproj[4],color='orange',ls='--')
plt.plot(fbs,np.ones(fbs.shape)*ILC_deproj[5],color='pink',ls='--')

plt.plot(fbs,epsilonT[0],color='k',lw=2,label=r'$\nu = 27$ GHz')
plt.plot(fbs,epsilonT[1],color='b',lw=2,label='39 GHz')
plt.plot(fbs,epsilonT[2],color='g',lw=2,label='93 GHz')
plt.plot(fbs,epsilonT[3],color='r',lw=2,label='145 GHz')
plt.plot(fbs,epsilonT[4],color='orange',lw=2,label='225 GHz')
plt.plot(fbs,epsilonT[5],color='pink',lw=2,label='280 GHz')

plt.ylabel(r'$\epsilon_\nu$')
plt.legend(loc=(0,1),frameon=False,fontsize='x-small',ncol=3,handlelength=1)
plt.xlabel(r'$f_b$')
plt.xlim(0,5)
plt.savefig('output/epsilons_varying_loss_functions_SLSQP_Shear.png',bbox_inches='tight')
plt.clf()

def bias_to_alens(partial_eps,fb=0.5):
   epsilon = np.array([0.]+list(partial_eps)+[-sum(partial_eps)])

   s2 =  2. * (ClkCmb + NqCmb_interp(epsilon))**2. / Nmodes 
   norm = np.sum(ClkCmb[I]**2. / s2[I])
   noise =  1./norm

   weights = ILC_weights_lmaxT_3000+epsilon
   #
   Cib_scalefactor = sum([cmb[0].cibPoissonFreqDpdceT(Nu[i])*weights[i]/cmb[0].cibPoissonFreqDpdceT(148.e9) for i in range(nBands)])
   Tsz_scalefactor = sum([cmb[0].tszFreqDpdceT(Nu[i])*weights[i]/cmb[0].tszFreqDpdceT(148.e9) for i in range(nBands)])
   Ksz_scalefactor = sum([cmb[0].kszFreqDpdceT(Nu[i])*weights[i]/cmb[0].kszFreqDpdceT(148.e9) for i in range(nBands)])
   Radiops_scalefactor = sum([cmb[0].radioPoissonFreqDpdceT(Nu[i])*weights[i]/cmb[0].radioPoissonFreqDpdceT(148.e9) for i in range(nBands)])
   #
   Cib = np.sum((2.*np.array(data['Shear primary, CIB'])[I]*Cib_scalefactor**2.+2.*np.array(data['Shear secondary, CIB'])[I]*Cib_scalefactor**2.+np.array(data['Shear trispectrum, CIB'])[I]*Cib_scalefactor**4. ) * ClkCmb[I] / s2[I]) / norm
   #
   Tsz = np.sum((2.*np.array(data['Shear primary, tSZ'])[I]*Tsz_scalefactor**2.+2.*np.array(data['Shear secondary, tSZ'])[I]*Tsz_scalefactor**2.+np.array(data['Shear trispectrum, tSZ'])[I]*Tsz_scalefactor**4. ) * ClkCmb[I] / s2[I]) / norm
   #
   Ksz = np.sum((2.*np.array(data['Shear primary, kSZ'])[I]*Ksz_scalefactor**2.+2.*np.array(data['Shear secondary, kSZ'])[I]*Ksz_scalefactor**2.+np.array(data['Shear trispectrum, kSZ'])[I]*Ksz_scalefactor**4. ) * ClkCmb[I] / s2[I]) / norm
   #
   Radiops = np.sum((2.*np.array(data['Shear primary, Radio PS'])[I]*Radiops_scalefactor**2.+2.*np.array(data['Shear secondary, Radio PS'])[I]*Radiops_scalefactor**2.+np.array(data['Shear trispectrum, Radio PS'])[I]*Radiops_scalefactor**4. ) * ClkCmb[I] / s2[I]) / norm

   return np.array([Cib,Tsz,Ksz,Radiops,np.sqrt(noise)])
      
    
plt.axhline(0,-10,10,color='k')  
plt.fill_between(fbs,[-bias_to_alens(peps)[4] for peps in result],\
[bias_to_alens(peps)[4] for peps in result],color='gray',alpha=0.5)
plt.plot(fbs,[bias_to_alens(peps)[0] for peps in result],label='CIB',lw=2)
plt.plot(fbs,[bias_to_alens(peps)[1] for peps in result],label='tSZ',lw=2)
plt.plot(fbs,[bias_to_alens(peps)[2] for peps in result],label='kSZ',lw=2)
plt.plot(fbs,[bias_to_alens(peps)[3] for peps in result],label='Radio PS',lw=2)
plt.legend(loc=0,frameon=True,framealpha=1,fontsize='x-small',ncol=2,\
           title=r'$\text{bias}(A_\text{lens})_s$')
plt.xlabel(r'$f_b$')
plt.savefig('output/prim_bias_and_noise_vs_fb_Shear.png',bbox_inches='tight')
plt.clf()



plt.axhline(0,-10,10,color='k')  
plt.plot([bias_to_alens(peps)[4] for peps in result],[bias_to_alens(peps)[0] for peps in result],label='CIB',lw=2)
plt.plot([bias_to_alens(peps)[4] for peps in result],[bias_to_alens(peps)[1] for peps in result],label='tSZ',lw=2)
plt.plot([bias_to_alens(peps)[4] for peps in result],[bias_to_alens(peps)[2] for peps in result],label='kSZ',lw=2)
plt.plot([bias_to_alens(peps)[4] for peps in result],[bias_to_alens(peps)[3] for peps in result],label='Radio PS',lw=2)
plt.legend(loc=0,frameon=True,framealpha=1,fontsize='x-small',ncol=2)
plt.xlabel(r'$\sigma(A_\text{lens})$')
plt.ylabel(r'$\text{bias}(A_\text{lens})$')
plt.savefig('output/bias_vs_noise_Shear.png',bbox_inches='tight')
plt.clf()






##########################################################################################################################
## Bias Hardened against point sources

def NqCmb(epsilon):  
   #fCtotal =  lambda l: ctot(weights,l)
   fCtotal = np.array([ctot(epsilon,l,lindex) for lindex,l in enumerate(L)])
   interp_ctot = interp1d(L, fCtotal, kind='linear', bounds_error=False, fill_value=0.)
   return baseMap.forecastN0KappaBH(cmb[0].flensedTT, interp_ctot, lMin=lMin, lMax=lMax, test=False)(lCen)

def fun_e(i):
   result = np.zeros(6)
   result[i] = 0.01
   return result

def fun_e2(i,j):
   result = np.zeros(6)
   result[i] = 0.01
   result[j] = 0.01
   if i == j: result[i] = 0.02
   return result

print('calculating ILC noise')
n0 = NqCmb(np.zeros(6))

normal_points = np.array([None]*6)
matrix_points = np.array([[None]*6]*6)

print('calculating normal points')
for i in range(6): normal_points[i] = NqCmb(fun_e(i))

print('calculating off-diagonal points')
for i in range(6):
   for j in range(i+1):
      matrix_points[i,j] = NqCmb(fun_e2(i,j))

hessian = np.array([[None]*6]*6)
der = np.array([None]*6)
for i in range(6):
   for j in range(i+1):    
      hessian[i,j] = matrix_points[i,j] - normal_points[i] - normal_points[j] + n0
      hessian[i,j] /= 0.01**2.

for i in range(6): 
   der[i] = normal_points[i]-n0
   der[i] /= 0.01

def NqCmb_interp(epsilon):
   result = n0.copy()
   for i in range(6): 
      result += 0.5 * hessian[i,i] * epsilon[i]**2.
      result += der[i] * epsilon[i]
   for i in range(6):
      for j in range(i):
         result += hessian[i,j] * epsilon[i] * epsilon[j]
   return result

plt.clf()
deltas = np.array([0.01,0.05,0.1,0.2,0.3])
for d in deltas: 
   ep = fun_e(2)*d/0.01
   plt.plot(lCen,NqCmb_interp(ep)/NqCmb(ep),label=str(d))

plt.legend(loc=0,fontsize='x-small',title=r'$\delta$')
plt.xlabel('L')
plt.ylabel(r'$N^\text{interp}_L(\delta\hat{e}_\nu)/N_L(\delta \epsilon_\nu)$')
plt.savefig('output/noise_interp_ratio_BH.png',bbox_inches='tight')
plt.clf()


def loss(partial_eps,fb=0.5):
   epsilon = np.array([0.]+list(partial_eps)+[-sum(partial_eps)])

   s2 =  2. * (ClkCmb + NqCmb_interp(epsilon))**2. / Nmodes 
   norm = np.sum(ClkCmb[I]**2. / s2[I])
   noise =  1./norm

   weights = ILC_weights_lmaxT_3000+epsilon
   #
   Cib_scalefactor = sum([cmb[0].cibPoissonFreqDpdceT(Nu[i])*weights[i]/cmb[0].cibPoissonFreqDpdceT(148.e9) for i in range(nBands)])
   Tsz_scalefactor = sum([cmb[0].tszFreqDpdceT(Nu[i])*weights[i]/cmb[0].tszFreqDpdceT(148.e9) for i in range(nBands)])
   Ksz_scalefactor = sum([cmb[0].kszFreqDpdceT(Nu[i])*weights[i]/cmb[0].kszFreqDpdceT(148.e9) for i in range(nBands)])
   Radiops_scalefactor = sum([cmb[0].radioPoissonFreqDpdceT(Nu[i])*weights[i]/cmb[0].radioPoissonFreqDpdceT(148.e9) for i in range(nBands)])
   #
   Cib = np.sum(np.abs(2.*np.array(data['PSH primary, CIB'])[I]*Cib_scalefactor**2.+2.*np.array(data['PSH secondary, CIB'])[I]*Cib_scalefactor**2.+np.array(data['PSH trispectrum, CIB'])[I]*Cib_scalefactor**4. ) * ClkCmb[I] / s2[I]) / norm
   #
   Tsz = np.sum(np.abs(2.*np.array(data['PSH primary, tSZ'])[I]*Tsz_scalefactor**2.+2.*np.array(data['PSH secondary, tSZ'])[I]*Tsz_scalefactor**2.+np.array(data['PSH trispectrum, tSZ'])[I]*Tsz_scalefactor**4. ) * ClkCmb[I] / s2[I]) / norm
   #
   Ksz = np.sum(np.abs(2.*np.array(data['PSH primary, kSZ'])[I]*Ksz_scalefactor**2.+2.*np.array(data['PSH secondary, kSZ'])[I]*Ksz_scalefactor**2.+np.array(data['PSH trispectrum, kSZ'])[I]*Ksz_scalefactor**4. ) * ClkCmb[I] / s2[I]) / norm
   #
   Radiops = np.sum(np.abs(2.*np.array(data['PSH primary, Radio PS'])[I]*Radiops_scalefactor**2.+2.*np.array(data['PSH secondary, Radio PS'])[I]*Radiops_scalefactor**2.+np.array(data['PSH trispectrum, Radio PS'])[I]*Radiops_scalefactor**4. ) * ClkCmb[I] / s2[I]) / norm

   bias = Cib**2. + Tsz**2. + Ksz**2. + Radiops**2.
   return np.log(noise + fb*bias)*1.e5
   

print('optimizing ILC weights')

result = []
for i,fb in enumerate(fbs):
   print('#########################################################################################')
   print(i)
   if len(result) == 0.: epsilon = scipy.optimize.minimize(loss,np.array([0.]*(nBands-2)),args=(fb),method='SLSQP') 
   else: epsilon = scipy.optimize.minimize(loss,result[-1],args=(fb),method='SLSQP')
   result.append(epsilon.x)
   
epsilons = np.zeros((len(result),nBands))
for i in range(len(result)):
   for j in range(nBands):
      if j == 0: epsilons[i,j]=0.
      elif j < nBands - 1: epsilons[i,j] = np.array(result)[i,j-1]
      elif j == nBands - 1: epsilons[i,j] = -sum(np.array(result)[i,:])
epsilonT = np.transpose(epsilons)


Nu = np.array([27.e9,39.e9,93.e9,145.e9,225.e9,280.e9])
from headers import*
ILC_deproj = cmbIlc.weightsDeprojTszCIB(3000) - cmbIlc.weightsIlcCmb(3000)
plt.plot(fbs,np.ones(fbs.shape)*ILC_deproj[0],color='k',ls='--')
plt.plot(fbs,np.ones(fbs.shape)*ILC_deproj[1],color='b',ls='--')
plt.plot(fbs,np.ones(fbs.shape)*ILC_deproj[2],color='g',ls='--')
plt.plot(fbs,np.ones(fbs.shape)*ILC_deproj[3],color='r',ls='--')
plt.plot(fbs,np.ones(fbs.shape)*ILC_deproj[4],color='orange',ls='--')
plt.plot(fbs,np.ones(fbs.shape)*ILC_deproj[5],color='pink',ls='--')

plt.plot(fbs,epsilonT[0],color='k',lw=2,label=r'$\nu = 27$ GHz')
plt.plot(fbs,epsilonT[1],color='b',lw=2,label='39 GHz')
plt.plot(fbs,epsilonT[2],color='g',lw=2,label='93 GHz')
plt.plot(fbs,epsilonT[3],color='r',lw=2,label='145 GHz')
plt.plot(fbs,epsilonT[4],color='orange',lw=2,label='225 GHz')
plt.plot(fbs,epsilonT[5],color='pink',lw=2,label='280 GHz')

plt.ylabel(r'$\epsilon_\nu$')
plt.legend(loc=(0,1),frameon=False,fontsize='x-small',ncol=3,handlelength=1)
plt.xlabel(r'$f_b$')
plt.xlim(0,5)
plt.savefig('output/epsilons_varying_loss_functions_SLSQP_BH.png',bbox_inches='tight')
plt.clf()

def bias_to_alens(partial_eps,fb=0.5):
   epsilon = np.array([0.]+list(partial_eps)+[-sum(partial_eps)])

   s2 =  2. * (ClkCmb + NqCmb_interp(epsilon))**2. / Nmodes 
   norm = np.sum(ClkCmb[I]**2. / s2[I])
   noise =  1./norm

   weights = ILC_weights_lmaxT_3000+epsilon
   #
   Cib_scalefactor = sum([cmb[0].cibPoissonFreqDpdceT(Nu[i])*weights[i]/cmb[0].cibPoissonFreqDpdceT(148.e9) for i in range(nBands)])
   Tsz_scalefactor = sum([cmb[0].tszFreqDpdceT(Nu[i])*weights[i]/cmb[0].tszFreqDpdceT(148.e9) for i in range(nBands)])
   Ksz_scalefactor = sum([cmb[0].kszFreqDpdceT(Nu[i])*weights[i]/cmb[0].kszFreqDpdceT(148.e9) for i in range(nBands)])
   Radiops_scalefactor = sum([cmb[0].radioPoissonFreqDpdceT(Nu[i])*weights[i]/cmb[0].radioPoissonFreqDpdceT(148.e9) for i in range(nBands)])
   #
   Cib = np.sum((2.*np.array(data['PSH primary, CIB'])[I]*Cib_scalefactor**2.+2.*np.array(data['PSH secondary, CIB'])[I]*Cib_scalefactor**2.+np.array(data['PSH trispectrum, CIB'])[I]*Cib_scalefactor**4. ) * ClkCmb[I] / s2[I]) / norm
   #
   Tsz = np.sum((2.*np.array(data['PSH primary, tSZ'])[I]*Tsz_scalefactor**2.+2.*np.array(data['PSH secondary, tSZ'])[I]*Tsz_scalefactor**2.+np.array(data['PSH trispectrum, tSZ'])[I]*Tsz_scalefactor**4. ) * ClkCmb[I] / s2[I]) / norm
   #
   Ksz = np.sum((2.*np.array(data['PSH primary, kSZ'])[I]*Ksz_scalefactor**2.+2.*np.array(data['PSH secondary, kSZ'])[I]*Ksz_scalefactor**2.+np.array(data['PSH trispectrum, kSZ'])[I]*Ksz_scalefactor**4. ) * ClkCmb[I] / s2[I]) / norm
   #
   Radiops = np.sum((2.*np.array(data['PSH primary, Radio PS'])[I]*Radiops_scalefactor**2.+2.*np.array(data['PSH secondary, Radio PS'])[I]*Radiops_scalefactor**2.+np.array(data['PSH trispectrum, Radio PS'])[I]*Radiops_scalefactor**4. ) * ClkCmb[I] / s2[I]) / norm

   return np.array([Cib,Tsz,Ksz,Radiops,np.sqrt(noise)])
      
    
plt.axhline(0,-10,10,color='k')  
plt.fill_between(fbs,[-bias_to_alens(peps)[4] for peps in result],\
[bias_to_alens(peps)[4] for peps in result],color='gray',alpha=0.5)
plt.plot(fbs,[bias_to_alens(peps)[0] for peps in result],label='CIB',lw=2)
plt.plot(fbs,[bias_to_alens(peps)[1] for peps in result],label='tSZ',lw=2)
plt.plot(fbs,[bias_to_alens(peps)[2] for peps in result],label='kSZ',lw=2)
plt.plot(fbs,[bias_to_alens(peps)[3] for peps in result],label='Radio PS',lw=2)
plt.legend(loc=0,frameon=True,framealpha=1,fontsize='x-small',ncol=2,\
           title=r'$\text{bias}(A_\text{lens})_s$')
plt.xlabel(r'$f_b$')
plt.savefig('output/prim_bias_and_noise_vs_fb_BH.png',bbox_inches='tight')
plt.clf()

plt.axhline(0,-10,10,color='k')  
plt.plot([bias_to_alens(peps)[4] for peps in result],[bias_to_alens(peps)[0] for peps in result],label='CIB',lw=2)
plt.plot([bias_to_alens(peps)[4] for peps in result],[bias_to_alens(peps)[1] for peps in result],label='tSZ',lw=2)
plt.plot([bias_to_alens(peps)[4] for peps in result],[bias_to_alens(peps)[2] for peps in result],label='kSZ',lw=2)
plt.plot([bias_to_alens(peps)[4] for peps in result],[bias_to_alens(peps)[3] for peps in result],label='Radio PS',lw=2)
plt.legend(loc=0,frameon=True,framealpha=1,fontsize='x-small',ncol=2)
plt.xlabel(r'$\sigma(A_\text{lens})$')
plt.ylabel(r'$\text{bias}(A_\text{lens})$')
plt.savefig('output/bias_vs_noise_BH.png',bbox_inches='tight')
plt.clf()
