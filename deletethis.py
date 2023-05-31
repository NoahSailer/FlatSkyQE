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

def NqCmb(epsilon): 
   #fCtotal =  lambda l: ctot(weights,l)
   fCtotal = np.array([ctot(epsilon,l,lindex) for lindex,l in enumerate(L)])
   interp_ctot = interp1d(L, fCtotal, kind='linear', bounds_error=False, fill_value=0.)
   return baseMap.forecastN0Kappa(cmb[0].funlensedTT, interp_ctot, lMin=lMin, lMax=lMax, test=False)(lCen)


def tSZ_power(weights,l,lindex):
   # here weights really mean epsilon
   tSZ = cmb[0].ftSZ(l)*0.

   for i in range(nBands):
      for j in range(nBands):
         tSZ += np.sqrt(cmb[i].ftSZ(l)*cmb[j].ftSZ(l))*(weightsIlcCmb[lindex][i]+weights[i])*(weightsIlcCmb[lindex][j]+weights[j])

         #include the cross later if you have time

   ctot = tSZ
   return ctot
   
def CtSZ(epsilon): 
   lindex3000 = np.where(L>=3000.)[0][0]
   return np.array([tSZ_power(epsilon,l,lindex) for lindex,l in enumerate(L)])[lindex3000]

def loss(partial_eps,fb=0.5):
   epsilon = np.array([0.]+list(partial_eps)+[-sum(partial_eps)])

   s2 =  2. * (ClkCmb + NqCmb(epsilon))**2. / Nmodes 
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

   bias = (Cib + Tsz + Ksz + Radiops)**2.
   return bias
   
   
ILC_deproj = cmbIlc.weightsDeprojTszCIB(3000) - cmbIlc.weightsIlcCmb(3000)


import sys
sys.exit()


plt.clf()
fbs = np.logspace(-4,1,300)
loss_ILC = np.array([loss(np.zeros(4),fb=fb) for fb in fbs])
loss_deproj = np.array([loss(ILC_deproj[1:-1],fb=fb) for fb in fbs])
loss_minimizer = np.array([loss(np.array(result[i]),fb=fbs[i]) for i in range(len(fbs))])
plt.semilogx(fbs,loss_ILC,label='ILC')
plt.semilogx(fbs,loss_deproj,label='deproj')
plt.semilogx(fbs,loss_minimizer,label='minimizer')
plt.legend(loc=0)
plt.ylabel('Loss with modulus and log')
plt.xlabel(r'$f_b$')
plt.savefig('output/mod_and_log_SLSQP.pdf',bbox_inches='tight')

plt.clf()
fbs = np.logspace(-4,1,300)
loss_ILC = np.array([loss_no_log(np.zeros(4),fb=fb) for fb in fbs])
loss_deproj = np.array([loss_no_log(ILC_deproj[1:-1],fb=fb) for fb in fbs])
loss_minimizer = np.array([loss_no_log(np.array(result_no_log[i]),fb=fbs[i]) for i in range(len(fbs))])
plt.loglog(fbs,loss_ILC,label='ILC')
plt.loglog(fbs,loss_deproj,label='deproj')
plt.loglog(fbs,loss_minimizer,label='minimizer')
plt.legend(loc=0)
plt.ylabel('Loss with modulus')
plt.xlabel(r'$f_b$')
plt.savefig('output/mod_SLSQP.pdf',bbox_inches='tight')


plt.clf()
fbs = np.logspace(-4,1,300)
loss_ILC = np.array([loss_no_log_no_mod(np.zeros(4),fb=fb) for fb in fbs])
loss_deproj = np.array([loss_no_log_no_mod(ILC_deproj[1:-1],fb=fb) for fb in fbs])
loss_minimizer = np.array([loss_no_log_no_mod(np.array(result_no_log_no_mod[i]),fb=fbs[i]) for i in range(len(fbs))])
plt.loglog(fbs,loss_ILC,label='ILC')
plt.loglog(fbs,loss_deproj,label='deproj')
plt.loglog(fbs,loss_minimizer,label='minimizer')
plt.ylabel('Original loss')
plt.legend(loc=0)
plt.xlabel(r'$f_b$')
plt.savefig('output/original_SLSQP.pdf',bbox_inches='tight')




epsilons = np.zeros((len(result),nBands))
for i in range(len(result)):
   for j in range(nBands):
      if j == 0: epsilons[i,j]=0.
      elif j < nBands - 1: epsilons[i,j] = np.array(result)[i,j-1]
      elif j == nBands - 1: epsilons[i,j] = -sum(np.array(result)[i,:])
epsilonT = np.transpose(epsilons)

epsilons_no_log = np.zeros((len(result),nBands))
for i in range(len(result)):
   for j in range(nBands):
      if j == 0: epsilons[i,j]=0.
      elif j < nBands - 1: epsilons_no_log[i,j] = np.array(result_no_log)[i,j-1]
      elif j == nBands - 1: epsilons_no_log[i,j] = -sum(np.array(result_no_log)[i,:])
epsilonT_no_log = np.transpose(epsilons_no_log)

epsilons_no_log_no_mod = np.zeros((len(result),nBands))
for i in range(len(result)):
   for j in range(nBands):
      if j == 0: epsilons[i,j]=0.
      elif j < nBands - 1: epsilons_no_log_no_mod[i,j] = np.array(result_no_log_no_mod)[i,j-1]
      elif j == nBands - 1: epsilons_no_log_no_mod[i,j] = -sum(np.array(result_no_log_no_mod)[i,:])
epsilonT_no_log_no_mod = np.transpose(epsilons_no_log_no_mod)


plt.clf()
plt.semilogx(fbs,epsilonT[0],color='k')
plt.semilogx(fbs,epsilonT[1],color='b')
plt.semilogx(fbs,epsilonT[2],color='g')
plt.semilogx(fbs,epsilonT[3],color='r',label='Original loss')
plt.semilogx(fbs,epsilonT[4],color='orange')
plt.semilogx(fbs,epsilonT[5],color='pink')

plt.semilogx(fbs,epsilonT_no_log[0],color='k',ls='--')
plt.semilogx(fbs,epsilonT_no_log[1],color='b',ls='--')
plt.semilogx(fbs,epsilonT_no_log[2],color='g',ls='--')
plt.semilogx(fbs,epsilonT_no_log[3],color='r',ls='--',label='Loss with modulus')
plt.semilogx(fbs,epsilonT_no_log[4],color='orange',ls='--')
plt.semilogx(fbs,epsilonT_no_log[5],color='pink',ls='--')

plt.semilogx(fbs,epsilonT_no_log_no_mod[0],color='k',ls='-.')
plt.semilogx(fbs,epsilonT_no_log_no_mod[1],color='b',ls='-.')
plt.semilogx(fbs,epsilonT_no_log_no_mod[2],color='g',ls='-.')
plt.semilogx(fbs,epsilonT_no_log_no_mod[3],color='r',ls='-.',label='Loss with modulus and log')
plt.semilogx(fbs,epsilonT_no_log_no_mod[4],color='orange',ls='-.')
plt.semilogx(fbs,epsilonT_no_log_no_mod[5],color='pink',ls='-.')
plt.ylabel(r'$\epsilon_\nu$')
plt.legend(loc=0,frameon=False,fontsize='x-small')
plt.xlabel(r'$f_b$')
plt.savefig('output/epsilons_varying_loss_functions_SLSQP.pdf',bbox_inches='tight')


plt.clf()
ts = np.linspace(0,1,100)
loss_vs_t = np.array([loss(ILC_deproj[1:-1]*t,fb=0.1) for t in ts])
plt.plot(ts,loss_vs_t,label=r'loss$(t\epsilon^\text{deproj}_\nu)$, $f_b=0.1$')

loss_vs_t = np.array([loss(ILC_deproj[1:-1]*t,fb=1) for t in ts])
plt.plot(ts,loss_vs_t,label=r'loss$(t\epsilon^\text{deproj}_\nu)$, $f_b=1$')

loss_vs_t = np.array([loss(ILC_deproj[1:-1]*t,fb=10) for t in ts])
plt.plot(ts,loss_vs_t,label=r'loss$(t\epsilon^\text{deproj}_\nu)$, $f_b=10$')

loss_vs_t = np.array([loss(ILC_deproj[1:-1]*t,fb=100) for t in ts])
plt.plot(ts,loss_vs_t,label=r'loss$(t\epsilon^\text{deproj}_\nu)$, $f_b=100$')
plt.xlabel('t')
plt.legend(loc=0,frameon=False,fontsize='x-small')
plt.show()



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

   return np.array([Cib,Tsz,Ksz,Radiops])
      
plt.clf()
bias_vs_t = np.transpose(np.array([bias_to_alens(ILC_deproj[1:-1]*t) for t in ts]))
plt.plot(ts,bias_vs_t[0],label='CIB')
plt.plot(ts,bias_vs_t[1],label='tSZ')
plt.plot(ts,bias_vs_t[2],label='kSZ')
plt.plot(ts,bias_vs_t[3],label='Radio')
plt.axhline(0,0,1,color='k')
plt.xlabel('t')
plt.ylabel(r'Bias to lensing amplitude')
plt.legend(loc=0,frameon=False,fontsize='x-small')
plt.show()


np.savetxt('modified_ILC_sum_squared.txt',np.array(result))
###############################################################################################################


#loss vs fb. Three curves. log and no log, with and without modulus 

#parameterize ILC -> deproj, see if there are local minima. 


#np.savetxt('hessian.txt',hessian)

#def minimize(fb,initial_guess,step=1.,stop=1.e-7):
#   result = initial_guess.copy()
#   alpha = np.ones(len(initial_guess))*step

#   diff = 1000.
#   while diff >= stop:
#      loss1 = loss(result,fb=fb)
#      grad = gradient(result,fb=fb)
#      result -= alpha*grad
#      result[-1] = -1.*sum(result[:-1])
#      loss2 = loss(result,fb=fb)
#      diff = np.abs(loss2-loss1)
#   return result




#def gradient(epsilon_lam,fb=0.5):
#   n = len(epsilon_lam)
#   result = np.zeros(n)
#   for i in range(n-1):
#      delta = np.zeros(n)
#      delta[i] = 0.01
#      der = loss(epsilon_lam+delta,fb=fb) - loss(epsilon_lam,fb=fb)
#      der /= 0.01
#      result[i] = der
#   return result

#def loss(epsilon,fb=0.5):  return noise_Alens(epsilon) + fb*bias_Alens(ILC_weights_lmaxT_3000+epsilon) 
