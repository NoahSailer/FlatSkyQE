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

#tmp = NqCmb(np.zeros(6))

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
deltas = np.array([0.01,0.05,0.1,0.15,0.2])
for d in deltas: 
   #ep = fun_e(2)*d/0.01
   ep = np.array([-d,0.,0.,0.,0.,d])
   plt.plot(lCen,NqCmb_interp(ep)/NqCmb(ep),label=str(d))

plt.legend(loc=0,fontsize='x-small')
plt.xlabel('L')
plt.ylabel(r'$N^\text{interp}_L(\delta\hat{e}_\nu)/N_L(\delta \epsilon_\nu)$')
plt.savefig('output/noise_interp_ratio.pdf',bbox_inches='tight')
plt.clf()


def loss(partial_eps,fb=0.5):
   epsilon = np.array([0.]+list(partial_eps)+[-sum(partial_eps)])

   noise = NqCmb_interp(epsilon)**2.
   
   lCen_100_index = np.where(lCen>=100.)[0][0]

   weights = ILC_weights_lmaxT_3000+epsilon
   #
   Cib_scalefactor = sum([cmb[0].cibPoissonFreqDpdceT(Nu[i])*weights[i]/cmb[0].cibPoissonFreqDpdceT(148.e9) for i in range(nBands)])
   Tsz_scalefactor = sum([cmb[0].tszFreqDpdceT(Nu[i])*weights[i]/cmb[0].tszFreqDpdceT(148.e9) for i in range(nBands)])
   Ksz_scalefactor = sum([cmb[0].kszFreqDpdceT(Nu[i])*weights[i]/cmb[0].kszFreqDpdceT(148.e9) for i in range(nBands)])
   Radiops_scalefactor = sum([cmb[0].radioPoissonFreqDpdceT(Nu[i])*weights[i]/cmb[0].radioPoissonFreqDpdceT(148.e9) for i in range(nBands)])
   #
   Cib = 2.*np.array(data['QE primary, CIB'])*Cib_scalefactor**2.+2.*np.array(data['QE secondary, CIB'])*Cib_scalefactor**2.+np.array(data['QE trispectrum, CIB'])*Cib_scalefactor**4.
   #
   Tsz = 2.*np.array(data['QE primary, tSZ'])*Tsz_scalefactor**2.+2.*np.array(data['QE secondary, tSZ'])*Tsz_scalefactor**2.+np.array(data['QE trispectrum, tSZ'])*Tsz_scalefactor**4. 
   #
   Ksz = 2.*np.array(data['QE primary, kSZ'])*Ksz_scalefactor**2.+2.*np.array(data['QE secondary, kSZ'])*Ksz_scalefactor**2.+np.array(data['QE trispectrum, kSZ'])*Ksz_scalefactor**4.  
   #
   Radiops = 2.*np.array(data['QE primary, Radio PS'])*Radiops_scalefactor**2.+2.*np.array(data['QE secondary, Radio PS'])*Radiops_scalefactor**2.+\
   np.array(data['QE trispectrum, Radio PS'])*Radiops_scalefactor**4. 

   bias = Cib**2. + Tsz**2. + Ksz**2. + Radiops**2.
   #print(noise[lCen_100_index],bias[lCen_100_index])
   return 1.e18*(noise[lCen_100_index] + fb*bias[lCen_100_index])


print('optimizing ILC weights')

result = []
for i,fb in enumerate(np.linspace(0.,500.,200)):
   print('#########################################################################################')
   print(i)
   if len(result) == 0.: epsilon = scipy.optimize.minimize(loss,np.array([0.]*(nBands-2)),args=(fb),method='CG')#,options={'eps':1.e-10,'gtol':1e-8}) 
   else: epsilon = scipy.optimize.minimize(loss,result[-1],args=(fb),method='CG')#,options={'eps':1.e-10,'gtol':1e-8})
   #else: epsilon = scipy.optimize.minimize(loss,np.ones(7),args=(fb),method='CG')
   #epsilon = scipy.optimize.minimize(loss,cmbIlc.weightsDeprojTszCIB(3000)-ILC_weights_lmaxT_3000,args=(fb),constraints={'type':'eq', 'fun': constraint},options={'ftol':0.01},method='SLSQP')
   result.append(epsilon.x)
np.savetxt('modified_ILC_sum_squared.txt',np.array(result))
