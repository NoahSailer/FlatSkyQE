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


##################################################################################
##################################################################################
print("Map properties")

# number of pixels for the flat map
nX = 400 #1200
nY = 400 #1200

# map dimensions in degrees
sizeX = 10.
sizeY = 10.

# basic map object
baseMap = FlatMap(nX=nX, nY=nY, sizeX=sizeX*np.pi/180., sizeY=sizeY*np.pi/180.)

# multipoles to include in the lensing reconstruction
lMin = 5.; lMax = 3500.#4.e3 #30, 3.5e3

# ell bins for power spectra
nBins = 21  # number of bins
lRange = (1., 2.*lMax)  # range for power spectra


##################################################################################
print("CMB experiment properties")

# Adjust the lMin and lMax to the assumptions of the analysis
# CMB S3 specs
cmb = CMB(beam=1.4, noise=6., lMin=lMin, lMaxT=lMax, lMaxP=lMax, atm=False, fg=True)
# Total power spectrum, for the lens reconstruction
forCtotal = lambda l: cmb.ftotalTT(l) #cmb.flensedTT(l) + cmb.fdetectorNoise(l) #cmb.ftotalTT(l) #cmb.flensedTT(l) #+ cmb.fdetectorNoise(l)
# reinterpolate: gain factor 10 in speed
L = np.logspace(np.log10(lMin/2.), np.log10(2.*lMax), 1001, 10.)
F = np.array(list(map(forCtotal, L)))
cmb.fCtotal = interp1d(L, F, kind='linear', bounds_error=False, fill_value=0.)


##################################################################################
print("CMB lensing power spectrum")

u = UnivPlanck15()
halofit = Halofit(u, save=False)
w_cmblens = WeightLensSingle(u, z_source=1100., name="cmblens")
p2d_cmblens = P2dAuto(u, halofit, w_cmblens, save=False)
clusterRadius = 2.2 * np.pi/(180. * 60.) # radians

l = np.genfromtxt('l.txt')
profile = np.genfromtxt('profile.txt')
uTsz = interp1d(l,profile,kind='linear') 

##################################################################################
##################################################################################
print("Compute the statistical uncertainty on the reconstructed lensing convergence")

### CHANGE BACK TO BEING UNLENSED????

print("- standard quadratic estimator")
fNqCmb_fft = baseMap.forecastN0Kappa(cmb.flensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False)
print("- magnification estimator")
fNdCmb_fft = baseMap.forecastN0KappaDilation(cmb.flensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, corr=True, test=False)
print("- shear E-mode estimator")
fNsCmb_fft = baseMap.forecastN0KappaShear(cmb.flensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, corr=True, test=False)
print("- point source hardened estimator")
fNkPSHCmb_fft = baseMap.forecastN0KappaBH(cmb.flensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False, cache=None)
print("- gaussian profile hardened estimator")
fNkPSHGCmb_fft = baseMap.forecastN0KappaBH(cmb.flensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False, cache=None, sigma=clusterRadius)
print("- delta function + gaussian profile hardened estimator")
fNkPSH2Cmb_fft = baseMap.forecastN0KappaBH2(cmb.flensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False, cache=None, sigma=clusterRadius)

fTsz_fft = baseMap.forecastN0KappaBH(cmb.flensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False, cache=None, u = uTsz)
fTsz2_fft = baseMap.forecastN0KappaBH2(cmb.flensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False, cache=None, u = uTsz)
#print("- gradient cleaned estimator")
#fNqGCCmb_fft = baseMap.forecastGradientCleanedKappa(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, cutoff=2300., test=False)
#print "- shear B-mode estimator"
#fNsBCmb_fft = baseMap.forecastN0KappaShearB(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, corr=True, test=False)
#print "- shear E x magnification"
#fNsdCmb_fft = baseMap.forecastN0KappaShearDilation(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMaxS=lMax, lMaxD=lMax, corr=True, test=False)
#print "- shear E x shear B. Not yet working."
#fNssBCmb_fft = baseMap.forecastN0KappaShearShearB(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, corr=True, test=False)


##################################################################################
# colors for plots

cQ = 'r'
cS = 'g'
cD = 'g'
cPSH = 'b'
cPSHG = 'orange'
cPSH2 = 'orange'


##################################################################################
det = baseMap.determinant(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False)

fig = plt.figure(0,figsize=(6,6))
ax = fig.add_subplot(111)

ax.loglog(L,1./det(L),c='k',lw=2,label=r'$(1-N^\kappa_L N^S_L \mathcal{R}^2_L)^{-1}$')
ax.loglog(L,1./(1.-det(L)),c='k',lw=2,linestyle='--',label=r'$(N^\kappa_L N^S_L \mathcal{R}^2_L)^{-1}$')

ax.legend(loc=2,fontsize='x-small',labelspacing=0.2)

ax.set_xlabel(r'$L$')
plt.xlim(40.,6500.)
plt.savefig('output/determinant.pdf', bbox_inches='tight')
plt.clf()

 
##################################################################################
print("Plot noise power spectra")

fig=plt.figure(0)
ax=fig.add_subplot(111)
#
ax.loglog(L, p2d_cmblens.fPinterp(L), 'k-', lw=3, label=r'signal',zorder=1)
#
Nq = fNqCmb_fft(L)
Ns = fNsCmb_fft(L)
Nd = fNdCmb_fft(L)
Npsh = fNkPSHCmb_fft(L)
#Ngc = fNqGCCmb_fft(L)
Npshg = fNkPSHGCmb_fft(L)
Npsh2 = fNkPSH2Cmb_fft(L)
#Nsd = fNsdCmb_fft(L)
#NsB = fNsBCmb_fft(L)
#NssB = fNssBCmb_fft(L)
#
ax.loglog(L, Npsh, c=cPSH, label='PSH',zorder=2)
ax.loglog(L,fTsz_fft(L),label='PH',c='purple')
ax.loglog(L,fTsz2_fft(L),label='PPH',c='orange')
#ax.loglog(L, Npshg, c=cPSHG, label='BHG',zorder=2)
#ax.loglog(L, Npsh2, c=cPSH2, label='BH2',zorder=2)
ax.loglog(L, Ns, c=cS, label=r'Shear E',zorder=1)
#ax.loglog(L, Nd, c=cD, label=r'Mag.')
ax.loglog(L, Nq, c=cQ, lw=3, label=r'QE',zorder=1)
#ax.loglog(L, Ngc, c='orange',label=r'GC')



#ax.loglog(L, NsB, c='y', label=r'shear B')a
#x.loglog(L, Nsd, c='c', ls='--', label=r'shear$\times$dilation')
#ax.loglog(L, 1./(1./Nd + 1./Ns), c=cQ, ls='-.', label=r'naive shear + dilation')
#ax.loglog(L, (Ns*Nd-Nsd**2)/(Ns+Nd-2.*Nsd), c=cQ, ls='--', lw=3, label=r'shear + dilation')
#ax.loglog(L, NssB, c='y', ls='-.', label=r'shear$\times$shear B')
#
ax.legend(loc=2, fontsize='x-small', labelspacing=0.1)
ax.set_xlabel(r'$L$')
plt.xlim(20, 6000)
plt.ylim(1.e-8,5.e-4)
plt.ylabel(r'$N^\kappa_L$')
path = "output/noise.pdf"
plt.savefig(path, bbox_inches='tight')
plt.clf()

##################################################################################
# Plotting the measured power spectra of the sources
lCen = np.genfromtxt('power_spectra_lCen.txt')
Cl = np.genfromtxt('power_spectra_Cl.txt')
ll = np.linspace(40.,7000.,1000000)

ClCIB = Cl[0]*lCen*(lCen+1.)/(2.*np.pi)
ClTsz = Cl[1]*lCen*(lCen+1.)/(2.*np.pi)
ClKsz = Cl[2]*lCen*(lCen+1.)/(2.*np.pi)
ClPS = Cl[3]*lCen*(lCen+1.)/(2.*np.pi)

plt.loglog(ll,cmb.fCtotal(ll)*ll*(ll+1.)/(2.*np.pi),c='k',label='total')
plt.loglog(ll,cmb.flensedTT(ll)*ll*(ll+1.)/(2.*np.pi),c='gray',label='lensed CMB')
plt.loglog(ll,cmb.fdetectorNoise(ll)*ll*(ll+1.)/(2.*np.pi),c='k',linestyle='--',label='detector noise')
plt.loglog(lCen,ClCIB,c='green',label='CIB')
plt.loglog(lCen,ClTsz,c='blue',label='tSZ')
plt.loglog(lCen,ClKsz,c='red',label='kSZ')
plt.loglog(lCen,ClPS,c='purple',label='radio PS')

plt.xlim(40.,7000.)
plt.ylim(0.01,1.e4)
plt.legend(loc=2, fontsize='x-small', labelspacing=0.1)
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\ell(\ell+1)C_\ell/(2\pi)$ [$\mu$K$^2$]')

path = 'output/foreground_spectra.pdf'
plt.savefig(path, bbox_inches='tight')
plt.clf()

##################################################################################



plt.semilogx(ll,uTsz(ll),color='k',lw=3)
plt.xlabel(r'$\ell$')
plt.savefig('output/tsz_profile.pdf',bbox_inches='tight')

import sys
sys.exit()


##################################################################################
#plt.loglog(L, Npsh, c='blue', label='PSH')

numLines = 300

radii = np.linspace(0.01,2.4,numLines)
norm = plt.Normalize()
colors = plt.cm.viridis(norm([i for i in range(numLines)]))
for i,clusterRadius in enumerate(( radii*np.pi/(180. * 60.) )):
   f = baseMap.forecastN0KappaBH(cmb.flensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False, cache=None, sigma=clusterRadius)
   plt.loglog(L,f(L),c=colors[i],lw=0.25)

plt.loglog(L, Npsh, c='b', label='PSH',zorder=2)
plt.loglog(L, fTsz_fft(L), label='PH',c='darkgreen')
plt.loglog(L, Nq, c='r',lw=3, label='QE')

plt.legend(loc=2, fontsize='x-small', labelspacing=0.1)
plt.xlabel(r'$L$')
plt.xlim(20, 6000)
plt.ylim(6.e-8,8.e-7)
plt.ylabel(r'$N^\kappa_L$')
path = "output/sigma_BH2_noise.pdf"
plt.savefig(path, bbox_inches='tight')
plt.clf()



##################################################################################

numMocks = 4
dataAuto = []
dataCross = []
for i in range(numMocks):
   print("Generate GRF unlensed CMB map (debeamed)")

   cmb0Fourier = baseMap.genGRF(cmb.funlensedTT, test=False)
   cmb0 = baseMap.inverseFourier(cmb0Fourier)
   print("plot unlensed CMB map")
   #baseMap.plot(cmb0)
   print("check the power spectrum")
   #lCen, Cl, sCl = baseMap.powerSpectrum(cmb0Fourier, theory=[cmb.funlensedTT], plot=True, save=False)


   ##################################################################################
   print("Generate GRF kappa map")

   kCmbFourier = baseMap.genGRF(p2d_cmblens.fPinterp, test=False)
   kCmb = baseMap.inverseFourier(kCmbFourier)
   print("plot kappa map")
   #baseMap.plot(kCmb)
   print("check the power spectrum")
   #lCen, Cl, sCl = baseMap.powerSpectrum(kCmbFourier, theory=[p2d_cmblens.fPinterp], plot=True, save=False)


   ##################################################################################
   print("Lens the CMB map")

   lensedCmb = baseMap.doLensing(cmb0, kappaFourier=kCmbFourier)
   #lensedCmb = baseMap.doLensingTaylor(cmb0, kappaFourier=kCmbFourier, order=1)
   lensedCmbFourier = baseMap.fourier(lensedCmb)
   print("plot lensed CMB map")
   #baseMap.plot(lensedCmb, save=False)
   print("check the power spectrum")
   #lCen, Cl, sCl = baseMap.powerSpectrum(lensedCmbFourier, theory=[cmb.funlensedTT, cmb.flensedTT], plot=True, save=False)


   ##################################################################################
   print("Add white detector noise (debeamed)")

   noiseFourier = baseMap.genGRF(cmb.fdetectorNoise, test=False)
   totalCmbFourier = lensedCmbFourier #+ noiseFourier
   totalCmb = baseMap.inverseFourier(totalCmbFourier)
   #baseMap.plot(totalCmb)
   print("check the power spectrum")
   #lCen, Cl, sCl = baseMap.powerSpectrum(totalCmbFourier,theory=[cmb.funlensedTT, cmb.flensedTT, cmb.fCtotal], plot=True, save=False)


   ##################################################################################
   #print("Reconstructing kappa: shear estimator")

   #pathSCmb = "./output/sCmb.txt"
   #baseMap.computeQuadEstKappaShearNormCorr(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=totalCmbFourier, test=False, path=pathSCmb)
   #sCmbFourier = baseMap.loadDataFourier(pathSCmb)

   #print("Auto-power: kappa_rec")
   #lCen, Cl, sCl = baseMap.powerSpectrum(sCmbFourier,theory=[p2d_cmblens.fPinterp, fNsCmb_fft], plot=False, save=False)

   #dataAuto.append(Cl)

   #print("Cross-power: kappa_rec x kappa_true")
   #lCen, Cl, sCl = baseMap.crossPowerSpectrum(sCmbFourier, kCmbFourier, theory=[p2d_cmblens.fPinterp, fNsCmb_fft], plot=False, save=False)
   
   #dataCross.append(Cl)


   ##################################################################################
   print("Reconstructing kappa: point source hardened estimator")

   pathPSHCmb = "./output/pshCmb.txt"
   baseMap.computeQuadEstKappaPointSourceHardenedNorm(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=totalCmbFourier, test=False, path=pathPSHCmb)#, sigma=clusterRadius)
   pshCmbFourier = baseMap.loadDataFourier(pathPSHCmb)

   print("Auto-power: kappa_rec")
   lCen, Cl, sCl = baseMap.powerSpectrum(pshCmbFourier,theory=[p2d_cmblens.fPinterp, fNkPSHCmb_fft,fNkPSHGCmb_fft], plot=True, save=False)

   dataAuto.append(Cl)

   print("Cross-power: kappa_rec x kappa_true")
   lCen, Cl, sCl = baseMap.crossPowerSpectrum(pshCmbFourier, kCmbFourier,theory=[p2d_cmblens.fPinterp,fNkPSHCmb_fft,fNkPSHGCmb_fft], plot=True, save=False)

   dataCross.append(Cl)

   import sys
   sys.exit()


   


dataAuto = np.array(dataAuto)
dataCross = np.array(dataCross)

dataAutoMean = np.mean(dataAuto, axis=0)
dataAutoStd = np.std(dataAuto, axis=0) / np.sqrt(numMocks)

dataCrossMean = np.mean(dataCross, axis=0)
dataCrossStd = np.std(dataCross, axis=0) / np.sqrt(numMocks)


fig=plt.figure(0)
ax=fig.add_subplot(111)
#
ax.loglog(L, p2d_cmblens.fPinterp(L), 'k-', lw=3, label=r'signal')
#
ax.loglog(lCen,dataAutoMean-fNkPSHGCmb_fft(lCen), c='blue', label=r'$\hat{\kappa}_L\times\hat{\kappa}_L - N_L$')
Up = dataAutoMean-fNkPSHGCmb_fft(lCen) + dataAutoStd
Down = dataAutoMean-fNkPSHGCmb_fft(lCen) - dataAutoStd
ax.fill_between(lCen, Down, Up, edgecolor='', facecolor='blue', alpha=0.6)
ax.legend(loc=0, fontsize='x-small', labelspacing=0.1)
ax.set_xlabel(r'$L$')
plt.xlim(20, 6000)
plt.ylim(1.e-10,1.e-6)
plt.ylabel(r'$C^\kappa_L$')
path = "output/qeAuto.pdf"
#plt.savefig(path, bbox_inches='tight')
plt.show()
plt.clf()


fig=plt.figure(0)
ax=fig.add_subplot(111)
#
ax.loglog(L, p2d_cmblens.fPinterp(L), 'k-', lw=3, label=r'signal')
#
ax.loglog(lCen,dataCrossMean, c='blue', label=r'$\hat{\kappa}_L\times\kappa_L')
Up = dataCrossMean + dataCrossStd
Down = dataCrossMean - dataCrossStd
ax.fill_between(lCen, Down, Up, edgecolor='', facecolor='blue', alpha=0.6)
ax.legend(loc=0, fontsize='x-small', labelspacing=0.1)
ax.set_xlabel(r'$L$')
plt.xlim(20, 6000)
plt.ylim(1.e-10,1.e-6)
plt.ylabel(r'$C^\kappa_L$')
path = "output/qeCross.pdf"
#plt.savefig(path, bbox_inches='tight')
plt.show()
plt.clf()



import sys
sys.exit()


##################################################################################
print("Reconstructing kappa: standard quadratic estimator")

pathQCmb = "./output/qCmb.txt"
baseMap.computeQuadEstKappaNorm(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=totalCmbFourier, test=False, path=pathQCmb)
qCmbFourier = baseMap.loadDataFourier(pathQCmb)

print("Auto-power: kappa_rec")
lCen, Cl, sCl = baseMap.powerSpectrum(qCmbFourier,theory=[p2d_cmblens.fPinterp, fNqCmb_fft], plot=False, save=False)



print("Cross-power: kappa_rec x kappa_true")
lCen, Cl, sCl = baseMap.crossPowerSpectrum(qCmbFourier, kCmbFourier, theory=[p2d_cmblens.fPinterp, fNqCmb_fft], plot=False, save=False)




##################################################################################
print("Reconstructing kappa: standard quadratic estimator")

pathQCmb = "./output/qCmb.txt"
baseMap.computeQuadEstKappaNorm(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=totalCmbFourier, test=False, path=pathQCmb)
qCmbFourier = baseMap.loadDataFourier(pathQCmb)

print("Auto-power: kappa_rec")
#lCen, Cl, sCl = baseMap.powerSpectrum(qCmbFourier,theory=[p2d_cmblens.fPinterp, fNqCmb_fft], plot=True, save=False)

print("Cross-power: kappa_rec x kappa_true")
#lCen, Cl, sCl = baseMap.crossPowerSpectrum(qCmbFourier, kCmbFourier, theory=[p2d_cmblens.fPinterp, fNqCmb_fft], plot=True, save=False)


##################################################################################
print("Reconstructing kappa: point source hardened estimator")

pathPSHCmb = "./output/pshCmb.txt"
baseMap.computeQuadEstKappaPointSourceHardenedNorm(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=totalCmbFourier, test=False, path=pathPSHCmb)
pshCmbFourier = baseMap.loadDataFourier(pathPSHCmb)

print("Auto-power: kappa_rec")
#lCen, Cl, sCl = baseMap.powerSpectrum(pshCmbFourier,theory=[p2d_cmblens.fPinterp, fNkPSHCmb_fft], plot=True, save=False)

print("Cross-power: kappa_rec x kappa_true")
#lCen, Cl, sCl = baseMap.crossPowerSpectrum(pshCmbFourier, kCmbFourier,theory=[p2d_cmblens.fPinterp,fNkPSHCmb_fft], plot=True, save=False)


##################################################################################
print("Reconstructing kappa: gradient cleaned estimator")

pathGCCmb = "./output/gcCmb.txt"
baseMap.computeGradientCleanedNorm(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=totalCmbFourier, dataFourier2=totalCmbFourier, cutoff=2300., test=False, path=pathGCCmb)
gcCmbFourier = baseMap.loadDataFourier(pathGCCmb)

print("Auto-power: kappa_rec")
lCen, Cl, sCl = baseMap.powerSpectrum(gcCmbFourier,theory=[p2d_cmblens.fPinterp, fNqGCCmb_fft], plot=True, save=False)

print("Cross-power: kappa_rec x kappa_true")
lCen, Cl, sCl = baseMap.crossPowerSpectrum(gcCmbFourier, kCmbFourier,theory=[p2d_cmblens.fPinterp,fNqGCCmb_fft], plot=True, save=False)
