import universe ; from universe import *
import halo_fit ; from halo_fit import *
import weight ; from weight import *
import pn_2d ; from pn_2d import *
import cmb ; from cmb import *
import flat_map ; from flat_map import *

##################################################################
##################################################################

# make plots prettier
import matplotlib
from matplotlib.pyplot import rc
import matplotlib.font_manager
rc('font',**{'size':'22','family':'serif','serif':['CMU serif']})
rc('mathtext', **{'fontset':'cm'})
rc('text', usetex=True)
rc('legend',**{'fontsize':'18'})
matplotlib.rcParams['axes.linewidth'] = 3
matplotlib.rcParams['axes.labelsize'] = 30
matplotlib.rcParams['xtick.labelsize'] = 25 
matplotlib.rcParams['ytick.labelsize'] = 25
matplotlib.rcParams['legend.fontsize'] = 25
#matplotlib.rcParams['legend.title_fontsize'] = 25
matplotlib.rcParams['xtick.major.size'] = 10
matplotlib.rcParams['ytick.major.size'] = 10
matplotlib.rcParams['xtick.minor.size'] = 5
matplotlib.rcParams['ytick.minor.size'] = 5
matplotlib.rcParams['xtick.major.width'] = 3
matplotlib.rcParams['ytick.major.width'] = 3
matplotlib.rcParams['xtick.minor.width'] = 1.5
matplotlib.rcParams['ytick.minor.width'] = 1.5
matplotlib.rcParams['axes.titlesize'] = 30
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

##################################################################
##################################################################
# Map properties

# number of pixels for the flat map
nX = 400 #1200
nY = 400 #1200

# map dimensions in degrees
sizeX = 10.
sizeY = 10.

# basic map object
baseMap = FlatMap(nX=nX, nY=nY, sizeX=sizeX*np.pi/180., sizeY=sizeY*np.pi/180.)

# multipoles to include in the lensing reconstruction
lMin = 30.; lMax = 3e3

# CMB experiment properties

# Adjust the lMin and lMax to the assumptions of the analysis
# CMB S3 specs
cmb = CMB(beam=1., noise=1., lMin=lMin, lMaxT=lMax, lMaxP=lMax, atm=False)

# Total power spectrum, for the lens reconstruction
forCtotal = lambda l: cmb.flensedTT(l) + cmb.fdetectorNoise(l)
#
# reinterpolate: gain factor 10 in speed
L = np.logspace(np.log10(lMin/2.), np.log10(2.*lMax), 1001, 10.)
F = np.array([forCtotal(ll) for ll in L])
cmb.fCtotal = interp1d(L, F, kind='linear', bounds_error=False, fill_value=0.)

u = UnivPlanck15()
halofit = Halofit(u, save=False)
w_cmblens = WeightLensSingle(u, z_source=1100., name="cmblens")
p2d_cmblens = P2dAuto(u, halofit, w_cmblens, save=False)

##################################################################
##################################################################

'''
msk = np.ones((nX,nY))
# mask the edges
n = 20
msk[:n,:] = 0. ; msk[:,:n] = 0.
msk[-n:,:] = 0. ; msk[:,-n:] =0.

def mask_bubble(j,i,r):
    for k in range(nX):
        for l in range(nY):
            if ((k-i)**2+(l-j)**2)**0.5 < r: msk[k,l] = 0.
# eyes
mask_bubble(120,140,20)
mask_bubble(280,140,20)
# mouth
xs = np.linspace(0,110,20)
for x in xs: 
    mask_bubble(85+x,270+10*x**0.3,15)
    mask_bubble(400-(85+x),270+10*x**0.3,15)
# nose
mask_bubble(200,220,10)
# brows
xs = np.linspace(80,160,20)
for x in xs:
    mask_bubble(x,80-x/10,10)
    mask_bubble(400-x,80,10)

# apodize by iterative smoothing
def apod(msk):
    m1 = np.roll(msk,1,axis=0)
    m2 = np.roll(msk,-1,axis=0)
    m3 = np.roll(msk,1,axis=1)
    m4 = np.roll(msk,-1,axis=1)
    return (m1+m2+m3+m4)/4. 
for i in range(50): msk = apod(msk)

# galaxy mask
msk_gal = np.ones((nX,nY))
def mask_bubble(j,i,r):
    for k in range(nX):
        for l in range(nY):
            if ((k-i)**2+(l-j)**2)**0.5 < r: msk_gal[k,l] = 0.
nholes = 100
xholes = np.random.randint(low=20,high=nX-20,size=nholes)
yholes = np.random.randint(low=20,high=nY-20,size=nholes)
rholes = np.abs(np.random.normal(loc=5,scale=2,size=nholes))
for i in range(nholes): mask_bubble(xholes[i],yholes[i],rholes[i])
'''

##################################################################
##################################################################

one = np.ones((nX,nY))
oneFourier = baseMap.fourierComplex(data=one)
twoPiSqDeltaD0 = oneFourier[0,0] # (2\pi)^2 \Delta^D_0

def getMskEff(L,theta,msk_pri):
    Lv = np.array([np.cos(theta),np.sin(theta)])*L/np.sqrt(2)
    mskEffFourier = baseMap.computeNonNormMatrixEffFourier(cmb.flensedTT, cmb.fCtotal, Lv, msk_pri, lMin=lMin, lMax=lMax)
    mskNorm      = (baseMap.computeNonNormMatrixEffFourier(cmb.flensedTT, cmb.fCtotal, Lv, one, lMin=lMin, lMax=lMax)[0,0]/twoPiSqDeltaD0)**-1
    mskEff = baseMap.inverseFourierComplex(dataFourier=mskEffFourier*mskNorm)
    mskEffreal = np.real(mskEff)
    mskEffimag = np.imag(mskEff)
    return mskEffreal,mskEffimag


##################################################################
##################################################################
# Ckg_convoled convolves the input Ckg with the masks
# Flat-sky equivalent to the NaMaster pseudo-Cell calculation

def convolve_thy(msk_kap,msk_gal,thy=p2d_cmblens.fPinterp):
    mskKapFourier = baseMap.fourierComplex(data=msk_kap)
    mskGalFourier = baseMap.fourierComplex(data=msk_gal)

    thyFourier = np.array(list(map(thy,baseMap.lc.flatten())))
    thyFourier = thyFourier.reshape(baseMap.lc.shape)

    one = np.ones((nX,nY))
    oneFourier = baseMap.fourierComplex(data=one)
    twoPiSqDeltaD0 = oneFourier[0,0] # (2\pi)^2 \Delta^D_0

    A = baseMap.inverseFourierComplex(dataFourier=np.conj(mskGalFourier) * mskKapFourier)
    B = baseMap.inverseFourierComplex(dataFourier=thyFourier)
    res = baseMap.fourier(A*B) / twoPiSqDeltaD0
    
    return res

def Ckg_convolved(msk_kap,msk_gal,thy=p2d_cmblens.fPinterp):
    Cobs = convolve_thy(msk_kap,msk_gal,thy)
    ell = baseMap.l.flatten()
    lEdges = np.logspace(np.log10(1.), np.log10(np.max(ell)), 51, 10.)
    Cl, _,_ = stats.binned_statistic(ell, Cobs.flatten(), statistic='mean', bins=lEdges)
    return np.nan_to_num(Cl)

##################################################################
##################################################################
# mock CMB lensing reconstrucion

msk_gal = np.genfromtxt('masks/msk_gal.txt')
msk_pri = np.genfromtxt('masks/msk_pri.txt') 

def create_mock(simidx):
    # check if it exists
    fname = f'mocks/power_spec_{simidx}.txt'
    if os.path.isfile(fname): return True 
    # primary CMB
    np.random.seed(simidx)
    cmb0Fourier = baseMap.genGRF(cmb.funlensedTT)
    cmb0 = baseMap.inverseFourier(cmb0Fourier)
    # kappa map
    np.random.seed(simidx+1000000)
    kCmbFourier = baseMap.genGRF(p2d_cmblens.fPinterp)
    kCmb = baseMap.inverseFourier(kCmbFourier)
    # lens the map
    lensedCmb = baseMap.doLensing(cmb0, kappaFourier=kCmbFourier)
    # mask the maps
    lensedMaskedCmbFourier = baseMap.fourier(lensedCmb*msk_pri)
    maskedGalFourier       = baseMap.fourier(kCmb*msk_gal)
    # reconstruct on masked map
    kHatFourier = baseMap.computeQuadEstKappaNorm(cmb.flensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=lensedMaskedCmbFourier, cache=1)
    #######################
    # power spectra
    lCen,truXtru,_ = baseMap.powerSpectrum(dataFourier = kCmbFourier)
    _,galXtru,_    = baseMap.crossPowerSpectrum(dataFourier1 = maskedGalFourier, dataFourier2=kCmbFourier)
    _,galXrec,_    = baseMap.crossPowerSpectrum(dataFourier1 = maskedGalFourier, dataFourier2=kHatFourier)

    dat = np.array([lCen,truXtru,galXtru,galXrec]).T
    np.savetxt(fname,dat)


##################################################################
##################################################################

def make_isotropic_masks(Ntheta=10):
    #L = np.array([30,50,80,100,200,500,1000,2000])
    #L = np.array([250,300,350,400])
    L = np.array([10])
    thetas = np.linspace(0,np.pi,Ntheta,endpoint=False)
    for LL in L:
        for i,t in enumerate(thetas): 
            mr,mi = getMskEff(LL,t,msk_pri)
            np.savetxt(f"masks/mskEff{LL}_{i}.txt",mr+mi*1.j) 
        #M = np.mean(M,axis=0)
        #np.savetxt(f"masks/mskEffIso{LL}.txt",M)