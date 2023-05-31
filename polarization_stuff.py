# imports
from __future__ import print_function
from __future__ import division
from builtins import map
import universe
from universe import *
import halo_fit
from halo_fit import *
import weight
from weight import *
import pn_2d
from pn_2d import *
import cmb
from cmb import *
import flat_map
from flat_map import *
import pandas as pd
import cmb_ilc
from cmb_ilc import *
import scipy
from headers import*
from SO_noise2 import*
from scipy.interpolate import interp1d
import vegas
import multiprocessing as mp
##################################################################################
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
##################################################################################
# create the CMB object
nu = 153.e9
#cmb = CMB(beam=0.25, noise=0.5, nu1=nu, nu2=nu, fg=True,atm=False)  #CMB-HD (cause why not)

cmb = CMB(beam=1.4, noise=1, nu1=nu, nu2=nu, fg=True,atm=False)
#cmb = CMB(beam=1.4, noise=6, nu1=nu, nu2=nu, fg=True,atm=False)

# get CMB lensing C_ell
u = UnivPlanck15()
halofit = Halofit(u, save=False)
w_cmblens = WeightLensSingle(u, z_source=1100., name="cmblens")
p2d_cmblens = P2dAuto(u, halofit, w_cmblens, save=False)

######################
# interpolate the Ctot's and inverse of covariance matrix to make things faster
lmin = 10
lmaxT = 3500
lmaxP = 5000
lmax = max(lmaxT,lmaxP)

L = np.logspace(np.log10(lmin/2.), np.log10(2.*lmax), 5001, 10.)
#
Ctt = np.array([cmb.ftotalTT(l) for l in L])
Ctt_ = interp1d(L, Ctt, kind='linear', bounds_error=False, fill_value=0.)
#
Cte = np.array([cmb.ftotalTE(l) for l in L])
Cte_ = interp1d(L, Cte, kind='linear', bounds_error=False, fill_value=0.)
#
Cee = np.array([cmb.ftotalEE(l) for l in L])
Cee_ = interp1d(L, Cee, kind='linear', bounds_error=False, fill_value=0.)
#

##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################

#Cbb = np.array([cmb.ftotalBB(l) for l in L])
Cbb = np.array([cmb.fdetectorNoise(l)*2 for l in L])
Cbb_ = interp1d(L, Cbb, kind='linear', bounds_error=False, fill_value=0.)
#
C00 = np.array([0. for l in L])
C00_ = interp1d(L, C00, kind='linear', bounds_error=False, fill_value=0.)

# covariance matrix in the basis {T, E, B}

CC_ = lambda l: np.array([[Ctt_(l), Cte_(l), C00_(l)],
                          [Cte_(l), Cee_(l), C00_(l)],
                          [C00_(l), C00_(l), Cbb_(l)]])


CCinv = np.array([np.linalg.inv(CC_(l)) for l in L])
CCinv_ = interp1d(L, CCinv, kind='linear', bounds_error=False, fill_value=0., axis=0)


##################################################################################
'''
nX = 400
nY = 400
size = 10.  # degrees, determined by the Sehgal cutouts
baseMap = FlatMap(nX=nX, nY=nY, sizeX=size*np.pi/180., sizeY=size*np.pi/180.)

lMin = 10.
lMax = 3500.
L = np.logspace(np.log10(lMin/2.), np.log10(2.*lMax), 501, 10.)
#
fCtotal = np.array([cmb.ftotalTT(l) for l in L])
interp_ctot = interp1d(L, fCtotal, kind='linear', bounds_error=False, fill_value=0.)
#
fNqCmb_fft = baseMap.forecastN0Kappa(cmb.flensedTT, interp_ctot, lMin=lMin, lMax=lMax, test=False)
'''
##################################################################################
##################################################################################
##################################################################################
# the real part of the code


##### MC integration parameters (change this in notebook)
nitn = 10
neval = 1000


##### Basic Geometry

def get_angle(l):
    '''
    l is a 2D array
    returns angle from x-axis (counter-clockwise) from [0,2*pi)
    '''
    lx,ly = l[0],l[1]
    l = np.sqrt(lx**2 + ly**2)
    if ly >= 0: return np.arccos(lx/l)
    else: return 2*np.pi - np.arccos(lx/l) 

def get_basics(l1vec,l2vec):
    '''
    l1vec and l2vec are 2D arrays
    '''
    l1 = np.sqrt(np.dot(l1vec,l1vec))
    phi1 = get_angle(l1vec)
    l2 = np.sqrt(np.dot(l2vec,l2vec))
    phi2 = get_angle(l2vec)
    Lvec = l1vec + l2vec
    L = np.sqrt(np.dot(Lvec,Lvec))
    prefac = 2*Lvec/L**2
    return l1,phi1,l2,phi2,Lvec,L,prefac

##### Linear Responses to lensing
##### assumes that l1vec,l2vec are arrays of length 2

def fTT(l1vec,l2vec,lensed=True):
    l1,phi1,l2,phi2,Lvec,L,prefac = get_basics(l1vec,l2vec)
    if lensed: 
        A = l1vec * cmb.flensedTT(l1)
        B = l2vec * cmb.flensedTT(l2)
    else: 
        A = l1vec * cmb.funlensedTT(l1) 
        B = l2vec * cmb.funlensedTT(l2)
    result = np.dot(prefac,A+B)
    return np.nan_to_num(result)

def fTE(l1vec,l2vec):
    l1,phi1,l2,phi2,Lvec,L,prefac = get_basics(l1vec,l2vec)
    A = l1vec * cmb.flensedTE(l1) * np.cos(2*phi1 - 2*phi2)
    B = l2vec * cmb.flensedTE(l2)
    result = np.dot(prefac,A+B)
    return result

def fTB(l1vec,l2vec):
    l1,phi1,l2,phi2,Lvec,L,prefac = get_basics(l1vec,l2vec)
    A = l1vec * cmb.flensedTE(l1) * np.sin(2*phi1 - 2*phi2)
    result = np.dot(prefac,A)
    return np.nan_to_num(result)

def fEE(l1vec,l2vec):
    l1,phi1,l2,phi2,Lvec,L,prefac = get_basics(l1vec,l2vec)
    A = l1vec * cmb.flensedEE(l1) * np.cos(2*phi1 - 2*phi2)
    B = l2vec * cmb.flensedEE(l2) * np.cos(2*phi1 - 2*phi2)
    result = np.dot(prefac,A+B)
    return np.nan_to_num(result)

def fEB(l1vec,l2vec):
    l1,phi1,l2,phi2,Lvec,L,prefac = get_basics(l1vec,l2vec)
    A = l1vec * cmb.flensedEE(l1) * np.sin(2*phi1 - 2*phi2)
    result = np.dot(prefac,A)
    return np.nan_to_num(result)

def f_matrix(l1vec,l2vec):
    result = np.array([[fTT(l1vec,l2vec), fTE(l1vec,l2vec), fTB(l1vec,l2vec)],
                       [fTE(l2vec,l1vec), fEE(l1vec,l2vec), fEB(l1vec,l2vec)],
                       [fTB(l2vec,l1vec), fEB(l2vec,l1vec), 0.              ]])
    return result


##### Linear Responses of foregrounds
##### I'm setting A = B = C = u = 1

def gTTT(l1vec,l2vec): return 1.
def gTET(l1vec,l2vec): return 0.
def gTBT(l1vec,l2vec): return 0.
def gEET(l1vec,l2vec):
    l1,phi1,l2,phi2,Lvec,L,prefac = get_basics(l1vec,l2vec)
    return np.cos(2*phi1 - 2*phi2)
def gEBT(l1vec,l2vec): 
    l1,phi1,l2,phi2,Lvec,L,prefac = get_basics(l1vec,l2vec)
    return np.sin(2*phi1 - 2*phi2)
def gBBT(l1vec,l2vec): 
    l1,phi1,l2,phi2,Lvec,L,prefac = get_basics(l1vec,l2vec)
    return np.cos(2*phi1 - 2*phi2)


def gTTE(l1vec,l2vec): return 0.
def gTEE(l1vec,l2vec): 
    l1,phi1,l2,phi2,Lvec,L,prefac = get_basics(l1vec,l2vec)
    phiL = get_angle(Lvec)
    return np.cos(2*phiL - 2*phi2)
def gTBE(l1vec,l2vec):
    l1,phi1,l2,phi2,Lvec,L,prefac = get_basics(l1vec,l2vec)
    phiL = get_angle(Lvec)
    return np.sin(2*phiL - 2*phi2)
def gEEE(l1vec,l2vec): return 0.
def gEBE(l1vec,l2vec): return 0.
def gBBE(l1vec,l2vec): return 0.


def gTTB(l1vec,l2vec): return 0.
def gTEB(l1vec,l2vec): 
    l1,phi1,l2,phi2,Lvec,L,prefac = get_basics(l1vec,l2vec)
    phiL = get_angle(Lvec)
    return np.sin(2*phi2 - 2*phiL)
def gTBB(l1vec,l2vec):
    l1,phi1,l2,phi2,Lvec,L,prefac = get_basics(l1vec,l2vec)
    phiL = get_angle(Lvec)
    return np.cos(2*phi2 - 2*phiL)
def gEEB(l1vec,l2vec): return 0.
def gEBB(l1vec,l2vec): return 0.
def gBBB(l1vec,l2vec): return 0.


def gT_matrix(l1vec,l2vec,A=1.,C=0.):
    result = np.array([[A*gTTT(l1vec,l2vec), gTET(l1vec,l2vec),   gTBT(l1vec,l2vec)],
                       [gTET(l2vec,l1vec),   C*gEET(l1vec,l2vec), C*gEBT(l1vec,l2vec)],
                       [gTBT(l2vec,l1vec),   C*gEBT(l2vec,l1vec), C*gBBT(l1vec,l2vec)]])
    return result
    
def gE_matrix(l1vec,l2vec,B=1.):
    result = np.array([[gTTE(l1vec,l2vec), gTEE(l1vec,l2vec), gTBE(l1vec,l2vec)],
                       [gTEE(l2vec,l1vec), gEEE(l1vec,l2vec), gEBE(l1vec,l2vec)],
                       [gTBE(l2vec,l1vec), gEBE(l2vec,l1vec), gBBE(l1vec,l2vec)]])
    return B*result
    
def gB_matrix(l1vec,l2vec,B=1.):
    result = np.array([[gTTB(l1vec,l2vec), gTEB(l1vec,l2vec), gTBB(l1vec,l2vec)],
                       [gTEB(l2vec,l1vec), gEEB(l1vec,l2vec), gEBB(l1vec,l2vec)],
                       [gTBB(l2vec,l1vec), gEBB(l2vec,l1vec), gBBB(l1vec,l2vec)]])
    return B*result

##### Minimum variance* weights
##### *forced to be FFTable

def F_MV(f,l1vec, l2vec, ftotX, ftotY, ftotXY, ell_maxX, ell_maxY):
    l1,phi1,l2,phi2,Lvec,L,prefac = get_basics(l1vec,l2vec)
    if l1 > ell_maxX or l2 > ell_maxY: return 0.
    Cl1 = ftotX(l1)
    Cl2 = ftotY(l2)
    if Cl1 <= 0 or Cl2 <= 0: return 0.
    return f(l1vec,l2vec)/Cl1/Cl2

def F_sym_MV(f,l1vec, l2vec, ftotX, ftotY, ftotXY, ell_maxX, ell_maxY):
    l1,phi1,l2,phi2,Lvec,L,prefac = get_basics(l1vec,l2vec)
    if l1 > ell_maxX or l2 > ell_maxY: return 0.
    ClX1 = ftotX(l1) ; ClX2 = ftotX(l2)
    ClY1 = ftotY(l1) ; ClY2 = ftotY(l2)
    if ClX1 <= 0 or ClX2 <= 0 or ClY1 <= 0 or ClY2 <= 0: return 0.
    return ( f(l1vec,l2vec) + f(l2vec,l1vec))/( ClX1*ClY2 + ClY1*ClX2)
    
def F_GMV(f, l1vec, l2vec, ell_maxT, ell_maxP):
    l1,phi1,l2,phi2,Lvec,L,prefac = get_basics(l1vec,l2vec)
    f_m = f(l1vec,l2vec)
    C1inv = CCinv_(l1)
    C2inv = CCinv_(l2)
    result = np.dot(np.dot(C1inv,f_m),C2inv) / 2
    # impose ell_max cuts
    TT = (l1 <= ell_maxT)*(l2 <= ell_maxT)
    PP = (l1 <= ell_maxP)*(l2 <= ell_maxP)
    TP = (l1 <= ell_maxT)*(l2 <= ell_maxP)
    PT = (l1 <= ell_maxP)*(l2 <= ell_maxT)
    ell_cut = np.array([[TT, TP, TP],
                        [PT, PP, PP],
                        [PT, PP, PP]])
    return result * ell_cut
    


# double checking my algebra    
def F_GMV_TMP(f, l1vec, l2vec, ell_maxT, ell_maxP):
    l1,phi1,l2,phi2,Lvec,L,prefac = get_basics(l1vec,l2vec)
    phiL = get_angle(Lvec)
    #
    CTT1 = CC_(l1)[0,0] ; CTT2 = CC_(l2)[0,0]
    CTE1 = CC_(l1)[0,1] ; CTE2 = CC_(l2)[0,1]
    CEE1 = CC_(l1)[1,1] ; CEE2 = CC_(l2)[1,1]
    CBB1 = CC_(l1)[2,2] ; CBB2 = CC_(l2)[2,2]
    D1 = CTT1*CEE1 - CTE1**2
    D2 = CTT2*CEE2 - CTE2**2
    #
    FTT = CEE1*CEE2*fTT(l1vec,l2vec) + CTE1*CTE2*fEE(l1vec,l2vec)
    FTT -= CEE1*CTE2*fTE(l1vec,l2vec) + CTE1*CEE2*fTE(l2vec,l1vec)
    FTT /= 2*D1*D2
    # 
    FTE = CEE1*CTT2*fTE(l1vec,l2vec) + CTE1*CTE2*fTE(l2vec,l2vec)
    FTE -= CTE1*CTT2*fEE(l1vec,l2vec) + CEE1*CTE2*fTT(l1vec,l2vec)
    FTE /= 2*D1*D2
    # 
    FET = CEE2*CTT1*fTE(l2vec,l1vec) + CTE2*CTE1*fTE(l1vec,l2vec)
    FET -= CTE2*CTT1*fEE(l2vec,l1vec) + CEE2*CTE1*fTT(l2vec,l1vec)
    FET /= 2*D1*D2
    #
    FTB = CEE1*fTB(l1vec,l2vec) - CTE1*fEB(l1vec,l2vec)
    FTB /= 2*D1*CBB2
    #
    FBT = CEE2*fTB(l2vec,l1vec) - CTE2*fEB(l2vec,l1vec)
    FBT /= 2*D2*CBB1
    #
    FEE = CTE1*CTE2*fTT(l1vec,l2vec) + CTT1*CTT2*fEE(l1vec,l2vec)
    FEE -= CTE1*CTT2*fTE(l1vec,l2vec) + CTT1*CTE2*fTE(l2vec,l1vec)
    FEE /= 2*D1*D2
    #
    FEB = CTT1*fEB(l1vec,l2vec) - CTE1*fTB(l1vec,l2vec)
    FEB /= 2*D1*CBB2
    #
    FBE = CTT2*fEB(l2vec,l1vec) - CTE1*fTB(l2vec,l1vec)
    FBE /= 2*D2*CBB1
    #
    result = np.array([[FTT, FTE, FTB],
                       [FET, FEE, FEB],
                       [FBT, FBE, 0. ]])
    # impose ell_max cuts
    TT = (l1 <= ell_maxT)*(l2 <= ell_maxT)
    PP = (l1 <= ell_maxP)*(l2 <= ell_maxP)
    TP = (l1 <= ell_maxT)*(l2 <= ell_maxP)
    PT = (l1 <= ell_maxP)*(l2 <= ell_maxT)
    ell_cut = np.array([[TT, TP, TP],
                        [PT, PP, PP],
                        [PT, PP, PP]])
    return result * ell_cut

##### Shear weights
##### 

def FTT_Shear(f,l1vec, l2vec, ftotX, ftotY, ftotXY, ell_maxX, ell_maxY):
    l1,phi1,l2,phi2,Lvec,L,prefac = get_basics(l1vec,l2vec)
    if l1 > ell_maxX or l2 > ell_maxY: return 0.
    Cl1 = ftotX(l1)
    dCdlnl = (cmb.flensedTT(l1+1) - cmb.flensedTT(l1)) / (np.log(l1+1) - np.log(l1))
    if Cl1 <= 0: return 0.
    phiL = get_angle(Lvec)
    return dCdlnl*np.cos(2*phiL-2*phi1)/Cl1**2

def FTE_Shear(f,l1vec, l2vec, ftotX, ftotY, ftotXY, ell_maxX, ell_maxY):
    l1,phi1,l2,phi2,Lvec,L,prefac = get_basics(l1vec,l2vec)
    if l1 > ell_maxX or l2 > ell_maxY: return 0.
    dlsqCdlnl = (cmb.flensedTE(l1+1) - cmb.flensedTE(l1)) / (np.log(l1+1) - np.log(l1))
    dlsqCdlnl += 2 * cmb.flensedTE(l1) 
    Cl1 = ftotX(l1)
    Cl2 = ftotY(l1)
    if Cl1 <= 0 or Cl2 <= 0: return 0.
    return dlsqCdlnl/Cl1/Cl2

def FTB_Shear(f,l1vec, l2vec, ftotX, ftotY, ftotXY, ell_maxX, ell_maxY):
    l1,phi1,l2,phi2,Lvec,L,prefac = get_basics(l1vec,l2vec)
    if l1 > ell_maxX or l2 > ell_maxY: return 0.
    phiL = get_angle(Lvec)
    s1 = np.sin(phiL-phi1)
    s3 = np.sin(3*phiL-3*phi1)
    Cl1 = ftotX(l1)
    Cl2 = ftotY(l1)
    if Cl1 <= 0 or Cl2 <= 0: return 0.
    return L*(s1+s3)*cmb.flensedTE(l1)/l1/Cl1/Cl2

def FEE_Shear(f,l1vec, l2vec, ftotX, ftotY, ftotXY, ell_maxX, ell_maxY):
    l1,phi1,l2,phi2,Lvec,L,prefac = get_basics(l1vec,l2vec)
    if l1 > ell_maxX or l2 > ell_maxY: return 0.
    Cl1 = ftotX(l1)
    dCdlnl = (cmb.flensedEE(l1+1) - cmb.flensedEE(l1)) / (np.log(l1+1) - np.log(l1))
    if Cl1 <= 0: return 0.
    phiL = get_angle(Lvec)
    return dCdlnl*np.cos(2*phiL-2*phi1)/Cl1**2

def FEB_Shear(f,l1vec, l2vec, ftotX, ftotY, ftotXY, ell_maxX, ell_maxY):
    l1,phi1,l2,phi2,Lvec,L,prefac = get_basics(l1vec,l2vec)
    if l1 > ell_maxX or l2 > ell_maxY: return 0.
    Cl1 = ftotX(l1)
    Cl2 = ftotY(l1)
    if Cl1 <= 0 or Cl2 <= 0: return 0.
    phiL = get_angle(Lvec)
    return cmb.flensedEE(l1)*np.sin(2*phiL-2*phi1)/Cl1/Cl2
    
    
def F_GMV_Shear(f, l1vec, l2vec, ell_maxT, ell_maxP):
    l1,phi1,l2,phi2,Lvec,L,prefac = get_basics(l1vec,l2vec)
    phiL = get_angle(Lvec)
    #
    CTT = CC_(l1)[0,0]
    CTE = CC_(l1)[0,1]
    CEE = CC_(l1)[1,1]
    CBB = CC_(l1)[2,2]
    D = CTT*CEE - CTE**2
    #
    FTT = CEE**2*(cmb.flensedTT(l1+1)-cmb.flensedTT(l1))/(np.log(l1+1)-np.log(l1))
    FTT += CTE**2*(cmb.flensedEE(l1+1)-cmb.flensedEE(l1))/(np.log(l1+1)-np.log(l1))
    FTT -= 2*CTE*CEE*(cmb.flensedTE(l1+1)-cmb.flensedTE(l1))/(np.log(l1+1)-np.log(l1))
    FTT *= np.cos(2*phiL-2*phi1) / 2 / D**2
    # 
    FTE = (CTT*CEE+CTE**2)*(2*cmb.flensedTE(l1) + (cmb.flensedTE(l1+1)-cmb.flensedTE(l1))/(np.log(l1+1)-np.log(l1)) )
    FTE -= CTE*CTT*cmb.flensedEE(l1)*(np.log((l1+1)**2*cmb.flensedEE(l1+1))-np.log(l1**2*cmb.flensedEE(l1)))/(np.log(l1+1)-np.log(l1))
    FTE -= CEE*CTE*cmb.flensedTT(l1)*(np.log((l1+1)**2*cmb.flensedTT(l1+1))-np.log(l1**2*cmb.flensedTT(l1)))/(np.log(l1+1)-np.log(l1))
    FTE /= 2*D**2
    #
    FTB = (CEE*cmb.flensedTE(l1)-CTE*cmb.flensedEE(l1)) * L / l1
    FTB *= (np.sin(phiL-phi1) + np.sin(3*phiL-3*phi1)) / 2 / D / CBB
    #
    FEE = CTE**2 * (cmb.flensedTT(l1+1)-cmb.flensedTT(l1))/(np.log(l1+1)-np.log(l1))
    FEE += CTT**2 * (cmb.flensedEE(l1+1)-cmb.flensedEE(l1))/(np.log(l1+1)-np.log(l1))
    FEE -= 2*CTE*CTT*(cmb.flensedTE(l1+1)-cmb.flensedTE(l1))/(np.log(l1+1)-np.log(l1))
    FEE *= np.cos(2*phiL-2*phi1) / 2 / D**2
    #
    FEB = CTT*cmb.flensedEE(l1)-CTE*cmb.flensedTE(l1)
    FEB *= np.sin(2*phiL-2*phi1) / D / CBB
    #
    result = np.array([[FTT, FTE, FTB],
                       [FTE, FEE, FEB],
                       [FTB, FEB, 0. ]])
                       
                       
    result = np.zeros((3,3))
    result[1,2] = FEB_Shear(f,l1vec, l2vec, Cee_, Cbb_, C00_, ell_maxP, ell_maxP)
    
     
    # impose ell_max cuts
    TT = (l1 <= ell_maxT)*(l2 <= ell_maxT)
    PP = (l1 <= ell_maxP)*(l2 <= ell_maxP)
    TP = (l1 <= ell_maxT)*(l2 <= ell_maxP)
    PT = (l1 <= ell_maxP)*(l2 <= ell_maxT)
    ell_cut = np.array([[TT, TP, TP],
                        [PT, PP, PP],
                        [PT, PP, PP]])
    return result * ell_cut
    
    
def F_GMV_Shear_resummed(f, l1vec, l2vec, ell_maxT, ell_maxP):
    l1,phi1,l2,phi2,Lvec,L,prefac = get_basics(l1vec,l2vec)
    phiL = get_angle(Lvec)
    #
    CTT1 = CC_(l1)[0,0] ; CTT2 = CC_(l2)[0,0]
    CTE1 = CC_(l1)[0,1] ; CTE2 = CC_(l2)[0,1]
    CEE1 = CC_(l1)[1,1] ; CEE2 = CC_(l2)[1,1]
    CBB1 = CC_(l1)[2,2] ; CBB2 = CC_(l2)[2,2]
    D1 = CTT1*CEE1 - CTE1**2
    D2 = CTT2*CEE2 - CTE2**2
    #
    FTT = CEE1*CEE2*(cmb.flensedTT(l1+1)-cmb.flensedTT(l1))/(np.log(l1+1)-np.log(l1))
    FTT += CTE1*CTE2*(cmb.flensedEE(l1+1)-cmb.flensedEE(l1))/(np.log(l1+1)-np.log(l1))
    FTT -= (CEE1*CTE2+CTE1*CEE2)*(cmb.flensedTE(l1+1)-cmb.flensedTE(l1))/(np.log(l1+1)-np.log(l1))
    FTT *= np.cos(2*phiL-2*phi1) / 2 / D1*D2
    # 
    FTE = (CEE1*CTT2+CTE1*CTE2)*(2*cmb.flensedTE(l1) + (cmb.flensedTE(l1+1)-cmb.flensedTE(l1))/(np.log(l1+1)-np.log(l1)) )
    FTE -= CTE1*CTT2*cmb.flensedEE(l1)*(np.log((l1+1)**2*cmb.flensedEE(l1+1))-np.log(l1**2*cmb.flensedEE(l1)))/(np.log(l1+1)-np.log(l1))
    FTE -= CEE1*CTE2*cmb.flensedTT(l1)*(np.log((l1+1)**2*cmb.flensedTT(l1+1))-np.log(l1**2*cmb.flensedTT(l1)))/(np.log(l1+1)-np.log(l1))
    FTE /= 2*D1*D2
    # 
    FET = (CEE2*CTT1+CTE2*CTE1)*(2*cmb.flensedTE(l2) + (cmb.flensedTE(l2+1)-cmb.flensedTE(l2))/(np.log(l2+1)-np.log(l2)) )
    FET -= CTE2*CTT1*cmb.flensedEE(l2)*(np.log((l2+1)**2*cmb.flensedEE(l2+1))-np.log(l2**2*cmb.flensedEE(l2)))/(np.log(l2+1)-np.log(l2))
    FET -= CEE2*CTE1*cmb.flensedTT(l2)*(np.log((l2+1)**2*cmb.flensedTT(l2+1))-np.log(l1**2*cmb.flensedTT(l2)))/(np.log(l2+1)-np.log(l2))
    FET /= 2*D1*D2
    #
    FTB = (CEE1*cmb.flensedTE(l1)-CTE1*cmb.flensedEE(l1)) * L / l1
    FTB *= (np.sin(phiL-phi1) + np.sin(3*phiL-3*phi1)) / 2 / D1 / CBB2
    #
    FBT = (CEE2*cmb.flensedTE(l2)-CTE2*cmb.flensedEE(l2)) * L / l2
    FBT *= (np.sin(phiL-phi2) + np.sin(3*phiL-3*phi2)) / 2 / D2 / CBB1
    #
    FEE = CTE1*CTE2 * (cmb.flensedTT(l1+1)-cmb.flensedTT(l1))/(np.log(l1+1)-np.log(l1))
    FEE += CTT1*CTT2 * (cmb.flensedEE(l1+1)-cmb.flensedEE(l1))/(np.log(l1+1)-np.log(l1))
    FEE -= (CTE1*CTT2+CTT1*CTE2)*(cmb.flensedTE(l1+1)-cmb.flensedTE(l1))/(np.log(l1+1)-np.log(l1))
    FEE *= np.cos(2*phiL-2*phi1) / 2 / D1*D2
    #
    FEB = CTT1*cmb.flensedEE(l1)-CTE1*cmb.flensedTE(l1)
    FEB *= np.sin(2*phiL-2*phi1) / D1 / CBB2
    #
    FBE = CTT2*cmb.flensedEE(l2)-CTE2*cmb.flensedTE(l2)
    FBE *= np.sin(2*phiL-2*phi2) / D2 / CBB1
    #
    result = np.array([[FTT, FTE, FTB],
                       [FET, FEE, FEB],
                       [FBT, FBE, 0. ]])
    # impose ell_max cuts
    TT = (l1 <= ell_maxT)*(l2 <= ell_maxT)
    PP = (l1 <= ell_maxP)*(l2 <= ell_maxP)
    TP = (l1 <= ell_maxT)*(l2 <= ell_maxP)
    PT = (l1 <= ell_maxP)*(l2 <= ell_maxT)
    ell_cut = np.array([[TT, TP, TP],
                        [PT, PP, PP],
                        [PT, PP, PP]])
    return result * ell_cut


##### Response function
##### 

def Response(L,f1,f2,ftotX, ftotY,ftotXY,ell_maxX,ell_maxY,F=F_MV):
    Lvec = np.array([L,0])
    def integrand(l,theta):
        lvec = np.array([np.cos(theta),np.sin(theta)]) * l
        result = F(f1,lvec,Lvec-lvec,ftotX,ftotY,ftotXY,ell_maxX,ell_maxY)
        result *= f2(lvec,Lvec-lvec) * l / (2*np.pi)**2
        return np.nan_to_num(result)
    
    ell_max = 2*max(ell_maxX,ell_maxY)
    def MC_integrand(x): return integrand(x[0],x[1])
    integ = vegas.Integrator([[10, ell_max], [0., 2.*np.pi]])
    result = integ(MC_integrand, nitn=nitn, neval=neval)
    return result.mean
    
    
def Response_GMV(L,f,g,ell_maxT,ell_maxP,F=F_GMV):
    Lvec = np.array([L,0])
    def integrand(l,theta):
        lvec = np.array([np.cos(theta),np.sin(theta)]) * l
        g_m = g(lvec,Lvec-lvec)
        F_m = F(f, lvec,Lvec-lvec, ell_maxT, ell_maxP)
        result = np.dot(g_m,F_m.T)
        return np.nan_to_num(np.trace(result) * l / (2*np.pi)**2)
    
    ell_max = 2*max(ell_maxT,ell_maxP)
    def MC_integrand(x): return integrand(x[0],x[1])
    integ = vegas.Integrator([[10, ell_max], [0., 2.*np.pi]])
    integral = integ(MC_integrand, nitn=nitn, neval=neval)
    result = 1/integral
    return result.mean

##### Bias hardened weights
##### 

def F_BH_1(f1,f2,L, ftotX, ftotY, ftotXY, ell_maxX, ell_maxY):
    # weights and norms
    F1 = lambda l1vec,l2vec: F_MV(f1,l1vec, l2vec, ftotX, ftotY, ftotXY, ell_maxX, ell_maxY)
    N1 = Norm(L,f1,ftotX, ftotY,ftotXY,ell_maxX,ell_maxY,F=F_MV)
    F2 = lambda l1vec,l2vec: F_MV(f2,l1vec, l2vec, ftotX, ftotY, ftotXY, ell_maxX, ell_maxY)
    N2 = Norm(L,f2,ftotX, ftotY,ftotXY,ell_maxX,ell_maxY,F=F_MV)
    # responses
    R = Response(L,f1,f2,ftotX, ftotY,ftotXY,ell_maxX,ell_maxY,F=F_MV)
    #
    FV = lambda l1v,l2v: np.array([N1*F1(l1v,l2v) , N2*F2(l1v,l2v)]) # array of normalized weights
    M = np.array([[1    , N1*R],
                  [N2*R , 1   ]])
    Minv = np.linalg.inv(M)
    return lambda l1v,l2v: np.dot(Minv,FV(l1v,l2v))[0]


def F_BH_2(f1,f2,f3,L, ftotX, ftotY, ftotXY, ell_maxX, ell_maxY):
    # weights and norms
    F1 = lambda l1vec,l2vec: F_MV(f1,l1vec, l2vec, ftotX, ftotY, ftotXY, ell_maxX, ell_maxY)
    N1 = Norm(L,f1,ftotX, ftotY,ftotXY,ell_maxX,ell_maxY,F=F_MV)
    F2 = lambda l1vec,l2vec: F_MV(f2,l1vec, l2vec, ftotX, ftotY, ftotXY, ell_maxX, ell_maxY)
    N2 = Norm(L,f2,ftotX, ftotY,ftotXY,ell_maxX,ell_maxY,F=F_MV)
    F3 = lambda l1vec,l2vec: F_MV(f3,l1vec, l2vec, ftotX, ftotY, ftotXY, ell_maxX, ell_maxY)
    N3 = Norm(L,f3,ftotX, ftotY,ftotXY,ell_maxX,ell_maxY,F=F_MV)
    # responses
    R12 = Response(L,f1,f2,ftotX, ftotY,ftotXY,ell_maxX,ell_maxY,F=F_MV)
    R13 = Response(L,f1,f3,ftotX, ftotY,ftotXY,ell_maxX,ell_maxY,F=F_MV)
    R23 = Response(L,f2,f3,ftotX, ftotY,ftotXY,ell_maxX,ell_maxY,F=F_MV)
    #
    FV = lambda l1v,l2v: np.array([N1*F1(l1v,l2v) , N2*F2(l1v,l2v), N3*F3(l1v,l2v)]) # norm weights
    M = np.array([[1      , N1*R12 , N1*R13],
                  [N2*R12 , 1      , N2*R23],
                  [N3*R13 , N3*R23 , 1     ]])
    Minv = np.linalg.inv(M)
    return lambda l1v,l2v: np.dot(Minv,FV(l1v,l2v))[0]
    
    
def F_BH_GMV(f,gT,gE,gB, L, ell_maxT, ell_maxP):
    # weights and norms
    Fk = lambda l1vec,l2vec: F_GMV(f, l1vec, l2vec, ell_maxT, ell_maxP)
    Nk = Norm_GMV(L,f,ell_maxT,ell_maxP)
    GT = lambda l1vec,l2vec: F_GMV(gT, l1vec, l2vec, ell_maxT, ell_maxP)
    NT = Norm_GMV(L,gT,ell_maxT,ell_maxP)
    GE = lambda l1vec,l2vec: F_GMV(gE, l1vec, l2vec, ell_maxT, ell_maxP)
    NE = Norm_GMV(L,gE,ell_maxT,ell_maxP)
    GB = lambda l1vec,l2vec: F_GMV(gB, l1vec, l2vec, ell_maxT, ell_maxP)
    NB = Norm_GMV(L,gB,ell_maxT,ell_maxP)
    # responses
    RkT = Response_GMV(L,f,gT,ell_maxT,ell_maxP)
    RkE = Response_GMV(L,f,gE,ell_maxT,ell_maxP)
    RkB = Response_GMV(L,f,gB,ell_maxT,ell_maxP)
    RTE = Response_GMV(L,gT,gE,ell_maxT,ell_maxP)
    RTB = Response_GMV(L,gT,gB,ell_maxT,ell_maxP)
    REB = Response_GMV(L,gE,gB,ell_maxT,ell_maxP)
    #
    FV = lambda l1v,l2v: np.array([Nk*Fk(l1v,l2v) , NT*GT(l1v,l2v), NE*GE(l1v,l2v), NB*GB(l1v,l2v)]) # norm weights
    M = np.array([[1      , Nk*RkT , Nk*RkE, Nk*RkB],
                  [NT*RkT , 1      , NT*RTE, NT*RTB],
                  [NE*RkE , NE*RTE , 1     , NE*REB],
                  [NB*RkB , NB*RTB , NB*REB, 1     ]])
    Minv = np.linalg.inv(M)
    Minv /= Minv[0,0] * Nk
    
    def result(l1v,l2v):
        ans = np.zeros((3,3))
        for i in  range(3):
            for j in range(3):
                ans[i,j] = np.dot(Minv,FV(l1v,l2v)[:,i,j])[0]
        return ans
    
    return result
    

##### Normalization
##### 

def Norm(L,f,ftotX, ftotY,ftotXY,ell_maxX,ell_maxY,F=F_MV, f2=None, f3=None):
    Lvec = np.array([L,0])
    #
    if F == F_BH_1: 
        FF = F_BH_1(f,f2,L,ftotX,ftotY,ftotXY,ell_maxX,ell_maxY)
    elif F == F_BH_2: 
        FF = F_BH_2(f,f2,f3,L,ftotX,ftotY,ftotXY,ell_maxX,ell_maxY)
    else: 
        FF = lambda l1v,l2v: F(f,l1v,l2v,ftotX,ftotY,ftotXY,ell_maxX,ell_maxY)
    #
    def integrand(l,theta):
        lvec = np.array([np.cos(theta),np.sin(theta)]) * l
        result = FF(lvec,Lvec-lvec)
        result *= f(lvec,Lvec-lvec) * l / (2*np.pi)**2
        return np.nan_to_num(result)
    
    ell_max = 2*max(ell_maxX,ell_maxY)
    def MC_integrand(x): return integrand(x[0],x[1])
    integ = vegas.Integrator([[10, ell_max], [0., 2.*np.pi]])
    integral = integ(MC_integrand, nitn=nitn, neval=neval)
    result = 1/integral
    return result.mean


def Norm_GMV(L,f,ell_maxT,ell_maxP,F=F_GMV):
    Lvec = np.array([L,0])
    def integrand(l,theta):
        lvec = np.array([np.cos(theta),np.sin(theta)]) * l
        f_m = f(lvec,Lvec-lvec)
        F_m = F(f, lvec,Lvec-lvec, ell_maxT, ell_maxP)
        result = np.dot(f_m,F_m.T)
        return np.nan_to_num(np.trace(result) * l / (2*np.pi)**2)
    
    ell_max = 2*max(ell_maxT,ell_maxP)
    def MC_integrand(x): return integrand(x[0],x[1])
    integ = vegas.Integrator([[10, ell_max], [0., 2.*np.pi]])
    integral = integ(MC_integrand, nitn=nitn, neval=neval)
    result = 1/integral
    return result.mean
    
    
def Norm_BH_GMV(L,f,gT,gE,gB,ell_maxT,ell_maxP,FF=None):
    Lvec = np.array([L,0])
    if FF is None: FF = F_BH_GMV(f,gT,gE,gB, L, ell_maxT, ell_maxP)
    def integrand(l,theta):
        lvec = np.array([np.cos(theta),np.sin(theta)]) * l
        f_m = f(lvec,Lvec-lvec)
        F_m = FF(lvec,Lvec-lvec)
        result = np.dot(f_m,F_m.T)
        return np.nan_to_num(np.trace(result) * l / (2*np.pi)**2)
    
    ell_max = 2*max(ell_maxT,ell_maxP)
    def MC_integrand(x): return integrand(x[0],x[1])
    integ = vegas.Integrator([[10, ell_max], [0., 2.*np.pi]])
    integral = integ(MC_integrand, nitn=nitn, neval=neval)
    result = 1/integral
    return result.mean

##### Noise
##### 

def Noise(L,f,ftotX, ftotY,ftotXY,ell_maxX,ell_maxY,F=F_MV,f2=None, f3=None):
    Lvec = np.array([L,0])
    #
    if F == F_BH_1: 
        FF = F_BH_1(f,f2,L,ftotX,ftotY,ftotXY,ell_maxX,ell_maxY)
    elif F == F_BH_2: 
        FF = F_BH_2(f,f2,f3,L,ftotX,ftotY,ftotXY,ell_maxX,ell_maxY)
    else: 
        FF = lambda l1v,l2v: F(f,l1v,l2v,ftotX,ftotY,ftotXY,ell_maxX,ell_maxY)
    #
    def integrand(l,theta):
        lvec = np.array([np.cos(theta),np.sin(theta)]) * l
        Ll = np.sqrt(np.dot(Lvec-lvec,Lvec-lvec))
        F_lLl = FF(lvec,Lvec-lvec)
        F_Lll = FF(Lvec-lvec,lvec)
        A = F_lLl**2 * ftotX(l) * ftotY(Ll)
        B = F_lLl * F_Lll * ftotXY(l) * ftotXY(Ll)
        return np.nan_to_num((A+B) * l / (2*np.pi)**2)
    
    ell_max = 2*max(ell_maxX,ell_maxY)
    def MC_integrand(x): return integrand(x[0],x[1])
    integ = vegas.Integrator([[10, ell_max], [0., 2.*np.pi]])
    integral = integ(MC_integrand, nitn=nitn, neval=neval)
    nonnorm = integral.mean
    norm = Norm(L,f,ftotX, ftotY,ftotXY,ell_maxX,ell_maxY,F=F,f2=f2,f3=f3)
    return nonnorm * norm**2
    
    
def Cross_Noise(L,X,Y,M,N,ell_maxT,ell_maxP,
                F1=F_MV,F2=F_MV,g1=None,g2=None,h1=None,h2=None):
    '''
    X,Y,M,N are either 0,1,2 (T,E,B)
    g1,g2 are the two responses to harden XY against
    h1,h2 are the two responses to harden MN against
    '''
    Lvec = np.array([L,0])
    # cross-spectra needed for integral
    CXM = lambda l: CC_(l)[X,M]
    CYN = lambda l: CC_(l)[Y,N]
    CXN = lambda l: CC_(l)[X,N]
    CYM = lambda l: CC_(l)[Y,M]
    # spectra needed for weights and normalizations
    CXX = lambda l: CC_(l)[X,X]
    CYY = lambda l: CC_(l)[Y,Y]
    CXY = lambda l: CC_(l)[X,Y]
    CMM = lambda l: CC_(l)[M,M]
    CNN = lambda l: CC_(l)[N,N]
    CMN = lambda l: CC_(l)[M,N]
    
    def fff(X,Y):
       if X == 0 and Y == 0: return fTT
       if X == 0 and Y == 1: return fTE
       if X == 0 and Y == 2: return fTB
       if X == 1 and Y == 1: return fEE
       if X == 1 and Y == 2: return fEB
       else: return 'something went wrong'
    def get_ellmax(X,Y):
       if X == 0 and Y == 0: return ell_maxT,ell_maxT
       if X == 0 and Y == 1: return ell_maxT,ell_maxP
       if X == 0 and Y == 2: return ell_maxT,ell_maxP
       if X == 1 and Y == 1: return ell_maxP,ell_maxP
       if X == 1 and Y == 2: return ell_maxP,ell_maxP
       else: return 'something went wrong'
       
    '''
    # XY weights
    ell_maxX,ell_maxY = get_ellmax(X,Y)
    FXY = lambda l1v,l2v: F_MV(fff(X,Y),l1v,l2v,CXX,CYY,CXY,ell_maxX,ell_maxY)
    # MN weights
    ell_maxM,ell_maxN = get_ellmax(M,N)
    FMN = lambda l1v,l2v: F_MV(fff(M,N),l1v,l2v,CMM,CNN,CMN,ell_maxM,ell_maxN)
    '''

    # get ell_cuts
    ell_maxX,ell_maxY = get_ellmax(X,Y)
    ell_maxM,ell_maxN = get_ellmax(M,N)
    
    # XY: decide which bias hardening to do
    if F1 == F_BH_1: 
        FXY = F_BH_1(fff(X,Y),g1,L,CXX,CYY,CXY,ell_maxX,ell_maxY)
    elif F1 == F_BH_2: 
        FXY = F_BH_2(fff(X,Y),g1,g2,L,CXX,CYY,CXY,ell_maxX,ell_maxY)
    else: 
        FXY = lambda l1v,l2v: F1(fff(X,Y),l1v,l2v,CXX,CYY,CXY,ell_maxX,ell_maxY)
    
    # MN: decide which bias hardening to do
    if F2 == F_BH_1: 
        FMN = F_BH_1(fff(M,N),h1,L,CMM,CNN,CMN,ell_maxM,ell_maxN)
    elif F2 == F_BH_2: 
        FMN = F_BH_2(fff(M,N),h1,h2,L,CMM,CNN,CMN,ell_maxM,ell_maxN)
    else: 
        FMN = lambda l1v,l2v: F2(fff(M,N),l1v,l2v,CMM,CNN,CMN,ell_maxM,ell_maxN)
    
        
    def integrand(l,theta):
        lvec = np.array([np.cos(theta),np.sin(theta)]) * l
        Ll = np.sqrt(np.dot(Lvec-lvec,Lvec-lvec))
        FXY_lLl = FXY(lvec,Lvec-lvec)
        FMN_lLl = FMN(lvec,Lvec-lvec)
        FMN_Lll = FMN(Lvec-lvec,lvec)
        result = FXY_lLl * (FMN_lLl * CXM(l) * CYN(Ll) + FMN_Lll * CXN(l) * CYM(Ll) )
        return np.nan_to_num(result * l / (2*np.pi)**2)
        
    ell_max = 2*max(ell_maxT,ell_maxP)
    def MC_integrand(x): return integrand(x[0],x[1])
    integ = vegas.Integrator([[10, ell_max], [0., 2.*np.pi]])
    integral = integ(MC_integrand, nitn=nitn, neval=neval)
    nonnorm = integral.mean
    normXY = Norm(L,fff(X,Y),CXX, CYY, CXY, ell_maxX, ell_maxY, F=F1, f2=g1, f3=g2)
    normMN = Norm(L,fff(M,N),CMM, CNN, CMN, ell_maxM, ell_maxN, F=F2, f2=h1, f3=h2)
    return nonnorm * normXY * normMN


def Noise_GMV(L,f,ell_maxT,ell_maxP,F=F_GMV):
    Lvec = np.array([L,0])
    def integrand(l,theta):
        lvec = np.array([np.cos(theta),np.sin(theta)]) * l
        Ll = np.sqrt(np.dot(Lvec-lvec,Lvec-lvec))
        C1 = CC_(l)
        C2 = CC_(Ll)
        F1 = F(f, lvec, Lvec-lvec, ell_maxT, ell_maxP)
        F2 = F(f, Lvec-lvec, lvec, ell_maxT, ell_maxP)
        result = np.dot(np.dot(np.dot(C1,F1),C2),F1.T)
        result += np.dot(np.dot(np.dot(C1,F1),C2),F2)
        return np.nan_to_num(np.trace(result) * l / (2*np.pi)**2)
    
    ell_max = 2*max(ell_maxT,ell_maxP)
    def MC_integrand(x): return integrand(x[0],x[1])
    integ = vegas.Integrator([[10, ell_max], [0., 2.*np.pi]])
    integral = integ(MC_integrand, nitn=nitn, neval=neval)
    nonnorm = integral.mean
    norm = Norm_GMV(L,f,ell_maxT,ell_maxP,F=F)
    return nonnorm * norm**2
    
    
def Noise_BH_GMV(L,f,gT,gE,gB,ell_maxT,ell_maxP):
    Lvec = np.array([L,0])
    FF = F_BH_GMV(f,gT,gE,gB, L, ell_maxT, ell_maxP)
    def integrand(l,theta):
        lvec = np.array([np.cos(theta),np.sin(theta)]) * l
        Ll = np.sqrt(np.dot(Lvec-lvec,Lvec-lvec))
        C1 = CC_(l)
        C2 = CC_(Ll)
        F1 = FF(lvec, Lvec-lvec)
        result = np.dot(np.dot(np.dot(C1,F1),C2),F1.T)
        return np.nan_to_num(2 * np.trace(result) * l / (2*np.pi)**2)
       
    ell_max = 2*max(ell_maxT,ell_maxP)
    def MC_integrand(x): return integrand(x[0],x[1])
    integ = vegas.Integrator([[10, ell_max], [0., 2.*np.pi]])
    integral = integ(MC_integrand, nitn=nitn, neval=neval)
    nonnorm = integral.mean
    norm = Norm_BH_GMV(L,f,gT,gE,gB,ell_maxT,ell_maxP,FF=FF)
    return nonnorm * norm**2

########################################################################
LL = np.logspace(np.log10(lmin),np.log10(6000),20)


def save_noise_levels(i):
    #
    # MV QE on individual paris
    #
    if i == 1:
        QE_TT = np.array([Noise(l,fTT,Ctt_,Ctt_,Ctt_,lmaxT,lmaxT) for l in LL])
        data = np.array([LL,QE_TT])
        np.savetxt('QE_TT.txt',data)
    if i == 2:
        QE_TE = np.array([Noise(l,fTE,Ctt_,Cee_,Cte_,lmaxT,lmaxP) for l in LL])
        data = np.array([LL,QE_TE])
        np.savetxt('QE_TE.txt',data)
    if i == 3:
        QE_TB = np.array([Noise(l,fTB,Ctt_,Cbb_,C00_,lmaxT,lmaxP) for l in LL])
        data = np.array([LL,QE_TB])
        np.savetxt('QE_TB.txt',data)
    if i == 4:
        QE_EE = np.array([Noise(l,fEE,Cee_,Cee_,Cee_,lmaxP,lmaxP) for l in LL])
        data = np.array([LL,QE_EE])
        np.savetxt('QE_EE.txt',data)
    if i == 5:
        QE_EB = np.array([Noise(l,fEB,Cee_,Cbb_,C00_,lmaxP,lmaxP) for l in LL])
        data = np.array([LL,QE_EB])
        np.savetxt('QE_EB.txt',data)
    #
    # Symmetrized estimator
    #
    if i == 6:
        QE_sym_TE = np.array([Noise(l,fTE,Ctt_,Cee_,Cte_,lmaxT,lmaxP,F=F_sym_MV) for l in LL])
        data = np.array([LL,QE_sym_TE])
        np.savetxt('QE_sym_TE.txt',data)
    #
    # Bias hardened individual pairs
    #
    if i == 7:
        BH_TT = np.array([Noise(l,fTT,Ctt_,Ctt_,Ctt_,lmaxT,lmaxT,F=F_BH_1,f2=gTTT) for l in LL])
        data = np.array([LL,BH_TT])
        np.savetxt('BH_TT.txt',data)
    if i == 8:
        BH_TE = np.array([Noise(l,fTE,Ctt_,Cee_,Cte_,lmaxT,lmaxP,F=F_BH_2,f2=gTEE,f3=gTEB) for l in LL])
        data = np.array([LL,BH_TE])
        np.savetxt('BH_TE.txt',data)
    if i == 9:
        BH_TB = np.array([Noise(l,fTB,Ctt_,Cbb_,C00_,lmaxT,lmaxP,F=F_BH_2,f2=gTBE,f3=gTBB) for l in LL])
        data = np.array([LL,BH_TB])
        np.savetxt('BH_TB.txt',data)
    if i == 10:
        BH_EE = np.array([Noise(l,fEE,Cee_,Cee_,Cee_,lmaxP,lmaxP,F=F_BH_1,f2=gEET) for l in LL])
        data = np.array([LL,BH_EE])
        np.savetxt('BH_EE.txt',data)
    if i == 11:
        BH_EB = np.array([Noise(l,fEB,Cee_,Cbb_,C00_,lmaxP,lmaxP,F=F_BH_1,f2=gEBT) for l in LL])
        data = np.array([LL,BH_EB])
        np.savetxt('BH_EB.txt',data)
    #
    # Shear individual pairs
    #
    if i == 12:
        Shear_TT = np.array([Noise(l,fTT,Ctt_,Ctt_,Ctt_,lmaxT,lmaxT,F=FTT_Shear) for l in LL])
        data = np.array([LL,Shear_TT])
        np.savetxt('Shear_TT.txt',data)
    if i == 13:
        Shear_TE = np.array([Noise(l,fTE,Ctt_,Cee_,Cte_,lmaxT,lmaxP,F=FTE_Shear) for l in LL])
        data = np.array([LL,Shear_TE])
        np.savetxt('Shear_TE.txt',data)
    if i == 14:
        Shear_EE = np.array([Noise(l,fEE,Cee_,Cee_,Cee_,lmaxP,lmaxP,F=FEE_Shear) for l in LL])
        data = np.array([LL,Shear_EE])
        np.savetxt('Shear_EE.txt',data)
    if i == 15:
        Shear_EB = np.array([Noise(l,fEB,Cee_,Cbb_,C00_,lmaxP,lmaxP,F=FEB_Shear) for l in LL])
        data = np.array([LL,Shear_EB])
        np.savetxt('Shear_EB.txt',data)
    if i == 16:
        Shear_TB = np.array([Noise(l,fTB,Ctt_,Cbb_,C00_,lmaxT,lmaxP,F=FTB_Shear) for l in LL])
        data = np.array([LL,Shear_TB])
        np.savetxt('Shear_TB.txt',data)
    #
    # global minimum variance
    #
    if i == 17:
        GMV = np.array([Norm_GMV(l,f_matrix,lmaxT,lmaxP) for l in LL])
        data = np.array([LL,GMV])
        np.savetxt('GMV.txt',data)
    #
    # Relevant cross power
    #
    if i == 18:
        TTTE = np.array([Cross_Noise(l,0,0,0,1,lmaxT,lmaxP) for l in LL])
        data = np.array([LL,TTTE])
        np.savetxt('TTTE.txt',data)
    if i == 19:
        TTEE = np.array([Cross_Noise(l,0,0,1,1,lmaxT,lmaxP) for l in LL])
        data = np.array([LL,TTEE])
        np.savetxt('TTEE.txt',data)
    if i == 20:
        TEEE = np.array([Cross_Noise(l,0,1,1,1,lmaxT,lmaxP) for l in LL])
        data = np.array([LL,TEEE])
        np.savetxt('TEEE.txt',data)
    if i == 21:
        TBEB = np.array([Cross_Noise(l,0,2,1,2,lmaxT,lmaxP) for l in LL])
        data = np.array([LL,TBEB])
        np.savetxt('TBEB.txt',data)
    #
    # bias hardening the GMV
    #
    if i == 22:
        BH_GMV = np.array([Norm_BH_GMV(l,f_matrix,gT_matrix,gE_matrix,gB_matrix,lmaxT,lmaxP) for l in LL])
        data = np.array([LL,BH_GMV])
        np.savetxt('BH_GMV.txt',data)
        
    #
    # Relevant cross power (bias hardened)
    #
    if i == 23:
        TTTE = np.array([Cross_Noise(l,0,0,0,1,lmaxT,lmaxP,F1=F_BH_1,F2=F_BH_2,g1=gTTT,g2=None,h1=gTEE,h2=gTEB) for l in LL])
        data = np.array([LL,TTTE])
        np.savetxt('TTTE_BH.txt',data)
    if i == 24:
        TTEE = np.array([Cross_Noise(l,0,0,1,1,lmaxT,lmaxP,F1=F_BH_1,F2=F_BH_1,g1=gTTT,g2=None,h1=gEET,h2=None) for l in LL])
        data = np.array([LL,TTEE])
        np.savetxt('TTEE_BH.txt',data)
    if i == 25:
        TEEE = np.array([Cross_Noise(l,0,1,1,1,lmaxT,lmaxP,F1=F_BH_2,F2=F_BH_1,g1=gTEE,g2=gTEB,h1=gEET,h2=None) for l in LL])
        data = np.array([LL,TEEE])
        np.savetxt('TEEE_BH.txt',data)
    if i == 26:
        TBEB = np.array([Cross_Noise(l,0,2,1,2,lmaxT,lmaxP,F1=F_BH_2,F2=F_BH_1,g1=gTBE,g2=gTBB,h1=gEBT,h2=None)  for l in LL])
        data = np.array([LL,TBEB])
        np.savetxt('TBEB_BH.txt',data)
    #
    # Sheared global minimum variance
    #
    if i == 27:
        GMV_Shear = np.array([Noise_GMV(l,f_matrix,lmaxT,lmaxP,F=F_GMV_Shear) for l in LL])
        data = np.array([LL,GMV_Shear])
        np.savetxt('GMV_Shear.txt',data)
    return ''
    
    
    
##############################################
## Compute the noise curves in parallel
'''
import time
ti = time.time()

if __name__ == '__main__':
    pool = mp.Pool(mp.cpu_count())
    xxx = [18,19,20,21] 
    results = pool.map(save_noise_levels, xxx)

tf = time.time()
print((tf-ti)/60,'minutes')
'''
