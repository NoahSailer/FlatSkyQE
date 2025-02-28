from __future__ import division
from builtins import map
from builtins import str
from builtins import range
from builtins import object
from past.utils import old_div
from headers import *

###############################################################################
# Class containing all templates for CMB components at various frequencies
# from Dunkley et al 2013.
# All the maps are debeamed, ie a shot noise component is ell-independent,
# and the detector noise grows exponentially with the beam.
# Signal and noise are assumed to be in muK^2*steradian.

class CMB(object):
   
   def __str__(self):
      return self.name
   
   def __init__(self):
      # required input:
      # self.name (string)
      # self.nu1 frequency 1 in Hertz
      # self.nu2 frequency 2 in Hertz
      # self.fwhm beam fwhm in rad
      # self.sensitivity sensitivity in muK.rad
      
      # constants
      self.c = 3.e8  # m/s
      self.h = 6.63e-34 # SI
      self.kB = 1.38e-23   # SI
      self.Tcmb = 2.726 # K
      self.Jansky = 1.e-26 # W/m^2/Hz
      
      # convert from Dl to pixel noise Cl
      #self.fdl_to_cl = lambda l: self.fbeam(l)**2 / ( l*(l+1.)/(2.*np.pi) )
      self.fdl_to_cl = lambda l: divide(1., l*(l+1.) / (2.*np.pi))
      
      ###########################################
      # unlensed primary T, E, B

      # unlensed CMB
      data = np.genfromtxt("./input/universe_Planck15/camb/lenspotentialCls.dat")
      self.funlensedTT_template = UnivariateSpline(data[:,0], data[:,1],k=1,s=0)
      lmin_unlensedCMB = data[0,0]
      lmax_unlensedCMB = data[-1,0]
      self.funlensedTT = lambda l: (l>=lmin_unlensedCMB and l<=lmax_unlensedCMB) * self.funlensedTT_template(l) * self.fdl_to_cl(l)
      
      # unlensed EE
      data = np.genfromtxt("./input/universe_Planck15/camb/lenspotentialCls.dat")
      self.funlensedEE_template = UnivariateSpline(data[:,0], data[:,2],k=1,s=0)
      lmin_unlensedEE = data[0,0]
      lmax_unlensedEE = data[-1,0]
      self.funlensedEE = lambda l: (l>=lmin_unlensedEE and l<=lmax_unlensedEE) * self.funlensedEE_template(l) * self.fdl_to_cl(l)
      
      # unlensed BB
      data = np.genfromtxt("./input/universe_Planck15/camb/lenspotentialCls.dat")
      self.funlensedBB_template = UnivariateSpline(data[:,0], data[:,3],k=1,s=0)
      lmin_unlensedBB = data[0,0]
      lmax_unlensedBB = data[-1,0]
      self.funlensedBB = lambda l: (l>=lmin_unlensedBB and l<=lmax_unlensedBB) * self.funlensedBB_template(l) * self.fdl_to_cl(l)
      
      # unlensed TE
      data = np.genfromtxt("./input/universe_Planck15/camb/lenspotentialCls.dat")
      self.funlensedTE_template = UnivariateSpline(data[:,0], data[:,4],k=1,s=0)
      lmin_unlensedTE = data[0,0]
      lmax_unlensedTE = data[-1,0]
      self.funlensedTE = lambda l: (l>=lmin_unlensedTE and l<=lmax_unlensedTE) * self.funlensedTE_template(l) * self.fdl_to_cl(l)
      
      
      ###########################################
      # lensed primary T, E, B
      
      # lensed CMB
      data = np.genfromtxt("./input/universe_Planck15/camb/lensedCls.dat")
      self.flensedTT_template = UnivariateSpline(data[:,0], data[:,1],k=1,s=0)
      lmin_lensedCMB = data[0,0]
      lmax_lensedCMB = data[-1,0]
      self.flensedTT = lambda l: (l>=lmin_lensedCMB and l<=lmax_lensedCMB) * self.flensedTT_template(l) * self.fdl_to_cl(l)

      # lensed EE
      data = np.genfromtxt("./input/universe_Planck15/camb/lensedCls.dat")
      self.flensedEE_template = UnivariateSpline(data[:,0], data[:,2],k=1,s=0)
      lmin_lensedEE = data[0,0]
      lmax_lensedEE = data[-1,0]
      self.flensedEE = lambda l: (l>=lmin_lensedEE and l<=lmax_lensedEE) * self.flensedEE_template(l) * self.fdl_to_cl(l)

      # lensed BB
      data = np.genfromtxt("./input/universe_Planck15/camb/lensedCls.dat")
      self.flensedBB_template = UnivariateSpline(data[:,0], data[:,3],k=1,s=0)
      lmin_lensedBB = data[0,0]
      lmax_lensedBB = data[-1,0]
      self.flensedBB = lambda l: (l>=lmin_lensedBB and l<=lmax_lensedBB) * self.flensedBB_template(l) * self.fdl_to_cl(l)

      # lensed TE
      data = np.genfromtxt("./input/universe_Planck15/camb/lensedCls.dat")
      self.flensedTE_template = UnivariateSpline(data[:,0], data[:,4],k=1,s=0)
      lmin_lensedTE = data[0,0]
      lmax_lensedTE = data[-1,0]
      self.flensedTE = lambda l: (l>=lmin_lensedTE and l<=lmax_lensedTE) * self.flensedTE_template(l) * self.fdl_to_cl(l)

      ###########################################
      # total primary T, E, B, w/o foregrounds: lensed + noise

      self.ftotalTT = lambda l: self.flensedTT(l) + self.fdetectorNoise(l) #+ self.fatmosphericNoiseTT(l)
      self.ftotalEE = lambda l: self.flensedEE(l) + 2.*self.fdetectorNoise(l) #+ self.fatmosphericNoisePP(l)
      self.ftotalBB = lambda l: self.flensedBB(l) + 2.*self.fdetectorNoise(l) #+ self.fatmosphericNoisePP(l)
      #
      self.ftotalTE = lambda l: self.flensedTE(l)

      ###########################################

      # tSZ: Dunkley et al 2013
      data = np.genfromtxt("./input/cmb/digitizing_SZ_template/tSZ.txt")
      ftSZ_template = UnivariateSpline(data[:,0], data[:,1],k=1,s=0)
      a_tSZ = 4.0
      lmin_tSZ = data[0,0]
      lmax_tSZ = data[-1,0]
      self.ftSZ = lambda l: old_div((l>=lmin_tSZ and l<=lmax_tSZ) * a_tSZ * self.freqDpdceTSZTemp(self.nu1)*self.freqDpdceTSZTemp(self.nu2),self.freqDpdceTSZTemp(150.e9)**2) * ftSZ_template(l) * self.fdl_to_cl(l)

      # kSZ: Dunkley et al 2013
      data = np.genfromtxt("./input/cmb/digitizing_SZ_template/kSZ.txt")
      fkSZ_template = UnivariateSpline(data[:,0], data[:,1],k=1,s=0)
      a_kSZ = 1.5  # 1.5 predicted by Battaglia et al 2010. Upper limit from Dunkley+13 is 5.
      lmin_kSZ = data[0,0]
      lmax_kSZ = data[-1,0]
      self.fkSZ = lambda l: (l>=lmin_kSZ and l<=lmax_kSZ) * a_kSZ * fkSZ_template(l) * self.fdl_to_cl(l)
   
      # tSZ x CMB: Dunkley et al 2013
      xi = 0.2 # upper limit at 95% confidence
      a_tSZ = 4.0
      a_CIBC = 5.7
      betaC = 2.1
      Td = 9.7
      # watch for the minus sign
      data = np.genfromtxt ("./input/cmb/digitizing_tSZCIB_template/minus_tSZ_CIB.txt")
      ftSZCIB_template = UnivariateSpline(data[:,0], data[:,1],k=1,s=0)
      lmin_tSZ_CIB = data[0,0]
      lmax_tSZ_CIB = data[-1,0]
      self.ftSZ_CIB = lambda l: old_div((l>=lmin_tSZ_CIB and l<=lmax_tSZ_CIB) * (-2.)*xi*np.sqrt(a_tSZ*a_CIBC)* self.fprime(self.nu1, self.nu2, betaC, Td),self.fprime(150.e9, 150.e9, betaC, Td)) * ftSZCIB_template(l) * self.fdl_to_cl(l)


   ###############################################################################
   # beam and detector noise

   def fbeamTheta(self, theta):
      sigma_beam = old_div(self.fwhm, np.sqrt(8.*np.log(2.)))
      return old_div(np.exp(old_div(-0.5*theta**2,sigma_beam**2)), (2.*np.pi*sigma_beam**2))
      
   def fbeam(self, l):
      sigma_beam = old_div(self.fwhm, np.sqrt(8.*np.log(2.)))
      return np.exp(-0.5*l**2 * sigma_beam**2)
   
   def fdetectorNoise(self, l):
      return old_div(self.sensitivity**2, self.fbeam(l)**2)


   ###############################################################################
   # functions for frequency dependence
   
   # blackbody function
   # nu in Hz
   # output in W / Hz / m^2 / sr
   def blackbody(self, nu, T):
      x = old_div(self.h*nu,(self.kB*T))
      result = old_div(2.*self.h*nu**3,self.c**2)
      result /= np.exp(x) - 1.
      return result
   
   # dlnBlackbody/dlnT
   # output in SI
   def dlnBdlnT(self, nu, T):
      x = old_div(self.h*nu,(self.kB*T))
      return old_div(x * np.exp(x), (np.exp(x) - 1.))
   
   # d(blackbody)/dT at T
   # output in SI
   def dBdT(self, nu, T):
      x = old_div(self.h*nu,(self.kB*T))
      result = 2.*self.h**2*nu**4
      result /= self.kB*T**2*self.c**2
      result *= old_div(np.exp(x), (np.exp(x) - 1.)**2)
      return result
   
   # dT/d(blackbody) at T
   # output in SI
   def g(self, nu, T):
      return 1./self.dBdT(nu, T)
   
   # blackbody modified with power law
   # expressed in temperature units
   # relevant for CIB
   def mu(self, nu, beta, T):
      return nu**beta * self.blackbody(nu, T) * self.g(nu, self.Tcmb)

   # frequency dependence for tSZ
   # dT/T = freqDpdceTSZTemp * y
   def freqDpdceTSZTemp(self, nu):
      x = old_div(self.h*nu,(self.kB*self.Tcmb))
      return old_div(x*(np.exp(x)+1.),(np.exp(x)-1.)) -4.
   
   # frequency dependence for tSZ
   # dI/I = freqDpdceTSZIntensity * y
   def freqDpdceTSZIntensity(self, nu):
      return self.freqDpdceTSZTemp(nu) * self.dlnBdlnT(nu, self.Tcmb)
   

   def fprime(self, nu1, nu2, beta, T):
      return self.freqDpdceTSZTemp(nu1) * self.mu(nu2, beta, T) + self.freqDpdceTSZTemp(nu2) * self.mu(nu1, beta, T)
   
   
   def plotFreqDpdce(self):
      #Nu = np.linspace(0., 800., 301)*1.e9   # in Hz
      Nu = np.logspace(np.log10(0.1), np.log10(1.e4), 501, 10.)*1.e9   # in Hz
      f = np.array(list(map(self.freqDpdceTSZTemp, Nu))) # freq dpdce of dT/T for tSZ
      
      MJy = 1.e-26 * 1.e6  # mega Jansky in SI

      # CMB mean and fluctuations specific intensity, in SI
      fluct = 110.e-6/2.726 # primary fluctuations, ~110muK
      f = lambda nu: self.blackbody(nu, self.Tcmb)
      blackbody = np.array(list(map(f, Nu)))
      CMBfluct = fluct * blackbody
      # tSZ
      y = 0.1e-6/2.726   # tSZ amplitude, ~0.1muK for 1.e13Msun halo
      freqDpdceTSZIntensity = np.array(list(map(self.freqDpdceTSZIntensity, Nu)))
      TSZ = freqDpdceTSZIntensity * blackbody * y
      # kSZ
      tauvc = 0.1e-6/2.726  # kSZ amplitude, ~0.1muK for 1e13Msun halo
      KSZ = blackbody * tauvc
      # CIB: dust temperature and power law index
      cibAmplitude = 1. # arbitrary number
      Td = 9.7
      betaP = 2.1
      cibFreqDpdceTemperature = np.array([self.mu(nu, betaP, Td) for nu in Nu])
      cibFreqDpdceTemperature /= self.mu(150.e9, betaP, Td)
      f = lambda nu: self.dlnBdlnT(nu, self.Tcmb)
      dlnBdlnT = np.array(list(map(f, Nu)))
      CIB = cibFreqDpdceTemperature * dlnBdlnT * blackbody
      
      
      # Plot CMB vs tSZ
      # in intensity, in MJy
      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      ax.plot(old_div(Nu,1.e9), old_div(blackbody, MJy), 'k', lw=2, label=r'CMB')
      ax.plot(old_div(Nu,1.e9), old_div(TSZ * 1.e6, MJy), 'b', lw=2, label=r'tSZ $\times \sim 10^6$')
      ax.axhline(0., color='k')
      #
      ax.legend(loc=1)
      #      ax.set_ylim(1.e-7, 1.e3)
      ax.set_xlabel(r'frequency $\nu$ [GHz]')
      ax.set_ylabel(r'specific intensity $I_\nu$ [MJy]')
      #
      #fig.savefig("./figures/cmb/tsz_freq_dpdce.pdf", bbox_inches='tight')
      

      # Plot CMB, tSZ, kSZ, CIB, with reasonable amplitudes
      # in intensity, in MJy
      fig=plt.figure(1)
      ax=fig.add_subplot(111)
      #
      ax.plot(old_div(Nu,1.e9), old_div(blackbody, MJy), 'k', lw=2, label=r'mean CMB $\sim 2.726$K')
      ax.plot(old_div(Nu,1.e9), old_div(CMBfluct, MJy), 'gray', lw=2, label=r'CMB fluctuations $\sim 110\mu$K')
      ax.plot(old_div(Nu,1.e9), old_div(TSZ, MJy), 'b', lw=2, label=r'tSZ $\sim 0.1\mu$K')
      ax.plot(old_div(Nu,1.e9), old_div(- TSZ, MJy), 'b--', lw=2)
      ax.plot(old_div(Nu,1.e9), old_div(KSZ, MJy), 'r', lw=2, label=r'kSZ $\sim 0.1\mu$K')
      ax.plot(old_div(Nu,1.e9), old_div(CIB, MJy), 'g', lw=2, label=r'CIB, arbitrary amplitude')
      #ax.axhline(0., color='k')
      #
      ax.legend(loc=2)
      ax.set_yscale('log', nonposy='mask')
      ax.set_xscale('log')
      ax.set_ylim(1.e-10, 1.e10)
      ax.set_xlabel(r'frequency $\nu$ [GHz]')
      ax.set_ylabel(r'specific intensity $I_\nu$ [MJy]')
      #
      #fig.savefig("./figures/cmb/freq_dpdces_loglog.pdf", bbox_inches='tight')

      plt.show()


   ###############################################################################
   # CIB Poisson and clustered

   def fCIBPoisson(self, l, nu1=None, nu2=None):
      a_CIBP = 7.0
      Td = 9.7
      betaP = 2.1
      if nu1 is None:
         nu1 = self.nu1
      if nu2 is None:
         nu2 = self.nu2
      return old_div(a_CIBP * (l/3000.)**2 * self.mu(nu1, betaP, Td)*self.mu(nu2, betaP, Td),self.mu(150.e9, betaP, Td)**2) * self.fdl_to_cl(l)

   def fCIBClustered(self, l, nu1=None, nu2=None):
      a_CIBC = 5.7
      n = 1.2
      Td = 9.7
      betaC = 2.1
      if nu1 is None:
         nu1 = self.nu1
      if nu2 is None:
         nu2 = self.nu2
      return old_div(a_CIBC * (l/3000.)**(2-n) * self.mu(nu1, betaC, Td)*self.mu(nu2, betaC, Td),self.mu(150.e9, betaC, Td)**2) * self.fdl_to_cl(l)
   
   def fCIB(self, l, nu1=None, nu2=None):
      return self.fCIBPoisson(l, nu1, nu2) + self.fCIBClustered(l, nu1, nu2)

   ###############################################################################
   # radio point sources, Poisson only

   def fradioPoisson(self, l):
      alpha_s = -0.5
      a_s = 3.2
      return old_div(a_s * (l/3000.)**2 * (old_div(self.nu1*self.nu2,150.e9**2))**alpha_s * self.g(self.nu1, self.Tcmb)*self.g(self.nu2, self.Tcmb),self.g(150.e9, self.Tcmb)**2) * self.fdl_to_cl(l)

   ###############################################################################
   # galactic dust

   def fgalacticDust(self, l):
      beta_g = 3.8
      n_g = -0.7
      a_ge = 0.9
      a_gs = 0.7  # 95% confidence limit
      return old_div(a_gs * (l/3000.)**2 * (old_div(self.nu1*self.nu2,150.e9**2))**beta_g * self.g(self.nu1, self.Tcmb)*self.g(self.nu2, self.Tcmb),self.g(150.e9, self.Tcmb)**2) * self.fdl_to_cl(l)
   
   ###############################################################################
   # atmospheric noise in temperature and polarization
   # only implemented for 150GHz
   # from Matthew Hasselfield's model for Simons observatory
   # getAtmosphere function from Mathew Madhavacheril
   
   def getAtmosphere(self):
      '''Get TT-lknee, TT-alpha, PP-lknee, PP-alpha
      '''
      # best fits from M.Hasselfield
      size = np.array([0.5,5.,7.]) # telescope size in meters
      ttalpha = -4.7
      ppalpha = np.array([-2.6,-3.8,-3.9])
      ttlknee = np.array([350.,3400.,4900.])
      pplknee = np.array([60,330,460])

      # convert telescope size to beam
      cspeed = 299792458.  # m/s
      wavelength = old_div(cspeed,self.nu1)  # m
      resin = old_div(1.22*wavelength,size)  # beam fwhm in rad
      
      # interpolate Matt's fits
      ttlkneeFunc = interp1d(resin,ttlknee,fill_value="extrapolate",kind="linear")
      ttalphaFunc = lambda x: ttalpha
      pplkneeFunc = interp1d(resin,pplknee,fill_value="extrapolate",kind="linear")
      ppalphaFunc = interp1d(resin,ppalpha,fill_value="extrapolate",kind="linear")
      
      b = self.fwhm  # beam fwhm in rad
      return ttlkneeFunc(b),ttalphaFunc(b),pplkneeFunc(b),ppalphaFunc(b)
   
   
   def fatmosphericNoiseTT(self, l):
      lKnee, alpha, _, _ = self.getAtmosphere()
      result = (old_div(lKnee,l))**(-alpha)
      result *= self.fdetectorNoise(l)
      return result

   def fatmosphericNoisePP(self, l):
      _, _, lKnee, alpha = self.getAtmosphere()
      result = (old_div(lKnee,l))**(-alpha)
      result *= self.fdetectorNoise(l)
      result *= 2.   # noise is larger in polarization
      return result
   

   ###############################################################################
   
   # total
   def ftotal(self, l):
      result = self.flensedTT(l)
      result += self.fCIBPoisson(l)
      result += self.fCIBClustered(l)
      result += self.ftSZ(l)
      result += self.fkSZ(l)
      result += self.ftSZ_CIB(l)
      result += self.fradioPoisson(l)
      result += self.fgalacticDust(l)
      result += self.fdetectorNoise(l)
      return result

   
   ###############################################################################


   def testInterpCMB(self):
      data = np.genfromtxt("./input/universe_FerraroHensley14/lensedCls.dat")
      L = np.logspace(np.log10(1.), np.log10(1.e4), 1.e4, 10.)
      Interp = np.array(list(map(self.fCMB_template, L)))
   
      fig=plt.figure(0)
      ax=plt.subplot(111)
      #
      ax.semilogx(L, abs(Interp), 'g')
      ax.semilogx(data[:,0], data[:,1], 'b.')
   
      plt.show()


   # compute the variance of the temperature at a given point
   # assuming infinitely small beam (no beam)
   # in muK
   def fsigmaNoBeam(self):
      
      # lensed CMB
      f = lambda l: old_div(old_div(self.flensedTT(l), self.fdl_to_cl(l)), l)
      result, error = integrate.quad(f, 1., 1.e4, epsabs=0., epsrel=1.e-5)
      result = np.sqrt(result)
      error = np.sqrt(error)
      #print "- temperature fluctuations due to CMB:", result, "muK"
      ##print "relative error on integral is", error/result

      # detector noise would diverge, because it is a constant divided by the beam**2

      # CIB Poisson and clustered
      f = lambda l: old_div(old_div(( self.fCIBPoisson(l) + self.fCIBClustered(l) ), self.fdl_to_cl(l)), l)
      result, error = integrate.quad(f, 1., 1.e4, epsabs=0., epsrel=1.e-5)
      result = np.sqrt(result)
      error = np.sqrt(error)
      #print "- temperature fluctuations due to CIB:", result, "muK"
      ##print "relative error on integral is", error/result
      
      # tSZ
      f = lambda l: old_div(old_div(self.ftSZ(l), self.fdl_to_cl(l)), l)
      result, error = integrate.quad(f, 1., 1.e4, epsabs=0., epsrel=1.e-5)
      result = np.sqrt(result)
      error = np.sqrt(error)
      #print "- temperature fluctuations due to tSZ:", result, "muK"
      ##print "relative error on integral is", error/result
      
      # kSZ
      f = lambda l: old_div(old_div(self.fkSZ(l), self.fdl_to_cl(l)), l)
      result, error = integrate.quad(f, 1., 1.e4, epsabs=0., epsrel=1.e-5)
      result = np.sqrt(result)
      error = np.sqrt(error)
      #print "- temperature fluctuations due to kSZ:", result, "muK"
      ##print "relative error on integral is", error/result
      
      return


   # compute the variance of the temperature at a given point
   # in the observed map
   # i.e. taking into account the beam
   # in muK
   def fsigmaWithBeam(self):
      
      # lensed CMB
      f = lambda l: old_div(self.flensedTT(l) * (l+1.),(2.*np.pi))
      result, error = integrate.quad(f, 1., 1.e4, epsabs=0., epsrel=1.e-5)
      result = np.sqrt(result)
      error = np.sqrt(error)
      #print "- temperature fluctuations due to CMB:", result, "muK"
      ##print "relative error on integral is", error/result
      
      # detector noise
      f = lambda l: old_div(self.fdetectorNoise(l) * (l+1.),(2.*np.pi))
      result, error = integrate.quad(f, 1., 1.e4, epsabs=0., epsrel=1.e-5)
      result = np.sqrt(result)
      error = np.sqrt(error)
      #print "- temperature fluctuations due to detector noise:", result, "muK"
      ##print "relative error on integral is", error/result
      
      # CIB Poisson and clustered
      f = lambda l: old_div(( self.fCIBPoisson(l) + self.fCIBClustered(l) ) * (l+1.),(2.*np.pi))
      result, error = integrate.quad(f, 1., 1.e4, epsabs=0., epsrel=1.e-5)
      result = np.sqrt(result)
      error = np.sqrt(error)
      #print "- temperature fluctuations due to CIB:", result, "muK"
      ##print "relative error on integral is", error/result
      
      # tSZ
      f = lambda l: old_div(self.ftSZ(l) * (l+1.),(2.*np.pi))
      result, error = integrate.quad(f, 1., 1.e4, epsabs=0., epsrel=1.e-5)
      result = np.sqrt(result)
      error = np.sqrt(error)
      #print "- temperature fluctuations due to tSZ:", result, "muK"
      ##print "relative error on integral is", error/result
      
      # kSZ
      f = lambda l: old_div(self.fkSZ(l) * (l+1.),(2.*np.pi))
      result, error = integrate.quad(f, 1., 1.e4, epsabs=0., epsrel=1.e-5)
      result = np.sqrt(result)
      error = np.sqrt(error)
      #print "- temperature fluctuations due to kSZ:", result, "muK"
      ##print "relative error on integral is", error/result
      
      
   # outputs the uncertainty on amplitude of profile
   # given the total power in the map
   # fprofile: isotropic profile (before beam convolution)
   # if none, use the beam as profile (ie point source)
   # If temperature map in muK, then output in muK*sr
   # If temperature map in Jy/sr, then output in Jy
   def fsigmaMatchedFilter(self, fprofile=None, ftotalTT=None):
      if ftotalTT is None:
         ftotalTT = self.ftotalTT
      if fprofile is None:
         f = lambda l: old_div(old_div(l,(2.*np.pi)), ftotalTT(l))
      else:
         f = lambda l: old_div(old_div(l,(2.*np.pi)) * fprofile(l), ftotalTT(l))
      result = integrate.quad(f, self.lMin, self.lMaxT, epsabs=0., epsrel=1.e-3)[0]
      result = 1./np.sqrt(result)
      return result


   def plotCl(self):
      
      Nl = 1001
      L = np.logspace(np.log10(1.), np.log10(3.6e4), Nl, 10.)
      
      UnlensedTT = np.array([self.funlensedTT(l) for l in L])
      LensedCMB = np.array([self.flensedTT(l) for l in L])
      CIBPoisson = np.array([self.fCIBPoisson(l) for l in L])
      CIBClustered = np.array([self.fCIBClustered(l) for l in L])
      TSZ = np.array([self.ftSZ(l) for l in L])
      KSZ = np.array([self.fkSZ(l) for l in L])
      TSZ_CIB = np.array([self.ftSZ_CIB(l) for l in L])
      RadioPoisson = np.array([self.fradioPoisson(l) for l in L])
      GalacticDust = np.array([self.fgalacticDust(l) for l in L])
      DetectorNoise = np.array([self.fdetectorNoise(l) for l in L])
      Total = np.array([self.ftotalTT(l) for l in L])
      
      '''
      # save arrays
      Data = np.zeros((Nl, 11))
      Data[:,0] = L
      Data[:,1] = LensedCMB
      Data[:,2] = CIBPoisson
      Data[:,3] = CIBClustered
      Data[:,4] = TSZ
      Data[:,5] = KSZ
      Data[:,6] = TSZ_CIB
      Data[:,7] = RadioPoisson
      Data[:,8] = GalacticDust
      Data[:,9] = DetectorNoise
      Data[:,10] = Total
      #np.savetxt("./output/dl_elldpdtILC_ACT148_ACT218.txt", Data)
      #np.savetxt("./output/dl_ACT148.txt", Data)
      '''
      
      
      # debeamed power spectrum
      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      ax.loglog(L, UnlensedTT, 'g', label=r'unlensed TT')
      ax.loglog(L, DetectorNoise, 'r', label=r'Detector noise')
      ax.loglog(L, Total, 'k', label=r'total TT')
      #
      ax.legend(loc=1)
      ax.set_xlabel(r'\ell')
      ax.set_ylabel(r'$C_\ell$')
      
      
      # debeamed Cl
      fig=plt.figure(1, figsize=(12, 8))
      ax=plt.subplot(111)
      #
      ax.loglog(L, abs(LensedCMB), 'r', lw=2, label=r'CMB')
      ax.loglog(L, CIBPoisson, 'b--', lw=2, label=r'CIB poisson')
      ax.loglog(L, CIBClustered, 'b--', lw=2, label=r'CIB clustered')
      ax.loglog(L, TSZ, 'g', lw=2, label=r'tSZ')
      ax.loglog(L, KSZ, 'g--', lw=2, label=r'kSZ')
      ax.loglog(L, np.abs(TSZ_CIB), 'm', lw=2, label=r'$|$ tSZ x CIB $|$')
      ax.loglog(L, RadioPoisson, 'y', lw=2, label=r'radio Poisson')
      ax.loglog(L, GalacticDust, 'r', lw=2, label=r'galactic dust')
      ax.loglog(L, DetectorNoise, 'k--', lw=2, label=r'detector noise')
      ax.loglog(L, Total, 'k', lw=2, label=r'total')
      ax.loglog(L, LensedCMB+DetectorNoise, 'k-.', lw=2, label=r'CMB+detector noise')
      #
      ax.grid()
      ax.legend(loc=1)
      ax.set_xlim((1.e2, 1.e4))
      #ax.set_xlim((100., 3.6e4))
      ax.set_ylim((1.e-8, 1.e1))
      ax.set_xlabel(r'$\ell$')
      ax.set_ylabel(r'$C_\ell$ [$(\mu K)^2$]')
      #ax.set_title(r'map noise, corrected for beam('+self.name+')')
      #
      #fig.savefig("./figures/pixel_noise/pixelnoiseCl_"+self.name+".pdf")
      
      '''
      # debeamed Dl
      fig=plt.figure(2, figsize=(12, 8))
      ax=plt.subplot(111)
      #
      ax.loglog(L, abs(LensedCMB)/self.fdl_to_cl(L), 'r', lw=2, label=r'CMB')
      ax.loglog(L, CIBPoisson/self.fdl_to_cl(L), 'b--', lw=2, label=r'CIB poisson')
      ax.loglog(L, CIBClustered/self.fdl_to_cl(L), 'b--', lw=2, label=r'CIB clustered')
      ax.loglog(L, TSZ/self.fdl_to_cl(L), 'g', lw=2, label=r'tSZ')
      ax.loglog(L, KSZ/self.fdl_to_cl(L), 'g--', lw=2, label=r'kSZ')
      ax.loglog(L, np.abs(TSZ_CIB)/self.fdl_to_cl(L), 'm', lw=2, label=r'$|$ tSZ x CIB $|$')
      ax.loglog(L, RadioPoisson/self.fdl_to_cl(L), 'y', lw=2, label=r'radio Poisson')
      ax.loglog(L, GalacticDust/self.fdl_to_cl(L), 'r', lw=2, label=r'galactic dust')
      ax.loglog(L, DetectorNoise/self.fdl_to_cl(L) * np.sqrt(2./(2.*L+1.)), 'k--', lw=2, label=r'detector noise')
      #ax.loglog(L, Total/self.fdl_to_cl(L), 'k', lw=2, label=r'total')
      #
      ax.grid()
      ax.legend(loc='center left')
      #ax.set_xlim((1.e2, 1.e4))
      ax.set_ylim((1.e-4, 1.e4))
      ax.set_xlabel(r'$\ell$')
      ax.set_ylabel(r'$D_\ell$ [$(\mu K)^2$]')
      ax.set_title(r'map noise, correcting for beam, and reducing noise ('+self.name+')')
      #
      #fig.savefig("./figures/pixel_noise/Dl_"+self.name+".pdf")
      '''
      plt.show()




   def plotTEB(self):
      
      Nl = 1001
      L = np.logspace(np.log10(1.), np.log10(3.6e4), Nl, 10.)

      unlensedTT = np.array(list(map(self.funlensedTT, L)))
      unlensedEE = np.array(list(map(self.funlensedEE, L)))
      unlensedTE = np.array(list(map(self.funlensedTE, L)))
      unlensedBB = np.array(list(map(self.funlensedBB, L)))
      #
      lensedTT = np.array(list(map(self.flensedTT, L)))
      lensedEE = np.array(list(map(self.flensedEE, L)))
      lensedTE = np.array(list(map(self.flensedTE, L)))
      lensedBB = np.array(list(map(self.flensedBB, L)))
      #
      totalTT = np.array(list(map(self.ftotalTT, L)))
      totalEE = np.array(list(map(self.ftotalEE, L)))
      totalTE = np.array(list(map(self.ftotalTE, L)))
      totalBB = np.array(list(map(self.ftotalBB, L)))
      #
      Noise = np.array(list(map(self.fdetectorNoise, L)))
      AtmNoiseTT = np.array(list(map(self.fatmosphericNoiseTT, L)))
      AtmNoisePP = np.array(list(map(self.fatmosphericNoisePP, L)))

      f = 1./self.fdl_to_cl(L)
      # Dl, beam corrected
      fig=plt.figure(1, figsize=(12, 8))
      ax=plt.subplot(111)
      #
#      ax.loglog(L, f*unlensedTT, 'k--', lw=2)
#      ax.loglog(L, f*unlensedEE, 'b--', lw=2)
#      ax.loglog(L, f*unlensedTE, 'r--', lw=2)
#      ax.loglog(L, f*unlensedBB, 'g--', lw=2)
      #
      ax.loglog(L, f*lensedTT, 'k', lw=2, label=r'TT')
      ax.loglog(L, f*lensedEE, 'b', lw=2, label=r'EE')
      ax.loglog(L, f*np.abs(lensedTE), 'r', lw=2, label=r'TE')
      ax.loglog(L, f*lensedBB, 'g', lw=2, label=r'BB')
      #
      ax.loglog(L, f*totalTT, 'k', lw=1)
#      ax.loglog(L, f*(Noise + AtmNoiseTT), 'k:')
      ax.loglog(L, f*totalEE, 'b', lw=1)
#      ax.loglog(L, f*(2.*Noise + AtmNoisePP), 'b:')
      ax.loglog(L, f*np.abs(totalTE), 'r', lw=1)
      ax.loglog(L, f*totalBB, 'g', lw=1)
#      ax.loglog(L, f*(2.*Noise + AtmNoisePP), 'g:')
      #
      ax.grid()
      ax.legend(loc=1)
      #ax.set_xlim((1.e2, 1.e4))
      ax.set_ylim((1.e-6, 1.e5))
      ax.set_xlabel(r'$\ell$')
      ax.set_ylabel(r'$D_\ell$ [$(\mu K)^2$]')
      #ax.set_title(r'unlensed/lensed/lensed+noise, corrected for beam ('+self.name+')')
      #
      path = "/Users/Emmanuel/Desktop/cmb_dl.pdf"
      #fig.savefig(path, bbox_inches='tight')
      
      plt.show()


   def plotCIB(self):
      Nl = 1001
      L = np.logspace(np.log10(1.), np.log10(3.6e4), Nl, 10.)
      
      # frequencies in Hz
      #Nu = np.array([100., 143., 217., 353., 545., 857.]) * 1.e9
      Nu = np.array([353.e9])
      nNu = len(Nu)
      
      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      for iNu in range(nNu):
         nu = Nu[iNu]
         CIBPoisson = np.array([self.fCIBPoisson(l, nu, nu) for l in L])
         CIBClustered = np.array([self.fCIBClustered(l, nu, nu) for l in L])
         #
         ax.plot(L, CIBClustered, c=plt.cm.rainbow(float(iNu)/nNu), lw=2, label=str(int(old_div(nu,1.e9)))+'GHz, clustered')
         ax.plot(L, CIBPoisson, c=plt.cm.rainbow(float(iNu)/nNu), ls='--', lw=2, label=str(int(old_div(nu,1.e9)))+'GHz, Poisson')
      #
      ax.set_xscale('log')
      ax.set_yscale('log', nonposx='clip')
      ax.legend(loc=1)
      ax.set_xlim((1.e1, 1.e4))
      ax.set_xlabel(r'$\ell$', fontsize=24)
      ax.set_ylabel(r'$C_\ell$ [$(\mu K)^2$]', fontsize=24)
      #
      #fig.savefig("./figures/cmb/cl_cib_dunkley+13.pdf", bbox_inches='tight')
      
      plt.show()





###############################################################################
###############################################################################
# ACT 148GHz map

class ACT148CMB(CMB):
   
   def __init__(self):
      # name
      self.name = "act148"
      # frequencies in Hz
      self.nu1 = 148.e9
      self.nu2 = 148.e9
      # beam fwhm in radians (1.4 arcmin at 148GHz, from Hasselfield et al 2013)
      self.fwhm = 1.4 * (np.pi/180.) / 60.
      # detector sensitivity in muK*rad. (12 muK.arcmin from a non-reliable source)
      self.sensitivity = 12.*(np.pi/180.)/60.
      
      super(ACT148CMB, self).__init__()


###############################################################################
###############################################################################
# Planck SMICA map

class PlanckSMICACMB(CMB):
   
   def __init__(self):
      # frequencies in Hz (irrelevant)
      self.nu1 = 143.e9
      self.nu2 = 143.e9
      # beam fwhm in radians (5 arcmin for SMICA, from Planck XII, table 1)
      self.fwhm = 5. * (np.pi/180.) / 60.
      # detector sensitivity in muK*rad.
      # from Planck XII, fig D-2: mostly 143GHz and 217GHz dominate the SMICA map
      # from Planck VI, table 4, the detector noise for 143 and 217 is 0.8 and 1 muK*deg
      # I will use 1 muK*deg = 60 muK*arcmin to be conservative
      self.sensitivity = 60.*(np.pi/180.)/60.
      #
      self.lMin = 30.
      self.lMaxT = 2.e3
      self.lMaxP = 2.e3
      # name
      self.name = "plancksmica_beam5.0_noise60_lmaxT2000_lmaxP2000"
      
      super(PlanckSMICACMB, self).__init__()


###############################################################################
###############################################################################
# ACTPol

class ACTPolCMB(CMB):
   
   def __init__(self):
      # name
      self.name = "actpol"
      # frequencies in Hz (irrelevant)
      self.nu1 = 143.e9
      self.nu2 = 143.e9
      # beam fwhm in radians (1.4 arcmin)
      self.fwhm = 1.4 * (np.pi/180.) / 60.
      # detector sensitivity in muK*rad.
      # 18 muK*arcmin
      self.sensitivity = 18.*(np.pi/180.)/60.
      
      super(ACTPolCMB, self).__init__()


###############################################################################
###############################################################################
# AdvACT

class AdvACTCMB(CMB):
   
   def __init__(self):
      # name
      self.name = "advact"
      # frequencies in Hz (irrelevant)
      self.nu1 = 143.e9
      self.nu2 = 143.e9
      # beam fwhm in radians (1.4 arcmin)
      self.fwhm = 1.4 * (np.pi/180.) / 60.
      # detector sensitivity in muK*rad.
      # 10 muK*arcmin
      self.sensitivity = 10.*(np.pi/180.)/60.
      
      super(AdvACTCMB, self).__init__()


###############################################################################
###############################################################################
# CMB Stage IV

class StageIVCMB(CMB):
   
   def __init__(self, beam=1., noise=1., lMin=30., lMaxT=3.e3, lMaxP=5.e3, atm=False, name=None):
      # name
      #      self.name = "cmbs4"
      self.name = "cmbs4_beam"+str(round(beam, 3))+"_noise"+str(round(noise, 3))+"_lmin"+str(int(lMin))+"_lmaxT"+str(int(lMaxT))+"_lmaxP"+str(int(lMaxP))
      if atm:
         self.name += "_atmnoise"
      if name is not None:
         self.name += "_"+name

      # frequencies in Hz (irrelevant)
      self.nu1 = 143.e9
      self.nu2 = 143.e9
      # beam fwhm in radians (1 arcmin)
      self.fwhm = beam * (np.pi/180.)/60.
      # detector sensitivity in muK*rad.
      # 1 muK*arcmin
      self.sensitivity = noise * (np.pi/180.)/60.
      # ell limits
      self.lMin = lMin
      self.lMaxT = lMaxT
      self.lMaxP = lMaxP
      
      super(StageIVCMB, self).__init__()
      
      # add atmospheric noise if needed
      if atm:
         self.ftotalTT = lambda l: self.flensedTT(l) + self.fdetectorNoise(l) + self.fatmosphericNoiseTT(l)
         self.ftotalEE = lambda l: self.flensedEE(l) + 2.*self.fdetectorNoise(l) + self.fatmosphericNoisePP(l)
         self.ftotalBB = lambda l: self.flensedBB(l) + 2.*self.fdetectorNoise(l) + self.fatmosphericNoisePP(l)
         self.ftotalTE = lambda l: self.flensedTE(l)

###############################################################################
###############################################################################
# the "reference CMB experiment" from Hu & Okamoto
# noise is 1muK.arcmin, beam fwhm=4arcmin

class HuOkamoto2002(CMB):
   
   def __init__(self):
      # name
      self.name = "huokamoto02"
      # frequencies in Hz (irrelevant)
      self.nu1 = 143.e9
      self.nu2 = 143.e9
      # beam fwhm in radians (4 arcmin)
      self.fwhm = 4 * (np.pi/180.) / 60.
      # detector sensitivity in muK*rad.
      # 1 muK*arcmin
      self.sensitivity = 1.*(np.pi/180.)/60.
      
      super(HuOkamoto2002, self).__init__()









