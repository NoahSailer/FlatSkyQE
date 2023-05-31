from headers import *

import cmb
from cmb import *

###############################################################################
# Simple ILC class, following Delabrouille Cardoso 2007

class CMBILC(object):
   
   def __str__(self):
      return self.name
   
   def __init__(self, Nu, Beam, Noise, cl=None, name="test", atm=False, lMaxT=3500., lMaxP=3500.):
      '''Simple ILC class, following Delabrouille Cardoso 2007.
      nui: various freqs to be included in the ILC, in Hz
      clij: matrix of functions for the various auto and cross-spectra.
      '''
      self.name = name
      self.Nu = Nu
      self.nNu = len(self.Nu)
      self.Beam = Beam
      self.Noise = Noise
      self.cl = cl
      #
      self.lMin = 10.
      self.lMaxT = lMaxT
      self.lMaxP = lMaxP


      # initialize the auto and cross spectra
      self.cmb = np.empty((self.nNu, self.nNu), dtype=object)
      for i in range(self.nNu):
         for j in range(self.nNu):
            beam = np.sqrt((self.Beam[i]**2 + self.Beam[j]**2) / 2.)
            if i==j:
               noise = self.Noise[i]
            else:
               noise = np.sqrt(self.Noise[i]*self.Noise[j]) # used to be 0
            self.cmb[i,j] = CMB(beam=beam, noise=noise, nu1=self.Nu[i], nu2=self.Nu[j], lMin=self.lMin, lMaxT=self.lMaxT, lMaxP=self.lMaxP, fg=True, atm=atm, name=None)




   ###############################################################################
   
   def powerIlc(self, w, l):
      '''Compute the power spectrum of the linear combination
      sum_i w_i T_i.
      '''
      # covariance matrix of the frequency channels
      c = np.zeros((self.nNu, self.nNu))
      for i in range(self.nNu):
         for j in range(self.nNu):
            c[i,j] = self.cmb[i,j].ftotalTT(l)
      # compute the power spectrum
      return (w.transpose()).dot(c.dot(w))
    
   def powerIlcEE(self, w, l):
      '''Compute the power spectrum of the linear combination
      sum_i w_i T_i.
      '''
      # covariance matrix of the frequency channels
      c = np.zeros((self.nNu, self.nNu))
      for i in range(self.nNu):
         for j in range(self.nNu):
            c[i,j] = self.cmb[i,j].ftotalEE(l)
      # compute the power spectrum
      return (w.transpose()).dot(c.dot(w))
    
   def powerIlcTE(self, wT, wE, l):
      '''Compute the power spectrum of the linear combination
      sum_i w_i T_i.
      '''
      # covariance matrix of the frequency channels
      c = np.zeros((self.nNu, self.nNu))
      for i in range(self.nNu):
         for j in range(self.nNu):
            c[i,j] = self.cmb[i,j].ftotalTE(l)
      # compute the power spectrum
      return (wT.transpose()).dot(c.dot(wE))


   ###############################################################################
   # CMB ILC

   def weightsIlcCmb(self, l):
      '''CMB ILC: min var unbiased CMB estimator, from linear combination of freq maps.
      Assumes the various freq channels are already calibrated to unit response to CMB unit.
      The weights include the covariance between freq channels.
      They add up to 1.
      '''
      # covariance matrix of the frequency channels
      c = np.zeros([self.nNu, self.nNu])
      for i in range(self.nNu):
         for j in range(self.nNu):
            c[i,j] = self.cmb[i,j].ftotalTT(l)
      # invert matrix
      cInv = np.linalg.inv(c)
      # generate the weights
      w = np.zeros(self.nNu)
      for i in range(self.nNu):
         w[i] = np.sum(cInv[i,:]) / np.sum(cInv)
      return w


   def weightsIlcCmbEE(self, l):
      '''CMB ILC: min var unbiased CMB estimator, from linear combination of freq maps.
      Assumes the various freq channels are already calibrated to unit response to CMB unit.
      The weights include the covariance between freq channels.
      They add up to 1.
      '''
      # covariance matrix of the frequency channels
      c = np.zeros([self.nNu, self.nNu])
      for i in range(self.nNu):
         for j in range(self.nNu):
            c[i,j] = self.cmb[i,j].ftotalEE(l)
      # invert matrix
      cInv = np.linalg.inv(c)
      # generate the weights
      w = np.zeros(self.nNu)
      for i in range(self.nNu):
         w[i] = np.sum(cInv[i,:]) / np.sum(cInv)
      return w


   def plotWeightsIlcCmb(self):
      nL = 501
      L = np.logspace(np.log10(self.lMin), np.log10(self.lMaxT), nL, 10.)
      W = np.array(map(self.weightsIlcCmb, L))

      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      for i in range(self.nNu):
         ax.plot(L, W[:,i], label=str(np.int(self.Nu[i]/1.e9))+' GHz')
      #
      ax.legend(loc=1, fontsize='x-small', labelspacing=0.1)
      ax.set_xscale('log', nonposx='clip')
      plt.show()


   def plotPowerIlcCmb(self):
      nL = 501
      L = np.logspace(np.log10(self.lMin), np.log10(self.lMaxT), nL, 10.)
      f = lambda l: self.powerIlc(self.weightsIlcCmb(l), l)
      powerIlcCmb = np.array(map(f, L))

      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      factor = L*(L+1.)/(2.*np.pi)
      # individual channels
      for i in range(self.nNu):
         cl = np.array(map(self.cmb[i,i].ftotalTT, L))
         ax.plot(L, factor * cl, '--', label=str(np.int(self.Nu[i]/1.e9))+' GHz')
      #
      # ILC
      ax.plot(L, factor * powerIlcCmb, 'k', label=r'ILC')
      # lensed CMB
      cl = np.array(map(self.cmb[i,i].flensedTT, L))
      ax.plot(L, factor * cl, 'grey', label=r'lensed CMB')
      #
      ax.legend(loc=3, fontsize='x-small', labelspacing=0.1)
      ax.set_xscale('log', nonposx='clip')
      ax.set_yscale('log', nonposy='clip')
      ax.set_xlabel(r'$\ell$')
      ax.set_ylabel(r'$\ell(\ell+1) C_\ell / (2\pi)$')
      ax.set_ylim((10., 1.e4))

      plt.show()


      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      # individual channels
      for i in range(self.nNu):
         cl = np.array(map(self.cmb[i,i].ftotalTT, L))
         ax.plot(L, cl / powerIlcCmb, '--', label=str(np.int(self.Nu[i]/1.e9))+' GHz')
      #
      # ILC
      ax.plot(L, np.ones_like(L), 'k', label=r'ILC')
      #
      ax.legend(loc=3, fontsize='x-small', labelspacing=0.1)
      ax.set_xscale('log', nonposx='clip')
      #ax.set_yscale('log', nonposy='clip')
      ax.set_xlabel(r'$\ell$')
      ax.set_ylabel(r'$C_\ell^{\nu_i, \nu_i} / C_\ell^\text{ILC}$')
      #ax.set_ylim((10., 1.e4))

      plt.show()


   ###############################################################################
   # Basic ILC for tSZ, ignoring the correlation with CIB

   def weightsIlcTsz(self, l):
      '''tSZ ILC: min var unbiased tSZ estimator, from linear combination of freq maps.
      Ignores the correlation with CIB, so will be biased in auto or cross.
      Assumes the various freq channels are calibrated to unit response to CMB unit.
      The weights include the covariance between freq channels.
      They do not add up to 1, since the freq maps are not calibrated to have unit response to tSZ.
      '''
      # covariance matrix of the frequency channels
      c = np.zeros([self.nNu, self.nNu])
      for i in range(self.nNu):
         for j in range(self.nNu):
            c[i,j] = self.cmb[i,j].ftotalTT(l)
            c[i,j] /= self.cmb[0,0].tszFreqDpdceT(self.Nu[i]) * self.cmb[0,0].tszFreqDpdceT(self.Nu[j])
      # invert matrix
      cInv = np.linalg.inv(c)
      # generate the weights
      w = np.zeros(self.nNu)
      for i in range(self.nNu):
         w[i] = np.sum(cInv[i,:]) / np.sum(cInv)
         w[i] /= self.cmb[0,0].tszFreqDpdceT(self.Nu[i])
      return w


   def plotWeightsIlcTsz(self):
      nL = 501
      L = np.logspace(np.log10(self.lMin), np.log10(self.lMaxT), nL, 10.)
      W = np.array(map(self.weightsIlcTsz, L))

      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      for i in range(self.nNu):
         ax.plot(L, W[:,i], label=str(np.int(self.Nu[i]/1.e9))+' GHz')
      #
      ax.legend(loc=1, fontsize='x-small', labelspacing=0.1)
      ax.set_xscale('log', nonposx='clip')

      plt.show()


   def plotPowerIlcTsz(self):
      nL = 501
      L = np.logspace(np.log10(self.lMin), np.log10(self.lMaxT), nL, 10.)
      f = lambda l: self.powerIlc(self.weightsIlcTsz(l), l)
      powerIlcTsz = np.array(map(f, L))

      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      factor = L*(L+1.)/(2.*np.pi)
      # individual channels
      for i in range(self.nNu):
         cl = np.array(map(self.cmb[i,i].ftotalTT, L))
         cl /= self.cmb[0,0].tszFreqDpdceT(self.Nu[i])**2
         ax.plot(L, factor * cl, '--', label=str(np.int(self.Nu[i]/1.e9))+' GHz')
      #
      # ILC
      ax.plot(L, factor * powerIlcTsz, 'k', label=r'ILC')
      # tSZ
      cl = np.array(map(self.cmb[0,0].ftSZ, L))
      cl /= self.cmb[0,0].tszFreqDpdceT(self.Nu[0])**2
      ax.plot(L, factor * cl, 'grey', label=r'$y$')
      #
      ax.legend(loc=3, fontsize='x-small', labelspacing=0.1)
      ax.set_xscale('log', nonposx='clip')
      ax.set_yscale('log', nonposy='clip')
      ax.set_xlabel(r'$\ell$')
      ax.set_ylabel(r'$\ell(\ell+1) C_\ell / (2\pi)$')
      #ax.set_ylim((10., 1.e4))

      plt.show()

      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      # individual channels
      for i in range(self.nNu):
         cl = np.array(map(self.cmb[i,i].ftotalTT, L))
         cl /= self.cmb[0,0].tszFreqDpdceT(self.Nu[i])**2
         ax.plot(L, cl / powerIlcTsz, '--', label=str(np.int(self.Nu[i]/1.e9))+' GHz')
      #
      # ILC
      ax.plot(L, np.ones_like(L), 'k', label=r'ILC')
      #
      ax.legend(loc=3, fontsize='x-small', labelspacing=0.1)
      ax.set_xscale('log', nonposx='clip')
      ax.set_yscale('log', nonposy='clip')
      ax.set_xlabel(r'$\ell$')
      ax.set_ylabel(r'$C_\ell^{\nu_i, \nu_i} / C_\ell^\text{ILC}$')
      #ax.set_ylim((10., 1.e4))

      plt.show()


   ###############################################################################
   # Constrained ILC: unbiased tSZ map, with zero response to CIB


   def weightsConstrainedIlcTszNoCib(self, l):
      '''tSZ constrained ILC, with null response to CIB.
      Min var unbiased tSZ estimator, from linear combination of freq maps, with zero response to CIB.
      Assumes the various freq channels are calibrated to unit response to CMB unit.
      The weights include the covariance between freq channels.
      They do not add up to 1, since the freq maps are not calibrated to have unit response to tSZ.
      '''
      # covariance matrix of the frequency channels,
      # not including the tSZ component and its correlation with CIB
      c = np.zeros([self.nNu, self.nNu])
      for i in range(self.nNu):
         for j in range(self.nNu):
            c[i,j] = self.cmb[i,j].flensedTT(l) + self.cmb[i,j].fkSZ(l) + self.cmb[i,j].fradioPoisson(l) + self.cmb[i,j].fdetectorNoise(l) + self.cmb[i,j].fCIB(l)
            c[i,j] /= self.cmb[0,0].tszFreqDpdceT(self.Nu[i]) * self.cmb[0,0].tszFreqDpdceT(self.Nu[j])
      # invert matrix
      cInv = np.linalg.inv(c)
      # freq dpdce of the CIB to be nulled
      f = np.array(map(self.cmb[0,0].cibPoissonFreqDpdceT, self.Nu))
      f /= np.array(map(self.cmb[0,0].tszFreqDpdceT, self.Nu))
      # useful numbers
      S0 = np.sum(cInv)
      S1 = np.sum(np.dot(cInv, f))
      S2 = f.transpose().dot(cInv.dot(f))
      # other vector needed
      one = np.ones(self.nNu)
      # generate the weights
      w = cInv.dot(S2*one - S1*f)
      w /= S0 * S2 - S1**2
      w /= np.array(map(self.cmb[0,0].tszFreqDpdceT, self.Nu))
      return w


   def plotWeightsConstrainedIlcTszNoCib(self):
      nL = 501
      L = np.logspace(np.log10(self.lMin), np.log10(self.lMaxT), nL, 10.)
      W = np.array(map(self.weightsConstrainedIlcTszNoCib, L))

      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      for i in range(self.nNu):
         ax.plot(L, W[:,i], label=str(np.int(self.Nu[i]/1.e9))+' GHz')
      #
      ax.legend(loc=1, fontsize='x-small', labelspacing=0.1)
      ax.set_xscale('log', nonposx='clip')

      plt.show()


   def plotPowerConstrainedIlcTszNoCib(self):
      nL = 501
      L = np.logspace(np.log10(self.lMin), np.log10(self.lMaxT), nL, 10.)
      f = lambda l: self.powerIlc(self.weightsConstrainedIlcTszNoCib(l), l)
      powerConstrainedIlcTszNoCib = np.array(map(f, L))

      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      factor = L*(L+1.)/(2.*np.pi)
      # individual channels
      for i in range(self.nNu):
         cl = np.array(map(self.cmb[i,i].ftotalTT, L))
         cl /= self.cmb[0,0].tszFreqDpdceT(self.Nu[i])**2
         ax.plot(L, factor * cl, '--', label=str(np.int(self.Nu[i]/1.e9))+' GHz')
      #
      # constrained ILC
      ax.plot(L, factor * powerConstrainedIlcTszNoCib, 'k', label=r'ILC')
      # tSZ
      cl = np.array(map(self.cmb[0,0].ftSZ, L))
      cl /= self.cmb[0,0].tszFreqDpdceT(self.Nu[0])**2
      ax.plot(L, factor * cl, 'grey', label=r'$y$')
      #
      ax.legend(loc=3, fontsize='x-small', labelspacing=0.1)
      ax.set_xscale('log', nonposx='clip')
      ax.set_yscale('log', nonposy='clip')
      ax.set_xlabel(r'$\ell$')
      ax.set_ylabel(r'$\ell(\ell+1) C_\ell / (2\pi)$')
      #ax.set_ylim((10., 1.e4))

      plt.show()



      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      # individual channels
      for i in range(self.nNu):
         cl = np.array(map(self.cmb[i,i].ftotalTT, L))
         cl /= self.cmb[0,0].tszFreqDpdceT(self.Nu[i])**2
         ax.plot(L, cl / powerConstrainedIlcTszNoCib, '--', label=str(np.int(self.Nu[i]/1.e9))+' GHz')
      #
      # ILC
      ax.plot(L, np.ones_like(L), 'k', label=r'ILC')
      #
      ax.legend(loc=3, fontsize='x-small', labelspacing=0.1)
      ax.set_xscale('log', nonposx='clip')
      ax.set_yscale('log', nonposy='clip')
      ax.set_xlabel(r'$\ell$')
      ax.set_ylabel(r'$C_\ell^{\nu_i, \nu_i} / C_\ell^\text{ILC}$')
      #ax.set_ylim((10., 1.e4))

      plt.show()


   ###############################################################################
   # Basic ILC for deprojecting tSZ and CIB

   def weightsDeprojTszCIB(self, l):
      '''CMB ILC: min var ILC with zero response to tSZ and CIB.
         They sum to 1.
      '''
      # covariance matrix of the frequency channels
      c = np.zeros([self.nNu, self.nNu])
      for i in range(self.nNu):
         for j in range(self.nNu):
            c[i,j] = self.cmb[i,j].ftotalTT(l)
      cinv = np.linalg.inv(c)
      # solving the constraint equations for zero response to tSZ and CIB
      ftSZ = np.array([self.cmb[0,0].tszFreqDpdceT(self.Nu[i]) for i in range(self.nNu)])
      fCIB = np.array([self.cmb[0,0].cibPoissonFreqDpdceT(self.Nu[i]) for i in range(self.nNu)])
      ftSZ /= 5.e2
      fCIB /= 4.e26 # hopefully this works
      one = np.ones(self.nNu)
      #
      a,b = np.dot(ftSZ,np.dot(cinv,ftSZ)) , np.dot(ftSZ,np.dot(cinv,fCIB))
      c,d = np.dot(fCIB,np.dot(cinv,fCIB)) , np.dot(one,np.dot(cinv,ftSZ))
      e = np.dot(one,np.dot(cinv,fCIB))
      Minv = np.array([[c,-b],[-b,a]])/(a*c-b**2.)
      x = -1.*np.array([d,e])
      lambdas = np.dot(Minv,x)
      # weights
      weights = np.dot(cinv, one + lambdas[0]*ftSZ + lambdas[1]*fCIB)
      normalization = np.dot(weights,one)
      return weights/normalization
      
      
   ###############################################################################
   # Basic ILC for deprojecting tSZ

   def weightsDeprojTsz(self, l):
      '''CMB ILC: min var ILC with zero response to tSZ and CIB.
         They sum to 1.
      '''
      # covariance matrix of the frequency channels
      c = np.zeros([self.nNu, self.nNu])
      for i in range(self.nNu):
         for j in range(self.nNu):
            c[i,j] = self.cmb[i,j].ftotalTT(l)
      cinv = np.linalg.inv(c)
      # solving the constraint equations for zero response to tSZ and CIB
      ftSZ = np.array([self.cmb[0,0].tszFreqDpdceT(self.Nu[i]) for i in range(self.nNu)])
      ftSZ /= 5.e2
      one = np.ones(self.nNu)
      #
      lambdas = -1.*np.dot(np.dot(one,cinv),ftSZ) / np.dot(np.dot(ftSZ,cinv),ftSZ)
      # weights
      weights = np.dot(cinv, one + lambdas*ftSZ )
      normalization = np.dot(weights,one)
      return weights/normalization


   ###############################################################################
   # Basic ILC for deprojecting CIB

   def weightsDeprojCIB(self, l):
      '''CMB ILC: min var ILC with zero response to tSZ and CIB.
         They sum to 1.
      '''
      # covariance matrix of the frequency channels
      c = np.zeros([self.nNu, self.nNu])
      for i in range(self.nNu):
         for j in range(self.nNu):
            c[i,j] = self.cmb[i,j].ftotalTT(l)
      cinv = np.linalg.inv(c)
      # solving the constraint equations for zero response to tSZ and CIB
      ftSZ = np.array([self.cmb[0,0].cibPoissonFreqDpdceT(self.Nu[i]) for i in range(self.nNu)])
      ftSZ /= 5.e2
      one = np.ones(self.nNu)
      #
      lambdas = -1.*np.dot(np.dot(one,cinv),ftSZ) / np.dot(np.dot(ftSZ,cinv),ftSZ)
      # weights
      weights = np.dot(cinv, one + lambdas*ftSZ )
      normalization = np.dot(weights,one)
      return weights/normalization