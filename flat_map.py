from __future__ import print_function
from __future__ import division
from builtins import str
from builtins import map
from builtins import range
from builtins import object
from headers import *

###################################################################

class FlatMap(object):
   
   def __init__(self, nX=256, nY=256, sizeX=5.*np.pi/180., sizeY=5.*np.pi/180., name="test"):
      """n is number of pixels on one side
      size is the angular size of the side, in radians
      """
      self.name = name
      self.nX = nX
      self.sizeX = sizeX
      self.dX = float(sizeX)/(nX-1)
      x = self.dX * np.arange(nX)   # the x value corresponds to the center of the cell
      #
      self.nY = nY
      self.sizeY = sizeY
      self.dY = float(sizeY)/(nY-1)
      y = self.dY * np.arange(nY)   # the y value corresponds to the center of the cell
      #
      self.x, self.y = np.meshgrid(x, y, indexing='ij')
      #
      self.fSky = self.sizeX*self.sizeY / (4.*np.pi)
      
      self.data = np.zeros((nX,nY))
   
      lx = np.zeros(nX)
      lx[:nX//2+1] = 2.*np.pi/sizeX * np.arange(nX//2+1)
      lx[nX//2+1:] = 2.*np.pi/sizeX * np.arange(-nX//2+1, 0, 1)
      ly = 2.*np.pi/sizeY * np.arange(nY//2+1)
      self.lx, self.ly = np.meshgrid(lx, ly, indexing='ij')
      
      self.l = np.sqrt(self.lx**2 + self.ly**2)
      self.dataFourier = np.zeros((nX,nY//2+1))
   
   def copy(self):
      newMap = FlatMap(nX=self.nX, nY=self.nY, sizeX=self.sizeX, sizeY=self.sizeY, name=self.name)
      newMap.data = self.data.copy()
      newMap.dataFourier = self.dataFourier.copy()
      return newMap
   
   def write(self, path=None):
      # primary hdu: size of map
      prihdr = fits.Header()
      prihdr['name'] = self.name
      prihdr['nX'] = self.nX
      prihdr['sizeX'] = self.sizeX
      prihdr['nY'] = self.nY
      prihdr['sizeY'] = self.sizeY
      hdu0 = fits.PrimaryHDU(header=prihdr)
      # secondary hdu: real-space and Fourier maps
      c1 = fits.Column(name='data', format='D', array=self.data.flatten())
      c2 = fits.Column(name='dataFourier', format='M', array=self.dataFourier.flatten())
      hdu1 = fits.BinTableHDU.from_columns([c1, c2])
      #
      hdulist = fits.HDUList([hdu0, hdu1])
      
      if path is None:
         path = "./output/lens_simulator/"+self.name+".fits"
      print("writing to "+path)
      hdulist.writeto(path, overwrite=True)

   
   def read(self, path=None):
      if path is None:
         path = "./output/lens_simulator/"+self.name+".fits"
      print("reading from "+path)
      hdulist = fits.open(path)
      #
      #self.name = hdulist[0].header['name']
      self.nX = hdulist[0].header['nX']
      self.sizeX = hdulist[0].header['sizeX']
      self.nY = hdulist[0].header['nY']
      self.sizeY = hdulist[0].header['sizeY']
      #
      self.__init__(nX=self.nX, nY=self.nY, sizeX=self.sizeX, sizeY=self.sizeY, name=self.name)
      #
      self.data = hdulist[1].data['data'].reshape((self.nX, self.nY))
      self.dataFourier = hdulist[1].data['dataFourier'][:self.nX*(self.nY//2+1)].reshape((self.nX, self.nY//2+1))
      #
      hdulist.close()
   
   
   def saveDataFourier(self, dataFourier, path):
      print("saving Fourier map to"+ path)
      # primary hdu: size of map
      prihdr = fits.Header()
      prihdr['nX'] = self.nX
      prihdr['sizeX'] = self.sizeX
      prihdr['nY'] = self.nY
      prihdr['sizeY'] = self.sizeY
      hdu0 = fits.PrimaryHDU(header=prihdr)
      # secondary hdu: maps
      c1 = fits.Column(name='dataFourier', format='M', array=dataFourier.flatten())
      hdu1 = fits.BinTableHDU.from_columns([c1])
      #
      hdulist = fits.HDUList([hdu0, hdu1])
      #
      hdulist.writeto(path, overwrite=True)

   def loadDataFourier(self, path):
      print("reading fourier map from"+path)
      hdulist = fits.open(path)
      dataFourier = hdulist[1].data['dataFourier'][:self.nX*(self.nY//2+1)].reshape((self.nX, self.nY//2+1))
      hdulist.close()
      return dataFourier
   
   ###############################################################################
   # change resolution of map
   
   def downResolution(self, nXNew, nYNew, data=None, test=False):
      """Expects nXNew < self.nX and nYNew < self.nY
      """
      if data is None:
         data = self.data.copy()
      # Fourier transform the map
      dataFourier = self.fourier(data)
      # truncate the fourier map
      newMap = FlatMap(nXNew, nYNew, sizeX=self.sizeX, sizeY=self.sizeY)
      IX = list(range(nXNew//2+1)) + list(range(-nXNew//2+1, 0))
      IY = list(range(nYNew//2+1))
      IX, IY = np.meshgrid(IX, IY, indexing='ij')
      newMap.dataFourier = dataFourier[IX, IY]
      # update real space map
      newMap.data = newMap.inverseFourier()
      if test:
         self.plot(data)
         newMap.plot(newMap.data)
      return newMap.data

   def upResolution(self, nXNew, nYNew, data=None, test=False):
      """Expects nXNew > self.nX and nYNew > self.nY
      """
      if data is None:
         data = self.data.copy()
      # Fourier transform the map
      dataFourier = self.fourier(data)
      # zero-pad the fourier map
      newMap = FlatMap(nXNew, nYNew, sizeX=self.sizeX, sizeY=self.sizeY)
      IX = list(range(self.nX//2+1)) + list(range(-self.nX//2+1, 0))
      IY = list(range(self.nY//2+1))
      newMap.dataFourier[IX, IY] = dataFourier
      # update real space map
      newMap.data = newMap.inverseFourier()
      if test:
         self.plot(data)
         newMap.plot(newMap.data)
      return newMap.data

   
   ###############################################################################
   # plots

   def plot(self, data=None, save=False, path=None, cmap='viridis'):
      if data is None:
         data = self.data.copy()
      sigma = np.std(data.flatten())
      vmin = np.min(data.flatten())
      vmax = np.max(data.flatten())

      fig=plt.figure(0,figsize=(8,8))
      ax=fig.add_subplot(111)
      #
      # pcolor wants x and y to be edges of cell,
      # ie one more element, and offset by half a cell
      x = self.dX * (np.arange(self.nX+1) - 0.5)
      y = self.dY * (np.arange(self.nY+1) - 0.5)
      x,y = np.meshgrid(x, y, indexing='ij')
      #
      cp=ax.pcolormesh(x*180./np.pi, y*180./np.pi, data, linewidth=0, rasterized=True)
      #
      # choose color map: jet, summer, winter, Reds, gist_gray, YlOrRd, bwr, seismic
      cp.set_cmap(cmap)
      #cp.set_clim(0.,255.)
      #cp.set_clim(-3.*sigma, 3.*sigma)
      cp.set_clim(vmin, vmax)
      fig.colorbar(cp)
      #
      plt.axis('scaled')
      ax.set_xlim(np.min(x)*180./np.pi, np.max(x)*180./np.pi)
      ax.set_ylim(np.min(y)*180./np.pi, np.max(y)*180./np.pi)
      ax.set_xlabel('$x$ [deg]')
      ax.set_ylabel('$y$ [deg]')
      #
      if save==True:
         if path is None:
            path = "./figures/lens_simulator/"+self.name+".pdf"
         print("saving plot to "+path)
         fig.savefig(path, bbox_inches='tight')
         fig.clf()
      else:
         plt.show()


   def plotFourier(self, dataFourier=None, save=False, name=None, cmap='viridis'):
      if dataFourier is None:
         dataFourier = self.dataFourier.copy()
      dataFourier = np.real(dataFourier)
      sigma = np.std(dataFourier.flatten())
      vmin = np.min(dataFourier.flatten())
      vmax = np.max(dataFourier.flatten())

      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      # pcolor wants x and y to be edges of cell,
      # ie one more element, and offset by half a cell
      #
      # left part of plot
      lxLeft = 2.*np.pi/self.sizeX * (np.arange(-self.nX/2+1, 1, 1) - 0.5)
      ly = 2.*np.pi/self.sizeY * (np.arange(self.nY//2+1+1) - 0.5)
      lx, ly = np.meshgrid(lxLeft, ly, indexing='ij')
      cp1=ax.pcolormesh(lx, ly, dataFourier[self.nX//2+1:,:], linewidth=0, rasterized=True)
      #
      # right part of plot
      lxRight = 2.*np.pi/self.sizeX * (np.arange(self.nX//2+1+1) - 0.5)
      ly = 2.*np.pi/self.sizeY * (np.arange(self.nY//2+1+1) - 0.5)
      lx, ly = np.meshgrid(lxRight, ly, indexing='ij')
      cp2=ax.pcolormesh(lx, ly, dataFourier[self.nX//2+1,:], linewidth=0, rasterized=True)
      #
      # choose color map: jet, summer, winter, Reds, gist_gray, YlOrRd, bwr, seismic
      cp1.set_cmap(cmap); cp2.set_cmap(cmap)
      #cp1.set_clim(0.,255.); cp2.set_clim(0.,255.)
      #cp1.set_clim(-3.*sigma, 3.*sigma); cp2.set_clim(-3.*sigma, 3.*sigma)
      cp1.set_clim(vmin, vmax); cp2.set_clim(vmin, vmax)
      #
      fig.colorbar(cp1)
      #
      plt.axis('scaled')
      ax.set_xlim(np.min(lxLeft), np.max(lxRight))
      ax.set_ylim(np.min(ly), np.max(ly))
      ax.set_xlabel('$\ell_x$')
      ax.set_ylabel('$\ell_y$')
      #
      if save==True:
         if name is None:
            name = self.name
         print("saving plot to "+"./figures/lens_simulator/"+name+".pdf")
         fig.savefig("./figures/lens_simulator/"+name+".pdf", bbox_inches='tight')
         fig.clf()
      else:
         plt.show()



   def plotHistogram(self, data=None, save=False, name=None, nBins=100):
      """plot a pixel histogram
      """
      if data is None:
         data = self.data.copy()
      
      # data mean, var, skewness, kurtosis
      mean = np.mean(data)
      sigma = np.std(data)
      skewness = np.mean((data-mean)**3) / sigma**3
      kurtosis = np.mean((data-mean)**4) / sigma**4
      
      print("mean ="+mean)
      print("std. dev ="+sigma)
      print("skewness ="+skewness)
      print("kurtosis ="+kurtosis)
      
      # data histogram, and error bars on it
      pdf, binEdges = np.histogram(data, density=False, bins=nBins)
      binWidth = (np.max(data)-np.min(data)) / nBins
      spdf = np.sqrt(pdf) / (np.sum(pdf)*binWidth) # absolute 1-sigma error on pdf
      pdf = pdf / (np.sum(pdf)*binWidth)
      
      # Gaussian histogram with same mean and var
      samples = np.random.normal(loc=mean, scale=sigma, size=self.nX*self.nY)
      pdfGauss, binEdgesGauss = np.histogram(samples, density=True, bins=nBins, range=(binEdges[0], binEdges[-1]))

      # Poisson histogram with same mean
      if mean>0.:
         samples = np.random.poisson(lam=mean, size=self.nX*self.nY)
         pdfPoiss, binEdgesPoiss = np.histogram(samples, density=True, bins=nBins, range=(binEdges[0], binEdges[-1]))

      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      # data histogram
      ax.step(binEdges[:-1], pdf, where='post', color='b', lw=2, label=r'map')
      # error band on data histogram
      ax.fill_between(np.linspace(binEdges[0], binEdges[-1], 100*len(pdf)),
                      np.repeat(pdf-spdf, 100),
                      np.repeat(pdf+spdf, 100),
                      color='b')
      #
      # Gaussian histogram
      ax.step(binEdgesGauss[:-1], pdfGauss, where='post', color='r', lw=2, label=r'Gaussian')
      ax.axvline(mean, color='k')
      ax.axvline(mean + 1.*sigma, color='gray')
      ax.axvline(mean + 2.*sigma, color='gray')
      ax.axvline(mean + 3.*sigma, color='gray')
      ax.axvline(mean + 4.*sigma, color='gray')
      ax.axvline(mean + 5.*sigma, color='gray')
      ax.axvline(mean + -1.*sigma, color='gray')
      ax.axvline(mean + -2.*sigma, color='gray')
      ax.axvline(mean + -3.*sigma, color='gray')
      ax.axvline(mean + -4.*sigma, color='gray')
      ax.axvline(mean + -5.*sigma, color='gray')
      #
      # Poisson histogram
      if mean>0.:
         ax.step(binEdgesPoiss[:-1], pdfPoiss, where='post', color='g', lw=2, label=r'Poisson')
      #
      ax.legend(loc=1)
      #ax.set_xlim((-4.*sigma, 4.*sigma))
      ax.set_yscale('log')
      #
      if save==True:
         if name is None:
            name = self.name
         print("saving plot to "+"./figures/lens_simulator/histogram_"+name+".pdf")
         fig.savefig("./figures/lens_simulator/histogram_"+name+".pdf", bbox_inches='tight')
         fig.clf()
      else:
         plt.show()


#   !!! the x and y axes might be reversed!
   def plotDeflectionFieldLines(self, dx, dy):
      d = np.sqrt(dx**2+dy**2)
      d /= np.max(d.flatten())
   
      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      strm = ax.streamplot(self.y*180./np.pi, self.x*180./np.pi, dy, dx, color=d, linewidth=3.*d, cmap=plt.cm.autumn, density=3.)
      fig.colorbar(strm.lines)
      #
      plt.axis('scaled')
      ax.set_xlim(np.min(self.x.flatten())*180./np.pi, np.max(self.x.flatten())*180./np.pi)
      ax.set_ylim(np.min(self.y.flatten())*180./np.pi, np.max(self.y.flatten())*180./np.pi)
      ax.set_xlabel('$x$ [deg]')
      ax.set_ylabel('$y$ [deg]')
      
      plt.show()
   

   def plotDeflectionArrows(self, dx, dy, save=False, name=None):
      d = np.sqrt(dx**2+dy**2)
      
      # Keep only some of the points
      skip = (slice(None, None, self.nX/25), slice(None, None, self.nX/25))
      
      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      #
      # pcolor wants x and y to be edges of cell,
      # ie one more element, and offset by half a cell
      x = self.dX * (np.arange(self.nX+1) - 0.5)
      y = self.dY * (np.arange(self.nY+1) - 0.5)
      x,y = np.meshgrid(x, y, indexing='ij')
      #
      sigma = self.data.std()
      cp=ax.pcolormesh(x*180./np.pi, y*180./np.pi, self.data, linewidth=0, rasterized=True, alpha=1, cmap=plt.cm.jet)
#      cp=ax.pcolormesh(x*180./np.pi, y*180./np.pi, self.data, linewidth=0, rasterized=True, alpha=1, vmin=-2.5*sigma, vmax=2.5*sigma, cmap=cmaps.viridis)
      fig.colorbar(cp)
      #
      #
#      ax.quiver(self.x[skip]*180./np.pi, self.y[skip]*180./np.pi, dx[skip]*180./np.pi, dy[skip]*180./np.pi, d[skip]*180./np.pi, units='xy', edgecolor='', width=0.007*self.sizeX*180./np.pi)
      ax.quiver(self.x[skip]*180./np.pi, self.y[skip]*180./np.pi, dx[skip]*180./np.pi, dy[skip]*180./np.pi, facecolor='k', units='xy', angles='xy', scale_units='xy', scale=1, width=0.003*self.sizeX*180./np.pi)
      #
      #
      plt.axis('scaled')
      ax.set_xlim(np.min(self.x.flatten())*180./np.pi, np.max(self.x.flatten())*180./np.pi)
      ax.set_ylim(np.min(self.y.flatten())*180./np.pi, np.max(self.y.flatten())*180./np.pi)
      ax.set_xlabel('$x$ [deg]')
      ax.set_ylabel('$y$ [deg]')
      #
      if save==True:
         if name is None:
            name = self.name
         print("saving plot to "+"./figures/lens_simulator/"+name+".pdf")
         fig.savefig("./figures/lens_simulator/"+name+"_arrows.pdf", bbox_inches='tight')
         fig.clf()
      else:
         plt.show()



   ###############################################################################
   # Fourier transforms, notmalized such that
   # f(k) = int dx e-ikx f(x)
   # f(x) = int dk/2pi eikx f(k)
   
   def fourier(self, data=None):
      """Fourier transforms, notmalized such that
      f(k) = int dx e-ikx f(x)
      f(x) = int dk/2pi eikx f(k)
      """
      if data is None:
         data = self.data.copy()
      # use numpy's fft
      result = np.fft.rfftn(data)
#      # use pyfftw's fft. Make sure the real-space data has type np.float128
#      result = pyfftw.interfaces.numpy_fft.rfftn((np.float128)(data))
      result *= self.dX * self.dY
      return result

   def inverseFourier(self, dataFourier=None):
      """Fourier transforms, notmalized such that
      f(k) = int dx e-ikx f(x)
      f(x) = int dk/2pi eikx f(k)
      """
      if dataFourier is None:
         dataFourier = self.dataFourier.copy()
      # use numpy's fft
      result = np.fft.irfftn(dataFourier)
#      # use pyfftw's fft. Make sure the Fourier data has type np.complex128
#      result = pyfftw.interfaces.numpy_fft.irfftn((np.complex128)(dataFourier))
      result /= self.dX * self.dY
      return result
   
   ###############################################################################
   # Measure power spectrum

   def crossPowerSpectrum(self, dataFourier1, dataFourier2, theory=[], fsCl=None, nBins=51, lRange=None, plot=False, name="test", save=False, dataLabel=None, theoryLabel=None, ylabel=None):

      # define ell bins
      ell = self.l.flatten()
      if lRange is None:
         lEdges = np.logspace(np.log10(1.), np.log10(np.max(ell)), nBins, 10.)
      else:
         lEdges = np.logspace(np.log10(lRange[0]), np.log10(lRange[-1]), nBins, 10.)
      
      # bin centers
      lCen, lEdges, binIndices = stats.binned_statistic(ell, ell, statistic='mean', bins=lEdges)
      # when bin is empty, replace lCen by a naive expectation
      lCenNaive = 0.5*(lEdges[:-1]+lEdges[1:])
      lCen[np.where(np.isnan(lCen))] = lCenNaive[np.where(np.isnan(lCen))]
      # number of modes
      Nmodes, lEdges, binIndices = stats.binned_statistic(ell, np.zeros_like(ell), statistic='count', bins=lEdges)
      Nmodes = np.nan_to_num(Nmodes)

      if np.array_equal(dataFourier1, dataFourier2):
         # power spectrum
         power = (dataFourier1 * np.conj(dataFourier2)).flatten()
         power = np.real(power)  # unnecessary in principle, but avoids binned_statistics to complain
         Cl, lEdges, binIndices = stats.binned_statistic(ell, power, statistic='mean', bins=lEdges)
         Cl = np.nan_to_num(Cl)
         # finite volume correction
         Cl /= self.sizeX*self.sizeY
         # 1sigma uncertainty on Cl
         if fsCl is None:
            sCl = Cl*np.sqrt(2)
         else:
            sCl = np.array(list(map(fsCl, lCen)))
      else:
         power1 = (dataFourier1 * np.conj(dataFourier1)).flatten()
         power1 = np.real(power1) 
         Cl1, lEdges1, binIndices1 = stats.binned_statistic(ell, power1, statistic='mean', bins=lEdges)
         Cl1 = np.nan_to_num(Cl1)
         Cl1 /= self.sizeX*self.sizeY

         power2 = (dataFourier2 * np.conj(dataFourier2)).flatten()
         power2 = np.real(power2) 
         Cl2, lEdges2, binIndices2 = stats.binned_statistic(ell, power2, statistic='mean', bins=lEdges)
         Cl2 = np.nan_to_num(Cl2)
         Cl2 /= self.sizeX*self.sizeY

         power3 = (dataFourier1 * np.conj(dataFourier2)).flatten()
         power3 = np.real(power3) 
         Cl, lEdges, binIndices = stats.binned_statistic(ell, power3, statistic='mean', bins=lEdges)
         Cl = np.nan_to_num(Cl) 
         Cl /= self.sizeX*self.sizeY
         
         if fsCl is None:
            sCl = np.sqrt(Cl**2. + Cl1 * Cl2)
         else:
            sCl = np.array(list(map(fsCl, lCen)))
         

      # In case of a cross-correlation, Cl may be negative.
      # the absolute value is then still some estimate of the error bar
      sCl = np.abs(sCl)
      sCl = divideArr(sCl,np.sqrt(Nmodes))
      sCl[np.where(np.isfinite(sCl)==False)] = 0.
      
      
      if plot:
         factor = 1. # lCen**2
         
         fig=plt.figure(0)
         ax=fig.add_subplot(111)
         #
         Ipos = np.where(Cl>=0.)
         Ineg = np.where(Cl<0.)
         ax.errorbar(lCen[Ipos], factor*Cl[Ipos], yerr=factor*sCl[Ipos], c='b', fmt='.',label=dataLabel)
         ax.errorbar(lCen[Ineg], -factor*Cl[Ineg], yerr=factor*sCl[Ineg], c='r', fmt='.')
         #
         if theory is not None:
            for f in theory:
               L = np.logspace(np.log10(1.), np.log10(np.max(ell)), 201, 10.)
               ClExpected = np.array(list(map(f, L)))
               ax.plot(L, factor*ClExpected, 'k')#, label=theoryLabel)
         #
         #ax.axhline(0.)
         ax.set_xscale('log')#, nonposx='clip')
         ax.set_yscale('log', nonposy='clip')
         ax.set_xlim(1.e1, 1.e4)
         #ax.set_ylim(1.e-5, 2.e5)
         ax.set_xlabel(r'$L$')
         ax.legend(loc=0)
         ax.set_ylabel(ylabel)
         #
         if save==True:
            if name is None:
               name = self.name
            print("saving plot to figures")
            fig.savefig("figures/"+name+".pdf",bbox_inches='tight')
            fig.clf()
         else:
            plt.show()
      
      return lCen, Cl, sCl



   def powerSpectrum(self, dataFourier=None, theory=[], fsCl=None, nBins=51, lRange=None, plot=False, name="test", save=False, dataLabel=None, theoryLabel=None, ylabel=None):
      if dataFourier is None:
         dataFourier = self.dataFourier.copy()
      return self.crossPowerSpectrum(dataFourier1=dataFourier, dataFourier2=dataFourier, theory=theory, fsCl=fsCl, nBins=nBins, lRange=lRange, plot=plot, name=name, save=save, dataLabel=dataLabel, theoryLabel=theoryLabel,ylabel=ylabel)



   def crossPowerSpectrumFiltered(self, dataFourier1, dataFourier2, norm1, norm2, theory=[], fsCl=None, nBins=51, lRange=None, plot=False, name="test", save=False, dataLabel=None, theoryLabel=None, ylabel=None):
      def f1(l):
         result = divide(1., norm1(l))
         if not np.isfinite(result):
            result = 0.
         return result

      def f2(l):
         result = divide(1., norm2(l))
         if not np.isfinite(result):
            result = 0.
         return result  

      dataFourier1_filtered = self.filterFourierIsotropic(f1, dataFourier=dataFourier1, test=False)
      dataFourier2_filtered = self.filterFourierIsotropic(f2, dataFourier=dataFourier2, test=False)
      return self.crossPowerSpectrum(dataFourier1=dataFourier1_filtered, dataFourier2=dataFourier2_filtered, theory=theory, fsCl=fsCl, nBins=nBins, lRange=lRange, plot=plot, name=name, save=save, dataLabel=dataLabel, theoryLabel=theoryLabel,ylabel=ylabel)



   def crossPowerSpectrumComp(self, dataFourier1, dataFourier2, dataFourier3, theory=[], fsCl=None, nBins=51, lRange=None, plot=False, name="test", save=False, label1=None, label2=None, ylabel=None):
      lCen, Cl, sCl = self.crossPowerSpectrum(dataFourier1=dataFourier1, dataFourier2=dataFourier3, theory=theory, fsCl=fsCl, nBins=nBins, lRange=lRange)
      lCen1, Cl1, sCl1 = self.crossPowerSpectrum(dataFourier1=dataFourier2, dataFourier2=dataFourier3, theory=theory, fsCl=fsCl, nBins=nBins, lRange=lRange)

      ell = self.l.flatten()

      if plot:
         factor = 1. # lCen**2
         
         fig=plt.figure(0,figsize=(11,8))
         ax=fig.add_subplot(211)
         ax2=fig.add_subplot(212)
         #
         Ipos = np.where(Cl>=0.)
         Ineg = np.where(Cl<0.)
         ax.errorbar(lCen[Ipos], factor*Cl[Ipos], yerr=factor*sCl[Ipos], c='b', fmt='.',label=label1)
         ax.errorbar(lCen[Ineg], -factor*Cl[Ineg], yerr=factor*sCl[Ineg], c='b', fmt='.')
         #
         Ipos1 = np.where(Cl1>=0.)
         Ineg1 = np.where(Cl1<0.)
         ax.errorbar(lCen1[Ipos1], factor*Cl1[Ipos1], yerr=factor*sCl1[Ipos1], c='r', fmt='.',label=label2)
         ax.errorbar(lCen1[Ineg1], -factor*Cl1[Ineg1], yerr=factor*sCl1[Ineg1], c='r', fmt='.')
         #
         for f in theory:
            L = np.logspace(np.log10(1.), np.log10(np.max(ell)), 201, 10.)
            ClExpected = np.array(list(map(f, L)))
            ax.plot(L, factor*ClExpected, 'k')
         #
         #ax.axhline(0.)
         ax.set_xscale('log', nonposx='clip')
         ax.set_yscale('log', nonposy='clip')
         ax2.set_xscale('log', nonposx='clip')
         ax2.set_yscale('log', nonposy='clip')
         #ax.set_xlim(1.e1, 4.e4)
         #ax.set_ylim(1.e-5, 2.e5)
         ax2.set_xlabel(r'$\ell$')
         #ax.set_ylabel(r'$\ell^2 C_\ell$')
         #
         ax.legend(loc=0)
         ax2.plot(lCen, Cl/Cl1,c='k',marker='.',label=label1+"/"+label2)
         print(Cl/Cl1)
         ax2.legend(loc=0)

         if save==True:
            if name is None:
               name = self.name
            print("saving plot to figures")
            fig.savefig("figures/test__power.png")
            fig.clf()
         else:
            plt.show()



   def binTheoryPowerSpectrum(self, fCl, nBins=17, lRange=None):
      """Bin a theory power spectrum to allow to compare it with the measured power spectrum of a map.
      """
      # define ell bins
      ell = self.l.flatten()
      if lRange is None:
         lEdges = np.logspace(np.log10(1.), np.log10(np.max(ell)), nBins, 10.)
      else:
         lEdges = np.logspace(np.log10(lRange[0]), np.log10(lRange[-1]), nBins, 10.)

      # bin centers
      lCen, lEdges, binIndices = stats.binned_statistic(ell, ell, statistic='mean', bins=lEdges)
      # when bin is empty, replace lCen by a naive expectation
      lCenNaive = 0.5*(lEdges[:-1]+lEdges[1:])
      lCen[np.where(np.isnan(lCen))] = lCenNaive[np.where(np.isnan(lCen))]

      # generate map with theory power spectrum
      clmapFourier = self.filterFourierIsotropic(fCl, dataFourier=np.ones_like(self.l), test=False)
      clmapFourier = np.real(clmapFourier.flatten())
      Cl, lEdges, binIndices = stats.binned_statistic(ell, clmapFourier, statistic='mean', bins=lEdges)
      Cl = np.nan_to_num(Cl)
      return lCen, Cl


   ###############################################################################
   # Gaussians and tests for Fourier transform conventions
   
   def genGaussian(self, meanX=0., meanY=0., sigma1d=1.):
      result = np.exp(-0.5*((self.x-meanX)**2 + (self.y-meanY)**2) / sigma1d**2)
      result /= 2.*np.pi*sigma1d**2
      return result

   def genGaussianFourier(self, meanLX=0., meanLY=0., sigma1d=1.):
      result = np.exp(-0.5*((self.lx-meanLX)**2 + (self.ly-meanLY)**2) / sigma1d**2)
      result /= 2.*np.pi*sigma1d**2
      return result

   def testFourierGaussian(self):
      """tests that the FT of a Gaussian is a Gaussian,
      with correct normalization and variance
      """
      # generate a quarter of a Gaussian
      sigma1d = self.sizeX / 10.
      self.data = self.genGaussian(sigma1d=sigma1d)
      # show it
      self.plot()

      # fourier transform it
      self.dataFourier = self.fourier()
      self.plotFourier()

      # computed expected Gaussian
      expectedFourier = self.genGaussianFourier(sigma1d=1./sigma1d)
      expectedFourier *= 2.*np.pi*(1./sigma1d)**2
      expectedFourier /= 4.   # because only one quadrant in real space

      #self.plotFourier(data=self.dataFourier/expectedFourier-1.)

      # compare along one axis
      plt.plot(self.dataFourier[0,:], 'k')
      plt.plot(expectedFourier[0,:], 'r')
      plt.show()

      # compare along other axis
      plt.plot(self.dataFourier[:,0], 'k')
      plt.plot(expectedFourier[:,0], 'r')
      plt.show()


   def testFourierCos(self):
      """tests that the FT of cos(k*x)
      peaks at the right k
      """
      # generate a quarter of a Gaussian
      ell = 100.
      self.data = np.cos(ell*self.x) + np.cos(ell*self.y)
      # show it
      self.plot()
      
      # fourier transform it
      self.dataFourier = self.fourier()
      #self.plotFourier()
   
      self.plotFourier()


   def testInverseFourier(self):
      """test that the inverse FT of the forward FT is the initial function
      """
      # generate a quarter of a Gaussian
      sigma1d = 2.
      self.data = self.genGaussian(sigma1d=sigma1d)
      # show it
      self.plot()

      # Fourier transform it
      self.dataFourier = self.fourier()
      # inverse Fourier transform it
      expectedData = self.inverseFourier()

      # compare along each axis
      plt.plot(self.y[0,:], self.data[0,:] / expectedData[0,:]-1., 'g')
      plt.plot(self.x[:,0], self.data[:,0] / expectedData[:,0]-1., 'b--')
      plt.show()

   ###############################################################################
   # generate Gaussian random field with any power spectrum

   '''
   #!!!!!!! Wrong on the axis ly=0: does not satisfy f(-l)=f(l)* there,
   # which means that the real-space map will be complex.
   #!!!!!!! Also, the modulus should not be Gaussian, but Rayleigh, and with mean non-zero!!!
   def genGRF(self, fCl, test=False):
      
      # flatten the Fourier data,
      # and put in it the value of Cl
      dataFourier = np.array(map(fCl, self.l.flatten()))
      dataFourier = np.nan_to_num(dataFourier)
      # rescale by finite volume
      dataFourier *= self.sizeX*self.sizeY
      # reshape it
      dataFourier = dataFourier.reshape(np.shape(self.l))
      
      if test:
         print(dataFourier[0,0])
         # plot Fourier power map
         self.plotFourier(dataFourier=self.l**2 * np.abs(dataFourier))
         # plot Fourier power
         plt.plot(self.l.flatten(), self.l.flatten()**2 * np.abs(dataFourier.flatten()), 'k.')
         plt.show()

      # for each ell vector,
      # generate a Gaussian random number with variance = cl
      f = lambda var: np.random.normal(0., scale=np.sqrt(var)+1.e-16) * np.exp(1j*np.random.uniform(0., 2.*np.pi))
      dataFourier = np.array(map(f, dataFourier.flatten()))
      # reshape it
      dataFourier = dataFourier.reshape(np.shape(self.l))
      
      if test:
         print(dataFourier[0,0])
         # plot Fourier map
         self.plotFourier(dataFourier=dataFourier)
         self.plotFourier(dataFourier=self.l**2 * np.abs(dataFourier)**2)
         # plot Fourier power
         plt.plot(self.l.flatten(), self.l.flatten()**2 * np.abs(dataFourier.flatten())**2, 'k.', alpha=0.1)
         plt.show()

      if test:
         print(dataFourier[0,0])
         data = self.inverseFourier()
         self.plot(data=data)
         self.plotFourier(dataFourier=dataFourier)

      return dataFourier
   '''
   
   def genGRF(self, fCl, test=False):
      
      # generate Gaussian white noise in real space
      data = np.zeros_like(self.data)
      data = np.random.normal(loc=0., scale=1./np.sqrt(self.dX*self.dY), size=len(self.x.flatten()))
      data = data.reshape(np.shape(self.x))
   
      # Fourier transform
      dataFourier = self.fourier(data)
      if test:
         # check that the power spectrum is Cl = 1
         self.powerSpectrum(dataFourier, theory=[lambda l:1.], plot=True)

      # multiply by desired power spectrum
      f = lambda l: np.sqrt(fCl(l))
      clFourier = np.array(list(map(f, self.l.flatten())))
      clFourier = np.nan_to_num(clFourier)
      clFourier = clFourier.reshape(np.shape(self.l))
      dataFourier *= clFourier
      if test:
         # check 0 mode
         print("l=0 mode is:"+dataFourier[0,0])
         # check that the power spectrum is the desired one
         self.powerSpectrum(dataFourier, theory=[fCl], plot=True)
         # show the fourier map
         self.plotFourier(dataFourier)
         # show the real space map
         data = self.inverseFourier(dataFourier)
         self.plot(data)

      return dataFourier


   def saveGRFMocks(self, fCl, nRand, directory=None, name=None):
      """create nRand GRF mock maps
      """
      if directory is None:
         directory = "./output/lens_simulator/mocks/"+self.name
      if name is None:
         name = "mock_"
      # create folder if needed
      if not os.path.exists(directory):
         os.makedirs(directory)
         
      for iRand in range(nRand):
         path = directory+"/"+name+str(iRand)+".fits"
         randomFourier = self.genGRF(fCl)
         self.saveDataFourier(randomFourier, path)

   def loadAllGRFMocks(self, nRand, directory=None, name=None):
      """DO NOT USE unless you have infinite RAM ;)
      """
      if directory is None:
         directory = "./output/lens_simulator/mocks/"+self.name
      if name is None:
         name = "mock_"

      self.mockDataFourier = {}
      for iRand in range(nRand):
         path = directory+"/"+name+str(iRand)+".fits"
         self.mockDataFourier[iRand] = self.loadDataFourier(path)


   def genCorrGRF(self, fC11, fC22, fC12, test=False):
      """Generate two correlated GRFs, with the correct auto and cross spectra.
      """
      # mao 1: generate GRF
      data1Fourier = self.genGRF(fC11, test=False)
      
      # map 2: start with part correlated with map 1
      f = lambda l: divide(fC12(l), fC11(l))
      data2Fourier = self.filterFourierIsotropic(f, dataFourier=data1Fourier, test=False)
      # map 2: add uncorrelated part
      f = lambda l: fC22(l) - divide(fC12(l)*fC12(l), fC11(l))
      data2Fourier += self.genGRF(f, test=False)
      # avoid nan and inf
      data1Fourier[np.where(np.isfinite(data1Fourier)==False)] = 0.
      data2Fourier[np.where(np.isfinite(data2Fourier)==False)] = 0.
      
      if test:
         self.powerSpectrum(data1Fourier, theory=[fC11], plot=True)
         self.powerSpectrum(data2Fourier, theory=[fC22], plot=True)
         self.crossPowerSpectrum(data1Fourier, data2Fourier, theory=[fC12], plot=True)
      
      return data1Fourier, data2Fourier
      


   ###############################################################################
   # generate map where T_ell has C_ell for modulus square,
   # and a random phase.
   # Useful to avoid sample variance
   
   def genMockIsotropicNoSampleVar(self, fCl, test=False, path=None):
      # generate map with correct modulus
      f = lambda l: np.sqrt(fCl(l))
      resultFourier = self.filterFourierIsotropic(f, dataFourier=np.ones_like(self.dataFourier), test=test)
      resultFourier *= np.sqrt(self.sizeX * self.sizeY)
   
      # generate random phases
      f = lambda lx,ly: np.exp(1j*np.random.uniform(0., 2.*np.pi))
      resultFourier = self.filterFourier(f, dataFourier=resultFourier)
      
      # keep it real ;)
      # if lx=ly=0, set to zero
      resultFourier[0,0] = 0.
      # if ly=0, make sure T[lx, 0] = T[-lx, 0]^*
      for i in range(self.nX//2+1, self.nX):
         resultFourier[i,0] = np.conj(resultFourier[self.nX-i, 0])
      
      # save if needed
      if path is not None:
         self.saveDataFourier(resultFourier, path)
      
      return resultFourier
      
   

   ###############################################################################
   # Generate Poisson white noise map

   def genPoissonWhiteNoise(self, nbar, fradioPoisson, norm=False, test=False):
      """Generate Poisson white noise.
      Returns real space map.
      nbar mean number density of objects, in 1/sr.
      if norm=False, returns a map of N (number)
      if norm=True, returns a map of delta = N/Nbar - 1
      """
      # number of objects per pixel
      Ngal = nbar * self.dX * self.dY
      if test:
         print("generate Poisson white noise")
         print(nbar+" objects per sr, i.e. "+Ngal+" objects per pixel")
      data = np.random.poisson(lam=Ngal, size=len(self.x.flatten()))
      if norm:
         data = data / Ngal - 1.
      data = data.reshape(np.shape(self.x))
      dataFourier = self.fourier(data)

      #area of single pixel
      singleA = (self.fSky * 4.*np.pi) / (self.nX*self.nY)
      #power spectrum of poisson field
      const = Ngal * singleA
      #set power spectrum to 1
      dataFourier /= np.sqrt(const)

      # multiply by desired power spectrum
      f = lambda l: np.sqrt(fradioPoisson(l))
      flFourier = np.array(list(map(f, self.l.flatten())))
      flFourier = np.nan_to_num(flFourier)
      flFourier = flFourier.reshape(np.shape(self.l))
      dataFourier *= flFourier
      
      data = self.inverseFourier(dataFourier)

      return data





   ###############################################################################
   # filter map
   
   def filterFourierIsotropic(self, fW, dataFourier=None, test=False):
      """the filter fW is assumed to be function of |ell|
      """
      if dataFourier is None:
         dataFourier = self.dataFourier.copy()
      W = np.array(list(map(fW, self.l.flatten())))
      W = W.reshape(self.l.shape)
      if test:
         self.plotFourier(dataFourier=W)
         #
         plt.plot(self.l.flatten(), W.flatten(), 'b.')
         plt.show()
      
      result = dataFourier * W
      result = np.nan_to_num(result)
      return result
   
   def filterFourier(self, fW, dataFourier=None, test=False):
      """the filter fW is assumed to be function of lx, ly
      """
      if dataFourier is None:
         dataFourier = self.dataFourier.copy()
      
      f = np.vectorize(fW)
      W = f(self.lx, self.ly)
      if test:
         self.plotFourier(dataFourier=W)
         #
         plt.plot(self.l.flatten(), W.flatten(), 'b.')
         plt.show()
      
      result = dataFourier * W
      result = np.nan_to_num(result)
      return result
   
   
   ###############################################################################
   # Matched filter and point source mask
   
   def matchedFilterIsotropic(self, fCl, fprof=None, dataFourier=None, test=False):
      """ Match-filter a surface brightness map into a flux map
      T_filt = (T_l * prof_l / C_l) / (\int d^2l/(2pi)^2 prof_l^2/C_l)
      If T(x) in Jy/sr, T_l in Jy,
      then T_filt(x) in Jy, T_filt_l in Jy*sr.
      fW = C_l is the total debeamed noise power.
      prof_l is the profile before beam convolution (i.e. 1 for point source).
      """
      if dataFourier is None:
         dataFourier = self.dataFourier.copy()
      if fprof is None:
         fprof = lambda l: 1.
         
      # filter function
      def fW(l):
         result = divide(fprof(l), fCl(l))
         if not np.isfinite(result):
            result = 0.
         return result
      # filter the map
      resultFourier = self.filterFourierIsotropic(fW, dataFourier=dataFourier, test=test)
      
      # normalization function
      def fNorm(l):
         result = divide(l/(2.*np.pi) * fprof(l)**2, fCl(l))
         if not np.isfinite(result):
            result = 0.
         return result
      # compute normalization
      normalization = integrate.quad(fNorm, self.l.min(), self.l.max(), epsabs=0., epsrel=1.e-3)[0]
      
      # normalize the filtered map
      resultFourier /= normalization
      if test:
         result = self.inverseFourier(resultFourier)
         self.plot(result)
      return resultFourier


   def pointSourceMaskMatchedFilterIsotropic(self, fCl, fluxCut, fprof=None, dataFourier=None, maskPatchRadius=None, test=False):
      """Returns the mask for point sources with flux above fluxCut.
      If T(x) in Jy/sr, T_l in Jy, Cl in Jy^2/sr, fluxCut in Jy.
      If T(x) in muK, T_l in muK*sr, Cl in (muK)^2*sr, fluxCut in muK*sr.
      prof_l is the profile before beam convolution (i.e. 1 for point source).
      The mask is 1 almost everywhere, and 0 on point sources.
      Patch is patch radius in rad, to mask around point sources
      """
      # matched-filter the map
      filteredFourier = self.matchedFilterIsotropic(fCl, fprof=fprof, dataFourier=dataFourier, test=test)
      filtered = self.inverseFourier(filteredFourier)
      
      # threshold the map
      mask = 1. * (np.abs(filtered) < fluxCut)
      
      if maskPatchRadius is not None:
         # make a Gaussian such that the fwhm is twice the maskPatchRadius
         fwhm = 2. * maskPatchRadius#5. * np.pi/(180.*60.)
         s = fwhm / np.sqrt(8.*np.log(2.))
         f = lambda l: np.exp(-0.5 * l**2 * s**2)
         gaussFourier = self.filterFourierIsotropic(f, dataFourier=np.ones_like(self.dataFourier))
         
         # normalize properly, accounting for Kronecker vs Dirac and finite pixel size
         gaussFourier /= special.erf(self.dX/(np.sqrt(8.)*s)) * special.erf(self.dY/(np.sqrt(8.)*s))
         
         # smooth the mask
         # Fourier transform 1-mask, to have zero everywhere except on the point sources
         maskFourier = self.fourier(1. - mask)
         mask = self.inverseFourier(maskFourier * gaussFourier)
         
         # threshold the smoothed mask at half-max
         mask = 1. * (mask < 0.5)   #*np.max(mask.flatten()))
   
      return mask
   
   
   
   
   ###############################################################################
   # pixel window function, Gaussian beam

   def pixelWindow(self, lx, ly, dX=None, dY=None):
      if dX is None:
         dX = self.dX
      if dY is None:
         dY = self.dY
      result = sinc(0.5 * lx * dX)
      result *= sinc(0.5 * ly * dY)
      return result

   def inversePixelWindow(self, lx, ly):
      result = 1./self.pixelWindow(lx, ly)
      if not np.isfinite(result):
         result = 0.
      return result

   def gaussianBeam(self, l, fwhm):
      """fwhm is in radians
      """
      sigma_beam = fwhm / np.sqrt(8.*np.log(2.))
      return np.exp(-0.5*l**2 * sigma_beam**2)

   def inverseBeam(self, l, fwhm):
      result = self.gaussianBeam(l, fwhm)
      result = 1./result
      if not np.isfinite(result):
         result = 0.
      return result

   def inverseBeamPixelWindow(self, lx, ly, fwhm):
      l = np.sqrt(lx**2+ly**2)
      result = self.pixelWindow(lx, ly) * self.gaussianBeam(l, fwhm)
      result = 1./result
      if not np.isfinite(result):
         result = 0.
      return result


   ###############################################################################
   
   def randomizePhases(self, dataFourier=None, test=False):
      """Generate new map with same Fourier amplitudes,
      but replacing the original phases
      with iid uniformly distributed Fourier phases.
      """
      if dataFourier is None:
         dataFourier = self.dataFourier.copy()
   
      f = lambda z: np.abs(z) * np.exp(1j*np.random.uniform(0., 2.*np.pi))
      resultFourier = np.array(list(map(f, dataFourier.flatten())))
      resultFourier = resultFourier.reshape(dataFourier.shape)
      return resultFourier
   
   ###############################################################################

#   def trispectrum(self, lmean=1000., dataFourier=None, theory=None, name="test", save=False, nBins=51, lRange=None):
#      """Collapsed 4pt function of map, from power spectrum of squared map.
#      Returns 4ptfunc = T + 2CC,
#      evaluated at (l, -l+L, l', -l'-L),
#      for various L and with l,l' \simeq lmean fixed
#      """
#      if dataFourier is None:
#         dataFourier = self.dataFourier.copy()
#   
##      # define filter function
##      lMean = 1.e3
##      sl = 2.e2
##      # defined so that int d^2l W^2 = 1?
##      # !!! normalization is wrong here, but doesn't matter for ratios
##      fW2 = lambda l: np.exp(-0.5*(l-lMean)**2/sl**2) / (2.*np.pi*sl**2)
##      fW = lambda l: np.sqrt(fW2(l))
#
#      # Window function normalized such that \int d^2l/(2\pi)^2 W(l)^2 very very close to 1
#      # to match Simone's
#      def fW(l):
#         lsigma = 50.
#         result = np.exp(-(l-lmean)**2 / (4. * lsigma**2)) )
#         result *= (2.*np.pi / lsigma**2)**0.25 / np.sqrt(lmean)
#         return result
#
#      # filter in Fourier space
#      dataFourier = self.filterFourierIsotropic(fW, dataFourier)
#      # update real space map
#      data = self.inverseFourier(dataFourier=dataFourier)
#      # square map in real space
#      data = data**2
#      # update Fourier map
#      dataFourier = self.fourier(data=data)
#   
#      # compute power spectrum of squared map
#      lCen, Cl, sCl = self.powerSpectrum(dataFourier=dataFourier, theory=theory, name=name, save=save, nBins=nBins, lRange=lRange)
#      return lCen, Cl, sCl


   def filterCollapsed4PtFunc(self, l, lMean):
      """Window function from Simone 
      normalized such that \int d^2l/(2\pi)^2 W(l)^2 very very close to 1 (but not exactly!)
      """
#      lsigma = 50.
#      lsigma = 100.
#      lsigma = 200.
#      lsigma = 300.
      lsigma = 500.
      result = np.exp(-(l-lMean)**2 / (4. * lsigma**2))
      result *= (2.*np.pi / lsigma**2)**0.25 / np.sqrt(lMean)
#      result *= (l>=lMean - 4.*lsigma) * (l<=lMean + 4.*lsigma)
#      result *= (l<0.5*self.lx.max())   # make sure there is no aliasing after map is squared
      return result

   def collapsed4PtFunc(self, lMean=1000., dataFourier=None, theory=None, name="test", save=False, nBins=51, lRange=None):
      """Almost-collapsed 4pt function of map, from power spectrum of squared map.
      Returns 4ptfunc = T + 2CC/Nmodes,
      evaluated at (l, -l+L, l', -l'-L),
      for various L and with l,l' \simeq lmean fixed.
      Nmodes is defined by the shape of the filter
      """
      if dataFourier is None:
         dataFourier = self.dataFourier.copy()

#      # define filter function
#      lMean = 1.e3
#      sl = 2.e2
#      # defined so that int d^2l W^2 = 1?
#      # !!! normalization is wrong here, but doesn't matter for ratios
#      fW2 = lambda l: np.exp(-0.5*(l-lMean)**2/sl**2) / (2.*np.pi*sl**2)
#      fW = lambda l: np.sqrt(fW2(l))

      # filter in Fourier space, to keep only l,l' \simeq lmean
      f = lambda l: self.filterCollapsed4PtFunc(l, lMean)
      dataFourier = self.filterFourierIsotropic(f, dataFourier)
      # update real space map
      data = self.inverseFourier(dataFourier=dataFourier)
      # square map in real space
      data = data**2
      # update Fourier map
      dataFourier = self.fourier(data=data)

      # compute power spectrum of squared map
      lCen, Cl, sCl = self.powerSpectrum(dataFourier=dataFourier, theory=theory, name=name, save=save, nBins=nBins, lRange=lRange)
      return lCen, Cl, sCl
   
   
   
   
   def saveTrispectrum(self, dataFourier=None, gaussDataFourier=None, path="./output/flat_map/test_", nBins=51, lRange=None):
      print("computing trispectrum")
      
      # array of the high map ells to evaluate
      lMean = np.logspace(np.log10(100.), np.log10(3.e3), 9, 10.) # 9
      
      # choose the potentially non-Gaussian map to analyze
      if dataFourier is None:
         dataFourier = self.dataFourier.copy()
      
      # analyze the potentially non-Gaussian map
      collapsed4PtFunc_NG = {}
      sCollapsed4PtFunc_NG = {}
      for ilMean in range(len(lMean)):
         # here L is the low ell, while lMean is the high ell
         L, collapsed4PtFunc_NG[ilMean], sCollapsed4PtFunc_NG[ilMean] = self.collapsed4PtFunc(lMean=lMean[ilMean], dataFourier=dataFourier, nBins=nBins, lRange=lRange)
         print("done "+str(ilMean+1)+" of "+str(len(lMean)))
      
      # analyze the Gaussian mock with the same power spectrum
      collapsed4PtFunc_G = {}
      sCollapsed4PtFunc_G = {}
      for ilMean in range(len(lMean)):
         # here L is the low ell, while lMean is the high ell
         L, collapsed4PtFunc_G[ilMean], sCollapsed4PtFunc_G[ilMean] = self.collapsed4PtFunc(lMean=lMean[ilMean], dataFourier=gaussDataFourier, nBins=nBins, lRange=lRange)
         print("done "+str(ilMean+1)+" of "+str(len(lMean)))


      # the 4pt func is 2 * C^2/Nmodes + trispectrum.
      # Infer C^2/Nmodes, C^2 and trispectrum
      C2Nmodes = np.zeros((len(L), len(lMean)))
      C2 = np.zeros((len(L), len(lMean)))
      Trispec = np.zeros((len(L), len(lMean)))
      #
      sC2Nmodes = np.zeros((len(L), len(lMean)))
      sC2 = np.zeros((len(L), len(lMean)))
      sTrispec = np.zeros((len(L), len(lMean)))
      for ilMean in range(len(lMean)):
         #
         # Number of modes included in filter around lMean
         f = lambda lnl: np.exp(lnl)**2 /(2.*np.pi) * self.filterCollapsed4PtFunc(np.exp(lnl), lMean[ilMean])**2
         #test = integrate.quad(f, np.log(1.), np.log(1.e5), epsabs=0., epsrel=1.e-3)
         #print(test[0], test[1] / test[0])
         Nmodes = integrate.quad(f, np.log(1.), np.log(1.e5), epsabs=0., epsrel=1.e-3)[0]
         Nmodes **= 2
         f = lambda lnl: np.exp(lnl)**2 / (2.*np.pi) * self.filterCollapsed4PtFunc(np.exp(lnl), lMean[ilMean])**4
         #test = integrate.quad(f, np.log(1.), np.log(1.e5), epsabs=0., epsrel=1.e-3)
         #print(test[0], test[1] / test[0])
         Nmodes /= integrate.quad(f, np.log(1.), np.log(1.e5), epsabs=0., epsrel=1.e-3)[0]
         #
         C2Nmodes[:, ilMean] = collapsed4PtFunc_G[ilMean].copy()
         sC2Nmodes[:, ilMean] = sCollapsed4PtFunc_G[ilMean].copy()
         Trispec[:, ilMean] = collapsed4PtFunc_NG[ilMean] - collapsed4PtFunc_G[ilMean]
         sTrispec[:, ilMean] = np.sqrt(2.) * sC2Nmodes[:, ilMean]
         #
         C2[:, ilMean] = C2Nmodes[:, ilMean] * Nmodes
         sC2[:, ilMean] = sC2Nmodes[:, ilMean] * Nmodes
      

      # save everything
      print("saving trispectrum to "+path)
      np.savetxt(path+"_Llow.txt", L)
      np.savetxt(path+"_lhigh.txt", lMean)
      np.savetxt(path+"_C2Nmodes.txt", C2Nmodes)
      np.savetxt(path+"_sC2Nmodes.txt", sC2Nmodes)
      np.savetxt(path+"_C2.txt", C2)
      np.savetxt(path+"_sC2.txt", sC2)
      np.savetxt(path+"_Trispec.txt", Trispec)
      np.savetxt(path+"_sTrispec.txt", sTrispec)




   def loadTrispectrum(self, path="./output/flat_map/test_"):
      # read everything
      print("loading trispectrum from "+path)
      L = np.genfromtxt(path+"_Llow.txt")
      l = np.genfromtxt(path+"_lhigh.txt")
      C2Nmodes = np.genfromtxt(path+"_C2Nmodes.txt")
      sC2Nmodes = np.genfromtxt(path+"_sC2Nmodes.txt")
      C2 = np.genfromtxt(path+"_C2.txt")
      sC2 = np.genfromtxt(path+"_sC2.txt")
      Trispec = np.genfromtxt(path+"_Trispec.txt")
      sTrispec = np.genfromtxt(path+"_sTrispec.txt")
      
      return L, l, C2Nmodes, sC2Nmodes, C2, sC2, Trispec, sTrispec
   
      
      
      

   ###############################################################################
   
   def computeGradient(self, dataFourier=None):
      """returns the Fourier maps of the 2 components of the gradient
      """
      if dataFourier is None:
         dataFourier = self.dataFourier.copy()
      dx = dataFourier * 1.j * self.lx
      dy = dataFourier * 1.j * self.ly
      return dx, dy

   def computeDivergence(self, dxFourier, dyFourier):
      divFourier = dxFourier*1.j*self.lx + dyFourier*1.j*self.ly
      return divFourier

   def computeLaplacian(self, dataFourier=None):
      if dataFourier is None:
         dataFourier = self.dataFourier.copy()
      laplacianFourier = - self.l**2 * dataFourier
      return laplacianFourier

   def phiFromKappa(self, kappaFourier):
      phiFourier = divideArr(-2. * kappaFourier, self.l**2)
      phiFourier[np.where(np.isfinite(phiFourier)==False)] = 0.
      return phiFourier

   def kappaFromPhi(self, phiFourier):
      """get kappa from phi, with:
      Delta phi = 2 kappa
      """
      kappaFourier = -0.5 * self.l**2 * phiFourier
      return kappaFourier

   def deflectionFromKappa(self, kappaFourier, test=False):
      """gives the two components of the lensing deflection
      should be called from a kappa map
      Delta phi = 2 kappa
      d = grad phi
      """
      dxFourier = divideArr(-2.j * self.lx, self.l**2) * kappaFourier
      dyFourier = divideArr(-2.j * self.ly, self.l**2) * kappaFourier
      #
      dxFourier[np.where(np.isfinite(dxFourier)==False)] = 0.
      dyFourier[np.where(np.isfinite(dyFourier)==False)] = 0.
      #
      dx = self.inverseFourier(dxFourier)
      dy = self.inverseFourier(dyFourier)
      #
      if test:
         self.plot(dx)
         self.plot(dy)
      return dx, dy

   def deflectionFromPhi(self, phiFourier):
      """gives the two components of the lensing deflection
      should be called from a phi map
      d = grad phi
      """
      dxFourier = 1j * phiMap.lx * phiFourier
      dyFourier = 1j * phiMap.ly * phiFourier
      #
      dxFourier = np.nan_to_num(dxFourier)
      dyFourier = np.nan_to_num(dyFourier)
      #
      dx = self.inverseFourier(dxFourier)
      dy = self.inverseFourier(dyFourier)
      return dx, dy
   
   ###############################################################################

   def doLensingTaylor(self, unlensed=None, kappaFourier=None, phiFourier=None, dxdy=None, order=3):
      """lenses the sky map by Taylor expansion
      should be called from the unlensed sky map
      input should be a kappa map, or a phi map, or [dx, dy],
      Convention: T(n) = T0(n-d(n)),
      which is only wrong if d is large compared to its coherence length,
      ie if d is large compared to the coherence length of the lens field
      ie if post-Born approximation is wrong
      the truth would be T(n+d(n)) = T0(n) instead of T(n) = T0(n-d(n)),
      but in practice, d~arcmin and coherence of lens field~degree, so ok!
      not OK for strong lensing, or with caustics...
      """
      if unlensed is None:
         unlensed = self.data
      unlensedFourier = self.fourier(unlensed)
      
      # get the deflection field
      if kappaFourier is not None:
         dx, dy = self.deflectionFromKappa(kappaFourier)
      elif phiFourier is not None:
         dx, dy = self.deflectionFromPhi(phiFourier)
      elif dxdyFourier is not None:
         dx, dy = dxdy
      else:
         print("error: no lensing map specified")
         return
      
      # CMB lensing convention: T(n) = T0(n-d),
      # so we get -d first
      dx = - dx
      dy = - dy

      # unlensed map
      lensed = unlensed.copy()
      # first order
      if order >= 1:
         lensed += self.inverseFourier(unlensedFourier*1j*self.lx) * dx
         lensed += self.inverseFourier(unlensedFourier*1j*self.ly) * dy
      # second order
      if order >= 2:
         lensed += 0.5 * self.inverseFourier(unlensedFourier*(1j*self.lx)**2) * dx**2
         lensed += 0.5 * self.inverseFourier(unlensedFourier*(1j*self.ly)**2) * dy**2
         lensed += self.inverseFourier(unlensedFourier*(1j*self.lx)*(1j*self.ly)) * dx*dy
      # third order
      if order >= 3:
         lensed += 1./6. * self.inverseFourier(unlensedFourier*(1j*self.lx)**3) * dx**3
         lensed += 1./6. * 3. * self.inverseFourier(unlensedFourier*(1j*self.lx)**2*(1j*self.ly)) * dx**2*dy
         lensed += 1./6. * 3. * self.inverseFourier(unlensedFourier*(1j*self.lx)*(1j*self.ly)**2) * dx*dy**2
         lensed += 1./6. * self.inverseFourier(unlensedFourier*(1j*self.ly)**3) * dy**3
      # fourth order
      if order >= 4:
         lensed += 1./24. * self.inverseFourier(unlensedFourier*(1j*self.lx)**4) * dx**4
         lensed += 1./24. * 4. * self.inverseFourier(unlensedFourier*(1j*self.lx)**3*(1j*self.ly)) * dx**3*dy
         lensed += 1./24. * 6. * self.inverseFourier(unlensedFourier*(1j*self.lx)**2*(1j*self.ly)**2) * dx**2*dy**2
         lensed += 1./24. * 4. * self.inverseFourier(unlensedFourier*(1j*self.lx)*(1j*self.ly)**3) * dx*dy**3
         lensed += 1./24. * self.inverseFourier(unlensedFourier*(1j*self.ly)**4) * dy**4
      return lensed


   def doLensing(self, unlensed=None, kappaFourier=None, phiFourier=None, dxdyFourier=None):
      """lenses the sky map by displacement and interpolation
      should be called from the unlensed sky map
      kappaMap should be a kappa map, or a phi map, or [dx, dy],
      depending on the input string
      Convention: T(n) = T0(n-d(n)),
      which is only wrong if d is large compared to its coherence length,
      ie if d is large compared to the coherence length of the lens field
      ie if post-Born approximation is wrong
      the truth would be T(n+d(n)) = T0(n) instead of T(n) = T0(n-d(n)),
      but in practice, d~arcmin and coherence of lens field~degree, so ok!
      not OK for strong lensing, or with caustics...
      """
      if unlensed is None:
         unlensed = self.data
      
      # get the deflection field
      if kappaFourier is not None:
         dx, dy = self.deflectionFromKappa(kappaFourier)
      elif phiFourier is not None:
         dx, dy = self.deflectionFromPhi(phiFourier)
      elif dxdyFourier is not None:
         dx, dy = dxdy
      else:
         print("error: no lensing map specified")
         return
      
      # displaced positions
      # CMB lensing convention: T(n) = T0(n-d),
      # ie we get the lensed map T at n by evaluating the unlensed map T0 at n0 = n-d
      x0 = self.x - dx
      y0 = self.y - dy
      # enforce periodic boundary conditions
      fx = lambda x: x - (self.sizeX+self.dX)*( (x+0.5*self.dX)//(self.sizeX+self.dX) )
      x0 = fx(x0)
      fy = lambda y: y - (self.sizeY+self.dY)*( (y+0.5*self.dY)//(self.sizeY+self.dY) )
      y0 = fy(y0)

      # interpolate the unlensed map
      #fInterp = RectBivariateSpline(self.x[:,0], self.y[0,:], unlensed, kx=1, ky=1, s=0)
      fInterp = RectBivariateSpline(self.x[:,0], self.y[0,:], unlensed, kx=3, ky=3, s=0)

      # create lensed the map
      lensed = 0. * unlensed
      for iX in range(self.nX):
         for iY in range(self.nY):
            lensed[iX, iY] = fInterp(x0[iX, iY], y0[iX, iY])

      return lensed



#   # lenses the sky map by displacement and interpolation
#   # should be called from the unlensed sky map
#   # kappaMap should be a kappa map, or a phi map, or [dx, dy],
#   # depending on the input string
#   # correctly uses T(n+d(n)) = T0(n)
#   # and NOT the incorrect T(n) = T0(n-d(n))
#   def doLensing(self, kappaMap, input="kappa", name="test"):
#   
#   
#      # get the deflection field
#      if input=="kappa":
#         dxMap, dyMap = self.deflectionFromKappa(kappaMap)
#      elif input=="phi":
#         dxMap, dyMap = self.deflectionFromPhi(kappaMap)
#      elif input=="dxdy":
#         dxMap, dyMap = kappaMap
#      
#      # displaced positions
#      # CMB lensing convention: T(n+d(n)) = T0(n)
#      xNew = self.x + dxMap.data
#      yNew = self.y + dyMap.data
#      # enforce periodic boundary conditions
#      fx = lambda x: x - (self.sizeX+self.dX)*( (x+0.5*self.dX)//(self.sizeX+self.dX) )
#      xNew = fx(xNew)
#      fy = lambda y: y - (self.sizeY+self.dY)*( (y+0.5*self.dY)//(self.sizeY+self.dY) )
#      yNew = fy(yNew)
#      
#      
#      # get (xOld, yOld) from (xNew, yNew)
#      fxOldFromNew = RectBivariateSpline(xNew[:,0], yNew[0,:], self.x, kx=1, ky=1, s=0)
#      fyOldFromNew = RectBivariateSpline(xNew[:,0], yNew[0,:], self.y, kx=1, ky=1, s=0)
#      
#      
#      # interpolate the unlensed map
#      fInterp = RectBivariateSpline(self.x[:,0], self.y[0,:], self.data, kx=1, ky=1, s=0)
#      
#      # create lensed the map
#      lensedCmbMap = self.copy()
#      lensedCmbMap.name = name
#      for iX in range(self.nX):
#         for iY in range(self.nY):
#            xOld = fxOldFromNew(self.x[iX, iY], self.y[iX, iY])
#            yOld = fyOldFromNew(self.x[iX, iY], self.y[iX, iY])
#            lensedCmbMap.data[iX, iY] = fInterp(xOld, yOld)
#
#      lensedCmbMap.dataFourier = lensedCmbMap.fourier()
#      return lensedCmbMap
   

   ###############################################################################
   ###############################################################################
   # Non-normalized quadratic estimator

   def quadEstPhiNonNorm(self, fC0, fCtot, lMin= 1., lMax=1.e5, dataFourier=None, dataFourier2=None, test=False):
      """non-normalized quadratic estimator
      fC0: ulensed power spectrum
      fCtot: lensed power spectrum + noise
      """
      if dataFourier is None:
         dataFourier = self.dataFourier.copy()
      if dataFourier2 is None:
         dataFourier2 = dataFourier.copy()

      # inverse-var weighted map
      def f(l):
         # cut off the high ells from input map
         if (l<lMin) or (l>lMax):
            return 0.
         result = 1./fCtot(l)
         if not np.isfinite(result):
            result = 0.
         return result
      iVarDataFourier = self.filterFourierIsotropic(f, dataFourier=dataFourier, test=test)
      iVarData = self.inverseFourier(iVarDataFourier)
      if test:
         print("showing the inverse var. weighted map")
         self.plot(data=iVarData)
         print("checking the power spectrum of this map")
         self.powerSpectrum(theory=f, dataFourier=iVarDataFourier, plot=True)

      # Wiener-filter the map
      def f(l):
         # cut off the high ells from input map
         if (l<lMin) or (l>lMax):
            return 0.
         result = divide(fC0(l), fCtot(l))
         if not np.isfinite(result):
            result = 0.
         return result

      WFDataFourier = self.filterFourierIsotropic(f, dataFourier=dataFourier2, test=test)
      if test:
         print("showing the WF map")
         WFData = self.inverseFourier(WFDataFourier)
         self.plot(data=WFData)
         print("checking the power spectrum of this map")
         theory = lambda l: f(l) * fC0(l)
         self.powerSpectrum(theory=theory, dataFourier=WFDataFourier, plot=True)

      # get Wiener-filtered gradient map
      WFDataXFourier, WFDataYFourier = self.computeGradient(dataFourier=WFDataFourier)
      WFDataX = self.inverseFourier(dataFourier=WFDataXFourier)
      WFDataY = self.inverseFourier(dataFourier=WFDataYFourier)
      if test:
         print("showing x gradient of WF map")
         self.plot(data=WFDataX)
         print("checking power spectrum of this map")
         theory = lambda l: 0.5*l**2 * f(l) * fC0(l)  # 0.5 is from average of cos^2
         self.powerSpectrum(theory=theory, dataFourier=WFDataXFourier, plot=True)
         print("showing y gradient of WF map")
         self.plot(data=WFDataY)
         print("checking power spectrum of this map")
         theory = lambda l: 0.5*l**2 * f(l) * fC0(l)  # 0.5 is from average of sin^2
         self.powerSpectrum(theory=theory, dataFourier=WFDataYFourier, plot=True)

      
      # product in real space
      productDataX = iVarData * WFDataX
      productDataY = iVarData * WFDataY
      productDataXFourier = self.fourier(data=productDataX)
      productDataYFourier = self.fourier(data=productDataY)
      
      # take divergence
      divergenceDataFourier = self.computeDivergence(productDataXFourier, productDataYFourier)
      if test:
         print("showing divergence map")
         divergenceData = self.inverseFourier(dataFourier=divergenceDataFourier)
         self.plot(data=divergenceData)
         print("checking the power spectrum of divergence map")
         self.powerSpectrum(dataFourier=divergenceDataFourier, plot=True)
      
      # cut off the high ells from phi map
      f = lambda l: (l<=2.*lMax)
      divergenceDataFourier = self.filterFourierIsotropic(f, dataFourier=divergenceDataFourier, test=test)
      
      return divergenceDataFourier
   

   ###############################################################################
   # Normalization for quadratic estimator, i.e. N_l^{0 phiphi}

   def computeQuadEstPhiNormalizationAna(self, fN_phi_TT, test=False):
      """the normalization is N_l^phiphi,
      obtained by evaluating the analytical calculation for the reconstruction noise
      """
      W = np.array(list(map(fN_phi_TT, self.l.flatten())))
      W = np.nan_to_num(W)
      W = W.reshape(self.l.shape)
      if test:
         self.plotFourier(dataFourier=W)
         #
         plt.plot(self.l.flatten(), W.flatten(), 'b.')
         plt.show()
      return W

   def computeQuadEstPhiNormalizationRand(self, fC0, fCtot, lMin=1., lMax=1.e5, nRand=1, path=None, dataFourier=None, test=False):
      """the normalization is N_l^phiphi,
      obtained as the inverse of the average power of the quad est on randoms,
      nRand: number of random realizations to average
      """
      if dataFourier is None:
         dataFourier = self.dataFourier.copy()

      result = np.zeros_like(self.dataFourier)
      
      for iRand in range(nRand):
#         # randomize the phases
#         randomFourier = self.randomizePhases(dataFourier=dataFourier)
         if path is not None:
            print("read mock "+str(iRand)+" from "+path+str(iRand)+".fits")
            randomFourier = self.loadDataFourier(path+str(iRand)+".fits")
         else:
            print("generate mock "+str(iRand))
            randomFourier = self.genGRF(fCtot, test=test)
         
         # get the non-normalized quad. est. for this map
         randomQuadFourier = self.quadEstPhiNonNorm(fC0, fCtot, lMin=lMin, lMax=lMax, dataFourier=randomFourier, test=test)
         # measure its power spectrum,
         # with enough bins...
         lCen, Cl, sCl = self.powerSpectrum(dataFourier=randomQuadFourier, nBins=0.1*self.nX, plot=test)
         # interpolate the inverse power spectrum
         f = UnivariateSpline(lCen, Cl, k=1, s=0)
         
         if test:
            # see if we used too many/few bins for power spectrum
            L = np.linspace(0., 8.e3, 10001)
            F = np.array(list(map(f, L)))
            plt.loglog(L, F, 'b')
            plt.loglog(L, -F, 'r')
            plt.show()
         
         # make it the normalization map
         W = np.array(list(map(f, self.l.flatten())))
         W = np.nan_to_num(1./W)
         W = W.reshape(self.l.shape)
         result += W
      result /= nRand
      
      if test:
         print("showing normalization map")
         self.plotFourier(dataFourier=result)
         print("checking power spectrum")
         self.powerSpectrum(dataFourier=result)
      
      return result



   def computeQuadEstPhiNormalizationFFT(self, fC0, fCtot, lMin=1., lMax=1.e5, test=False, cache=None):
      """the normalization is N_l^phiphi,
      Works great, and super fast.
      """
   
      # Actual calculation
      def doCalculation():
         print("Doing full calculation: computeQuadEstPhiNormalizationFFT")
         # inverse-var weighted map
         def f(l):
            if (l<lMin) or (l>lMax):
               return 0.
            result = 1./fCtot(l)
            if not np.isfinite(result):
               result = 0.
            return result
         iVarFourier = np.array(list(map(f, self.l.flatten())))
         iVarFourier = iVarFourier.reshape(self.l.shape)
         iVar = self.inverseFourier(dataFourier=iVarFourier)

         # C map
         def f(l):
            if (l<lMin) or (l>lMax):
               return 0.
            result = divide(fC0(l)**2, fCtot(l))
            if not np.isfinite(result):
               result = 0.
            return result
         CFourier = np.array(list(map(f, self.l.flatten())))
         CFourier = CFourier.reshape(self.l.shape)

         # term 1x
         term1x = self.inverseFourier(dataFourier= self.lx**2 * CFourier)
         term1x *= iVar
         term1xFourier = self.fourier(data=term1x)
         term1xFourier *= self.lx**2
         #
         # term 1y
         term1y = self.inverseFourier(dataFourier= self.ly**2 * CFourier)
         term1y *= iVar
         term1yFourier = self.fourier(data=term1y)
         term1yFourier *= self.ly**2
         #
         # term 1xy
         term1xy = self.inverseFourier(dataFourier= 2. * self.lx * self.ly * CFourier)
         term1xy *= iVar
         term1xyFourier = self.fourier(data=term1xy)
         term1xyFourier *= self.lx * self.ly
         
         if test:
            self.plotFourier(term1xFourier)
            self.plotFourier(term1yFourier)
            self.plotFourier(term1xyFourier)
            self.plotFourier(term1xFourier+term1yFourier+term1xyFourier)
         

         # WF map
         def f(l):
            if (l<lMin) or (l>lMax):
               return 0.
            result = divide(fC0(l), fCtot(l))
            # artificial factor of i such that f(-l) = f(l)*,
            # such that f(x) is real
            result *= 1.j
            if not np.isfinite(result):
               result = 0.
            return result
         WFFourier = np.array(list(map(f, self.l.flatten())))
         WFFourier = WFFourier.reshape(self.l.shape)

         # term 2
         term2_x = self.inverseFourier(dataFourier= self.lx * WFFourier)
         term2_y = self.inverseFourier(dataFourier= self.ly * WFFourier)
         #
         # term 2x
         term2x = term2_x**2
         term2xFourier = self.fourier(data=term2x)
         term2xFourier *= self.lx**2
         # minus sign to correct the artificial i**2
         term2xFourier *= -1.
         #
         # term 2y
         term2y = term2_y**2
         term2yFourier = self.fourier(data=term2y)
         term2yFourier *= self.ly**2
         # minus sign to correct the artificial i**2
         term2yFourier *= -1.
         #
         # term 2xy
         term2xy = 2. * term2_x * term2_y
         term2xyFourier = self.fourier(data=term2xy)
         term2xyFourier *= self.lx * self.ly
         # minus sign to correct the artificial i**2
         term2xyFourier *= -1.

         if test:
            self.plotFourier(term2xFourier)
            self.plotFourier(term2yFourier)
            self.plotFourier(term2xyFourier)
            self.plotFourier(term2xFourier+term2yFourier+term2xyFourier)

         # add all terms
         resultFourier = term1xFourier + term1yFourier + term1xyFourier
         resultFourier += term2xFourier + term2yFourier + term2xyFourier

         # cut off the high ells from phi normalization map
         f = lambda l: (l<=2.*lMax)
         resultFourier = self.filterFourierIsotropic(f, dataFourier=resultFourier, test=False)

         # invert
         resultFourier = 1./resultFourier
         resultFourier[np.where(np.isfinite(resultFourier)==False)] = 0.

         if test:
            plt.loglog(self.l.flatten(), resultFourier.flatten(), 'b.')
            plt.show()
      
         return resultFourier

      # Caching boiler plate
      # if no caching is desired, just compute
      if cache is None:
         resultFourier = doCalculation()
      # if caching is desired
      else:
         # if first call with caching, set up the cache dictionary
         if not hasattr(self.computeQuadEstPhiNormalizationFFT.__func__, "cache"):
            self.computeQuadEstPhiNormalizationFFT.__func__.cache = {}
         # if the calculation has been done before
         if cache in self.computeQuadEstPhiNormalizationFFT.cache:
            resultFourier = self.computeQuadEstPhiNormalizationFFT.cache[cache].copy()
         # if this calculation was not done before
         else:
            resultFourier = doCalculation()
            self.computeQuadEstPhiNormalizationFFT.cache[cache] = resultFourier.copy()


      return resultFourier


   def forecastN0Kappa(self, fC0, fCtot, fCfg=None, lMin=1., lMax=1.e5, test=False, cache=None):
      """Interpolates the result for N_L^kappa = f(L),
      to be used for forecasts on lensing reconstruction
      """
      print("computing the reconstruction noise")
      # Standard reconstruction noise
      if fCfg is None:
         n0Phi = self.computeQuadEstPhiNormalizationFFT(fC0, fCtot, lMin=lMin, lMax=lMax, test=test, cache=cache)
      # Gaussian noise contribution from the foregrounds only
      else:
         n0Phi = self.computeQuadEstPhiNormalizationFgNoiseFFT(fC0, fCtot, fCfg, lMin=lMin, lMax=lMax, test=test)
      # keep only the real part (the imag. part should be zero, but is tiny in practice)
      n0Phi = np.real(n0Phi)
      # remove the nans
      n0Phi = np.nan_to_num(n0Phi)
      # make sure every value is positive
      n0Phi = np.abs(n0Phi)
      
      # convert from phi to kappa
      n0Kappa = 0.25 * self.l**4 * n0Phi
      
      # fix the issue of the wrong ell=0 value
      # replace it by the value lx=0, ly=fundamental
      n0Kappa[0,0] = n0Kappa[0,1]
      
      # interpolate
      where = (self.l.flatten()>0.)*(self.l.flatten()<2.*lMax)
      L = self.l.flatten()[where]
      N = n0Kappa.flatten()[where]
      lnfln = interp1d(np.log(L), np.log(N), kind='linear', bounds_error=False, fill_value=np.inf)
      
      where = lnfln(np.log(L)) < np.inf
      lmin = float(L[where][0])   
            
      f = lambda l: np.exp(lnfln(np.log(np.maximum(l,lmin))))
      return f
   
   
   
   def computeQuadEstKappaNorm(self, fC0, fCtot, lMin=1., lMax=1.e5, dataFourier=None, dataFourier2=None, path=None, test=False, cache=None):
      '''Returns the normalized quadratic estimator for kappa in Fourier space,
      and saves it to file if needed.
      '''
      # non-normalized QE for phi
      resultFourier = self.quadEstPhiNonNorm(fC0, fCtot, lMin=lMin, lMax=lMax, dataFourier=dataFourier, dataFourier2=dataFourier2, test=test)
      # convert from phi to kappa
      resultFourier = self.kappaFromPhi(resultFourier)
      # compute normalization
      normalizationFourier = self.computeQuadEstPhiNormalizationFFT(fC0, fCtot, lMin=lMin, lMax=lMax, test=test, cache=cache)
      # normalized (not mean field-subtracted) QE for kappa
      resultFourier *= normalizationFourier
      # save to file if needed
      if path is not None:
         self.saveDataFourier(resultFourier, path)
      return resultFourier
   
   
   
   
   ###############################################################################
   # Gaussian N0 term from foregrounds in the CMB map
   
   
   def computeQuadEstPhiNormalizationFgNoiseFFT(self, fC0, fCtot, fCfg, lMin=1., lMax=1.e5, test=False, cache=None):
      """Computes the N^{0 fg phiphi}_L due to foreground power in the temperature map
      N^{0 fg phiphi}_L = N^{0 phiphi}_L^2 * integral.
      """
      # inverse-var weighted map
      def f(l):
         if (l<lMin) or (l>lMax):
            return 0.
         result = 1./fCtot(l)
         result *= divide(fCfg(l),fCtot(l))
         if not np.isfinite(result):
            result = 0.
         return result
      iVarFourier = np.array(list(map(f, self.l.flatten())))
      iVarFourier = iVarFourier.reshape(self.l.shape)
      iVar = self.inverseFourier(dataFourier=iVarFourier)

      # C map
      def f(l):
         if (l<lMin) or (l>lMax):
            return 0.
         result = divide(fC0(l)**2, fCtot(l))
         result *= divide(fCfg(l), fCtot(l))
         if not np.isfinite(result):
            result = 0.
         return result
      CFourier = np.array(list(map(f, self.l.flatten())))
      CFourier = CFourier.reshape(self.l.shape)

      # term 1x
      term1x = self.inverseFourier(dataFourier= self.lx**2 * CFourier)
      term1x *= iVar
      term1xFourier = self.fourier(data=term1x)
      term1xFourier *= self.lx**2
      #
      # term 1y
      term1y = self.inverseFourier(dataFourier= self.ly**2 * CFourier)
      term1y *= iVar
      term1yFourier = self.fourier(data=term1y)
      term1yFourier *= self.ly**2
      #
      # term 1xy
      term1xy = self.inverseFourier(dataFourier= 2. * self.lx * self.ly * CFourier)
      term1xy *= iVar
      term1xyFourier = self.fourier(data=term1xy)
      term1xyFourier *= self.lx * self.ly
      
      if test:
         self.plotFourier(term1xFourier)
         self.plotFourier(term1yFourier)
         self.plotFourier(term1xyFourier)
         self.plotFourier(term1xFourier+term1yFourier+term1xyFourier)
      

      # WF map
      def f(l):
         if (l<lMin) or (l>lMax):
            return 0.
         result = divide(fC0(l), fCtot(l))
         result *= divide(fCfg(l), fCtot(l))
         # artificial factor of i such that f(-l) = f(l)*,
         # such that f(x) is real
         result *= 1.j
         if not np.isfinite(result):
            result = 0.
         return result
      WFFourier = np.array(list(map(f, self.l.flatten())))
      WFFourier = WFFourier.reshape(self.l.shape)

      # term 2
      term2_x = self.inverseFourier(dataFourier= self.lx * WFFourier)
      term2_y = self.inverseFourier(dataFourier= self.ly * WFFourier)
      #
      # term 2x
      term2x = term2_x**2
      term2xFourier = self.fourier(data=term2x)
      term2xFourier *= self.lx**2
      # minus sign to correct the artificial i**2
      term2xFourier *= -1.
      #
      # term 2y
      term2y = term2_y**2
      term2yFourier = self.fourier(data=term2y)
      term2yFourier *= self.ly**2
      # minus sign to correct the artificial i**2
      term2yFourier *= -1.
      #
      # term 2xy
      term2xy = 2. * term2_x * term2_y
      term2xyFourier = self.fourier(data=term2xy)
      term2xyFourier *= self.lx * self.ly
      # minus sign to correct the artificial i**2
      term2xyFourier *= -1.

      if test:
         self.plotFourier(term2xFourier)
         self.plotFourier(term2yFourier)
         self.plotFourier(term2xyFourier)
         self.plotFourier(term2xFourier+term2yFourier+term2xyFourier)

      # add all terms
      result = term1xFourier + term1yFourier + term1xyFourier
      result += term2xFourier + term2yFourier + term2xyFourier

      # cut off the high ells from phi normalization map
      f = lambda l: (l<=2.*lMax)
      result = self.filterFourierIsotropic(f, dataFourier=result, test=False)

      # normalize by the standard N0 squared
      result *= self.computeQuadEstPhiNormalizationFFT(fC0, fCtot, lMin=lMin, lMax=lMax, test=False, cache=cache)**2
      
      # clean up
      result[np.where(np.isfinite(result)==False)] = 0.
      
      if test:
         plt.loglog(self.l.flatten(), result.flatten(), 'b.')
         plt.show()

      return result



   ###############################################################################
   # Lensing multiplicative bias from lensed foregrounds

   def computeMultBiasLensedForegrounds(self, fC0, fCtot, fCfgBias, lMin=1., lMax=1.e5, test=False, cache=None):
      """Multiplicative bias to CMB lensing
      from lensed foregrounds in the CMB map.
      fCfgBias: unlensed foreground power spectrum.
      Such that:
      <QE> = kappa_CMB + (multiplicative bias) * kappa_foreground + noise.
      """
      # inverse-var weighted map
      def f(l):
         if (l<lMin) or (l>lMax):
            return 0.
         result = 1./fCtot(l)
         if not np.isfinite(result):
            result = 0.
         return result
      iVarFourier = np.array(list(map(f, self.l.flatten())))
      iVarFourier = iVarFourier.reshape(self.l.shape)
      iVar = self.inverseFourier(dataFourier=iVarFourier)

      # C map
      def f(l):
         if (l<lMin) or (l>lMax):
            return 0.
         result = divide(fC0(l)*fCfgBias(l), fCtot(l))
         if not np.isfinite(result):
            result = 0.
         return result
      CFourier = np.array(list(map(f, self.l.flatten())))
      CFourier = CFourier.reshape(self.l.shape)

      # term 1x
      term1x = self.inverseFourier(dataFourier= self.lx**2 * CFourier)
      term1x *= iVar
      term1xFourier = self.fourier(data=term1x)
      term1xFourier *= self.lx**2
      #
      # term 1y
      term1y = self.inverseFourier(dataFourier= self.ly**2 * CFourier)
      term1y *= iVar
      term1yFourier = self.fourier(data=term1y)
      term1yFourier *= self.ly**2
      #
      # term 1xy
      term1xy = self.inverseFourier(dataFourier= 2. * self.lx * self.ly * CFourier)
      term1xy *= iVar
      term1xyFourier = self.fourier(data=term1xy)
      term1xyFourier *= self.lx * self.ly
      
      if test:
         self.plotFourier(term1xFourier)
         self.plotFourier(term1yFourier)
         self.plotFourier(term1xyFourier)
         self.plotFourier(term1xFourier+term1yFourier+term1xyFourier)
      

      # WF map
      def f(l):
         if (l<lMin) or (l>lMax):
            return 0.
         result = divide(fC0(l), fCtot(l))
         # artificial factor of i such that f(-l) = f(l)*,
         # such that f(x) is real
         result *= 1.j
         if not np.isfinite(result):
            result = 0.
         return result
      WFFourier = np.array(list(map(f, self.l.flatten())))
      WFFourier = WFFourier.reshape(self.l.shape)
      # term 2
      term2_x = self.inverseFourier(dataFourier= self.lx * WFFourier)
      term2_y = self.inverseFourier(dataFourier= self.ly * WFFourier)

      # WF map, for foreground
      def f(l):
         if (l<lMin) or (l>lMax):
            return 0.
         result = divide(fCfgBias(l), fCtot(l))
         # artificial factor of i such that f(-l) = f(l)*,
         # such that f(x) is real
         result *= 1.j
         if not np.isfinite(result):
            result = 0.
         return result
      fgWFFourier = np.array(list(map(f, self.l.flatten())))
      fgWFFourier = fgWFFourier.reshape(self.l.shape)
      # term 2
      term2_x_fg = self.inverseFourier(dataFourier= self.lx * fgWFFourier)
      term2_y_fg = self.inverseFourier(dataFourier= self.ly * fgWFFourier)
      
      
      # term 2x
      term2x = term2_x * term2_x_fg
      term2xFourier = self.fourier(data=term2x)
      term2xFourier *= self.lx**2
      # minus sign to correct the artificial i**2
      term2xFourier *= -1.
      #
      # term 2y
      term2y = term2_y * term2_y_fg
      term2yFourier = self.fourier(data=term2y)
      term2yFourier *= self.ly**2
      # minus sign to correct the artificial i**2
      term2yFourier *= -1.
      #
      # term 2xy
      term2xy = term2_x * term2_y_fg
      term2xy += term2_x_fg * term2_y
      term2xyFourier = self.fourier(data=term2xy)
      term2xyFourier *= self.lx * self.ly
      # minus sign to correct the artificial i**2
      term2xyFourier *= -1.

      if test:
         self.plotFourier(term2xFourier)
         self.plotFourier(term2yFourier)
         self.plotFourier(term2xyFourier)
         self.plotFourier(term2xFourier+term2yFourier+term2xyFourier)

      # add all terms
      result = term1xFourier + term1yFourier + term1xyFourier
      result += term2xFourier + term2yFourier + term2xyFourier

      # cut off the high ells from phi normalization map
      f = lambda l: (l<=2.*lMax)
      result = self.filterFourierIsotropic(f, dataFourier=result, test=False)

      # normalize by the standard N0
      result *= self.computeQuadEstPhiNormalizationFFT(fC0, fCtot, lMin=lMin, lMax=lMax, test=False, cache=cache)
      result[np.where(np.isfinite(result)==False)] = 0.

      if test:
         plt.loglog(self.l.flatten(), np.abs(result.flatten()), 'k.')
         plt.loglog(self.l.flatten(), np.abs(np.real(result.flatten())), 'b.')
         plt.loglog(self.l.flatten(), np.imag(result.flatten()), 'g.')
         plt.show()

      return result


   def forecastMultBiasLensedForegrounds(self, fC0, fCtot, fCfgBias, lMin=1., lMax=1.e5, test=False):
      """Interpolates the multiplicative bias to CMB lensing
      due to lensed foregrounds in the map
      """
      print("computing the multiplicative bias")
      result = self.computeMultBiasLensedForegrounds(fC0, fCtot, fCfgBias, lMin=lMin, lMax=lMax, test=test)
      # keep only the real part (the imag. part should be zero, but is tiny in practice)
      result = np.real(result)
      # remove the nans
      result = np.nan_to_num(result)

      # fix the issue of the wrong ell=0 value
      # replace it by the value lx=0, ly=fundamental
      result[0,0] = result[0,1]
      
      # interpolate, preserving the sign
      lnfln = interp1d(np.log(self.l.flatten()), np.log(np.abs(result).flatten()), kind='linear', bounds_error=False, fill_value=0.)
      signln = interp1d(np.log(self.l.flatten()), np.sign(result.flatten()), kind='linear', bounds_error=False, fill_value=0.)
      f = lambda l: np.exp(lnfln(np.log(l))) * signln(np.log(l))
      return f



   ###############################################################################
   # Converting a map trispectrum into a non-Gaussian N_L^kappa
   
   
   def computeConversionTrispecToNoisePhi(self, fC0, fCtot, lMin=1., lMax=1.e5, test=False, cache=None):
      """Computes conversion factor, such that:
      N_L^phi = Gaussian + (conversion factor) * (white trispectrum of temperature map)
      The conversion factor has units of 1/C_l^2 (omitting radians)
      The integral to compute is very similar to that for the
      Gaussian N_L^phi.
      """
      # inverse-var weighted map
      def f(l):
         if (l<lMin) or (l>lMax):
            return 0.
         result = 1./fCtot(l)
         if not np.isfinite(result):
            result = 0.
         return result
      iVarFourier = np.array(list(map(f, self.l.flatten())))
      iVarFourier = iVarFourier.reshape(self.l.shape)
      iVar = self.inverseFourier(dataFourier=iVarFourier)

      # C map
      def f(l):
         if (l<lMin) or (l>lMax):
            return 0.
         result = divide(fC0(l), fCtot(l))
         # artificial factor of i such that f(-l) = f(l)*,
         # such that f(x) is real
         result *= 1.j
         if not np.isfinite(result):
            result = 0.
         return result
      CFourier = np.array(list(map(f, self.l.flatten())))
      CFourier = CFourier.reshape(self.l.shape)

      # term x
      termx = self.inverseFourier(dataFourier= self.lx * CFourier)
      termx *= iVar
      termxFourier = self.fourier(data=termx)
      termxFourier *= self.lx
      # correct for the artificial factor of i
      termxFourier *= -1.j
      #
      # term y
      termy = self.inverseFourier(dataFourier= self.ly * CFourier)
      termy *= iVar
      termyFourier = self.fourier(data=termy)
      termyFourier *= self.ly
      # correct for the artificial factor of i
      termyFourier *= -1.j

      if test:
         self.plotFourier(termxFourier)
         self.plotFourier(termyFourier)
         self.plotFourier(termxFourier+termyFourier)

      # add all terms
      resultFourier = termxFourier + termyFourier

      # cut off the high ells from phi normalization map
      f = lambda l: (l<=2.*lMax)
      resultFourier = self.filterFourierIsotropic(f, dataFourier=resultFourier, test=False)

      if test:
         plt.loglog(self.l.flatten(), resultFourier.flatten(), 'b.')
         plt.show()
      
      # take modulus squared
      resultFourier = np.abs(resultFourier)**2

      # normalize by the squared Gaussian N0
      n0GFourier = self.computeQuadEstPhiNormalizationFFT(fC0, fCtot, lMin=lMin, lMax=lMax, test=False, cache=cache)
      n0GFourier = np.abs(n0GFourier)**2
      resultFourier *= n0GFourier

      if test:
         plt.loglog(self.l.flatten(), resultFourier.flatten(), 'b.')
         plt.show()
      
      return resultFourier

   def computeConversionTrispecToNoiseKappa(self, fC0, fCtot, lMin=1., lMax=1.e5, test=False):
      """Computes conversion factor, such that:
      N_L^kappa = Gaussian + (conversion factor) * (white trispectrum of temperature map)
      The conversion factor has units of 1/C_l^2 (omitting radians)
      The integral to compute is very similar to that for the
      Gaussian N_L^kappa.
      """
      print("computing the conversion factor")
      convPhi = self.computeConversionTrispecToNoisePhi(fC0, fCtot, lMin=lMin, lMax=lMax, test=test)
      # keep only the real part (the imag. part should be zero, but is tiny in practice)
      convPhi = np.real(convPhi)
      # remove the nans
      convPhi = np.nan_to_num(convPhi)
      # make sure every value is positive
      convPhi = np.abs(convPhi)
      
      # convert from phi to kappa
      convKappa = 0.25 * self.l**4 * convPhi
            
      # fix the issue of the wrong ell=0 value
      # replace it by the value lx=0, ly=fundamental
      convKappa[0,0] = convKappa[0,1]
      
      # interpolate
      f = interp1d(self.l.flatten(), convKappa.flatten(), kind='linear', bounds_error=False, fill_value=0.)
      return f



      

   ###############################################################################
   # Non-normalized mean field for quadratic estimator

   def computeQuadEstPhiMeanFieldNonNormRand(self, fC0, fCtot, lMin=1., lMax=1.e5, nRand=1, path=None, dataFourier=None, test=False):
      """Mean field for quadratic estimator from randoms
      """
      if dataFourier is None:
         dataFourier = self.dataFourier.copy()
      
      result = np.zeros_like(self.dataFourier)
      
      for iRand in range(nRand):
         #         # randomize the phases
         #         randomFourier = self.randomizePhases(dataFourier=dataFourier)
         if path is not None:
            print("read mock "+str(iRand)+" from "+path+str(iRand)+".fits")
            randomFourier = self.loadDataFourier(path+str(iRand)+".fits")
         else:
            print("generate mock"+str(iRand))
            randomFourier = self.genGRF(fCtot, test=test)
         
         # get the non-normalized quad. est. for this map
         randomQuadFourier = self.quadEstPhiNonNorm(fC0, fCtot, lMin=lMin, lMax=lMax, dataFourier=randomFourier, test=test)
         result += randomQuadFourier
      result /= nRand
      
      if test:
         print("showing normalization map")
         self.plotFourier(dataFourier=result)
         print("checking power spectrum")
         self.powerSpectrum(dataFourier=result)

      return result


   ###############################################################################
   # Smarter estimator for C_L^kk, smarter than the auto of the kappa QE:
   # avoids the N0 bias, and the secondary foreground biases.


   def computeQuadEstPhiInverseDataNormalizationFFT(self, fC0, fCtot, lMin=1., lMax=1.e5, dataFourier=None, test=False):
      """Analogous to the standard QE normalization 1/N_L,
      but from the data.
      Used to estimate C_L^phiphi auto from cross only,
      and to reduce secondary foreground bias.
      """
      if dataFourier is None:
         dataFourier = self.dataFourier.copy()
      
      # inverse-var weighted map
      def f(l):
         if (l<lMin) or (l>lMax):
            return 0.
         result = 1./fCtot(l)**2
         if not np.isfinite(result):
            result = 0.
         return result
      iVarFourier = np.array(list(map(f, self.l.flatten())))
      iVarFourier = iVarFourier.reshape(self.l.shape)
      # multiply by data squared modulus
      iVarFourier *= np.abs(dataFourier)**2
      iVar = self.inverseFourier(dataFourier=iVarFourier)

      # C map
      def f(l):
         if (l<lMin) or (l>lMax):
            return 0.
         result = divide(fC0(l)**2, fCtot(l)**2)
         if not np.isfinite(result):
            result = 0.
         return result
      CFourier = np.array(list(map(f, self.l.flatten())))
      CFourier = CFourier.reshape(self.l.shape)
      # multiply by data squared modulus
      CFourier *= np.abs(dataFourier)**2

      # term 1x
      term1x = self.inverseFourier(dataFourier= self.lx**2 * CFourier)
      term1x *= iVar
      term1xFourier = self.fourier(data=term1x)
      term1xFourier *= self.lx**2
      #
      # term 1y
      term1y = self.inverseFourier(dataFourier= self.ly**2 * CFourier)
      term1y *= iVar
      term1yFourier = self.fourier(data=term1y)
      term1yFourier *= self.ly**2
      #
      # term 1xy
      term1xy = self.inverseFourier(dataFourier= 2. * self.lx * self.ly * CFourier)
      term1xy *= iVar
      term1xyFourier = self.fourier(data=term1xy)
      term1xyFourier *= self.lx * self.ly
      
      if test:
         self.plotFourier(term1xFourier)
         self.plotFourier(term1yFourier)
         self.plotFourier(term1xyFourier)
         self.plotFourier(term1xFourier+term1yFourier+term1xyFourier)
      

      # WF map
      def f(l):
         if (l<lMin) or (l>lMax):
            return 0.
         result = divide(fC0(l), fCtot(l)**2)
         # artificial factor of i such that f(-l) = f(l)*,
         # such that f(x) is real
         result *= 1.j
         if not np.isfinite(result):
            result = 0.
         return result
      WFFourier = np.array(list(map(f, self.l.flatten())))
      WFFourier = WFFourier.reshape(self.l.shape)
      # multiply by data squared modulus
      WFFourier *= np.abs(dataFourier)**2

      # term 2
      term2_x = self.inverseFourier(dataFourier= self.lx * WFFourier)
      term2_y = self.inverseFourier(dataFourier= self.ly * WFFourier)
      #
      # term 2x
      term2x = term2_x**2
      term2xFourier = self.fourier(data=term2x)
      term2xFourier *= self.lx**2
      # minus sign to correct the artificial i**2
      term2xFourier *= -1.
      #
      # term 2y
      term2y = term2_y**2
      term2yFourier = self.fourier(data=term2y)
      term2yFourier *= self.ly**2
      # minus sign to correct the artificial i**2
      term2yFourier *= -1.
      #
      # term 2xy
      term2xy = 2. * term2_x * term2_y
      term2xyFourier = self.fourier(data=term2xy)
      term2xyFourier *= self.lx * self.ly
      # minus sign to correct the artificial i**2
      term2xyFourier *= -1.

      if test:
         self.plotFourier(term2xFourier)
         self.plotFourier(term2yFourier)
         self.plotFourier(term2xyFourier)
         self.plotFourier(term2xFourier+term2yFourier+term2xyFourier)

      # add all terms
      resultFourier = term1xFourier + term1yFourier + term1xyFourier
      resultFourier += term2xFourier + term2yFourier + term2xyFourier

      # cut off the high ells from phi normalization map
      f = lambda l: (l<=2.*lMax)
      resultFourier = self.filterFourierIsotropic(f, dataFourier=resultFourier, test=False)

      # invert
#      resultFourier = 1./resultFourier
      resultFourier[np.where(np.isfinite(resultFourier)==False)] = 0.

      if test:
         plt.loglog(self.l.flatten(), resultFourier.flatten(), 'b.')
         plt.show()

      return resultFourier



   def computeQuadEstKappaAutoCorrectionMap(self, fC0, fCtot, lMin=1., lMax=1.e5, dataFourier=None, path=None, test=False, cache=None):
      """This is the magic map m_L, such that when binned in L-annuli,
      it gives the correction to the kappa auto-spectrum,
      i.e. it subtracts the auto-only terms that cause the N0
      and the secondary foreground biases.
      To use: take the auto-spectrum, then subtract m_L to the auto-spectrum of the kappa QE.
      """
      # non-normalized phi inverse normalization map
      resultFourier = self.computeQuadEstPhiInverseDataNormalizationFFT(fC0, fCtot, lMin=lMin, lMax=lMax, dataFourier=dataFourier, test=test)
      # convert from phi to kappa
      resultFourier = self.kappaFromPhi(resultFourier)
      # do it again, since this is effectively a "power spectrum map"
      resultFourier = self.kappaFromPhi(resultFourier)
      # compute normalization
      normalizationFourier = self.computeQuadEstPhiNormalizationFFT(fC0, fCtot, lMin=lMin, lMax=lMax, test=test, cache=cache)
      # normalized correction for QE kappa auto-spectrum correction map
      resultFourier *= normalizationFourier**2
#!!!!!!!!! weird factor needed. I haven't figured out why
      resultFourier /= self.sizeX*self.sizeY
      # take square root, so that all you have to do is to take the power spectrum
      resultFourier = np.sqrt(np.real(resultFourier))
      # save to file if needed
      if path is not None:
         self.saveDataFourier(resultFourier, path)
      return resultFourier






   ###############################################################################
   ###############################################################################
   # Dilation-only estimator


   def quadEstPhiDilationNonNorm(self, fC0, fCtot, lMin=5.e2, lMax=3000., dataFourier=None, dataFourier2=None, test=False):
      """Non-normalized quadratic estimator for phi from dilation only
      fC0: ulensed power spectrum
      fCtot: lensed power spectrum + noise
      ell cuts are performed to remain in the regime L_phi < l_T
      """
      if dataFourier is None:
         dataFourier = self.dataFourier.copy()
      if dataFourier2 is None:
         dataFourier2 = dataFourier.copy()
      
      # cut off high ells
      f = lambda l: 1.*(l>=lMin)*(l<=lMax)
      FDataFourier = self.filterFourierIsotropic(f, dataFourier=dataFourier, test=test)
      FData = self.inverseFourier(FDataFourier)
      
      # weight function for dilation
      def fdLnl2C0dLnl(l):
         e = 0.01
         lup = l*(1.+e)
         ldown = l*(1.-e)
         result = lup**2 * fC0(lup)
         result /= ldown**2 * fC0(ldown)
         result = np.log(result) / (2.*e)
         return result
      if test:
         print("testing derivative")
         F = np.array(list(map(fdLnl2C0dLnl, self.l.flatten())))
         plt.semilogx(self.l.flatten(), F, 'b.')
         plt.show()

      # sort of Dilation Wiener-filter
      def f(l):
         # cut off the high ells from input map
         if (l<lMin) or (l>lMax):
            return 0.
         result = divide(fC0(l), fCtot(l)**2)
         result *= fdLnl2C0dLnl(l) # for isotropic dilation
         result /= 0.5
         if not np.isfinite(result):
            result = 0.
         return result
      WFDataFourier = self.filterFourierIsotropic(f, dataFourier=dataFourier2, test=test)
      WFData = self.inverseFourier(WFDataFourier)
      if test:
         print("showing the WF map")
         self.plot(data=WFData)
         print("checking the power spectrum of this map")
         theory = lambda l: f(l)**2 * fC0(l)
         self.powerSpectrum(dataFourier=WFDataFourier, theory=[theory], plot=True)

      # product in real space
      product = FData * WFData
      productFourier = self.fourier(product)
      
      # get phi from kappa
      productFourier *= divide(- 2., self.l**2)
      productFourier[np.where(np.isfinite(productFourier)==False)] = 0.
      if test:
         print("checking the power spectrum of phi map")
         self.powerSpectrum(dataFourier=productFourier, plot=True)

#      # keep only L < lMin,
#      # to always be in the regime L << l
#      f = lambda l: (l<=lMin)
#      productFourier = self.filterFourierIsotropic(f, dataFourier=productFourier, test=test)

      if test:
         print("Show real space map")
         product = self.inverseFourier(productFourier)
         self.plot(product)
      
      return productFourier
      


   def computeQuadEstPhiDilationNormalizationFFT(self, fC0, fCtot, fC0wg=None, lMin=5.e2, lMax=3.e3, test=False):
      """Multiplicative normalization for phi estimator from dilation only,
      computed with FFT.
      This normalization does not correct for the estimator's multiplicative bias.
      C0wg: a hypothetical wiggle-only or no-wiggle unlensed CMB power spectrum.
      """
      if fC0wg is None:
         fC0wg = fC0
      
      def fdLnl2C0dLnl(l):
         e = 0.01
         lup = l*(1.+e)
         ldown = l*(1.-e)
         result = lup**2 * fC0wg(lup)
         result /= ldown**2 * fC0wg(ldown)
         result = np.log(result) / (2.*e)
         return result

      def f(l):
         # cut off the high ells from input map
         if (l<lMin) or (l>lMax):
            return 0.
         result = divide(fC0wg(l), fCtot(l))
         result *= fdLnl2C0dLnl(l) # for isotropic dilation
         result = result**2
         result /= 0.5
         if not np.isfinite(result):
            result = 0.
         return result
      
      dataFourier = self.filterFourierIsotropic(f, dataFourier=np.ones_like(self.l), test=test)
      
      # compute the integral
      integral = np.sum(dataFourier)
      integral /= self.sizeX * self.sizeY
      if test:
         #check that my integral is properly normalized, by comparing to the (slower) FFT
         integralFFT = self.inverseFourier(dataFourier)
         integralFFT = integralFFT[0,0] / 2. # I think the factor 2 is about the half/full Fourier plane
         print("Integral from sum = "+str(integral))
         print("Same integral from FFT = "+str(integralFFT))
      
      # fill a Fourier map with this value
      resultFourier = np.ones_like(dataFourier)
      resultFourier /= integral

      return resultFourier


   def computeQuadEstPhiDilationNormalizationCorrectedFFT(self, fC0, fCtot, fC0wg=None, lMin=5.e2, lMax=3.e3, test=False, cache=None):
      """Multiplicative normalization for phi estimator from dilation only,
      computed with FFT.
      This normalization corrects for the multiplicative bias in the estimator.
      C0wg: a hypothetical wiggle-only or no-wiggle unlensed CMB power spectrum.
      """
      
      if fC0wg is None:
         fC0wg = fC0
      
      def doCalculation():
         print("doing full calculation: computeQuadEstPhiDilationNormalizationCorrectedFFT")
         def fdLnl2C0dLnl(l):
            e = 0.01
            lup = l*(1.+e)
            ldown = l*(1.-e)
            result = lup**2 * fC0wg(lup)
            result /= ldown**2 * fC0wg(ldown)
            result = np.log(result) / (2.*e)
            return result

         def fDilation(l):
            # cut off the high ells from input map
            if (l<lMin) or (l>lMax):
               return 0.
            result = divide(fC0wg(l), fCtot(l)**2)
            result *= fdLnl2C0dLnl(l) # for isotropic dilation
            result /= 0.5
            if not np.isfinite(result):
               result = 0.
            return result

         # generate dilation map
         dilationFourier = self.filterFourierIsotropic(fDilation, dataFourier=np.ones_like(self.l), test=test)
         dilation = self.inverseFourier(dilationFourier)
         
         # generate gradient C0 map
         f = lambda l: fC0(l) * (l>=lMin) * (l<=lMax)
         c0Fourier = self.filterFourierIsotropic(f, dataFourier=np.ones_like(self.l), test=test)
         # the factor i in the gradient makes the Fourier function Hermitian
         gradXFourier, gradYFourier = self.computeGradient(dataFourier=c0Fourier)
         gradX = self.inverseFourier(gradXFourier) # extra factor of i will be cancelled later
         gradY = self.inverseFourier(gradYFourier) # extra factor of i will be cancelled later

         # generate ell limit map
         f = lambda l: (l>=lMin) * (l<=lMax)
         ellLimitsFourier = self.filterFourierIsotropic(f, dataFourier=np.ones_like(self.l), test=test)
         ellLimits = self.inverseFourier(ellLimitsFourier)


         # First, the asymmetric term
         # term1x
         term1XFourier = self.fourier(gradX * dilation)
         term1XFourier *= divideArr(2. * self.lx, self.l**2) / 1.j # factor of i to cancel the one in the gradient
         term1XFourier[np.where(np.isfinite(term1XFourier)==False)] = 0.
         # term1y
         term1YFourier = self.fourier(gradY * dilation)
         term1YFourier *= divideArr(2. * self.ly, self.l**2) / 1.j # factor of i to cancel the one in the gradient
         term1YFourier[np.where(np.isfinite(term1YFourier)==False)] = 0.
         # sum
         term1Fourier = term1XFourier + term1YFourier
         if test:
            print("showing term1XFourier")
            self.plotFourier(term1XFourier)
            print("showing term1YFourier")
            self.plotFourier(term1YFourier)
            print("showing term1Fourier")
            self.plotFourier(term1Fourier)
         
         
         # Second, the symmetric term
         # term2x
         term2X = self.inverseFourier(gradXFourier * dilationFourier)
         term2XFourier = self.fourier(term2X * ellLimits)
         term2XFourier *= divideArr(2. * self.lx, self.l**2) / 1.j # factor of i to cancel the one in the gradient
         term2XFourier[np.where(np.isfinite(term2XFourier)==False)] = 0.
         # term2y
         term2Y = self.inverseFourier(gradYFourier * dilationFourier)
         term2YFourier = self.fourier(term2Y * ellLimits)
         term2YFourier *= divideArr(2. * self.ly, self.l**2) / 1.j # factor of i to cancel the one in the gradient
         term2YFourier[np.where(np.isfinite(term2XFourier)==False)] = 0.
         # sum
         term2Fourier = term2XFourier + term2YFourier
         if test:
            print("showing term2XFourier")
            self.plotFourier(term2XFourier)
            print("showing term2YFourier")
            self.plotFourier(term2YFourier)
            print("showing term2Fourier")
            self.plotFourier(term2Fourier)

         # sum and invert
         resultFourier = 1. / (term1Fourier + term2Fourier)
         resultFourier[np.where(np.isfinite(resultFourier)==False)] = 0.
         # remove L > 2 lMax
         f = lambda l: (l <= 2. * lMax)
         resultFourier = self.filterFourierIsotropic(f, dataFourier=resultFourier, test=test)
         if test:
            print("showing sum")
            self.plotFourier(resultFourier)
         
         return resultFourier


      # Caching boiler plate
      # if no caching is desired, just compute
      if cache is None:
         resultFourier = doCalculation()
      # if caching is desired
      else:
         # if first call with caching, set up the cache dictionary
         if not hasattr(self.computeQuadEstPhiDilationNormalizationCorrectedFFT.__func__, "cache"):
            self.computeQuadEstPhiDilationNormalizationCorrectedFFT.__func__.cache = {}
         # if the calculation has been done before
         if cache in self.computeQuadEstPhiDilationNormalizationCorrectedFFT.cache:
            resultFourier = self.computeQuadEstPhiDilationNormalizationCorrectedFFT.cache[cache].copy()
         # if this calculation was not done before
         else:
            resultFourier = doCalculation()
            self.computeQuadEstPhiDilationNormalizationCorrectedFFT.cache[cache] = resultFourier.copy()


      return resultFourier


   def computeQuadEstKappaDilationNormCorr(self, fC0, fCtot, fC0wg=None, lMin=1., lMax=1.e5, dataFourier=None, dataFourier2=None, path=None, corr=True, test=False, cache=None):
      '''Returns the dilation-only normalized quadratic estimator for kappa in Fourier space,
      and saves it to file if needed.
      The normalization includes or not the correction for the multiplicative bias.
      '''
      # non-normalized QE for phi
      resultFourier = self.quadEstPhiDilationNonNorm(fC0, fCtot, lMin=lMin, lMax=lMax, dataFourier=dataFourier, dataFourier2=dataFourier2, test=test)
      # convert from phi to kappa
      resultFourier = self.kappaFromPhi(resultFourier)
      # compute normalization (no mean field subtraction)
      if corr:
         resultFourier *= self.computeQuadEstPhiDilationNormalizationCorrectedFFT(fC0, fCtot, fC0wg=fC0wg, lMin=lMin, lMax=lMax, test=test, cache=cache)
      else:
         resultFourier *= self.computeQuadEstPhiDilationNormalizationFFT(fC0, fCtot, fC0wg=fC0wg, lMin=lMin, lMax=lMax, test=test)
      # save to file if needed
      if path is not None:
         self.saveDataFourier(resultFourier, path)
      return resultFourier
   

   def computeQuadEstKappaDilationNoiseFFT(self, fC0, fCtot, fCfg=None, fC0wg=None, lMin=1., lMax=1.e5, corr=True, test=False, cache=None):
      """Computes the lensing noise power spectrum N_L^kappa
      for the dilation estimator.
      fCfg: gives the N0 due only to the foreground component in the map.
      fC0wg: to replace consistently C0 by a wiggles-only power spectrum.
      """
      if fCfg is None:
         fCfg = fCtot
      if fC0wg is None:
         fC0wg = fC0
      
      def fdLnl2C0dLnl(l):
         e = 0.01
         lup = l*(1.+e)
         ldown = l*(1.-e)
         result = lup**2 * fC0wg(lup)
         result /= ldown**2 * fC0wg(ldown)
         result = np.log(result) / (2.*e)
         return result

      def g(l):
         # cut off the high ells from input map
         if (l<lMin) or (l>lMax):
            return 0.
         result = divide(fC0wg(l), fCtot(l)**2)
         result *= fdLnl2C0dLnl(l) # for isotropic dilation
         result /= 0.5
         if not np.isfinite(result):
            result = 0.
         return result
   
      # First, the symmetric term
      f = lambda l: g(l) * fCfg(l)
      term1Fourier = self.filterFourierIsotropic(f, dataFourier=np.ones_like(self.l), test=test)
      term1 = self.inverseFourier(term1Fourier)
      term1Fourier = self.fourier(term1**2)
      if test:
         self.plotFourier(term1Fourier)
      
      # Second, the asymmetric term
      f = lambda l: g(l)**2 * fCfg(l)
      term2aFourier = self.filterFourierIsotropic(f, dataFourier=np.ones_like(self.l), test=test)
      term2a = self.inverseFourier(term2aFourier)
      #
      f = lambda l: fCfg(l) * (l>=lMin) * (l<=lMax)
      term2bFourier = self.filterFourierIsotropic(f, dataFourier=np.ones_like(self.l), test=test)
      term2b = self.inverseFourier(term2bFourier)
      #
      term2Fourier = self.fourier(term2a * term2b)
      if test:
         self.plotFourier(term2Fourier)
      
      # add term1 and term2
      resultFourier = term1Fourier + term2Fourier
      if test:
         self.plotFourier(term1Fourier + term2Fourier)

      # compute normalization
      if corr:
         normalizationFourier = self.computeQuadEstPhiDilationNormalizationCorrectedFFT(fC0, fCtot, fC0wg=fC0wg, lMin=lMin, lMax=lMax, test=test, cache=cache)
      else:
         normalizationFourier = self.computeQuadEstPhiDilationNormalizationFFT(fC0, fCtot, fC0wg=fC0wg, lMin=lMin, lMax=lMax, test=test)

      # multiply by squared normalization
      resultFourier *= normalizationFourier**2
      
      # remove L > 2 lMax
      f = lambda l: (l <= 2. * lMax)
      resultFourier = self.filterFourierIsotropic(f, dataFourier=resultFourier, test=test)

      return resultFourier


   def forecastN0KappaDilation(self, fC0, fCtot, fCfg=None, fC0wg=None, lMin=1., lMax=1.e5, corr=True, test=False, cache=None):
      """Interpolates the result for N_L^{kappa_dilation} = f(l),
      to be used for forecasts on lensing reconstruction.
      fCfg: gives the N0 due only to the foreground component in the map.
      fC0wg: to replace consistently C0 by a wiggles-only power spectrum.
      """
      print("computing the reconstruction noise")
      n0Kappa = self.computeQuadEstKappaDilationNoiseFFT(fC0, fCtot, fCfg=fCfg, fC0wg=fC0wg, lMin=lMin, lMax=lMax, corr=corr, test=test, cache=cache)
      # keep only the real part (the imag. part should be zero, but is tiny in practice)
      n0Kappa = np.real(n0Kappa)
      # remove the nans
      n0Kappa = np.nan_to_num(n0Kappa)
      # make sure every value is positive
      n0Kappa = np.abs(n0Kappa)
      
      # fix the issue of the wrong ell=0 value
      # replace it by the value lx=0, ly=fundamental
      n0Kappa[0,0] = n0Kappa[0,1]
      
      # interpolate
      where = (self.l.flatten()>0.)*(self.l.flatten()<2.*lMax)
      L = self.l.flatten()[where]
      N = n0Kappa.flatten()[where]
      lnfln = interp1d(np.log(L), np.log(N), kind='linear', bounds_error=False, fill_value=np.inf)
#      lnfln = interp1d(np.log(self.l.flatten()), np.log(n0Kappa.flatten()), kind='linear', bounds_error=False, fill_value=0.)

      where = lnfln(np.log(L)) < np.inf
      lmin = float(L[where][0])   
            
      f = lambda l: np.exp(lnfln(np.log(np.maximum(l,lmin))))
      return f


   ###############################################################################
   # Lensing multiplicative bias from lensed foregrounds
   

   def computeMultBiasLensedForegroundsDilation(self, fC0, fCtot, fCfgBias, lMin=5.e2, lMax=3.e3, test=False, cache=None):
      """Multiplicative bias to the dilation estimator
      from lensed foregrounds in the CMB map.
      fCfgBias: unlensed foreground power spectrum.
      Such that:
      <dilation> = kappa_CMB + (multiplicative bias) * kappa_foreground + noise.
      """

      def fdLnl2C0dLnl(l):
         e = 0.01
         lup = l*(1.+e)
         ldown = l*(1.-e)
         result = lup**2 * fC0(lup)
         result /= ldown**2 * fC0(ldown)
         result = np.log(result) / (2.*e)
         return result

      def fDilation(l):
         # cut off the high ells from input map
         if (l<lMin) or (l>lMax):
            return 0.
         result = divide(fC0(l), fCtot(l)**2)
         result *= fdLnl2C0dLnl(l) # for isotropic dilation
         result /= 0.5
         if not np.isfinite(result):
            result = 0.
         return result

      # generate dilation map
      dilationFourier = self.filterFourierIsotropic(fDilation, dataFourier=np.ones_like(self.l), test=test)
      dilation = self.inverseFourier(dilationFourier)
      
      # generate gradient C0 map
      f = lambda l: fCfgBias(l) * (l>=lMin) * (l<=lMax)
      c0Fourier = self.filterFourierIsotropic(f, dataFourier=np.ones_like(self.l), test=test)
      # the factor i in the gradient makes the Fourier function Hermitian
      gradXFourier, gradYFourier = self.computeGradient(dataFourier=c0Fourier)
      gradX = self.inverseFourier(gradXFourier) # extra factor of i will be cancelled later
      gradY = self.inverseFourier(gradYFourier) # extra factor of i will be cancelled later

      # generate ell limit map
      f = lambda l: (l>=lMin) * (l<=lMax)
      ellLimitsFourier = self.filterFourierIsotropic(f, dataFourier=np.ones_like(self.l), test=test)
      ellLimits = self.inverseFourier(ellLimitsFourier)


      # First, the asymmetric term
      # term1x
      term1XFourier = self.fourier(gradX * dilation)
      term1XFourier *= divideArr(2. * self.lx, self.l**2) / 1.j # factor of i to cancel the one in the gradient
      term1XFourier[np.where(np.isfinite(term1XFourier)==False)] = 0.
      # term1y
      term1YFourier = self.fourier(gradY * dilation)
      term1YFourier *= divideArr(2. * self.ly, self.l**2) / 1.j # factor of i to cancel the one in the gradient
      term1YFourier[np.where(np.isfinite(term1YFourier)==False)] = 0.
      # sum
      term1Fourier = term1XFourier + term1YFourier
      if test:
         print("showing term1XFourier")
         self.plotFourier(term1XFourier)
         print("showing term1YFourier")
         self.plotFourier(term1YFourier)
         print("showing term1Fourier")
         self.plotFourier(term1Fourier)
      
      
      # Second, the symmetric term
      # term2x
      term2X = self.inverseFourier(gradXFourier * dilationFourier)
      term2XFourier = self.fourier(term2X * ellLimits)
      term2XFourier *= divideArr(2. * self.lx, self.l**2) / 1.j # factor of i to cancel the one in the gradient
      term2XFourier[np.where(np.isfinite(term2XFourier)==False)] = 0.
      # term2y
      term2Y = self.inverseFourier(gradYFourier * dilationFourier)
      term2YFourier = self.fourier(term2Y * ellLimits)
      term2YFourier *= divideArr(2. * self.ly, self.l**2) / 1.j # factor of i to cancel the one in the gradient
      term2YFourier[np.where(np.isfinite(term2XFourier)==False)] = 0.
      # sum
      term2Fourier = term2XFourier + term2YFourier
      if test:
         print("showing term2XFourier")
         self.plotFourier(term2XFourier)
         print("showing term2YFourier")
         self.plotFourier(term2YFourier)
         print("showing term2Fourier")
         self.plotFourier(term2Fourier)

      # sum
      resultFourier = term1Fourier + term2Fourier
      resultFourier[np.where(np.isfinite(resultFourier)==False)] = 0.
      # remove L > 2 lMax
      f = lambda l: (l <= 2. * lMax)
      resultFourier = self.filterFourierIsotropic(f, dataFourier=resultFourier, test=test)

      # normalize with the standard normalization
      resultFourier *= self.computeQuadEstPhiDilationNormalizationCorrectedFFT(fC0, fCtot, fC0wg=None, lMin=lMin, lMax=lMax, test=False, cache=cache)

      resultFourier[np.where(np.isfinite(resultFourier)==False)] = 0.
      if test:
         print("showing sum")
         self.plotFourier(resultFourier)

      return resultFourier



   def forecastMultBiasLensedForegroundsDilation(self, fC0, fCtot, fCfgBias, lMin=1., lMax=1.e5, test=False):
      """Interpolates the multiplicative bias to the dilation estimator
      due to lensed foregrounds in the map
      """
      print("computing the multiplicative bias")
      result = self.computeMultBiasLensedForegroundsDilation(fC0, fCtot, fCfgBias, lMin=lMin, lMax=lMax, test=test)
      # keep only the real part (the imag. part should be zero, but is tiny in practice)
      result = np.real(result)
      # remove the nans
      result = np.nan_to_num(result)

      # fix the issue of the wrong ell=0 value
      # replace it by the value lx=0, ly=fundamental
      result[0,0] = result[0,1]
      
      # interpolate, preserving the sign
      lnfln = interp1d(np.log(self.l.flatten()), np.log(np.abs(result).flatten()), kind='linear', bounds_error=False, fill_value=0.)
      signln = interp1d(np.log(self.l.flatten()), np.sign(result.flatten()), kind='linear', bounds_error=False, fill_value=0.)
      f = lambda l: np.exp(lnfln(np.log(l))) * signln(np.log(l))
      return f



   ###############################################################################
   ###############################################################################
   # Shear-only estimator


   def quadEstPhiShearNonNorm(self, fC0, fCtot, lMin=5.e2, lMax=3000., dataFourier=None, dataFourier2=None, test=False):
      '''Non-normalized quadratic estimator for phi from shear only
      fC0: ulensed power spectrum
      fCtot: lensed power spectrum + noise
      '''
      if dataFourier is None:
         dataFourier = self.dataFourier.copy()
      if dataFourier2 is None:
         dataFourier2 = dataFourier.copy()
      
      # cut off high ells
      f = lambda l: 1.*(l>=lMin)*(l<=lMax)
      FDataFourier = self.filterFourierIsotropic(f, dataFourier=dataFourier, test=test)
      FData = self.inverseFourier(FDataFourier)
      if test:
         print("show Fourier data")
         self.plotFourier(dataFourier=FDataFourier)
      
      # weight function for shear
      def fdLnC0dLnl(l):
         e = 0.01
         lup = l*(1.+e)
         ldown = l*(1.-e)
         result = divide(fC0(lup), fC0(ldown))
         result = np.log(result) / (2.*e)
         return result
      if test:
         F = np.array(list(map(fdLnC0dLnl, self.l.flatten())))
         plt.semilogx(self.l.flatten(), F, 'b.')
         plt.show()

      # sort of shear Wiener-filter
      def f(l):
         # cut off the high ells from input map
         if (l<lMin) or (l>lMax):
            return 0.
         result = divide(fC0(l), fCtot(l)**2)
         result *= fdLnC0dLnl(l) # for shear
         result /= 0.5
         if not np.isfinite(result):
            result = 0.
         return result
      WFDataFourier = self.filterFourierIsotropic(f, dataFourier=dataFourier2, test=test)
      if test:
         print("showing the WF map")
         self.plotFourier(dataFourier=WFDataFourier)
         print("checking the power spectrum of this map")
         theory = lambda l: f(l)**2 * fC0(l)
         self.powerSpectrum(theory=[theory], dataFourier=WFDataFourier, plot=True)
      
      # multiplication by cos 2 theta_{L,l}
      #
      # term 1
      def f(lx, ly):
         l2 = lx**2 + ly**2
         if l2==0:
            return 0.
         else:
            return divide((lx**2-ly**2), l2)
      term1Fourier = self.filterFourier(f, dataFourier=WFDataFourier, test=test)
      term1 = self.inverseFourier(dataFourier=term1Fourier)
      term1 *= FData
      term1Fourier = self.fourier(data=term1)
      term1Fourier = self.filterFourier(f, dataFourier=term1Fourier, test=test)
      #
      # term 2
      def f(lx, ly):
         l2 = lx**2 + ly**2
         if l2==0:
            return 0.
         else:
            return divide(lx * ly, l2)
      term2Fourier = self.filterFourier(f, dataFourier=WFDataFourier, test=test)
      term2 = self.inverseFourier(dataFourier=term2Fourier)
      term2 *= FData
      term2Fourier = self.fourier(data=term2)
      term2Fourier = self.filterFourier(f, dataFourier=term2Fourier, test=test)
      term2Fourier *= 4.
      #
      # add term1 and term2 to get kappa
      resultFourier = term1Fourier + term2Fourier
      #
      if test:
         print("show term 1")
         self.plotFourier(term1Fourier)
         print("show term 2")
         self.plotFourier(term2Fourier)
         print("show term 1 + term 2")
         self.plotFourier(term1Fourier + term2Fourier)
      
      # get phi from kappa
      resultFourier *= divideArr(-2. * np.ones_like(self.l), self.l**2)
      resultFourier[np.where(np.isfinite(resultFourier)==False)] = 0.
      if test:
         print("checking the power spectrum of phi map")
         self.powerSpectrum(dataFourier=resultFourier, plot=True)

      if test:
         print("Show real-space phi map")
         result = self.inverseFourier(resultFourier)
         self.plot(result)
      
      return resultFourier



   def computeQuadEstPhiShearNormalizationFFT(self, fC0, fCtot, lMin=5.e2, lMax=3.e3,  test=False):
      """Multiplicative normalization for phi estimator from shear only,
      computed with FFT
      ell cuts are performed to remain in the regime L_phi < l_T
      """

      # weight function for shear
      def fdLnC0dLnl(l):
         e = 0.01
         lup = l*(1.+e)
         ldown = l*(1.-e)
         result = divide(fC0(lup), fC0(ldown))
         result = np.log(result) / (2.*e)
         return result
      if test:
         F = np.array(list(map(fdLnC0dLnl, self.l.flatten())))
         plt.semilogx(self.l.flatten(), F, 'b.')
         plt.show()

      # sort of shear Wiener-filter
      def f(l):
         # cut off the high ells from input map
         if l<lMin or l>lMax:
            return 0.
         result = divide(fC0(l), fCtot(l))
         result *= fdLnC0dLnl(l) # for shear
         result **= 2.
         result /= 0.5
         if not np.isfinite(result):
            result = 0.
         return result
      WFDataFourier = self.filterFourierIsotropic(f, dataFourier=np.ones_like(self.l), test=test)
      if test:
         print("showing the WF map")
         self.plotFourier(dataFourier=WFDataFourier)

      # mean value of cos^2 is 1/2
      integral = np.sum(0.5 * WFDataFourier)
      integral /= self.sizeX * self.sizeY
      if test:
         #check that my integral is properly normalized, by comparing to the (slower) FFT
         integralFFT = self.inverseFourier(WFDataFourier)
         integralFFT = integralFFT[0,0] / 2. # I think the factor 2 is about the half/full Fourier plane
         print("Integral from sum = "+str(integral))
         print("Same integral from FFT = "+str(integralFFT))
      
      # fill a Fourier map with this value
      resultFourier = np.ones_like(WFDataFourier)
      resultFourier /= integral

      return resultFourier


   def computeQuadEstPhiShearNormalizationCorrectedFFT(self, fC0, fCtot, lMin=5.e2, lMax=3.e3, test=False, cache=None):
      """Multiplicative normalization for phi estimator from shear only,
      computed with FFT.
      This normalization corrects for the multiplicative bias in the estimator.
      ell cuts are performed to remain in the regime L_phi < l_T
      """
      
      def doCalculation():
         print("doing full calculation: computeQuadEstPhiShearNormalizationCorrectedFFT")
         # weight function for shear
         def fdLnC0dLnl(l):
            e = 0.01
            lup = l*(1.+e)
            ldown = l*(1.-e)
            result = divide(fC0(lup), fC0(ldown))
            result = np.log(result) / (2.*e)
            return result

         def f(l):
            # cut off the high ells from input map
            if (l<lMin) or (l>lMax):
               return 0.
            result = divide(fC0(l), fCtot(l)**2)
            result *= fdLnC0dLnl(l) # for shear
            result /= 0.5
            if not np.isfinite(result):
               result = 0.
            return result

         # generate shear maps
         shearFourier = self.filterFourierIsotropic(f, dataFourier=np.ones_like(self.l), test=test)
         #
         cosXFourier = divideArr(shearFourier * (self.lx**2 - self.ly**2), self.l**2)
         cosXFourier[np.where(np.isfinite(cosXFourier)==False)] = 0.
         cosX = self.inverseFourier(cosXFourier)
         #
         cosYFourier = divideArr(shearFourier * self.lx * self.ly, self.l**2)
         cosYFourier[np.where(np.isfinite(cosYFourier)==False)] = 0.
         cosY = self.inverseFourier(cosYFourier)

         # generate gradient C0 map
         f = lambda l: fC0(l) * (l>=lMin) * (l<=lMax)
         c0Fourier = self.filterFourierIsotropic(f, dataFourier=np.ones_like(self.l), test=test)
         # the factor i in the gradient makes the Fourier function Hermitian
         gradXFourier, gradYFourier = self.computeGradient(dataFourier=c0Fourier)
         gradX = self.inverseFourier(gradXFourier) # extra factor of i will be cancelled later
         gradY = self.inverseFourier(gradYFourier) # extra factor of i will be cancelled later

         # generate ell limit map
         f = lambda l: (l>=lMin) * (l<=lMax)
         ellLimitsFourier = self.filterFourierIsotropic(f, dataFourier=np.ones_like(self.l), test=test)
         ellLimits = self.inverseFourier(ellLimitsFourier)


         # Term 1
         grad1XcosXFourier = self.fourier(gradX * cosX)
         grad1XcosXFourier *= (self.lx**2 - self.ly**2)   # for cos
         grad1XcosXFourier *= divideArr(2. * self.lx, self.l**4) / 1.j # for grad
         grad1XcosXFourier[np.where(np.isfinite(grad1XcosXFourier)==False)] = 0.
         #
         grad1YcosXFourier = self.fourier(gradY * cosX)
         grad1YcosXFourier *= (self.lx**2 - self.ly**2)   # for cos
         grad1YcosXFourier *= divideArr(2. * self.ly, self.l**4) / 1.j # for grad
         grad1YcosXFourier[np.where(np.isfinite(grad1YcosXFourier)==False)] = 0.
         #
         grad1XcosYFourier = self.fourier(gradX * cosY)
         grad1XcosYFourier *= 4. * self.lx * self.ly   # for cos
         grad1XcosYFourier *= divideArr(2. * self.lx, self.l**4) / 1.j # for grad
         grad1XcosYFourier[np.where(np.isfinite(grad1XcosYFourier)==False)] = 0.
         #
         grad1YcosYFourier = self.fourier(gradY * cosY)
         grad1YcosYFourier *= 4. * self.lx * self.ly   # for cos
         grad1YcosYFourier *= divideArr(2. * self.ly, self.l**4) / 1.j # for grad
         grad1YcosYFourier[np.where(np.isfinite(grad1YcosYFourier)==False)] = 0.
         #
         # sum
         term1Fourier = grad1XcosXFourier + grad1YcosXFourier + grad1XcosYFourier + grad1YcosYFourier
         if test:
            print("showing various terms")
            self.plotFourier(grad1XcosXFourier)
            self.plotFourier(grad1YcosXFourier)
            self.plotFourier(grad1XcosYFourier)
            self.plotFourier(grad1YcosYFourier)
            self.plotFourier(term1Fourier)


         # Term 2
         grad2XcosX = self.inverseFourier(gradXFourier * cosXFourier)
         grad2XcosXFourier = self.fourier(grad2XcosX * ellLimits)
         grad2XcosXFourier *= (self.lx**2 - self.ly**2)   # for cos
         grad2XcosXFourier *= divideArr(2. * self.lx, self.l**4) / 1.j # for grad
         grad2XcosXFourier[np.where(np.isfinite(grad2XcosXFourier)==False)] = 0.
         #
         grad2YcosX = self.inverseFourier(gradYFourier * cosXFourier)
         grad2YcosXFourier = self.fourier(grad2YcosX * ellLimits)
         grad2YcosXFourier *= (self.lx**2 - self.ly**2)   # for cos
         grad2YcosXFourier *= divideArr(2. * self.ly, self.l**4) / 1.j # for grad
         grad2YcosXFourier[np.where(np.isfinite(grad2YcosXFourier)==False)] = 0.
         #
         grad2XcosY = self.inverseFourier(gradXFourier * cosYFourier)
         grad2XcosYFourier = self.fourier(grad2XcosY * ellLimits)
         grad2XcosYFourier *= 4. * self.lx * self.ly   # for cos
         grad2XcosYFourier *= divideArr(2. * self.lx, self.l**4) / 1.j # for grad
         grad2XcosYFourier[np.where(np.isfinite(grad2XcosYFourier)==False)] = 0.
         #
         grad2YcosY = self.inverseFourier(gradYFourier * cosYFourier)
         grad2YcosYFourier = self.fourier(grad2YcosY * ellLimits)
         grad2YcosYFourier *= 4. * self.lx * self.ly   # for cos
         grad2YcosYFourier *= divideArr(2. * self.ly, self.l**4) / 1.j # for grad
         grad2YcosYFourier[np.where(np.isfinite(grad2YcosXFourier)==False)] = 0.
         #
         # sum
         term2Fourier = grad2XcosXFourier + grad2YcosXFourier + grad2XcosYFourier + grad2YcosYFourier
         if test:
            print("showing various terms")
            self.plotFourier(grad2XcosXFourier)
            self.plotFourier(grad2YcosXFourier)
            self.plotFourier(grad2XcosYFourier)
            self.plotFourier(grad2YcosYFourier)
            self.plotFourier(term2Fourier)


         # sum and invert
         resultFourier = 1. / (term1Fourier + term2Fourier)
         resultFourier[np.where(np.isfinite(resultFourier)==False)] = 0.
         
         # remove L > 2 lMax
         f = lambda l: (l<=2.*lMax)
         resultFourier = self.filterFourierIsotropic(f, dataFourier=resultFourier, test=test)
         if test:
            print("showing result")
            self.plotFourier(resultFourier)
         return resultFourier


      # Caching boiler plate
      # if no caching is desired, just compute
      if cache is None:
         resultFourier = doCalculation()
      # if caching is desired
      else:
         # if first call with caching, set up the cache dictionary
         if not hasattr(self.computeQuadEstPhiShearNormalizationCorrectedFFT.__func__, "cache"):
            self.computeQuadEstPhiShearNormalizationCorrectedFFT.__func__.cache = {}
         # if the calculation has been done before
         if cache in self.computeQuadEstPhiShearNormalizationCorrectedFFT.cache:
            resultFourier = self.computeQuadEstPhiShearNormalizationCorrectedFFT.cache[cache].copy()
         # if this calculation was not done before
         else:
            resultFourier = doCalculation()
            self.computeQuadEstPhiShearNormalizationCorrectedFFT.cache[cache] = resultFourier.copy()


      return resultFourier


   def computeQuadEstKappaShearNormCorr(self, fC0, fCtot, lMin=1., lMax=1.e5, dataFourier=None, dataFourier2=None, path=None, corr=True, test=False, cache=None):
      '''Returns the shear-only normalized quadratic estimator for kappa in Fourier space,
      and saves it to file if needed.
      The normalization includes or not the correction for the multiplicative bias.
      '''
      # non-normalized QE for phi
      resultFourier = self.quadEstPhiShearNonNorm(fC0, fCtot, lMin=lMin, lMax=lMax, dataFourier=dataFourier, dataFourier2=dataFourier2, test=test)
      # convert from phi to kappa
      resultFourier = self.kappaFromPhi(resultFourier)
      # compute normalization
      if corr:
         normalizationFourier = self.computeQuadEstPhiShearNormalizationCorrectedFFT(fC0, fCtot, lMin=lMin, lMax=lMax, test=test, cache=cache)
      else:
         normalizationFourier = self.computeQuadEstPhiShearNormalizationFFT(fC0, fCtot, lMin=lMin, lMax=lMax, test=test)
      # normalized (not mean field-subtracted) QE for kappa
      resultFourier *= normalizationFourier
      # save to file if needed
      if path is not None:
         self.saveDataFourier(resultFourier, path)
      return resultFourier




   def computeQuadEstKappaShearNoiseFFT(self, fC0, fCtot, fCfg=None, lMin=1., lMax=1.e5, corr=True, test=False, cache=None):
      """Computes the lensing noise power spectrum N_L^kappa
      for the shear estimator.
      """
      if fCfg is None:
         fCfg = fCtot
      
      def fdLnC0dLnl(l):
         e = 0.01
         lup = l*(1.+e)
         ldown = l*(1.-e)
         result = divide(fC0(lup), fC0(ldown))
         result = np.log(result) / (2.*e)
         return result

      def g(l):
         # cut off the high ells from input map
         if (l<lMin) or (l>lMax):
            return 0.
         result = divide(fC0(l), fCtot(l)**2)
         result *= fdLnC0dLnl(l) # for isotropic dilation
         result /= 0.5
         if not np.isfinite(result):
            result = 0.
         return result
      
      # useful terms
      fx = divideArr(self.lx**2-self.ly**2, self.l**2)
      fx[np.where(np.isfinite(fx)==False)] = 0.
      #
      fy = divideArr(self.lx*self.ly, self.l**2)
      fy[np.where(np.isfinite(fy)==False)] = 0.
   
      # First, the symmetric term
      f = lambda l: g(l) * fCfg(l)
      term1Fourier = self.filterFourierIsotropic(f, dataFourier=np.ones_like(self.l), test=test)
      #
      term1xFourier = term1Fourier * fx
      term1x = self.inverseFourier(term1xFourier)
      #
      term1yFourier = term1Fourier * fy
      term1y = self.inverseFourier(term1yFourier)
      
      term1xxFourier = self.fourier(term1x**2)
      term1xxFourier *= fx**2
      #
      term1yyFourier = self.fourier(term1y**2)
      term1yyFourier *= (4.*fy)**2
      #
      term1xyFourier = self.fourier(term1x*term1y)
      term1xyFourier *= fx * 4. * fy
      term1xyFourier *= 2.
      #
      term1Fourier = term1xxFourier + term1yyFourier + term1xyFourier
      if test:
         self.plotFourier(term1xxFourier)
         self.plotFourier(term1yyFourier)
         self.plotFourier(term1xyFourier)
         self.plotFourier(term1Fourier)


      # Second, the asymmetric term
      f = lambda l: fCfg(l) * (l>=lMin) * (l<=lMax)
      term2bFourier = self.filterFourierIsotropic(f, dataFourier=np.ones_like(self.l), test=test)
      term2b = self.inverseFourier(term2bFourier)
      #
      f = lambda l: g(l)**2 * fCfg(l)
      term2aFourier = self.filterFourierIsotropic(f, dataFourier=np.ones_like(self.l), test=test)

      term2xxFourier = term2aFourier * fx**2
      term2xx = self.inverseFourier(term2xxFourier)
      term2xxFourier = self.fourier(term2xx * term2b)
      term2xxFourier *= fx**2
      #
      term2yyFourier = term2aFourier * fy**2
      term2yy = self.inverseFourier(term2yyFourier)
      term2yyFourier = self.fourier(term2yy * term2b)
      term2yyFourier *= (4.*fy)**2
      #
      term2xyFourier = term2aFourier * fx * fy
      term2xy = self.inverseFourier(term2xyFourier)
      term2xyFourier = self.fourier(term2xy * term2b)
      term2xyFourier *= (fx * 4.*fy)
      term2xyFourier *= 2.
      #
      term2Fourier = term2xxFourier + term2yyFourier + term2xyFourier
      if test:
         self.plotFourier(term2xxFourier)
         self.plotFourier(term2yyFourier)
         self.plotFourier(term2xyFourier)
         self.plotFourier(term2Fourier)

      # add terms
      resultFourier = term1Fourier + term2Fourier

      # compute normalization
      if corr:
         normalizationFourier = self.computeQuadEstPhiShearNormalizationCorrectedFFT(fC0, fCtot, lMin=lMin, lMax=lMax, test=test, cache=cache)
      else:
         normalizationFourier = self.computeQuadEstPhiShearNormalizationFFT(fC0, fCtot, lMin=lMin, lMax=lMax, test=test)

      # divide by squared normalization
      resultFourier *= normalizationFourier**2

      # remove L > 2 lMax
      f = lambda l: (l <= 2. * lMax)
      resultFourier = self.filterFourierIsotropic(f, dataFourier=resultFourier, test=test)

      return resultFourier


   def forecastN0KappaShear(self, fC0, fCtot, fCfg=None, lMin=1., lMax=1.e5, corr=True, test=False, cache=None):
      """Interpolates the result for N_L^{kappa_shear} = f(l),
      to be used for forecasts on lensing reconstruction.
      """
      print("computing the reconstruction noise")
      n0Kappa = self.computeQuadEstKappaShearNoiseFFT(fC0, fCtot, fCfg=fCfg, lMin=lMin, lMax=lMax, corr=corr, test=test, cache=cache)
      # keep only the real part (the imag. part should be zero, but is tiny in practice)
      n0Kappa = np.real(n0Kappa)
      # remove the nans
      n0Kappa = np.nan_to_num(n0Kappa)
      # make sure every value is positive
      n0Kappa = np.abs(n0Kappa)
      
      # fix the issue of the wrong ell=0 value
      # replace it by the value lx=0, ly=fundamental
      n0Kappa[0,0] = n0Kappa[0,1]
      
      # interpolate
      where = (self.l.flatten()>0.)*(self.l.flatten()<2.*lMax)
      L = self.l.flatten()[where]
      N = n0Kappa.flatten()[where]
      lnfln = interp1d(np.log(L), np.log(N), kind='linear', bounds_error=False, fill_value=np.inf)
#      lnfln = interp1d(np.log(self.l.flatten()), np.log(n0Kappa.flatten()), kind='linear', bounds_error=False, fill_value=0.)

      where = lnfln(np.log(L)) < np.inf
      lmin = float(L[where][0])   
            
      f = lambda l: np.exp(lnfln(np.log(np.maximum(l,lmin))))
      return f


   ###############################################################################
   # Lensing multiplicative bias from lensed foregrounds
   

   def computeMultBiasLensedForegroundsShear(self, fC0, fCtot, fCfgBias, lMin=5.e2, lMax=3.e3, test=False, cache=None):
      """Multiplicative bias to the shear estimator
      from lensed foregrounds in the CMB map.
      fCfgBias: unlensed foreground power spectrum.
      Such that:
      <shear> = kappa_CMB + (multiplicative bias) * kappa_foreground + noise.
      """
      
      # weight function for shear
      def fdLnC0dLnl(l):
         e = 0.01
         lup = l*(1.+e)
         ldown = l*(1.-e)
         result = divide(fC0(lup), fC0(ldown))
         result = np.log(result) / (2.*e)
         return result

      def f(l):
         # cut off the high ells from input map
         if (l<lMin) or (l>lMax):
            return 0.
         result = divide(fC0(l), fCtot(l)**2)
         result *= fdLnC0dLnl(l) # for shear
         result /= 0.5
         if not np.isfinite(result):
            result = 0.
         return result

      # generate shear maps
      shearFourier = self.filterFourierIsotropic(f, dataFourier=np.ones_like(self.l), test=test)
      #
      cosXFourier = divideArr(shearFourier * (self.lx**2 - self.ly**2), self.l**2)
      cosXFourier[np.where(np.isfinite(cosXFourier)==False)] = 0.
      cosX = self.inverseFourier(cosXFourier)
      #
      cosYFourier = divideArr(shearFourier * self.lx * self.ly, self.l**2)
      cosYFourier[np.where(np.isfinite(cosYFourier)==False)] = 0.
      cosY = self.inverseFourier(cosYFourier)

      # generate gradient C0 map
      f = lambda l: fCfgBias(l) * (l>=lMin) * (l<=lMax)
      c0Fourier = self.filterFourierIsotropic(f, dataFourier=np.ones_like(self.l), test=test)
      # the factor i in the gradient makes the Fourier function Hermitian
      gradXFourier, gradYFourier = self.computeGradient(dataFourier=c0Fourier)
      gradX = self.inverseFourier(gradXFourier) # extra factor of i will be cancelled later
      gradY = self.inverseFourier(gradYFourier) # extra factor of i will be cancelled later

      # generate ell limit map
      f = lambda l: (l>=lMin) * (l<=lMax)
      ellLimitsFourier = self.filterFourierIsotropic(f, dataFourier=np.ones_like(self.l), test=test)
      ellLimits = self.inverseFourier(ellLimitsFourier)


      # Term 1
      grad1XcosXFourier = self.fourier(gradX * cosX)
      grad1XcosXFourier *= (self.lx**2 - self.ly**2)   # for cos
      grad1XcosXFourier *= divideArr(2. * self.lx, self.l**4) / 1.j # for grad
      grad1XcosXFourier[np.where(np.isfinite(grad1XcosXFourier)==False)] = 0.
      #
      grad1YcosXFourier = self.fourier(gradY * cosX)
      grad1YcosXFourier *= (self.lx**2 - self.ly**2)   # for cos
      grad1YcosXFourier *= divideArr(2. * self.ly, self.l**4) / 1.j # for grad
      grad1YcosXFourier[np.where(np.isfinite(grad1YcosXFourier)==False)] = 0.
      #
      grad1XcosYFourier = self.fourier(gradX * cosY)
      grad1XcosYFourier *= 4. * self.lx * self.ly   # for cos
      grad1XcosYFourier *= divideArr(2. * self.lx, self.l**4) / 1.j # for grad
      grad1XcosYFourier[np.where(np.isfinite(grad1XcosYFourier)==False)] = 0.
      #
      grad1YcosYFourier = self.fourier(gradY * cosY)
      grad1YcosYFourier *= 4. * self.lx * self.ly   # for cos
      grad1YcosYFourier *= divideArr(2. * self.ly, self.l**4) / 1.j # for grad
      grad1YcosYFourier[np.where(np.isfinite(grad1YcosYFourier)==False)] = 0.
      #
      # sum
      term1Fourier = grad1XcosXFourier + grad1YcosXFourier + grad1XcosYFourier + grad1YcosYFourier
      if test:
         print("showing various terms")
         self.plotFourier(grad1XcosXFourier)
         self.plotFourier(grad1YcosXFourier)
         self.plotFourier(grad1XcosYFourier)
         self.plotFourier(grad1YcosYFourier)
         self.plotFourier(term1Fourier)


      # Term 2
      grad2XcosX = self.inverseFourier(gradXFourier * cosXFourier)
      grad2XcosXFourier = self.fourier(grad2XcosX * ellLimits)
      grad2XcosXFourier *= (self.lx**2 - self.ly**2)   # for cos
      grad2XcosXFourier *= divideArr(2. * self.lx, self.l**4) / 1.j # for grad
      grad2XcosXFourier[np.where(np.isfinite(grad2XcosXFourier)==False)] = 0.
      #
      grad2YcosX = self.inverseFourier(gradYFourier * cosXFourier)
      grad2YcosXFourier = self.fourier(grad2YcosX * ellLimits)
      grad2YcosXFourier *= (self.lx**2 - self.ly**2)   # for cos
      grad2YcosXFourier *= divideArr(2. * self.ly, self.l**4) / 1.j # for grad
      grad2YcosXFourier[np.where(np.isfinite(grad2YcosXFourier)==False)] = 0.
      #
      grad2XcosY = self.inverseFourier(gradXFourier * cosYFourier)
      grad2XcosYFourier = self.fourier(grad2XcosY * ellLimits)
      grad2XcosYFourier *= 4. * self.lx * self.ly   # for cos
      grad2XcosYFourier *= divideArr(2. * self.lx, self.l**4) / 1.j # for grad
      grad2XcosYFourier[np.where(np.isfinite(grad2XcosYFourier)==False)] = 0.
      #
      grad2YcosY = self.inverseFourier(gradYFourier * cosYFourier)
      grad2YcosYFourier = self.fourier(grad2YcosY * ellLimits)
      grad2YcosYFourier *= 4. * self.lx * self.ly   # for cos
      grad2YcosYFourier *= divideArr(2. * self.ly, self.l**4) / 1.j # for grad
      grad2YcosYFourier[np.where(np.isfinite(grad2YcosXFourier)==False)] = 0.
      #
      # sum
      term2Fourier = grad2XcosXFourier + grad2YcosXFourier + grad2XcosYFourier + grad2YcosYFourier
      if test:
         print("showing various terms")
         self.plotFourier(grad2XcosXFourier)
         self.plotFourier(grad2YcosXFourier)
         self.plotFourier(grad2XcosYFourier)
         self.plotFourier(grad2YcosYFourier)
         self.plotFourier(term2Fourier)


      # sum
      resultFourier = term1Fourier + term2Fourier
      resultFourier[np.where(np.isfinite(resultFourier)==False)] = 0.
      
      # remove L > 2 lMax
      f = lambda l: (l<=2.*lMax)
      resultFourier = self.filterFourierIsotropic(f, dataFourier=resultFourier, test=test)

      # normalize with the standard normalization
      resultFourier *= self.computeQuadEstPhiShearNormalizationCorrectedFFT(fC0, fCtot, lMin=lMin, lMax=lMax, test=False, cache=cache)
      if test:
         print("showing result")
         self.plotFourier(resultFourier)

      return resultFourier

   

   def forecastMultBiasLensedForegroundsShear(self, fC0, fCtot, fCfgBias, lMin=1., lMax=1.e5, test=False):
      """Interpolates the multiplicative bias to the shear estimator
      due to lensed foregrounds in the map
      """
      print("computing the multiplicative bias")
      result = self.computeMultBiasLensedForegroundsShear(fC0, fCtot, fCfgBias, lMin=lMin, lMax=lMax, test=test)
      # keep only the real part (the imag. part should be zero, but is tiny in practice)
      result = np.real(result)
      # remove the nans
      result = np.nan_to_num(result)

      # fix the issue of the wrong ell=0 value
      # replace it by the value lx=0, ly=fundamental
      result[0,0] = result[0,1]
      
      # interpolate, preserving the sign
      lnfln = interp1d(np.log(self.l.flatten()), np.log(np.abs(result).flatten()), kind='linear', bounds_error=False, fill_value=0.)
      signln = interp1d(np.log(self.l.flatten()), np.sign(result.flatten()), kind='linear', bounds_error=False, fill_value=0.)
      f = lambda l: np.exp(lnfln(np.log(l))) * signln(np.log(l))
      return f



   ###############################################################################
   ###############################################################################
   # Noise cross-power between shear and dilation


   def computeQuadEstKappaShearDilationNoiseFFT(self, fC0, fCtot, fCfg=None, lMin=1., lMaxS=1.e5, lMaxD=1.e5, corr=True, test=False, cache=None):
      """Computes the lensing noise cross-spectrum N_L^{kappa_shear * kappa_dilation}
      between the shear and the dilation estimators.
      lMaxS: maximum temperature multipole used in the shear estimator
      lMaxD: maximum temperature multipole used in the dilation estimator
      """
      if fCfg is None:
         fCfg = fCtot
      
      # Numerator: keep the minimum lMax
      lMax = min(lMaxS, lMaxD)
      
      def fdLnC0dLnl(l):
         e = 0.01
         lup = l*(1.+e)
         ldown = l*(1.-e)
         result = divide(fC0(lup), fC0(ldown))
         result = np.log(result) / (2.*e)
         return result

      def fdLnl2C0dLnl(l):
         e = 0.01
         lup = l*(1.+e)
         ldown = l*(1.-e)
         result = lup**2 * fC0(lup)
         result /= ldown**2 * fC0(ldown)
         result = np.log(result) / (2.*e)
         return result
      
      def gShear(l):
         # cut off the high ells from input map
         if (l<lMin) or (l>lMax):
            return 0.
         result = divide(fC0(l), fCtot(l)**2)
         result *= fdLnC0dLnl(l) # for isotropic dilation
         result /= 0.5
         if not np.isfinite(result):
            result = 0.
         return result
      
      def gDilation(l):
         # cut off the high ells from input map
         if (l<lMin) or (l>lMax):
            return 0.
         result = divide(fC0(l), fCtot(l)**2)
         result *= fdLnl2C0dLnl(l) # for isotropic dilation
         result /= 0.5
         if not np.isfinite(result):
            result = 0.
         return result
      
      # useful terms
      fx = divideArr((self.lx**2-self.ly**2), self.l**2)
      fx[np.where(np.isfinite(fx)==False)] = 0.
      #
      fy = divideArr(self.lx*self.ly, self.l**2)
      fy[np.where(np.isfinite(fy)==False)] = 0.
   
      # First, the symmetric term
      f = lambda l: gDilation(l) * fCfg(l)
      term1DFourier = self.filterFourierIsotropic(f, dataFourier=np.ones_like(self.l), test=test)
      term1D = self.inverseFourier(term1DFourier)
      #
      f = lambda l: gShear(l) * fCfg(l)
      term1SFourier = self.filterFourierIsotropic(f, dataFourier=np.ones_like(self.l), test=test)

      term1xFourier = term1SFourier * fx
      term1x = self.inverseFourier(term1xFourier)
      term1xFourier = self.fourier(term1x * term1D)
      term1xFourier *= fx
      #
      term1yFourier = term1SFourier * fy
      term1y = self.inverseFourier(term1yFourier)
      term1yFourier = self.fourier(term1y * term1D)
      term1yFourier *= 4.*fy
      #
      term1Fourier = term1xFourier + term1yFourier
      if test:
         self.plotFourier(term1xFourier)
         self.plotFourier(term1yFourier)
         self.plotFourier(term1Fourier)


      # Second, the asymmetric term
      f = lambda l: fCfg(l) * (l>=lMin) * (l<=lMax)
      term2CFourier = self.filterFourierIsotropic(f, dataFourier=np.ones_like(self.l), test=test)
      term2C = self.inverseFourier(term2CFourier)
      #
      f = lambda l: gShear(l) * gDilation(l) * fCfg(l)
      term2SDFourier = self.filterFourierIsotropic(f, dataFourier=np.ones_like(self.l), test=test)

      term2xFourier = term2SDFourier * fx
      term2x = self.inverseFourier(term2xFourier)
      term2xFourier = self.fourier(term2x * term2C)
      term2xFourier *= fx
      #
      term2yFourier = term2SDFourier * fy
      term2y = self.inverseFourier(term2yFourier)
      term2yFourier = self.fourier(term2y * term2C)
      term2yFourier *= 4.*fy
      #
      term2Fourier = term2xFourier + term2yFourier
      if test:
         self.plotFourier(term2xFourier)
         self.plotFourier(term2yFourier)
         self.plotFourier(term2Fourier)
         self.plotFourier(term1Fourier + term2Fourier)

      # add terms
      resultFourier = term1Fourier + term2Fourier

      # compute normalization
      if corr:
         normalizationFourier = self.computeQuadEstPhiShearNormalizationCorrectedFFT(fC0, fCtot, lMin=lMin, lMax=lMaxS, test=test, cache=cache)
         normalizationFourier *= self.computeQuadEstPhiDilationNormalizationCorrectedFFT(fC0, fCtot, lMin=lMin, lMax=lMaxD, test=test, cache=cache)
      else:
         normalizationFourier = self.computeQuadEstPhiShearNormalizationFFT(fC0, fCtot, lMin=lMin, lMax=lMaxS, test=test)
         normalizationFourier = self.computeQuadEstPhiDilationNormalizationFFT(fC0, fCtot, lMin=lMin, lMax=lMaxD, test=test)

      # divide by squared normalization
      resultFourier *= normalizationFourier

      # remove L > 2 lMax
      f = lambda l: (l <= 2. * lMax)
      resultFourier = self.filterFourierIsotropic(f, dataFourier=resultFourier, test=test)

      return resultFourier


   def forecastN0KappaShearDilation(self, fC0, fCtot, fCfg=None, lMin=1., lMaxS=1.e5, lMaxD=1.e5, corr=True, test=False, cache=None):
      """Interpolates the result for N_L^{kappa_shear * kappa_dilation} = f(l),
      to be used for forecasts on lensing reconstruction.
      lMaxS: maximum temperature multipole used in the shear estimator
      lMaxD: maximum temperature multipole used in the dilation estimator
      """
      # One can only form the cross-correlation for lensing modes < 2*lMax,
      # where lMax = min(lMaxS, lMaxD).
      lMax = min(lMaxS, lMaxD)
      
      print("computing the reconstruction noise")
      n0Kappa = self.computeQuadEstKappaShearDilationNoiseFFT(fC0, fCtot, fCfg=fCfg, lMin=lMin, lMaxS=lMaxS, lMaxD=lMaxD, corr=corr, test=test, cache=cache)
      # keep only the real part (the imag. part should be zero, but is tiny in practice)
      n0Kappa = np.real(n0Kappa)
      # remove the nans
      n0Kappa = np.nan_to_num(n0Kappa)

      # fix the issue of the wrong ell=0 value
      # replace it by the value lx=0, ly=fundamental
      n0Kappa[0,0] = n0Kappa[0,1]
      
      # interpolate
      where = (self.l.flatten()>0.)*(self.l.flatten()<2.*lMax)
      L = self.l.flatten()[where]
      N = n0Kappa.flatten()[where]
      lnfln = interp1d(np.log(L), np.log(np.abs(N)), kind='linear', bounds_error=False, fill_value=np.inf)
      sgnfln = interp1d(np.log(L), np.sign(N), kind='linear', bounds_error=False, fill_value=np.inf)
      f = lambda l: np.exp(lnfln(np.log(l))) * sgnfln(np.log(l))
      return f



   ###############################################################################
   ###############################################################################
   # Shear B-mode estimator

   def quadEstPhiShearBNonNorm(self, fC0, fCtot, lMin=5.e2, lMax=3000., dataFourier=None, dataFourier2=None, test=False):
      '''Non-normalized quadratic estimator for phi from shear B-mode only
      fC0: ulensed power spectrum
      fCtot: lensed power spectrum + noise
      '''
      if dataFourier is None:
         dataFourier = self.dataFourier.copy()
      if dataFourier2 is None:
         dataFourier2 = dataFourier.copy()
      
      # cut off high ells
      f = lambda l: 1.*(l>=lMin)*(l<=lMax)
      FDataFourier = self.filterFourierIsotropic(f, dataFourier=dataFourier, test=test)
      FData = self.inverseFourier(FDataFourier)
      if test:
         print("show Fourier data")
         self.plotFourier(dataFourier=FDataFourier)
      
      # weight function for shear
      def fdLnC0dLnl(l):
         e = 0.01
         lup = l*(1.+e)
         ldown = l*(1.-e)
         result = divide(fC0(lup), fC0(ldown))
         result = np.log(result) / (2.*e)
         return result
      if test:
         F = np.array(list(map(fdLnC0dLnl, self.l.flatten())))
         plt.semilogx(self.l.flatten(), F, 'b.')
         plt.show()

      # sort of shear Wiener-filter
      def f(l):
         # cut off the high ells from input map
         if (l<lMin) or (l>lMax):
            return 0.
         result = divide(fC0(l), fCtot(l)**2)
         result *= fdLnC0dLnl(l) # for shear
         result /= 0.5
         if not np.isfinite(result):
            result = 0.
         return result
      WFDataFourier = self.filterFourierIsotropic(f, dataFourier=dataFourier2, test=test)
      if test:
         print("showing the WF map")
         self.plotFourier(dataFourier=WFDataFourier)
         print("checking the power spectrum of this map")
         theory = lambda l: f(l)**2 * fC0(l)
         self.powerSpectrum(theory=[theory], dataFourier=WFDataFourier, plot=True)
      
      # multiplication by sin 2 theta_{L,l}
      
      def fDiff(lx, ly):
         l2 = lx**2 + ly**2
         if l2==0:
            return 0.
         else:
            return divide((lx**2-ly**2),l2)
      
      def fProd(lx, ly):
         l2 = lx**2 + ly**2
         if l2==0:
            return 0.
         else:
            return divide(lx * ly,l2)
      
      
      # term 1
      term1Fourier = self.filterFourier(fProd, dataFourier=WFDataFourier, test=test)
      term1 = self.inverseFourier(dataFourier=term1Fourier)
      term1 *= FData
      term1Fourier = self.fourier(data=term1)
      term1Fourier = self.filterFourier(fDiff, dataFourier=term1Fourier, test=test)
      term1Fourier *= 2.
      
      # term 2
      term2Fourier = self.filterFourier(fDiff, dataFourier=WFDataFourier, test=test)
      term2Fourier *= -1.  # because we want ly^2-lx^2 here
      term2 = self.inverseFourier(dataFourier=term2Fourier)
      term2 *= FData
      term2Fourier = self.fourier(data=term2)
      term2Fourier = self.filterFourier(fProd, dataFourier=term2Fourier, test=test)
      term2Fourier *= 2.

      # add term1 and term2 to get kappa
      resultFourier = term1Fourier + term2Fourier
      #
      if test:
         print("show term 1")
         self.plotFourier(term1Fourier)
         print("show term 2")
         self.plotFourier(term2Fourier)
         print("show term 1 + term 2")
         self.plotFourier(term1Fourier + term2Fourier)

      # get phi from kappa
      resultFourier *= divideArr(-2.*np.ones_like(self.l), self.l**2)
      resultFourier[np.where(np.isfinite(resultFourier)==False)] = 0.
      if test:
         print("checking the power spectrum of phi map")
         self.powerSpectrum(dataFourier=resultFourier, plot=True)

      if test:
         print("Show real-space phi map")
         result = self.inverseFourier(resultFourier)
         self.plot(result)

      return resultFourier


   def computeQuadEstPhiShearBNormalizationFFT(self, fC0, fCtot, lMin=5.e2, lMax=3.e3,  test=False):
      """Multiplicative normalization for phi estimator from shear B-mode only,
      computed with FFT
      ell cuts are performed to remain in the regime L_phi < l_T.
      Same as for shear E-mode, since <cos^2> = <sin^2> = 1/2.
      """
      resultFourier = self.computeQuadEstPhiShearNormalizationFFT(fC0, fCtot, lMin=lMin, lMax=lMax,  test=test)
      return resultFourier


   def computeQuadEstPhiShearBNormalizationCorrectedFFT(self, fC0, fCtot, lMin=5.e2, lMax=3.e3, test=False, cache=None):
      """The shear B-mode estimator has zero response to lensing,
      so the corrected normalization would be to divide by zero,
      i.e. the unbiased normalization should be +infinity.
      Instead, here we decide to use the same normalization as the shear E-mode,
      so that shear E-mode and B-modes can be compared.
      """
      resultFourier = self.computeQuadEstPhiShearNormalizationCorrectedFFT(fC0, fCtot, lMin=lMin, lMax=lMax, test=test, cache=cache)
      return resultFourier


   def computeQuadEstKappaShearBNormCorr(self, fC0, fCtot, lMin=1., lMax=1.e5, dataFourier=None, dataFourier2=None, path=None, corr=True, test=False, cache=None):
      '''Returns the shear B-mode-only normalized quadratic estimator for kappa in Fourier space,
      and saves it to file if needed.
      The normalization includes or not the correction for the multiplicative bias.
      '''
      # non-normalized QE for phi
      resultFourier = self.quadEstPhiShearBNonNorm(fC0, fCtot, lMin=lMin, lMax=lMax, dataFourier=dataFourier, dataFourier2=dataFourier2, test=test)
      # convert from phi to kappa
      resultFourier = self.kappaFromPhi(resultFourier)
      # compute normalization
      if corr:
         normalizationFourier = self.computeQuadEstPhiShearBNormalizationCorrectedFFT(fC0, fCtot, lMin=lMin, lMax=lMax, test=test, cache=cache)
      else:
         normalizationFourier = self.computeQuadEstPhiShearBNormalizationFFT(fC0, fCtot, lMin=lMin, lMax=lMax, test=test)
      # normalized (not mean field-subtracted) QE for kappa
      resultFourier *= normalizationFourier
      # save to file if needed
      if path is not None:
         self.saveDataFourier(resultFourier, path)
      return resultFourier


   def computeQuadEstKappaShearBNoiseFFT(self, fC0, fCtot, fCfg=None, lMin=1., lMax=1.e5, corr=True, test=False, cache=None):
      """Computes the lensing noise power spectrum N_L^kappa
      for the shear B-mode estimator.
      """
      if fCfg is None:
         fCfg = fCtot
      
      def fdLnC0dLnl(l):
         e = 0.01
         lup = l*(1.+e)
         ldown = l*(1.-e)
         result = divide(fC0(lup), fC0(ldown))
         result = np.log(result) / (2.*e)
         return result

      def g(l):
         # cut off the high ells from input map
         if (l<lMin) or (l>lMax):
            return 0.
         result = divide(fC0(l), fCtot(l)**2)
         result *= fdLnC0dLnl(l) # for isotropic dilation
         result /= 0.5
         if not np.isfinite(result):
            result = 0.
         return result
      
      # useful terms
      fDiff = divideArr((self.lx**2-self.ly**2), self.l**2)
      fDiff[np.where(np.isfinite(fDiff)==False)] = 0.
      #
      fProd = divideArr(self.lx*self.ly, self.l**2)
      fProd[np.where(np.isfinite(fProd)==False)] = 0.
   
      # First, the symmetric term
      f = lambda l: g(l) * fCfg(l)
      term1Fourier = self.filterFourierIsotropic(f, dataFourier=np.ones_like(self.l), test=test)
      #
      term1xFourier = term1Fourier * fDiff
      term1xFourier *= -1.
      term1x = self.inverseFourier(term1xFourier)
      #
      term1yFourier = term1Fourier * fProd
      term1y = self.inverseFourier(term1yFourier)
      
      term1xxFourier = self.fourier(term1x**2)
      term1xxFourier *= fProd**2
      term1xxFourier *= 4.
      #
      term1yyFourier = self.fourier(term1y**2)
      term1yyFourier *= fDiff**2
      term1yyFourier *= 4.
      #
      term1xyFourier = self.fourier(term1x*term1y)
      term1xyFourier *= fDiff * fProd
      term1xyFourier *= 4. * 2.
      #
      term1Fourier = term1xxFourier + term1yyFourier + term1xyFourier
      if test:
         self.plotFourier(term1xxFourier)
         self.plotFourier(term1yyFourier)
         self.plotFourier(term1xyFourier)
         self.plotFourier(term1Fourier)


      # Second, the asymmetric term
      f = lambda l: fCfg(l) * (l>=lMin) * (l<=lMax)
      term2bFourier = self.filterFourierIsotropic(f, dataFourier=np.ones_like(self.l), test=test)
      term2b = self.inverseFourier(term2bFourier)
      #
      f = lambda l: g(l)**2 * fCfg(l)
      term2aFourier = self.filterFourierIsotropic(f, dataFourier=np.ones_like(self.l), test=test)

      term2xxFourier = term2aFourier * fProd**2
      term2xx = self.inverseFourier(term2xxFourier)
      term2xxFourier = self.fourier(term2xx * term2b)
      term2xxFourier *= fDiff**2
      term2xxFourier *= 4.
      #
      term2yyFourier = term2aFourier * fDiff**2
      term2yy = self.inverseFourier(term2yyFourier)
      term2yyFourier = self.fourier(term2yy * term2b)
      term2yyFourier *= fProd**2
      term2yyFourier *= 4.
      #
      term2xyFourier = term2aFourier * fDiff * fProd
      term2xyFourier*= -1.
      term2xy = self.inverseFourier(term2xyFourier)
      term2xyFourier = self.fourier(term2xy * term2b)
      term2xyFourier *= fDiff*fProd
      term2xyFourier *= 4.
      term2xyFourier *= 2.
      #
      term2Fourier = term2xxFourier + term2yyFourier + term2xyFourier
      if test:
         self.plotFourier(term2xxFourier)
         self.plotFourier(term2yyFourier)
         self.plotFourier(term2xyFourier)
         self.plotFourier(term2Fourier)
         self.plotFourier(term1Fourier + term2Fourier)

      # add terms
      resultFourier = term1Fourier + term2Fourier

      # compute normalization
      if corr:
         normalizationFourier = self.computeQuadEstPhiShearBNormalizationCorrectedFFT(fC0, fCtot, lMin=lMin, lMax=lMax, test=test, cache=cache)
      else:
         normalizationFourier = self.computeQuadEstPhiShearBNormalizationFFT(fC0, fCtot, lMin=lMin, lMax=lMax, test=test)

      # divide by squared normalization
      resultFourier *= normalizationFourier**2

      # remove L > 2 lMax
      f = lambda l: (l <= 2. * lMax)
      resultFourier = self.filterFourierIsotropic(f, dataFourier=resultFourier, test=test)

      return resultFourier


   def forecastN0KappaShearB(self, fC0, fCtot, fCfg=None, lMin=1., lMax=1.e5, corr=True, test=False, cache=None):
      """Interpolates the result for N_L^{kappa_shearB} = f(l),
      to be used for forecasts on lensing reconstruction.
      """
      print("computing the reconstruction noise")
      n0Kappa = self.computeQuadEstKappaShearBNoiseFFT(fC0, fCtot, fCfg=fCfg, lMin=lMin, lMax=lMax, corr=corr, test=test, cache=cache)
      # keep only the real part (the imag. part should be zero, but is tiny in practice)
      n0Kappa = np.real(n0Kappa)
      # remove the nans
      n0Kappa = np.nan_to_num(n0Kappa)
      # make sure every value is positive
      n0Kappa = np.abs(n0Kappa)
      
      # fix the issue of the wrong ell=0 value
      # replace it by the value lx=0, ly=fundamental
      n0Kappa[0,0] = n0Kappa[0,1]
      
      # interpolate
      where = (self.l.flatten()>0.)*(self.l.flatten()<2.*lMax)
      L = self.l.flatten()[where]
      N = n0Kappa.flatten()[where]
      lnfln = interp1d(np.log(L), np.log(N), kind='linear', bounds_error=False, fill_value=np.inf)
#      lnfln = interp1d(np.log(self.l.flatten()), np.log(n0Kappa.flatten()), kind='linear', bounds_error=False, fill_value=0.)
      f = lambda l: np.exp(lnfln(np.log(l)))
      return f



   ###############################################################################
   ###############################################################################
   # Noise cross-spectrum for Shear E and Shear B estimators
   #!!! Not working: gives weird results, that are not independent of the direction of L.


   def computeQuadEstKappaShearShearBNoiseFFT(self, fC0, fCtot, fCfg=None, lMin=1., lMaxS=1.e5, lMaxSB=1.e5, corr=True, test=False, cache=None):
      """Computes the lensing noise power spectrum N_L^kappa
      for the shear B-mode estimator.
      !!! Not working: gives weird results, that are not independent of the direction of L. Also, the result goes down as the numerical precision improves.
      !!! This probably suggests that there is a fundamental reason why shear E and shear B should be uncorrelated.
      """
      if fCfg is None:
         fCfg = fCtot
      
      # Numerator: keep the minimum lMax
      lMax = min(lMaxS, lMaxSB)
      
      def fdLnC0dLnl(l):
         e = 0.01
         lup = l*(1.+e)
         ldown = l*(1.-e)
         result = divide(fC0(lup), fC0(ldown))
         result = np.log(result) / (2.*e)
         return result

      def g(l):
         # cut off the high ells from input map
         if (l<lMin) or (l>lMax):
            return 0.
         result = divide(fC0(l), fCtot(l)**2)
         result *= fdLnC0dLnl(l) # for isotropic dilation
         result /= 0.5
         if not np.isfinite(result):
            result = 0.
         return result
      
      # useful terms
      fDiff = divideArr((self.lx**2-self.ly**2), self.l**2)
      fDiff[np.where(np.isfinite(fDiff)==False)] = 0.
      #
      fProd = divideArr(self.lx*self.ly, self.l**2)
      fProd[np.where(np.isfinite(fProd)==False)] = 0.

      # First, the symmetric term
      f = lambda l: g(l) * fCfg(l)
      termSFourier = self.filterFourierIsotropic(f, dataFourier=np.ones_like(self.l), test=test)

      termEx = self.inverseFourier(termSFourier * fDiff)
      termEy = self.inverseFourier(termSFourier * 4. *fProd)
      #
      termBx = self.inverseFourier(termSFourier * 2. * fProd)
      termBy = self.inverseFourier(termSFourier * (-2. * fDiff))

      term1xxFourier = self.fourier(termEx * termBx)
      term1xxFourier *= fDiff**2
      #
      term1yyFourier = self.fourier(termEy * termBy)
      term1yyFourier *= fProd**2
      #
      term1xyFourier = self.fourier(termEx * termBy)
      term1xyFourier *= fDiff * fProd
      #
      term1yxFourier = self.fourier(termEy * termBx)
      term1yxFourier *= fProd * fDiff
      #
      term1Fourier = term1xxFourier + term1yyFourier + term1xyFourier + term1yxFourier
      if test:
         self.plotFourier(term1xxFourier)
         self.plotFourier(term1yyFourier)
         self.plotFourier(term1xyFourier)
         self.plotFourier(term1yxFourier)
         self.plotFourier(term1Fourier)


      # Second, the asymmetric term
      f = lambda l: g(l)**2 * fCfg(l)
      term2aFourier = self.filterFourierIsotropic(f, dataFourier=np.ones_like(self.l), test=test)
      #
      f = lambda l: fCfg(l) * (l>=lMin) * (l<=lMax)
      term2bFourier = self.filterFourierIsotropic(f, dataFourier=np.ones_like(self.l), test=test)
      term2b = self.inverseFourier(term2bFourier)

      term2xx = self.inverseFourier(term2aFourier * fDiff * 2.*fProd)
      term2xxFourier = self.fourier(term2xx * term2b)
      term2xxFourier *= fDiff**2
      #
      term2yy = self.inverseFourier(term2aFourier * 4.*fProd * (-2.*fDiff))
      term2yyFourier = self.fourier(term2yy * term2b)
      term2yyFourier *= fProd**2
      #
      term2xy = self.inverseFourier(term2aFourier * fDiff * (-2.*fDiff))
      term2xyFourier = self.fourier(term2xy * term2b)
      term2xyFourier *= fDiff * fProd
      #
      term2yx = self.inverseFourier(term2aFourier * 4.*fProd * 2.*fProd)
      term2yxFourier = self.fourier(term2yx * term2b)
      term2yxFourier *= fProd * fDiff
      #
      term2Fourier = term2xxFourier + term2yyFourier + term2xyFourier + term2yxFourier
      if test:
         self.plotFourier(term2xxFourier)
         self.plotFourier(term2yyFourier)
         self.plotFourier(term2xyFourier)
         self.plotFourier(term2yxFourier)
         self.plotFourier(term2Fourier)

      # add terms
      resultFourier = term1Fourier + term2Fourier
      
      if test:
         self.plotFourier(resultFourier)
         plt.loglog(self.l.flatten(), resultFourier.flatten(), '.')
         plt.show()
   

      # compute normalization: same for shear and shear B
      if corr:
         normalizationFourier = self.computeQuadEstPhiShearNormalizationCorrectedFFT(fC0, fCtot, lMin=lMin, lMax=lMaxS, test=test, cache=cache)
         if lMaxS==lMaxSB:
            normalizationFourier *= normalizationFourier
         else:
            normalizationFourier *= self.computeQuadEstPhiShearNormalizationCorrectedFFT(fC0, fCtot, lMin=lMin, lMax=lMaxSB, test=test, cache=cache)
      
      else:
         normalizationFourier = self.computeQuadEstPhiShearNormalizationFFT(fC0, fCtot, lMin=lMin, lMax=lMaxS, test=test)
         if lMaxS==lMaxSB:
            normalizationFourier *= normalizationFourier
         else:
            normalizationFourier *= self.computeQuadEstPhiShearNormalizationFFT(fC0, fCtot, lMin=lMin, lMax=lMaxSB, test=test)

      # divide by squared normalization
      resultFourier *= normalizationFourier

      # remove L > 2 lMax
      f = lambda l: (l <= 2. * lMax)
      resultFourier = self.filterFourierIsotropic(f, dataFourier=resultFourier, test=test)

      return resultFourier



   def forecastN0KappaShearShearB(self, fC0, fCtot, fCfg=None, lMin=1., lMaxS=1.e5, lMaxSB=1.e5, corr=True, test=False, cache=None):
      """Interpolates the result for N_L^{kappa_shear * kappa_shearB} = f(l),
      to be used for forecasts on lensing reconstruction.
      """
      # One can only form the cross-correlation for lensing modes < 2*lMax,
      # where lMax = min(lMaxS, lMaxD).
      lMax = min(lMaxS, lMaxSB)
      print("computing the reconstruction noise")
      n0Kappa = self.computeQuadEstKappaShearShearBNoiseFFT(fC0, fCtot, fCfg=fCfg, lMin=lMin, lMaxS=lMaxS, lMaxSB=lMaxSB, corr=corr, test=test, cache=cache)
      # keep only the real part (the imag. part should be zero, but is tiny in practice)
      n0Kappa = np.real(n0Kappa)
      # remove the nans
      n0Kappa = np.nan_to_num(n0Kappa)
      # make sure every value is positive
      n0Kappa = np.abs(n0Kappa)

      # fix the issue of the wrong ell=0 value
      # replace it by the value lx=0, ly=fundamental
      n0Kappa[0,0] = n0Kappa[0,1]

      # interpolate
      where = (self.l.flatten()>0.)*(self.l.flatten()<2.*lMax)
      L = self.l.flatten()[where]
      N = n0Kappa.flatten()[where]
      lnfln = interp1d(np.log(L), np.log(N), kind='linear', bounds_error=False, fill_value=np.inf)
#      lnfln = interp1d(np.log(self.l.flatten()), np.log(n0Kappa.flatten()), kind='linear', bounds_error=False, fill_value=0.)
      f = lambda l: np.exp(lnfln(np.log(l)))
      return f



   ###############################################################################
   ###############################################################################
   # Point source hardened estimator with arbitrary profile
   
   def computeResponseFFT(self, fC0, fCtot, lMin=1., lMax=1.e5, test=False, cache=None, sigma=0., u=None):
      '''calculate the response R_L'''
       
      # Response calculation
      def doCalculation():
         print("Doing full calculation: computeResponseFFT")
         # inverse-var weighted map
         def f(l):
            if (l<lMin) or (l>lMax):
               return 0.
            result = 1./fCtot(l)
            if u is not None: result *= u(l)
            else: result *= np.exp(- (sigma**2.) * (l**2.) / 2.)
            if not np.isfinite(result):
               result = 0.
            return result
         iVarFourier = np.array(list(map(f, self.l.flatten())))
         iVarFourier = iVarFourier.reshape(self.l.shape)
         iVar = self.inverseFourier(dataFourier=iVarFourier)

         # WF map
         def f(l):
            if (l<lMin) or (l>lMax):
               return 0.
            result = divide(fC0(l),fCtot(l))
            # artificial factor of i such that f(-l) = f(l)*,
            # such that f(x) is real
            result *= 1.j
            if u is not None: result *= u(l)
            else: result *= np.exp(- (sigma**2.) * (l**2.) / 2.)
            if not np.isfinite(result):
               result = 0.
            return result
         WFFourier = np.array(list(map(f, self.l.flatten())))
         WFFourier = WFFourier.reshape(self.l.shape)

         # term 1x
         term1x = self.inverseFourier(dataFourier= self.lx * WFFourier)
         term1x *= iVar
         term1xFourier = self.fourier(data=term1x)
         term1xFourier *= divideArr(self.lx, (self.lx**2. + self.ly**2.))
	 # correction for the artificial i in WFFourier
         term1xFourier /= 1.j
         #
         # term 1y
         term1y = self.inverseFourier(dataFourier= self.ly * WFFourier)
         term1y *= iVar 
         term1yFourier = self.fourier(data=term1y)
         term1yFourier *= divideArr(self.ly, (self.lx**2. + self.ly**2.))
	 # correction for the artificial i in WFFourier
         term1yFourier /= 1.j
         
         if test:
            self.plotFourier(term1xFourier)
            self.plotFourier(term1yFourier)
            self.plotFourier(term1xFourier+term1yFourier)

         # add all terms
         resultFourier = term1xFourier + term1yFourier
	 # Normalization factor
         resultFourier *= 2.
         # scale by profile
         if u is not None: resultFourier /= u(self.l)
         else: resultFourier *= np.exp( (sigma**2.) * (self.l**2.) / 2.)

         # cut off the high ells from phi normalization map
         f = lambda l: (l<=2.*lMax)
         resultFourier = self.filterFourierIsotropic(f, dataFourier=resultFourier, test=False)

         if test:
            plt.loglog(self.l.flatten(), resultFourier.flatten(), 'b.')
            plt.show()

         return resultFourier

      # Caching boiler plate
      # if no caching is desired, just compute
      if cache is None:
         resultFourier = doCalculation()
      # if caching is desired
      else:
         # if first call with caching, set up the cache dictionary
         if not hasattr(self.computeResponseFFT.__func__, "cache"):
            self.computeResponseFFT.__func__.cache = {}
         # if the calculation has been done before
         if cache in self.computeResponseFFT.cache:
            resultFourier = self.computeResponseFFT.cache[cache].copy()
         # if this calculation was not done before
         else:
            resultFourier = doCalculation()
            self.computeResponseFFT.cache[cache] = resultFourier.copy()

      return resultFourier



   def computeSNormalizationFFT(self, fCtot, lMin=1., lMax=1.e5, test=False, cache=None, sigma=0., u=None):
      '''calculate the normalization for S^2'''
       
      # Actual calculation
      def doCalculation():
         print("Doing full calculation: computeS2NormalizationFFT")
         # inverse-variance
         def f(l):
            if (l<lMin) or (l>lMax):
               return 0.
            result = 1./fCtot(l)
            if u is not None: result *= u(l)**2.
            else: result *= np.exp(- (sigma**2.) * (l**2.))
            if not np.isfinite(result):
               result = 0.
            return result
         iVarFourier = np.array(list(map(f, self.l.flatten())))
         iVarFourier = iVarFourier.reshape(self.l.shape)
         iVar = self.inverseFourier(dataFourier=iVarFourier)
         
         # term 1 (it's the only term)
         term1 = iVar**2.
         term1Fourier = self.fourier(data=term1)
         
         if test: self.plotFourier(term1Fourier)

         # renaming the term for consistency
         resultFourier = term1Fourier / 2.
         # multiplying by profile
         if u is not None: resultFourier /= u(self.l)**2.
         else: resultFourier *= np.exp( (sigma**2.) * (self.l**2.) )

         # cut off the high ells 
         f = lambda l: (l<=2.*lMax)
         resultFourier = self.filterFourierIsotropic(f, dataFourier=resultFourier, test=False)

	 # invert
         resultFourier = 1./resultFourier
         resultFourier[np.where(np.isfinite(resultFourier)==False)] = 0.

         if test:
            plt.loglog(self.l.flatten(), resultFourier.flatten(), 'b.')
            plt.show()
         
         return resultFourier

      # Caching boiler plate
      # if no caching is desired, just compute
      if cache is None:
         resultFourier = doCalculation()
      # if caching is desired
      else:
         # if first call with caching, set up the cache dictionary
         if not hasattr(self.computeS2NormalizationFFT.__func__, "cache"):
            self.computeS2NormalizationFFT.__func__.cache = {}
         # if the calculation has been done before
         if cache in self.computeS2NormalizationFFT.cache:
            resultFourier = self.computeS2NormalizationFFT.cache[cache].copy()
         # if this calculation was not done before
         else:
            resultFourier = doCalculation()
            self.computeS2NormalizationFFT.cache[cache] = resultFourier.copy()

      return resultFourier



   def quadEstSNonNorm(self, fCtot, lMin= 1., lMax=1.e5, dataFourier=None, dataFourier2=None, test=False, sigma=0., u=None):
      """non-normalized quadratic estimator for S2
      fC0: ulensed power spectrum
      fCtot: lensed power spectrum + noise
      """
      if dataFourier is None:
         dataFourier = self.dataFourier.copy()
      if dataFourier2 is None:
         dataFourier2 = dataFourier.copy()
 
      # inverse-var weighted map
      def f(l):
         # cut off the high ells from input map
         if (l<lMin) or (l>lMax):
            return 0.
         result = 1./fCtot(l)
         if u is not None: result *= u(l)
         else: result *= np.exp(- (sigma**2.) * (l**2.) / 2.)
         if not np.isfinite(result):
            result = 0.
         return result
      iVarDataFourier = self.filterFourierIsotropic(f, dataFourier=dataFourier, test=test)
      iVarData = self.inverseFourier(iVarDataFourier)
      iVarData2Fourier = self.filterFourierIsotropic(f, dataFourier=dataFourier2, test=test)
      iVarData2 = self.inverseFourier(iVarData2Fourier)
      if test:
         print("showing the inverse var. weighted map")
         self.plot(data=iVarData)
         print("checking the power spectrum of this map")
         self.powerSpectrum(theory=f, dataFourier=iVarDataFourier, plot=True)

      # term 1 (it's the only term)
      term1 = iVarData*iVarData2
      term1Fourier = self.fourier(data=term1)

      if test: self.plotFourier(term1Fourier)

      # renaming the term for consistency
      resultFourier = term1Fourier / 2.
      # multiplying by profile
      if u is not None: resultFourier /= u(self.l)
      else: resultFourier *= np.exp( (sigma**2.) * (self.l**2.) / 2.)
      
      # cut off the high ells from phi normalization map
      f = lambda l: (l<=2.*lMax)
      resultFourier = self.filterFourierIsotropic(f, dataFourier=resultFourier, test=False)

      if test:
         plt.loglog(self.l.flatten(), resultFourier.flatten(), 'b.')
         plt.show()
      
      return resultFourier



   def computeQuadEstSNorm(self, fCtot, lMin=1., lMax=1.e5, dataFourier=None, dataFourier2=None, path=None, test=False, cache=None, sigma=0., u=None):
      '''Returns the normalized quadratic estimator for S2 in Fourier space,
      and saves it to file if needed.
      '''
      # non-normalized QE for S2
      resultFourier = self.quadEstSNonNorm(fCtot, lMin=lMin, lMax=lMax, dataFourier=dataFourier, dataFourier2=dataFourier2, test=test, sigma=sigma, u=u)
      # compute normalization
      normalizationFourier = self.computeSNormalizationFFT(fCtot, lMin=lMin, lMax=lMax, test=test, cache=cache, sigma=sigma, u=u)
      # normalized (not mean field-subtracted) QE for kappa
      resultFourier *= normalizationFourier
      # save to file if needed
      if path is not None:
         self.saveDataFourier(resultFourier, path)
      return resultFourier



   def computeQuadEstKappaPointSourceHardenedNorm(self, fC0, fCtot, lMin=1., lMax=1.e5, dataFourier=None, dataFourier2=None, path=None, test=False, cache=None, sigma=0., u=None):
      '''Returns the normalized bias hardened quadratic estimator for kappa in Fourier space,
      and saves it to file if needed.
      '''
      # minimum variance kappa estimator
      kappaMinVarFourier = self.computeQuadEstKappaNorm(fC0, fCtot, lMin=lMin, lMax=lMax, dataFourier=dataFourier, dataFourier2=dataFourier2, test=test, cache=cache)
      # minimum variance S2 estimator
      SMinVarFourier = self.computeQuadEstSNorm(fCtot, lMin=lMin, lMax=lMax, dataFourier=dataFourier, dataFourier2=dataFourier2, test=test, cache=cache, sigma=sigma, u=u)
      # normalization for phi
      NphiFourier = self.computeQuadEstPhiNormalizationFFT(fC0, fCtot, lMin=lMin, lMax=lMax, test=test, cache=cache)
      # convert from phi to kappa
      NkappaFourier = 0.25 * self.l**4 * NphiFourier
      # normalization for S2
      NSFourier = self.computeSNormalizationFFT(fCtot, lMin=lMin, lMax=lMax, test=test, cache=cache, sigma=sigma, u=u)
      # response
      RFourier = self.computeResponseFFT(fC0, fCtot, lMin=lMin, lMax=lMax, test=test, cache=cache, sigma=sigma, u=u)
   
      # inverting the matrix {{1,a},{b,1}}
      a = NkappaFourier * RFourier
      b = NSFourier * RFourier
      determinant = 1. - (a * b)
      # invert the determinant
      invDeterminant = 1./determinant
      invDeterminant[np.where(np.isfinite(invDeterminant)==False)] = 0.
      # calculate the point source hardened estimator
      resultFourier = invDeterminant * (kappaMinVarFourier - a * SMinVarFourier)
      # save to file if needed
      if path is not None:
         self.saveDataFourier(resultFourier, path)
      return resultFourier



   def forecastN0S(self, fCtot, lMin=1., lMax=1.e5, test=False, sigma=0., u=None):
      """Interpolates the result for N_L^kappa = f(L),
      to be used for forecasts on lensing reconstruction
      """
      print("computing the reconstruction noise")
      # Standard reconstruction noise
      n0S = self.computeSNormalizationFFT(fCtot=fCtot, lMin=lMin, lMax=lMax, test=test, cache=None, sigma=sigma, u=u)
      # keep only the real part (the imag. part should be zero, but is tiny in practice)
      n0S = np.real(n0S)
      # remove the nans
      n0S = np.nan_to_num(n0S)
      # make sure every value is positive
      n0S = np.abs(n0S)
      # fix the issue of the wrong ell=0 value
      # replace it by the value lx=0, ly=fundamental
      n0S[0,0] = n0S[0,1]
      
      # interpolate
      where = (self.l.flatten()>0.)*(self.l.flatten()<2.*lMax)
      L = self.l.flatten()[where]
      N = n0S.flatten()[where]
      lnfln = interp1d(np.log(L), np.log(N), kind='linear', bounds_error=False, fill_value=np.inf)
      f = lambda l: np.exp(lnfln(np.log(l)))
      return f



   def response(self, fC0, fCtot, lMin=1., lMax=1.e5, test=False, cache=None, sigma=0., u=None):
      print("computing response")
      # Standard reconstruction noise
      r = self.computeResponseFFT(fC0, fCtot, lMin, lMax, test, cache, sigma, u)
      # keep only the real part (the imag. part should be zero, but is tiny in practice)
      r = np.real(r)
      # remove the nans
      r = np.nan_to_num(r)
      # make sure every value is positive
      r = np.abs(r)
      # fix the issue of the wrong ell=0 value
      # replace it by the value lx=0, ly=fundamental
      r[0,0] = r[0,1]
      
      # interpolate
      where = (self.l.flatten()>0.)*(self.l.flatten()<2.*lMax)
      L = self.l.flatten()[where]
      R = r.flatten()[where]
      lnfln = interp1d(np.log(L), np.log(R), kind='linear', bounds_error=False, fill_value=np.inf)
      f = lambda l: np.exp(lnfln(np.log(l)))
      return f



   def forecastN0KappaBH(self, fC0, fCtot, lMin=1., lMax=1.e5, test=False, cache=None, sigma=0., u=None):
      print("computing the response")
      # Standard reconstruction noise
      r = self.computeResponseFFT(fC0, fCtot, lMin, lMax, test, cache, sigma=sigma, u=u)
      ns = self.computeSNormalizationFFT(fCtot, lMin, lMax, test, cache, sigma=sigma, u=u)
      nphi = self.computeQuadEstPhiNormalizationFFT(fC0, fCtot, lMin=lMin, lMax=lMax, test=test, cache=cache)
      # convert from phi to kappa
      nk = 0.25 * self.l**4 * nphi
      # keep only the real part (the imag. part should be zero, but is tiny in practice)
      r = np.real(r)
      ns = np.real(ns)
      nk = np.real(nk)
      # remove the nans
      r = np.nan_to_num(r)
      ns = np.nan_to_num(ns)
      nk = np.nan_to_num(nk)
      # make sure every value is positive
      r = np.abs(r)
      ns = np.abs(ns)
      nk = np.abs(nk)
      # fix the issue of the wrong ell=0 value
      # replace it by the value lx=0, ly=fundamental
      r[0,0] = r[0,1]
      ns[0,0] = ns[0,1]
      nk[0,0] = nk[0,1]
      
      nkPSH = divideArr(nk, (1. - nk*ns*r*r))

      # interpolate
      where = (self.l.flatten()>0.)*(self.l.flatten()<2.*lMax)
      L = self.l.flatten()[where]
      N = nkPSH.flatten()[where]
      lnfln = interp1d(np.log(L), np.log(N), kind='linear', bounds_error=False, fill_value=np.inf)

      where = lnfln(np.log(L)) < np.inf
      lmin = float(L[where][0])   
            
      f = lambda l: np.exp(lnfln(np.log(np.maximum(l,lmin))))
      return f

   def determinant(self, fC0, fCtot, lMin=1., lMax=1.e5, test=False, cache=None, sigma=0., u=None):
      print("computing the response")
      # Standard reconstruction noise
      r = self.computeResponseFFT(fC0, fCtot, lMin, lMax, test, cache, sigma=sigma, u=u)
      ns = self.computeSNormalizationFFT(fCtot, lMin, lMax, test, cache, sigma=sigma, u=u)
      nphi = self.computeQuadEstPhiNormalizationFFT(fC0, fCtot, lMin=lMin, lMax=lMax, test=test, cache=cache)
      # convert from phi to kappa
      nk = 0.25 * self.l**4 * nphi
      # keep only the real part (the imag. part should be zero, but is tiny in practice)
      r = np.real(r)
      ns = np.real(ns)
      nk = np.real(nk)
      # remove the nans
      r = np.nan_to_num(r)
      ns = np.nan_to_num(ns)
      nk = np.nan_to_num(nk)
      
      d = 1. - ns * nk * r**2.
      # make sure every value is positive
      d = np.abs(d)
      # fix the issue of the wrong ell=0 value
      # replace it by the value lx=0, ly=fundamental
      d[0,0] = d[0,1]
      # interpolate
      where = (self.l.flatten()>0.)*(self.l.flatten()<2.*lMax)
      L = self.l.flatten()[where]
      N = d.flatten()[where]
      lnfln = interp1d(np.log(L), np.log(N), kind='linear', bounds_error=False, fill_value=np.inf)
      f = lambda l: np.exp(lnfln(np.log(l)))
      return f



   ###############################################################################
   ###############################################################################
   # Bias hardening with both a delta function and arbitrary profile
   

   def computeResponseSourceSourceFFT(self, fC0, fCtot, lMin=1., lMax=1.e5, test=False, cache=None, sigma=0., u=None):
      '''calculate the response between two source estimators R_L'''
       
      # Response calculation
      def doCalculation():
         print("Doing full calculation: computeResponseFFT")
         # inverse-var weighted map
         def f(l):
            if (l<lMin) or (l>lMax):
               return 0.
            result = 1./fCtot(l)
            if u is not None: result *= u(l)
            else: result *= np.exp(- (sigma**2.) * (l**2.) / 2.)
            if not np.isfinite(result):
               result = 0.
            return result
         iVarFourier = np.array(list(map(f, self.l.flatten())))
         iVarFourier = iVarFourier.reshape(self.l.shape)
         iVar = self.inverseFourier(dataFourier=iVarFourier)

         # term 1 (it's the only term)
         term1 = iVar**2.
         term1Fourier = self.fourier(data=term1)
         
         if test: self.plotFourier(term1Fourier)

         # renaming the term for consistency
         resultFourier = term1Fourier / 2.
         
         # scale by profile
         if u is not None: resultFourier /= u(self.l)
         else: resultFourier *= np.exp( (sigma**2.) * (self.l**2.) / 2.)

         # cut off the high ells from phi normalization map
         f = lambda l: (l<=2.*lMax)
         resultFourier = self.filterFourierIsotropic(f, dataFourier=resultFourier, test=False)

         if test:
            plt.loglog(self.l.flatten(), resultFourier.flatten(), 'b.')
            plt.show()

         return resultFourier

      # Caching boiler plate
      # if no caching is desired, just compute
      if cache is None:
         resultFourier = doCalculation()
      # if caching is desired
      else:
         # if first call with caching, set up the cache dictionary
         if not hasattr(self.computeResponseFFT.__func__, "cache"):
            self.computeResponseFFT.__func__.cache = {}
         # if the calculation has been done before
         if cache in self.computeResponseFFT.cache:
            resultFourier = self.computeResponseFFT.cache[cache].copy()
         # if this calculation was not done before
         else:
            resultFourier = doCalculation()
            self.computeResponseFFT.cache[cache] = resultFourier.copy()

      return resultFourier 



   def computeQuadEstBiasHardened2Norm(self, fC0, fCtot, lMin=1., lMax=1.e5, dataFourier=None, dataFourier2=None, path=None, test=False, cache=None, sigma=0., u=None):
      '''Returns the normalized bias hardened quadratic estimator for kappa in Fourier space,
      and saves it to file if needed.

      The matrix now takes the form 

             1  a  c
             b  1  d
             e  f  1

      '''
      # minimum variance kappa estimator
      kappaMinVarFourier = self.computeQuadEstKappaNorm(fC0, fCtot, lMin=lMin, lMax=lMax, dataFourier=dataFourier, dataFourier2=dataFourier2, test=test, cache=cache)
      # delta function profile source estimator
      SFourier = self.computeQuadEstSNorm(fCtot, lMin=lMin, lMax=lMax, dataFourier=dataFourier, dataFourier2=dataFourier2, test=test, cache=cache, sigma=0.)
      # Gaussian profile source estimator
      SgFourier = self.computeQuadEstSNorm(fCtot, lMin=lMin, lMax=lMax, dataFourier=dataFourier, dataFourier2=dataFourier2, test=test, cache=cache, sigma=sigma, u=u)
      # normalization for phi
      NphiFourier = self.computeQuadEstPhiNormalizationFFT(fC0, fCtot, lMin=lMin, lMax=lMax, test=test, cache=cache)
      # convert from phi to kappa
      NkappaFourier = 0.25 * self.l**4 * NphiFourier
      # normalization for S
      NSFourier = self.computeSNormalizationFFT(fCtot, lMin=lMin, lMax=lMax, test=test, cache=cache, sigma=0.)
      # normalization for Sg
      NSgFourier = self.computeSNormalizationFFT(fCtot, lMin=lMin, lMax=lMax, test=test, cache=cache, sigma=sigma, u=u)
      # response kappa, S
      RkappaSFourier = self.computeResponseFFT(fC0, fCtot, lMin=lMin, lMax=lMax, test=test, cache=cache, sigma=0.)
      # response kappa, Sg
      RkappaSgFourier = self.computeResponseFFT(fC0, fCtot, lMin=lMin, lMax=lMax, test=test, cache=cache, sigma=sigma, u=u)
      # response S, Sg
      RSSgFourier = self.computeResponseSourceSourceFFT(fC0, fCtot, lMin=lMin, lMax=lMax, test=test, cache=cache, sigma=sigma, u=u)
   
      # inverting the matrix 
      a = NkappaFourier * RkappaSFourier
      b = NSFourier * RkappaSFourier
      c = NkappaFourier * RkappaSgFourier
      e = NSgFourier * RkappaSgFourier
      d = NSFourier * RSSgFourier
      f = NSgFourier * RSSgFourier
      determinant = 1. - (a*b) + (a*d*e) + (b*c*f) - (c*e) - (d*f)
      # invert the determinant
      invDeterminant = 1./determinant
      invDeterminant[np.where(np.isfinite(invDeterminant)==False)] = 0.
      # calculate the point source hardened estimator
      resultFourier = invDeterminant * ( (1.-d*f)*kappaMinVarFourier + (c*f-a)*SFourier + (a*d-c)*SgFourier)
      # save to file if needed
      if path is not None:
         self.saveDataFourier(resultFourier, path)
      return resultFourier  



   def forecastN0KappaBH2(self, fC0, fCtot, lMin=1., lMax=1.e5, test=False, cache=None, sigma=0., u=None):
      # normalization for phi
      NphiFourier = self.computeQuadEstPhiNormalizationFFT(fC0, fCtot, lMin=lMin, lMax=lMax, test=test, cache=cache)
      # convert from phi to kappa
      NkappaFourier = 0.25 * self.l**4 * NphiFourier
      # normalization for S
      NSFourier = self.computeSNormalizationFFT(fCtot, lMin=lMin, lMax=lMax, test=test, cache=cache, sigma=0.)
      # normalization for Sg
      NSgFourier = self.computeSNormalizationFFT(fCtot, lMin=lMin, lMax=lMax, test=test, cache=cache, sigma=sigma, u=u)
      # response kappa, S
      RkappaSFourier = self.computeResponseFFT(fC0, fCtot, lMin=lMin, lMax=lMax, test=test, cache=cache, sigma=0.)
      # response kappa, Sg
      RkappaSgFourier = self.computeResponseFFT(fC0, fCtot, lMin=lMin, lMax=lMax, test=test, cache=cache, sigma=sigma, u=u)
      # response S, Sg
      RSSgFourier = self.computeResponseSourceSourceFFT(fC0, fCtot, lMin=lMin, lMax=lMax, test=test, cache=cache, sigma=sigma, u=u)

      a = NkappaFourier * RkappaSFourier
      b = NSFourier * RkappaSFourier
      c = NkappaFourier * RkappaSgFourier
      e = NSgFourier * RkappaSgFourier
      d = NSFourier * RSSgFourier
      f = NSgFourier * RSSgFourier
      determinant = 1. - (a*b) + (a*d*e) + (b*c*f) - (c*e) - (d*f)
      invDeterminant = 1./determinant
      invDeterminant[np.where(np.isfinite(invDeterminant)==False)] = 0.
      prefactor = 1. - d*f
      # keep only the real part (the imag. part should be zero, but is tiny in practice)
      nk = np.real(NkappaFourier)
      invDet = np.real(invDeterminant)
      prefactor = np.real(prefactor)
      # remove the nans
      nk = np.nan_to_num(nk)
      invDet = np.nan_to_num(invDet)
      prefactor = np.nan_to_num(prefactor)
      # make sure every value is positive
      nk = np.abs(nk)
      invDet = np.abs(invDet)
      prefactor = np.abs(prefactor)
      # fix the issue of the wrong ell=0 value
      # replace it by the value lx=0, ly=fundamental
      nk[0,0] = nk[0,1]
      invDet[0,0] = invDet[0,1]
      prefactor[0,0] = prefactor[0,1]
      
      nkBH3 = nk * invDet * prefactor

      # interpolate
      where = (self.l.flatten()>0.)*(self.l.flatten()<2.*lMax)
      L = self.l.flatten()[where]
      N = nkBH3.flatten()[where]
      lnfln = interp1d(np.log(L), np.log(N), kind='linear', bounds_error=False, fill_value=np.inf)

      where = lnfln(np.log(L)) < np.inf
      lmin = float(L[where][0])   
            
      f = lambda l: np.exp(lnfln(np.log(np.maximum(l,lmin))))
      return f



   ###############################################################################
   ###############################################################################
   # Gradient cleaned estimator


   def computeGradientCleanedNormalizationFFT(self, fC0, fCtot, lMin=1., lMax=1.e5, cutoff=2000., test=False, cache=None):
      """the normalization is N_l^phiphi,
      Works great, and super fast.
      """
   
      # Actual calculation
      def doCalculation():
         print("Doing full calculation: computeGradientCleanedNormalizationFFT")
         # inverse-var weighted map
         def f(l):
            if (l<lMin) or (l>lMax):
               return 0.
            result = 1./fCtot(l)
            if not np.isfinite(result):
               result = 0.
            return result
         iVarFourier = np.array(list(map(f, self.l.flatten())))
         iVarFourier = iVarFourier.reshape(self.l.shape)
         iVar = self.inverseFourier(dataFourier=iVarFourier)

         # low inverse-var weighted map
         def f(l):
            if (l<lMin) or (l>min(lMax,cutoff)):
               return 0.
            result = 1./fCtot(l)
            if not np.isfinite(result):
               result = 0.
            return result
         iVarLowFourier = np.array(list(map(f, self.l.flatten())))
         iVarLowFourier = iVarLowFourier.reshape(self.l.shape)
         iVarLow = self.inverseFourier(dataFourier=iVarLowFourier)

         # C map
         def f(l):
            if (l<lMin) or (l>lMax):
               return 0.
            result = divide(fC0(l)**2, fCtot(l))
            if not np.isfinite(result):
               result = 0.
            return result
         CFourier = np.array(list(map(f, self.l.flatten())))
         CFourier = CFourier.reshape(self.l.shape)
      
         # C low map
         def f(l):
            if (l<lMin) or (l>min(lMax,cutoff)):
               return 0.
            result = divide(fC0(l)**2, fCtot(l))
            if not np.isfinite(result):
               result = 0.
            return result
         CLowFourier = np.array(list(map(f, self.l.flatten())))
         CLowFourier = CLowFourier.reshape(self.l.shape)

         # term 1xA
         term1xA = self.inverseFourier(dataFourier= self.lx**2 * CLowFourier)
         term1xA *= iVar/2.
         term1xAFourier = self.fourier(data=term1xA)
         term1xAFourier *= self.lx**2
         # term 1xB
         term1xB = self.inverseFourier(dataFourier= self.lx**2 * CFourier)
         term1xB *= iVarLow/2.
         term1xBFourier = self.fourier(data=term1xB)
         term1xBFourier *= self.lx**2
         # term 1x
         term1xFourier = term1xAFourier + term1xBFourier
         #
         # term 1yA
         term1yA = self.inverseFourier(dataFourier= self.ly**2 * CLowFourier)
         term1yA *= iVar/2.
         term1yAFourier = self.fourier(data=term1yA)
         term1yAFourier *= self.ly**2
         # term 1yB
         term1yB = self.inverseFourier(dataFourier= self.ly**2 * CFourier)
         term1yB *= iVarLow/2.
         term1yBFourier = self.fourier(data=term1yB)
         term1yBFourier *= self.ly**2
         # term 1y
         term1yFourier = term1yAFourier + term1yBFourier
         #
         # term 1xyA
         term1xyA = self.inverseFourier(dataFourier= 2. * self.lx * self.ly * CLowFourier)
         term1xyA *= iVar/2.
         term1xyAFourier = self.fourier(data=term1xyA)
         term1xyAFourier *= self.lx * self.ly
         # term 1xyB
         term1xyB = self.inverseFourier(dataFourier= 2. * self.lx * self.ly * CFourier)
         term1xyB *= iVarLow/2.
         term1xyBFourier = self.fourier(data=term1xyB)
         term1xyBFourier *= self.lx * self.ly
         # term 1xy
         term1xyFourier = term1xyAFourier + term1xyBFourier
         
         if test:
            self.plotFourier(term1xFourier)
            self.plotFourier(term1yFourier)
            self.plotFourier(term1xyFourier)
            self.plotFourier(term1xFourier+term1yFourier+term1xyFourier)
         

         # WF map
         def f(l):
            if (l<lMin) or (l>lMax):
               return 0.
            result = divide(fC0(l), fCtot(l))
            # artificial factor of i such that f(-l) = f(l)*,
            # such that f(x) is real
            result *= 1.j
            if not np.isfinite(result):
               result = 0.
            return result
         WFFourier = np.array(list(map(f, self.l.flatten())))
         WFFourier = WFFourier.reshape(self.l.shape)

         # WF low map
         def f(l):
            if (l<lMin) or (l>min(lMax,cutoff)):
               return 0.
            result = divide(fC0(l), fCtot(l))
            # artificial factor of i such that f(-l) = f(l)*,
            # such that f(x) is real
            result *= 1.j
            if not np.isfinite(result):
               result = 0.
            return result
         WFLowFourier = np.array(list(map(f, self.l.flatten())))
         WFLowFourier = WFLowFourier.reshape(self.l.shape)

         # term 2
         term2_x = self.inverseFourier(dataFourier= self.lx * WFFourier)
         term2_y = self.inverseFourier(dataFourier= self.ly * WFFourier)
         term2Low_x = self.inverseFourier(dataFourier= self.lx * WFLowFourier)
         term2Low_y = self.inverseFourier(dataFourier= self.ly * WFLowFourier)
         #
         # term 2x
         term2x = term2_x * term2Low_x
         term2xFourier = self.fourier(data=term2x)
         term2xFourier *= self.lx**2
         # minus sign to correct the artificial i**2
         term2xFourier *= -1.
         #
         # term 2y
         term2y = term2_y * term2Low_y
         term2yFourier = self.fourier(data=term2y)
         term2yFourier *= self.ly**2
         # minus sign to correct the artificial i**2
         term2yFourier *= -1.
         #
         # term 2xyA
         term2xyA = term2_x * term2Low_y
         term2xyAFourier = self.fourier(data=term2xyA)
         term2xyAFourier *= self.lx * self.ly
         # minus sign to correct the artificial i**2
         term2xyAFourier *= -1.
         #
         # term 2xyB
         term2xyB = term2Low_x * term2_y
         term2xyBFourier = self.fourier(data=term2xyB)
         term2xyBFourier *= self.lx * self.ly
         # minus sign to correct the artificial i**2
         term2xyBFourier *= -1.
         #
         # term 2xy
         term2xyFourier = term2xyAFourier + term2xyBFourier

         if test:
            self.plotFourier(term2xFourier)
            self.plotFourier(term2yFourier)
            self.plotFourier(term2xyFourier)
            self.plotFourier(term2xFourier+term2yFourier+term2xyFourier)

         # add all terms
         resultFourier = term1xFourier + term1yFourier + term1xyFourier
         resultFourier += term2xFourier + term2yFourier + term2xyFourier

         # cut off the high ells from phi normalization map
         f = lambda l: (l<=2.*lMax)
         resultFourier = self.filterFourierIsotropic(f, dataFourier=resultFourier, test=False)

         # invert
         resultFourier = 1./resultFourier
         resultFourier[np.where(np.isfinite(resultFourier)==False)] = 0.

         if test:
            plt.loglog(self.l.flatten(), resultFourier.flatten(), 'b.')
            plt.show()

         a = 0.25 * self.l**4 * resultFourier
      
         return a

      return doCalculation()


   def computeGradientCleanedSecondIntegralFFT(self, fC0, fCtot, lMin=1., lMax=1.e5, cutoff=2000., test=False, cache=None):
      """the normalization is N_l^phiphi,
      Works great, and super fast.
      """
   
      # Actual calculation
      def doCalculation():
         print("Doing full calculation: computeGradientCleanedNormalizationFFT")
         # inverse-var weighted map
         def f(l):
            if (l<lMin) or (l>lMax):
               return 0.
            result = 1./fCtot(l)
            if not np.isfinite(result):
               result = 0.
            return result
         iVarFourier = np.array(list(map(f, self.l.flatten())))
         iVarFourier = iVarFourier.reshape(self.l.shape)
         iVar = self.inverseFourier(dataFourier=iVarFourier)

         # low inverse-var weighted map
         def f(l):
            if (l<lMin) or (l>min(lMax,cutoff)):
               return 0.
            result = 1./fCtot(l)
            if not np.isfinite(result):
               result = 0.
            return result
         iVarLowFourier = np.array(list(map(f, self.l.flatten())))
         iVarLowFourier = iVarLowFourier.reshape(self.l.shape)
         iVarLow = self.inverseFourier(dataFourier=iVarLowFourier)

         # C map
         def f(l):
            if (l<lMin) or (l>lMax):
               return 0.
            result = divide(fC0(l)**2, fCtot(l))
            if not np.isfinite(result):
               result = 0.
            return result
         CFourier = np.array(list(map(f, self.l.flatten())))
         CFourier = CFourier.reshape(self.l.shape)
      
         # C low map
         def f(l):
            if (l<lMin) or (l>min(lMax,cutoff)):
               return 0.
            result = divide(fC0(l)**2, fCtot(l))
            if not np.isfinite(result):
               result = 0.
            return result
         CLowFourier = np.array(list(map(f, self.l.flatten())))
         CLowFourier = CLowFourier.reshape(self.l.shape)         

         # WF map
         def f(l):
            if (l<lMin) or (l>lMax):
               return 0.
            result = divide(fC0(l), fCtot(l))
            # artificial factor of i such that f(-l) = f(l)*,
            # such that f(x) is real
            result *= 1.j
            if not np.isfinite(result):
               result = 0.
            return result
         WFFourier = np.array(list(map(f, self.l.flatten())))
         WFFourier = WFFourier.reshape(self.l.shape)

         # WF low map
         def f(l):
            if (l<lMin) or (l>min(lMax,cutoff)):
               return 0.
            result = divide(fC0(l), fCtot(l))
            # artificial factor of i such that f(-l) = f(l)*,
            # such that f(x) is real
            result *= 1.j
            if not np.isfinite(result):
               result = 0.
            return result
         WFLowFourier = np.array(list(map(f, self.l.flatten())))
         WFLowFourier = WFLowFourier.reshape(self.l.shape)
         
         # term 1x
         term1x = self.inverseFourier(dataFourier= self.lx**2 * CLowFourier)
         term1x *= iVarLow
         term1xFourier = self.fourier(data=term1x)
         term1xFourier *= self.lx**2
         #
         # term 1y
         term1y = self.inverseFourier(dataFourier= self.ly**2 * CLowFourier)
         term1y *= iVarLow
         term1yFourier = self.fourier(data=term1y)
         term1yFourier *= self.ly**2
         #
         # term 1xy
         term1xy = self.inverseFourier(dataFourier= 2. * self.lx * self.ly * CLowFourier)
         term1xy *= iVarLow
         term1xyFourier = self.fourier(data=term1xy)
         term1xyFourier *= self.lx * self.ly
         #
         term2_x = self.inverseFourier(dataFourier= self.lx * WFFourier)
         term2_y = self.inverseFourier(dataFourier= self.ly * WFFourier)
         term2Low_x = self.inverseFourier(dataFourier= self.lx * WFLowFourier)
         term2Low_y = self.inverseFourier(dataFourier= self.ly * WFLowFourier)
         # term 2x
         term2x = term2Low_x**2
         term2xFourier = self.fourier(data=term2x)
         term2xFourier *= self.lx**2
         # minus sign to correct the artificial i**2
         term2xFourier *= -1.
         #
         # term 2y
         term2y = term2Low_y**2
         term2yFourier = self.fourier(data=term2y)
         term2yFourier *= self.ly**2
         # minus sign to correct the artificial i**2
         term2yFourier *= -1.
         #
         # term 2xy
         term2xy = 2. * term2Low_x * term2Low_y
         term2xyFourier = self.fourier(data=term2xy)
         term2xyFourier *= self.lx * self.ly
         # minus sign to correct the artificial i**2
         term2xyFourier *= -1.

         # add all terms
         resultFourier = term1xFourier + term1yFourier + term1xyFourier
         resultFourier += term2xFourier + term2yFourier + term2xyFourier

         # cut off the high ells from phi normalization map
         f = lambda l: (l<=2.*lMax)
         resultFourier = self.filterFourierIsotropic(f, dataFourier=resultFourier, test=False)

         # invert
         resultFourier = 1./resultFourier
         resultFourier[np.where(np.isfinite(resultFourier)==False)] = 0.

         if test:
            plt.loglog(self.l.flatten(), resultFourier.flatten(), 'b.')
            plt.show()

         b = 0.25 * self.l**4 * resultFourier
      
         return b

      return doCalculation()



   def gradientCleanedNonNorm(self, fC0, fCtot, lMin= 1., lMax=1.e5, dataFourier=None, dataFourier2=None, cutoff=2000., test=False):
      """non-normalized GC estimator
      fC0: ulensed power spectrum
      fCtot: lensed power spectrum + noise
      dataFourier = low and high ell map
      dataFourier2 = low ell map
      """
      if dataFourier2 is None:
         print("error: no input for low ell map")

      # inverse-var weighted map
      def f(l):
         # cut off the high ells from input map
         if (l<lMin) or (l>lMax):
            return 0.
         result = 1./fCtot(l)
         if not np.isfinite(result):
            result = 0.
         return result
      iVarDataFourier = self.filterFourierIsotropic(f, dataFourier=dataFourier, test=test)
      iVarData = self.inverseFourier(iVarDataFourier)

      # inverse-var weighted low ell map 
      def f(l):
         # cut off the high ells from input map
         if (l<lMin) or (l>min(lMax,cutoff)):
            return 0.
         result = 1./fCtot(l)
         if not np.isfinite(result):
            result = 0.
         return result
      iVarDataLowFourier = self.filterFourierIsotropic(f, dataFourier=dataFourier2, test=test)
      iVarDataLow = self.inverseFourier(iVarDataLowFourier)

      # Wiener-filter the map
      def f(l):
         # cut off the high ells from input map
         if (l<lMin) or (l>lMax):
            return 0.
         result = divide(fC0(l), fCtot(l))
         if not np.isfinite(result):
            result = 0.
         return result
      WFDataFourier = self.filterFourierIsotropic(f, dataFourier=dataFourier, test=test)

      # Wiener-filter the low ell map
      def f(l):
         # cut off the high ells from input map
         if (l<lMin) or (l>min(lMax,cutoff)):
            return 0.
         result = divide(fC0(l), fCtot(l))
         if not np.isfinite(result):
            result = 0.
         return result
      WFDataLowFourier = self.filterFourierIsotropic(f, dataFourier=dataFourier2, test=test)
      # get Wiener-filtered gradient map
      WFDataXFourier, WFDataYFourier = self.computeGradient(dataFourier=WFDataFourier)
      WFDataX = self.inverseFourier(dataFourier=WFDataXFourier)
      WFDataY = self.inverseFourier(dataFourier=WFDataYFourier)
      # product in real space
      productDataX = iVarDataLow * WFDataX
      productDataY = iVarDataLow * WFDataY
      productDataXFourier = self.fourier(data=productDataX)
      productDataYFourier = self.fourier(data=productDataY)
      # take divergence
      divergenceDataFourier = self.computeDivergence(productDataXFourier, productDataYFourier)
      # get Wiener-filtered gradient map
      WFDataXFourier, WFDataYFourier = self.computeGradient(dataFourier=WFDataLowFourier)
      WFDataX = self.inverseFourier(dataFourier=WFDataXFourier)
      WFDataY = self.inverseFourier(dataFourier=WFDataYFourier)
      # product in real space
      productDataX = iVarData * WFDataX
      productDataY = iVarData * WFDataY
      productDataXFourier = self.fourier(data=productDataX)
      productDataYFourier = self.fourier(data=productDataY)
      # take divergence
      divergenceDataFourier += self.computeDivergence(productDataXFourier, productDataYFourier)
      divergenceDataFourier /= self.l**2.
      # correct a factor of -1
      divergenceDataFourier *= -1.
      # cut off the high ells
      f = lambda l: (l<=2.*lMax)
      divergenceDataFourier = self.filterFourierIsotropic(f, dataFourier=divergenceDataFourier, test=test)
      
      return divergenceDataFourier



   def computeGradientCleanedNorm(self, fC0, fCtot, lMin=1., lMax=1.e5, dataFourier=None, dataFourier2=None, cutoff=2000., test=False, cache=None, path=None):
      """normalized GC estimator
      fC0: ulensed power spectrum
      fCtot: lensed power spectrum + noise
      dataFourier = low and high ell map
      dataFourier2 = low ell map
      """
      if dataFourier2 is None:
         print("error: no input for low ell map")

      nonNorm = self.gradientCleanedNonNorm(fC0=fC0, fCtot=fCtot, lMin=lMin, lMax=lMax, dataFourier=dataFourier, dataFourier2=dataFourier2, cutoff=cutoff, test=test)
      norm = self.computeGradientCleanedNormalizationFFT(fC0, fCtot, lMin=lMin, lMax=lMax, cutoff=cutoff, test=test, cache=cache)
      resultFourier = norm * nonNorm
      if path is not None:
         self.saveDataFourier(resultFourier, path)
      return resultFourier



   def forecastGradientCleanedKappa(self, fC0, fCtot, lMin=1., lMax=1.e5, cutoff=2000., test=False, cache=None):
      print("computing the gradient cleaned noise")
      # lMax must be >= 2000.
      # Standard reconstruction noise
      a = self.computeGradientCleanedNormalizationFFT(fC0, fCtot, lMin=lMin, lMax=lMax, cutoff=cutoff, test=test, cache=cache)
      b = self.computeGradientCleanedSecondIntegralFFT(fC0, fCtot, lMin=lMin, lMax=lMax, cutoff=cutoff, test=test, cache=cache)
      # keep only the real part (the imag. part should be zero, but is tiny in practice)
      a = np.real(a)
      b = np.real(b)
      # remove the nans
      a = np.nan_to_num(a)
      b = np.nan_to_num(b)
      # make sure every value is positive
      a = np.abs(a)
      b = np.abs(b)
      # fix the issue of the wrong ell=0 value
      # replace it by the value lx=0, ly=fundamental
      a[0,0] = a[0,1]
      b[0,0] = b[0,1]
      gc = 0.5 * a + 0.5 * (a**2.) / b
      # interpolate
      where = (self.l.flatten()>0.)*(self.l.flatten()<2.*lMax)
      L = self.l.flatten()[where]
      N = gc.flatten()[where]
      lnfln = interp1d(np.log(L), np.log(N), kind='linear', bounds_error=False, fill_value=np.inf)
      where = lnfln(np.log(L)) < np.inf
      lmin = float(L[where][0])       
      f = lambda l: np.exp(lnfln(np.log(np.maximum(l,lmin))))
      return f
 
