'''
#############################################################################
Acknowledgments to paper: Kalinova et al. 2016, MNRAS
"The inner mass distribution of late-type spiral galaxies from SAURON stellar kinematic maps".

Copyright (c) 2016, Veselina Kalinova, Dario Colombo, Erik Rosolowsky
University of Alberta
E-mails: kalinova@mpifr.de, dcolombo@mpifr.de, erosolow@ualberta.ca
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided 
that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of 
conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of 
conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
Neither the name of the Astropy Team nor the names of its contributors may be used to endorse 
or promote products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR 
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND 
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, 
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

############################################################################

NAME:
  JAM_MCMC

PURPOSE:
   This procedure calculates the Markov Chain Monte Carlo (MCMC) circular 
   velocity in the equatorial plane of an axisymmetric galaxy model described 
   by a Multi-Gaussian Expansion parametrization (MGE; Monnet,Bacon & Emsellem 1992). 
   We use the "EMCEE code" of Foreman-Mackey et al. 2013 (http://dan.iel.fm/emcee/current/),
   an implementation of an affine invariant ensemble sampler for the MCMC method of parameter estimations,
   together with the Jeans Anisotropic MGE (JAM) code of the dynamics of axisymmetric galaxies by Cappellari 2008 
   (http://purl.org/cappellari/software).

CALLING SEQUENCE:

  res = JAM_MCMC(gal, qmin, dist, surf_lum, sigobs_lum, qobs_lum,surf_pot, sigobs_pot, qobs_pot, \
    xmod, ymod, Vrmsbin, dVrmsbin, Mbh, sigmapsf, pixscale, nwalks, burn_steps, steps, threads,ideg_in)


INPUT PARAMETERS (as described in "EMCEE code" and "JAM code"):
  GAL: name of the galaxy 
  QMIN: the axis ratio of the flattest gaussian in Multi-Gaussian Expansion (MGE) method.
  DISTANCE: distance of the galaxy in Mpc.
  SURF_LUM: vector of length N containing the peak surface brightness of the
      MGE Gaussians describing the galaxy surface brightness in units of
      Lsun/pc^2 (solar luminosities per parsec^2).
  SIGMA_LUM: vector of length N containing the dispersion in arcseconds of
      the MGE Gaussians describing the galaxy surface brightness.
  QOBS_LUM: vector of length N containing the observed axial ratio of the MGE
      Gaussians describing the galaxy surface brightness.
  SURF_POT: vector of length M containing the peak value of the MGE Gaussians
      describing the galaxy surface density in units of Msun/pc^2 (solar
      masses per parsec^2). This is the MGE model from which the model
      potential is computed.
    - In a common usage scenario, with a self-consistent model, one has
      the same Gaussians for both the surface brightness and the potential.
      This implies SURF_POT = SURF_LUM, SIGMA_POT = SIGMA_LUM and
      QOBS_POT = QOBS_LUM. The global M/L of the model is fitted by the
      routine when passing the RMS and ERMS keywords with the observed kinematics.
  SIGMA_POT: vector of length M containing the dispersion in arcseconds of
      the MGE Gaussians describing the galaxy surface density.
  QOBS_POT: vector of length M containing the observed axial ratio of the MGE
      Gaussians describing the galaxy surface density.
  XBIN: Vector of length P with the X coordinates in arcseconds of the bins
      (or pixels) at which one wants to compute the model predictions. The
      X-axis is assumed to coincide with the galaxy projected major axis. The
      galaxy center is at (0,0).
  YBIN: Vector of length P with the Y coordinates in arcseconds of the bins
      (or pixels) at which one wants to compute the model predictions. The
      Y-axis is assumed to concide with the projected galaxy symmetry axis.
  VRMSBIN: Vector of length P with the input observed stellar
      V_RMS=sqrt(velBin^2 + sigBin^2) at the coordinates positions given by
      the vectors XBIN and YBIN.
  DVRMSBIN: Error of VRMSBIN (vector of length P)
  MBH: Mass of a nuclear supermassive black hole in solar masses.
  SIGMAPSF: Vector of length Q with the dispersion in arcseconds of the
      circular Gaussians describing the PSF of the observations.    
  PIXSIZE: Size in arcseconds of the (square) spatial elements at which the
      kinematics is obtained. This may correspond to the side of the spaxel
      or lenslets of an integral-field spectrograph. This size is used to
      compute the kernel for the seeing and aperture convolution.    
  NWALKS: Number of Goodman & Weare walkers, which should be equal or 
      greater than twice the dimension, i.e twice the number of the fitted parameters)
  BURN_STEPS: Number of the steps for the burn-in process
  STEPS: Number of the steps after burn-in process, i.e steps for the final chains of the parameters
  THREADS: Number of threads to use for parallelization, where threads > 1 is for multiprocessing 
  IDEG_IN: galaxy inclination in degrees


RETURN (as described in "EMCEE code" and "JAM code"):
  TABLE: median values of the distributions of the velocity anisotropy, mass-to-light ratio and inclination, 
    together with their 75 and 25 percentile errors 
  BURN_CHAINS: (burn-in phase) A pointer to the Markov chain itself, where the shape of this array is (k, iterations, dim).
  BURN_LNS: (burn-in phase) A pointer to the matrix of the value of lnprobfn (a function that takes a vector in the parameter space 
    as input and returns the natural logarithm of the posterior probability for that position) produced at each step 
    for each walker. The shape is (k, iterations).
  BURN_FLATCHAINS: (burn-in phase) A shortcut for accessing burn-in chain flattened along the zeroth (walker) axis.
  BURN_FLATLNS: (burn-in phase) A shortcut to return the equivalent of lnprobability but aligned to flatchain rather than chain.
  FINAL_CHAINS: (posterior phase) A pointer to the Markov chain itself, where the shape of this array is (k, iterations, dim)
  FINAL_LNS: (posterior phase) A pointer to the matrix of the value of lnprobfn (a function that takes a vector in the parameter space 
    as input and returns the natural logarithm of the posterior probability for that position) produced at each step 
    for each walker. The shape is (k, iterations).
  FINAL_FLATCHAINS: (posterior phase) A shortcut for accessing burn-in chain flattened along the zeroth (walker) axis.
  FINAL_FLATLNS: (posterior phase) A shortcut to return the equivalent of lnprobability but aligned to flatchain rather than chain.
  VRMSMOD:Vector of length P with the model predictions for the velocity
      second moments V_RMS ~ sqrt(vel^2 + sig^2) for each bin.
  CHI2: Reduced chi^2 describing the quality of the fit
  FLUX:In output this contains a vector of length P with the PSF-convolved
       MGE surface brightness of each bin in Lsun/pc^2, used to plot the
       isophotes on the model results.
  
METHODS:
  rms_logprob: sub-routine for defining the log-probability
  runjam: sub-routine for running of JAM-MCMC analysis
  make_fig: sub-routine for plotting of burn-in and final chains of the parameters, corner figure, and the best fit VRMS model.
  make_vcirc: sub-routine for plotting the circular velocity of the galaxy using the distributions of the parameters and the
  best fit value of the M/L from JAM-MCMC analysis.

USAGE EXAMPLE:
   A simple usage example "test_JAM_MCMC.py" is given in the same directory. 

REQUIRED ROUTINES:
      By M. Cappellari (included in the JAM distribution; http://purl.org/cappellari/software):
      - JAM modelling package in python (jam_axi_rms.py, mge_vcirc.py, cap_symmetrize_velfield.py, cap_loess_2d.py)
      
      By D. Foreman-Mackey (http://dan.iel.fm/emcee/current/)
      - emcee
      - corner

MODIFICATION HISTORY:
V1.0: Written and tested as part of the implementation of
    JAM-MCMC method described in Kalinova et al. (2016). 
    Veselina Kalinova, Dario Colombo, Erik Rosolowsky; 
    University of Alberta, 2016

'''

######################
# MCMC JAM model
######################
#...Import packages
from jam_axi_rms import jam_axi_rms
from mge_vcirc import mge_vcirc
from cap_plot_velfield import plot_velfield

import emcee
import corner
import fish
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
from astropy.table import Table
from pdb import set_trace as stop



# sub-routine for defining the log-probability
def rms_logprob(p,surf_lum, sigobs_lum, qobs_lum, surf_pot, sigobs_pot, qobs_pot, \
                       Mbh, dist, xbin, ybin, Vrmsbin, dVrmsbin, qmin, sigmapsf, pixsize,ideg_in):
    
    beta_scalar=p[0]
    ml=p[1]
    

    ideg_lim = np.arccos(qmin)*180./np.pi

    # defining the distribution of the priors
    priors =  ss.uniform.logpdf(beta_scalar,loc=-1.0,scale=2.0)+\
              ss.uniform.logpdf(ml,loc=0.0,scale=10.0)
               
    
    #condition related to the limitations of the priors
    if np.isfinite(priors) == False:
        return -np.inf
    
    #...Call JAM code
    _, _, chi2,_ = \
    jam_axi_rms(surf_lum, sigobs_lum, qobs_lum, surf_pot, sigobs_pot, qobs_pot,
                     ideg_in, Mbh, dist, xbin, ybin, plot=False, quiet=True, rms=Vrmsbin,ml=ml, erms=dVrmsbin,
                      sigmapsf=sigmapsf, beta=beta_scalar+(surf_lum*0), pixsize=pixsize)
    
    #calculating the log-probability throght the chi2 of the Vrms_model fit 
    lp = -chi2*len(Vrmsbin) + priors
    if np.isnan(lp):
        return -np.inf

    return lp

  

def runjam(gal, qmin, dist, surf_lum, sigobs_lum, qobs_lum,
  surf_pot, sigobs_pot, qobs_pot, xbin, ybin, Vrmsbin, dVrmsbin, Mbh, sigmapsf, pixscale, 
  nwalks, burn_steps, steps, threads,ideg_in):
  
  ideg_lim = np.arccos(qmin)*180./np.pi

  #print 'ideg_enter=',ideg_lim

  # Set the number of the free parameters and the walkers
  ndim, nwalkers = 2,nwalks
  p0 = np.zeros((nwalkers,ndim))
  # set the start position of the walkers
  p0[:,0] = np.random.uniform(-1,1.0,nwalkers) 
  p0[:,1] = np.random.uniform(0,10,nwalkers) 

  #stop()
  # Call EMCEE code
  sampler = emcee.EnsembleSampler(nwalkers, ndim, rms_logprob,
                                    args=[surf_lum, sigobs_lum, qobs_lum, surf_pot, sigobs_pot, qobs_pot, \
                                            Mbh, dist, xbin, ybin, Vrmsbin, dVrmsbin, qmin, sigmapsf, pixscale,ideg_in], \
                                            threads=threads)

  print "%&%&%&%&%&%&%&%&%&%&%&%&%&"
  print   "Run MCMC analysis"
  print "%&%&%&%&%&%&%&%&%&%&%&%&%&"

  # result from the EMCEE code
  print("Burn-in...")
  #pos, prob, state = sampler.run_mcmc(p0, burn_steps)

  peixe = fish.ProgressFish(total=burn_steps)
  for j, results in enumerate(sampler.sample(p0, iterations=burn_steps)):
    peixe.animate(amount=j+1)

  pos = results[0]

  burn_chains = sampler.chain.copy()
  burn_lns = sampler.lnprobability.copy()
  burn_flatchains = sampler.flatchain.copy()
  burn_flatlns = sampler.flatlnprobability.copy()
  ####################################################
  #... save the data in a file
  np.savez('data_output/Burn_in/'+gal+'_burn_lnP', chain_JAM=sampler.chain, lnprobability_JAM=sampler.lnprobability)
  #... save the data in a file
  np.savez('data_output/Burn_in/'+gal+'_burn_flatlnP', flatchain_JAM=sampler.flatchain, flatlnprobability_JAM=sampler.flatlnprobability)    
  ##################################################  

  # RUN again MCMC after burn-in
  print("Running MCMC...")
  sampler.reset()
  #pos,prob,state = sampler.run_mcmc(pos, 10)

  peixe = fish.ProgressFish(total=steps)
  for j, results in enumerate(sampler.sample(pos, iterations=steps)):
    peixe.animate(amount=j+1)


  final_chains = sampler.chain
  final_lns = sampler.lnprobability
  final_flatchains = sampler.flatchain
  final_flatlns = sampler.flatlnprobability

  ####################################################
  #... save the data in a file
  np.savez('data_output/Chains/'+gal+'_final_lnP', chain_JAM=sampler.chain, lnprobability_JAM=sampler.lnprobability)
  #... save the data in a file
  np.savez('data_output/Chains/'+gal+'_final_flatlnP', flatchain_JAM=sampler.flatchain, flatlnprobability_JAM=sampler.flatlnprobability)
  ####################################################

  # make distributions of the parameters
  beta_dist = final_flatchains[:,0]
  beta_md=np.median(beta_dist)
  beta_plus=np.percentile(final_flatchains[:,0], 75)- np.median(final_flatchains[:,0])
  beta_minus=np.median(final_flatchains[:,0]) - np.percentile(final_flatchains[:,0], 25)

  ml_dist = final_flatchains[:,1]
  ml_md=np.median(ml_dist)
  ml_plus=np.percentile(final_flatchains[:,1], 75)- np.median(final_flatchains[:,1])
  ml_minus=np.median(final_flatchains[:,1]) - np.percentile(final_flatchains[:,1], 25)
  

  #-----------------------------------------------
  
  medians = [beta_md, ml_md, ideg_in]

  # Array for the upper 75th percentiles
  ups = [beta_plus, ml_plus, 0.0]

  # Array for the lower 25th percentiles
  lws = [beta_minus, ml_minus, 0.0]

  # make table 
  table = Table([medians, ups, lws], names = ['#medians(Bz, M/L, ideg_in)','ups','lws '])
  table.write("data_output/Tables/"+gal+"_bestfit.txt",format="ascii.tab",delimiter=",")
  #-------------------------------------------------
  print "%&%&%&%&%&%&%&%&%&%&%&%&%&"
  print   "Final best fit values"
  print "%&%&%&%&%&%&%&%&%&%&%&%&%&"
  print 'Beta_z:' , np.median(final_flatchains[:,0]), \
      '+', np.percentile(final_flatchains[:,0], 75) - np.median(final_flatchains[:,0]),\
      '-', np.median(final_flatchains[:,0]) - np.percentile(final_flatchains[:,0], 25) 
  print 'M/L: ', np.median(final_flatchains[:,1]), \
      '+', np.percentile(final_flatchains[:,1], 75) - np.median(final_flatchains[:,1]),\
      '-', np.median(final_flatchains[:,1]) - np.percentile(final_flatchains[:,1], 25) 
  #---------------------------------------------------------------


  print "%&%&%&%&%&%&%&%&%&%&%&%&%&"
  print      "Final Vrms model"
  print        'ideg_in=', ideg_in
  print "%&%&%&%&%&%&%&%&%&%&%&%&%&"  
  rmsmodel, _, chi2, flux = \
    jam_axi_rms(surf_lum, sigobs_lum, qobs_lum, surf_pot, sigobs_pot, qobs_pot,
                     ideg_in, Mbh, dist, xbin, ybin, plot=False, rms=Vrmsbin,ml=ml_md, 
                      erms=dVrmsbin, sigmapsf=sigmapsf, beta=beta_md+(surf_lum*0), pixsize=pixscale)

  chi2 = chi2*len(Vrmsbin)/(len(Vrmsbin)-2)

  return table, burn_chains, burn_lns, burn_flatchains, burn_flatlns, \
  final_chains, final_lns, final_flatchains, final_flatlns, rmsmodel, chi2, flux


class JAM_MCMC(object):

  def __init__(self, gal, qmin, dist, surf_lum, sigobs_lum, qobs_lum,
  surf_pot, sigobs_pot, qobs_pot, xbin, ybin, Vrmsbin, dVrmsbin, Mbh, sigmapsf, pixscale, nwalks,
  burn_steps, steps, threads,ideg_in):

    if nwalks < 6:
      print("NWALKERS must be equal or greater than twice the dimension")
      nwalks = 6

    # Galaxy parameters
    self.gal = gal
    self.qmin = qmin
    self.dist = dist


    # MGE inputs
    self.surf_lum = surf_lum
    self.sigobs_lum = sigobs_lum
    self.qobs_lum = qobs_lum
    self.surf_pot = surf_pot
    self.sigobs_pot = sigobs_pot
    self.qobs_pot = qobs_pot

    # Kinematic inputs
    self.xbin = xbin
    self.ybin = ybin
    self.Vrmsbin = Vrmsbin
    self.dVrmsbin = dVrmsbin
    self.Mbh = Mbh
    self.sigmapsf = sigmapsf
    self.pixscale = pixscale
    self.ideg_in=ideg_in
  


    # Run JAM with MCMC
    self.table, self.burn_chains, self.burn_lns, self.burn_flatchains, self.burn_flatlns, \
    self.final_chains, self.final_lns, self.final_flatchains, self.final_flatlns, self.Vrmsmod, self.chi2, self.flux = runjam(self.gal, self.qmin, self.dist, \
      self.surf_lum, self.sigobs_lum, self.qobs_lum, self.surf_pot, self.sigobs_pot, self.qobs_pot, \
      self.xbin, self.ybin, self.Vrmsbin, self.dVrmsbin, self.Mbh, self.sigmapsf, self.pixscale, nwalks, burn_steps, steps, threads,self.ideg_in)


  def make_fig(self, gal, vmin = None, vmax = None):

    #------------------------------------------
    #Plot the distribution of the burn-in process
    fig = plt.figure(figsize=(5,10))
    plt.subplot(2,1,1)
    plt.title(self.gal)
    plt.plot(self.burn_chains[:,:,0].T)
    plt.ylabel(r'Chain for $\beta^{JAM}_z$')
    plt.subplot(2,1,2)
    plt.plot(self.burn_chains[:,:,1].T)
    plt.ylabel(r'Chain for $\Upsilon$') 
    plt.tight_layout() # This tightens up the spacing
    plt.savefig('figures/Burn_in/'+self.gal+'_burnchains.png')
    plt.close()

    #------------------------------------------
    #Plot the distribution of final process
    fig = plt.figure(figsize=(5,10))
    plt.subplot(2,1,1)
    plt.title(self.gal)
    plt.plot(self.final_chains[:,:,0].T)
    plt.ylabel(r'Chain for $\beta^{JAM}_z$')
    plt.subplot(2,1,2)
    plt.plot(self.final_chains[:,:,1].T)
    plt.ylabel(r'Chain for $\Upsilon$')
    plt.tight_layout() # This tightens up the spacing
    plt.savefig('figures/Chains/'+self.gal+'_finalchains.png')
    plt.close()

    #------------------------------------------
    # Corner Figure of the final Flatchain 
    figure=corner.corner(self.final_flatchains, labels=[r"$\beta^{JAM}_z$", "$\Upsilon$"], \
      quantiles=[0.25, 0.50, 0.75],show_titles=True, title_fmt=".3f",title_args={"fontsize": 12} ) 
    figure.gca().annotate(self.gal+ ' JAM', xy=(0.5, 1.0), xycoords="figure fraction", xytext=(0, -5), \
      textcoords="offset points", ha="center", va="top")
    figure.savefig('figures/Corner/'+self.gal+'_corner.png')
    #------------------------------------------
    # Plot Vrmsbin and Vrmsmodel
            
    if (vmin is None) or (vmax is None):
      vmin, vmax = ss.scoreatpercentile(self.Vrmsbin, [0.5, 99.5])

    fig = plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plot_velfield(self.xbin, self.ybin, self.Vrmsbin, vmin=vmin, vmax=vmax, flux=self.flux, colorbar=True, \
      orientation='horizontal', xlab='arcsec', ylab='arcsec', nodots=True)
    plt.title(r"$V^{OBS}_{rms}$")

    plt.subplot(1,2,2)
    plot_velfield(self.xbin, self.ybin, self.Vrmsmod, vmin=vmin, vmax=vmax, flux=self.flux, colorbar=True, \
      xlab='arcsec', ylab='arcsec', nodots=True)
    plt.title(r"$V^{MOD}_{rms}$, $\chi^2$="+str("{0:.2f}".format(self.chi2)))
    plt.tight_layout()
    plt.savefig('figures/Vrms/'+self.gal+'_Vrms.png')
    plt.close()     


  def make_vcirc(self, rad, gal, ideg_in):

    # load the parameter distributions 
    beta_dist=self.final_flatchains[:,0]
    ml_dist=self.final_flatchains[:,1]
    
     

    vcircs = np.zeros([len(rad),len(ml_dist)])


    for i in range(len(ml_dist)):
    
      ml= ml_dist[i]

    
      #condition
      inc = np.radians(ideg_in)      # ...convert inclination to radians
      qintr = self.qobs_lum**2 - np.cos(inc)**2
    
      w=np.where(  (qintr > 0.0)  &  (np.sqrt(qintr)/np.sin(inc) > 0.05)  )
    
      surf_lum=self.surf_lum[w] 
      sigobs_lum=self.sigobs_lum[w] 
      qobs_lum=self.qobs_lum[w] 

      vcircs[:,i] = mge_vcirc(surf_lum*ml, sigobs_lum, qobs_lum, ideg_in, self.Mbh, self.dist, rad)

    vcirc_med = np.median(vcircs, axis = 1)
    vcirc_up = np.percentile(vcircs, 75, axis = 1) - np.median(vcircs, axis = 1)
    vcirc_dn = np.median(vcircs, axis = 1) - np.percentile(vcircs, 25, axis = 1)
    
    #... save Vcirc in a file
    np.savez('data_output/Vcirc/Vc_'+self.gal, rad=rad, vcirc_med=vcirc_med, vcirc_up=vcirc_up, vcirc_dn=vcirc_dn)


    #...Plot Vcirc
    fig = plt.figure(figsize=(10,6))    
    plt.clf()
    plt.plot(rad, vcirc_med, '-o')
    plt.errorbar(rad,vcirc_med, yerr = (vcirc_up, vcirc_dn), color = 'r' )
    plt.xlabel('R (arcsec)')
    plt.ylabel(r'$V^{JAM}_{c}$ (km/s)')
    plt.title(self.gal)
    plt.tight_layout()
    plt.savefig('figures/Vcirc/Vc_'+self.gal+'_Vc.png')
    plt.close()

    self.vcirc_med = vcirc_med
    self.vcirc_up = vcirc_up
    self.vcirc_dn = vcirc_dn





 
