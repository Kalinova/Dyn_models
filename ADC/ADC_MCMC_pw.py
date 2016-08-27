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
  ADC_MCMC_pw

PURPOSE:
   PURPOSE:
   This procedure calculates the Markov Chain Monte Carlo (MCMC) circular 
   velocity using the thin disk approximation assumption in the 
   Axisymmetric Drift Correction approach (ADC; Binney & Tremaine 2008).
   We use the velocity and velocity radial profiles of the galaxy, derived from its stellar kinematics via the
   kinemetry routine (Krajnovic et al., 2006) and the surface brightness as provided by the Multi-Gaussian Expansion 
   parametrization method (MGE; Monnet,Bacon & Emsellem 1992). 
   We use the "EMCEE code" of Foreman-Mackey et al. 2013 (http://dan.iel.fm/emcee/current/),
   an implementation of an affine invariant ensemble sampler for the MCMC method of parameter estimations.


CALLING SEQUENCE:

    res = ADC_MOC_pw(gal, incl, Vinf_in, Rc_in,sig0_in,ksig_in,R,vobs,evobs,sobs,esobs,I0obs,spobs,nwalks, burn_steps, steps, threads)


INPUT PARAMETERS:
    GAL: name of the galaxy
    INCL: inclination 
    VINF_IN: initial guess for the asymptotic velocity parameter in the power-law fitting model to the velocity radial profile
    RC_IN: initial guess for the core radius parameter in the power-law fitting model of the velocity radial profile
    SIG0_IN: initial guess for the y-intercept of the linear fit of the velocity dispersion radial profile
    KSIG_IN: initial guess for the slope of the linear fit of the velocity dispersion radial profile
    R: radius of the velocity profile
    VOBS: observed velocity radial profile (e.g., using kinemetry routine of Krajnovic et al., 2006)
    EVOBS: error of VOBS
    SOBS: observed velocity dispersion radial profile (e.g., using kinemetry routine of Krajnovic et al., 2006)
    ESOBS: error of SOBS
    I0OBS: vector of length N containing the peak surface brightness of the
      MGE Gaussians describing the galaxy surface brightness in units of
      Lsun/pc^2 (solar luminosities per parsec^2) 
    SPOBS: vector of length N containing the dispersion in arcseconds of
      the MGE Gaussians describing the galaxy surface brightness.
    NWALKS: Number of Goodman & Weare walkers, which should be equal or 
      greater than twice the dimension, i.e twice the number of the fitted parameters)
    BURN_STEPS: Number of the steps for the burn-in process
    STEPS: Number of the steps after burn-in process, i.e steps for the final chains of the parameters
    THREADS: Number of threads to use for parallelization, where threads > 1 is for multiprocessing 


RETURN:
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
  CHIV: the chi2 of the velocity fit, where chi2v = np.sum((vobs-vmod)**2/(evobs)**2)
  CHIS: the chi2 of the velocity dispersion fit, where chi2s = np.sum((sobs-smod)**2/(esobs)**2)
  CHI2V_RED: reduced chi2 of the velocity profile fit, where chi2v_red = chi2v/(len(R) - 4)
  CHI2S_RED: reduced chi2 of the velocity dispersion profile fit, where chi2s_red = chi2s/(len(R) - 4)
  VMOD: the model fit of the velocity radial profile
  SMOD: the model fit of the velocity dispersion profile


METHODS:
  rms_logprob: sub-routine for defining the log-probability
  runadc: sub-routine for running of ADC-MCMC analysis
  make_fig: sub-routine for plotting of burn-in and final chains of the parameters, and corner figure
  make_fig_curves: sub-routine for plotting the velocity radial profiles, velocity dispersion radial profiles, azimuthal velocity radial profile,
    deprojected velocity dispersion profile, and circular velocity curve of the galaxy using the distributions of 
    the parameters and their best fit values.


USAGE EXAMPLE:
   A simple usage example "test_ADC_MCMC_PW.py" is given in the same directory.


REQUIRED ROUTINES:
      By D. Foreman-Mackey (http://dan.iel.fm/emcee/current/)
      - emcee
      - corner

MODIFICATION HISTORY:
V1.0: Written and tested as part of the implementation of
    ADC-MCMC method described in Kalinova et al. (2016). 
    Veselina Kalinova, Dario Colombo, Erik Rosolowsky; 
    University of Alberta, 2016

'''


import emcee
import math
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import scipy.stats as ss
import scipy.interpolate as interp
from astropy.table import Table
from readcol import readcol
import corner
from matplotlib.colors import LogNorm
import fish 
from pdb import set_trace as stop

def log_adc(p, R, vobs, sobs, evobs, esobs, Vinf_in, Rc_in,sig0_in,ksig_in,incl):


    Vinf, Rc, beta, sig0, ksig = p[0], p[1], p[2], p[3], p[4]
    
    ar = Rc**2/(Rc**2+R**2)
    sigr = sig0+ksig*R

    vmod = (Vinf*R)*np.sin(math.pi/180*incl)/np.sqrt(Rc**2+R**2)
    smod = np.sqrt(1 - beta*np.cos(math.pi/180*incl)**2 + 0.5*(ar-1)*np.sin(math.pi/180*incl)**2)*sigr
    
    
    #print 'incl:', incl       
    priors = ss.uniform.logpdf(Vinf,loc=0,scale=400)+\
            ss.uniform.logpdf(Rc,loc=0,scale=50)+\
            ss.uniform.logpdf(beta,loc=-1.5,scale=2.5)+\
            ss.uniform.logpdf(sig0,loc=0,scale=300)+\
            ss.uniform.logpdf(ksig,loc=-5,scale=10)+\
            ss.norm.logpdf(Vinf,loc=Vinf_in,scale=10)+\
            ss.norm.logpdf(Rc,loc=Rc_in,scale=1)+\
            ss.norm.logpdf(sig0,loc=sig0_in,scale=1)+\
            ss.norm.logpdf(ksig,loc=ksig_in,scale=0.5)
             
             

    if np.isfinite(priors) == False:
        return -np.inf

    p1 = (vmod-vobs)**2/evobs**2
    p1 = np.nansum(p1)
        
    p2 = (smod-sobs)**2/esobs**2
    p2 = np.nansum(p2)
                
    lp = - p1 - p2 + priors

    if np.isnan(lp):
        return -np.inf

    return lp


def runadc(gal, incl, Vinf_in, Rc_in,sig0_in,ksig_in,R,vobs,evobs,sobs,esobs,I0obs,spobs,nwalks, burn_steps, steps, threads):

    # %&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
    # MCMC for Vobs and sigma_obs
    # %&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&

    # Set the walkers

    ndim, nwalkers = 5,nwalks
    p0 = np.zeros((nwalkers,ndim))
    p0[:,0] = np.random.randn(nwalkers)*10+Vinf_in
    p0[:,1] = np.random.randn(nwalkers)*1+Rc_in
    p0[:,2] = np.random.uniform(-1.5,1,nwalkers) #beta
    p0[:,3] = np.random.randn(nwalkers)*1+sig0_in
    p0[:,4] = np.random.randn(nwalkers)*0.5+ksig_in
    # p0[:,0] = np.random.randn(nwalkers)*10+Vinf_in
    # p0[:,1] = np.random.randn(nwalkers)*1+Rc_in
    # p0[:,2] = np.random.randn(nwalkers)*1+ideg_in  #incl
    # p0[:,3] = np.random.uniform(-1,1,nwalkers) #beta
    # p0[:,4] = np.random.randn(nwalkers)*1+sig0_in
    # p0[:,5] = np.random.randn(nwalkers)*0.5+ksig_in
    
    
    
    
    ####### Call EMCEE code #######   
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_adc,
              args=[R,vobs,sobs,evobs,esobs, Vinf_in, Rc_in,sig0_in,ksig_in, incl], threads=threads)

    # burn-in
    #pos, prob, state = sampler.run_mcmc(p0, burn_steps)

    ####### Chain #######
    #sampler.reset()
    #pos,prob,state = sampler.run_mcmc(pos,steps)
    #--------------------------------------------
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
    #---------------------------------------------
    Vinf_dist = final_flatchains[:,0]
    Rc_dist = final_flatchains[:,1]
    betaz_dist = final_flatchains[:,2]
    sig0_dist = final_flatchains[:,3]
    ksig_dist = final_flatchains[:,4]

    Vinf_med = np.median(final_flatchains[:,0])
    Rc_med = np.median(final_flatchains[:,1])
    betaz_med = np.median(final_flatchains[:,2])
    sig0_med = np.median(final_flatchains[:,3])
    ksig_med = np.median(final_flatchains[:,4])

    Vinf_plus = np.percentile(final_flatchains[:,0], 75)- np.median(final_flatchains[:,0])
    Rc_plus = np.percentile(final_flatchains[:,1], 75)- np.median(final_flatchains[:,1])
    betaz_plus = np.percentile(final_flatchains[:,2], 75)- np.median(final_flatchains[:,2])
    sig0_plus = np.percentile(final_flatchains[:,3], 75)- np.median(final_flatchains[:,3])
    ksig_plus = np.percentile(final_flatchains[:,4], 75)- np.median(final_flatchains[:,4])

    Vinf_minus = np.median(final_flatchains[:,0]) - np.percentile(final_flatchains[:,0], 25)
    Rc_minus = np.median(final_flatchains[:,1]) - np.percentile(final_flatchains[:,1], 25)
    betaz_minus = np.median(final_flatchains[:,2]) - np.percentile(final_flatchains[:,2], 25)
    sig0_minus = np.median(final_flatchains[:,3]) - np.percentile(final_flatchains[:,3], 25)
    ksig_minus = np.median(final_flatchains[:,4]) - np.percentile(final_flatchains[:,4], 25)

    #-------------------------------------------------
    # Tables with all parameters
    #-----------------------------------------------
    # Array for the medians
    medians = [Vinf_med, Rc_med, 0.0, sig0_med,ksig_med, betaz_med]

    # Array for the upper percentiles
    ups = [Vinf_plus, Rc_plus, 0.0, sig0_plus,ksig_plus, betaz_plus]

    # Array for the lower percentiles
    lws = [Vinf_minus, Rc_minus, 0.0, sig0_minus,ksig_minus,betaz_minus]

    # make table
    table = Table([medians, ups, lws], names = ('medians(Vinf, Rc, kv, sig0,ksig,betaz)','ups','lws'))
    table.write("data_output/Tables/"+gal+".txt",format="ascii.tab",delimiter=",")
    print 'Print table'
    #-------------------------------------------------


    print "%&%&%&%&%&%&%&%&%&%&%&%&%&"
    print   "Final best fit values"
    print "%&%&%&%&%&%&%&%&%&%&%&%&%&"

    print 'Vinf: ', np.median(final_flatchains[:,0]), \
        '+', np.percentile(final_flatchains[:,0], 75) - np.median(final_flatchains[:,0]),\
        '-', np.median(final_flatchains[:,0]) - np.percentile(final_flatchains[:,0], 25)
    print 'Rc: ', np.median(final_flatchains[:,1]), \
        '+', np.percentile(final_flatchains[:,1], 75) - np.median(final_flatchains[:,1]),\
        '-', np.median(final_flatchains[:,1]) - np.percentile(final_flatchains[:,1], 25) 
    
    print 'betaz: ', np.median(final_flatchains[:,2]), \
        '+', np.percentile(final_flatchains[:,2], 75) - np.median(final_flatchains[:,2]),\
        '-', np.median(final_flatchains[:,2]) - np.percentile(final_flatchains[:,2], 25)
    print 'sig0: ', np.median(final_flatchains[:,3]), \
        '+', np.percentile(final_flatchains[:,3], 75) - np.median(final_flatchains[:,3]),\
        '-', np.median(final_flatchains[:,3]) - np.percentile(final_flatchains[:,3], 25)
    print 'ksig: ', np.median(final_flatchains[:,4]), \
        '+', np.percentile(final_flatchains[:,4], 75) - np.median(final_flatchains[:,4]),\
        '-', np.median(final_flatchains[:,4]) - np.percentile(final_flatchains[:,4], 25)             


    #-------------------------------------------------
    # Final Chi2
    #-----------------------------------------------

    ar_med = Rc_med**2/(Rc_med**2+R**2)
    sigr_med = sig0_med+ksig_med*R

    vmod = (Vinf_med*R)*np.sin(math.pi/180*incl)/np.sqrt(Rc_med**2+R**2)
    smod = np.sqrt(1 - betaz_med*np.cos(math.pi/180*incl)**2 + 0.5*(ar_med-1)*np.sin(math.pi/180*incl)**2)*sigr_med

    chi2v = np.sum((vobs-vmod)**2/(evobs)**2)
    chi2s = np.sum((sobs-smod)**2/(esobs)**2)

    chi2v_red = chi2v/(len(R) - 5)
    chi2s_red = chi2s/(len(R) - 5)

    RES_v=np.median(np.abs( (vobs/vmod) -1 ))
    RES_s=np.median(np.abs( (sobs/smod) -1 ))

    vschi2=[chi2v,chi2s]
    vschi2_red=[chi2v_red,chi2s_red]
    vsRES=[RES_v,RES_s]

    print 'Chi2v_red=', chi2v_red
    print 'Chi2s_red=', chi2s_red
    print 'RES_v=',RES_v
    print 'RES_s=',RES_s

    table2 = Table([vschi2,vschi2_red,vsRES], names = ('vschi2','vschi2_red','vsRES'))
    table2.write("data_output/Tables/Chi2/"+gal+"_chi2.txt",format="ascii.tab",delimiter=",")
    print 'Print table chi2'
    #stop()



    return table, burn_chains, burn_lns, burn_flatchains, burn_flatlns, \
    final_chains, final_lns, final_flatchains, final_flatlns,chi2v,chi2s,chi2v_red,chi2s_red,vmod,smod  

class ADC_MOC_pw(object):

    def __init__(self,gal, incl, Vinf_in, Rc_in,sig0_in,ksig_in,R,vobs,evobs,sobs,esobs,I0obs,spobs,nwalks, burn_steps, steps, threads):
        
        if nwalks < 10:
            print("NWALKERS must be equal or greater than twice the dimension)")
            nwalks = 10
    
        # Galaxy parameters
        self.gal = gal
        self.incl=incl
        


        # guess for the velocity fitted parameters
        self.Vinf_in= Vinf_in
        self.Rc_in=Rc_in
        self.sig0_in=sig0_in
        self.ksig_in=ksig_in
        
        # observables
        self.R=R
        self.vobs=vobs
        self.evobs=evobs
        self.sobs=sobs
        self.esobs=esobs

        # MGE parameters
        self.I0obs=I0obs
        self.spobs=spobs

        # Run ADC with MCMC
        self.table,  self.burn_chains, self.burn_lns, self.burn_flatchains, self.burn_flatlns, \
        self.final_chains, self.final_lns, self.final_flatchains, self.final_flatlns,\
        self.chi2v,self.chi2s,self.chi2v_red,self.chi2v_red,self.vmod,self.smod=runadc(self.gal, self.incl, \
         self.Vinf_in, self.Rc_in,self.sig0_in,self.ksig_in,self.R,self.vobs,self.evobs,self.sobs,\
         self.esobs,self.I0obs,self.spobs, nwalks, burn_steps, steps, threads)
        ###########################################################################

    def make_fig(self,incl):

        fig = plt.figure(figsize=(10,6))
        plt.subplot(3,2,1)
        plt.title(self.gal)
        #plt.plot(burn_chains[:,:,0].T,marker=".",lw=0,color="k")
        plt.plot(self.burn_chains[:,:,0].T)
        plt.ylabel(r'Chain for $v_{\infty}$')
        plt.subplot(3,2,2)
        plt.plot(self.burn_chains[:,:,1].T)
        plt.ylabel(r'Chain for $R_c$')
        plt.subplot(3,2,3)
        plt.plot(self.burn_chains[:,:,2].T)
        plt.ylabel(r'Chain for $\beta_z$')
        plt.subplot(3,2,4)
        plt.plot(self.burn_chains[:,:,3].T)
        plt.ylabel(r'Chain for $\sigma_0$')
        plt.subplot(3,2,5)
        plt.plot(self.burn_chains[:,:,4].T)
        plt.ylabel(r'Chain for $k_{\sigma}$')
        plt.tight_layout() # This tightens up the spacing
        plt.savefig("figures/Burn_in/"+self.gal+"_burnin.png")
        plt.close()

        #sel= np.where(final_flatlns + 10 > np.max(final_flatlns))

        fig = plt.figure(figsize=(18,10))
        plt.subplot(3,2,1)
        plt.title(self.gal)
        #plt.hist2d(self.final_chains[:,:,0].T,bins=40, norm=LogNorm(),cmap='gray')
        plt.plot(self.final_chains[:,:,0].T)
        #plt.plot(self.final_chains[:,:,0].T,marker=".",lw=0,color="#A9A9A9")
        plt.ylabel(r'Chain for $v_{\infty}$')
        plt.subplot(3,2,2)
        plt.plot(self.final_chains[:,:,1].T)
        plt.ylabel(r'Chain for $R_c$')
        plt.subplot(3,2,3)
        plt.plot(self.final_chains[:,:,2].T)
        plt.ylabel(r'Chain for $\beta_z$')
        plt.subplot(3,2,4)
        plt.plot(self.final_chains[:,:,3].T)
        plt.ylabel(r'Chain for $\sigma_0$')
        plt.subplot(3,2,5)
        plt.plot(self.final_chains[:,:,4].T)
        plt.ylabel(r'Chain for $k_{\sigma}$')
        plt.tight_layout() # This tightens up the spacing
        plt.savefig("figures/Chain/"+self.gal+"_chain.png")
        plt.close()

        #sel= np.where(self.final_flatlns + 10 > np.max(self.final_flatlns))


        figure=corner.corner(self.final_flatchains, labels=["$v_{\infty}$", "$R_c$", r'$\beta_{z}^{ADC}$', 
          "$\sigma_{0}$", "$k_{\sigma}$"], quantiles=[0.25, 0.50, 0.75],show_titles=True, title_fmt=".3f",title_args={"fontsize": 12} )
        figure.gca().annotate(self.gal+' ADC Power-law', xy=(0.5, 1.0), xycoords="figure fraction", xytext=(0, -5), 
          textcoords="offset points", ha="center", va="top")
        figure.savefig("figures/Corner/"+self.gal+"_corner.png")
        plt.close()    

        


    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def make_fig_curves(self, incl):

        R=self.R
        Vinf_dist = self.final_flatchains[:,0]
        Rc_dist = self.final_flatchains[:,1]
        betaz_dist = self.final_flatchains[:,2]
        sig0_dist = self.final_flatchains[:,3]
        ksig_dist = self.final_flatchains[:,4]


        # observables
        vobs=self.vobs
        evobs=self.evobs
        sobs=self.sobs
        esobs=self.esobs

        # MGE parameters
        I0obs=self.I0obs
        spobs=self.spobs


        Vphi_mod = np.zeros([len(R),len(self.final_flatchains[:,0])])
        alphar_mod = np.zeros([len(R),len(self.final_flatchains[:,0])])
        sigr_mod = np.zeros([len(R),len(self.final_flatchains[:,0])])
        dlnSigR2_dlnR_mod = np.zeros([len(R),len(self.final_flatchains[:,0])])


        Vphi_obs = np.zeros([len(R),len(self.final_flatchains[:,0])])
        sigr_obs = np.zeros([len(R),len(self.final_flatchains[:,0])])
        sobs_fin= np.zeros([len(R),len(self.final_flatchains[:,0])])
        vobs_fin= np.zeros([len(R),len(self.final_flatchains[:,0])])

        for j in range(len(R)):

            Vphi_mod[j,:] = Vinf_dist*R[j]/np.sqrt(R[j]**2+Rc_dist**2)
            alphar_mod[j,:] = Rc_dist**2/(R[j]**2+Rc_dist**2)
            sigr_mod[j,:] = sig0_dist + ksig_dist*R[j]
            dlnSigR2_dlnR_mod[j,:] = 2*ksig_dist*R[j]/(sig0_dist + ksig_dist*R[j]) 


        for j in range(len(R)):

            Vphi_obs[j,:] = vobs[j]/np.sin(math.pi/180.*incl)
            sigr_obs[j,:] = sobs[j]/np.sqrt(1 - \
                                            betaz_dist*np.cos(math.pi/180*incl)**2 + \
                                            0.5*(Rc_dist**2/(R[j]**2+Rc_dist**2)-1)*np.sin(math.pi/180*incl)**2)
            sobs_fin[j,:] = (sig0_dist + ksig_dist*R[j])*np.sqrt(1 - \
                                            betaz_dist*np.cos(math.pi/180*incl)**2 + \
                                            0.5*(Rc_dist**2/(R[j]**2+Rc_dist**2)-1)*np.sin(math.pi/180*incl)**2)

            vobs_fin[j,:] = Vinf_dist*R[j]*np.sin(math.pi/180.*incl)/np.sqrt(Rc_dist**2+R[j]**2)
            
                                            
        # Itot = I(R), dItot = sum(I0j*exp(-0.5R^2/spj^2)*(-R/spj^2))    
        Itot = np.zeros(len(R))
        dItot = np.zeros(len(R))
        for j in range(len(R)):
            for i in range(len(I0obs)):
                
                Itot[j] = Itot[j] + I0obs[i]*np.exp(-0.5*R[j]**2/(spobs[i]**2)) 
                dItot[j] = dItot[j] + (-R[j]/(spobs[i]**2))*I0obs[i]*np.exp(-0.5*R[j]**2/(spobs[i]**2))

        dlnI_dlnR = R*dItot/Itot 
        dlnI_dlnRs = np.tile(dlnI_dlnR,(Vphi_mod.shape[1],1))
        dlnI_dlnRs = dlnI_dlnRs.T
        #===========================
        
        # %&%&%&%&%&%&%&%&%
        #    Final ADC
        # %&%&%&%&%&%&%&%&%    

        # From model...
        Vc2_mod = Vphi_mod**2 + sigr_mod**2*(-dlnI_dlnRs - dlnSigR2_dlnR_mod - 0.5*(1-alphar_mod))
        Vc_mod = np.sqrt(Vc2_mod)     

        # From observation + model...
        Vc2_obmod = Vphi_obs**2 + sigr_obs**2*(-dlnI_dlnRs - dlnSigR2_dlnR_mod - 0.5*(1-alphar_mod))
        Vc_obmod = np.sqrt(Vc2_obmod)

        Vc2_oblit = Vphi_obs**2 + sigr_obs**2

        #stop()

        ############################# PLOT FIGURES ###############################################
        fig = plt.figure(figsize=(10,6))
        plt.plot(R,np.median(Vc_obmod, axis = 1), 'bo')
        eVctop0 = np.percentile(Vc_obmod, 75, axis = 1) - np.median(Vc_obmod, axis = 1)
        eVcbot0 = np.median(Vc_obmod, axis = 1) - np.percentile(Vc_obmod, 25, axis = 1)
        plt.errorbar(R,np.median(Vc_obmod, axis = 1), yerr = (eVctop0, eVcbot0), color = 'b' )

        plt.plot(R,np.median(Vc_mod, axis = 1), 'ro', label='ADC')
        eVctop = np.percentile(Vc_mod, 75, axis = 1) - np.median(Vc_mod, axis = 1)
        eVcbot = np.median(Vc_mod, axis = 1) - np.percentile(Vc_mod, 25, axis = 1)
        plt.errorbar(R,np.median(Vc_mod, axis = 1), yerr = (eVctop, eVcbot), color = 'r' )

        #---------------------------------------------------
        plt.xlabel('R [arcsec]')
        plt.ylabel('$V_{c,ADC}$')
        plt.savefig("figures/Vcirc/"+self.gal+"_Vcirc.png")
        plt.close()

        #######################################
        #... save Vcirc in a file
        np.savez('data_output/Vcirc/Vc_'+self.gal, rad=R, vcirc_med=Vc_mod, vcirc_up=eVctop, vcirc_dn=eVcbot)
        ######################################

        #... Vphi profiles
        fig = plt.figure(figsize=(10,6))
        plt.plot(R,np.median(Vphi_obs, axis=1), 'bo')
        eVctop1 = np.percentile(Vphi_obs, 75, axis = 1) - np.median(Vphi_obs, axis = 1)
        eVcbot1 = np.median(Vphi_obs, axis = 1) - np.percentile(Vphi_obs, 25, axis = 1)
        plt.errorbar(R,np.median(Vphi_obs, axis = 1), yerr = (eVctop1, eVcbot1), color = 'b' )

        plt.plot(R,np.median(Vphi_mod,axis=1), 'r-')
        eVctop2 = np.percentile(Vphi_mod, 75, axis = 1) - np.median(Vphi_mod, axis = 1)
        eVcbot2 = np.median(Vphi_mod, axis = 1) - np.percentile(Vphi_mod, 25, axis = 1)
        plt.errorbar(R,np.median(Vphi_mod, axis = 1), yerr = (eVctop2, eVcbot2), color = 'r' )

        plt.plot(R,vobs, 'g-')
        plt.errorbar(R,vobs, yerr = (evobs, evobs), color = 'g' )

        plt.xlabel('R [arcsec]')
        plt.ylabel('$v_{\phi}$')
        plt.savefig("figures/Vphi/"+self.gal+"_Vphi.png")
        plt.close()

        #... SigR profiles
        fig = plt.figure(figsize=(10,6))
        plt.plot(R,np.median(sigr_obs, axis=1), 'o')
        eVctop3 = np.percentile(sigr_obs, 75, axis = 1) - np.median(sigr_obs, axis = 1)
        eVcbot3 = np.median(sigr_obs, axis = 1) - np.percentile(sigr_obs, 25, axis = 1)
        plt.errorbar(R,np.median(sigr_obs, axis = 1), yerr = (eVctop3, eVcbot3), color = 'b' )

        plt.plot(R,np.median(sigr_mod,axis=1), 'ro')
        eVctop4 = np.percentile(sigr_mod, 75, axis = 1) - np.median(sigr_mod, axis = 1)
        eVcbot4 = np.median(sigr_mod, axis = 1) - np.percentile(sigr_mod, 25, axis = 1)
        plt.errorbar(R,np.median(sigr_mod, axis = 1), yerr = (eVctop4, eVcbot4), color = 'r' )

        plt.xlabel('R [arcsec]')
        plt.ylabel('$\sigma_{R}$')
        plt.savefig("figures/SgR/"+self.gal+"_SgR.png")
        plt.close()

        #... Sobs profiles
        fig = plt.figure(figsize=(10,6))
        plt.plot(R,sobs, 'go')
        plt.errorbar(R,sobs, yerr = (esobs, esobs), color = 'g' )

        plt.plot(R,np.median(sobs_fin,axis=1), 'ro')
        eVctop5 = np.percentile(sobs_fin, 75, axis = 1) - np.median(sobs_fin, axis = 1)
        eVcbot5 = np.median(sobs_fin, axis = 1) - np.percentile(sobs_fin, 25, axis = 1)
        plt.errorbar(R,np.median(sobs_fin, axis = 1), yerr = (eVctop5, eVcbot5), color = 'r' )
        
        plt.xlabel('R [arcsec]')
        plt.ylabel('$\sigma_{obs}$')
        plt.savefig("figures/Sobs/"+self.gal+"_Sobs.png")
        plt.close()

        #...Vobs_profile
        fig = plt.figure(figsize=(10,6))
        plt.plot(R,vobs, 'go')
        plt.errorbar(R,vobs, yerr = (esobs, esobs), color = 'g' )

        plt.plot(R,np.median(vobs_fin,axis=1), 'ro')
        eVctop6 = np.percentile(vobs_fin, 75, axis = 1) - np.median(vobs_fin, axis = 1)
        eVcbot6 = np.median(vobs_fin, axis = 1) - np.percentile(vobs_fin, 25, axis = 1)
        plt.errorbar(R,np.median(vobs_fin, axis = 1), yerr = (eVctop6, eVcbot6), color = 'r' )

        plt.xlabel('R [arcsec]')
        plt.ylabel('$V_{obs}$')
        plt.savefig("figures/Vobs/"+self.gal+"_Vobs.png")
        plt.close()
        ########################################################################################        
        print 'Plotting done!'

        print 'Save in files'
        #... save Vcirc in a file
        np.savez('data_output/Vcirc/Vc_'+self.gal, \
        #...V and S observed values
        R=R, vobs=vobs, evobs=evobs, sobs=sobs, esobs=esobs,\
        #...V and S Modeled values
        vobs_fin=vobs_fin, eVctop6=eVctop6, eVcbot6=eVcbot6,\
        sobs_fin=sobs_fin, eVctop5=eVctop5, eVcbot5=eVcbot5,\
           #...Vph and SigR observed
        sigr_obs=sigr_obs, eVctop3=eVctop3, eVcbot3=eVcbot3,\
        Vphi_obs=Vphi_obs, eVctop1=eVctop1, eVcbot1=eVcbot1,\
        #...Vph and SigR modeled
        sigr_mod=sigr_mod, eVctop4=eVctop4, eVcbot4=eVcbot4,\
        Vphi_mod=Vphi_mod, eVctop2=eVctop2, eVcbot2=eVcbot2,\
        #...Vc_ADC observed
        Vc_obmod=Vc_obmod, eVctop0=eVctop0, eVcbot0=eVcbot0,\
        # Vc_ADC modeled
        Vc_mod=Vc_mod, eVctop=eVctop, eVcbot=eVcbot)








