# Dyn_models
Markov Chain Monte Carlo (MCMC) implementation of Axisymmetric Drift Correction (ADC) and Jeans Anisotropic Models (JAM) approaches.

ADC-MCMC:

REQUIRED ROUTINES:
By D. Foreman-Mackey (http://dan.iel.fm/emcee/current/)
  - emcee
  - corner

PURPOSE:
This procedure calculates the Markov Chain Monte Carlo (MCMC) circular 
velocity using the thin disk approximation assumption in the 
Axisymmetric Drift Correction approach (ADC; Binney & Tremaine 2008).
We use the velocity and velocity radial profiles of the galaxy, derived from its stellar kinematics via the
kinemetry routine (Krajnovic et al., 2006) and the surface brightness as provided by the Multi-Gaussian Expansion 
parametrization method (MGE; Monnet,Bacon & Emsellem 1992). 
We use the "EMCEE code" of Foreman-Mackey et al. 2013 (http://dan.iel.fm/emcee/current/),
an implementation of an affine invariant ensemble sampler for the MCMC method of parameter estimations.



JAM-MCMC:

This procedure calculates the Markov Chain Monte Carlo (MCMC) circular 
velocity in the equatorial plane of an axisymmetric galaxy model described 
by a Multi-Gaussian Expansion parametrization (MGE; Monnet,Bacon & Emsellem 1992). 
We use the "EMCEE code" of Foreman-Mackey et al. 2013 (http://dan.iel.fm/emcee/current/),
an implementation of an affine invariant ensemble sampler for the MCMC method of parameter estimations,
together with the Jeans Anisotropic MGE (JAM) code of the dynamics of axisymmetric galaxies by Cappellari 2008 
(http://purl.org/cappellari/software).


REQUIRED ROUTINES:

By M. Cappellari (included in the JAM distribution; http://purl.org/cappellari/software):
  - JAM modelling package in python

By D. Foreman-Mackey (http://dan.iel.fm/emcee/current/)
  - emcee
  - corner
