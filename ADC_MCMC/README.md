#ADC MCMC package

This procedure calculates the Markov Chain Monte Carlo (MCMC) circular 
velocity using the thin disk approximation assumption in the 
Axisymmetric Drift Correction approach (ADC; Binney & Tremaine 2008).
We use the velocity and velocity radial profiles of the galaxy, derived from its stellar kinematics via the
kinemetry routine (Krajnovic et al., 2006) and the surface brightness as provided by the Multi-Gaussian Expansion 
parametrization method (MGE; Monnet,Bacon & Emsellem 1992). 
We use the "EMCEE code" of Foreman-Mackey et al. 2013 (http://dan.iel.fm/emcee/current/),
an implementation of an affine invariant ensemble sampler for the MCMC method of parameter estimations.
