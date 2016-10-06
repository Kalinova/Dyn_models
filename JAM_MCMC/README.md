#JAM MCMC package

REQUIRED ROUTINES:

By M. Cappellari (included in the JAM distribution; http://purl.org/cappellari/software):
  - JAM modelling package in python

By D. Foreman-Mackey (http://dan.iel.fm/emcee/current/)
  - emcee
  - corner

PURPOSE:

This procedure calculates the Markov Chain Monte Carlo (MCMC) circular 
velocity in the equatorial plane of an axisymmetric galaxy model described 
by a Multi-Gaussian Expansion parametrization (MGE; Monnet,Bacon & Emsellem 1992). 
We use the "EMCEE code" of Foreman-Mackey et al. 2013 (http://dan.iel.fm/emcee/current/),
an implementation of an affine invariant ensemble sampler for the MCMC method of parameter estimations,
together with the Jeans Anisotropic MGE (JAM) code of the dynamics of axisymmetric galaxies by Cappellari 2008 
(http://purl.org/cappellari/software).
