"""
############################################################################

Copyright (C) 2003-2014, Michele Cappellari
E-mail: cappellari_at_astro.ox.ac.uk

Updated versions of the software are available from my web page
http://purl.org/cappellari/software

If you have found this software useful for your research,
I would appreciate an acknowledgment to the use of the
"JAM modelling package of Cappellari (2008)"

This software is provided as is without any warranty whatsoever.
Permission to use, for non-commercial purposes is granted.
Permission to modify for personal or internal use is granted,
provided this copyright and disclaimer are included unchanged
at the beginning of the file. All other rights are reserved.

############################################################################

NAME:
  MGE_VCIRC

PURPOSE:
   This procedure calculates the circular velocity in the equatorial plane of
   an axisymmetric galaxy model described by a Multi-Gaussian Expansion
   parametrization. This implementation follows the approach described in
   Appendix A of Cappellari et al. (2002, ApJ, 578, 787), which allows for
   quick and accurate calculations also at very small and very large radii.

CALLING SEQUENCE:
   MGE_VCIRC, surf_pot, sigma_pot, qObs_pot, $
       inc_deg, mbh, distance, rad, vcirc, SOFT=soft

INPUT PARAMETERS:
  SURF_POT: vector of length M containing the peak value of the MGE Gaussians
      describing the galaxy surface density in units of Msun/pc**2 (solar
      masses per parsec**2). This is the MGE model from which the model
      potential is computed.
  SIGMA_POT: vector of length M containing the dispersion in arcseconds of
      the MGE Gaussians describing the galaxy surface density.
  QOBS_POT: vector of length M containing the observed axial ratio of the MGE
      Gaussians describing the galaxy surface density.
  INC_DEG: inclination in degrees (90 being edge-on).
  MBH: Mass of a nuclear supermassive black hole in solar masses.
  DISTANCE: distance of the galaxy in Mpc.
  RAD: Vector of length P with the radius in arcseconds, measured from the
      galaxy centre, at which one wants to compute the model predictions.

KEYWORDS:
  SOFT: Softening length in arcsec for the Keplerian potential of the black
      hole. When this keyword is nonzero the black hole potential will be
      replaced by a Plummer potential with the given scale length.

OUTPUT PARAMETER:
  VCIRC: Vector of length P with the model predictions for the circular
      velocity at the given input radii RAD.

USAGE EXAMPLE:
   A simple usage example is given in the procedure
   TEST_MGE_CIRCULAR_VELOCITY at the end of this file.

REQUIRED ROUTINES:
      By M. Cappellari (included in the JAM distribution):
      - ANY
      - DIFF
      - QUADVA
      - RANGE

MODIFICATION HISTORY:
V1.0: Written and tested as part of the implementation of
    Schwarzschild's numerical orbit superposition method described in
    Cappellari et al. (2006). Michele Cappellari, Leiden, 3 February 2003
V3.0: This version retains only the few routines required for the computation
    of the circular velocity. All other unnecessary modelling routines have
    been removed. MC, Leiden, 22 November 2005
V3.01: Minor code polishing. MC, Oxford, 9 November 2006
V3.02: First released version. Included documentation. QUADVA integrator.
    MC, Windhoek, 1 October 2008
V4.0: Translated from IDL into Python. MC, Oxford, 10 April 2014
V4.01: Support both Python 2.6/2.7 and Python 3.x. MC, Oxford, 25 May 2014
 
"""

from __future__ import print_function

import numpy as np
from cap_quadva import quadva

#
# The following set of routines computes the R acceleration
# for a density parametrized via the Multi-Gaussian Expansion method.
# The routines are designed to GUARANTEE a maximum relative error of
# 1e-4 in the case of positive Gaussians. This maximum error is reached
# only at the extremes of the usable radial range and only for a very
# flattened Gaussian (q=0.1). Inside the radial range normally adopted
# during orbit integration the error is instead <1e-6.
#
#----------------------------------------------------------------------

def _accelerationR_dRRcapitalh(u, r2, z2, e2, s2):
    #
    # Computes: -D[H[R,z,u],R]/R
    
    u2 = u**2
    p2 = 1. - e2*u2
    us2 = u2/s2
    return np.exp(-0.5*us2*(r2+z2/p2))*us2/np.sqrt(p2) # Cfr. equation (A3)

#----------------------------------------------------------------------

def _accR(R, z, dens, sigma, qintr, bhMass, soft):

    mgepot = np.empty_like(R)
    pot = np.empty_like(dens)
    e2 = 1. - qintr**2
    s2 = sigma**2
    r2 = R**2
    z2 = z**2
    d2 = r2 + z2
    
    for k in range(R.size):
        for j in range(dens.size):
            if (d2[k] < s2[j]/240.**2):
                e = np.sqrt(e2[j]) # pot is Integral in {u,0,1} of -D[H[R,z,u],R]/R at (R,z)=0
                pot[j] = (np.arcsin(e)/e - qintr[j])/(2*e2[j]*s2[j]) # Cfr. equation (A5)
            elif (d2[k] < s2[j]*245**2):
                pot[j] = quadva(_accelerationR_dRRcapitalh, [0.,1.], 
                                args=(r2[k], z2[k], e2[j], s2[j]))[0]
            else: # R acceleration in Keplerian limit (Cappellari et al. 2002)
               pot[j] = np.sqrt(np.pi/2)*sigma[j]/d2[k]**1.5 # Cfr. equation (A4)
        mgepot[k] = np.sum(s2*qintr*dens*pot)
    
    G = 0.00430237    # (km/s)**2 pc/Msun [6.674e-11 SI units (CODATA-10)]
    
    return -R*(4*np.pi*G*mgepot + G*bhMass/(d2 + soft**2)**1.5)

#----------------------------------------------------------------------

def mge_vcirc(surf_pc, sigma_arcsec, qobs, 
                            inc_deg, mbh, distance, rad, soft=0.):

    pc = distance*np.pi/0.648 # Constant factor to convert arcsec --> pc
    
    soft_pc = soft*pc           # Convert from arcsec to pc
    Rcirc = rad*pc              # Convert from arcsec to pc
    sigma = sigma_arcsec*pc     # Convert from arcsec to pc
    
    # Axisymmetric deprojection of total mass.
    # See equation (12)-(14) of Cappellari (2008)
    #
    inc = np.radians(inc_deg)      # Convert inclination to radians
    qintr = qobs**2 - np.cos(inc)**2
    if np.any(qintr <= 0.0): 
        raise ValueError('Inclination too low for deprojection')
    qintr = np.sqrt(qintr)/np.sin(inc)
    if np.any(qintr <= 0.05):
        raise ValueError('q < 0.05 components')
    dens = surf_pc*qobs/(qintr*sigma*np.sqrt(2*np.pi)) # MGE deprojection
    
    # Equality of gravitational and centrifugal acceleration accR at z=0
    # R Vphi**2 == accR --> R (vcirc/R)**2 == accR
    #
    accR = _accR(Rcirc, Rcirc*0, dens, sigma, qintr.clip(0.001,0.999), mbh, soft_pc)
    vcirc = np.sqrt(Rcirc*np.abs(accR))  # circular velocity at rcirc
    
    return vcirc

#----------------------------------------------------------------------------

def test_mge_vcirc():
    """
    Usage example for mge_vcirc()
    It takes a fraction of a second on a 2GHz computer
    
    """    
    import matplotlib.pyplot as plt
    
    # Realistic MGE galaxy surface brightness
    # 
    surf = np.array([39483, 37158, 30646, 17759, 5955.1, 1203.5, 174.36, 21.105, 2.3599, 0.25493])
    sigma = np.array([0.153, 0.515, 1.58, 4.22, 10, 22.4, 48.8, 105, 227, 525])
    qObs = np.array([0.57, 0.57, 0.57, 0.57, 0.57, 0.57, 0.57, 0.57, 0.57, 0.57])
    
    inc = 60. # Inclination in degrees
    mbh = 1e6 # BH mass in solar masses
    distance = 10. # Mpc
    rad = np.logspace(-1,2,25) # Radii in arscec where Vcirc has to be computed
    ml = 5.0 # Adopted M/L ratio
    
    vcirc = mge_vcirc(surf*ml, sigma, qObs, inc, mbh, distance, rad)
    
    plt.clf()
    plt.plot(rad, vcirc, '-o')
    plt.xlabel('R (arcsec)')
    plt.ylabel(r'$V_{circ}$ (km/s)')

#----------------------------------------------------------------------

if __name__ == '__main__':
    
    from time import clock
    t = clock()
    test_mge_vcirc()
    print('Elapsed time:', clock()-t, ' seconds')
