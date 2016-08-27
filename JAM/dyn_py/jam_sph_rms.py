"""
##############################################################################

Copyright (C) 2004-2014, Michele Cappellari
E-mail: cappellari_at_astro.ox.ac.uk

Updated versions of the software are available from my web page
http://purl.org/cappellari/software

If you have found this software useful for your research,
I would appreciate an acknowledgment to the use of the
"JAM modelling method of Cappellari (2008)"

This software is provided as is without any warranty whatsoever.
Permission to use, for non-commercial purposes is granted.
Permission to modify for personal or internal use is granted,
provided this copyright and disclaimer are included unchanged
at the beginning of the file. All other rights are reserved.

##############################################################################

NAME:
  JAM_SPHERICAL_RMS

PURPOSE:
   This procedure calculates a prediction for the projected second
   velocity moments V_RMS = sqrt(V^2 + sigma^2), or for a non-rotating
   galaxy V_RMS = sigma, for an anisotropic spherical galaxy model.
   It implements the solution of the anisotropic Jeans equations
   presented in equation (50) of Cappellari (2008, MNRAS, 390, 71).
   PSF convolution is done as described in the Appendix of that paper:
   http://adsabs.harvard.edu/abs/2008MNRAS.390...71C

CALLING SEQUENCE:
   rmsModel, ml, chi2 = \
       jam_sph_rms(surf_lum, sigma_lum, surf_pot, sigma_pot, mbh, distance, rad,
                   beta=None, normpsf=1, pixsize=0, sigmapsf=0, step=0, nrad=50,
                   rms=None, erms=None, ml=None, quiet=False, plot=True)

 NOTE:
     The documentation below is taken from the IDL version, which is functionally
     identical to the Phython version. The various parameters below have the same
     meaning here as in the IDL version, but the documentation has not yet been
     fully updated for Python.

INPUT PARAMETERS:
  SURF_LUM: vector of length N containing the peak surface brightness of the
      MGE Gaussians describing the galaxy surface brightness in units of
      Lsun/pc^2 (solar luminosities per parsec^2).
  SIGMA_LUM: vector of length N containing the dispersion in arcseconds of
      the MGE Gaussians describing the galaxy surface brightness.
  SURF_POT: vector of length M containing the peak value of the MGE Gaussians
      describing the galaxy surface density in units of Msun/pc^2 (solar
      masses per parsec^2). This is the MGE model from which the model
      potential is computed.
    - In a common usage scenario, with a self-consistent model, one has
      the same Gaussians for both the surface brightness and the potential.
      This implies SURF_POT = SURF_LUM, SIGMA_POT = SIGMA_LUM.
      The M/L, by which SURF_POT has to be multiplied to best match the
      data, is fitted by the routine when passing the RMS and ERMS
      keywords with the observed kinematics.
  SIGMA_POT: vector of length M containing the dispersion in arcseconds of
      the MGE Gaussians describing the galaxy surface density.
  MBH: Mass of a nuclear supermassive black hole in solar masses.
    - VERY IMPORTANT: The model predictions are computed assuming SURF_POT
      gives the total mass. In the common self-consistent case one has
      SURF_POT = SURF_LUM and if requested (keyword ML) the program can scale
      the output RMSMODEL to best fit the data. The scaling is equivalent to
      multiplying *both* SURF_POT and MBH by a factor M/L. To avoid mistakes,
      the actual MBH used by the output model is printed on the screen.
  DISTANCE: distance of the galaxy in Mpc.
  RAD: Vector of length P with the (positive) radius from the galaxy center
      in arcseconds of the bins (or pixels) at which one wants to compute
      the model predictions.
    - When no PSF/pixel convolution is performed (SIGMAPSF=0 or PIXSIZE=0)
      there is a singularity at RAD=0 which should be avoided.

KEYWORDS:
  BETA: Vector of length N with the anisotropy
      beta = 1 - (sigma_theta/sigma_R)^2 of the individual MGE Gaussians. A
      scalar can be used if the model has constant anisotropy.
  CHI2: Reduced chi^2 describing the quality of the fit
      chi^2 = total( ((rms-rmsModel)/erms)^2 ) / n_elements(rms)
  ERMS: Vector of length P with the 1sigma errors associated to the RMS
      measurements. From the error propagation
      ERMS = sqrt((dVel*velBin)^2 + (dSig*sigBin)^2)/RMS,
      where velBin and sigBin are the velocity and dispersion in each bin
      and dVel and dSig are the corresponding errors
      (Default: constant errors ERMS=0.05*MEDIAN(RMS)).
  ML: Mass-to-light ratio to multiply the values given by SURF_POT.
      Setting this keyword is completely equivalent to multiplying the
      output RMSMODEL by SQRT(M/L) after the fit. This implies that the
      BH mass becomes MBH*(M/L).
    - If this keyword is set to a negative number in input, the M/L is
      fitted from the data and the keyword returns the best-fitting M/L
      in output. The BH mass of the best-fitting model is MBH*(M/L).
  NORMPSF: Vector of length Q with the fraction of the total PSF flux
      contained in the various circular Gaussians describing the PSF of the
      observations. It has to be total(NORMPSF) = 1. The PSF will be used for
      seeing convolution of the model kinematics.
  NRAD: Number of logarithmically spaced radial positions for which the
      models is evaluated before interpolation and PSF convolution. One may
      want to increase this value if the model has to be evaluated over many
      orders of magnitutes in radius (default: NRAD=50).
  PIXSIZE: Size in arcseconds of the (square) spatial elements at which the
      kinematics is obtained. This may correspond to the size of the spaxel
      or lenslets of an integral-field spectrograph. This size is used to
      compute the kernel for the seeing and aperture convolution.
    - If this is not set, or PIXSIZE = 0, then convolution is not performed.
  /PLOT: Set this keyword to produce a plot at the end of the calculation.
  /QUIET: Set this keyword not to print values on the screen.
  RMS: Vector of length P with the input observed stellar
      V_RMS=sqrt(velBin^2 + sigBin^2) at the coordinates positions
      given by the vector RAD.
    - If RMS is set and ML is negative or not set, then the model is fitted to
      the data, otherwise the adopted ML is used and just the chi^2 is returned.
  SIGMAPSF: Vector of length Q with the dispersion in arcseconds of the
      circular Gaussians describing the PSF of the observations.
    - If this is not set, or SIGMAPSF = 0, then convolution is not performed.
    - IMPORTANT: PSF convolution is done by creating a 2D image, with pixels
      size given by STEP=MAX(SIGMAPSF,PIXSIZE/2)/4, and convolving it with
      the PSF + aperture. If the input radii RAD are very large with respect
      to STEP, the 2D image may require a too large amount of memory. If this
      is the case one may compute the model predictions at small radii
      separately from those at large radii, where PSF convolution is not
      needed.
  STEP: Spatial step for the model calculation and PSF convolution in arcsec.
      This value is automatically computed by default as
      STEP=MAX(SIGMAPSF,PIXSIZE/2)/4. It is assumed that when PIXSIZE or
      SIGMAPSF are big, high resolution calculations are not needed. In some
      cases however, e.g. to accurately estimate the central Vrms in a very
      cuspy galaxy inside a large aperture, one may want to override the
      default value to force smaller spatial pixels using this keyword.
    - Use this keyword to set the desired scale of the model when no PSF or
      pixel convolution is performed (SIGMAPSF=0 or PIXSIZE=0).

OUTPUT PARAMETER:
  RMSMODEL: Vector of length P with the model predictions for the velocity
      second moments (sigma in the spherical non-rotating case) of each bin.

USAGE EXAMPLE:
   A simple usage example is given in the procedure test_jam_sph_rms()
   at the bottom of this file.

REQUIRED ROUTINES:
      By M. Cappellari (included in the JAM distribution):
      - CAP_QUADVA

MODIFICATION HISTORY:
    V1.0.0: Written and tested isotropic case.
       Michele Cappellari, Vicenza, 10 August 2004
    V2.0.0: Included anisotropic case with 1D integral. MC, Oxford, 4 April 2008
    V3.1.0: First released version. MC, Oxford, 12 August 2008
    V3.2.0: Updated documentation. MC, Oxford, 14 August 2008
    V4.0.0: Implemented PSF convolution using interpolation on polar grid.
        Dramatic speed-up of calculation. Further documentation.
        MC, Oxford, 11 September 2008
    V4.0.1: Included keyword STEP. MC, Windhoek, 29 September 2008
    V4.0.2: Added keywords NRAD. Thanks to Michael Williams for reporting possible
        problems with too coarse interpolation. MC, Oxford, 21 November 2008
    V4.1: Added keywords CHI2, ERMS, ML, /PRINT, /QUIET, RMS as in the
        JAM_AXISYMMETRIC_RMS routine. Updated the usage example routine
        TEST_JAM_SPHERICAL_RMS. MC, Oxford, 04 February 2010
    V4.1.1: Correct missing value at R=0 in plot. MC, Oxford, 29 September 2011
    V4.1.2: Use renamed CAP_* routines to avoid potential naming conflicts.
       MC, Paranal, 8 November 2013
    V5.0.0: Translated from IDL into Python. MC, Oxford, 3 April 2014
    V5.0.1: Support both Python 2.6/2.7 and Python 3.x. MC, Oxford, 25 May 2014
    V5.0.2: Updated documentation. MC, Oxford, 5 August 2014

"""

from __future__ import print_function

from scipy import special, ndimage, signal
import numpy as np
from time import clock
import matplotlib.pyplot as plt

from cap_quadva import quadva

##############################################################################

def bilinear_interpolate(xv, yv, im, xout, yout, fill_value=0):
    """
    The input array has size im[ny,nx] as in the output
    of im = f(meshgrid(xv,yv))
    xv and yv are vectors of size nx and ny respectively.
    map_coordinates is equivalent to IDL's INTERPOLATE.

    """
    ny, nx = np.shape(im)
    if  (nx, ny) != (xv.size, yv.size):
        raise ValueError("Input arrays dimensions do not match")

    xi = (nx-1.)/(xv[-1] - xv[0]) * (xout - xv[0])
    yi = (ny-1.)/(yv[-1] - yv[0]) * (yout - yv[0])
    return ndimage.map_coordinates(im.T, [xi, yi], cval=fill_value, order=1)

##############################################################################

def ibetam(a, b, x):
    #
    # Incomplete beta function defined in the same
    # way as the Mathematica function Beta[x,a,b]:
    # Beta[x,a,b] = Integral[t^(a-1) * (1-t)^(b-1), {t,0,x}]
    # This routine works for x<1 and was tested against Mathematica.
    #
    # V1.0: Michele Cappellari, Oxford, 01/APR/2008
    # V2.0: Use Hypergeometric function for negative parameters.
    #    From equation (6.6.8) of Abramoviz & Stegun
    #    http://www.nrbook.com/abramowitz_and_stegun/page_263.htm
    #    MC, Oxford, 04/APR/2008

    if a > 0 and b > 0:
        ib = special.betainc(a,b,x)*special.beta(a,b)
    else:
        ib = x**a/a * special.hyp2f1(a, 1. - b, a + 1., x)

    return ib

##############################################################################

def _integrand(r, sig_l, sig_m, lum, mass, Mbh, rmin, beta):
    #
    # This function implements the integrand of equation (50) of Cappellari (2008).
    # The routine tries to speed up the calculation by treating differently the three
    # cases: (i) isotropic, (ii) constant-anisotropy and (iii) variable-anisotropy.

    h = r[:,np.newaxis]/(np.sqrt(2)*sig_m)
    mass_r = Mbh + np.sum(mass*(special.erf(h) - 2./np.sqrt(np.pi)*h*np.exp(-h**2)), axis=1) # equation (49)

    if np.all(beta == beta[0]):               # Faster constant-anisotropy model
        if beta[0] == 0:                      # Isotropic case
            er = np.sqrt(r**2-rmin**2)        # equation (44)
        else:                                 # Anisotropic case
            rat = (rmin/r)**2
            er = 0.5*rmin/rat**beta[0]*(
                 beta[0]*ibetam(0.5 + beta[0], 0.5, rat) - ibetam(beta[0] - 0.5, 0.5, rat)
                 + np.sqrt(np.pi)*(1.5 - beta[0])*special.gamma(beta[0] - 0.5)/special.gamma(beta[0]) )
        fun = er * np.sum(lum*np.exp(-0.5*(r[:,np.newaxis]/sig_l)**2)/(np.sqrt(2*np.pi)*sig_l)**3, axis=1) # equation (47)
    else:                          # Slower variable-anisotropy model
        fun = 0
        rat = (rmin/r)**2
        for j in range(lum.size):
            if beta[j] == 0:                  # Isotropic case
                er = np.sqrt(r**2-rmin**2)    # equation (44)
            else:                             # Anisotropic case
                er = 0.5*rmin/rat**beta[j]*(  # equation (43)
                     beta[j]*ibetam(0.5 + beta[j], 0.5, rat) - ibetam(beta[j] - 0.5, 0.5, rat)
                     + np.sqrt(np.pi)*(1.5 - beta[j])*special.gamma(beta[j] - 0.5)/special.gamma(beta[j]) )
            fun += er * lum[j]*np.exp(-0.5*(r/sig_l[j])**2)/(np.sqrt(2*np.pi)*sig_l[j])**3

    # This routine returns a vector of values computed at different values of r
    #
    G = 0.00430237    # (km/s)^2 pc/Msun [6.674e-11 SI units (CODATA-10)]

    return 2.*G*fun*mass_r/r**2

##############################################################################

def _second_moment(R, sig_l, sig_m, lum, mass, Mbh, beta,
                     sigmaPsf, normPsf, step, nrad, surf_l, pixSize):
    #
    # This routine gives the second V moment after convolution with a PSF.
    # The convolution is done using interpolation of the model on a
    # polar grid, as described in Appendix A of Cappellari (2008).

    if (max(sigmaPsf) > 0) and (pixSize > 0): # PSF convolution

        # Kernel step is 1/4 of largest value between sigma(min) and 1/2 pixel side.
        # Kernel half size is the sum of 3*sigma(max) and 1/2 pixel diagonal.
        #
        if step == 0:
            step = max(pixSize/2., np.min(sigmaPsf))/4.
        mx = 3*np.max(sigmaPsf) + pixSize/np.sqrt(2)

        # Make grid linear in log of radius RR
        #
        rmax = np.max(R) + mx # Radius of circle containing all data + convolution
        logRad = np.linspace(np.log(step),np.log(rmax),nrad) # Linear grid in log(RR)
        rr = np.exp(logRad)

        # The model Vrms computation is only performed on the radial grid
        # which is then used to interpolate the values at any other location
        #
        wm2Pol = np.empty_like(rr)
        mgePol = np.empty_like(rr)
        rup = 3*np.max(sig_l)
        for j in range(rr.size): # Integration of equation (50)
            wm2Pol[j] = quadva(_integrand, [rr[j], rup],
                               args=(sig_l, sig_m, lum, mass, Mbh, rr[j], beta))[0]
            mgePol[j] = np.sum( surf_l * np.exp(-0.5*(rr[j]/sig_l)**2) )

        nx = np.ceil(rmax/step)
        x1 = np.linspace(-nx, nx, 2*nx)*step
        xCar, yCar = np.meshgrid(x1, x1)  # Cartesian grid for convolution

        # Interpolate MGE model and Vrms over cartesian grid
        #
        r1 = 0.5*np.log(xCar**2 + yCar**2) # Log radius of cartesian grid
        wm2Car = np.interp(r1, logRad, wm2Pol)
        mgeCar = np.interp(r1, logRad, mgePol)

        nk = np.ceil(mx/step)
        kgrid = np.linspace(-nk, nk, 2*nk)*step
        xgrid, ygrid = np.meshgrid(kgrid, kgrid) # Kernel is square

        # Compute kernel with equation (A6) of Cappellari (2008).
        # Normaliztion is irrelevant here as it cancels out.
        #
        kernel = xgrid*0
        dx = pixSize/2
        sp = np.sqrt(2)*sigmaPsf
        for j in range(len(sigmaPsf)):
            kernel = kernel + normPsf[j] \
                * (special.erf((dx-xgrid)/sp[j]) + special.erf((dx+xgrid)/sp[j])) \
                * (special.erf((dx-ygrid)/sp[j]) + special.erf((dx+ygrid)/sp[j]))
        kernel = kernel/np.sum(kernel)

        # Seeing and aperture convolution with equation (A3)
        #
        muCar = np.sqrt(signal.fftconvolve(wm2Car, kernel, mode='same')
                      / signal.fftconvolve(mgeCar, kernel, mode='same'))

        # Interpolate convolved image at observed apertures.
        # Aperture integration was already included in the kernel.
        #
        mu = bilinear_interpolate(x1, x1, muCar, R/np.sqrt(2), R/np.sqrt(2))

    else: # No PSF convolution: just compute values

        mu = np.empty_like(R)
        rmax = 3*np.max(sig_l)
        for j in range(R.size):
            wm2Pol = quadva(_integrand, [R[j], rmax],
                            args=(sig_l, sig_m, lum, mass, Mbh, R[j], beta))[0]
            mgePol = np.sum( surf_l * np.exp(-0.5*(R[j]/sig_l)**2) )
            mu[j] = np.sqrt(wm2Pol/mgePol)

    return mu

##############################################################################

def jam_sph_rms(surf_lum, sigma_lum, surf_pot, sigma_pot, mbh, distance, rad,
                beta=None, normpsf=1, pixsize=0, sigmapsf=0, step=0, nrad=50,
                rms=None, erms=None, ml=None, quiet=False, plot=True):

    if beta is None:
        beta = np.zeros_like(surf_lum)
    if (erms is None) and (rms is not None):
        erms = rms*0 + np.median(rms)*0.05 # Constant ~5% errors

    sigmapsf = np.atleast_1d(sigmapsf)
    normpsf = np.atleast_1d(normpsf)

    pc = distance*np.pi/0.648 # Constant factor to convert arcsec --> pc

    sigmaPsf = sigmapsf*pc
    pixSize = pixsize*pc
    step_pc = step*pc

    sigma_lum_pc = sigma_lum*pc     # Convert from arcsec to pc
    lum = 2*np.pi*surf_lum*sigma_lum_pc**2

    sigma_pot_pc = sigma_pot*pc     # Convert from arcsec to pc
    mass = 2*np.pi*surf_pot*sigma_pot_pc**2

    t = clock()
    rmsModel = _second_moment(rad*pc, sigma_lum_pc, sigma_pot_pc, lum, mass, mbh,
                              beta, sigmaPsf, normpsf, step_pc, nrad, surf_lum, pixSize)

    if not quiet:
        print('jam_sph_rms elapsed time sec: %.2f' % (clock() - t))

    ####### Output and optional M/L fit
    # If RMS keyword is not given all this section is skipped

    if rms is not None:

        if (ml is None) or (ml <= 0):

            # y1 = rms# dy1 = erms (y1 are the data, y2 the model)
            # scale = total(y1*y2/dy1^2)/total(y2^2/dy1^2)  (equation 51)
            #
            scale = np.sum(rms*rmsModel/erms**2) / np.sum((rmsModel/erms)**2)
            ml = scale**2

        else:
            scale = np.sqrt(ml)

        rmsModel *= scale
        chi2 = np.sum( ((rms-rmsModel)/erms)**2 ) / rms.size

        if not quiet:
            print('beta=%.2g; M/L=%.3g; BH=%.3g; chi2/DOF=%.3g' \
                % (beta[0], ml, mbh*ml, chi2))
            mass = 2*np.pi*surf_pot*sigma_pot**2
            print('Total mass MGE: %.4g' % np.sum(mass*ml))

        if plot:
            rad1 = rad.clip(0.38*pixsize)
            ax = plt.gca()
            ax.set_xscale('log')
            ax.set_xlabel('R (arcsec)')
            ax.set_ylabel(r'$\sigma$ (km/s)')
            plt.errorbar(rad1, rms, yerr=erms, fmt='o')
            plt.plot(rad1, rmsModel, 'r')

    else:

        ml = None
        chi2 = None

    return rmsModel, ml, chi2

##############################################################################

def test_jam_sph_rms():
    #
    # This example takes 1s on a 2GHz computer

    # Realistic MGE galaxy surface brightness.
    # The surface brightness is in L_sun/pc^2 and the sigma in arcsec
    #
    surf_pc = np.array([6229., 3089., 5406., 8443., 4283., 1927., 708.8, 268.1, 96.83])
    sigma_arcsec = np.array([0.0374, 0.286, 0.969, 2.30, 4.95, 8.96, 17.3, 36.9, 128.])

    # Realistic observed stellar kinematics. It comes from AO observations
    # at R<2" and seeing-limited long slit observations at larger radii.
    # The galaxy has negligible rotation and we can use sigma as V_RMS
    #
    sig = np.array([395.,390.,387.,385.,380.,365.,350.,315.,310.,290.,260.]) # km/s
    erms = sig*0.02 # assume 2% errors
    rad = np.array([0.15, 0.2, 0.3, 0.4, 0.5, 1, 1.5, 3, 5, 9, 15]) # arcsec

    # Realistic anisotropy profile from a Schwarzschild model.
    # The anisotropy varies smoothly between the following three regimes:
    # 1. beta = -1 for R < 1"
    # 2. beta = 0.3 for 1" < R < 30"
    # 3. beta = -0.2 for R > 30"
    #
    beta = sigma_arcsec*0
    beta[sigma_arcsec < 1] = -1.0
    beta[(sigma_arcsec > 1) & (sigma_arcsec <= 30)] = 0.3
    beta[sigma_arcsec > 30] = -0.2

    # Compute V_RMS profiles and optimize M/L to best fit the data.
    # Assume self-consistency: same MGE for luminosity and potential.
    #
    pixSize = 0.1   # Spaxel size in arcsec
    psf = 0.3/2.355 # sigma of the PSF in arcsec from AO observations
    mbh = 1.5e8 # Black hole mass in solar masses before multiplication by M/L
    distance = 20.    # Mpc

    rmsModel, ml, chi2 = \
        jam_sph_rms(surf_pc, sigma_arcsec, surf_pc, sigma_arcsec,
                    mbh, distance, rad, beta=beta, sigmapsf=psf,
                    pixsize=pixSize, rms=sig, erms=erms, plot=True)    

##############################################################################

if __name__ == '__main__':
    plt.clf()
    test_jam_sph_rms()
