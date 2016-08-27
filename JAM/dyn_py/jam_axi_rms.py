"""
#############################################################################

Copyright (C) 2003-2014, Michele Cappellari
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

#############################################################################

NAME:
  jam_axi_rms()

PURPOSE:
   This procedure calculates a prediction for the projected second
   velocity moment V_RMS = sqrt(V^2 + sigma^2), or optionally any
   of the six components of the symmetric proper motion dispersion
   tensor, for an anisotropic axisymmetric galaxy model.
   It implements the solution of the anisotropic Jeans equations presented
   in equation (28) and note 5 of Cappellari (2008, MNRAS, 390, 71).
   PSF convolution in done as described in the Appendix of that paper:
   http://adsabs.harvard.edu/abs/2008MNRAS.390...71C

CALLING SEQUENCE:

   rmsModel, ml, chi2, flux = \
       jam_axi_rms(surf_lum, sigma_lum, qobs_lum, surf_pot, sigma_pot, qobs_pot,
                    inc, mbh, distance, xbin, ybin, ml=None, normpsf=1, pixang=0,
                    pixsize=0, plot=True, rms=None, erms=None, sigmapsf=0,
                    goodbins=None, quiet=False, beta=None, step=0, rbh=0.01,
                    nrad=20, nang=10, tensor='zz', vmin=None, vmax=None)

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
  INC_DEG: inclination in degrees (90 being edge-on).
  MBH: Mass of a nuclear supermassive black hole in solar masses.
    - VERY IMPORTANT: The model predictions are computed assuming SURF_POT
      gives the total mass. In the common self-consistent case one has
      SURF_POT = SURF_LUM and if requested (keyword ML) the program can scale
      the output RMSMODEL to best fit the data. The scaling is equivalent to
      multiplying *both* SURF_POT and MBH by a factor M/L. To avoid mistakes,
      the actual MBH used by the output model is printed on the screen.
  DISTANCE: distance of the galaxy in Mpc.
  XBIN: Vector of length P with the X coordinates in arcseconds of the bins
      (or pixels) at which one wants to compute the model predictions. The
      X-axis is assumed to coincide with the galaxy projected major axis. The
      galaxy center is at (0,0).
    - When no PSF/pixel convolution is performed (SIGMAPSF=0 or PIXSIZE=0)
      there is a singularity at (0,0) which should be avoided by the input
      coordinates.
  YBIN: Vector of length P with the Y coordinates in arcseconds of the bins
      (or pixels) at which one wants to compute the model predictions. The
      Y-axis is assumed to concide with the projected galaxy symmetry axis.

KEYWORDS:
  BETA: Vector of length N with the anisotropy
      beta_z = 1 - (sigma_z/sigma_R)^2 of the individual MGE Gaussians.
      A scalar can be used if the model has constant anisotropy.
  CHI2: Reduced chi^2 describing the quality of the fit
       chi^2 = total( ((rms[goodBins]-rmsModel[goodBins])/erms[goodBins])^2 )
             / n_elements(goodBins)
  ERMS: Vector of length P with the 1sigma errors associated to the RMS
       measurements. From the error propagation
       ERMS = sqrt((dVel*velBin)^2 + (dSig*sigBin)^2)/RMS,
       where velBin and sigBin are the velocity and dispersion in each bin
       and dVel and dSig are the corresponding errors.
       (Default: constant errors ERMS=0.05*MEDIAN(RMS))
  FLUX: In output this contains a vector of length P with the PSF-convolved
      MGE surface brightness of each bin in Lsun/pc^2, used to plot the
      isophotes on the model results.
  GOODBINS: Vector of length <=P with the indices of the bins which have to
       be included in the fit (if requested) and chi^2 calculation.
       (Default: fit all bins).
  ML: Mass-to-light ratio to multiply the values given by SURF_POT.
      Setting this keyword is completely equivalent to multiplying the
      output RMSMODEL by SQRT(M/L) after the fit. This implies that the
      BH mass becomes MBH*(M/L).
    - If this keyword is set to a negative number in input, the M/L is
      fitted from the data and the keyword returns the best-fitting M/L
      in output. The BH mass of the best-fitting model is MBH*(M/L).
  NORMPSF: Vector of length Q with the fraction of the total PSF flux
      contained in the circular Gaussians describing the PSF of the
      observations. It has to be total(NORMPSF) = 1. The PSF will be used
      for seeing convolution of the model kinematics.
  NRAD: Number of logarithmically spaced radial positions for which the
      models is evaluated before interpolation and PSF convolution. One may
      want to increase this value if the model has to be evaluated over many
      orders of magnitutes in radius (default: NRAD=50). The computation time
      scales as NRAD*NANG.
  NANG: Same as for NRAD, but for the number of angular intervals
      (default: NANG=10).
  PIXANG: angle between the observed spaxels and the galaxy major axis X.
  PIXSIZE: Size in arcseconds of the (square) spatial elements at which the
      kinematics is obtained. This may correspond to the side of the spaxel
      or lenslets of an integral-field spectrograph. This size is used to
      compute the kernel for the seeing and aperture convolution.
    - If this is not set, or PIXSIZE = 0, then convolution is not performed.
  /PLOT: Set this keyword to produce a plot at the end of the calculation.
  /QUIET: Set this keyword not to print values on the screen.
  RBH: This scalar gives the sigma in arcsec of the Gaussian representing the
      central black hole of mass MBH (See Section 3.1.2 of Cappellari 2008).
      The gravitational potential is indistinguishable from a point source
      for radii > 2*RBH, so the default RBH=0.01 arcsec is appropriate in
      most current situations.
    - RBH should not be decreased unless actually needed!
  RMS: Vector of length P with the input observed stellar
      V_RMS=sqrt(velBin^2 + sigBin^2) at the coordinates positions given by
      the vectors XBIN and YBIN.
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
  TENSOR: String specifying the component of the velocity dispersion tensor.
    - TENSOR='xx' gives sigma_xx=sqrt<V_x'^2>  of the component of the
      proper motion dispersion tensor in the direction parallel to the
      projected major axis.
    - TENSOR='yy' gives sigma_yy=sqrt<V_y'^2> of the component of the
      proper motion dispersion tensor in the direction parallel to the
      projected symmetry axis.
    - TENSOR='zz' (default) gives the usual line-of-sight V_rms=sqrt<V_z'^2>.
    - TENSOR='xy' gives the mixed component <V_x'V_y'> of the proper
      motion dispersion tensor.
    - TENSOR='xz' gives the mixed component <V_x'V_z'> of the proper
      motion dispersion tensor.
    - TENSOR='yz' gives the mixed component <V_y'V_z'> of the proper
      motion dispersion tensor.

OUTPUT PARAMETER:
  RMSMODEL: Vector of length P with the model predictions for the velocity
      second moments V_RMS ~ sqrt(vel^2 + sig^2) for each bin.
      Any of the six components of the symmetric proper motion dispersion
      tensor can be provided in output when the TENSOR keyword is used.

USAGE EXAMPLE:
   A simple usage example is given in the procedure test_jam_axi_rms()
   at the end of this file.

REQUIRED ROUTINES:
     By M. Cappellari (included in the JAM distribution):
     - CAP_QUADVA
     - CAP_PLOTVELFIELD

MODIFICATION HISTORY:
    V1.0.0: Written and tested by Michele Cappellari, Vicenza, 19 November 2003
    V2.0.0: Introduced new solution of the MGE Jeans equations with constant
     anisotropy sig_R = b*sig_z. MC, Oxford, 20 September 2007
    V3.1.3: First released version. MC, Oxford, 12 August 2008
    V3.2.0: Updated documentation. MC, Oxford, 14 August 2008
    V4.0.0: Implemented PSF convolution using interpolation on polar grid.
       Dramatic speed-up of calculation. Further documentation.
       MC, Oxford, 11 September 2008
    V4.0.1: Bug fix: when ERMS was not given, the default was not properly set.
       Included keyword STEP. The keyword FLUX is now only used for output:
       the surface brightness for plotting is computed from the MGE model.
       MC, Windhoek, 29 September 2008
    V4.0.2: Added keywords NRAD and NANG. Thanks to Michael Williams for
       reporting possible problems with too coarse interpolation.
       MC, Oxford, 21 November 2008
    V4.0.3: Added keyword RBH. MC, Oxford 4 April 2009
    V4.0.4: Compute FLUX even when not plotting. MC, Oxford, 29 May 2009
    V4.0.5: Skip unnecessary interpolation when computing few points
       without PSF convolution. After feedback from Eric Emsellem.
       MC, Oxford, 6 July 2009
    V4.0.6: Updated documentation. The routine TEST_JAM_AXISYMMETRIC_RMS with
       the usage example now adopts a more realistic input kinematics.
       MC, Oxford, 08 February 2010
    V4.0.7: Forces q_lum && q_pot < 1. MC, Oxford, 01 March 2010
    V4.0.8: Use linear instead of smooth interpolation. After feedback
       from Eric Emsellem. MC, Oxford, 09 August 2010
    V4.0.9: Plot and output with FLUX keyword the PSF-convolved MGE
     surface brightness. MC, Oxford, 15 September 2010
    V4.1.0: Include TENSOR keyword to calculate any of the six components of
     the symmetric proper motion dispersion tensor (as in note 5 of the paper).
     MC, Oxford 19 October 2010
    V4.1.1: Only calculates FLUX if required. MC, Oxford, 8 December 2011
    V4.1.2: Updated documentation. MC, Oxford, 28 May 2012
    V4.1.3: Output FLUX in Lsun/pc^2. MC, Oxford, 1 February 2013
    V4.1.4: Include _EXTRA and RANGE keywords for plotting.
     MC, Oxford, 12 February 2013
    V4.1.5: Use renamed CAP_* routines to avoid potential naming conflicts.
      MC, Paranal, 8 November 2013
    V5.0.0: Translated from IDL into Python. MC, Paranal, 11 November 2013
    V5.0.1: Plot bi-symmetrized V_rms as in IDL version.
      MC, Oxford, 24 February 2014
    V5.0.2: Support both Python 2.6/2.7 and Python 3.x. MC, Oxford, 25 May 2014

"""

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy import special, signal, ndimage, stats
from time import clock
from warnings import simplefilter

from cap_quadva import quadva
from cap_plot_velfield import plot_velfield
from cap_symmetrize_velfield import symmetrize_velfield

simplefilter('ignore', RuntimeWarning)

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

def rotate_points(x, y, ang):
    """
    Rotates points conter-clockwise by an angle ANG in degrees.
    Michele cappellari, Paranal, 10 November 2013

    """
    theta = np.radians(ang)
    xNew = x*np.cos(theta) - y*np.sin(theta)
    yNew = x*np.sin(theta) + y*np.cos(theta)
    return xNew, yNew

##############################################################################

def _integrand(u1,
               dens_lum, sigma_lum, q_lum,
               dens_pot, sigma_pot, q_pot,
               x1, y1, inc, beta, tensor):
    """
    This routine computes the integrand of Eq.(28) of Cappellari (2008; C08)
    for a model with constant anisotropy sigma_R**2 = b*sigma_z**2 and <V_R*V_z> = 0.
    The components of the proper motion dispersions tensor are calculated as
    described in note 5 of C08.
    See Cappellari (2012; C12 http://arxiv.org/abs/1211.7009)
    for explicit formulas for the proper motion tensor.

    """

    dens_lum = dens_lum[:, np.newaxis, np.newaxis]
    sigma_lum = sigma_lum[:, np.newaxis, np.newaxis]
    q_lum = q_lum[:, np.newaxis, np.newaxis]
    beta = beta[:, np.newaxis, np.newaxis]

    dens_pot = dens_pot[np.newaxis, :, np.newaxis]
    sigma_pot = sigma_pot[np.newaxis, :, np.newaxis]
    q_pot = q_pot[np.newaxis, :, np.newaxis]

    u = u1[np.newaxis, np.newaxis, :]

    kani = 1./(1. - beta) # Anisotropy ratio b = (sig_R/sig_z)**2
    ci = np.cos(inc)
    si = np.sin(inc)
    si2 = si**2
    ci2 = ci**2
    x2 = x1**2
    y2 = y1**2
    u2 = u**2

    s2_lum = sigma_lum**2
    q2_lum = q_lum**2
    e2_lum = 1. - q2_lum
    s2q2_lum = s2_lum*q2_lum

    s2_pot = sigma_pot**2
    e2_pot = 1. - q_pot**2

    # Double summation over (j,k) of eq.(28) for all values of integration variable u.
    # The triple loop in (j,k,u) is replaced by broadcasted numpy array operations.
    # The j-index refers to the Gaussians describing the total mass,
    # from which the potential is derived, while the k-index is used
    # for the MGE components describing the galaxy stellar luminosity.

    e2u2_pot = e2_pot*u2
    a = 0.5*(u2/s2_pot + 1./s2_lum)               # equation (29) in C08
    b = 0.5*(e2u2_pot*u2/(s2_pot*(1. - e2u2_pot)) + e2_lum/s2q2_lum) # equation (30) in C08
    c = e2_pot - s2q2_lum/s2_pot                  # equation (22) in C08
    d = 1. - kani*q2_lum - ((1. - kani)*c + e2_pot*kani)*u2 # equation (23) in C08
    e = a + b*ci2
    if tensor == 'xx':
        f = kani*s2q2_lum + d*((y1*ci*(a+b)/e)**2 + si2/(2.*e)) # equation (4) in C12
    elif tensor == 'yy':
        f = s2q2_lum*(si2 + kani*ci2) + d*x2*ci2  # equation (5) in C12
    elif tensor == 'zz':
        f = s2q2_lum*(ci2 + kani*si2) + d*x2*si2  # z' LOS equation (28) in C08
    elif tensor == 'xy':
        f = -d*np.abs(x1*y1)*ci2*(a+b)/e          # equation (6) in C12
    elif tensor == 'xz':
        f = d*np.abs(x1*y1)*si*ci*(a+b)/e         # equation (7) in C12
    elif tensor == 'yz':
        f = si*ci*(s2q2_lum*(1. - kani) - d*x2)   # equation (8) in C12
    else:
        raise RuntimeError('Incorrect TENSOR string')

    # arr has the dimensions (q_lum.size, q_pot.size, u.size)

    arr = q_pot*dens_pot*dens_lum*u2*f*np.exp(-a*(x2 + y2*(a + b)/e)) \
        / ((1. - c*u2)*np.sqrt((1. - e2u2_pot)*e))

    G = 0.00430237    # (km/s)^2 pc/Msun [6.674e-11 SI units (CODATA-10)]

    return 4.*np.pi**1.5*G*np.sum(arr, axis=(0, 1))

##############################################################################

def _vrms2(x, y, inc_deg,
           surf_lum, sigma_lum, qobs_lum,
           surf_pot, sigma_pot, qobs_pot,
           beta, tensor, sigmaPsf, normPsf,
           pixSize, pixAng, step, nrad, nang):
    """
    This routine gives the second V moment after convolution with a PSF.
    The convolution is done using interpolation of the model on a
    polar grid, as described in Appendix A of Cappellari (2008).

    """

    # Axisymmetric deprojection of both luminous and total mass.
    # See equation (12)-(14) of Cappellari (2008)
    #
    inc = np.radians(inc_deg)

    qintr_lum = qobs_lum**2 - np.cos(inc)**2
    if np.any(qintr_lum <= 0):
        raise RuntimeError('Inclination too low q < 0')
    qintr_lum = np.sqrt(qintr_lum)/np.sin(inc)
    if np.any(qintr_lum < 0.05):
        raise RuntimeError('q < 0.05 components')
    dens_lum = surf_lum*qobs_lum / (sigma_lum*qintr_lum*np.sqrt(2*np.pi))

    qintr_pot = qobs_pot**2 - np.cos(inc)**2
    if np.any(qintr_pot <= 0):
        raise RuntimeError('Inclination too low q < 0')
    qintr_pot = np.sqrt(qintr_pot)/np.sin(inc)
    if np.any(qintr_pot < 0.05):
        raise RuntimeError('q < 0.05 components')
    dens_pot = surf_pot*qobs_pot / (sigma_pot*qintr_pot*np.sqrt(2*np.pi))

    # Define parameters of polar grid for interpolation
    #
    w = np.where(sigma_lum < max(abs(x))) # Characteristic MGE axial ratio in observed range

    if len(sigma_lum[w]) < 3:
        qmed = np.median(qobs_lum)
    else:
        qmed = np.median(qobs_lum[w])

    rell = np.sqrt(x**2 + (y/qmed)**2) # Elliptical radius of input (x,y)

    psfConvolution = (max(sigmaPsf) > 0) and (pixSize > 0)

    # Kernel step is 1/4 of largest value between sigma(min) and 1/2 pixel side.
    # Kernel half size is the sum of 3*sigma(max) and 1/2 pixel diagonal.
    #
    if (nrad*nang > x.size) and (not psfConvolution): # Just calculate values

        xPol = x
        yPol = y

    else:  # Interpolate values on polar grid

        if psfConvolution:   # PSF convolution
            if step == 0:
                step = max(pixSize/2., min(sigmaPsf))/4.
            mx = 3*max(sigmaPsf) + pixSize/np.sqrt(2)
        else:                                   # No convolution
            step = min(rell.clip(1)) # Minimum radius of 1pc
            mx = 0

        # Make linear grid in log of elliptical radius RAD and eccentric anomaly ANG
        # See Appendix A
        #
        rmax = max(rell) + mx # Major axis of ellipse containing all data + convolution
        logRad = np.linspace(np.log(step), np.log(rmax), nrad) # Linear grid in np.log(rell)
        ang = np.linspace(0, np.pi/2, nang) # Linear grid in eccentric anomaly
        radGrid, angGrid = np.meshgrid(np.exp(logRad), ang)
        radGrid = np.ravel(radGrid)
        angGrid = np.ravel(angGrid)
        xPol = radGrid*np.cos(angGrid)
        yPol = radGrid*np.sin(angGrid) * qmed

    # The model Vrms computation is only performed on the polar grid
    # which is then used to interpolate the values at any other location
    #
    wm2Pol = np.empty_like(xPol)
    mgePol = np.empty_like(xPol)
    for j in range(xPol.size):
        wm2Pol[j] = quadva(_integrand, [0., 1.],
                            args=(dens_lum, sigma_lum, qintr_lum,
                                  dens_pot, sigma_pot, qintr_pot,
                                  xPol[j], yPol[j], inc, beta, tensor))[0]
        mgePol[j] = np.sum(surf_lum * np.exp(-0.5/sigma_lum**2 *
                           (xPol[j]**2 + (yPol[j]/qobs_lum)**2)))


    if psfConvolution: # PSF convolution

        nx = np.ceil(rmax/step)
        ny = np.ceil(rmax*qmed/step)
        x1 = np.linspace(-nx, nx, 2*nx)*step
        y1 = np.linspace(-ny, ny, 2*ny)*step
        xCar, yCar = np.meshgrid(x1, y1)  # Cartesian grid for convolution

        # Interpolate MGE model and Vrms over cartesian grid
        #
        r1 = 0.5*np.log(xCar**2 + (yCar/qmed)**2) # Log elliptical radius of cartesian grid
        e1 = np.arctan2(abs(yCar/qmed), abs(xCar))    # Eccentric anomaly of cartesian grid

        wm2Car = bilinear_interpolate(logRad, ang, wm2Pol.reshape(nang, nrad), r1, e1)
        mgeCar = bilinear_interpolate(logRad, ang, mgePol.reshape(nang, nrad), r1, e1)

        nk = np.ceil(mx/step)
        kgrid = np.linspace(-nk, nk, 2*nk)*step
        xgrid, ygrid = np.meshgrid(kgrid, kgrid) # Kernel is square
        if pixAng != 0:
            xgrid, ygrid = rotate_points(xgrid, ygrid, pixAng)

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
        muCar = signal.fftconvolve(wm2Car, kernel, mode='same') \
              / signal.fftconvolve(mgeCar, kernel, mode='same')

        # Interpolate convolved image at observed apertures.
        # Aperture integration was already included in the kernel.
        #
        mu = bilinear_interpolate(x1, y1, muCar, x, y)

    else: # No PSF convolution

        muPol = wm2Pol/mgePol

        if nrad*nang > x.size:      # Just returns values
            mu = muPol
        else:                      # Interpolate values
            r1 = 0.5*np.log(x**2 + (y/qmed)**2) # Log elliptical radius of input (x,y)
            e1 = np.arctan2(abs(y/qmed), abs(x))    # Eccentric anomaly of input (x,y)
            mu = bilinear_interpolate(logRad, ang, muPol.reshape(nang, nrad), r1, e1)

    return mu

##############################################################################

def jam_axi_rms(surf_lum, sigma_lum, qobs_lum, surf_pot, sigma_pot, qobs_pot,
                  inc, mbh, distance, xbin, ybin, ml=None, normpsf=1, pixang=0,
                  pixsize=0, plot=True, rms=None, erms=None, sigmapsf=0,
                  goodbins=None, quiet=False, beta=None, step=0, nrad=20,
                  nang=10, rbh=0.01, tensor='zz', vmin=None, vmax=None):

    """
    This procedure calculates a prediction for the projected second
    velocity moment V_RMS = sqrt(V^2 + sigma^2), or optionally any
    of the six components of the symmetric proper motion dispersion
    tensor, for an anisotropic axisymmetric galaxy model.
    It implements the solution of the anisotropic Jeans equations presented
    in equation (28) and note 5 of Cappellari (2008, MNRAS, 390, 71).
    PSF convolution in done as described in the Appendix of that paper.
    http://adsabs.harvard.edu/abs/2008MNRAS.390...71C

    CALLING SEQUENCE:

    rmsModel, ml, chi2, flux = \
       jam_axi_rms(surf_lum, sigma_lum, qobs_lum, surf_pot, sigma_pot, qobs_pot,
                    inc, mbh, distance, xbin, ybin, ml=None, normpsf=1, pixang=0,
                    pixsize=0, plot=True, rms=None, erms=None, sigmapsf=0,
                    goodbins=None, quiet=False, beta=None, step=0, nrad=20, nang=10,
                    rbh=0.01, tensor='zz', vmin=None, vmax=None)

    See the file jam_axi_rms.py for detailed documentation.

    """

    if beta is None:
        beta = surf_lum*0 # Anisotropy parameter beta = 1 - (sig_z/sig_R)**2
    if (beta.size != surf_lum.size) or (sigma_lum.size != surf_lum.size) \
        or (qobs_lum.size != surf_lum.size):
        raise ValueError("The luminous MGE components do not match")
    if (sigma_pot.size != surf_pot.size) or (qobs_pot.size != surf_pot.size):
        raise ValueError("The total mass MGE components do not match")
    if (erms is None) and (rms is not None):
        erms = rms*0 + np.median(rms)*0.05 # Constant ~5% errors

    sigmapsf = np.atleast_1d(sigmapsf)
    normpsf = np.atleast_1d(normpsf)

    pc = distance*np.pi/0.648 # Constant factor to convert arcsec --> pc

    surf_lum_pc = surf_lum
    surf_pot_pc = surf_pot
    sigma_lum_pc = sigma_lum*pc         # Convert from arcsec to pc
    sigma_pot_pc = sigma_pot*pc         # Convert from arcsec to pc
    xbin_pc = xbin*pc                   # Convert all distances to pc
    ybin_pc = ybin*pc
    pixSize_pc = pixsize*pc
    sigmaPsf_pc = sigmapsf*pc
    step_pc = step*pc

    # Add a Gaussian with small sigma and the same total mass as the BH.
    # The Gaussian provides an excellent representation of the second moments
    # of a point-like mass, to 1% accuracy out to a radius 2*sigmaBH.
    # The error increses to 14% at 1*sigmaBH, independently of the BH mass.
    #
    if mbh > 0:
        sigmaBH_pc = rbh*pc # Adopt for the BH just a very small size
        surfBH_pc = mbh/(2*np.pi*sigmaBH_pc**2)
        surf_pot_pc = np.append(surfBH_pc, surf_pot_pc) # Add Gaussian to potential only!
        sigma_pot_pc = np.append(sigmaBH_pc, sigma_pot_pc)
        qobs_pot = np.append(1., qobs_pot)  # Make sure vectors do not have extra dimensions

    qobs_lum = qobs_lum.clip(0, 0.999)
    qobs_pot = qobs_pot.clip(0, 0.999)

    t = clock()
    rmsModel = _vrms2(xbin_pc, ybin_pc, inc, surf_lum_pc, sigma_lum_pc,
                     qobs_lum, surf_pot_pc, sigma_pot_pc, qobs_pot, beta,
                     tensor, sigmaPsf_pc, normpsf, pixSize_pc, pixang,
                     step_pc, nrad, nang)
    if not quiet:
        print('jam_axi_rms elapsed time sec: %.2f' % (clock() - t))

    if tensor in ('xx', 'yy', 'zz'):
        rmsModel = np.sqrt(rmsModel.clip(0))   # Return SQRT and fix possible rounding errors
    if tensor in ('xy', 'xz'):
        rmsModel *= np.sign(xbin*ybin) # Calculation was done in positive quadrant

    # Analytic convolution of the MGE model with an MGE circular PSF
    # using Equations (4,5) of Cappellari (2002, MNRAS, 333, 400)
    #
    lum = surf_lum_pc*qobs_lum*sigma_lum**2 # Luminosity/(2np.pi) of each Gaussian
    flux = xbin*0 # Total MGE surface brightness for plotting
    for k in range(len(sigmapsf)): # loop over the PSF Gaussians
        sigmaX = np.sqrt(sigma_lum**2 + sigmapsf[k]**2)
        sigmaY = np.sqrt((sigma_lum*qobs_lum)**2 + sigmapsf[k]**2)
        surfConv = lum / (sigmaX*sigmaY) # PSF-convolved in Lsun/pc**2
        for j in range(len(surf_lum_pc)): # loop over the galaxy MGE Gaussians
            flux += normpsf[k]*surfConv[j]*np.exp(-0.5*((xbin/sigmaX[j])**2 + (ybin/sigmaY[j])**2))

    ####### Output and optional M/L fit
    # If RMS keyword is not given all this section is skipped

    if rms is not None:

        # Only consider the good bins for the chi**2 estimation
        #
        if goodbins is None:
            goodbins = np.arange(xbin.size)

        if (ml is None) or (ml <= 0):

            # y1 = rms# dy1 = erms (y1 are the data, y2 the model)
            # scale = total(y1*y2/dy1**2)/total(y2**2/dy1**2)  (equation 51)
            #
            scale = np.sum(rms[goodbins]*rmsModel[goodbins]/erms[goodbins]**2) \
                  / np.sum((rmsModel[goodbins]/erms[goodbins])**2)
            ml = scale**2

        else:
            scale = np.sqrt(ml)

        rmsModel *= scale
        chi2 = np.sum(((rms[goodbins]-rmsModel[goodbins])/erms[goodbins])**2) / len(goodbins)

        if not quiet:
            print('inc=%.1f beta_z=%.2f M/L=%.3g BH=%.2e chi2/DOF=%.3g' % (inc, beta[0], ml, mbh*ml, chi2))
            mass = 2*np.pi*surf_pot_pc*qobs_pot*sigma_pot_pc**2
            print('Total mass MGE: %.4g' % np.sum(mass*ml))

        if plot:

            rms1 = rms.copy() # Only symmetrize good bins
            rms1[goodbins] = symmetrize_velfield(xbin[goodbins], ybin[goodbins], rms[goodbins])
            chi2_sym = np.sum(((rms1[goodbins]-rmsModel[goodbins])/erms[goodbins])**2) / len(goodbins)
            
            if (vmin is None) or (vmax is None):
                vmin, vmax = stats.scoreatpercentile(rms1[goodbins], [0.5, 99.5])

            plt.clf()
            plt.subplot(121)
            plot_velfield(xbin, ybin, rms1, vmin=vmin, vmax=vmax, flux=flux, colorbar=True)
            plt.title(r"Input $V_{\rm rms}$")

            plt.subplot(122)
            plot_velfield(xbin, ybin, rmsModel, vmin=vmin, vmax=vmax, flux=flux, colorbar=True)
            plt.title(r"Model $V_{\rm rms}$, chi2="+str(chi2_sym))
            plt.tight_layout()

    else:

        ml = None
        chi2 = None

    return rmsModel, ml, chi2, flux

##############################################################################

def test_jam_axi_rms():

    """
    Usage example for jam_axi_rms.
    It takes about 2s on a 2.5 GHz computer

    """
    np.random.seed(123)
    xbin, ybin = np.random.uniform(low=[-60,-40], high=[60,40], size=[1000,2]).T

    inc = 60.                                                # Assumed galaxy inclination
    r = np.sqrt(xbin**2 + (ybin/np.cos(np.radians(inc)))**2) # Radius in the plane of the disk
    a = 40                                                   # Scale length in arcsec
    vr = 2000*np.sqrt(r)/(r+a)                               # Assumed velocity profile
    vel = vr * np.sin(np.radians(inc))*xbin/r                # Projected velocity field
    sig = 8700/(r+a)                                         # Assumed velocity dispersion profile
    rms = np.sqrt(vel**2 + sig**2)                           # Vrms field in km/s

    surf = np.array([39483., 37158., 30646., 17759., 5955.1, 1203.5, 174.36, 21.105, 2.3599, 0.25493])
    sigma = np.array([0.153, 0.515, 1.58, 4.22, 10, 22.4, 48.8, 105, 227, 525])
    qObs = sigma*0 + 0.57

    distance = 16.5   # Assume Virgo distance in Mpc (Mei et al. 2007)
    mbh = 1e8 # Black hole mass in solar masses
    beta = surf*0 + 0.2

    surf_lum = surf # Assume self-consistency
    sigma_lum = sigma
    qobs_lum = qObs
    surf_pot = surf
    sigma_pot = sigma
    qobs_pot = qObs

    sigmapsf = 0.6
    pixsize = 0.8

    # The model is similar but not identical to the adopted kinematics!

    rmsModel, ml, chi2, flux = \
        jam_axi_rms(surf_lum, sigma_lum, qobs_lum, surf_pot, sigma_pot, qobs_pot,
                     inc, mbh, distance, xbin, ybin, plot=True, rms=rms,
                     sigmapsf=sigmapsf, beta=beta, pixsize=pixsize)

##############################################################################

if __name__ == '__main__':
    test_jam_axi_rms()
