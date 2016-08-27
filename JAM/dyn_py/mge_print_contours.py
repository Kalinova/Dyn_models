"""
#####################################################################

Copyright (C) 1999-2014, Michele Cappellari
E-mail: cappellari_at_astro.ox.ac.uk

For details on the method see:
  Cappellari M., 2002, MNRAS, 333, 400

Updated versions of the software are available from my web page
http://purl.org/cappellari/software

If you have found this software useful for your
research, I would appreciate an acknowledgment to use of
`the MGE fitting method and software by Cappellari (2002)'.

This software is provided as is without any warranty whatsoever.
Permission to use, for non-commercial purposes is granted.
Permission to modify for personal or internal use is granted,
provided this copyright and disclaimer are included unchanged
at the beginning of the file. All other rights are reserved.

#####################################################################

NAME:
    MGE_PRINT_CONTOURS

AUTHOR:
      Michele Cappellari, Astrophysics Sub-department, University of Oxford, UK

PURPOSE:
      Produces a contour plot comparing a convolved
      MGE model to the original fitted image.

CALLING SEQUENCE:
      MGE_PRINT_CONTOURS, Img, Ang, Xc, Yc, Sol, $
          FILE=file, MAGRANGE=magRange, SCALE=scale, BINNING=binning, $
          SIGMAPSF=[sigma1,sigma2,...], NORMPSF=[norm1,norm2,...], $
          MODEL=model, /CONVOL

INPUTS:
      Img = array containing the image that was fitted by MGE_FIT_SECTORS
      Ang = Scalar giving the common Position Angle of the Gaussians.
              This is measured counterclockwise from the image Y axis to
              the Gaussians major axis, as measured by FIND_GALAXY.
      Xc = Scalar giving the common X coordinate in pixels of the
              center of the Gaussians.
      Yc = Scalar giving the common Y coordinate in pixels of the
              center of the Gaussians.
      SOL - Array containing a 3xNgauss array with the MGE best-fitting
              solution as produced by MGE_FIT_SECTORS:
              1) sol[0,*] = TotalCounts, of the Gaussians components.
                  The relation TotalCounts = Height*(2*!PI*Sigma^2*qObs)
                  can be used compute the Gaussian central surface
                  brightness (Height)
              2) sol[1,*] = Sigma, is the dispersion of the best-fitting
                  Gaussians in pixels.
              3) sol[2,*] = qObs, is the observed axial ratio of the
                  best-fitting Gaussian components.

OUTPUTS:
      No output parameters. The results are plotted in a PS file.
      But see also the optional MODEL keyword.

OPTIONAL KEYWORDS:
      FILE - String: name of the output PS file.
      MAGRANGE - Scalar giving the range in magnitudes of the equally
              spaced contours to plot, in steps of 0.5 mag/arcsec^2,
              starting from the model maximum value.
              (default: from maximum of the model to 10 magnitudes below that)
      SCALE - The pixel scale in arcsec/pixels used for the plot axes.
              (default: 1)
      BINNING - Pixels to bin together before plotting.
              Helps producing MUCH smaller PS files (default: no binning).
      /CONVOL - Set this keyword to use Gaussian convolution instead of the
              default SMOOTH routine to smooth the galaxy image before rebining
              (when the BINNING keyword is used). Noise suppression is in principle
              more effective when this keyword is set but the computation
              time and memory requirement can be significantly larger.
      SIGMAPSF - Scalar giving the sigma of the PSF, or vector with the
              sigma of an MGE model for the circular PSF. (Default: no convolution)
      NORMPSF - This is optional if only a scalar is given for SIGMAPSF,
              otherwise it must contain the normalization of each MGE component
              of the PSF, whose sigma is given by SIGMAPSF. The vector needs to
              have the same number of elements of SIGMAPSF and the condition
              TOTAL(normpsf) = 1 must be verified. In other words the MGE PSF
              needs to be normalized. (default: 1).
      MODEL -- Named variable that will contain in output an image with
              the MGE model, convolved with the given PSF.
      _EXTRA - Additional parameters can be passed to the IDL CONTOUR
              procedure via the _EXTRA mechanism

EXAMPLE:
      The sequence of commands below was used to generate the complete
      MGE model of the Figures 8-9 in Cappellari (2002).
      1) The FITS file is read sky is subtracted#
      2) the photometry along sectors is performed with SECTORS_PHOTOMETRY#
      3) the resulting measurements are fitted with MGE_FIT_SECTORS#
      4) the contour are printed on a PS file with MGE_PRINT_CONTOURS.
      The geometric parameters of the galaxy (eps,ang,xc,yc) were
      previously determined using FIND_GALAXY.

          fits_read, 'ngc4342_f814w.fits', img, h

          skylev = 0.55       # In counts
          img = img - skylev  # Subtract the sky from the image
          scale = 0.0455      # WFPC2/PC1 CCD pixel scale in arcsec/pixels

          sigmaPSF = 0.8      # Here I use one single Gaussian PSF
          eps = 0.66          # These values were measured with FIND_GALAXY
          ang = 35.7
          xc = 366
          yc = 356

          sectors_photometry, img, eps, ang, xc, yc, radius, angle, counts, $
              MINLEVEL=0.2, N_SECTORS=19

          MGE_fit_sectors, radius, angle, counts, eps, $
              NGAUSS=13, SIGMAPSF=sigmaPSF, SOL=sol, /PRINT, SCALE=scale

          MGE_print_contours, img, ang, xc, yc, sol, BINNING=3, $
              FILE='ngc4342.ps', SCALE=scale, MAGRANGE=9, SIGMAPSF=sigmaPSF

PROCEDURES USED:
      The following procedures are contained in the main MGE_PRINT_CONTOURS program.
          GAUSS2D_MGE  -- Returns a 2D Gaussian image
          MULTI_GAUSS  -- Returns a 2D MGE expansion by calling GAUSS2D_MGE
      Astronomy User's Library (http://idlastro.gsfc.nasa.gov/) routines needed:
          CONVOLVE  -- Fast convolution using FFT

MODIFICATION HISTORY:
      V1.0.0 First implementation, Padova, February 1999, Michele Cappellari
      V2.0.0 Major revisions, Leiden, January 2000, MC
      V2.1.0 Updated documentation, Leiden, 8 October 2001, MC
      V2.2.0 Implemented MGE PSF, Leiden, 29 October 2001, MC
      V2.3.0 Added MODEL keyword, Leiden, 30 October 2001, MC
      V2.3.1 Added compilation options, MC, Leiden 20 May 2002
      V2.3.2: Use N_ELEMENTS instead of KEYWORD_SET to test
          non-logical keywords. Leiden, 9 May 2003, MC
      V2.4.0: Convolve image with a Gaussian kernel instead of using
          the SMOOTH function before binning. Always shows contours
          in steps of 0.5 mag/arcsec^2. Replaced LOGRANGE and NLEVELS
          keywords with MAGRANGE. Leiden, 30 April 2005, MC
      V2.4.1: Added /CONVOL keyword. MC, Oxford, 23 September 2008
      V2.4.2: Use Coyote Library to select red contour levels for MGE model.
          MC, Oxford, 8 August 2011
      V3.0.0: Translated from IDL into Python.
          MC, Aspen Airport, 8 February 2014
      V3.0.1: Use input scale to label axis if given. Avoid use of log.
          Use data rather than model as reference for the contour levels.
          Allow for a scalar sigmaPSF. MC, Oxford, 18 September 2014

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

#----------------------------------------------------------------------------

def _gauss2d_mge(n, xc, yc, sx, sy, pos_ang):
    """
    Returns a 2D Gaussian image with size N[0]xN[1], center (XC,YC),
    sigma (SX,SY) along the principal axes and position angle POS_ANG, measured
    from the positive Y axis to the Gaussian major axis (positive counter-clockwise).

    """
    ang = np.radians(pos_ang - 90.)
    x, y = np.ogrid[0:n[0], 0:n[1]] - np.array([xc, yc])

    xcosang = np.cos(ang)/(np.sqrt(2.)*sx)*x
    ysinang = np.sin(ang)/(np.sqrt(2.)*sx)*y
    xsinang = np.sin(ang)/(np.sqrt(2.)*sy)*x
    ycosang = np.cos(ang)/(np.sqrt(2.)*sy)*y

    im = (xcosang + ysinang)**2 + (ycosang - xsinang)**2

    return np.exp(-im)

#----------------------------------------------------------------------------

def _multi_gauss(pars, img, sigmaPSF, normPSF, xpeak, ypeak, theta):

    ngauss = pars.size // 3
    lum = pars[0, :]
    sigma = pars[1, :]
    q = pars[2, :]

    # Analytic convolution with an MGE circular Gaussian
    # Eq.(4,5) in Cappellari (2002)
    #
    u = 0.
    for j in range(ngauss):
        for k in range(sigmaPSF.size):
            sx = np.sqrt(sigma[j]**2 + sigmaPSF[k]**2)
            sy = np.sqrt((sigma[j]*q[j])**2 + sigmaPSF[k]**2)
            g = _gauss2d_mge(img.shape, xpeak, ypeak, sx, sy, theta)
            u += lum[j]*normPSF[k]/(2.*np.pi*sx*sy) * g

    return u

#----------------------------------------------------------------------------

def mge_print_contours(img, ang, xc, yc, sol, magrange=10, scale=None,
                       binning=None, sigmapsf=0, normpsf=1):

    sigmapsf = np.atleast_1d(sigmapsf)
    normpsf = np.atleast_1d(normpsf)

    if normpsf.size != sigmapsf.size:
        raise ValueError('sigmaPSF and normPSF must have the same length')
    if round(np.sum(normpsf)*100) != 100:
        raise ValueError('Error: PSF not normalized')

    model = _multi_gauss(sol, img, sigmapsf, normpsf, xc, yc, ang)
    levels = img[xc,yc] * 10**(-0.4*np.arange(0, magrange, 0.5)) # 0.5 mag/arcsec^2 steps

    if binning is None:
        binning = 1
    else:
        model = ndimage.filters.gaussian_filter(model, binning/2.355)
        model = ndimage.zoom(model, 1./binning, order=1)
        img = ndimage.filters.gaussian_filter(img, binning/2.355)
        img = ndimage.zoom(img, 1./binning, order=1)

    ax = plt.gca()
    ax.axis('equal')
    ax.set_adjustable('box-forced')
    s = img.shape

    if scale is None:
        extent = [0, s[0], 0, s[1]]
        plt.xlabel("pixels")
        plt.ylabel("pixels")
    else:
        extent = np.array([0, s[0], 0, s[1]])*scale*binning
        plt.xlabel("arcsec")
        plt.ylabel("arcsec")

    ax.contour(img, levels, colors = 'k', linestyles='solid', extent=extent)
    ax.contour(model, levels, hold='on', colors='r', linestyles='solid', extent=extent)

#----------------------------------------------------------------------------
