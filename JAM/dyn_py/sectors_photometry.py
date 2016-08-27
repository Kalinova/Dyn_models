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
      SECTORS_PHOTOMETRY

AUTHOR:
      Michele Cappellari, Astrophysics Sub-department, University of Oxford, UK

PURPOSE:
      Perform photometry of a galaxy image along sectors equally spaced in
      angle. This routine assumes four-fold symmetry, so measurements in
      the four quadrants are averaged together. This routine is useful to
      generate the input photometry required by the MGE fitting routine
      MGE_FIT_SECTORS.

EXPLANATION:
      Further information on SECTORS_PHOTOMETRY is available in
      Cappellari M., 2002, MNRAS, 333, 400
      http://adsabs.harvard.edu/abs/2002MNRAS.333..400C

CALLING SEQUENCE:
      SECTORS_PHOTOMETRY, Img, Eps, Ang, Xc, Yc, Radius, Angle, Counts,
          N_SECTORS=n_sectors, SECTOR_WIDTH=sector_width,
          BADPIXELS=badpixels, MINLEVEL=minlevel

INPUTS:
      Img = The galaxy images as a 2D array.
      Eps = The galaxy "average" ellipticity Eps = 1-b/a = 1-q'.
          Photometry will be measured along elliptical annuli with
          constant axial ellipticity Eps. The four quantities
          (Eps,Ang,Xc,Yc) can be measured with the routine FIND_GALAXY.
      Ang = Position angle measured counterclockwise from
          the image Y axis, to the galaxy major axis.
      Xc = X coordinate of the galaxy center in pixels.
      Yc = Y coordinate of the galaxy center in pixels.

OUTPUTS:
      Radius = Vector containing the radius of the surface brightness
              measurements, taken from the galaxy center. This is given
              in units of PIXELS (!).
      Angle = Vector containing the polar angle of the surface brightness
              measurements, taken from the galaxy major axis.
      Counts = Vector containing the actual surface brightness measurements
              in COUNTS (!) at the polar coordinates specified by the vectors
              Radius and Angle. These three vectors need to have the same
              number of elements.

OPTIONAL INPUT KEYWORDS:
      N_SECTORS - Number of the sectors, equally spaced in angle, from
              the galaxy major axis to the minor axis (one quadrant).
              (default: 19 to cover the whole image with 5 degrees width).
      SECTOR_WIDTH - Scalar giving the width of the sectors
              in degrees (default: 5 degrees)
      BADPIXELS - Vector containing the indices of the pixels in the
              array Img that have to be excluded from the photometry
              calculation. Pixels indices can be selected using the
              WHERE or POLYFILLV commands or interactively with
              the DEFROI routine.
      MINLEVEL - Scalar giving the minimum COUNTS level to include
              in the photometry. The measurement along one profile
              will stop when the counts go below this level.

EXAMPLE:
      The sequence of commands below was used to generate the complete
      MGE model of the Figures 8-9 in Cappellari (2002).
      1) The FITS file with the galaxy is read and the sky is subtracted#
      2) the photometry along sectors is performed with SECTORS_PHOTOMETRY#
      3) the resulting measurements are fitted with MGE_FIT_SECTORS#
      4) the contour are printed on a PS file with MGE_PRINT_CONTOURS.
      The geometric parameters of the galaxy (eps,ang,xc,yc) were
      previously determined using FIND_GALAXY.

          fits_read, 'ngc4342_f814w.fits', img, h

          skylev = 0.55
          img = img - skylev
          scale = 0.0455  # WFPC2/PC1 CCD pixel scale in arcsec/pixels

          sigmaPSF = 0.8  # Here I use one single Gaussian PSF
          eps = 0.66      # These values were measured with FIND_GALAXY
          ang = 35.7
          xc = 366
          yc = 356
          ngauss = 13

          sectors_photometry, img, eps, ang, xc, yc, radius, angle, counts, $
              MINLEVEL=0.2, /PLOT

          MGE_fit_sectors, radius, angle, counts, eps, $
              NGAUSS=ngauss, SIGMAPSF=sigmaPSF, SOL=sol, /PRINT, SCALE=scale

          MGE_print_contours, img, ang, xc, yc, sol, BINNING=3, $
              FILE='ngc4342.ps', SCALE=scale, MAGRANGE=9, SIGMAPSF=sigmaPSF

PROCEDURES USED:
      The following procedures are contained in the main SECTORS_PHOTOMETRY program.
          _ANGLES_IMAGE   -- construct an image of position angles
          _ELLIPSES_IMAGE -- construct an image of elliptical radii
          _PROFILE        -- perform photometry along ONE position angle

MODIFICATION HISTORY:
      V1.0.0: First implementation for the NGC2681 photometric modeling.
          Michele Cappellari, ESO Garching, 27 september 1999
      V2.0.0: Major revisions, to use it with MGE_FIT_SECTORS.
          Leiden, January 2000, MC
      V2.1.0: Further updates, Padova, August 2000, MC
      V2.1.1: Added compilation options, MC, Leiden 20 May 2002
      V2.1.2: Allow for N_SECTORS=1 to get a single profile centered at
          a given PA. Use biweight mean insted of sigma-clipped mean.
          MC, Leiden, 30 April 2004
      V2.1.3: Reduced amount of verbose output. MC, Leiden, 24 October 2004
      V2.1.4: Replaced LOGRANGE keyword in example with the new MAGRANGE.
          MC, Leiden, 1 May 2005
      V2.1.5: Forces image to be positive when computing weighted radius
          to prevent possible negative radii at large radii. Thanks to
          Michael Williams for reporting the problem and the fix.
          MC, Oxford, 16 February 2009
      V3.0.0: Translated from IDL into Python. MC, Aspen Airport, 8 February 2014
      V3.0.1: Support both Python 2.6/2.7 and Python 3.x. MC, Oxford, 25 May 2014
      V3.1.0: Improved image visualization of sampled photometric grid.
          Sample angles uniformly in eccentric anomaly rather than polar angle.
          Removed Scipy dependency. MC, Oxford, 23 September 2014

"""

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------

def _biweight_mean(y, itmax=10):
    """
    Biweight estimate of the location (mean).
    Implements the approach described in
    "Understanding Robust and Exploratory Data Analysis"
    Hoaglin, Mosteller, Tukey ed., 1983

    """
    y = np.ravel(y)
    c = 6.
    fracmin = 0.03*np.sqrt(0.5/(y.size - 1))
    y0 = np.median(y)
    mad = np.median(np.abs(y - y0))

    for it in range(itmax):
        u2 = ((y - y0)/(c*mad))**2
        u2 = u2.clip(0, 1)
        w = (1 - u2)**2
        y0 += np.sum(w*(y - y0))/np.sum(w)
        mad_old = mad
        mad = np.median(np.abs(y - y0))
        frac = np.abs(mad_old - mad)/mad
        if frac < fracmin:
            break

    return y0

#----------------------------------------------------------------------------

def _coordinates(q, pos_ang, xc, yc, s):

    ang = np.radians(90 - pos_ang)   # x-axis is major axis
    x, y = np.ogrid[:s[0], :s[1]] - np.array([xc, yc])
    x, y = x*np.cos(ang) - y*np.sin(ang), x*np.sin(ang) + y*np.cos(ang)
    x2, y2 = x**2, y**2
    rad = np.sqrt(x2 + y2)                  # Radius
    rell = np.sqrt(x2 + y2/q**2)            # Elliptical radius
    ecc = np.arctan2(np.abs(y/q), np.abs(x))   # Eccentric anomaly [0, pi/2]

    return rad, rell, ecc

#----------------------------------------------------------------------------

class sectors_photometry(object):

    def __init__(self, img, eps, ang, xc, yc, badpixels=None,
                  n_sectors=19, minlevel=0, plot=False):
        """
        This routine performs photometry along sectors linearly spaced
        in eccentric anomaly between the major and minor axis of a galaxy.
        In output it returns the three vectors RADIUS, ANGLE, CNT,
        containing the photometric measurements in polar coordinates.

        """        
        xc, yc = int(round(xc)), int(round(yc))
        s = img.shape
        q = 1 - eps
        minlevel = max(minlevel, 0)

        rad, rell, ecc = _coordinates(q, ang, xc, yc, s)
        rad[xc, yc] = 0.38  # Average radius within the central pixel
        rell[xc, yc] = 0.38

        if plot:
            self.mask = np.zeros_like(img, dtype=bool)

        # Sample radii with 24 isophotes per decade: factor 1.1 spacing.
        # Sample eccentric anomaly with n_sectors from 0-pi/2

        rell = np.round(24.2*np.log10(rell))
        ecc = np.round(2*(n_sectors - 1)/np.pi*ecc)

        if badpixels is not None:
             ecc.flat[badpixels] = -1  # Negative flag value

        self.radius = self.counts = self.angle = []
        eccGrid = np.linspace(0, np.pi/2, n_sectors)       # Eccentric anomaly
        angGrid = np.degrees(np.arctan(np.tan(eccGrid)*q)) # Polar angle

        for k, angj in enumerate(angGrid):
            radj, cntj = self._profile(
                    img, xc, yc, rad, rell, ecc, k, plot, minlevel)
            self.radius = np.append(self.radius, radj)
            self.counts = np.append(self.counts, cntj)
            self.angle = np.append(self.angle, radj*0 + angj)

        if plot:
            plt.imshow(np.log(img.clip(img[xc, yc]/1e4)), cmap='hot',
                       origin='lower', interpolation='none')
            plt.imshow(self.mask, cmap='binary', alpha=0.3,
                       origin='lower', interpolation='none')
            plt.xlabel("pixels")
            plt.ylabel("pixels")

#----------------------------------------------------------------------------

    def _profile(self, data, xc, yc, rad, rell, ecc, k, plot, minlevel):

        if ecc[xc, yc] != -1:
            ecc[xc, yc] = k  # Always include central pixel unless bad
        good = np.flatnonzero(ecc == k)
        img = rell.flat[good]
        levels = np.unique(img)  # get unique levels within sector
        cnt = np.empty_like(levels)
        radius = np.empty_like(levels)

        for j, lev in enumerate(levels):
            sub = good[img == lev]
            if sub.size > 10:  # Evaluate a biweight mean
                cnt[j] = _biweight_mean(data.flat[sub])
            else:
                cnt[j] = np.mean(data.flat[sub])  # Usual mean

            if plot:
                self.mask.flat[sub] = (lev + k % 2) % 2

            if cnt[j] < minlevel:  # drop last value
                cnt = cnt[:j]
                radius = radius[:j]
                break

            # Luminosity-weighted average radius in pixels

            flx = data.flat[sub].clip(0)
            radius[j] = np.sum(rad.flat[sub]*flx)/np.sum(flx)

        j = np.argsort(radius)
        cnt = cnt[j]
        radius = radius[j]

        return radius, cnt

#----------------------------------------------------------------------------
