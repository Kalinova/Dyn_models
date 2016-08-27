#!/usr/bin/env python

"""
    This example illustrates how to obtain an MGE fit from a galaxy image
    using the mge_fit_sectors package and how to verify the results.

    V1.0.0: Translated from the corresponding IDL version.
        Michele Cappellari, Aspen Airport, 8 February 2014
    V1.0.1: Fixed incorrect offset in high-res contour plot.
        Use arcsec pixel scale. MC, Oxford, 18 September 2014

"""

import pyfits
import matplotlib.pyplot as plt

from sectors_photometry import sectors_photometry
from mge_print_contours import mge_print_contours
from find_galaxy import find_galaxy
from mge_fit_sectors import mge_fit_sectors

#----------------------------------------------------------------------------

def mge_fit_example():
    """
    This procedure reproduces Figures 8-9 in Cappellari (2002)
    This example illustrates a simple MGE fit to one single HST/WFPC2 image.

    """

    file = "ngc4342_f814w_pc.fits"

    hdu = pyfits.open(file)
    img = hdu[0].data

    skylev = 0.55 # counts/pixel
    img -= skylev  # subtract sky
    scale = 0.0455 # arcsec/pixel
    minlevel = 0.2 # counts/pixel
    ngauss = 12

    # Here we use an accurate four gaussians MGE PSF for
    # the HST/WFPC2/F814W filter, taken from Table 3 of
    # Cappellari et al. (2002, ApJ, 578, 787)

    sigmaPSF = [0.494, 1.44, 4.71, 13.4]      # In PC1 pixels
    normPSF = [0.294, 0.559, 0.0813, 0.0657]  # total(normPSF)=1

    # Here we use FIND_GALAXY directly inside the procedure. Usually you may want
    # to experiment with different values of the FRACTION keyword, before adopting
    # given values of Eps, Ang, Xc, Yc.

    plt.clf()
    f = find_galaxy(img, fraction=0.04, plot=True)
    plt.pause(0.01)  # Allow plot to appear on the screen

    # Perform galaxy photometry

    plt.clf()
    s = sectors_photometry(img, f.eps, f.theta, f.xpeak, f.ypeak,
                           minlevel=minlevel, plot=True)
    plt.pause(0.01)  # Allow plot to appear on the screen

    # Do the actual MGE fit

    m = mge_fit_sectors(s.radius, s.angle, s.counts, f.eps,
                        ngauss=ngauss, sigmaPSF=sigmaPSF, normPSF=normPSF,
                        scale=scale, plot=False, bulge_disk=False)

    # Show contour plots of the results

    plt.subplot(121)
    mge_print_contours(img, f.theta, f.xpeak, f.ypeak, m.sol, scale=scale,
                       binning=7, sigmapsf=sigmaPSF, normpsf=normPSF, magrange=9)

    # Extract the central part of the image to plot at high resolution.
    # The MGE is centered to fractional pixel accuracy to ease visual comparson.

    n = 50
    img = img[f.xpeak-n:f.xpeak+n, f.ypeak-n:f.ypeak+n]
    xc, yc = n - f.xpeak + f.xmed, n - f.ypeak + f.ymed
    plt.subplot(122)
    mge_print_contours(img, f.theta, xc, yc, m.sol,
                       sigmapsf=sigmaPSF, normpsf=normPSF, scale=scale)

#----------------------------------------------------------------------------

if __name__ == '__main__':
    mge_fit_example()
