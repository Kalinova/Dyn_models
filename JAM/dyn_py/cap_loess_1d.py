###############################################################################
#
# Copyright (C) 2010-2015, Michele Cappellari
# E-mail: cappellari_at_astro.ox.ac.uk
#
# Updated versions of the software are available from my web page
# http://purl.org/cappellari/software
#
# If you have found this software useful for your research,
# I would appreciate an acknowledgment to the use of the
# "CAP_LOESS_1D routine of Cappellari et al. (2013b), which implements
# the univariate LOESS algorithm of Cleveland & Devlin (1988)"
#
# This software is provided as is without any warranty whatsoever.
# Permission to use, for non-commercial purposes is granted.
# Permission to modify for personal or internal use is granted,
# provided this copyright and disclaimer are included unchanged
# at the beginning of the file. All other rights are reserved.
#
###############################################################################
#+
# NAME:
#       CAP_LOESS_1D
# PURPOSE:
#       Local regression LOESS smoothing http://en.wikipedia.org/wiki/Local_regression
#       Univariate: Cleveland (1979) http://www.jstor.org/stable/2286407
#
# CALLING EXAMPLE:
#       xout, yout, wout = loess_1d(x, y, frac=0.5, degree=1, rotate=False)
#
# INPUT PARAMETERS:
#   X, Y: vectors of equal number of elements containing the coordinate X
#       and the corresponding function values Y to be smoothed.
#   XOUT: vector of X coordinates for the YOUT values (same size as Y).
#       If rotate=False (default) then XOUT = X.
#   YOUT: vector of smoothed Y values (same size as Y).
#   WOUT: vector of biweights used in the fit (same size as Y).
#
# KEYWORDS:
#   DEGREE: degree of the local approximation (typically 1 or 2)
#   FRAC: Fraction of points to consider in the local approximation.
#       Typical values are between ~0.2-0.8. Note that the values are
#       weighted by their distance from the point under consideration.
#       This implies that the effective fraction of points contributing
#       to a given value is much smaller that FRAC.
#   NPOINTS: Number of points to consider in the local approximation.
#       This is an alternative to giving FRAC=NPOINTS/n_elements(x).
#   ROTATE: Rotate the (X,Y) coordinates to make the X axis the axis
#       of maximum variance. This is useful to give comparable contribution
#       to the X and Y variables. It is mostly useful for testing.
#   SIGY: 1-sigma errors for the Y values. If this keyword is used
#       the biweight fit is done assuming those errors. If this keyword
#       is *not* used, the biweight fit determines the errors in Y
#       from the scatter of the neighbouring points.
#   WOUT: Output weights used in the fit. This can be used to
#       identify outliers: wout=0 for outliers deviations >4sigma.
#
# MODIFICATION HISTORY:
#   V1.0.0: Michele Cappellari Oxford, 15 December 2010
#   V1.1.0: Rescale after rotating to axis of maximum variance.
#       MC, Vicenza, 30 December 2010
#   V1.1.1: Fix: use ABS() for proper computation of "r".
#       MC, Oxford, 07 March 2011
#   V1.1.2: Return values unchanged if FRAC=0. MC, Oxford, 25 July 2011
#   V1.1.3: Check when outliers don't change to stop iteration.
#       MC, Oxford, 2 December 2011
#   V1.1.4: Updated documentation. MC, Oxford, 16 May 2013
#   V1.3.2: Test whether input (X,Y,Z) have the same size.
#       Included NPOINTS keyword. MC, Oxford, 12 October 2013
#   V1.3.3: Use CAP_POLYFIT_2D. Removed /QUARTIC keyword and replaced
#       by DEGREE keyword like CAP_LOESS_1D. MC, Oxford, 31 October 2013
#   V1.3.4: Include SIGZ and WOUT keywords. Updated documentation.
#       MC, Paranal, 7 November 2013
#   V2.0.0: Translated from IDL into Python. MC, Oxford, 26 February 2014
#   V2.0.1: Removed SciPy dependency. MC, Oxford, 10 July 2014
#   V2.0.2: Returns weights also when frac=0 for consistency.
#       MC, Oxford, 3 November 2014
#   V2.0.3: Updated documentation. Minor polishing. MC, Oxford, 8 December 2014
#   V2.0.4: Converted from 2D to 1D. MC, Oxford, 23 February 2015
#   V2.0.5: Updated documentation. MC, Oxford, 26 March 2015
#-
#------------------------------------------------------------------------

import numpy as np


def polyfit_1d(x, y, degree, sigy=None, weights=None):
    """
    Fit a univariate polynomial of given DEGREE to a set of points
    (X, Y), assuming errors SIGY in the Y variable only.

    For example with DEGREE=1 this function fits a line

       y = a + b*x

    while with DEGREE=2 the function fits a parabola

       y = a + b*x + c*x^2
       
    """

    if weights is None:
        if sigy is None:
            sw = 1.
        else:
            sw = 1./sigy
    else:
        sw = np.sqrt(weights)

    npol = degree + 1

    a = x[:, None]**np.arange(npol)
    coeff = np.linalg.lstsq(a*sw[:, None], y*sw)[0]

    return a.dot(coeff)

#----------------------------------------------------------------------------------

def biweight_sigma(y, zero=False):
    """
    Biweight estimate of the scale (standard deviation).
    Implements the approach described in
    "Understanding Robust and Exploratory Data Analysis"
    Hoaglin, Mosteller, Tukey ed., 1983, Chapter 12B, pg. 417

    """
    y = np.ravel(y)
    if zero:
        d = y
    else:
        d = y - np.median(y)

    mad = np.median(np.abs(d))
    u2 = (d / (9.*mad))**2  # c = 9
    good = u2 < 1.
    u1 = 1. - u2[good]
    num = y.size * ((d[good]*u1**2)**2).sum()
    den = (u1*(1. - 5.*u2[good])).sum()
    sigma = np.sqrt(num/(den*(den - 1.)))  # see note in above reference

    return sigma

#----------------------------------------------------------------------------

def rotate_points(x, y, ang):
    """
    Rotates points counter-clockwise by an angle ANG in degrees.
    Michele cappellari, Paranal, 10 November 2013

    """
    theta = np.radians(ang)
    xNew = x*np.cos(theta) - y*np.sin(theta)
    yNew = x*np.sin(theta) + y*np.cos(theta)

    return xNew, yNew

#------------------------------------------------------------------------

def loess_1d(x1, y1, frac=0.5, degree=1, rotate=False,
             npoints=None, sigy=None):
    """
    yout, wout = loess_1d(x, y, frac=0.5, degree=1)
    gives a LOESS smoothed estimate of the quantity y at the x coordinates.

    """

    if frac == 0:
        return y1, np.ones_like(y1)

    n = x1.size
    if n != y1.size:
        raise ValueError('Input vectors (X, Y) must have the same size')

    if npoints is None:
        npoints = np.ceil(n*frac)

    if rotate:

        # Robust calculation of the axis of maximum variance
        #
        nsteps = 180
        angles = np.arange(nsteps)
        sig = np.zeros(nsteps)
        for j, ang in enumerate(angles):
            x2, y2 = rotate_points(x1, y1, ang)
            sig[j] = biweight_sigma(x2)
        k = np.argmax(sig)  # Find index of max value
        x, y = rotate_points(x1, y1, angles[k])

    else:

        x = x1
        y = y1

    yout = np.empty(n)
    wout = np.empty(n)

    for j, xj in enumerate(x):

        dist = np.abs(x - xj)
        w = np.argsort(dist)[:npoints]
        distWeights = (1 - (dist[w]/dist[w[-1]])**3)**3  # tricube function distance weights
        yfit = polyfit_1d(x[w], y[w], degree, weights=distWeights)

        # Robust fit from Sec.2 of Cleveland (1979)
        # Use errors if those are known.
        #
        bad = []
        for p in range(10):  # do at most 10 iterations
        
            if sigy is None:  # Errors are unknown
                aerr = np.abs(yfit - y[w])  # Note ABS()
                mad = np.median(aerr)  # Characteristic scale
                uu = (aerr/(6*mad))**2  # For a Gaussian: sigma=1.4826*MAD
            else:  # Errors are assumed known
                uu = ((yfit - y[w])/(4*sigy[w]))**2  # 4*sig ~ 6*mad
                
            uu = uu.clip(0, 1)
            biWeights = (1 - uu)**2
            totWeights = distWeights*biWeights
            yfit = polyfit_1d(x[w], y[w], degree, weights=totWeights)
            badOld = bad
            bad = np.where(biWeights < 0.34)[0] # 99% confidence outliers
            if np.array_equal(badOld, bad):
                break

        yout[j] = yfit[0]
        wout[j] = biWeights[0]

    if rotate:
        xout, yout = rotate_points(x, yout, -angles[k])
        j = np.argsort(xout)
        xout = xout[j]
        yout = yout[j]
    else:
        xout = x

    return xout, yout, wout

#------------------------------------------------------------------------

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    n = 200
    np.random.seed(123)
    x = np.random.uniform(-1, 1, n)
    x.sort()

    y = np.sin(3*x)
    sigy = 0.4
    yran = np.random.normal(y, sigy)

    nbad = n*0.1 # 10% outliers
    w = np.random.randint(0, n, nbad)  # random indices from 0-n
    yran[w] += np.random.normal(0, 5*sigy, nbad)

    xout, yout, weigts = loess_1d(x, yran, frac=0.3)

    w = weigts < 0.34  # select outliers
    plt.clf()
    plt.plot(x, yran, 'ro', label='Noisy')
    plt.plot(xout, yout, 'b', linewidth=4, label='LOESS')
    plt.plot(x, y, color='limegreen', linewidth=4, label='True')
    plt.plot(x[w], yran[w], '+k', ms=20, label='outliers')
    plt.legend(loc='lower right')
    plt.pause(0.01)
