"""
#####################################################################

Copyright (C) 1999-2014, Michele Cappellari
E-mail: cappellari_at_astro.ox.ac.uk

Updated versions of the software are available from my web page
http://purl.org/cappellari/software

This software is provided as is without any warranty whatsoever.
Permission to use, for non-commercial purposes is granted.
Permission to modify for personal or internal use is granted,
provided this copyright and disclaimer are included unchanged
at the beginning of the file. All other rights are reserved.

#####################################################################

NAME:
      FIND_GALAXY

AUTHOR:
      Michele Cappellari, Astrophysics Sub-department, University of Oxford, UK

PURPOSE:
      Find the largest region of connected pixels (after smoothing)
      lying above a given intensity level of the image.
      This is useful to automatically identify the location and orientation of
      a galaxy in an image, assuming it is the largest positive fluctuation.
      The convention used by this routine are the same as for the rest
      of the MGE_FIT_SECTORS package.

EXPLANATION:
      This procedure uses the weighted first and second moments of the intensity
      distribution for the computation of the galaxy center and position angle.
      Further information on FIND_GALAXY is available in
      Cappellari M., 2002, MNRAS, 333, 400

CALLING SEQUENCE:
      FIND_GALAXY, Img, Majoraxis, Eps, Ang, Xpeak, Ypeak, Xmed, Ymed, $
          FRACTION=fraction, INDEX=ind, LEVEL=level, NBLOB=nblob, $
          /PLOT, /QUIET, X=x, Y=y

INPUTS:
      Img = The galaxy images as a 2D array.

OUTPUTS:
      Eps = The galaxy "average" ellipticity Eps = 1-b/a = 1-q'.
          Photometry will be measured along elliptical annuli with
          constant axial ellipticity Eps. The four quantities
          (Eps,Ang,Xc,Yc) can be measured with the routine FIND_GALAXY.
      Ang = Position angle measured from the image Y axis,
          counter-clockwise to the galaxy major axis.
          All angles are measured with respect to this direction
      Xpeak = X coordinate of the galaxy center in pixels. To be precise this
          coordinate represents the coordinate of the brightest pixels within
          a 40x40 pixels box centered on the galaxy weighted average center.
      Ypeak = Y coordinate of the galaxy center in pixels.
      Xmed = X coordinate of luminosity weighted galaxy center.
      Ymed = Y coordinate of luminosity weighted galaxy center.

OPTIONAL INPUT KEYWORDS:
      FRACTION - This corresponds (approximately) to the fraction
          [0 < FRACTION < 1] of the image pixels that one wants to
          consider to estimate galaxy parameters (default 0.1 = 10%)
      LEVEL - Level above which to select pixels to consider in the
          estimate of the galaxy parameters. This is an alternative
          to the use of the FRACTION keyword.
      PLOT - display an image in the current graphic window showing
          the pixels used in the computation of the moments.
      NBLOB - If NBLOB=1 (default) the procedure selects the largest feature
          in the image, if NBLOB=2 the second largest is selected, and so
          on for increasing values of NBLOB. This is useful when the
          galaxy is not the largest feature in the image.
      QUIET - do not print numerical values on the screen.

OPTIONAL OUTPUT KEYWORDS:
      INDEX - img[index] contain the pixels of the galaxy from which the
          inertia ellipsoid and the ellipticity and PA are derived.
      X - The x-coordinate of the pixels with index given above.
      Y - The y-coordinate of the pixels with index given above.

EXAMPLE:
      This command locates the position and orientation of a galaxy
      in the image IMG and produces a plot showing the obtained results
      and the region of the image used in the computation:

          f = find_galaxy(img, plot=True)

      This command only uses the 2% of the image pixels to estimate the
      intensity weighted moments and show the results:

          f = find_galaxy(img, fraction=0.02, plot=True)

MODIFICATION HISTORY:
      V1.0.0: Written by Michele Cappellari, Padova, 30 Giugno 1999
      V1.0.1: Michele Cappellari, ESO Garching, 27 september 1999
      V1.1.0: Made a more robust automatic level selection, MC, Leiden, August 2001
      V1.1.1: Added compilation options, MC, Leiden 20 May 2002
      V1.1.2: Load proper color table for plot. MC, Leiden, 9 May 2003
      V1.2.0: Do not use a widget to display the image. Just resample the
          image if needed. Added /QUIET keyword and (xmed,ymed) output.
          After suggestion by R. McDermid. MC, Leiden, 29 July 2003
      V1.2.1: Make the procedure work also with very small images,
          where it does not make sense to extract a subimage.
          MC, Leiden, 1 August 2003
      V1.2.2: Added NBLOB keyword. Useful when the galaxy is not the
          largest feature in the image. MC, Leiden, 9 October 2004
      V1.2.3: Gives an error message if IMG is not a two-dimensional array.
          MC, Leiden, 11 May 2006
      V1.2.4: Included optional output keywords INDEX, X and Y.
          MC, Munich, 14 December 2006
      V1.2.5: Included LEVELS input keyword. MC, Oxford, 3 June 2009
      V1.2.6: Minor changes to the plotting. MC, Oxford, 14 October 2009
      V1.2.7: Perform computations in DOUBLE. MC, Oxford, 25 April 2010
      V1.2.8: Added /DEVICE keyword to TVELLIPSE call, due to a change in
          that astrolib routine. MC, Oxford, 9 January 2013
      V2.0.0: Translated from IDL into Python. MC, Aspen Airport, 8 February 2014
      V2.0.1: Fixed bug in determining largest blob and plotting the ellipse.
          Thanks to Davor Krajnovic for reporting the problems with examples.
          MC, Oxford, 22 February 2014
      V2.0.2: Support both Python 2.6/2.7 and Python 3.x. MC, Oxford, 25 May 2014
      V2.0.3: Use unravel_index. Do not interpolate image for plotting.
          Use imshow(...origin='lower'). MC, Oxford, 21 September 2014

"""

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy import signal, ndimage, stats

#--------------------------------------------------------------------

class find_galaxy(object):

    def __init__(self, img, fraction=0.1, plot=False, quiet=False,
                 nblob=1, level=None):
        """
        With nblob=1 find the ellipse of inertia of the largest
        connected region in the image, with nblob=2 find the second
        in size and so on...

        """
        if len(img.shape) != 2:
            raise ValueError('IMG must be a two-dimensional array')

        a = signal.medfilt(img, 5)

        if level is None:
            level = stats.scoreatpercentile(a, (1 - fraction)*100)

        mask = a > level
        labels, nb = ndimage.label(mask)   # Get blob indices
        sizes = ndimage.sum(mask, labels, np.arange(nb + 1))
        j = np.argsort(sizes)[-nblob]      # find the nblob-th largest blob
        ind = np.flatnonzero(labels == j)

        self.second_moments(img, ind)

        if not quiet:
            print(' Pixels used:', ind.size)
            print(' Peak (x,y):', self.xpeak, self.ypeak)
            print(' Mean (x,y): %.2f %.2f' % (self.xmed, self.ymed))
            print(' Theta (deg): %.1f' % self.theta)
            print(' Eps: %.3f' % self.eps)
            print(' Sigma along major axis (pix): %.1f' % self.majoraxis)

        if plot:
            ax = plt.gca()
            ax.imshow(np.log(img.clip(img[self.xpeak, self.ypeak]/1e4)),
                      cmap='hot', origin='lower', interpolation='none')
            ax.imshow(mask, cmap='binary', interpolation='none',
                      origin='lower', alpha=0.3)
            ax.autoscale(False) # prevents further scaling after imshow()
            mjr = 3.5*self.majoraxis
            yc, xc = self.xmed, self.ymed
            ellipse = patches.Ellipse(xy=(xc, yc), width=2*mjr, fill=False,
                                      height=2*mjr*(1-self.eps), angle=-self.theta,
                                      color='red', linewidth=3)
            ax.add_artist(ellipse)
            ang = np.array([0,np.pi]) - np.radians(self.theta)
            ax.plot(xc - mjr*np.sin(ang), yc + mjr*np.cos(ang), 'g--',
                    xc + mjr*np.cos(ang), yc + mjr*np.sin(ang), 'g-', linewidth=3)
            ax.set_xlabel("pixels")
            ax.set_ylabel("pixels")

#-------------------------------------------------------------------------

    def second_moments(self, img, ind):
        #
        # Restrict the computation of the first and second moments to
        # the region containing the galaxy, defined by vector IND.

        img1 = img.flat[ind]
        s = img.shape
        x, y = np.unravel_index(ind, s)

        # Compute coefficients of the moment of inertia tensor.
        #
        i = np.sum(img1)
        self.xmed = np.sum(img1*x)/i
        self.ymed = np.sum(img1*y)/i
        x2 = np.sum(img1*x**2)/i - self.xmed**2
        y2 = np.sum(img1*y**2)/i - self.ymed**2
        xy = np.sum(img1*x*y)/i - self.xmed*self.ymed

        # Diagonalize the moment of inertia tensor.
        # theta is the angle, measured counter-clockwise,
        # from the image Y axis to the galaxy major axis.
        #
        self.theta = np.degrees(np.arctan2(2*xy, x2 - y2)/2.) + 90.
        a2 = (x2 + y2)/2. + np.sqrt(((x2 - y2)/2.)**2 + xy**2)
        b2 = (x2 + y2)/2. - np.sqrt(((x2 - y2)/2.)**2 + xy**2)
        self.eps = 1. - np.sqrt(b2/a2)
        self.majoraxis = np.sqrt(a2)

        # If the image has many pixels then compute the coordinates of the
        # highest pixel value inside a 40x40 pixels region centered on the
        # first intensity moments (Xmed,Ymed), otherwise just return the
        # coordinates of the highest pixel value in the whole image.
        #
        n = 20
        xmed1 = int(round(self.xmed))
        ymed1 = int(round(self.ymed))   # Check if subimage fits...
        if n <= xmed1 <= s[0]-n and n <= ymed1 <= s[1]-n:
            img2 = img[xmed1-n:xmed1+n, ymed1-n:ymed1+n]
            ij = np.unravel_index(np.argmax(img2), img2.shape)
            self.xpeak, self.ypeak = ij + np.array([xmed1, ymed1]) - n
        else:            # ...otherwise use full image
            self.xpeak, self.ypeak = np.unravel_index(np.argmax(img), s)

#-------------------------------------------------------------------------
