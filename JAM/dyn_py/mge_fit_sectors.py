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
    MGE_FIT_SECTORS

AUTHOR:
      Michele Cappellari, Astrophysics Sub-department, University of Oxford, UK

PURPOSE:
      Fit a Multi-Gaussian Expansion (MGE) model to a set of galaxy surface
      brightness measurements. The measurements are usually taken along
      sectors with a previous call to the routine SECTORS_PHOTOMETRY.
      The MGE model is intended to be used as a parametrization for
      the galaxy surface brightness. All measurements within this program
      are in the instrumental units of PIXELS and COUNTS.
      This routine fits MGE models with constant position angle and common center.

EXPLANATION:
      Further information on MGE_FIT_SECTORS is available in
      Cappellari M., 2002, MNRAS, 333, 400

CALLING SEQUENCE:
      MGE_FIT_SECTORS, Radius, Angle, Counts, Eps,
          NGAUSS=ngauss, SIGMAPSF=[sigma1,sigma2,...], NORMPSF=[norm1,norm2,...],
          SCALE=scale, RBOUNDS=[rmin,rmax], QBOUNDS=[qmin,qmax],
          /PRINT, /LINEAR, /NEGATIVE, /BULGE_DISK, SOL=sol,
          OUTER_SLOPE=outer_slope, ABSDEV=absdev, /QUIET

INPUTS:
      Radius = Vector containing the radius of the surface brightness
              measurements, taken from the galaxy center. This is given
              in units of PIXELS (!) of the high resolution image.
      Angle = Vector containing the polar angle of the surface brightness
              measurements, taken from the galaxy major axis.
      Counts = Vector containing the actual surface brightness measurements
              in COUNTS (!) at the polar coordinates specified by the vectors
              Radius and Angle. These three vectors need to have the same
              number of elements.
      Eps = Estimate for the galaxy `average' ellipticity Eps = 1-b/a = 1-q'

OUTPUTS:
      No output parameters. The results are printed on the screen, plotted
      in a PS file with the /PRINT keyword and passed with the optional
      keyword SOL

OPTIONAL INPUT KEYWORDS:
      /BULGE_DISK - Set this keyword to perform a non-parametric bulge/diks
              decomposition using MGE. When this keyword is set, the Gaussians
              are divided into two sets, each with a unique axial ratio. The two
              sets are meant to describe and model the contribution of bulge and
              disks in spiral or lenticular galxies, or nuclear disk in ellipticals.
            - When this keyword is set one may have to increase NGAUSS.
            - When this keyword is set it is advisable to either remove QBOUNDS
              or to specify four elements in QBOUNDS, for the even/odd Gaussians.
      /FASTNORM - Set this keyword to activate a faster but less stable
              computation of the Chi**2 (rarely needed...)
      /LINEAR - Set this keyword to start with the fully linear algorithm
              and then optimize the fit with the nonlinear method
              (see Cappellari [2002, Section 3.4] for details). Much slower that
              the standard method, and not often used in practice, but may
              be very useful in critical situations.
      /NEGATIVE - Set this keyword to allow for negative Gaussians in the fit.
              Use this only if everything else didn't work or if there is clear
              evidence that negative Gaussians are actually needed.
              Negative Gaussians are needed e.g. when fitting a boxy bulge.
      NGAUSS - Number of Gaussians desired in the MGE fit.
              Typical values are in the range 10-20 when the /LINEAR
              keyword is NOT set (default: 12) and 20**2-40**2 when the
              /LINEAR keyword is set (default: 30**2).
      NORMPSF - This is optional if only a scalar is given for SIGMAPSF,
              otherwise it must contain the normalization of each MGE component
              of the PSF, whose sigma is given by SIGMAPSF. The vector needs to
              have the same number of elements of SIGMAPSF and the condition
              TOTAL(normpsf) = 1 must be verified. In other words the MGE PSF
              needs to be normalized. (default: 1).
      OUTER_SLOPE - This scalar forces the surface brightness profile of
              the MGE model to decrease at least as fast as R**(-OUTER_SLOPE)
              at the largest measured radius (Default: OUTER_SLOPE=2).
      /PRINT - Set this keyword to print the best-fitting MGE profiles in IDL.PS
      QBOUNDS - This can be either a two or a four elements vector.
            - If QBOUNDS has two elements, it gives the minimum and maximum
              axial ratio Q allowed in the MGE fit.
            - If QBOUNDS has four elements [[qMin1,qMax1],[qMin2,qMax2]], then
              the first two elements give the limits on Q for the even Gaussians,
              while the last two elements give the limits on Q for the odd Gaussians.
              In this way QBOUNDS can be used to perform disk/bulges
              decompositions in a way similar to the /BULGE_DISK keyword.
      /QUIET: Set this keyword to suppress printed output.
      RBOUNDS - Two elements vector giving the minimum and maximum sigma
              allowed in the MGE fit. This is in PIXELS units.
      SCALE - the pixel scale in arcsec/pixels. This is only needed
              for the scale on the plots (default: 1)
      SIGMAPSF - Scalar giving the sigma of the PSF of the high resolution
              image (see Cappellari [2002, pg. 406] for details), or vector
              with the sigma of an MGE model for the circular PSF.
              This has to be in pixels, as the vector RADIUS above.
              (Default: no convolution)
        SOL - If this keyword has at least 6 elements in input, the Sigma
              and qObs will be used as starting point for the optimization.
              This is useful for testing but is never needed otherwise.
              The format is described in the OUTPUT KEYWORDS below.

OPTIONAL OUTPUT KEYWORDS:
        SOL - Output keyword containing a 3xNgauss array with the
              best-fitting solution:
              1) sol[0,*] = TotalCounts, of the Gaussians components.
                  The relation TotalCounts = Height*(2*!PI*Sigma**2*qObs)
                  can be used compute the Gaussian central surface
                  brightness (Height)
              2) sol[1,*] = Sigma, is the dispersion of the best-fitting
                  Gaussians in pixels.
              3) sol[2,*] = qObs, is the observed axial ratio of the
                  best-fitting Gaussian components.
      ABSDEV - Mean absolute deviation between the fitted MGE and the
              data expressed as a fraction. Good fits to high S/N images
              can reach values of ABSDEV < 0.02 = 2%.

EXAMPLE:
      The sequence of commands below was used to generate the complete
      MGE model of the Figures 8-9 in Cappellari (2002).
      1) The FITS file is read and sky is subtracted#
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
              MINLEVEL=0.2

          MGE_fit_sectors, radius, angle, counts, eps, $
              NGAUSS=13, SIGMAPSF=sigmaPSF, SOL=sol, /PRINT, SCALE=scale

          MGE_print_contours, img, ang, xc, yc, sol, BINNING=3, $
              FILE='ngc4342.ps', SCALE=scale, MAGRANGE=9, SIGMAPSF=sigmaPSF

PROCEDURES USED:
      The following procedures are contained in the main MGE_FIT_SECTORS program.
          MGE_FIT_SECTORS_PRINT  -- called during the fit to show the fit profiles
          FITFUNC_MGE_SECTORS    -- returns the residuals between data and model
          MGE_FIT_SECTORS_LINEAR -- only called if the /LINEAR keyword is set

      Other IDL routines needed:
          BVLS  -- Michele Cappellari porting of Lawson & Hanson generalized NNLS
                   http://purl.org/cappellari/idl
          CAP_MPFIT -- Craig Markwardt porting of Levenberg-Marquardt MINPACK-1

MODIFICATION HISTORY:
      V1.0.0: First implementation, Padova, February 1999, Michele Cappellari
      V2.0.0: Major revisions, Leiden, January 2000, MC
      V3.0.0: Significant changes, Padova, July 2000, MC
      V3.1.0: More robust definition of err in FITFUNC_MGE_SECTORS,
          Leiden, 27 April 2001, MC
      V3.2.0: Graphical changes: always show about 7 sectors on the screen,
          and print plots with shared axes. Leiden, 8 July 2001, MC
      V3.3.0: Added MGE PSF convolution, central pixel integration and changed
          program input parameters to make it independent from SECTORS_PHOTOMETRY
          Leiden, 26 July 2001, MC
      V3.4.0: Added /FASTNORM keyword, Leiden, 20 September 2001, MC
      V3.5.0: Updated documentation, Leiden, 8 October 2001, MC
      V3.6.0: Modified implementation of /NEGATIVE keyword.
          Leiden, 23 October 2001, MC
      V3.7.0: Added explicit stepsize (STEP) of numerical derivative in
          parinfo structure, after suggestion by Craig B. Markwardt.
          Leiden, 23 February 2002, MC
      V3.7.1: Added compilation options, Leiden 20 May 2002, MC
      V3.7.2: Added ERRMSG keyword to MPFIT call. Leiden, 13 October 2002, MC
      V3.7.3: Force the input parameters to the given bounds if they
          fall outside the required range before starting the fit.
          After feedback from Remco van den Bosch.
          Leiden, 7 March 2003, MC
      V3.7.4: Use N_ELEMENTS instead of KEYWORD_SET to test
          non-logical keywords. Leiden, 9 May 2003, MC
      V3.7.5: Corrected small bug introduced in V3.73.
          Thanks to Arend Sluis. Leiden 23 July 2003, MC.
      V3.7.6: Use updated calling sequence for BVLS. Leiden, 20 March 2004, MC
      V3.8.0: Force the surface brightness of the MGE model to decrease at
          least as R**-2 at the largest measured radius. Leiden, 8 May 2004, MC
      V3.8.1: Make sure this routine uses the Nov/2000 version of Craig Markwardt
          MPFIT which was renamed MGE_MPFIT to prevent potential conflicts with
          more recent versions of the same routine. Vicenza, 23 August 2004, MC.
      V3.9.0: Allow forcing the outer slope of the surface brightness profile of
          the MGE model to decrease at least as R**-n at the largest measured
          radius (cfr. version 3.8).
          Clean the solution at the end of the nonlinear fit as already done in
          the /LINEAR implementation. It's almost always redundant, but quick.
          Leiden, 23 October 2004, MC
      V3.9.1 Replaced LOGRANGE keyword in example with the new MAGRANGE.
          MC, Leiden, 1 May 2005
      V3.9.2 Print iterations of the longer part at the end, not of the
          short `Gaussian cleaning' part. MC, Leiden, 11 October 2005
      V3.9.3: Changed axes labels in plots. Leiden, 18 October 2005, MC
      V3.9.4: Use more robust la_least_squares (IDL 5.6) instead of SVDC with
          /NEGATIVE keyword. MC, Oxford, 16 May 2008
      V3.9.5: Force Gaussians smaller than the PSF, which have a degenerate
          axial ratio, to have the same axial ratio as the mean of the first
          two well determined Gaussians. MC, Oxford, 24 September 2008
      V4.0.0: Added /BULGE_DISK keyword to perform non-parametric bulge/disks
          decompositions using MGE. Updated MPFIT to version v1.52 2008/05/04,
          to fix a bug with the required parinfo.tied mechanism. In the new
          version of MPFIT, which I again renamed MGE_MPFIT, I implemented
          my previous important modification to improve convergence with
          MGE_FIT_SECTORS. MC, Windhoek, 5 October 2008
      V4.0.1: Added output keyword ABSDEV. Fixed display not being updated
          while iterating under Windows. MC, Oxford, 6 June 2009
      V4.1.0: Allow QBOUNDS to have four elements, to perform bulge/disk
          decompositions similarly to the /BULGE_DISK option.
          MC, Oxford, 22 April 2010
      V4.1.1: Added keyword /QUIET. MC, Oxford, 12 November 2010
      V4.1.2: Small change to the treatment of the innermost unresolved
          Gaussians. MC, Oxford, 24 April 2012
      V4.1.3: Explained optional usage of SOL in input.
          Removed stop when MPFIT reports over/underflow.
          MC, Oxford, 23 January 2013
      V5.0.0: Translated from IDL into Python. MC, Aspen Airport, 8 February 2014
      V5.0.1: Support both Python 2.6/2.7 and Python 3.x. MC, Oxford, 25 May 2014
      V5.0.2: Improved plotting. MC, Oxford, 24 September 2014

"""

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize, linalg, special
from time import clock
from cap_mpfit import mpfit

#----------------------------------------------------------------------------

def _linspace_open(a, b, n):
        x, dx = np.linspace(a, b, n, endpoint=False, retstep=True)
        return x + dx/2.

#----------------------------------------------------------------------------

class mge_fit_sectors(object):

    def __init__(self, radius, angle, counts, eps,
                 ngauss=None, negative=False, sigmaPSF=0., normPSF=1.,
                 scale=1., rbounds=None, qbounds=None, linear=False,
                 quiet=False, outer_slope=4, bulge_disk=False, sol=0, plot=False):
        """
        This is the main routine. It is the only one that has to be called from outside.

        """

        # Starting from here the biggest part of this routine is devoted to input
        # checking and loading of shared parameters into a common block
        #
        if np.any(counts <= 0):
            raise ValueError('Error: negative input counts')
        n1 = len(radius)
        if ((len(angle) != n1) or (len(counts) != n1)):
            raise ValueError('Error: Input vectors must have the same length')
        outer_slope = np.clip(outer_slope, 1, 4)

        # load data vectors into the COMMON block
        #
        self.radius = radius
        self.counts = counts
        self.angle = angle
        self.scale = scale
        sol = np.asarray(sol)
        self.sigmaPSF = np.atleast_1d(sigmaPSF)
        self.normPSF = np.atleast_1d(normPSF)
        self.negative = negative
        self.scale = scale
        self.quiet = quiet
        self.plot = plot

        nPsf = self.sigmaPSF.size
        if nPsf > 1:
            if self.normPSF.size != nPsf:
                raise ValueError('sigmaPSF and normPSF must have the same length')
            if round(np.sum(normPSF)*100.) != 100:
                raise ValueError('Error: PSF not normalized')

        self.sectors = np.unique(angle) # Finds the different position angles

        # Open grid in the range [rminLog,rmaxLog]
        # The logarithmic slope of a Gaussian is slope(R) = -(R/sigma)**2
        # so to have a slope -n one needs to be at R = sqrt(n)*sigma.
        # Below we constrain the largest Gaussian to have sigma < rmax/sqrt(n)
        # to force the surface brightness of the MGE model to decrease
        # at least as fast as R**-n at the largest measured radius.
        #
        if rbounds is None:
            rmin = np.min(radius)
            rmax = np.max(radius)
            rminLog = np.log10(rmin)
            rmaxLog = np.log10(rmax/np.sqrt(outer_slope))
        else:
            rminLog = np.log10(rbounds[0])
            rmaxLog = np.log10(rbounds[1])

        if qbounds is None:
            qbounds = np.array([0.05, 1.]) # no gaussians flatter than q=0.05

        # If the smallest intrinsic Gaussian has sigma=0.75*sigmaPSF it will produce an
        # observed Gaussian with sigmaObs=SQRT(sigma**2+sigmaPSF**2)=1.25*sigmaPSF.
        # We require the sigma of the Gaussians to be larger than 0.75*sigmaPSF,
        # or the smallest measured radius, whichever is larger.
        #
        if np.sum(sigmaPSF) > 0:
            rminLog = max(rminLog, np.log10(0.75*np.min(sigmaPSF)))

        # Here the actual calculation starts. The path is different depending on whether the
        # user has requested the nonlinear method or the linear one by setting the /LINEAR keyword
        #
        t = clock()
        if linear:
            if bulge_disk:
                raise ValueError('BULGE_DISK not supported with LINEAR keyword')
            if ngauss is None:
                ngauss = 100**2
            elif ngauss < 10**2:
                raise ValueError('Too few Gaussians for the LINEAR method')
            if not self.quiet:
                print('Starting the LINEAR fit with ',  ngauss, ' Gaussians. Please wait...')
            bestNorm = 1e30
            nIter = 0
        else:
            if ngauss is None:
                ngauss = 15
            elif ngauss > 35:
                raise ValueError('Too many Gaussians for the non-linear method')
            if sol.size < 6:
                logsigma = _linspace_open(rminLog, rmaxLog, ngauss) # open grid
                pars = np.append(logsigma, np.zeros(ngauss) + np.clip(1. - eps, qbounds[0], qbounds[1]))
            else:
                ngauss = sol.size//3
                pars = np.zeros(ngauss*2)
                pars[:ngauss] = np.log10(sol[1, :]).clip(rminLog, rmaxLog) # Log(sigma)
                pars[ngauss:] = sol[2, :].clip(qbounds[0], qbounds[1])     # qObs

            parinfo = [{'step':0.01, 'limits':[rminLog, rmaxLog], 'limited':[1, 1], 'tied':''}
                        for j in range(2*ngauss)]
            if qbounds.size == 4: # Allow for different constraints for bulge and disk
                pars[ngauss::2] = pars[ngauss::2].clip(qbounds[0], qbounds[1])     # qObs
                pars[ngauss+1::2] = pars[ngauss+1::2].clip(qbounds[2], qbounds[3])     # qObs
                for j in range(ngauss, ngauss*2, 2):
                    parinfo[j]['limits'] = qbounds[:2]
                    parinfo[j+1]['limits'] = qbounds[2:]
            else:
                for j in range(ngauss, ngauss*2):
                    parinfo[j]['limits'] = qbounds

            if bulge_disk:
                for j in range(0, ngauss-3, 2):
                    parinfo[ngauss+j+2]['tied'] = 'p['+str(ngauss+0)+']' # Ties axial ratio of even Gaussians
                    parinfo[ngauss+j+3]['tied'] = 'p['+str(ngauss+1)+']' # Ties axial ratio of odd Gaussians

            mp = mpfit(self._fitfunc, pars, iterfunct=self._print,
                       nprint=10, parinfo=parinfo, quiet=quiet)
            sol = mp.params
            bestNorm = mp.fnorm
            nIter = mp.niter # Use iterations of the longer part

        if (not bulge_disk) and (qbounds.size == 2):
            sol, bestNorm = self._linear_fit(
                sol, bestNorm, rminLog, rmaxLog, qbounds, ngauss)

        # Print the results for the nonzero Gaussians
        #
        ngauss = sol.size//2
        logSigma = sol[:ngauss]
        q = sol[ngauss:]

        w = np.nonzero(self.soluz)[0]
        m = w.size
        logSigma = logSigma[w]
        q = q[w]
        self.soluz = self.soluz[w]
        j = np.argsort(logSigma) # Sort by increasing sigma

        self.sol = np.empty((3, m))
        self.sol[0, :] = self.soluz[j]
        self.sol[1, :] = 10.**logSigma[j]
        self.sol[2, :] = q[j]

        # Force Gaussians with minor axis smaller than the PSF, which have
        # a degenerate axial ratio, to have the same axial ratio as the mean
        # of the first two well determined Gaussians. Flux is conserved by
        # PSF convolution so no other changes are required
        #
        if not bulge_disk:
            sig = self.sol[1, :]*self.sol[2, :]
            w1 = np.where(sig <= np.min(sigmaPSF))[0]
            w2 = np.where(sig > np.min(sigmaPSF))[0]
            if w1.size > 0:
                self.sol[2, w1] = np.mean(self.sol[2, w2[:2]])

        if not self.quiet:
            print('############################################')
            print(' Computation time: %.2f seconds' % (clock() - t))
            print('  Total Iterations: ', nIter)
            print(' Nonzero Gaussians: ', m)
            print('  Unused Gaussians: ', ngauss - m)
            print(' Sectors used in the fit: ', self.sectors.size)
            print(' Total number of points fitted: ', n1)
            print(' Chi2: %.4g ' % bestNorm)
            print(' STDEV: %.4g ' % np.std(self.err))
            print(' MEANABSDEV: %.4g ' % np.mean(np.abs(self.err)))
            print('############################################')
            print(' Total_Counts   Sigma_Pixels    qObs')
            print('############################################')
            print(self.sol.T)
            print('++++++++++++++++++++++++++++++++++++++++++++')

#----------------------------------------------------------------------------

    def _fitfunc(self, pars, fjac=None):

        ngauss = pars.size//2
        npoints = len(self.radius)
        logsigma = pars[:ngauss]
        q = pars[ngauss:]
        self.gauss = np.zeros((npoints, ngauss))
        sigma2 = 10.**(2.*logsigma)

        w = np.where(self.radius < 0.5)[0] # Will perform flux integration inside the central pixel
        ang = np.radians(self.angle)
        cosa = np.cos(ang)
        sina = np.sin(ang)
        r2 = self.radius**2

        for j in range(ngauss): # loop over the galaxy MGE Gaussians
            for k in range(len(self.sigmaPSF)): # loop over the PSF Gaussians

                # Analytic convolution with an MGE circular PSF
                # Equations (4,5) in Cappellari (2002)
                #
                sigmaX = np.sqrt(sigma2[j] + self.sigmaPSF[k]**2)
                sigmaY = np.sqrt(sigma2[j]*q[j]**2 + self.sigmaPSF[k]**2)

                # Evaluate the normalized (volume=1) 2d Gaussian in polar coordinates
                #
                g = np.exp(-0.5*r2 * ((cosa/sigmaX)**2 + (sina/sigmaY)**2)) \
                        / (2*np.pi*sigmaX*sigmaY)

                # Analytic integral of the Gaussian on the central pixel.
                # Below we assume the central pixel is aligned with the galaxy axes.
                # This is generally not the case, but the error due to this
                # approximation is negligible in realistic situations.
                #
                if len(w) > 0:
                    g[w] = special.erf(2**-1.5/sigmaX) * special.erf(2**-1.5/sigmaY)

                # Each convolved MGE Gaussian is weighted with the
                # normalization of the corresponding PSF component
                #
                self.gauss[:, j] += self.normPSF[k] * g

        A = self.gauss/self.counts[:, None] # gauss*SQRT(weights) = gauss/y
        b = np.ones(npoints)  # y*SQRT(weights) = 1 <== weights = 1/sigma**2 = 1/y**2

        if self.negative:   # Solution by LAPACK linear least-squares
            self.soluz = linalg.lstsq(A, b)[0]
        else:               # Solution by NNLS
            self.soluz = optimize.nnls(A, b)[0]

        self.yfit = self.gauss.dot(self.soluz)   # Evaluate predictions by matrix multiplications
        self.err = 1. - self.yfit / self.counts  # relative error: yfit, counts are positive quantities
        self.chi2 = np.sum(self.err**2) # rnorm**2 = TOTAL(err**2) (this value is only used with the /LINEAR keyword)

        return [0, self.err.astype(np.float32)] # err is a vector. Important: Solve BVLS in DOUBLE but MPFIT in FLOAT

#----------------------------------------------------------------------------

    def _linear_fit(self, sol, chi2Best, rminLog, rmaxLog, qBounds, ngauss):
        """
        This implements the algorithm described in Sec.3.4 of Cappellari (2002)

        """

        if sol.size < 6:
            neps = round(np.sqrt(ngauss)) # Adopt neps=nrad. This may not always be optimal
            nrad = round(np.sqrt(ngauss))
            q = np.linspace(qBounds[0], qBounds[1], neps)
            logSigma = np.linspace(rminLog, rmaxLog, nrad)
            q, logSigma = np.meshgrid(q, logSigma)
            sol = np.append(logSigma, q)
            self.nIter = 0
            self._fitfunc(sol)                  # Get initial chi**2
            chi2Best = self.chi2
            self._print(0, sol, 0, chi2Best)    # Show initial fit

        ########
        # Starting from the best linear solution we iteratively perform the following steps:
        # 1) Eliminate the Gaussians = 0
        # 2) Eliminate all Gaussians whose elimination increase chi2 less than "factor"
        # 3) Perform nonlinear optimization of these Gaussians (chi2 can only decrese)
        # 4) if the number of Gaussians decreased go back to step (1)
        ########

        factor = 1.01  # Maximum accepted factor of increase in chi**2 from the best solution

        while True:
            sol = sol.reshape(2, -1)
            ngauss = sol.shape[1]
            sol = sol[:, self.soluz != 0] # Extract the nonzero Gaussians
            m = sol.shape[1]
            if not self.quiet:
                print('Nonzero Gaussians:', m)
                print('Eliminating not useful Gaussians...')
            while True:
                chi2v = np.zeros(m)
                for k in range(m):
                    tmp = np.delete(sol, k, axis=1)  # Drop element k from the solution
                    tmp = self._fitfunc(tmp.ravel()) # Try the new solution
                    chi2v[k] = self.chi2

                k = np.argmin(chi2v)
                if chi2v[k] > factor*chi2Best:
                    break
                sol = np.delete(sol, k, axis=1)  # Delete element k from the solution
                m -= 1                           # Update the gaussian count
                if not self.quiet:
                    print('ngauss:', m, '          chi2: %.3g' % chi2v[k])

            if m == ngauss:  # all Gaussians are needed
                break
            ngauss = m

            parinfo = [{'step':0.01, 'limits':[rminLog, rmaxLog], 'limited':[1, 1]}
                        for j in range(2*ngauss)]

            for j in range(ngauss, 2*ngauss):
                parinfo[j]['limits'] = qBounds

            if not self.quiet:
                print('Starting nonlinear fit...')

            mp = mpfit(self._fitfunc, sol.ravel(), iterfunct=self._print,
                       nprint=10, parinfo=parinfo, quiet=self.quiet)
            sol = mp.params

            if mp.fnorm < chi2Best:
                chi2Best = mp.fnorm

        # Load proper values in self.soluz and chi2best before returning

        tmp = self._fitfunc(sol.ravel())
        chi2Best = self.chi2

        return [sol.ravel(), chi2Best]

#----------------------------------------------------------------------------

    def _print(self, myfunct, logsigma, it, fnorm,
                functkw=None, parinfo=None, dof=None, quiet=0):
        """
        This is a plotting routine that is called every NPRINT iteration of MPFIT

        """

        if not self.quiet:
            print('Iteration: ', it, '  chi2: %.4g' % fnorm, ' Nonzero:', np.sum(self.soluz != 0))

        if self.plot:
            # Select an x and y plot range that is the same for all plots
            #
            minrad = np.min(self.radius)*self.scale
            maxrad = np.max(self.radius)*self.scale
            mincnt = np.min(self.counts)
            maxcnt = np.max(self.counts)
            xran = minrad * (maxrad/minrad)**np.array([-0.02, +1.02])
            yran = mincnt * (maxcnt/mincnt)**np.array([-0.05, +1.05])

            n = self.sectors.size
            dn = int(round(n/6.))
            nrows = (n-1)//dn + 1 # integer division

            plt.clf()
            fig, ax = plt.subplots(nrows, 2, sharex=True, num=1)
            fig.subplots_adjust(hspace=0)

            row = 0
            for j in range(0, n, dn):
                w = np.where(self.angle == self.sectors[j])[0]
                w = w[np.argsort(self.radius[w])]
                r = self.radius[w]*self.scale
                txt = "$%.f^\circ$" % self.sectors[j]

                ax[row, 0].set_xlim(xran)
                ax[row, 0].set_ylim(yran)
                ax[row, 0].loglog(r, self.counts[w], 'bo')
                ax[row, 0].set_xlabel("arcsec")
                ax[row, 0].set_ylabel("counts")
                ax[row, 0].loglog(r, self.yfit[w], 'r', linewidth=2)
                ax[row, 0].text(maxrad, maxcnt, txt, ha='right', va='top')
                ax[row, 0].loglog(r, self.gauss[w, :]*self.soluz[None, :])

                ax[row, 1].semilogx(r, self.err[w]*100, 'bo')
                ax[row, 1].axhline(linestyle='--', color='r', linewidth=2)
                ax[row, 1].set_xlabel("arcsec")
                ax[row, 1].set_ylabel("error (%)")
                ax[row, 1].yaxis.tick_right()
                ax[row, 1].yaxis.set_ticks_position('both')
                ax[row, 1].yaxis.set_label_position("right")
                ax[row, 1].set_ylim([-19.5, 20])

                row += 1

            plt.pause(0.01)

#----------------------------------------------------------------------------
