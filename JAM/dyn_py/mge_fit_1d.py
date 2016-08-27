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
    mge_fit_1d()

AUTHOR:
      Michele Cappellari, Astrophysics Sub-department, University of Oxford, UK

PURPOSE:
      Perform a Multi Gaussian Expansion fit to a one-dimensional profile.

EXPLANATION:
      Further information on MGE_FIT_1D is available in
      Cappellari M., 2002, MNRAS, 333, 400

CALLING SEQUENCE:
      p = mge_fit_1d(x, y, negative=0, ngauss=12, 
                     rbounds=None, outer_slope=4, quiet=0):

INPUTS:
      X = Vector of the logarithmically sampled (Important !)
          abscissa for the profile that has to be fitted.
      Y = Ordinate of the profile evaluated at the abscissa X.

OUTPUTS:
      No output parameters. The results are printed on the screen
      and passed with the optional keyword SOL

OPTIONAL INPUT KEYWORDS:
      NGAUSS - Number of Gaussian on want to fit. Typical values are 10-20.
      /NEGATIVE - Set this keyword to allow for negative Gaussians in the fit.
              Use this only if there is clear evidence that negative Gaussians 
              are actually needed e.g. to fit a radially increasing profile.
      OUTER_SLOPE - This scalar forces the surface brightness profile of
              the MGE model to decrease at least as fast as R^(-OUTER_SLOPE)
              at the largest measured radius (Default: OUTER_SLOPE=2).
      RBOUNDS - Two elements vector giving the minimum and maximum sigma
              allowed in the MGE fit. This is in the same units of X.
      /QUIET: Set this keyword to suppress plots and printed output.

EXAMPLE:
      The sequence of commands below was used to generate the
      one-dimensional MGE fit of Fig.3 in Cappellari (2002).

          n = 300 # Number of sampled points
          R = range(0.01,300,n,/LOG) # logarithmically sampled radii
          rho = (1 + R)^(-4) # The profile is logarithmically sampled!
          mge_fit_1d, R, rho, NGAUSS=16, SOL=sol
     
       In the common case in which rho represents an intrinsic density
       in Msun/pc^3 and R is in pc, the output keyword sol[0,*] will 
       contain the surface density of the projected Gaussians in Msun/pc^2. 
       This is already the required input format for the JAM modelling 
       routines here http://www-astro.physics.ox.ac.uk/~mxc/idl/

OPTIONAL OUTPUT KEYWORDS:
      SOL - Output keyword containing a 2xNgauss array with the
              best-fitting solution:
              1) sol[0,*] = TotalCounts, of the Gaussians components.
                  The relation TotalCounts = Height*SQRT(2*!PI)*Sigma
                  can be used compute the Gaussian central surface
                  brightness (Height).
                  IMPORTANT: TotalCounts is defined as the integral under a
                  1D Gaussian curve and not the one of a 2D Gaussian surface.
              2) sol[1,*] = sigma, is the dispersion of the best-fitting
                  Gaussians in units of X.
PROCEDURES USED:
      The following procedures are contained in the main MGE_FIT_1D program.
          MGE_FIT_1D_PRINT   -- Show intermediate results while fitting
          FITFUNC_MGE_1D     -- This the function that is optimized during the fit.

      Other IDL routines needed:
          BVLS  -- Michele Cappellari porting of Lawson & Hanson generalized NNLS
                   http://www.strw.leidenuniv.nl/~mcappell/idl/
          CAP_MPFIT -- Craig Markwardt porting of Levenberg-Marquardt MINPACK-1

MODIFICATION HISTORY:
      V1.0.0 Michele Cappellari, Padova, February 2000
      V1.1.0 Minor revisions, MC, Leiden, June 2001
      V1.2.0 Updated documentation, MC, 18 October 2001
      V1.3.0 Added compilation options and explicit stepsize in parinfo
           structure of MPFIT, MC, Leiden 20 May 2002
      V1.3.1 Added ERRMSG keyword to MPFIT call. Leiden, 13 October 2002, MC
      V1.3.2: Use N_ELEMENTS instead of KEYWORD_SET to test
          non-logical keywords. Leiden, 9 May 2003, MC
      V1.3.3: Use updated calling sequence for BVLS. Leiden, 20 March 2004, MC
      V1.4.0: Force the surface brightness of the MGE model to decrease at
          least as R^-2 at the largest measured radius. Leiden, 8 May 2004, MC
      V1.4.1: Make sure this routine uses the Nov/2000 version of Craig Markwardt
          MPFIT which was renamed MGE_MPFIT to prevent potential conflicts with
          more recent versions of the same routine. Vicenza, 23 August 2004, MC.
      V1.4.2: Allow forcing the outer slope of the surface brightness profile of
          the MGE model to decrease at least as R^-n at the largest measured
          radius (cfr. version 1.4). Leiden, 23 October 2004, MC
      V1.4.3: Changed axes labels in plots. Leiden, 18 October 2005, MC
      V1.4.4: Fixed display not being updated while iterating under Windows. 
          MC, Frankfurt, 21 July 2009 
      V1.4.5: Fixed minor plotting bug introduced in V1.43.
          Show individual values with diamonds in profiles.
          MC, Oxford, 10 November 2009
      V1.4.6: Added keyword /QUIET. MC, Oxford, 12 November 2010
      V1.4.7: Added /NEGATIVE keyword as in the other routines of the package, 
          after feedback from Nora Lutzgendorf. MC, Oxford, 15 June 2011
      V1.4.8: Suppress both printing and plotting with /QUIET keyword.
          MC, Oxford, 17 September 2011
      V2.0.0: Translated from IDL into Python. MC, Aspen Airport, 8 February 2014
      V2.0.1: Support both Python 2.6/2.7 and Python 3.x. MC, Oxford, 25 May 2014
 
"""

from __future__ import print_function

import numpy as np    
import matplotlib.pyplot as plt
from scipy import optimize, linalg
from time import clock
from cap_mpfit import mpfit

#----------------------------------------------------------------------------

def _linspace_open(a, b, n):    
        x, dx = np.linspace(a, b, n, endpoint=False, retstep=True)
        return x + dx/2.        

#----------------------------------------------------------------------------

class mge_fit_1d(object):

    def __init__(self, x, y, negative=0, ngauss=12, 
                  rbounds=None, outer_slope=4, quiet=False):
        """
        This is the main routine that has to be called from outside programs
        
        """
        self.x = x # load parameters into structure
        self.y = y
        self.negative = negative
        
        outer_slope = np.clip(outer_slope, 1, 4)
        if rbounds is None:
            rminLog = np.log10(min(x))
            rmaxLog = np.log10(max(x)/np.sqrt(outer_slope))
        else:
            rminLog = np.log10(rbounds[0])
            rmaxLog = np.log10(rbounds[1])
        
        logsigma = _linspace_open(rminLog, rmaxLog, ngauss)
        parinfo = [{'step':0.01, 'limits':[rminLog, rmaxLog], 'limited':[1,1]} 
                    for j in range(ngauss)] 
        
        if not quiet: 
            iterfunc = self._print
        else:
            iterfunc = None
            
        t = clock()
        mp = mpfit(self._fitfunc, logsigma, iterfunct=iterfunc,
                   nprint=10, parinfo=parinfo, quiet=quiet)

        if mp.status <= 0:
            print("Mpfit error, status:", mp.errmsg, mp.status)
            
        # Print the results for the nonzero Gaussians

        w = np.nonzero(mp.params)[0]
        logSigma = mp.params[w]
        self.soluz = self.soluz[w]
        j = np.argsort(logSigma) # Sort by increasing sigma
        
        m = len(logSigma)
        self.sol = np.empty((2, m))
        self.sol[0, :] = self.soluz[j]
        self.sol[1, :] = 10.**logSigma[j]
        
        if not quiet:
            self._print(0, logSigma, mp.niter, mp.fnorm) # plot best fitting solution
            print('############################################')
            print(' Computation time: %.2f seconds' % (clock() - t))
            print(' Total Iterations: ', mp.niter)
            print('Nonzero Gaussians: ', m)
            print(' Unused Gaussians: ', ngauss - m)
            print(' Chi2: %.4g ' % mp.fnorm)
            print(' STDEV: %.4g' % np.std(self.err))
            print(' MEANABSDEV: %.4g' % np.mean(np.abs(self.err)))
            print('############################################')
            print(' Total_Counts         Sigma')
            print('############################################')
            print(self.sol.T)
            print('############################################')

#----------------------------------------------------------------------------
                     
    def _print(self, myfunct, logsigma, it, fnorm, 
                functkw=None, parinfo=None, dof=None, quiet=False):
        """
        This is a plotting routine that is called every NPRINT iteration of MPFIT
        
        """        
        print('Iteration: ', it, '#  chi2: %.4g' % fnorm)

        plt.clf()
        f, ax = plt.subplots(2, 1, sharex=True, num=1)
        f.subplots_adjust(hspace=0)
        
        ax[0].set_ylim([min(self.y), max(self.y)])
        ax[0].loglog(self.x, self.y, 'o')
        ax[0].loglog(self.x, self.yfit)
        ax[0].loglog(self.x, self.gauss*self.soluz[np.newaxis, :])
        ax[0].set_ylabel("arcsec")
        ax[0].set_ylabel("counts")
        
        ax[1].set_ylim([-20, 19.9])
        ax[1].semilogx(self.x, self.err*100)
        ax[1].semilogx(self.x, self.x*0)
        ax[1].semilogx(10**np.vstack([logsigma, logsigma]), [-20, -15])
        ax[1].set_xlabel("arcsec")
        ax[1].set_ylabel("error (%)")
        
        plt.pause(0.01) # Fixes display not being updated under Windows.
        
        return
    
#----------------------------------------------------------------------------

    def _fitfunc(self, logsigma, fjac=None):
        
        sigma2 = 10.**(2.*logsigma)        
        self.gauss = np.exp(-self.x[:, np.newaxis]**2/(2.*sigma2)) \
                   / np.sqrt(2.*np.pi*sigma2)
        A = self.gauss / self.y[:, np.newaxis]        
        b = np.ones_like(self.x)
        
        if self.negative:  # Solution by LAPACK linear least-squares
            self.soluz = linalg.lstsq(A, b)[0]
        else:             # Solution by NNLS
            self.soluz = optimize.nnls(A, b)[0]
        
        self.yfit = self.gauss.dot(self.soluz)
        self.err = 1. - self.yfit / self.y
        
        return [0, self.err.astype(np.float32)] # err is a vector. Important: Solve BVLS in DOUBLE but MPFIT in FLOAT

#----------------------------------------------------------------------------

def test_mge_fit_1d():
    """
    Usage example for mge_fit_1d().
    It takes about 3s on a 2.5 GHz computer
    
    """
    
    # This example reproduces Figure 3 in Cappellari (2002)
    # See the next procedure mge_fit_1d_hernquist_model 
    # for an example using physical units.
    
    n = 300 # number of sampled points
    x = np.logspace(np.log10(0.01), np.log10(300), n) # logarithmically spaced radii
    y = (1. + x)**(-4) # The profile should be logarithmically sampled!
    p = mge_fit_1d(x, y, ngauss=16)

#----------------------------------------------------------------------------

if __name__ == '__main__':
    test_mge_fit_1d()
