##############################################################################
#
# Copyright (C) 2007-2014, Michele Cappellari
# E-mail: cappellari_at_astro.ox.ac.uk
#
# Updated versions of the software are available from my web page
# http://purl.org/cappellari/software
#
# If you have found this software useful for your research,
# I would appreciate an acknowledgment and a link to the website.
# 
# This software is provided as is without any warranty whatsoever.
# Permission to use, for non-commercial purposes is granted.
# Permission to modify for personal or internal use is granted,
# provided this copyright and disclaimer are included unchanged
# at the beginning of the file. All other rights are reserved.
#
##############################################################################
#
# function [Ifx,errbnd] = quadva(fun,interval,reltol,abstol)
# When INTERVAL = [A,B], computes the integral of a continuous function
# f(x) from A to B for A < B.  A can be -Inf and/or B can be Inf. FUN
# accepts a row vector X and returns a vector Y with Y(m) = f(X(m))
# for m = 1,...,length(X). FUN can be a string that QUADVA will convert
# to a vectorized, inline function.
#
# QUADVA returns an approximation Ifx to the integral and optionally an
# approximate bound, ERRBND, on the error |integral - Ifx|. It attempts
# to compute Ifx such that |Ifx - integral| <= max(ABSTOL,RELTOL*|Ifx|).
# If QUADVA is unsuccessful, Ifx and ERRBND are still meaningful, so a
# warning is issued that includes ERRBND.
#
# If the interval is infinite, say [a,Inf), then for the integral to
# exist, f(x) must decay as x -> Inf and QUADVA requires it to decay
# rapidly. Special methods should be used for oscillatory functions on
# infinite intervals, but QUADVA can be used if f(x) decays fast enough.
#
# QUADVA will integrate f(x) that are singular at finite end points if
# the singularities are not too strong. For example, it will integrate
# f(x) that behave like log|x-c| or |x-c|^p for p >= -1/2 with c = a
# and/or c = b. If f(x) is singular at points inside (A,B), write the
# integral as a sum of integrals over subintervals with the singular
# points as end points, compute them with QUADVA, and add the results.
#
# QUADVA starts with samples of f(x) at 150 points in (A,B). It must be
# able to recognize the behavior of f(x) from these samples, so if f(x)
# oscillates very rapidly or has sharp peaks, it may be necessary to
# subdivide the interval. To do this, make INTERVAL an array with entries
# (breakpoints) that increase from A = INTERVAL(1) to B = INTERVAL(end).
# If f(x) is only piecewise smooth, the points of discontinuity should
# be breakpoints.
# 
# Based on the algorithm "Vectorized Adaptive Quadrature in Matlab"
# L.F. Shampine, Journal of Computational and Applied Mathematics 211 (2008) 131-140
# http://dx.doi.org/10.1016/j.cam.2006.11.021
#
# MODIFICATION HISTORY:
# V1.0: Written and tested against the corresponding MATLAB version,
#       Michele Cappellari, Oxford, 22 October, 2007
# V1.1: Allow function parameters to be passed via the FUNCTARGS keyword.
#       MC, Oxford, 23 October 2007
# V1.11: Added STATUS keyword. Provide more informative error messages on failure.
#       MC, Windoek, 5 October 2008
# V1.12: Renamed CAP_QUADVA to avoid potential naming conflicts.
#       MC, Paranal, 8 November 2013
# V2.0: Translated from IDL into Python. MC, Paranal, 14 November 2013
# V2.01: Fixed possible program stop. MC, Oxford, 27 January 2014
# V2.02: Support both Python 2.6/2.7 and Python 3.x. MC, Oxford, 25 May 2014
# 
#----------------------------------------------------------------------

from __future__ import print_function

import numpy as np    

EPS = 100.*np.MachAr().eps

#----------------------------------------------------------------------
def _qva_split(interval):

    # If breakpoints are specified, split subintervals in
    # half as needed to get a minimum of 10 subintervals.
    v = interval
    while True:
        npts = interval.size
        if npts >= 11: 
            break
        v = np.zeros(npts*2-1)
        v[0::2] = interval
        v[1::2] = interval[:-1] + 0.5*np.diff(interval)
        interval = v
    
    return v
#----------------------------------------------------------------------
def _qva_check_spacing(x):

    ax = np.abs(x) 
    too_close = np.any(np.diff(x) <= EPS*np.maximum(ax[:-1],ax[1:]))
    
    return too_close
#----------------------------------------------------------------------
def _qva_f1(fun,t,a,b,args=()):

    # Transform to weaken singularities at both ends: [a,b] -> [-1,1]
    Tt = 0.25*(b-a)*t*(3. - t**2) + 0.5*(b+a)
    too_close = _qva_check_spacing(Tt)
    if  too_close:
        y = []
    else:
        y = fun(Tt,*args)
        y = 0.75*(b-a)*y*(1. - t**2)        
        
    return y, too_close
#----------------------------------------------------------------------
def _qva_f2(fun,t,a,b,args=()):

    # Transform to weaken singularity at left end: [a,np.Inf) -> [0,Inf).
    # Then transform to finite interval: [0,Inf) -> [0,1].
    Tt = t / (1. - t)
    T2t = a + Tt**2
    too_close = _qva_check_spacing(T2t)
    if  too_close:
        y = []
    else:
        y = fun(T2t,*args)
        y =  2.*Tt * y / (1. - t)**2
        
    return y, too_close
#----------------------------------------------------------------------    
def _qva_f3(fun,t,a,b,args=()):

    # Transform to weaken singularity at right end: (-Inf,b] -> (-Inf,b].
    # Then transform to finite interval: (-Inf,b] -> (-1,0].
    Tt = t / (1. + t)
    T2t = b - Tt**2
    too_close = _qva_check_spacing(T2t)
    if too_close:
        y = []
    else:
        y = fun(T2t,*args)
        y = -2.*Tt * y / (1. + t)**2
        
    return y, too_close
#----------------------------------------------------------------------    
def _qva_f4(fun,t,a,b,args=()):

    # Transform to finite interval: (-Inf,Inf) -> (-1,1).
    Tt = t / (1. - t**2)
    too_close = _qva_check_spacing(Tt)
    if too_close:
        y = []
    else:
        y = fun(Tt,*args)
        y = y * (1. + t**2) / (1. - t**2)**2
        
    return y, too_close
#----------------------------------------------------------------------        
def _qva_Vadapt(f,tinterval,rtol,atol, samples,nodes,wt,ewt,fun,a,b,args=()):

    tbma = np.abs(tinterval[-1] - tinterval[0]) # length of transformed interval
    
    # Initialize array of subintervals of [a,b].
    subs = np.column_stack([tinterval[:-1],tinterval[1:]]) # Two columns array[n,2]
    
    # Initialize partial sums.
    IfxOK = 0
    errOK = 0
    
    # Initialize main loop
    OK = True # true
    first = True # true
    Ifx = 0.
    errbnd = 0.

    while True:
        # SUBS contains subintervals of [a,b] where the integral is not
        # sufficiently accurate.  The first row of SUBS holds the left end
        # points and the second row, the corresponding right end points.
        midpt = np.sum(subs,axis=1)/2.      # midpoints of the subintervals
        halfh = np.diff(subs,axis=1)/2.     # half the lengths of the subintervals
        x = nodes * halfh + midpt[:,np.newaxis] # broadcasting midpt
        halfh = halfh.ravel()
        x = x.ravel()
    
        fx, too_close = f(fun,x,a,b,args=args)
    
        # Quit if mesh points are too close or too close to a
        # singular point or got into trouble on first evaluation.
        not_finite = np.any(np.isinf(fx))
        if too_close or not_finite: 
            break
    
        fx = fx.reshape(-1,samples)
    
        # Quantities for subintervals.
        Ifxsubs = fx.dot(wt) * halfh
        errsubs = fx.dot(ewt) * halfh
    
        # Quantities for all of [a,b].
        Ifx = np.sum(Ifxsubs) + IfxOK
        errbnd = abs(np.sum(errsubs) + errOK)
    
        # Test for convergence:
        tol = max(atol, rtol*np.abs(Ifx))
        if errbnd <= tol: 
            return Ifx, errbnd, OK
    
        # Locate subintervals where the approximate integrals are
        # sufficiently accurate and use them to update partial sums.
        ndx = np.where(np.abs(errsubs) <= (2./tbma)*halfh*tol)[0]
        errOK += np.sum(errsubs[ndx])
        IfxOK += np.sum(Ifxsubs[ndx])
        #Remove subintervals with accurate approximations.
        subs = np.delete(subs, ndx, axis=0)
        if subs.size == 0: 
            return Ifx, errbnd, OK # all intervals are accurate
    
        # Split the remaining subintervals in half. Quit if splitting
        # results in too many subintervals.
        many_subint = 2.*subs.size > 650 # multiplied limit by 10x MC 26/FEB/2008 
        if many_subint: 
            break 
        midpt = np.sum(subs,axis=1)/2.
        tmp = np.column_stack([subs[:,0], midpt, midpt, subs[:,1]])
        subs = tmp.reshape(-1,2) # ---> subs[n,2]
        first = False

    if first:
        if too_close: 
            print('***Sub intervals too close.')
        elif not_finite:
            print('***Infinite values in integrand.')
        elif many_subint: 
            print('***Too many sub intervals.')
        OK = False

    return Ifx, errbnd, OK
#----------------------------------------------------------------------
def quadva(fun, interval, epsrel=1e-5, epsabs=1e-10, args=()):

    interval = np.asarray(interval)    
    nint = interval.size
    if nint < 2:
        raise ValueError("INTERVAL must be a real vector of at least two entries.")
    if np.any(np.diff(interval) <= 0):
        raise ValueError("Entries of INTERVAL must strictly increase.")

    a = interval[0]
    b = interval[-1]
    
    # Generally the error test is a mixed one, but pure absolute error
    # and pure relative error are allowed.  If a pure relative error
    # test is specified, the tolerance must be at least 100*EPS. 
    #
    rtol = max(epsrel, EPS)
    atol = epsabs
    
    # Gauss-Kronrod (7,15) pair. Use symmetry in defining nodes and weights.
    #
    samples = 15
    pnodes = np.array([0.2077849550078985, 0.4058451513773972, 0.5860872354676911, 
                       0.7415311855993944, 0.8648644233597691, 0.9491079123427585, 
                       0.9914553711208126])
    nodes = np.hstack([-pnodes[::-1], 0, pnodes])
    pwt = np.array([0.2044329400752989, 0.1903505780647854, 0.1690047266392679, 
                    0.1406532597155259, 0.1047900103222502, 0.06309209262997855, 
                    0.02293532201052922])
    wt = np.hstack([pwt[::-1], 0.2094821410847278, pwt])
    pwt7 = np.array([0, 0.3818300505051189, 0, 0.2797053914892767, 
                     0, 0.1294849661688697, 0])
    ewt = wt - np.hstack([pwt7[::-1], 0.4179591836734694, pwt7])
    
    # Identify the task. If breakpoints are specified, work out
    # how they map into the standard interval.
    #
    if a != -np.Inf and b != np.Inf:
        if nint > 2:
            # Analytical transformation suggested by K.L. Metlov:
            alpha = 2.*np.sin( np.arcsin((a + b - 2.*interval[1:-1])/(a - b))/3. )
            tinterval = np.hstack([-1., alpha, 1.])
            tinterval = _qva_split(tinterval)
        else:
            tinterval = np.linspace(-1,1,11)
        Ifx,errbnd,OK = _qva_Vadapt(_qva_f1,tinterval,rtol,atol,samples,
                                   nodes,wt,ewt,fun,a,b,args=args)
    elif a != -np.Inf and b == np.Inf:
        if nint > 2:
            alpha = np.sqrt(interval[1:-1] - a)
            tinterval = np.hstack([0., alpha/(1. + alpha), 1.])
            tinterval = _qva_split(tinterval)
        else:
            tinterval = np.linspace(0,1,11)
        Ifx,errbnd,OK = _qva_Vadapt(_qva_f2,tinterval,rtol,atol,samples,
                                   nodes,wt,ewt,fun,a,b,args=args)
    elif a == -np.Inf and b != np.Inf:
        if nint > 2:
            alpha = np.sqrt(b - interval[1:-1])
            tinterval = np.hstack([-1., -alpha/(1. + alpha), 0.])
            tinterval = _qva_split(tinterval)
        else:
            tinterval = np.linspace(-1,0,11)
        Ifx,errbnd,OK = _qva_Vadapt(_qva_f3,tinterval,rtol,atol,samples,
                                   nodes,wt,ewt,fun,a,b,args=args)
    elif a == -np.Inf and b == np.Inf:
        if nint > 2:
            # Analytical transformation suggested by K.L. Metlov:
            alpha = np.tanh( np.arcsinh(2.*interval[1:-1])/2. )
            tinterval = np.hstack([-1., alpha, 1.])
            tinterval = _qva_split(tinterval)
        else:
            tinterval = np.linspace(-1,1,11)
        Ifx,errbnd,OK = _qva_Vadapt(_qva_f4,tinterval,rtol,atol,samples,
                                   nodes,wt,ewt,fun,a,b,args=args)
    
    if OK:
        status = 0 # success   
    else: 
        print('***Integral does not satisfy error test.')
        print('***Approximate bound on error is', errbnd)
        status = 1
    
    return Ifx, errbnd, status
#----------------------------------------------------------------------        
def quadva_test1(x,a=3,b=5):
    # Gladwell's problem no1. limits x=[0,8]
    # Precise result: 0.33333333332074955152
    
    return np.exp(-a*x)-np.cos(b*np.pi*x)
#----------------------------------------------------------------------        
def quadva_test2(x):
    # Gladwell's problem no2. limits x=[-1,2]
    # Precise result: 5.9630898453302550932
    
    return np.abs(x-1/np.sqrt(3)) + np.abs(x+1/np.sqrt(2))
#----------------------------------------------------------------------
def quadva_test3(x, a=2.0/3):
    # Gladwell's problem no3. limits x=[0,1]
    # Precise result: 3 with x^(-2d/3d)
    
    return x**a
#----------------------------------------------------------------------
def quadva_test4(x):
    # Gladwell's problem no3. limits x=[0,1]
    # Precise result: 2.5066282746310005024
    
    return np.exp(-x**2/2)
#----------------------------------------------------------------------
def quadva_examples():

    print('test 1 ###############################')
    ifx, errbnd, ok = quadva(quadva_test1, [0,8], args=(3,5), epsrel=0, epsabs=1e-12)
    print(ifx, '+/-', errbnd, ok) 
    print('actual error:', ifx - 0.33333333332074955152)
    print('test 2 ###############################')
    ifx, errbnd, ok = quadva(quadva_test2, [-1,2], epsrel=0, epsabs=1e-12)
    print(ifx, '+/-', errbnd, ok) 
    print('actual error:', ifx - 5.9630898453302550932)
    print('test 2 with breakpoints ##############')
    ifx, errbnd, ok = quadva(quadva_test2, [-1,-1/np.sqrt(2),1/np.sqrt(3),2], epsrel=0, epsabs=1e-12)
    print(ifx, '+/-', errbnd, ok) 
    print('actual error:', ifx - 5.9630898453302550932)
    print('test 3 ###############################')
    ifx, errbnd, ok = quadva(quadva_test3, [0,1], args=(-2.0/3,), epsrel=0, epsabs=1e-12)
    print(ifx, '+/-', errbnd, ok) 
    print('actual error:', ifx - 3)
    print('test 4 ###############################')
    ifx, errbnd, ok = quadva(quadva_test4, [-np.Inf,np.Inf], epsrel=1e-5, epsabs=0)
    print(ifx, '+/-', errbnd, ok) 
    print('actual error:', ifx - 2.5066282746310005024)
    ifx, errbnd, ok = quadva(quadva_test4, [-6,6], epsrel=1e-5, epsabs=0)
    print(ifx, '+/-', errbnd, ok) 
    print('actual error:', ifx - 2.5066282746310005024)
#----------------------------------------------------------------------

if __name__ == '__main__':
    quadva_examples()
