"""
NAME:
    display_bins(x, y, binNum, velBin)
    
AUTHOR:
    Michele Cappellari, University of Oxford
    E-mail: michele.cappellari_at_physics.ox.ac.uk

PURPOSE:
    This simple routine illustrates how to display a Voronoi binned map.
    
INPUTS:
    (x, y): (length npix) Coordinates of the original spaxels before binning;
    binNum: (length npix) Bin number corresponding to each (x, y) pair,
            as provided in output by the voronoi_2d_binning() routine;
    velBin: (length nbins) Quantity associated to each bin, resulting
            e.g. from the kinematic extraction from the binned spectra.
          
MODIFICATION HISTORY:
    V1.0.0: Michele Cappellari, Oxford, 15 January 2015
    V1.0.1: Further input checks. MC, Oxford, 15 July 2015
    
"""

import numpy as np

from cap_display_pixels import display_pixels

def display_bins(x, y, binNum, velBin):

    if not (x.size == y.size == binNum.size):
        raise ValueError('The vectors (x, y, binNum) must have the same size')

    if np.uniq(binNum).size != velBin.size:
        raise ValueError('velBin size does not match number of bins')
        
    img = display_pixels(x, y, velBin[binNum])
    
    return img
