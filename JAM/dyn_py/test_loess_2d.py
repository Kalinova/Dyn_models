##############################################################################
#
# Usage example for the procedure loess_2d
#
# MODIFICATION HISTORY:
#   V1.0: Written by Michele Cappellari, Oxford 26 February 2014
#
##############################################################################

import numpy as np
import matplotlib.pyplot as plt

from cap_loess_2d import loess_2d
from cap_plot_velfield import plot_velfield

def test_loess_2d():
    """
    Usage example for loess_2d

    """
    n = 200
    x = np.random.uniform(-1,1,n)
    y = np.random.uniform(-1,1,n)
    z = x**2 - y**2
    sigz = 0.2
    zran = np.random.normal(z, sigz)

    zout, wout = loess_2d(x, y, zran)

    plt.clf()
    plt.subplot(131)
    plot_velfield(x, y, z)
    plt.title("Underlying Function")

    plt.subplot(132)
    plot_velfield(x, y, zran)
    plt.title("With Noise Added")

    plt.subplot(133)
    plot_velfield(x, y, zout)
    plt.title("LOESS Recovery")
    plt.pause(0.01)

#------------------------------------------------------------------------

if __name__ == '__main__':
    test_loess_2d()
