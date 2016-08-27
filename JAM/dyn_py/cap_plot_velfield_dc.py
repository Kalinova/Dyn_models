"""
  Copyright (C) 2013-2014, Michele Cappellari
  E-mail: cappellari_at_astro.ox.ac.uk
  http://purl.org/cappellari/software

  See example at the bottom for usage instructions.

"""

# V1.0: Michele Cappellari, Paranal, 11 November 2013
# V1.0.1: Clip values before contouring. MC, Oxford, 26 February 2014
# V1.0.2: Include SAURON colormap. MC, Oxford, 29 January 2014
# V1.0.3: Call set_aspect(1). MC, Oxford, 22 February 2014
# V1.0.4: Call autoscale_view(tight=True). Overplot small dots by default.
#    MC, Oxford, 25 February 2014
# V1.0.5: Use axis('image'). MC, Oxford, 29 March 2014
# V1.0.6: Allow changing colormap. MC, Oxford, 29 July 2014
# V1.0.7: Added optional fixpdf keyword to remove PDF artifacts like below:
#    http://stackoverflow.com/questions/15822159/aliasing-when-saving-matplotlib-filled-contour-plot-to-pdf-or-eps
#  - Make nice tick levels for colorbar. Added nticks keyword for colorbar.
#    MC, Oxford, 16 October 2014
#
##############################################################################

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator

from sauron_colormap import sauron
from califa_vfield_ct import califa

##############################################################################

def plot_velfield(x, y, vel, vmin=None, vmax=None, ncolors=64, nodots=False,
                    colorbar=False, label=None, flux=None, fixpdf=False,
                    nticks=7, xlab=None, ylab=None, **kwargs):

    if vmin is None:
        vmin = np.min(vel)

    if vmax is None:
        vmax = np.max(vel)

    x, y, vel = map(np.ravel, [x, y, vel])
    levels = np.linspace(vmin, vmax, ncolors)

    ax = plt.gca()

    cs = ax.tricontourf(x, y, vel.clip(vmin, vmax), levels=levels,
                       cmap=kwargs.get("cmap", sauron))
    ax.axis('image')  # Equal axes and no rescaling
    ax.minorticks_on()
    ax.tick_params(length=5, which='major')
    ax.tick_params(length=2.5, which='minor')

    if flux is not None:
        ax.tricontour(x, y, -2.5*np.log10(flux/np.max(flux).ravel()),
                      levels=np.arange(20), colors='k') # 1 mag contours

    if fixpdf:  # remove white contour lines in PDF at expense of larger file size
        ax.tricontour(x, y, vel.clip(vmin, vmax), levels=levels, zorder=0,
                      cmap=kwargs.get("cmap", sauron))

    if not nodots:
        ax.plot(x, y, '.k', markersize=kwargs.get("markersize", 3))

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)    

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        ticks = MaxNLocator(nticks).tick_values(vmin, vmax)
        cbar = plt.colorbar(cs, cax=cax, ticks=ticks, orientation='horizontal')
        if label:
            cbar.set_label(label)


    return cs

##############################################################################

# Usage example for display_pixels()

if __name__ == '__main__':

    xbin, ybin = np.random.uniform(low=[-30, -20], high=[30, 20], size=(300, 2)).T
    inc = 60.                       # assumed galaxy inclination
    r = np.sqrt(xbin**2 + (ybin/np.cos(np.radians(inc)))**2) # Radius in the plane of the disk
    a = 40                          # Scale length in arcsec
    vr = 2000*np.sqrt(r)/(r+a)      # Assumed velocity profile
    vel = vr * np.sin(np.radians(inc))*xbin/r # Projected velocity field
    flux = np.exp(-r/10)

    plt.clf()
    plt.title('Velocity')
    plot_velfield(xbin, ybin, vel, flux=flux, colorbar=True, label='km/s')
