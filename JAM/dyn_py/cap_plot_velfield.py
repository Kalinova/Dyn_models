"""
  Copyright (C) 2013-2015, Michele Cappellari
  E-mail: michele.cappellari_at_physics.ox.ac.uk
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
#    http://stackoverflow.com/questions/15822159/
#  - Make nice tick levels for colorbar. Added nticks keyword for colorbar.
#    MC, Oxford, 16 October 2014
# V1.0.8: Return axis of main plot. MC, Oxford, 26 March 2015
# V1.0.9: Clip values within +/-eps of vmin/vmax, to assign clipped values
#    the top colour in the colormap, rather than having an empty contour.
#    MC, Oxford, 18 May 2015
# V1.0.10: Removed optional fixpdf keyword and replaced with better solution.
#    MC, Oxford, 5 October 2015
#
##############################################################################

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator

from sauron_colormap import sauron
from califa_vfield_ct import califa
from califa_intens_ct import califa_int
##############################################################################


def plot_velfield(x, y, vel, vmin=None, vmax=None, ncolors=64, nodots=False,
                    colorbar=False, label=None, flux=None, eps=1e-7, 
                    nticks=7, xlab=None, ylab=None, title=None, **kwargs):

    x, y, vel = map(np.ravel, [x, y, vel])

    if not (x.size == y.size == vel.size):
        raise ValueError('The vectors (x, y, vel) must have the same size')

    if vmin is None:
        vmin = np.min(vel)

    if vmax is None:
        vmax = np.max(vel)

    levels = np.linspace(vmin, vmax, ncolors)

    ax = plt.gca()

    cnt = ax.tricontourf(x, y, vel.clip(vmin + eps, vmax - eps), levels=levels,
                       cmap=kwargs.get("cmap", sauron))

    for c in cnt.collections:    # Remove white gaps in contours levels of PDF
        c.set_edgecolor("face")  # http://stackoverflow.com/a/32911283/

    ax.axis('image')  # Equal axes and no rescaling
    ax.minorticks_on()
    ax.tick_params(length=3, which='major')
    ax.tick_params(length=1.5, which='minor')

    if flux is not None:
        ax.tricontour(x, y, -2.5*np.log10(flux/np.max(flux).ravel()),
                      levels=np.arange(20), colors='k')  # 1 mag contours

    if not nodots:
        ax.plot(x, y, '.k', markersize=kwargs.get("markersize", 3))

    
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    
    if title:
        ax.set_title(title)

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad=0.5)
        ticks = MaxNLocator(nticks).tick_values(vmin, vmax)
        cbar = plt.colorbar(cnt, cax=cax, ticks=ticks, orientation='horizontal')
        cbar.solids.set_edgecolor("face")  # Remove gaps in PDF http://stackoverflow.com/a/15021541
        if label:
            cbar.set_label(label)

    return ax

##############################################################################

# Usage example for plot_velfield()

if __name__ == '__main__':

    np.random.seed(123)
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
    plt.pause(0.01)
