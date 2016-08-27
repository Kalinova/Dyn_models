"""
    Copyright (C) 2015, Michele Cappellari
    E-mail: michele.cappellari_at_physics.ox.ac.uk

    Updated versions of the software are available from my web page
    http://purl.org/cappellari/software

    See example at the bottom for usage instructions.

    V1.0.0: Created to emulate my IDL procedure with the same name.
        Michele Cappellari, Oxford, 28 March 2014
    V1.0.1: Fixed treatment of optional parameters. MC, Oxford, 6 June 2014
    V1.0.2: Avoid potential runtime warning. MC, Oxford, 2 October 2014
    V1.0.3: Return axis. MC, Oxford, 26 March 2015
    V1.0.4: Return image instead of axis. MC, Oxford, 15 July 2015
    V1.0.5: Removes white gaps from rotated images using edgecolors.
        MC, Oxford, 5 October 2015

"""
import matplotlib.pyplot as plt
from scipy.spatial import distance
import numpy as np

from sauron_colormap import sauron

##############################################################################

def display_pixels(x, y, val, pixelsize=None, angle=None, **kwargs):
    """
    Display vectors of square pixels at coordinates (x,y) coloured with "val".
    An optional rotation around the origin can be applied to the whole image.
    
    The pixels are assumed to be taken from a regular cartesian grid with 
    constant spacing (like an image), but not all elements of the grid are 
    required (missing data are OK).

    This routine is designed to be fast even with large images and to produce
    minimal file sizes when the output is saved in a vector format like PDF.

    """
    if pixelsize is None:
        pixelsize = np.min(distance.pdist(np.column_stack([x, y])))

    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    nx = round((xmax - xmin)/pixelsize) + 1
    ny = round((ymax - ymin)/pixelsize) + 1
    j = np.round((x - xmin)/pixelsize).astype(int)
    k = np.round((y - ymin)/pixelsize).astype(int)
    mask = np.ones((nx, ny), dtype=bool)
    img = np.empty((nx, ny))
    mask[j, k] = 0
    img[j, k] = val
    img = np.ma.masked_array(img, mask)

    ax = plt.gca()

    if (angle is None) or (angle == 0):

        img = ax.imshow(np.rot90(img), interpolation='none',
                        cmap=kwargs.get("cmap", sauron),
                        extent=[xmin-pixelsize/2, xmax+pixelsize/2,
                                ymin-pixelsize/2, ymax+pixelsize/2])

    else:

        x, y = np.ogrid[xmin-pixelsize/2 : xmax+pixelsize/2 : (nx+1)*1j,
                        ymin-pixelsize/2 : ymax+pixelsize/2 : (ny+1)*1j]
        ang = np.radians(angle)
        x, y = x*np.cos(ang) - y*np.sin(ang), x*np.sin(ang) + y*np.cos(ang)

        mask1 = np.ones_like(x, dtype=bool)
        mask1[:-1, :-1] *= mask  # Flag the four corners of the mesh
        mask1[:-1, 1:] *= mask
        mask1[1:, :-1] *= mask
        mask1[1:, 1:] *= mask
        x = np.ma.masked_array(x, mask1)  # Mask is used for proper plot range
        y = np.ma.masked_array(y, mask1)

        img = ax.pcolormesh(x, y, img, cmap=kwargs.get("cmap", sauron), edgecolors="face")
        ax.axis('image')

    ax.minorticks_on()
    ax.tick_params(length=10, width=1, which='major')
    ax.tick_params(length=5, width=1, which='minor')

    return img

##############################################################################

# Usage example for display_pixels()

if __name__ == '__main__':

    n = 41  # 1 arcsec pixels
    x = np.linspace(-20, 20, n)
    y = np.linspace(-20, 20, n)
    xx, yy = np.meshgrid(x,y)
    counts = xx**2 - 2*yy**2
    w = xx**2 + 2*yy**2 < 20.1**2

    plt.clf()
    ax = display_pixels(xx[w], yy[w], counts[w], pixelsize=x[1]-x[0], angle=20)
    plt.pause(0.01)
