import math
import time
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import scipy.stats as ss
import scipy.interpolate as interp
from scipy import special, signal, ndimage, integrate, stats
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.io.idl import readsav
from astropy.table import Table
from time import clock

from subpro.readcol import readcol
from subpro.jam_axi_rms import jam_axi_rms
from subpro.mge_vcirc import mge_vcirc
from subpro.jam_axi_vel import jam_axi_vel
from subpro.cap_symmetrize_velfield import symmetrize_velfield as symfield
from subpro.cap_plot_velfield import plot_velfield
from subpro.cap_display_pixels import display_pixels
from subpro.cap_quadva import quadva
from subpro.cap_display_bins import display_bins
from subpro.cap_loess_2d import loess_2d





t = clock()
Mbh=10**6
gals=['488','772','4102','5678','3949','4030','2964','628','864','4254','1042','3346','3423','4487','2805','4775','5585','5668']
dist_all=[32.1,35.6,15.5,31.3,14.6,21.1,20.6,9.8,21.8,19.4,18.1,18.9,14.7,14.7,28.2,22.5,8.2,23.9]
pa_all=[5,126,35,5,122,37,96,25,26,50,174,100,41,77,125,96,38,120]
Vsys_all=[2299,2506,838,1896,808,1443,1324,703,1606,2384,1404,1257,1001,1016,1742,1547,312,1569]

#ideg_all=[39.8,48.75,56.3,58.35,50.25,40.55,56.65,35.95,47.2,43.15,44.8,32.9,39.65,50.95,40.55,30.15,50.25,32.35]
beta_all=[-0.059,0.0,0.35,0.40,0.60,0.0,0.60,-0.15,-0.03,0.10,0.10,0.0,-0.10,0.50,0.30,0.50,0.60,-0.1]
ml_all=[1.19,0.98,0.62,1.20,1.43,0.74,1.37,1.11,1.60,0.49,1.95,2.02,2.36,2.12,2.05,1.03,3.46,1.41]

#qmin_all=[0.77,0.66,0.555,0.525,0.64,0.76,0.55,0.81,0.68,0.73,0.71,0.84,0.77,0.63,0.76,0.865,0.64,0.845]
ideg_all=[39.8,49,57,59,51,41,57,83,65,44,45,42,40,54,41,34,51,37]
#ideg_all=[42,54,55,61,55,44,57,24,41,29,39,29,32,47,41,21,50,24]

#kin_par = readcol('data_input/tables/Kin_par.txt', twod=False, skipline=1)
#ideg_all=kin_par[1]
#pa_all=kin_par[2]

for j in  [7]: 
#for j in range(0,len(gals),1):
	
	#t = clock()
	gal=gals[j]
	dist=dist_all[j]
	pa=pa_all[j]
	Vsys=Vsys_all[j]

	ideg=ideg_all[j]
	beta_scalar=beta_all[j]
	ml=ml_all[j] 



	print 'gal, ideg, pa=', gal, ideg, pa

    
	# read MGEs....
	data = readcol('data_input/mge/mge_NGC'+gal+'.txt', twod=False, skipline=2)
	nr=data[0]
	I0obs=data[1]
	Sigobs=data[2]
	qobs=data[3]
	ic=data[4]
	print 'qobs=',qobs


	hlist = fits.open('data_input/stellar_kinematics/PXF_bin_MS_NGC'+gal+'_r1_MILESstars_SN60.fits')
	tab=hlist[1].data
	xbin = tab['xs']
	ybin = tab['ys']
	Vbin = tab['vpxf'] -Vsys # subtract systemic velocity 
	Sbin = tab['spxf']
	dVbin = tab['dvp']
	dSbin = tab['dsp']
	flux_org=tab['flux']

	#Vrmsbin=np.sqrt(Vbin**2+Sbin**2)
	#dVrmsbin = sqrt((Vbin/Vrmsbin)**2*dVbin**2 + (Sbin/Vrmsbin)**2*dSbin**2)
	#dVrmsbin=np.sqrt((dVbin*Vbin)**2 + (dSbin*Sbin)**2)/Vrmsbin

	

	surf_lum=I0obs
	sigobs_lum=Sigobs
	qobs_lum=qobs
	surf_pot=I0obs
	sigobs_pot=Sigobs
	qobs_pot=qobs


	 
	sVrmsbin = np.sqrt(Vbin**2+Sbin**2)
	Vrmsbin=symfield(xbin,ybin,sVrmsbin,sym=2,pa=pa)
	Vrms, RMSw = loess_2d(xbin, ybin, sVrmsbin, frac=0.5, degree=1, rescale=False)
	dVrmsbin = np.sqrt((Vbin/Vrmsbin)**2*dVbin**2 + (Sbin/Vrmsbin)**2*dSbin**2)
    #erms, erms_w = loess_2d(xbin, ybin, dVrmsbin, frac=0.5, degree=1, rescale=False)

    #....
	sPA = np.sin(pa*np.pi/180.)
	cPA = np.cos(pa*np.pi/180.)
	xmod = -sPA*xbin+cPA*ybin
	ymod = -cPA*xbin-sPA*ybin

	
	
	

	ml=-1
	
	#...run JAM model

	rmsModel, ml, chi2, flux = \
	    jam_axi_rms(surf_lum, sigobs_lum, qobs_lum, surf_pot, sigobs_pot, qobs_pot,
	                     ideg, Mbh, dist, xmod, ymod, plot=False, rms=Vrms, ml=ml,
	                     sigmapsf=0.6, beta=beta_scalar+(surf_lum*0), pixsize=0.8)
   

	#...Plot maps    
	plt.switch_backend('pdf')
	#plt.clf()
	fig = plt.figure(figsize=(18,4))
	plt.subplots_adjust(wspace = 0.01)

	ax = fig.add_subplot(1,6,1)
	lg_flux=np.log(flux_org)
	vmin_flux, vmax_flux = stats.scoreatpercentile(lg_flux, [0.5, 99.5])
	plot_velfield(xbin, ybin, np.log(flux_org), flux=flux_org, colorbar=True,nodots=True, vmin=vmin_flux, vmax=vmax_flux, 
		label='NGC'+gal+'/ Log I OBS') 

	ax= fig.add_subplot(1,6,2)
	plot_velfield(xbin, ybin, Vbin, flux=flux_org, colorbar=True, nodots=True)

	

	#ax = display_pixels(xmod, ymod, Vbin, pixelsize=0.8, angle=8)
    

	ax= fig.add_subplot(1,6,3)
	plot_velfield(xbin, ybin, Sbin, flux=flux_org, colorbar=True, nodots=True)
	

	vmin, vmax = stats.scoreatpercentile(Vrmsbin, [0.05, 99.5])
	ax= fig.add_subplot(1,6,4)
	plot_velfield(xbin, ybin, Vrmsbin, flux=flux, colorbar=True,nodots=True, vmin=vmin, vmax=vmax)

	ax= fig.add_subplot(1,6,5)
	plot_velfield(xbin, ybin, rmsModel, flux=flux, colorbar=True,nodots=True, vmin=vmin, vmax=vmax)

	

	# Residual map
	res=(Vrmsbin/rmsModel)-1
	ax= fig.add_subplot(1,6,6)
	vmin_res, vmax_res = (0,0.3) #stats.scoreatpercentile(res, [0.5, 99.5])
	plot_velfield(xbin, ybin, res, flux=flux_org, colorbar=True,nodots=True, vmin=vmin_res, vmax=vmax_res)

	fig.tight_layout()
	fig.savefig("figures/NGC"+gal+"_maps.pdf")
	plt.close()
    
print "Process time: %.2f" % (time.clock() - t) , "sec, %.2f" % ((time.clock() - t)/60), "min, %.3f" % ((time.clock() - t)/3600), 'h.'
	
