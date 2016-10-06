from JAM_MCMC import JAM_MCMC
from cap_symmetrize_velfield import symmetrize_velfield as symfield
from cap_loess_2d import loess_2d
from cap_plot_velfield import plot_velfield
import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace as stop
from dyn_py.readcol import readcol
from astropy.io import fits


# MCMC parameters
nwalks=20         # number of walkers (minimum twice the number of the fitting parameters)
threads=1        # number of the used cores for the parallel process
burn_steps=50	   # number of the burn-in steps		
steps=50		   # number of the posterior steps

# resolution
sigmapsf = 0.6 # arcsec/pixel ; resolution of the photometry
pixscale = 0.8	# resolution of the stellar kinematics

#galaxy parameters
gal='NGC0488'	# galaxy name
dist=32.1       # distance of the galaxy in Mpc
Vsys=2299		# systemic velocity		
qmin=0.77		# the minimum flatening of the MGEs
eps=0.230		# ellipticity of the galaxy
Re=53.6			# effective radius of the galaxy in arcsec
PA_gal=5 		# position angle of the galaxy in degrees
Mbh = 1e6 		# Black hole mass in solar masses


#calculating the inclination of the galaxy from the ellipticity 
qo=0.2
q=1-eps
ideg = np.arccos(np.sqrt(q**2-qo**2/(1-qo**2)))*180./np.pi

print 'gal=', gal
plt.switch_backend('pdf')

# read MGEs....
data = readcol('data_input/mge_'+gal+'.txt', twod=False, skipline=2)
nr=data[0]
surf=data[1]
sigma=data[2]
qObs=data[3]
ic=data[4]

surf_lum = surf # Assume self-consistency
sigobs_lum = sigma
qobs_lum = qObs
surf_pot = surf
sigobs_pot = sigma
qobs_pot = qObs

#checking limit of the inclination
qmin  = np.min(qObs)
ideg_lim=np.arccos(np.sqrt((qmin**2-0.051**2)/(1-0.051**2)))*180./np.pi			
ideg_in=np.nanmax([ideg,ideg_lim])



#... reading stellar kinematics
hlist = fits.open('data_input/'+gal+'_stellar_kin.fits')
tab=hlist[1].data
xbin = tab['xs']
ybin = tab['ys']
Vbin = tab['vpxf'] -Vsys # subtract systemic velocity 
Sbin = tab['spxf']
dVbin = tab['dvp']
dSbin = tab['dsp']
flux_org=tab['flux']

#...symmetrising the velocity fields and calculating the corresponding error
Vrmsbin = np.sqrt(Vbin**2+Sbin**2)

dVrmsbin2 = ((dVbin*Vbin)**2 + (dSbin*Sbin)**2)/Vrmsbin**2
dVrmsbin2 = symfield(xbin, ybin, dVrmsbin2, sym=2, pa=PA_gal)
dVrmsbin = 2*np.sqrt(dVrmsbin2)

Vrmsbin = symfield(xbin, ybin, Vrmsbin, sym=2, pa=PA_gal)

minidx = np.zeros(len(Vrmsbin), dtype = np.int)

for ii in range(len(Vrmsbin)):

	xbini = 1.*xbin[ii]
	ybini = 1.*ybin[ii]

	bdists = np.zeros(len(Vrmsbin))

	for jj in range(len(Vrmsbin)):

		xbinj = 1.*xbin[jj]
		ybinj = 1.*ybin[jj]

		bdists[jj] = np.sqrt((xbini - xbinj)**2 + (ybini - ybinj)**2)

	minidx[ii] = np.where(bdists == np.min(bdists[bdists != 0]))[0][0]

pdVrmsbin = np.abs(Vrmsbin - Vrmsbin[minidx])/2.

dVrmsbin = np.max((pdVrmsbin,dVrmsbin),axis=0)

# deprojecting the velocity fields
sPA = np.sin(PA_gal*np.pi/180.)
cPA = np.cos(PA_gal*np.pi/180.)
xmod = -sPA*xbin+cPA*ybin
ymod = -cPA*xbin-sPA*ybin


#plotting Vbin, Sbin and Vrms,obs for check
fig = plt.figure(figsize=(15,7))
plt.subplot(1,3,1)
plot_velfield(xmod, ymod, Vbin, flux=flux_org, colorbar=True, label='km/s',orientation='horizontal', xlab='arcsec', ylab='arcsec', nodots=True)
plt.title(r"$V_{obs}$")

plt.subplot(1,3,2)
plot_velfield(xmod, ymod, Sbin, flux=flux_org, colorbar=True, label='km/s',orientation='horizontal', xlab='arcsec', ylab='arcsec', nodots=True)
plt.title(r"$\sigma_{obs}$")

plt.subplot(1,3,3)
plot_velfield(xmod, ymod, Vrmsbin, flux=flux_org, colorbar=True, label='km/s',orientation='horizontal', xlab='arcsec', ylab='arcsec', nodots=True)
plt.title(r"$V_{rms,obs}$")

fig.tight_layout()
fig.savefig('figures/Vbin/Vobs_'+gal+'.png')
#stop()


#calling JAM-MCMC
res = JAM_MCMC(gal, qmin, dist, surf_lum, sigobs_lum, qobs_lum,
	surf_pot, sigobs_pot, qobs_pot, xmod, ymod, Vrmsbin, dVrmsbin, Mbh, sigmapsf, pixscale, 
	nwalks, burn_steps, steps, threads,ideg_in)



# Plot the results
res.make_fig(gal)

# Make the circular velocity curve
#rad = np.linspace(0,30,15) # Radii in arscec where Vcirc has to be computed
data = readcol('data_input/vel_'+gal+'.txt',twod=False,comment='#')
rad = data[0]
res.make_vcirc(rad, gal, ideg_in)

	
	#stop()
