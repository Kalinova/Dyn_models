from ADC_MOC_pw import ADC_MOC_pw
import numpy as np
from sub_pro.readcol import readcol
from pdb import set_trace as stop
from astropy.io import fits


gal='NGC0488' # galaxy name
incl=42.06    # inclination in degrees
PA=5          # position angle

# Initial guess for the fitting parameters
Vinf_in=220.52 
Rc_in= 7.9
sig0_in=240.79
ksig_in=-4.42

# MCMC parameters
nwalks=60       # number of walkers (minimum twice the number of the parameters)
burn_steps=400  # number of burn-in steps
steps=1000 		# number of posterior steps 
threads=1       # number of the used cores for multiple process


#-------------------------------------
# reading the stellar kinematics in order to estimate the velocity uncertainties
hlist = fits.open('data_input/'+gal+'_stellar_kin.fits')
tab=hlist[1].data
xbin = tab['xs']
ybin = tab['ys']
dVbin = tab['dvp']
dSbin = tab['dsp']


# de-project the velocity fields
sPA = np.sin(PA*np.pi/180.)
cPA = np.cos(PA*np.pi/180.)
xmod = -sPA*xbin+cPA*ybin
ymod = -cPA*xbin-sPA*ybin

# radial extend of the stellar kinematics
Rmod = (xmod**2 + (ymod/np.cos(ideg*np.pi/180.))**2)**0.5
Rm=np.max(Rmod)


print 'gal, ideg, Rmax =', gal, ideg, Rm

#-----------------

# read V adn S kinemetry profiles
R_arsec,PA_rad, er_PA_rad,q,er_q,k1,erk1,k51,erk51,Vsys=readcol('data_input/vel_'+gal+'.txt',skipline=1,twod = False )
R_arsec2,PA_rad, er_PA_rad,q,er_q,k0,erk0,k51,erk51=readcol('data_input/sigma_'+gal+'.txt',skipline=1,twod = False )
R= R_arsec  # radius of the profiles
vobs=k1 # velocity profile
sobs=k0 # velocity dispersion profile

#evobs=erk1 # the error of the velocity from kinemetry routine 
#esobs=erk0 # the error of the velocity dispersion from kinemetry routine

# calculation of the error of V and Sigma profiles within the ellipses, define in kinemetry routine
dVbins=np.zeros(len(R))
dSbins=np.zeros(len(R))	
for j in range(0,len(R),1):
	dVbin_new = dVbin[(Rmod >= R[j]-0.5) & (Rmod < R[j]+0.5)]
	dSbin_new = dSbin[(Rmod >= R[j]-0.5) & (Rmod < R[j]+0.5)]		
	dVbins[j] = np.nanmedian(dVbin_new)
	dSbins[j] = np.nanmedian(dSbin_new)

evobs=dVbins # calculated error of V profile
esobs=dSbins # calculated error of Sigma profile


# read MGEs....
data = readcol('data_input/mge/mge_'+gal+'.txt', twod=False, skipline=2)
nr=data[0]
I0obs=data[1]
spobs=data[2]
qObs=data[3]
ic=data[4]


plt.switch_backend('pdf')

# call ADC-MCMC
res = ADC_MOC_pw(gal, incl, Vinf_in, Rc_in,sig0_in,ksig_in,R,vobs,evobs,sobs,esobs,I0obs,spobs,nwalks, burn_steps, steps, threads)

         
# Plot the results
res.make_fig(incl)


# fitted_curves
res.make_fig_curves(incl)
