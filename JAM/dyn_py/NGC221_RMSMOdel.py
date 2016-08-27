from scipy.io.idl import readsav
from jam_axi_rms import jam_axi_rms
from mge_vcirc import mge_vcirc




 # Realistic MGE galaxy surface brightness
    # 
surf_log = np.array([6.187 , 5.774, 5.766, 5.613, 5.311, 4.774, 4.359, 4.087, 3.682, 3.316, 2.744, 1.618])
sigma_log = np.array([-1.762, -1.143, -0.839, -0.438, -0.104, 0.232, 0.560, 0.835, 1.160, 1.414, 1.703, 2.249])
surf=10**surf_log
sigma=10**sigma_log
qobs = np.array([0.790, 0.741, 0.786, 0.757, 0.720, 0.724, 0.725, 0.743, 0.751, 0.838, 0.835, 0.720])
   
#surf = np.array([39483, 37158, 30646, 17759, 5955.1, 1203.5, 174.36, 21.105, 2.3599, 0.25493])
#sigma = np.array([0.153, 0.515, 1.58, 4.22, 10, 22.4, 48.8, 105, 227, 525])
#qobs = np.array([0.57, 0.57, 0.57, 0.57, 0.57, 0.57, 0.57, 0.57, 0.57, 0.57])   

   
inc = 50. # Inclination in degrees
mbh = 1e6 # BH mass in solar masses
distance = 0.773 # Mpc
rad = np.logspace(-1,2,25) # Radii in arscec where Vcirc has to be computed
ml = 1.25 # Adopted M/L ratio
    
vcirc = mge_vcirc(surf*ml, sigma, qobs, inc, mbh, distance, rad)
  

    
plt.clf()
plt.plot(rad, vcirc, '-o')
plt.xlabel('R (arcsec)')
plt.ylabel(r'$V_{circ}$ (km/s)')
plt.ylim(0., 160.)

