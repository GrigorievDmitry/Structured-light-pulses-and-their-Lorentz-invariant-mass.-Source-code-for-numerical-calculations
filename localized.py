import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, assoc_laguerre, eval_hermite, erf
import time
import os
from numba import njit, prange
from pulse import pulse
import field_plotter as fp

#============================MODELING_FUNCTIONS================================

#Defines field boundary conditions
def field(point, name, w0, scalar=False):

    x = point[0]
    y = point[1]

    if name == 'G':
        E = np.exp(-(x**2 + y**2)/w0**2)
        if scalar:
            Ex = E/np.sqrt(2)
            Ey = E/np.sqrt(2)
        else:
            alpha = np.arctan2(y,x)
            Ex = - E * np.sin(alpha)
            Ey = E * np.cos(alpha)
        return Ex * field_modulation(x/w0, y/w0), Ey * field_modulation(x/w0, y/w0)

    if name == 'BG':
        beta = 1./w0
        r = beta * np.sqrt(x**2 + y**2)
        E = jv(1, r)
        Ex, Ey = field(point, 'G', w0)
        Ex = Ex * E
        Ey = Ey * E
        return Ex, Ey

    if name == 'LG':
        l = 1
        r = (np.sqrt(2*x**2 + 2*y**2)/w0)
        G = assoc_laguerre(r**2, l)
        E = r**l * G
        Ex, Ey = field(point, 'G', w0)
        Ex = Ex * E
        Ey = Ey * E
        return Ex, Ey

    if name == 'HG':
        l = 1
        m = 1
        E = eval_hermite(l, np.sqrt(2)*x/w0) * eval_hermite(m, np.sqrt(2)*y/w0)
        Ex, Ey = field(point, 'G', w0)
        Ex = Ex * E
        Ey = Ey * E
        return Ex, Ey
    
#@njit(nogil=True, parallel=True)
def spec_envelop(omega_range, omega0, k, tp):
    env = np.zeros(omega_range.shape[0], dtype=np.complex128)
    for i in prange(omega_range.shape[0]):
        omega = omega_range[i]
        try:
            env[i] = 1j * tp * np.exp(1j*(omega - omega0)*k*tp) * (np.sqrt(2) * np.exp(k**2/2 - (omega - omega0)**2 * tp**2/2) *\
                    (erf(k/np.sqrt(2) + 1j*(omega - omega0)*k*tp/np.sqrt(2)) + erf(k/np.sqrt(2) - 1j*(omega - omega0)*k*tp/np.sqrt(2))) -\
                    2*k*np.sinc((omega - omega0)*k*tp))
        except Exception:
            pass
    return env
#    return np.exp(-(omega - omega0)**2/delta_omega**2)
        
def temporal_envelop(t, k, tp, omega0):
    env = (np.exp(k**2/2 - (t - k*tp)**2/2/tp**2) - 1) * np.sin(omega0*t)
    return env

#Boundary additional modulation
def field_modulation(x, y):
    return 1.
    #return np.cos(x**2 + y**2)

def saleh_teich(x, y, z, t):
    rho = np.sqrt(x**2 + y**2)
    tau0 = 1./delta_omega/np.pi
    N = omega0 * tau0
    z0 = np.pi * w0**2/lambda0
    rho0 = np.pi * N * w0 * z/z0
    t_rho = t - rho**2/(2*c*z)
    tau_rho = tau0 * np.sqrt(1 + rho**2/rho0**2)
    I = np.exp(-2*np.pi*N * rho**2/(rho**2 + rho0**2))/(1 + rho**2/rho0**2) * \
        np.exp(-2*t_rho**2/tau_rho**2)/(1 + t_rho**2/(np.pi*N*tau0)**2)
    return I


#================================PARAMETERS====================================
n = 100 #Spatial grid steps
c = 3 * 10**8 #Speed of light
lambda0 = 404 * 10**(-9) #Wavelength
w0 = 10 * lambda0 #Waist
W = 1. #Total energy of pulse.
k = 2. #Scale factor of z-axis
t_scale = 30. #Time scale factor
t0 = 0 #Initial timestep
T = 100 #Number of timesteps
l0 = 10. #Transverse window size in [w0]
time_window_number = 1 #Number of different space scales (for different time)
tp = 3*10**(-13)

f_type = 'G' #Pulse type ('G', 'BG', 'LG', 'HG')
r_type = 'abs' #'abs' for sqrt(E*E.conj); 'osc' for 1/2*(F+F.conj)
paraxial = False #Use of paraxial approximation
scalar = False #Evaluate scalar field
folder_suffix = 'pure' #Data will be writen in the new foler with given suffix
#Data structure: pic/(f_type)_(folder_suffix)/files
#==============================================================================



#================================EVALUATION====================================
t1 = time.time()
fold = 'pic/' + f_type + '_' + folder_suffix
params = {'n': n, 'c': c, 'lambda0': lambda0, 'w0': w0, 'W': W, 'k': k,
          't_scale': t_scale, 't0': t0, 'T': T, 'l0': l0, 'paraxial': paraxial, 'scalar': scalar}
if not os.path.exists(fold):
    os.makedirs(fold)

omega0 = c/lambda0
delta_omega = omega0/10.
l = np.linspace(-l0*w0, l0*w0, n)

mu = []
mass = []
m = []
intensity = []
velosity = []
angle = []
z_offset = []
saleh_teich_intensity = []

for twn in range(time_window_number):
    loc_pulse = pulse(field, l*(twn + 1), r_type, *(f_type, w0, scalar))
    loc_pulse.spatial_bound_ft()
    temp_range = np.linspace(0, 2*k*tp)
    loc_pulse.temporal_bound_ft(temporal_envelop, temp_range, *(k, tp, omega0))
    loc_pulse.define_Ekz()
    loc_pulse.make_ksi_propagator(paraxial)
    y, z, x = np.meshgrid(l, l/k, l)
    for t in range(T):
        print(t)
        z = (t + t0 + twn * T) * w0/c
#        I = saleh_teich(x, y, z, tau)
#        saleh_teich_intensity.append(I)

        loc_pulse.propagate(z)
        loc_pulse.magnetic()
        loc_pulse.evolution()

        p4k = loc_pulse.momentum()
#        energy, px, py, pz = [pulse.tripl_integrate(p4k[i], (loc_pulse.lk, loc_pulse.lk, loc_pulse.lkz)) for i in range(4)]


        if t == 0:
            energy0 = 1#energy

#        mass_t = W * (1/2/np.pi/c**2) * np.sqrt(energy**2 - (px**2 + py**2 + pz**2)) / energy0
#        mu_t = W * (1/4/np.pi/c**2) * np.sqrt((loc_pulse.E_sq - loc_pulse.H_sq)**2/4 + loc_pulse.EH**2) / energy0
        intensity_t = loc_pulse.E_sq / energy0
#        m_t = pulse.tripl_integrate(mu_t, (l, l, l/k))
#        velosity_t = np.sqrt(1 - (mass_t**2 * c**4) * energy0**2/energy**2)
#        offset = (velosity_t * c * tau)%(2*l0*w0/k)
#        velosity_t = velosity_t - 1.
#        angle_t = 180/np.pi * np.arccos(loc_pulse.EH / np.sqrt(loc_pulse.E_sq * loc_pulse.H_sq))
#
#        mu.append(mu_t)
#        mass.append(mass_t)
#        m.append(m_t)
        intensity.append(intensity_t)
#        velosity.append(velosity_t)
#        angle.append(angle_t)
#        z_offset.append(int(offset//(2*l0*w0/k/n)))

#mass = np.array(mass)
#m = np.array(m)
#velosity = np.array(velosity)
#==============================================================================



#===================================PLOTS======================================
#mu - Mass density; shape(T,n,n,n)
#m - Integrated mass density; shape(T)
#mass - Mass; shape(T), all elements are equal

duration = 1 #Time in seconds for each frame in animation

fp.plot(intensity, l, 'intensity', fold, t_scale * w0/c)
fp.anim('intensity', fold, duration)

#fp.plot(mu, l, 'mu')
#fp.anim('mu', duration)
#
#fig, (ax1, ax2) = plt.subplots(1, 2)
#ax1.plot(m)
#field_fig_preset(ax1, 't', 'Integrated mass density')
#ax2.plot(velosity)
#field_fig_preset(ax2, 't', 'Velosity')
#plt.savefig(fold + '/velosity.png')
#
#with open(fold + '/params.txt', 'w') as file:
#    for par in params.keys():
#        if len(par) < 7:
#            file.write(par + ':\t\t')
#        else:
#            file.write(par + ':\t')
#        val = '%.3e' %params[par] + '\n'
#        file.write(val)
#
#np.savetxt(fold + '/mass.txt', np.array([mass[0]]), '%.6e')
#print('Mass = %.6e' %(mass[0]))
#
#fig, ax = plt.subplots()
#plt.plot(mass)
t2 = time.time()
print('Exec_time: %f' %(t2-t1))
