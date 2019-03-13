import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, assoc_laguerre, eval_hermite, erf
import time
import os
from numba import njit, prange
from pulse import pulse, save_result
import field_plotter as fp

#============================MODELING_FUNCTIONS================================

#Defines field boundary conditions
def field(point, name, w0, scalar=False):

    x = point[0]
    y = point[1]

    if name == 'G':
        E = np.exp(-(x**2 + y**2)/2/w0**2)
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
    x = np.empty(t.shape[0], dtype=np.complex128)
    for i in range(t.shape[0]):
        if t[i] >= 0 and t[i] <= 2*k*tp:
            x[i] = 1j*(np.exp(k**2/2 - (t[i] - k*tp)**2/2/tp**2) - 1) * np.exp(-1j*omega0*t[i])
        else:
            x[i] = 0
    return x
# def temporal_envelop1(t, k, tp, omega0):
#     x = 1j*(np.exp(-t**2/2/tp**2)) * np.exp(-1j*omega0*t)
#     return x




#Boundary additional modulation
def field_modulation(x, y):
    return 1.
    #return np.cos(x**2 + y**2)

# def saleh_teich(x, y, z, t):
#     rho = np.sqrt(x**2 + y**2)
#     tau0 = 1./delta_omega/np.pi
#     N = omega0 * tau0
#     z0 = np.pi * w0**2/lambda0
#     rho0 = np.pi * N * w0 * z/z0
#     t_rho = t - rho**2/(2*c*z)
#     tau_rho = tau0 * np.sqrt(1 + rho**2/rho0**2)
#     I = np.exp(-2*np.pi*N * rho**2/(rho**2 + rho0**2))/(1 + rho**2/rho0**2) * \
#         np.exp(-2*t_rho**2/tau_rho**2)/(1 + t_rho**2/(np.pi*N*tau0)**2)
#     return I


#================================PARAMETERS====================================
n = 50 #Spatial grid steps
nn = 10
k = 2
c = 3 * 10**8 #Speed of light
lambda0 = 404 * 10**(-9) #Wavelength
omega0 = 2*np.pi*c/lambda0
tp = np.pi*nn/k/omega0
w0 = 10 * lambda0 #Waist
W = 1. #Total energy of pulse.
t_scale = 30. #Time scale factor
t0 = 0 #Initial timestep
T = 200 #Number of timesteps
l0 = 5. #Transverse window size in [w0]
time_window_number = 1 #Number of different space scales (for different time)



#===FUNDAMENTAL PARAMETRS OF A PULSE================#
lambda0 = 0.4 (# microns#) (#10**(-4) cantimeters #)
omega0 =  2*np.pi* 2.99792458/4 #(10**15 seconds^(-1)#
n_burst = 40
tp_full = (2*np.pi/omega0)*n_burst #(femtoseconds)#  (#10**(-15) seconds#)
w0 = 10 * lambda0 #(microns)# (#10**(-4) cantimeters#)

#====CALCULATION AND PLOT SCALES ====================#

#1 . FOR Boundary CONDITIONS#
#1.1 INITIAL SCALES FOR SPATIAL BOUNDARY CONDITIONS #
scale_x = 5*w0
scale_y = 5*w0
points_x = 100
points_y = 100
x = np.linspace(-scale_x,scale_x, points_x)
y = np.linspace(-scale_y,scale_y, points_y)
#1.2 INITIAL SCALES FOR TEMPORAL BOUNDARY CONDITIONS #
tp_max = (1/2)*tp_full
scale_t = 5*tp_full
points_t = 100 (#Number of points is choosen in accordance with spectrum detalization(quality requirements#)
t = np.linspace(- scale_t, scale_t, points_t) + tp_max
#1.3 SCALES OF Z - COORDINATE#
scale_factor = 5 #NUBLER OF PULSE LENGTH IN Z COORDINATE#
z_scale = scale_factor*(lambda0*nburst)
points_z = scale_factor*100
z = np.linspace(0, z_scale, points_z)


f_type = 'G' #Pulse type ('G', 'BG', 'LG', 'HG')
r_type = 'abs' #'abs' for sqrt(E*E.conj); 'osc' for 1/2*(F+F.conj)
paraxial = False #Use of paraxial approximation
scalar = True #Evaluate scalar field
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


loc_pulse = pulse(field, l, r_type, *(f_type, w0, scalar))
loc_pulse.spatial_bound_ft()
temp_range = np.linspace(0, 20*k*tp, 10*20 + 1)
loc_pulse.temporal_bound_ft(temporal_envelop, temp_range, *(k, tp, omega0))
loc_pulse.define_Ekz()
#y, z, x = np.meshgrid(l, l/k, l)
for j in range(T):
    print(j)
    z = (j) * nn * w0/T
#        I = saleh_teich(x, y, z, tau)
#        saleh_teich_intensity.append(I)
    loc_pulse.make_ksi_propagator(z, paraxial)
    loc_pulse.propagate()
    loc_pulse.magnetic()
    loc_pulse.evolution()

    p4k = loc_pulse.momentum()
#        energy, px, py, pz = [pulse.tripl_integrate(p4k[i], (loc_pulse.lk, loc_pulse.lk, loc_pulse.lkz)) for i in range(4)]


    if t == 0:
        energy0 = 1#energy

#        mass_t = W * (1/2/np.pi/c**2) * np.sqrt(energy**2 - (px**2 + py**2 + pz**2)) / energy0
#        mu_t = W * (1/4/np.pi/c**2) * np.sqrt((loc_pulse.E_sq - loc_pulse.H_sq)**2/4 + loc_pulse.EH**2) / energy0
    intensity_t = loc_pulse.S_abs
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

#duration = 1 #Time in seconds for each frame in animation
#
#fp.plot(intensity, l, 'intensity', fold, t_scale * w0/c)
#fp.anim('intensity', fold, duration)

save_result(intensity)
plt.plot(loc_pulse.l_omega, np.abs(loc_pulse.spec_envelop.ravel()))
# fp.plot2d(np.abs(loc_pulse.Ek_bound[1]), loc_pulse.lk)
plt.show()

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
