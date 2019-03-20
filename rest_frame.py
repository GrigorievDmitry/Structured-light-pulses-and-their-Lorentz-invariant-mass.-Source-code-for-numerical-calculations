import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, assoc_laguerre, eval_hermite, erf
import time
import os
from numba import njit, prange
from pulse import pulse, save_result

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
        Ex, Ey = field(point, 'G', w0, scalar)
        Ex = Ex * E
        Ey = Ey * E
        return Ex, Ey

    if name == 'LG':
        l = 1
        q = 0
        r = (np.sqrt(2*x**2 + 2*y**2)/w0)
        G = assoc_laguerre(r**2, l, q)
        E = r**l * G * np.exp(-1j*l*np.arctan2(y,x))
        Ex, Ey = field(point, 'G', w0, scalar)
        Ex = Ex * E
        Ey = Ey * E
        return Ex, Ey

    if name == 'HG':
        l = 1
        m = 1
        E = eval_hermite(l, np.sqrt(2)*x/w0) * eval_hermite(m, np.sqrt(2)*y/w0)
        Ex, Ey = field(point, 'G', w0, scalar)
        Ex = Ex * E
        Ey = Ey * E
        return Ex, Ey

@njit(nogil=True, parallel=True)
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

def temporal_envelop_sin(t, k, tp, omega0):
    x = np.empty(t.shape[0], dtype=np.complex128)
    for i in range(t.shape[0]):
        if t[i] >= 0 and t[i] <= 2*k*tp:
#            x[i] = -np.exp(1j*omega0*t[i])
            x[i] = -1j*np.sin(omega0*t[i]/2)*np.exp(1j*omega0*t[i]/2)
        else:
            x[i] = 0
    return x

@njit(nogil=True, parallel=True)
def spectral_envelop(omega, tp, omega0):
    x = np.empty(omega.shape[0], dtype=np.complex128)
    for i in prange(omega.shape[0]):
        if np.real(omega[i]) > 0 and np.imag(omega[i]) == 0.:
#            x[i] = 1/2 * tp * np.exp(-1j*omega[i]*tp/2) * (np.sinc(omega[i]*tp/2/np.pi) - np.exp(1j*omega0*tp/2) * np.sinc((omega[i]-omega0)*tp/2/np.pi))
            x[i] = -1j * np.sinc((omega[i] - omega0)*tp/2/np.pi) * np.exp(1j*(omega[i] - omega0)*tp/2)
        else:
            x[i] = 0
    return x

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
#===FUNDAMENTAL PARAMETRS OF A PULSE================#
lambda0 = 0.4# * 10**(-4) # microns# #10**(-4) cantimeters #
c = 0.299792458# * 10**(11) #Speed of light [microns/femtoseconds]
omega0 =  2*np.pi*c/lambda0 #(10**15 seconds^(-1)#
n_burst = 400
tp_full = (2*np.pi/omega0)*n_burst #(femtoseconds)#  (#10**(-15) seconds#)
w0 = 100 * lambda0 #(microns)# (#10**(-4) cantimeters#)
k = 1
W = 10**5 * 10**(2*4 - 2*15) #erg -> g*micron**2/femtosec**2
#====CALCULATION AND PLOT SCALES ====================#

#1 . FOR Boundary CONDITIONS#
#1.1 INITIAL SCALES FOR SPATIAL BOUNDARY CONDITIONS #
scale_x = 50*w0
scale_y = 50*w0
points_x = 50
points_y = 50
x = np.linspace(-scale_x, scale_x, points_x)
y = np.linspace(-scale_y, scale_y, points_y)
#1.2 INITIAL SCALES FOR TEMPORAL BOUNDARY CONDITIONS #
tp_max = (1/2)*tp_full
scale_t = 10*tp_full
points_t = 20 #Number of points is choosen in accordance with spectrum detalization(quality requirements#)
t = np.linspace(0, 2*scale_t, points_t)

enable_shift = False
f_type = 'G' #Pulse type ('G', 'BG', 'LG', 'HG')
r_type = 'abs' #'abs' for sqrt(E*E.conj); 'osc' for 1/2*(F+F.conj)
paraxial = False #Use of paraxial approximation
scalar = True #Evaluate scalar field

delimiter = '\\'
#==============================================================================



#================================EVALUATION====================================
t1 = time.time()

mu = []
intensity = []
angle = []
enrg1 = []
enrg2 = []


saleh_teich_intensity = []


loc_pulse = pulse(field, x, y, r_type, *(f_type, w0, scalar))
loc_pulse.spatial_bound_ft()
omega_range = np.linspace(0.99*omega0, 1.01*omega0, 100)
loc_pulse.set_spec_envelop(spectral_envelop, omega_range, *(tp_full, omega0))
loc_pulse.define_Ekz()
loc_pulse.magnetic()

p4k = loc_pulse.momentum()
energy, px, py, pz = [pulse.tripl_integrate(p4k[i], (loc_pulse.lkx, loc_pulse.lky, loc_pulse.l_omega)) for i in range(4)]
N = W/energy #g*micron**2/femtosec**2
loc_pulse.normalize_fields(N)

p4k = loc_pulse.momentum()
energy, px, py, pz = [pulse.tripl_integrate(p4k[i], (loc_pulse.lkx, loc_pulse.lky, loc_pulse.l_omega)) for i in range(4)]
mass = (1/c**2) * np.sqrt(energy**2 - c**2*(px**2 + py**2 + pz**2))
print(mass)
velosity = np.sqrt(1 - (mass**2 * c**4)/energy**2)
print(velosity)
beta = velosity
gamma = 1./np.sqrt(1. - beta**2)

loc_pulse.transform_fields(beta)

#1.3 SCALES OF Z - COORDINATE#
scale_factor = 5 #NUMBER OF PULSE LENGTH IN Z COORDINATE#
scale_z = scale_factor * (lambda0*n_burst)# * 1./np.sqrt(1 - beta**2)
points_z = scale_factor * 20
z = np.linspace(-2*gamma*c*tp_full, gamma*scale_z, points_z)
batch_size = 100
T = np.linspace(0, 2*gamma*scale_t, points_t)

for (j, z_point) in enumerate(z):
    print(j)

    loc_pulse.make_transformed_propagator(z_point)
    loc_pulse.propagate()
#    loc_pulse.inverse_ft()
    
    loc_pulse.transformed_ift(T)

    intensity_t = loc_pulse.S_abs
    angle_t = 180/np.pi * np.arccos(loc_pulse.EH / np.sqrt(loc_pulse.E_sq * loc_pulse.H_sq))
    
    intensity.append(intensity_t * 10**(10))
    angle.append(angle_t)
    
    if (j+1)%batch_size == 0:
        intensity = np.array(intensity)
        save_result(intensity, 'intensity', delimiter, number=(j+1)//batch_size)
        intensity = []
#==============================================================================



#===================================PLOTS======================================
#mu - Mass density; shape(T,n,n,n)
#m - Integrated mass density; shape(T)
#mass - Mass; shape(T), all elements are equal

#plt.plot(enrg)
#plt.plot(enrg1)

fold = os.getcwd() + delimiter + 'data'

np.savetxt(fold + delimiter + 'type.txt', [f_type], '%s')
file = fold + delimiter + 'space.npy'
print(x.shape)
np.save(file, x)
file = fold + delimiter + 't_scale.npy'
np.save(file, loc_pulse.t.max()/loc_pulse.nt)
file = fold + delimiter + 'z_range.npy'
np.save(file, np.array([0, scale_z/points_z * (batch_size - 1)]))
plt.plot(loc_pulse.l_omega, np.abs(loc_pulse.spec_envelop[:,0,0]))
# fp.plot2d(np.abs(loc_pulse.Ek_bound[1]), loc_pulse.lk)
plt.show()

np.savetxt(fold + delimiter + 'mass.txt', np.array([mass]), '%.3e')
print('Mass = %.6e [g]' %(mass))

t2 = time.time()
print('Exec_time: %f' %(t2-t1))
