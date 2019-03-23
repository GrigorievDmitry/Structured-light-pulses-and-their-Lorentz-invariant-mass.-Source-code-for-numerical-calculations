import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, assoc_laguerre, eval_hermite, erf, airy
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
        r = np.sqrt(x**2 + y**2)/w0
        G = assoc_laguerre(2*r**2, l, q)
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
    
    if name == 'AG':
        E = airy(x/w0)[0] * airy(y/w0)[0] * np.exp(-(x/w0 + y/w0)**2/2)
        alpha = np.arctan2(y,x)
        Ex = - E * np.sin(alpha)
        Ey = E * np.cos(alpha)
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
            x[i] = -1j*np.exp(1j*omega0*t[i])
#            x[i] = -1j*np.sin(omega0*t[i]/2)*np.exp(1j*omega0*t[i]/2)
#            x[i] = np.sin(omega0*t[i])
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
w_range = np.linspace(3, 30, 50)
fig_m, axes_m = plt.subplots()
fig_v, axes_v = plt.subplots()
f_types = ['G', 'BG', 'LG', 'HG', 'AG']
Scalar = {'G':True, 'BG':False, 'LG':False, 'HG':False, 'AG':True}
linestyles = {'G':'-', 'BG':'--', 'LG':':', 'HG':'-.', 'AG':'-'}
for f_type in f_types:
    Mass = []
    Velosity = []
    for w in w_range:
        print(w)
        #===FUNDAMENTAL PARAMETRS OF A PULSE================#
        lambda0 = 0.4# * 10**(-4) # microns# #10**(-4) cantimeters #
        c = 0.299792458# * 10**(11) #Speed of light [microns/femtoseconds]
        omega0 =  2*np.pi*c/lambda0 #(10**15 seconds^(-1)#
        n_burst = 400
        tp_full = (2*np.pi/omega0)*n_burst #(femtoseconds)#  (#10**(-15) seconds#)
        w0 = w * lambda0 #(microns)# (#10**(-4) cantimeters#)
        k = 1
        W = 10**5 * 10**(2*4 - 2*15) #erg -> g*micron**2/femtosec**2
        #====CALCULATION AND PLOT SCALES ====================#
        
        #1 . FOR Boundary CONDITIONS#
        #1.1 INITIAL SCALES FOR SPATIAL BOUNDARY CONDITIONS #
        scale_x = 10*w0
        scale_y = 10*w0
        points_x = 100
        points_y = 100
        x = np.linspace(-scale_x, scale_x, points_x)
        y = np.linspace(-scale_y, scale_y, points_y)
        #1.2 INITIAL SCALES FOR TEMPORAL BOUNDARY CONDITIONS #
        tp_max = (1/2)*tp_full
        scale_t = 10*tp_full
        points_t = 100 #Number of points is choosen in accordance with spectrum detalization(quality requirements#)
        t = np.linspace(0, 2*scale_t, points_t)
        #1.3 SCALES OF Z - COORDINATE#
        scale_factor = 5 #NUMBER OF PULSE LENGTH IN Z COORDINATE#
        scale_z = scale_factor * (lambda0*n_burst)
        points_z = scale_factor * 20
        z = np.linspace(0, scale_z, points_z)
        
        enable_shift = False
        #f_type = 'G' #Pulse type ('G', 'BG', 'LG', 'HG')
        r_type = 'abs' #'abs' for sqrt(E*E.conj); 'osc' for 1/2*(F+F.conj)
        paraxial = False #Use of paraxial approximation
        scalar = Scalar[f_type] #Evaluate scalar field
        
        delimiter = '\\'
        batch_size = 100
        #==============================================================================
        
        
        
        #================================EVALUATION====================================
        t1 = time.time()
        
        loc_pulse = pulse(field, x, y, r_type, *(f_type, w0, scalar))
        loc_pulse.spatial_bound_ft()
        loc_pulse.temporal_bound_ft(temporal_envelop_sin, t, enable_shift, *(k, tp_max, omega0))
        loc_pulse.center_spectral_range(omega0)
        #loc_pulse.make_spectral_range()
        loc_pulse.define_Ekz()
        loc_pulse.magnetic()
        
        p4k = loc_pulse.momentum()
        energy, px, py, pz = [pulse.tripl_integrate(p4k[i], (loc_pulse.lkx, loc_pulse.lky, loc_pulse.l_omega)) for i in range(4)]
        N = W/energy #g*micron**2/femtosec**2
        loc_pulse.normalize_fields(N)
        
        p4k = loc_pulse.momentum()
        energy, px, py, pz = [pulse.tripl_integrate(p4k[i], (loc_pulse.lkx, loc_pulse.lky, loc_pulse.l_omega)) for i in range(4)]
        mass = (1/c**2) * np.sqrt(energy**2 - c**2*(px**2 + py**2 + pz**2))
        velosity = (1. - np.sqrt(1 - (mass**2 * c**4)/energy**2)) * 10**3
        
        Mass.append(mass)
        Velosity.append(velosity)
    
    fold = os.getcwd() + delimiter + 'data'
    file = fold + delimiter + f_type + '_m_w0.npy'
    np.save(file, Mass)
    file = fold + delimiter + f_type + '_v_w0.npy'
    np.save(file, Velosity)
    
    if f_type is not 'AG':
        axes_m.plot(w_range, Mass, linestyle=linestyles[f_type], color='black')
        axes_v.plot(w_range, Velosity, linestyle=linestyles[f_type], color='black')
    else:
        axes_m.plot(w_range, Mass, marker='.', color='black')
        axes_v.plot(w_range, Velosity, marker='.', color='black')
    axes_m.legend(f_types)
    axes_v.legend(f_types)
    

#mu = []
#intensity = []
#angle = []
#enrg1 = []
#enrg2 = []
#saleh_teich_intensity = []
#
##y, z, x = np.meshgrid(l, l/k, l)
#for (j, z_point) in enumerate(z):
#    print(j)
##        I = saleh_teich(x, y, z, tau)
##        saleh_teich_intensity.append(I)
#    loc_pulse.make_t_propagator(z_point, paraxial)
#    loc_pulse.propagate()
#    loc_pulse.inverse_ft()
#
##    mu_t = W * (1/4/np.pi/c**2) * np.sqrt((loc_pulse.E_sq - loc_pulse.H_sq)**2/4 + loc_pulse.EH**2) / energy0
#    intensity_t = W * loc_pulse.S_abs / energy0
#    angle_t = 180/np.pi * np.arccos(loc_pulse.EH / np.sqrt(loc_pulse.E_sq * loc_pulse.H_sq))
#    
##    mu.append(mu_t * 10**(-13)) #[g]
#    intensity.append(intensity_t * 10**(10))
#    angle.append(angle_t)
#    
##    enrg1.append(energy)
##    enrg2.append(loc_pulse.E_sq + loc_pulse.H_sq)
#    
#    
#    if (j+1)%batch_size == 0:
#        intensity = np.array(intensity)
##        mu = np.array(mu)
#        save_result(intensity, 'intensity', delimiter, number=(j+1)//batch_size)
##        save_result(mu, 'mu', delimiter, number=(j+1)//batch_size)
#        intensity = []
##        mu = []
#
##enrg1 = np.array(enrg1)
##enrg2 = np.transpose(np.array(enrg2), (1,0,2,3))/8/np.pi
##enrg = [pulse.tripl_integrate(enrg2[i], (x, y, z)) for i in range(len(t))]
#==============================================================================



#===================================PLOTS======================================
#mu - Mass density; shape(T,n,n,n)
#m - Integrated mass density; shape(T)
#mass - Mass; shape(T), all elements are equal

#plt.plot(enrg)
#plt.plot(enrg1)

#fold = os.getcwd() + delimiter + 'data'
#
#np.savetxt(fold + delimiter + 'type.txt', [f_type], '%s')
#file = fold + delimiter + 'space.npy'
#print(x.shape)
#np.save(file, x)
#file = fold + delimiter + 't_scale.npy'
#np.save(file, 2*scale_t/points_t)
#file = fold + delimiter + 'z_range.npy'
#np.save(file, np.array([0, scale_z/points_z * (batch_size - 1)]))
#plt.plot(loc_pulse.l_omega, np.abs(loc_pulse.spec_envelop.ravel()))
## fp.plot2d(np.abs(loc_pulse.Ek_bound[1]), loc_pulse.lk)
#plt.show()
#
#np.savetxt(fold + delimiter + 'mass.txt', np.array([mass]), '%.3e')
#print('Mass = %.6e [g]' %(mass))
#
#t2 = time.time()
#print('Exec_time: %f' %(t2-t1))
