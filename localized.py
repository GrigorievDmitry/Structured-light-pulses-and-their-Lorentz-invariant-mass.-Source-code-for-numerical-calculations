import numpy as np
import matplotlib.pyplot as plt
import boundaries as bnd
import time
from pulse import pulse
from data_manipulation import save_result, save_environment

#================================PARAMETERS====================================
#===FUNDAMENTAL PARAMETRS OF A PULSE================#
lambda0 = 0.4# * 10**(-4) # microns# #10**(-4) cantimeters #
c = 0.299792458# * 10**(11) #Speed of light [microns/femtoseconds]
omega0 =  2*np.pi*c/lambda0 #(10**15 seconds^(-1)#
n_burst = 400
tp_full = (2*np.pi/omega0)*n_burst #(femtoseconds)#  (#10**(-15) seconds#)
w0 = 10 * lambda0 #(microns)# (#10**(-4) cantimeters#)
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

enable_shift = True
f_type = 'LG' #Pulse type ('G', 'BG', 'LG', 'HG')
r_type = 'abs' #'abs' for sqrt(E*E.conj); 'osc' for 1/2*(F+F.conj)
paraxial = False #Use of paraxial approximation
scalar = False #Evaluate scalar field

delimiter = '\\'
#==============================================================================



#================================EVALUATION====================================
t1 = time.time()

loc_pulse = pulse(bnd.field, x, y, r_type, *(f_type, w0, scalar))
loc_pulse.spatial_bound_ft()
loc_pulse.temporal_bound_ft(bnd.temporal_envelop_sin, t, enable_shift, *(k, tp_max, omega0))
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
velosity = 1. - np.sqrt(1 - (mass**2 * c**4)/energy**2)

mu = []
intensity = []
angle = []
enrg1 = []
enrg2 = []
saleh_teich_intensity = []

for (j, z_point) in enumerate(z):
    print(j)

    loc_pulse.make_t_propagator(z_point, paraxial)
    loc_pulse.propagate()
    loc_pulse.inverse_ft()

    intensity_t = loc_pulse.S_abs
    angle_t = 180/np.pi * np.arccos(loc_pulse.EH / np.sqrt(loc_pulse.E_sq * loc_pulse.H_sq))
    
    intensity.append(intensity_t * 10**(10))
    angle.append(angle_t)
    
    
    if (j+1)%batch_size == 0:
        intensity = np.array(intensity)
        save_result(intensity, 'intensity', delimiter, f_type, number=(j+1)//batch_size)
        intensity = []

#==============================================================================



#===================================PLOTS======================================
save_environment(x, 2*scale_t/points_t, scale_z/points_z * (batch_size - 1), mass, delimiter, f_type)

print(x.shape)
plt.plot(loc_pulse.l_omega, np.abs(loc_pulse.spec_envelop.ravel()))
plt.show()
print('Mass = %.6e [g]' %(mass))

t2 = time.time()
print('Exec_time: %f' %(t2-t1))
