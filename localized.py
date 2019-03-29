import numpy as np
import matplotlib.pyplot as plt
import time
from pulse import parameter_container
from data_manipulation import save_result, save_environment
from main_calculation_part import field_core

#================================PARAMETERS====================================
#===FUNDAMENTAL PARAMETRS OF A PULSE================#
lambda0 = 0.4# * 10**(-4) # microns# #10**(-4) cantimeters #
w0 = 10 #[lambda0]
n_burst = 400
W = 10**5 #erg

presets = {}
presets['enable_shift'] = True
presets['f_type'] = 'LG' #Pulse type ('G', 'BG', 'LG', 'HG')
presets['r_type'] = 'abs' #'abs' for sqrt(E*E.conj); 'osc' for 1/2*(F+F.conj)
presets['paraxial'] = False #Use of paraxial approximation
presets['scalar'] = False #Evaluate scalar field

delimiter = '\\'
pars = parameter_container(lambda0, n_burst, w0, W, delimiter)
#==============================================================================



#================================EVALUATION====================================
t1 = time.time()

loc_pulse, mass, velosity = field_core(pars, presets)

intensity = []
angle = []

for (j, z_point) in enumerate(pars.z):
    print(j)

    loc_pulse.make_t_propagator(z_point, presets['paraxial'])
    loc_pulse.propagate()
    loc_pulse.inverse_ft()

    intensity_t = loc_pulse.S_abs
    angle_t = 180/np.pi * np.arccos(loc_pulse.EH / np.sqrt(loc_pulse.E_sq * loc_pulse.H_sq))
    
    intensity.append(intensity_t * 10**(10))
    angle.append(angle_t)
    
    
    if (j+1)%pars.batch_size == 0:
        intensity = np.array(intensity)
        save_result(intensity, 'intensity', delimiter, presets['f_type'], number=(j+1)//pars.batch_size)
        intensity = []

#==============================================================================



#===================================PLOTS======================================
save_environment(pars.x, pars.t[1] - pars.t[0], (pars.z[1] - pars.z[0])*(pars.batch_size - 1), \
                 mass, delimiter, presets['f_type'])

print(pars.x.shape)
plt.plot(loc_pulse.l_omega, np.abs(loc_pulse.spec_envelop.ravel()))
plt.show()
print('Mass = %.6e [g]' %(mass))

t2 = time.time()
print('Exec_time: %f' %(t2-t1))
