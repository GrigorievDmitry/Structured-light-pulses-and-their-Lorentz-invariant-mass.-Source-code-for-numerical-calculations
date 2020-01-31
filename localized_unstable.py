import numpy as np
import matplotlib.pyplot as plt
import time
from pulse import parameter_container
from data_manipulation import save_result, save_environment
from main_calculation_part import field_core, make_preset

#================================PARAMETERS====================================
#===FUNDAMENTAL PARAMETRS OF A PULSE================#
lambda0 = 0.4# * 10**(-4) # microns# #10**(-4) cantimeters #
w0 = 10 #[lambda0]
n_burst = 400
W = 10**5 #erg

presets = make_preset('G', True)
delimiter = '\\'
pars = parameter_container(lambda0, n_burst, w0, W, delimiter)
#==============================================================================



#================================EVALUATION====================================
t1 = time.time()

loc_pulse, mass, velosity = field_core(pars, presets)

print(pars.x.shape)
plt.plot(loc_pulse.l_omega, np.abs(loc_pulse.spec_envelop.ravel()))
plt.show()

fields = []

for (j, z_point) in enumerate(pars.z):
    t1 = time.time()
    print(j)

    loc_pulse.make_t_propagator(z_point, presets['paraxial'])
    loc_pulse.propagate()
    loc_pulse.inverse_ft()
    
    fields.append(loc_pulse.get_field())
    
    t2 = time.time()
    print('Exec_time: %f' %(t2-t1))

fields = np.array(fields).transpose(1, 2, 0, 3, 4)
save_result(fields, 'fields', delimiter, presets['f_type'] + '_new')

#==============================================================================



#===================================PLOTS======================================
save_environment(pars.x, pars.t[1] - pars.t[0], (pars.z[1] - pars.z[0])*(pars.batch_size - 1), \
                 mass, velosity, delimiter, presets['f_type'] + '_new')

print('Mass = %.6e [g]' %(mass))
