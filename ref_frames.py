import numpy as np
import matplotlib.pyplot as plt
import time
from pulse import parameter_container
from data_manipulation import save_result, save_environment
from main_calculation_part import field_core, make_preset, transform_field

#================================PARAMETERS====================================
#===FUNDAMENTAL PARAMETRS OF A PULSE================#
lambda0 = 0.4# * 10**(-4) # microns# #10**(-4) cantimeters #
w0 = 10 #[lambda0]
n_burst = 400
W = 10**5 #erg

presets = make_preset('G', False)
delimiter = '\\'
pars = parameter_container(lambda0, n_burst, w0, W, delimiter)
#==============================================================================



#================================EVALUATION====================================
t1 = time.time()

loc_pulse, mass, velosity = field_core(pars, presets)

z_point = pars.z[5]
loc_pulse.make_t_propagator(z_point, presets['paraxial'])
loc_pulse.propagate()
loc_pulse.inverse_ft()
fields = loc_pulse.get_field()

#points = np.random.randint(2, size=fields[0].shape).astype(bool)
invariants = (sum([fields[i]**2 for i in range(3)]) -\
            sum([fields[i]**2 for i in range(3, 6)])).ravel()
fields_transformed = transform_field(fields, 0.5)
invariants_transformed = sum([fields_transformed[i]**2 for i in range(3)]) -\
            sum([fields_transformed[i]**2 for i in range(3, 6)])
            
rel_abs_error = np.abs((invariants_transformed - invariants)/invariants)
print(rel_abs_error)


