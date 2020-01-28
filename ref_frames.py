import numpy as np
import matplotlib.pyplot as plt
import time
from pulse import parameter_container
from data_manipulation import save_environment
from main_calculation_part import field_core, make_preset, 

#================================PARAMETERS====================================
#===FUNDAMENTAL PARAMETRS OF A PULSE================#
lambda0 = 0.4# * 10**(-4) # microns# #10**(-4) cantimeters #
w0 = 10 #[lambda0]
n_burst = 400
W = 10**5 #erg

presets = make_preset('LG', False)
delimiter = '\\'
pars = parameter_container(lambda0, n_burst, w0, W, delimiter)

loc_pulse, mass, velosity = field_core(pars, presets)


print(pars.x.shape)
plt.plot(loc_pulse.l_omega, np.abs(loc_pulse.spec_envelop.ravel()))
plt.show()
