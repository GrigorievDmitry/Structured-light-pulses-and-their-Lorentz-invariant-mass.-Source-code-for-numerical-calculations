import numpy as np
import os
from main_calculation_part import change_ref_frame as crf
from main_calculation_part import transform_field
from pulse import parameter_container

folder_suffix = '_new' #Data will be writen in the new foler with given suffix
delimiter = '\\'
f_type = 'G'

path = os.getcwd() + delimiter + 'data' + delimiter + f_type + folder_suffix + delimiter

f_name = 'fields.npy'
fields = np.load(path + f_name)
print(fields.shape)

lambda0 = 0.4# * 10**(-4) # microns# #10**(-4) cantimeters #
w0 = 10 #[lambda0]
n_burst = 400
W = 10**5 #erg

pars = parameter_container(lambda0, n_burst, w0, W, delimiter)
# mass = float(np.loadtxt(path + 'mass.txt'))