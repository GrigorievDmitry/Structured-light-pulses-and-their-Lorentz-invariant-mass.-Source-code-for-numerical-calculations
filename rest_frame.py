import numpy as np
import os
from main_calculation_part import change_ref_frame
from main_calculation_part import translate_coordinates
from pulse import pulse, parameter_container

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
ranges = [pars.t, pars.z, pars.x, pars.y]
# v = float(np.loadtxt(path + 'velosity.txt'))
# beta = 1. - v
beta = 0.5

def generate_grid(pars, beta, sample_size=None):
    t_min, z_max = translate_coordinates(pulse.c*pars.t[0], pars.z[-1], beta)
    t_max, z_min = translate_coordinates(pulse.c*pars.t[-1], pars.z[0], beta)
    z = np.linspace(z_min, z_max, len(pars.z))
    t = np.linspace(t_min, t_max, len(pars.t))/pulse.c
    x, y, t, z = np.meshgrid(pars.y, pars.x, t, z)
    points = np.vstack((t.ravel(), z.ravel(), x.ravel(), y.ravel())).T
    
    if sample_size:
        ids = np.random.choice(np.arange(points.shape[0]), size=int(sample_size)).tolist()
        points = points[ids]
    
    return points

points = generate_grid(pars, beta, sample_size=10**7)
fields_out, points = change_ref_frame(fields, points, beta, ranges)



