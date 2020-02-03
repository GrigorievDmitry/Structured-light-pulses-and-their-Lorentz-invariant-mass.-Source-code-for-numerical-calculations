import numpy as np
import os
from main_calculation_part import change_ref_frame
from main_calculation_part import translate_coordinates
from pulse import pulse, parameter_container
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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
beta = 0.9

def generate_grid(pars, beta, sample_size=None):
    # t_min, z_max = translate_coordinates(pulse.c*pars.t[1], pars.z[-2], beta)
    # t_max, z_min = translate_coordinates(pulse.c*pars.t[-2], pars.z[1], beta)
    z = np.linspace(pars.z[1], pars.z[-1], 100)
    t = np.linspace(pars.t[1], pars.t[-1]/20, 100)/pulse.c
    t, z, x = np.meshgrid(t, z, pars.x)
    points = np.vstack((t.ravel(), z.ravel(), x.ravel(), np.ones(x.ravel().shape) * pars.y[50])).T
    
    if sample_size:
        ids = np.random.choice(np.arange(points.shape[0]), size=int(sample_size)).tolist()
        points = points[ids]
    
    return points

points = generate_grid(pars, beta)
fields_out, points = change_ref_frame(fields, points, beta, ranges)

fields_out = [fields_out[i].reshape(100, 100, 100) for i in range(6)]

for i in range(1, 20, 5):
    Ex_out = fields_out[2][:, i, :]
    plt.figure()
    plt.imshow(Ex_out, interpolation='lanczos', cmap=cm.RdBu, origin='lower')

