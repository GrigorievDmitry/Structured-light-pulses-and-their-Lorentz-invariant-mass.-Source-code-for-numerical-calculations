import numpy as np
import os
import imageio
from main_calculation_part import change_ref_frame
from main_calculation_part import translate_coordinates
from pulse import pulse, parameter_container
import matplotlib.pyplot as plt
import matplotlib.cm as cm

folder_suffix = '_new' #Data will be writen in the new foler with given suffix
delimiter = os.sep
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
v = float(np.loadtxt(path + 'velosity.txt'))
beta = 1. - v
print(beta)
# beta = 0.

def generate_grid(pars, beta, sample_size=None):
    z = np.linspace(pars.z[0], pars.z[1]/2, 300)
    t = np.linspace(pars.t[0], pars.t[1]/2, 300)/pulse.c
    
    # lab_t, lab_z = translate_coordinates(pulse.c*t, z, -beta)
    # lab_t = lab_t/pulse.c
    
    t, z, x = np.meshgrid(t, z, pars.x)
    points = np.vstack((t.ravel(), z.ravel(), x.ravel(), np.ones(x.ravel().shape) * pars.y[50])).T
    
    if sample_size:
        ids = np.random.choice(np.arange(points.shape[0]), size=int(sample_size)).tolist()
        points = points[ids]
    
    # return lab_t, lab_z
    return points

# lab_t, lab_z = generate_grid(pars, beta)
# print(lab_t.min(), lab_t.max(), pars.t.min(), pars.t.max())
# print(lab_z.min(), lab_z.max(), pars.z.min(), pars.z.max())

points = generate_grid(pars, beta)
fields_out, points = change_ref_frame(fields, points, beta, ranges, target="cpu")
#%%
fields_out = [fields_out[i].reshape(300, 300, 100) for i in range(6)]
I = sum([fields_out[i]**2 for i in range(3)])
# I_min, I_max = I[0].min(), I[0].max()
#%%
pic_path = "pic" + os.sep + "rest_frame" + os.sep
k = 0
for i in range(0, 300, 5):
    fig = plt.figure()
    plt.imshow(I[:, i, :], interpolation='lanczos', cmap=cm.RdBu, origin='lower')
    fig.savefig(pic_path + f"{k}.png")
    k += 1
#%%
images = []
i = 0
while True:
    try:
        if i%1 ==0:
            filename = pic_path + f"{i}.png"
            images.append(imageio.imread(filename))
        i += 1
    except FileNotFoundError:
        break
imageio.mimsave(pic_path + 'rest_frame.gif', images, duration=0.1)
