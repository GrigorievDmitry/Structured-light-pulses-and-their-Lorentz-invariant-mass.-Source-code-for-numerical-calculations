import numpy as np
import field_plotter as fp
import os
#================================PARAMETERS====================================
folder_suffix = '' #Data will be writen in the new foler with given suffix
delimiter = '\\'
f_type = 'G'

path = os.getcwd() + delimiter + 'data' + delimiter + f_type + delimiter

fold = os.getcwd() + delimiter + 'pic' + delimiter + f_type + '_' + folder_suffix
if not os.path.exists(fold):
    os.makedirs(fold)

f_name = 'intensity1.npy'
intensity = np.load(path + f_name)
f_name = 'space.npy'
x = np.load(path + f_name)
f_name = 't_scale.npy'
t_scale = np.load(path + f_name)
f_name = 'z_range.npy'
z_range = np.load(path + f_name)

intensity = np.transpose(intensity, (1,0,2,3))
print(intensity.shape)

fps = 0.1
print(x.shape)
figures = fp.plot(intensity, x, 'intensity', fold, t_scale, z_range, delimiter)
#for (i, fig) in enumerate(figures):
#    filename = fold + delimiter + 'intensity' + f'_{i}.png'
#    fig.savefig(filename)
#fp.anim('intensity', fold, fps, delimiter)
