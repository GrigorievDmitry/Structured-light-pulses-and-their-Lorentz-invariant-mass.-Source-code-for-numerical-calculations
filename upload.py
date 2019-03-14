import numpy as np
import field_plotter as fp
import os
#================================PARAMETERS====================================

f_type = 'G' #Pulse type ('G', 'BG', 'LG', 'HG')
folder_suffix = 'pure' #Data will be writen in the new foler with given suffix
delimiter = '\\'

fold = os.getcwd() + delimiter + 'pic' + delimiter + f_type + '_' + folder_suffix
if not os.path.exists(fold):
    os.makedirs(fold)

path = os.getcwd() + delimiter + 'data' + delimiter
f_name = 'intensity.npy'
intensity = np.load(path + f_name)
f_name = 'space.npy'
x = np.load(path + f_name)
f_name = 't_scale.npy'
t_scale = np.load(path + f_name)

intensity = np.transpose(intensity, (1,0,2,3))
print(intensity.shape)

fps = 0.1
print(x.shape)
fp.plot(intensity, x, 'intensity', fold, t_scale, delimiter, mode = 'uniform')
fp.anim('intensity', fold, fps, delimiter)
