import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, assoc_laguerre, eval_hermite
from pulse import pulse
import field_plotter as fp
import os
#================================PARAMETERS====================================
c = 3 * 10**8 #Speed of light
n = 50 #Spatial grid steps
lambda0 = 404 * 10**(-9) #Wavelength
w0 = 10 * lambda0 #Waist
t_scale =  w0/c
l0 = 10. #Transverse window size in [w0]
l = np.linspace(-l0*w0, l0*w0, n)
f_type = 'G' #Pulse type ('G', 'BG', 'LG', 'HG')
folder_suffix = 'pure' #Data will be writen in the new foler with given suffix
#Data structure: pic/(f_type)_(folder_suffix)/files

#================================EVALUATION====================================

fold = 'pic/' + f_type + '_' + folder_suffix
if not os.path.exists(fold):
    os.makedirs(fold)
#==============================================================================


path = os.getcwd() + '/test'
f_name = '/test.npy'
intensity = np.load(path + f_name)
intensity = np.transpose(intensity, (1,0,2,3))
print(intensity.shape)

fps = 1
fp.plot(intensity, l, 'intensity', fold, t_scale)
fp.anim('intensity', fold, fps)
