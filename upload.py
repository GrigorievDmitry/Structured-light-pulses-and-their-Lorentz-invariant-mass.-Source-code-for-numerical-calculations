import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, assoc_laguerre, eval_hermite

import os
#================================PARAMETERS====================================
n = 100 #Spatial grid steps
c = 3 * 10**8 #Speed of light
lambda0 = 404 * 10**(-9) #Wavelength
w0 = 10 * lambda0 #Waist
W = 1. #Total energy of pulse.
t_scale =  w0/c
z_scale = 10*w0
z0 = 0 #Initial timestep
Z = n #Number of timesteps
l0 = 10. #Transverse window size in [w0]
time_window_number = 1 #Number of different space scales (for different time)
l = np.linspace(-l0*w0, l0*w0, n)

f_type = 'G' #Pulse type ('G', 'BG', 'LG', 'HG')
r_type = 'abs' #'abs' for sqrt(E*E.conj); 'osc' for 1/2*(F+F.conj)
paraxial = False #Use of paraxial approximation
scalar = True #Evaluate scalar field
folder_suffix = 'pure' #Data will be writen in the new foler with given suffix
#Data structure: pic/(f_type)_(folder_suffix)/files

#================================EVALUATION====================================

fold = 'pic/' + f_type + '_' + folder_suffix
if not os.path.exists(fold):
    os.makedirs(fold)


from pulse import pulse
import field_plotter as fp

f_type = 'G' #Pulse type ('G', 'BG', 'LG', 'HG')
r_type = 'abs' #'abs' for sqrt(E*E.conj); 'osc' for 1/2*(F+F.conj)
paraxial = False #Use of paraxial approximation
scalar = True #Evaluate scalar field
folder_suffix = 'pure' #Data will be writen in the new foler with given suffix
#Data structure: pic/(f_type)_(folder_suffix)/files
#==============================================================================


path = os.getcwd() + '\\test'
f_name = '\\test.npy'
intensity = np.load(path + f_name)
intensity = np.transpose(intensity, (1,0,2,3))
print(intensity.shape)

fps = 1
fp.plot(intensity, l, 'intensity', fold, z_scale)
fp.anim('intensity', fold, fps)
