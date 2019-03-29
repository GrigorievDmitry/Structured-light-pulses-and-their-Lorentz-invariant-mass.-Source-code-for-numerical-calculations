import numpy as np
import matplotlib.pyplot as plt
import time
import os
from pulse import pulse
import boundaries as bnd

#================================PARAMETERS====================================
w_range = np.linspace(3, 30, 50)
fig_m, axes_m = plt.subplots()
fig_v, axes_v = plt.subplots()
f_types = ['G', 'BG', 'LG', 'HG', 'AG']
Scalar = {'G':True, 'BG':False, 'LG':False, 'HG':False, 'AG':True}
linestyles = {'G':'-', 'BG':'--', 'LG':':', 'HG':'-.', 'AG':'-'}
for f_type in f_types:
    Mass = []
    Velosity = []
    for w in w_range:
        print(w)
        #===FUNDAMENTAL PARAMETRS OF A PULSE================#
        lambda0 = 0.4# * 10**(-4) # microns# #10**(-4) cantimeters #
        c = 0.299792458# * 10**(11) #Speed of light [microns/femtoseconds]
        omega0 =  2*np.pi*c/lambda0 #(10**15 seconds^(-1)#
        n_burst = 400
        tp_full = (2*np.pi/omega0)*n_burst #(femtoseconds)#  (#10**(-15) seconds#)
        w0 = w * lambda0 #(microns)# (#10**(-4) cantimeters#)
        k = 1
        W = 10**5 * 10**(2*4 - 2*15) #erg -> g*micron**2/femtosec**2
        #====CALCULATION AND PLOT SCALES ====================#
        
        #1 . FOR Boundary CONDITIONS#
        #1.1 INITIAL SCALES FOR SPATIAL BOUNDARY CONDITIONS #
        scale_x = 10*w0
        scale_y = 10*w0
        points_x = 100
        points_y = 100
        x = np.linspace(-scale_x, scale_x, points_x)
        y = np.linspace(-scale_y, scale_y, points_y)
        #1.2 INITIAL SCALES FOR TEMPORAL BOUNDARY CONDITIONS #
        tp_max = (1/2)*tp_full
        scale_t = 10*tp_full
        points_t = 100 #Number of points is choosen in accordance with spectrum detalization(quality requirements#)
        t = np.linspace(0, 2*scale_t, points_t)
        #1.3 SCALES OF Z - COORDINATE#
        scale_factor = 5 #NUMBER OF PULSE LENGTH IN Z COORDINATE#
        scale_z = scale_factor * (lambda0*n_burst)
        points_z = scale_factor * 20
        z = np.linspace(0, scale_z, points_z)
        
        enable_shift = False
        #f_type = 'G' #Pulse type ('G', 'BG', 'LG', 'HG')
        r_type = 'abs' #'abs' for sqrt(E*E.conj); 'osc' for 1/2*(F+F.conj)
        paraxial = False #Use of paraxial approximation
        scalar = Scalar[f_type] #Evaluate scalar field
        
        delimiter = '\\'
        batch_size = 100
        #==============================================================================
        
        
        
        #================================EVALUATION====================================
        t1 = time.time()
        
        loc_pulse = pulse(bnd.field, x, y, r_type, *(f_type, w0, scalar))
        loc_pulse.spatial_bound_ft()
        loc_pulse.temporal_bound_ft(bnd.temporal_envelop_sin, t, enable_shift, *(k, tp_max, omega0))
        loc_pulse.center_spectral_range(omega0)
        #loc_pulse.make_spectral_range()
        loc_pulse.define_Ekz()
        loc_pulse.magnetic()
        
        p4k = loc_pulse.momentum()
        energy, px, py, pz = [pulse.tripl_integrate(p4k[i], (loc_pulse.lkx, loc_pulse.lky, loc_pulse.l_omega)) for i in range(4)]
        N = W/energy #g*micron**2/femtosec**2
        loc_pulse.normalize_fields(N)
        
        p4k = loc_pulse.momentum()
        energy, px, py, pz = [pulse.tripl_integrate(p4k[i], (loc_pulse.lkx, loc_pulse.lky, loc_pulse.l_omega)) for i in range(4)]
        mass = (1/c**2) * np.sqrt(energy**2 - c**2*(px**2 + py**2 + pz**2))
        velosity = (1. - np.sqrt(1 - (mass**2 * c**4)/energy**2)) * 10**3
        
        Mass.append(mass)
        Velosity.append(velosity)
    
    fold = os.getcwd() + delimiter + 'data'
    file = fold + delimiter + f_type + '_m_w0.npy'
    np.save(file, Mass)
    file = fold + delimiter + f_type + '_v_w0.npy'
    np.save(file, Velosity)
    
    if f_type is not 'AG':
        axes_m.plot(w_range, Mass, linestyle=linestyles[f_type], color='black')
        axes_v.plot(w_range, Velosity, linestyle=linestyles[f_type], color='black')
    else:
        axes_m.plot(w_range, Mass, marker='.', color='black')
        axes_v.plot(w_range, Velosity, marker='.', color='black')
    axes_m.legend(f_types)
    axes_v.legend(f_types)
