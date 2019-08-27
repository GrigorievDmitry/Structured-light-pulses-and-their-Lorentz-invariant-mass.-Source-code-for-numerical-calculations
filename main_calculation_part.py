import numpy as np
from pulse import pulse
from pulse_gpu import pulse_gpu
import boundaries as bnd
import matplotlib.pyplot as plt
import time
from pulse import parameter_container
from data_manipulation import save_mass_calc as smc

def field_core(pars, presets):
    loc_pulse = pulse(bnd.field, pars.x, pars.y, presets['r_type'], *(presets['f_type'], pars.w0, presets['scalar']))
    loc_pulse.spatial_bound_ft()
    loc_pulse.temporal_bound_ft(bnd.temporal_envelop_sin, pars.t, presets['enable_shift'], *(1., pars.tp_max, pars.omega0))
    loc_pulse.center_spectral_range(pars.omega0)
    #loc_pulse.make_spectral_range()
    loc_pulse.define_Ekz()
    loc_pulse.magnetic()
    
    p4k = loc_pulse.momentum()
    energy, px, py, pz = [pulse.tripl_integrate(p4k[i], (loc_pulse.lkx, loc_pulse.lky, loc_pulse.l_omega)) for i in range(4)]
    N = pars.W/energy #g*micron**2/femtosec**2
    loc_pulse.normalize_fields(N)
            
    p4k = loc_pulse.momentum()
    energy, px, py, pz = [pulse.tripl_integrate(p4k[i], (loc_pulse.lkx, loc_pulse.lky, loc_pulse.l_omega)) for i in range(4)]
    mass = (1/pulse.c**2) * np.sqrt(energy**2 - pulse.c**2*(px**2 + py**2 + pz**2))
    velosity = 1. - np.sqrt(1 - (mass**2 * pulse.c**4)/energy**2)
    
    return loc_pulse, mass, velosity

def make_preset(f_type, scalar, enable_shift=True, paraxial=False, r_type='abs'):
    presets = {}
    presets['enable_shift'] = enable_shift
    presets['f_type'] = f_type #Pulse type ('G', 'BG', 'LG', 'HG')
    presets['r_type'] = r_type #'abs' for sqrt(E*E.conj); 'osc' for 1/2*(F+F.conj)
    presets['paraxial'] = paraxial #Use of paraxial approximation
    presets['scalar'] = scalar #Evaluate scalar field
    return presets

def compute_mass(var, var_id, delimiter):
    inputs = set_default_inputs()
    fig_m, axes_m = plt.subplots()
    fig_v, axes_v = plt.subplots()
    f_types = ['G', 'BG', 'LG', 'HG', 'AG']
    Scalar = {'G':True, 'BG':False, 'LG':False, 'HG':False, 'AG':True}
    linestyles = {'G':'-', 'BG':'--', 'LG':':', 'HG':'-.', 'AG':'-'}
    for f_type in f_types:
        Mass = []
        Velosity = []
        t1 = time.time()
        for v in var:
            print(v)
            inputs[var_id] = v
            
            presets = make_preset(f_type, Scalar[f_type])        
            pars = parameter_container(inputs['lambda0'], inputs['n_burst'], inputs['w0'], \
                                       inputs['W'], delimiter)
    
            loc_pulse, mass, velosity = field_core(pars, presets)
            Mass.append(mass)
            Velosity.append(velosity * 10**3)
        
        smc(Mass, Velosity, presets['f_type'], delimiter, var_id)
        
        if f_type is not 'AG':
            axes_m.plot(var, Mass, linestyle=linestyles[f_type], color='black')
            axes_v.plot(var, Velosity, linestyle=linestyles[f_type], color='black')
        else:
            axes_m.plot(var, Mass, marker='.', color='black')
            axes_v.plot(var, Velosity, marker='.', color='black')
        axes_m.legend(f_types)
        axes_v.legend(f_types)
        
        t2 = time.time()
        print('Exec_time: %f' %(t2-t1))

def set_default_inputs():
    default_inputs = {}
    default_inputs['lambda0'] = 0.4# * 10**(-4) # microns# #10**(-4) cantimeters #
    default_inputs['w0'] = 10 #[lambda0]
    default_inputs['n_burst'] = 400
    default_inputs['W'] = 10**5 #erg
    return default_inputs
    