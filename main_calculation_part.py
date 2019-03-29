import numpy as np
from pulse import pulse
import boundaries as bnd

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