import numpy as np
from pulse import pulse
from pulse_gpu import pulse_gpu
import boundaries as bnd
import matplotlib.pyplot as plt
import time
import numba as nb
from numba import cuda, njit, prange
from pulse import parameter_container
from data_manipulation import save_mass_calc as smc
from functools import reduce
from tqdm import tqdm
import pandas as pd

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


@cuda.jit(device=True)
def moveaxis(mtx, axis):
    if axis == 0:
        return (mtx[0,:], mtx[1,:], mtx[2,:])
    else:
        return (mtx[:,0], mtx[:,1], mtx[:,2])


@cuda.jit(device=True) 
def _round(x):
    if (x - int(x)) > 0.5:
        return int(x) + 1
    else:
        return int(x)


@cuda.jit
def interpolate_kernel(field, points, steps, zero, field_out):
    n = cuda.grid(1)
    
    if n < len(points):
        point = points[n]
        nearest_idx = cuda.local.array(2, dtype=nb.int8)
        grads = cuda.local.array(3, dtype=nb.float32)
        hess = cuda.local.array((2, 2), dtype=nb.float32)
        offset = cuda.local.array(2, dtype=nb.float32)
        
        for i in range(2):
            nearest_idx[i] = _round((point[i] - zero[i])/steps[i])
        
        calc_grid = field[
                    nearest_idx[0]-1:nearest_idx[0]+2,
                    nearest_idx[1]-1:nearest_idx[1]+2,
                    _round((point[2] - zero[2])/steps[2]),
                ]
        
        correction = 0
        for i in range(2):
            grad_grid = moveaxis(calc_grid, i)
            for q in range(3):
                grads[q] = ((grad_grid[2][q] - 2*grad_grid[1][q] +
                                     grad_grid[0][q])/2/steps[i])
            
            for j in range(2):
                if i == j:
                    hess[i, j] = grads[1]*2/steps[j]
                else:
                    hess[i, j] = (grads[2] - 2*grads[1] + grads[0])/2/steps[j]
        
            offset[i] = point[i] - nearest_idx[i] * steps[i]
            
            correction += grads[1] * offset[i]
            order2_temp = 0
            for j in range(2):
                order2_temp += hess[i, j] * offset[j]
            correction += 0.5 * order2_temp * offset[i]
        
        field_out[n] = calc_grid[(1, 1)] + correction


def crate_batch(field, points, steps, zero, axis=3):
    ids = []
    slices = []
    points = pd.DataFrame(points, columns=[0, 1, 2, 3])
    zeros = points[axis].unique()
    groups = points.groupby(axis).groups
    for coord in zeros:
        ids.append(groups[coord])
        slices.append(int((coord - zero[axis])/steps[axis]))
    
    return ids, zeros, slices
    


def interpolate(field, points, steps, zero):
    print("Interpolating {} points".format(len(points)))
    field_out = np.empty(len(points), dtype=np.float32)
    axis = 3
    ids, zeros, slices = crate_batch(field, points, steps, zero, axis=axis)
    field = np.moveaxis(field, axis, 0)
    with cuda.gpus[0]:
        for i in range(len(ids)):
            zero[axis] = zeros[i]
            # field_out_gpu = cuda.device_array(len(ids[i]), dtype=np.float32)
            # nblocks = len(ids[i])//512 + 1
            # interpolate_kernel[nblocks, 512](np.ascontiguousarray(np.moveaxis(field, axis, 0)[slices[i]]),
            #                                  np.ascontiguousarray(points[ids[i]]),
            #                                  steps, zero, field_out_gpu)
            # field_out[ids[i]] = field_out_gpu.copy_to_host()
            field_out[ids[i]] = interpolate_kernel_cpu(field[slices[i]],
                                              points[ids[i]], steps, zero)
    
    return field_out


@njit(parallel=True, nogil=True)
def interpolate_kernel_cpu(field, points, steps, zero):
    field_out = np.empty(len(points), dtype=np.float32)
    for n in prange(len(points)):
        point = points[n]
        nearest_idx = np.empty(2, dtype=np.int8)
        grads = np.empty(3, dtype=np.float32)
        hess = np.empty((2, 2), dtype=np.float32)
        offset = np.empty(2, dtype=np.float32)
        
        for i in range(2):
            nearest_idx[i] = round((point[i] - zero[i])/steps[i])
        
        calc_grid = field[
                    nearest_idx[0]-1:nearest_idx[0]+2,
                    nearest_idx[1]-1:nearest_idx[1]+2,
                    round((point[2] - zero[2])/steps[2]),
                ]
        
        correction = 0
        for i in range(2):
            if i == 0:
                grad_grid = (calc_grid[0,:], calc_grid[1,:], calc_grid[2,:])
            else:
                grad_grid = (calc_grid[:,0], calc_grid[:,1], calc_grid[:,2])
            
            for q in range(3):
                grads[q] = ((grad_grid[2][q] - 2*grad_grid[1][q] +
                                     grad_grid[0][q])/2/steps[i])
            
            for j in range(2):
                if i == j:
                    hess[i, j] = grads[1]*2/steps[j]
                else:
                    hess[i, j] = (grads[2] - 2*grads[1] + grads[0])/2/steps[j]
        
            offset[i] = point[i] - nearest_idx[i] * steps[i]
            
            correction += grads[1] * offset[i]
            order2_temp = 0
            for j in range(2):
                order2_temp += hess[i, j] * offset[j]
            correction += 0.5 * order2_temp * offset[i]
        
        field_out[n] = calc_grid[(1, 1)] + correction
    


class InterpolationError(Exception):
    pass


def translate_coordinates(ct, z, beta):
    print("Translating coordinates")
    gamma = 1/np.sqrt(1 - beta**2)
    return gamma*ct - beta*gamma*z, gamma*z - beta*gamma*ct


@njit(parallel=True, nogil=True)
def transform_field(fields, beta):
    print("Transforming fields")
    n = len(fields[0])
    fields_transformed = np.zeros((6, n))
    gamma = 1/np.sqrt(1 - beta**2)
    jacobian = np.array(
                    [[gamma, -beta * gamma, 0., 0.],
                     [-beta * gamma, gamma, 0., 0.],
                     [0., 0., 1., 0.],
                     [0., 0., 0., 1.]]
                )
    for i in prange(n):
        f_tensor = np.array(
                    [[0., fields[0][i], fields[1][i], fields[2][i]],
                     [-fields[0][i], 0., -fields[5][i], fields[4][i]],
                     [-fields[1][i], fields[5][i], 0., -fields[3][i]],
                     [-fields[2][i], -fields[4][i], fields[3][i], 0.]]
                )
        f_tensor = jacobian @ f_tensor @ jacobian
        fields_transformed[:, i] = np.array([
                    f_tensor[0, 1], f_tensor[0, 2],
                    f_tensor[0, 3], f_tensor[3, 2],
                    f_tensor[1, 3], f_tensor[2, 1],
                ])
    
    return fields_transformed


def change_ref_frame(fields, points, beta, ranges):
    """
    Performs the field transformation to the moving reference frame.
    
    Axis order: t, z, x, y
    fields - [Ex, Ey, Ez, Hx, Hy, Hz]
    points - points in moving reference frame
    beta - v/c
    ranges - grid in the frame where fields were calculated
    """
    steps = np.array([ranges[i][1] - ranges[i][0] for i in range(4)])
    zero = np.array([ranges[i][0] for i in range(4)])
    
    ct, z = translate_coordinates(pulse.c*points[:, 0:1], points[:, 1:2], -beta)
    points_lab_frame = np.hstack((ct/pulse.c, z, np.array([points[:, 2],
                                                           points[:, 3]]).T))
    
    fields_out = []
    for field in fields:
        fields_out.append(interpolate(field, points_lab_frame, steps, zero))
    fields_out = transform_field(np.array(fields_out), beta)
    
    return fields_out, points


def test_interpolation(field, ranges):
    steps = np.array([ranges[i][1] - ranges[i][0] for i in range(4)])
    zero = np.array([ranges[i][0] for i in range(4)])
    z_mesh, x_mesh = np.meshgrid(ranges[1][1:-1] + 8, ranges[2])
    points_zx = np.vstack((z_mesh.ravel(), x_mesh.ravel())).T
    
    points = []
    for p in points_zx:
        points.append(np.array([ranges[0][1], p[0], p[1], ranges[3][50]]))
    points = np.array(points).reshape(-1, 4)
    field_out = interpolate(field, points, steps, zero)
    
    return field_out, points


def plotter(field_out, points, pars, intensity):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    field_xz = []
    gridz = []
    gridx = []
    for i, point in enumerate(points):
        if point[1] < pars.z[15]:
            field_xz.append(field_out[i])
            gridz.append(point[1])
            gridx.append(point[2])
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    z, x = np.meshgrid(pars.z[1:15], pars.x)
    z, x = z.ravel(), x.ravel()
    I = intensity[1, 1:15, :, 50].T.ravel()
    
    ax.scatter(x, z, I)
    ax.scatter(gridx, gridz, field_xz)
    
            
    
        
