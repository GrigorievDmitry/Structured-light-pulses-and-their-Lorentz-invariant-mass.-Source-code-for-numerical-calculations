import os
import numpy as np

def save_result(result, name, delimiter, f_type, number=''):
    path = os.getcwd() + delimiter + 'data' + delimiter + f_type
    if not os.path.exists(path):
        os.makedirs(path)
    f_name = delimiter + name + str(number) + '.npy'
    np.save(path + f_name, result)

def save_environment(x, t, z, mass, delimiter, f_type):
    fold = os.getcwd() + delimiter + 'data' + delimiter + f_type
    if not os.path.exists(fold):
            os.makedirs(fold)
    
    np.savetxt(fold + delimiter + 'type.txt', [f_type], '%s')
    file = fold + delimiter + 'space.npy'
    np.save(file, x)
    file = fold + delimiter + 't_scale.npy'
    np.save(file, t)
    file = fold + delimiter + 'z_range.npy'
    np.save(file, np.array([0, z]))
    np.savetxt(fold + delimiter + 'mass.txt', np.array([mass]), '%.3e')
