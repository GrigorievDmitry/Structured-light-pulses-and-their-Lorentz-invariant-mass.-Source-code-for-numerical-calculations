import numpy as np
from main_calculation_part import compute_mass

delimiter = '\\'
var_id = 'w0'

if var_id == 'w0':
    var = np.linspace(3, 30, 50)
elif var_id == 'n_burst':
    var = np.arange(7, 300, 4)
elif var_id == 'W':
    var = np.linspace(1, 30, 50)/2 * 10**5

compute_mass(var, var_id, delimiter)
