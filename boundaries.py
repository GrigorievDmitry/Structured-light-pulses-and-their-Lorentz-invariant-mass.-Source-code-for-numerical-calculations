import numpy as np
from scipy.special import jv, assoc_laguerre, eval_hermite, erf, airy
from numba import njit, prange

#Defines field boundary conditions
def field(point, name, w0, scalar=False):

    x = point[0]
    y = point[1]

    if name == 'G':
        E = np.exp(-(x**2 + y**2)/2/w0**2)
        if scalar:
            Ex = E/np.sqrt(2)
            Ey = E/np.sqrt(2)
        else:
            alpha = np.arctan2(y,x)
            Ex = - E * np.sin(alpha)
            Ey = E * np.cos(alpha)
        return Ex * field_modulation(x/w0, y/w0), Ey * field_modulation(x/w0, y/w0)

    if name == 'BG':
        beta = 1./w0
        r = beta * np.sqrt(x**2 + y**2)
        E = jv(1, r)
        Ex, Ey = field(point, 'G', w0, scalar)
        Ex = Ex * E
        Ey = Ey * E
        return Ex, Ey

    if name == 'LG':
        l = 1
        q = 1
        r = np.sqrt(x**2 + y**2)/w0
        G = assoc_laguerre(2*r**2, l, q)
        E = r**l * G * np.exp(-1j*l*np.arctan2(y,x))
        Ex, Ey = field(point, 'G', w0, scalar)
        Ex = Ex * E
        Ey = Ey * E
        return Ex, Ey

    if name == 'HG':
        l = 1
        m = 1
        E = eval_hermite(l, np.sqrt(2)*x/w0) * eval_hermite(m, np.sqrt(2)*y/w0)
        Ex, Ey = field(point, 'G', w0, scalar)
        Ex = Ex * E
        Ey = Ey * E
        return Ex, Ey
    
    if name == 'AG':
        E = airy(x/w0)[0] * airy(y/w0)[0] * np.exp(-(x/w0 + y/w0)**2/2)
        alpha = np.arctan2(y,x)
        Ex = - E * np.sin(alpha)
        Ey = E * np.cos(alpha)
        Ex = Ex * E
        Ey = Ey * E
        return Ex, Ey
    
    if name == 'AG_x':
        E = airy(x/w0)[0] * np.exp(-(x/w0 + y/w0)**2/2)
        alpha = np.arctan2(y,x)
        Ex = - E * np.sin(alpha)
        Ey = E * np.cos(alpha)
        Ex = Ex * E
        Ey = Ey * E
        return Ex, Ey

@njit(nogil=True, parallel=True)
def spec_envelop(omega_range, omega0, k, tp):
    env = np.zeros(omega_range.shape[0], dtype=np.complex128)
    for i in prange(omega_range.shape[0]):
        omega = omega_range[i]
        try:
            env[i] = 1j * tp * np.exp(1j*(omega - omega0)*k*tp) * (np.sqrt(2) * np.exp(k**2/2 - (omega - omega0)**2 * tp**2/2) *\
                    (erf(k/np.sqrt(2) + 1j*(omega - omega0)*k*tp/np.sqrt(2)) + erf(k/np.sqrt(2) - 1j*(omega - omega0)*k*tp/np.sqrt(2))) -\
                    2*k*np.sinc((omega - omega0)*k*tp))
        except Exception:
            pass
    return env
#    return np.exp(-(omega - omega0)**2/delta_omega**2)

def temporal_envelop(t, k, tp, omega0):
    x = np.empty(t.shape[0], dtype=np.complex128)
    for i in range(t.shape[0]):
        if t[i] >= 0 and t[i] <= 2*k*tp:
            x[i] = 1j*(np.exp(k**2/2 - (t[i] - k*tp)**2/2/tp**2) - 1) * np.exp(-1j*omega0*t[i])
        else:
            x[i] = 0
    return x

def temporal_envelop_sin(t, k, tp, omega0):
    x = np.empty(t.shape[0], dtype=np.complex128)
    for i in range(t.shape[0]):
        if t[i] >= 0 and t[i] <= 2*k*tp:
            x[i] = -1j*np.exp(1j*omega0*t[i])
#            x[i] = -1j*np.sin(omega0*t[i]/2)*np.exp(1j*omega0*t[i]/2)
#            x[i] = np.sin(omega0*t[i])
        else:
            x[i] = 0
    return x

#Boundary additional modulation
def field_modulation(x, y):
    return 1.
    #return np.cos(x**2 + y**2)

# def saleh_teich(x, y, z, t):
#     rho = np.sqrt(x**2 + y**2)
#     tau0 = 1./delta_omega/np.pi
#     N = omega0 * tau0
#     z0 = np.pi * w0**2/lambda0
#     rho0 = np.pi * N * w0 * z/z0
#     t_rho = t - rho**2/(2*c*z)
#     tau_rho = tau0 * np.sqrt(1 + rho**2/rho0**2)
#     I = np.exp(-2*np.pi*N * rho**2/(rho**2 + rho0**2))/(1 + rho**2/rho0**2) * \
#         np.exp(-2*t_rho**2/tau_rho**2)/(1 + t_rho**2/(np.pi*N*tau0)**2)
#     return I