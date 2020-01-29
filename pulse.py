import numpy as np
import scipy.integrate as spint
from scipy.fftpack import fftn, ifftn, fftshift, ifftshift, ifft, fft
import os

class pulse():

    c = 0.299792458# * 10**(11) #Speed of light [microns/femtoseconds]
    def __init__(self, boundary, x_range, y_range, real_type='abs', *args):
        self.x = x_range
        self.nx = len(x_range)
        self.y = y_range
        self.ny = len(y_range)
        self.r_type = real_type
        self.freq_shift = 0
        self.E_bound = boundary(np.meshgrid(self.x, self.y), *args)
        
    def get_field(self):
        return self.E_real + self.H_real

    def spatial_bound_ft(self):
        self.lkx = 2*np.pi*np.linspace(-self.nx/2/(self.x[-1] - self.x[0]), self.nx/2/(self.x[-1] - self.x[0]), self.nx)
        self.lky = 2*np.pi*np.linspace(-self.ny/2/(self.y[-1] - self.y[0]), self.ny/2/(self.y[-1] - self.y[0]), self.ny)
        self.Ek_bound = [fftshift(ifftn(Eb)) for Eb in self.E_bound]

    def temporal_bound_ft(self, temp_envelop, temporal_range, enable_shift, *args):
        self.t = temporal_range
        self.spectral_shift = enable_shift
        self.nt = len(temporal_range)
        self.spec_envelop = fft(temp_envelop(self.t, *args)).reshape(self.nt, 1, 1)
        if self.spectral_shift:
            self.spec_envelop = fftshift(self.spec_envelop)
            self.freq_shift = -1

    def make_spectral_range(self, enable_shift=False):
        self.l_omega = 2*np.pi*np.linspace(self.freq_shift*self.nt/2/(self.t[-1] - self.t[0]), \
                        (2+self.freq_shift)*self.nt/2/(self.t[-1] - self.t[0]), self.nt)

    def center_spectral_range(self, omega0):
        self.make_spectral_range(self.spectral_shift)
        while self.l_omega.max() < omega0 or self.l_omega.min() > omega0:
            if self.l_omega.max() < omega0:
                self.freq_shift += 2
                self.make_spectral_range()
            elif self.l_omega.min() > omega0:
                self.freq_shift -= 2
                self.make_spectral_range()

    def define_Ekz(self):
        self.ky, self.omega, self.kx = np.meshgrid(self.lky, self.l_omega, self.lkx)
        self.kz = np.sqrt(self.omega**2/pulse.c**2 - self.ky**2 - self.kx**2, dtype=np.complex128) + 10**(-200)
        self.kz = self.kz.conjugate()
        Ekz = -(self.Ek_bound[0] * self.kx + self.Ek_bound[1] * self.ky)/self.kz
        self.Ek_bound.append(Ekz)

    def set_spec_envelop(self, spec_envelop, spec_range, *args):
        self.l_omega = spec_range
        self.nt = len(spec_range)
        self.spec_envelop = spec_envelop(self.l_omega, *args).reshape(self.nt, 1, 1)
        self.t = 2*np.pi*np.linspace(0, self.nt/(self.l_omega[-1] - self.l_omega[0]), self.nt)
        self.spectral_shift = False
    
    def magnetic(self):
#        scal = (self.Ek[0]*self.kx + self.Ek[1]*self.ky + self.Ek[2]*self.kz).max()
#        if scal > 10**(10):
#            raise Exception('Non orthogonal E and k, (E,k) = %e' %scal)
        norm = pulse.c/self.omega
        Hx_bound = (self.Ek_bound[2] * self.ky - self.Ek_bound[1] * self.kz) * norm
        Hy_bound = (self.Ek_bound[0] * self.kz - self.Ek_bound[2] * self.kx) * norm
        Hz_bound = (self.Ek_bound[1] * self.kx - self.Ek_bound[0] * self.ky) * norm
        self.Hk_bound = [Hx_bound, Hy_bound, Hz_bound]

    def make_t_propagator(self, z, paraxial):
        if not paraxial:
            self.propagator = np.exp(-1j*self.kz*z)
        else:
            self.propagator = np.exp(-1j*z*self.omega/pulse.c*(1 - pulse.c**2*(self.kx*2 + self.ky**2)/2/self.omega**2))

    def make_ksi_propagator(self, z, paraxial):
        if not paraxial:
            self.propagator = np.exp(1j*z*(self.omega/pulse.c - self.kz))
        else:
            self.propagator = np.exp(1j*z*pulse.c*(self.kx*2 + self.ky**2)/2/self.omega)

    def propagate(self):
        self.Ek = [Eb * self.spec_envelop * self.propagator for Eb in self.Ek_bound]
        self.Hk = [Hb * self.spec_envelop * self.propagator for Hb in self.Hk_bound]

    def inverse_ft(self):
        self.E = []
        for F in self.Ek:
            if self.spectral_shift:
                F = ifftshift(F)
            else:
                F = ifftshift(F, axes=(1,2))
            self.E.append(ifft(fftn(F, axes=(1,2)), axis=0))
        self.E_sq = 0.
        for F in self.E: self.E_sq += (pulse.real(F, self.r_type))**2
        self.H = []
        try:
            for F in self.Hk:
                if self.spectral_shift:
                    F = ifftshift(F)
                else:
                    F = ifftshift(F, axes=(1,2))
                self.H.append(ifft(fftn(F, axes=(1,2)), axis=0))
            self.H_sq = 0.
            for F in self.H: self.H_sq += (pulse.real(F, self.r_type))**2
            self.EH = 0.
            for i in range(3): self.EH += pulse.real(self.E[i], self.r_type) * pulse.real(self.H[i], self.r_type)
        except AttributeError:
            pass
        self.E_real = [pulse.real(Eb, self.r_type) for Eb in self.E]
        self.H_real = [pulse.real(Hb, self.r_type) for Hb in self.H]
        self.S = [self.E_real[1]*self.H_real[2] - self.E_real[2]*self.H_real[1], self.E_real[2]*self.H_real[0] - self.E_real[0]*self.H_real[2], self.E_real[0]*self.H_real[1] - self.E_real[1]*self.H_real[0]]
        self.S_abs = pulse.c/4/np.pi*np.sqrt(self.S[0]**2 + self.S[1]**2 + self.S[2]**2)

    def momentum(self):
        eps = 0.
        for Eb in self.Ek_bound:
            eps += np.abs(Eb * self.spec_envelop)**2
        eps = eps * np.real(self.kz)/self.omega * pulse.c**2
        px = eps * self.kx/self.omega
        py = eps * self.ky/self.omega
        pz = eps * self.kz/self.omega
        if np.imag(eps).all() != 0 or np.imag(px).all() != 0 or np.imag(py).all() != 0 or np.imag(pz).all() != 0:
            raise ValueError('Complex energy!')
        return np.real(eps), np.real(px), np.real(py), np.real(pz)
    
    def set_filter(self, filt_function, *args):
        filt = filt_function(np.meshgrid(self.x, self.y), *args)
        filt = fftshift(ifftn(filt))
        self.propagator = self.propagator * filt
    
    def transform_fields(self, beta):
        self.beta = beta
        self.gamma = 1./np.sqrt(1 - self.beta**2)
        
        Ekx_temp = self.gamma * (self.Ek_bound[0] + self.beta * self.Hk_bound[1])
        Eky_temp = self.gamma * (self.Ek_bound[1] - self.beta * self.Hk_bound[0])
        
        self.Hk_bound[0] = self.gamma * (self.Hk_bound[0] - self.beta * self.Ek_bound[1])
        self.Hk_bound[1] = self.gamma * (self.Hk_bound[1] + self.beta * self.Ek_bound[0])
        
        self.Ek_bound[0] = Ekx_temp
        self.Ek_bound[1] = Eky_temp
    
    def t_factor(self, t):
        factor = np.exp(1j * self.gamma * 2 * self.l_omega * t)
        return factor.reshape(self.nt, 1, 1)
    
    def transformed_ift(self, T):
        self.E = []
        for F in self.Ek:
            if self.spectral_shift:
                F = ifftshift(F)
            else:
                F = ifftshift(F, axes=(1,2))
            E = fftn(F, axes=(1,2))
            Et = []
            for t in T:
                factor = self.t_factor(t)
                Et.append(factor * E)
            Et = np.array(Et)
            E = spint.simps(Et, self.l_omega, axis=1)
            self.E.append(E)
        self.E_sq = 0.
        for F in self.E: self.E_sq += (pulse.real(F, self.r_type))**2
        self.H = []
        try:
            for F in self.Hk:
                if self.spectral_shift:
                    F = ifftshift(F)
                else:
                    F = ifftshift(F, axes=(1,2))
                E = fftn(F, axes=(1,2))
                Et = []
                for t in T:
                    factor = self.t_factor(t)
                    Et.append(factor * E)
                Et = np.array(Et)
                E = spint.simps(Et, self.l_omega, axis=1)
                self.H.append(E)
            self.H_sq = 0.
            for F in self.H: self.H_sq += (pulse.real(F, self.r_type))**2
            self.EH = 0.
            for i in range(3): self.EH += pulse.real(self.E[i], self.r_type) * pulse.real(self.H[i], self.r_type)
        except AttributeError:
            pass
        self.E_real = [pulse.real(Eb, self.r_type) for Eb in self.E]
        self.H_real = [pulse.real(Hb, self.r_type) for Hb in self.H]
        self.S = [self.E_real[1]*self.H_real[2] - self.E_real[2]*self.H_real[1], self.E_real[2]*self.H_real[0] - self.E_real[0]*self.H_real[2], self.E_real[0]*self.H_real[1] - self.E_real[1]*self.H_real[0]]
        self.S_abs = pulse.c/4/np.pi*np.sqrt(self.S[0]**2 + self.S[1]**2 + self.S[2]**2)
    
    def make_transformed_propagator(self, z):
        self.propagator = np.exp(-1j*self.gamma*z * (self.kz - self.beta*self.omega/pulse.c))
    
    def normalize_fields(self, N):
        for i in range(3):
            self.Ek_bound[i] = self.Ek_bound[i] * np.sqrt(N)
            self.Hk_bound[i] = self.Hk_bound[i] * np.sqrt(N)

    @staticmethod
    def tripl_integrate(M, l):
        return spint.simps(spint.simps(spint.simps(M, l[0]), l[1]), l[2])

    @staticmethod
    def real(F, r_type):
        if r_type == 'abs':
            return np.real(F)
        elif r_type == 'osc':
            return np.real(1./2. * (F + F.conjugate()))        

class parameter_container():
    
    def __init__(self, lambda0, n_burst, w0, W, delimiter):
        self.lambda0 = lambda0
        self.omega0 =  2*np.pi*pulse.c/self.lambda0 #(10**15 seconds^(-1)#
        self.n_burst = n_burst
        self.w0 = w0 * self.lambda0 #(microns)# (#10**(-4) cantimeters#)
        self.W = W * 10**(2*4 - 2*15) #erg -> g*micron**2/femtosec**2
        
        pars = parse_config(delimiter)
        #====CALCULATION AND PLOT SCALES ====================#
        
        #1 . FOR Boundary CONDITIONS#
        #1.1 INITIAL SCALES FOR SPATIAL BOUNDARY CONDITIONS #
        scale_x = pars['scale_x'] * self.w0
        scale_y = pars['scale_y'] * self.w0
        points_x = pars['points_x']
        points_y = pars['points_y']
        self.x = np.linspace(-scale_x, scale_x, points_x)
        self.y = np.linspace(-scale_y, scale_y, points_y)
        #1.2 INITIAL SCALES FOR TEMPORAL BOUNDARY CONDITIONS #
        self.tp_full = (2*np.pi/self.omega0) * self.n_burst #(femtoseconds)#  (#10**(-15) seconds#)
        self.tp_max = self.tp_full/2.
        scale_t = pars['scale_t'] * self.tp_full
        points_t = pars['points_t'] #Number of points is choosen in accordance with spectrum detalization(quality requirements#)
        self.t = np.linspace(0, 2*scale_t, points_t)
        #1.3 SCALES OF Z - COORDINATE#
        scale_factor = pars['scale_factor'] #NUMBER OF PULSE LENGTH IN Z COORDINATE#
        scale_z = scale_factor * (self.lambda0 * self.n_burst)
        points_z = pars['points_z']
        self.z = np.linspace(0, scale_z, points_z)
        self.batch_size = pars['batch_size']
        
    def get_parameters_list(self):
        print('lambda0 omega0 n_burst w0 W \nx y z t tp_full tp_max batch_size')

def parse_config(delimiter, name='config.txt'):
    file = os.getcwd() + delimiter + name
    par_names = ('scale_x', 'scale_y', 'points_x', 'points_y', 'scale_t', 'points_t', \
                 'scale_factor', 'points_z', 'batch_size')
    pars = {}
    with open(file, 'r') as f:
        for (i, line) in enumerate(f):
            val = line.split('=')[1].strip()
            pars[par_names[i]] = int(val)
    return pars