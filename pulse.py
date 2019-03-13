import numpy as np
import scipy.integrate as spint
from scipy.fftpack import fftn, ifftn, fftshift, ifftshift, ifft, fft
import os

def save_result(result):
    path = os.getcwd() + '/test'
    if not os.path.exists(path):
        os.makedirs(path)
    f_name = '/test.npy'
    np.save(path + f_name, result)

class pulse():

    c = 3 * 10**8
    def __init__(self, boundary, trvs_range, real_type='abs', *args):
        self.l = trvs_range
        self.n = len(trvs_range)
        self.r_type = real_type
        self.E_bound = boundary(np.meshgrid(self.l, self.l), *args)
        #self.E_bound = boundary(self.l)

    def spatial_bound_ft(self):
        self.lk = 2*np.pi*np.linspace(-self.n/2/(self.l[-1] - self.l[0]), self.n/2/(self.l[-1] - self.l[0]), self.n)
        self.Ek_bound = [fftshift(ifftn(Eb)) for Eb in self.E_bound]
        #self.Ek_bound = fftshift(fftn(self.E_bound))

    def temporal_bound_ft(self, temp_envelop, temporal_range, *args):
        self.t = temporal_range
        self.nt = len(temporal_range)
        self.spec_envelop = fft(temp_envelop(self.t, *args)).reshape(self.nt, 1, 1)
        self.l_omega = 2*np.pi*np.linspace(4*self.nt/2/(temporal_range[-1] - temporal_range[0]), 6*self.nt/2/(temporal_range[-1] - temporal_range[0]), self.nt)

    def define_Ekz(self):
        self.ky, self.omega, self.kx = np.meshgrid(self.lk, self.l_omega, self.lk)
        self.kz = np.sqrt(self.omega**2/pulse.c**2 - self.ky**2 - self.kx**2, dtype=np.complex128) + 10**(-200)
        self.kz = self.kz.conjugate()
        Ekz = -(self.Ek_bound[0] * self.kx + self.Ek_bound[1] * self.ky)/self.kz
        self.Ek_bound.append(Ekz)

    def set_spec_envelop(self, spec_envelop, spec_range):
        self.spec_envelop = spec_envelop
        self.l_omega = spec_range
        self.nt = len(spec_range)
        self.t = 2*np.pi*np.linspace(0, self.nt/(spec_range[-1] - spec_range[0]), self.nt)

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

    def evolution(self):
        self.E = []
        for F in self.Ek:
            F = ifftshift(F, axes=(1,2))
            self.E.append(ifft(fftn(F, axes=(1,2)), axis=0))
        self.E_sq = 0.
        for F in self.E: self.E_sq += (pulse.real(F, self.r_type))**2
        self.H = []
        try:
            for F in self.Hk:
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

    def magnetic(self):
#        scal = (self.Ek[0]*self.kx + self.Ek[1]*self.ky + self.Ek[2]*self.kz).max()
#        if scal > 10**(10):
#            raise Exception('Non orthogonal E and k, (E,k) = %e' %scal)
        norm = 1./self.omega
        Hx = (self.Ek[2] * self.ky - self.Ek[1] * self.kz) * norm
        Hy = (self.Ek[0] * self.kz - self.Ek[2] * self.kx) * norm
        Hz = (self.Ek[1] * self.kx - self.Ek[0] * self.ky) * norm
        self.Hk = [Hx, Hy, Hz]

    def momentum(self):
        eps = pulse.real(self.Ek[0], self.r_type)**2 + pulse.real(self.Ek[1], self.r_type)**2 + pulse.real(self.Ek[2], self.r_type)**2
        r = self.omega
        px = eps * self.kx/r
        py = eps * self.ky/r
        pz = eps * self.kz/r
        return eps, px, py, pz

    @staticmethod
    def tripl_integrate(M, l):
        return spint.simps(spint.simps(spint.simps(M, l[0]), l[1]), l[2])

    @staticmethod
    def real(F, r_type):
        if r_type == 'abs':
            return np.real(F)
        elif r_type == 'osc':
            return np.real(1./2. * (F + F.conjugate()))
