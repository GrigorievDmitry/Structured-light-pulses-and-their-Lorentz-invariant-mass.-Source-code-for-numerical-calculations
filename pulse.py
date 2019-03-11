import numpy as np
import scipy.integrate as spint
from scipy.fftpack import fftn, ifftn, fftshift, ifftshift

class pulse():

    c = 3 * 10**8
    def __init__(self, boundary, trvs_range, real_type, *args):
        self.l = trvs_range
        self.n = len(trvs_range)
        self.r_type = real_type
        self.E_bound = boundary(np.meshgrid(self.l, self.l), *args)

    def spatial_bound_ft(self):
        self.lk = np.linspace(-self.n/2/(self.l[-1] - self.l[0]), self.n/2/(self.l[-1] - self.l[0]), self.n)
        self.Ek_bound = [fftshift(fftn(Eb)) for Eb in self.E_bound]

    def temporal_bound_ft(self, temp_envelop, temporal_range, *args):
        self.t = temporal_range
        self.nt = len(temporal_range)
        self.spec_envelop = ifft(temp_envelop(self.t, *args))
        self.l_omega = np.linspace(0, self.nt/(temporal_range[-1] - temporal_range[0]), self.nt)

    def define_Ekz(self):
        self.ky, self.omega, self.kx = np.meshgrid(self.lk, self.l_omega, self.lk)
        self.kz = np.sqrt(self.omega**2/pulse.c**2 - self.ky**2 - self.kx**2, dtype=np.complex128)
        Ekz = -(self.Ek_bound[0] * self.kx + self.Ek_bound[1] * self.ky)/self.kz
        self.Ek_bound.append(Ekz)

    def set_spec_envelop(self, spec_envelop, spec_range):
        self.spec_envelop = spec_envelop
        self.l_omega = spec_range
        self.nt = len(spec_range)
        self.t = np.linspace(0, self.nt/(spec_range[-1] - spec_range[0]), self.nt)

    def make_t_propagator(self, paraxial):
        if not paraxial:
            self.propagator = np.exp(-1j*np.kz)
        else:
            self.propagator = np.exp(-1j*self.omega/pulse.c*(1 - pulse.c**2*(self.kx*2 + self.ky**2)/2/self.omega**2)

    def make_ksi_propagator(self, paraxial):
        if not paraxial:
            self.propagator = np.exp(1j*(self.omega/pulse.c - self.kz)
        else:
            self.propagator = np.exp(1j*pulse.c*(self.kx*2 + self.ky**2)/2/self.omega)

    def propagate(self, z):
        self.Ek = [Eb * self.propagator**z for Eb in self.Ek_bound]

    def evolution(self):
        self.E = []
        for F in self.Ek:
            F = ifftshift(F, axes=(1,2))
            self.E.append(fft(ifftn(F, axes=(1,2)), axis=0))
        self.E_sq = 0.
        for F in self.E: self.E_sq += pulse.real(F, self.r_type)**2
        self.H = []
        try:
            for F in self.Hk:
                F = ifftshift(F, axes=(1,2))
                self.H.append(fft(ifftn(F, axes=(1,2)), axis=0))
            self.H_sq = 0.
            for F in self.H: self.H_sq += pulse.real(F, self.r_type)**2
            self.EH = 0.
            for i in range(3): self.EH += pulse.real(self.E[i], self.r_type) * pulse.real(self.H[i], self.r_type)
        except AttributeError:
            pass

    def magnetic(self):
#        scal = (self.Ek[0]*self.kx + self.Ek[1]*self.ky + self.Ek[2]*self.kz).max()
#        if scal > 10**(10):
#            raise Exception('Non orthogonal E and k, (E,k) = %e' %scal)
        norm = 1./np.sqrt(self.kx**2 + self.ky**2 + self.kz**2)
        Hx = (self.Ek[2] * self.ky - self.Ek[1] * self.kz) * norm
        Hy = (self.Ek[0] * self.kz - self.Ek[2] * self.kx) * norm
        Hz = (self.Ek[1] * self.kx - self.Ek[0] * self.ky) * norm
        self.Hk = [Hx, Hy, Hz]

    def momentum(self):
        eps = pulse.real(self.Ek[0], self.r_type)**2 + pulse.real(self.Ek[1], self.r_type)**2 + pulse.real(self.Ek[2], self.r_type)**2
        r = np.sqrt(self.kx**2 + self.ky**2 + self.kz**2)
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
            return abs(F)
        elif r_type == 'osc':
            return np.real(1./2. * (F + F.conjugate()))
