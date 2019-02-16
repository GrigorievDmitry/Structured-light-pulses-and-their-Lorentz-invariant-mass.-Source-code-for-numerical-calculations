import numpy as np
import scipy.integrate as spint
from scipy.fftpack import fftn, ifftn, fftshift, ifftshift

class pulse():
    
    c = 3 * 10**8
    def __init__(self, boundary, trvs_range, z_scale, real_type, *args):
        self.l = trvs_range
        self.n = len(trvs_range)
        self.z_scale = z_scale
        self.r_type = real_type
        
        self.lk = np.linspace(-self.n/2/(trvs_range[-1] - trvs_range[0]), self.n/2/(trvs_range[-1] - trvs_range[0]), self.n)
        self.lkz = (self.lk + self.n/2/(trvs_range[-1] - trvs_range[0]) + 10**(-200)) * self.z_scale 
        self.ky, self.kz, self.kx = np.meshgrid(self.lk, self.lkz, self.lk)
        
        self.E_bound = boundary(np.meshgrid(self.l, self.l), *args)
        self.ft_on_bound()
    
    def ft_on_bound(self):
        self.Ek_bound = [fftshift(fftn(Eb)) for Eb in self.E_bound]
        Ekz = -(self.Ek_bound[0] * self.kx + self.Ek_bound[1] * self.ky)/self.kz
        self.Ek_bound.append(Ekz)
          
    def propagate(self, spec_envelop, t, paraxial, *args):
        if not paraxial:
            kr = np.sqrt(self.kx**2 + self.ky**2 + self.kz**2)
            propagator = spec_envelop(pulse.c * kr, *args) * pulse.c * self.kz/kr * np.exp(-1j * pulse.c * kr * t)
        else:
            kr = self.kz * (1 + (self.kx**2 + self.ky**2)/2/(self.kz**2 + 10**(-200)))
            propagator = spec_envelop(pulse.c * kr, *args) * pulse.c/(1 +
                                 (self.kx**2 + self.ky**2)/2/(self.kz**2 + 10**(-200))) * np.exp(-1j * pulse.c * kr * t)
        self.Ek = [Eb * propagator for Eb in self.Ek_bound]
        
    def evolution(self):
        self.E = []
        for F in self.Ek:
            F = ifftshift(F, axes=(1,2))
            self.E.append(ifftn(F))
        self.E_sq = 0.
        for F in self.E: self.E_sq += pulse.real(F, self.r_type)**2
        self.H = []
        try:
            for F in self.Hk:
                F = ifftshift(F, axes=(1,2))
                self.H.append(ifftn(F))
            self.H_sq = 0.
            for F in self.H: self.H_sq += pulse.real(F, self.r_type)**2
            self.EH = 0.
            for i in range(3): self.EH += pulse.real(self.E[i]*self.H[i], self.r_type)
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
        
    def rescale(self, scale):
        pass



    
'''
class vec_field(np.ndarray):
    
    def __init__(self, matrixes, real_type):
        self.x = matrixes[0]
        self.y = matrixes[1]
        self.z = matrixes[2]
        self.real_type = real_type
    
    def __pow__(self, n):
        if self.real_type == 'abs':
            return abs(self.x)**n + abs(self.z)**n + abs(self.z)**n
        elif self.real_type == 'osc':
            pw_x = np.real(1./2. * (self.x + self.x.conjugate()))**n
            pw_y = np.real(1./2. * (self.y + self.y.conjugate()))**n
            pw_z = np.real(1./2. * (self.z + self.z.conjugate()))**n
            return pw_x + pw_y + pw_z
        else:
            return self.x**2 + self.y**2 + self.z**2
'''