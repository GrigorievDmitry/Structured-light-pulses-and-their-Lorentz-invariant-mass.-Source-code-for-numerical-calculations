import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
import skcuda.fft as cufft
from pulse import pulse
from scipy.fftpack import fftshift

class pulse_gpu(pulse):

    def spatial_bound_ft(self):
        self.lkx = fftshift(2*np.pi*np.linspace(-self.nx/2/(self.x[-1] - self.x[0]), self.nx/2/(self.x[-1] - self.x[0]), self.nx))
        self.lky = fftshift(2*np.pi*np.linspace(-self.ny/2/(self.y[-1] - self.y[0]), self.ny/2/(self.y[-1] - self.y[0]), self.ny))
        self.Ek_bound = []
        self.plan2 = cufft.Plan((self.ny, self.nx), np.complex128, np.complex128)
        for Eb in self.E_bound:
            E_gpu = gpuarray.to_gpu(Eb)
            cufft.ifft(E_gpu, E_gpu, self.plan2)
            self.Ek_bound.append(E_gpu)
        self.plan2 = cufft.Plan((1, self.ny, self.nx), np.complex128, np.complex128)
            
    def make_spectral_range(self, enable_shift=False):
        self.l_omega = 2*np.pi*np.linspace(self.freq_shift*self.nt/2/(self.t[-1] - self.t[0]), \
                        (2+self.freq_shift)*self.nt/2/(self.t[-1] - self.t[0]), self.nt)
        if self.spectral_shift:
            self.spec_envelop = fftshift(self.l_omega)
            self.freq_shift = -1

    def temporal_bound_ft(self, temp_envelop, temporal_range, enable_shift, *args):
        self.t = temporal_range
        self.spectral_shift = enable_shift
        self.nt = len(temporal_range)
        self.plan1 = cufft.Plan(self.nt, np.complex128, np.complex128)
        self.spec_envelop = gpuarray.to_gpu(temp_envelop(self.t, *args))
        cufft.fft(self.spec_envelop, self.spec_envelop, self.plan1)
        self.spec_envelop = self.spec_envelop.reshape(self.nt, 1, 1)
        self.plan1 = cufft.Plan((self.nt, 1, 1), np.complex128, np.complex128)
        
    def propagate(self):
        prop = gpuarray.to_gpu(self.propagator)
        self.Ek = [Eb * self.spec_envelop * prop for Eb in self.Ek_bound]
        self.Hk = [Hb * self.spec_envelop * prop for Hb in self.Hk_bound]

    def inverse_ft(self):
        self.E = []
        F_host = np.empty((self.nt, self.nx, self.ny))
        for F in self.Ek:
            cufft.fft(F, F, self.plan2)
            cufft.ifft(F, F, self.plan1)
            F.get(F_host)
            self.E.append(F_host)
        self.E_sq = 0.
        for F in self.E: self.E_sq += (pulse.real(F, self.r_type))**2
        self.H = []
        try:
            for F in self.Hk:
                cufft.fft(F, F, self.plan2)
                cufft.ifft(F, F, self.plan1)
                F.get(F_host)
                self.H.append(F_host)
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
    
#    def set_filter(self, filt_function, *args):
#        filt = filt_function(np.meshgrid(self.x, self.y), *args)
#        filt = fftshift(ifftn(filt))
#        self.propagator = self.propagator * filt
    
#    def transformed_ift(self, T):
#        self.E = []
#        for F in self.Ek:
#            if self.spectral_shift:
#                F = ifftshift(F)
#            else:
#                F = ifftshift(F, axes=(1,2))
#            E = fftn(F, axes=(1,2))
#            Et = []
#            for t in T:
#                factor = self.t_factor(t)
#                Et.append(factor * E)
#            Et = np.array(Et)
#            E = spint.simps(Et, self.l_omega, axis=1)
#            self.E.append(E)
#        self.E_sq = 0.
#        for F in self.E: self.E_sq += (pulse.real(F, self.r_type))**2
#        self.H = []
#        try:
#            for F in self.Hk:
#                if self.spectral_shift:
#                    F = ifftshift(F)
#                else:
#                    F = ifftshift(F, axes=(1,2))
#                E = fftn(F, axes=(1,2))
#                Et = []
#                for t in T:
#                    factor = self.t_factor(t)
#                    Et.append(factor * E)
#                Et = np.array(Et)
#                E = spint.simps(Et, self.l_omega, axis=1)
#                self.H.append(E)
#            self.H_sq = 0.
#            for F in self.H: self.H_sq += (pulse.real(F, self.r_type))**2
#            self.EH = 0.
#            for i in range(3): self.EH += pulse.real(self.E[i], self.r_type) * pulse.real(self.H[i], self.r_type)
#        except AttributeError:
#            pass
#        self.E_real = [pulse.real(Eb, self.r_type) for Eb in self.E]
#        self.H_real = [pulse.real(Hb, self.r_type) for Hb in self.H]
#        self.S = [self.E_real[1]*self.H_real[2] - self.E_real[2]*self.H_real[1], self.E_real[2]*self.H_real[0] - self.E_real[0]*self.H_real[2], self.E_real[0]*self.H_real[1] - self.E_real[1]*self.H_real[0]]
#        self.S_abs = pulse.c/4/np.pi*np.sqrt(self.S[0]**2 + self.S[1]**2 + self.S[2]**2)
