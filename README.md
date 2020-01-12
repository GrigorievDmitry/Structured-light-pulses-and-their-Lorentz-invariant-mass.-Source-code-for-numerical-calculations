# Light pulses calculation code.
This code provides numerical simulation of localized light pulses with arbitrary spatial and temporal envelops. Calculations are based on the approach, considered in "Stephen V Vintskevich and Dmitry A Grigoriev 2019 Laser Phys. 29 086001" https://doi.org/10.1088/1555-6611/ab1aa0. Both authors equally contributed to the result. The work is supported by the Russian Foundation for Basic Research under Grant No. 18-32-00906.

# Abstract
The Lorentz-invariant mass and mean propagation speed have been found for structured light pulses in a vacuum considered as relativistic objects. We have solved the boundary problem for such widely known field configurations as Gauss, Laguerre–Gauss, Bessel–Gauss, Hermite–Gauss and Airy–Gauss. The pulses were taken as having a temporal envelope of finite duration. We discovered that Lorentz-invariant mass and mean propagation speed significantly depend on the spatial–temporal structure of pulses. We found that mean propagation speed is independent of the full energy of the pulse and is less than the speed of light.

# The following manual describes functions, modules, and objects included.
## Module 'pulse'.
This module contains function `save_result`, that saves data during calculation and the main class `pulse`.

Class `pulse`:
```python
pulse(boundary, x_range, y_range, real_type='abs', *args)
```
Generates `pulse` class instance from spatial boundary condition, spatial ranges and defines the way of real field getting from complex amplitude.

Attributes:


Methods (methods are in-place unless otherwise indicated):
* Provides FFT transformation of spatial boundary conditions.
```python
spatial_bound_ft()
```
* Defines temporal boundary conditions and provides their FFT transformation.
```python
temporal_bound_ft(temp_envelop, temporal_range, enable_shift, *args)
```
* Makes spectral range from temporal range.
```python
make_spectral_range()
```
* Chooses the right frequency interval in accordance with Nyquist–Shannon sampling theorem.
```python
center_spectral_range(omega0)
```
* Chooses the right frequency interval in accordance with Nyquist–Shannon sampling theorem.
```python
define_Ekz()
```
* Allows to define spectral envelope manualy.
```python
set_spec_envelop(spec_envelop, spec_range, *args)
```
* Calculates magnetic field on boundary.
```python
magnetic()
```
* Defines transfer function with respect to variables `z` and `t`.
```python
make_t_propagator(z, paraxial)
```
* Defines transfer function with respect to variables `z - ct` and `t`.
```python
make_ksi_propagator(z, paraxial)
```
* Calculates fields with given transfer function (at the defined `z` point) in frequency domain.
```python
propagate()
```
* Transforms fields to space-time domain.
```python
inverse_ft()
```
* Returns 4-momentum vector (NOT in-place method).
```python
momentum()
```
* Calculates phase mask of an arbitrary filter and transforms transfer function respectively.
```python
set_filter(filt_function, *args)
```
* Applies normalization on all fields.
```python
normalize_fields(N)
```
* Static method for computing of triple integrals.
```python
tripl_integrate(M, l)
```
* Static method that takes the real part of complex amplitude.
```python
real(F, r_type)
```
