# Light pulses calculation code.
This code provides numerical simulation of localized light pulses with arbitrary spatial and temporal envelops. Calculations are based on the approach, considered in [reference]. The work is supported by the Russian Foundation for Basic Research, grant 18-32-00906.

The following manual describes functions, modules, and objects included.
# Module 'pulse'.
This module contains function `save_result(result, name, delimiter, number='')`, that saves data during calculation and the main class `pulse`.

Class `pulse`:
```python
pulse(boundary, x_range, y_range, real_type='abs', *args)
```
Generates `pulse` class instance from spatial boundary condition, spatial ranges and defines the way of real field getting from complex amplitude.

Attributes:


Methods (methods are in-place unless otherwise indicated):
```python
spatial_bound_ft()
```
Provides FFT transformation of spatial boundary conditions.
```python
temporal_bound_ft(temp_envelop, temporal_range, enable_shift, *args)
```
Defines temporal boundary conditions and provides their FFT transformation.
```python
make_spectral_range()
```
Makes spectral range from temporal range.
```python
center_spectral_range(omega0)
```
Chooses the right frequency interval in accordance with Nyquist–Shannon sampling theorem.
```python
define_Ekz()
```
Defines `z`-component of electric field from the zero divergence condition.
```python
set_spec_envelop(spec_envelop, spec_range, *args)
```
Allows to define spectral envelope manualy.
```python
magnetic()
```
Calculates magnetic field on boundary.
```python
make_t_propagator(z, paraxial)
```
Defines transfer function with respect to variables `z` and `t`.
```python
make_ksi_propagator(z, paraxial)
```
Defines transfer function with respect to variables `z - ct` and `t`.
```python
propagate()
```
Calculates fields with given transfer function (at the defined `z` point) in frequency domain.
```python
inverse_ft()
```
Transforms fields to space-time domain.
```python
momentum()
```
Returns 4-momentum vector (NOT in-place method).
```python
set_filter(filt_function, *args)
```
Calculates phase mask of an arbitrary filter and transforms transfer function respectively.
```python
normalize_fields(N)
```
Applies normalization on all fields.
```python
tripl_integrate(M, l)
```
Static method for computing of triple integrals.
```python
real(F, r_type)
```
Static method that takes the real part of complex amplitude.
