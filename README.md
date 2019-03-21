# Light pulses calculation code.
This code provides numerical simulation of localized light pulses with arbitrary spatial and temporal envelops. Calculations are based on the approach, considered in [reference]. The work is supported by the Russian Foundation for Basic Research, grant 18-32-00906.

The following manual describes functions, modules, and objects included.
# Module 'pulse'.
This module contains function ```python save_result(result, name, delimiter, number='')```, that saves data during calculation and the main class ```python pulse```.

Class ```python pulse```:
```python
pulse(boundary, x_range, y_range, real_type='abs', *args)
```
Generates ```python pulse``` class instance from spatial boundary condition, spatial ranges.
Attributes:

Methods (methods are in-place unless otherwise indicated):
```python
spatial_bound_ft()
```
```python
temporal_bound_ft(temp_envelop, temporal_range, enable_shift, *args)
```
```python
make_spectral_range()
```
```python
center_spectral_range(omega0)
```
```python
define_Ekz()
```
```python
set_spec_envelop(spec_envelop, spec_range, *args)
```
```python
magnetic()
```
```python
make_t_propagator(z, paraxial)
```
```python
make_ksi_propagator(z, paraxial)
```
```python
propagate()
```
```python
inverse_ft()
```
```python
momentum()
```
```python
```
```python
```
```python
```
