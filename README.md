# Installation Instructions

## pypi
```bash
pip install lwrad
```
## editable install
```bash
git clone https://github.com/xsgeng/lwrad.git
cd lwrad
pip install --user -e .
```

# Usage
## Command Line Interface (CLI)
```bash
# help
python -m lwrad --help

# trajectories in hdf5
python -m lwrad \
--smilei \
--ek-max 1.0 \
--nek 128 \
--theta-max 0.5 \
--ntheta 128 \
--direction x \
--theta-plane xy \
--savefig \
trajecories.h5

# smilei TrackParticles
python -m lwrad \
--smilei \
--ek-max 1.0 \
--nek 128 \
--theta-max 0.5 \
--ntheta 128 \
--direction x \
--theta-plane xy \
--savefig \
smilei/result/path \
species_name

# EPOCH
# TODO
```
The `trajecories.h5` file contains `x,y,z,ux,uy,uz` in SI units.

CUDA Acceleration
```bash
python -m lwrad --backend cuda [OPTIONS] PATH NAME
```

MPI
```bash
# multithreaded MPI
srun -n 2 -c 32 -u python -m lwrad [OPTIONS] PATH NAME
# 1 thread per MPI
srun -n 64 -u python -m lwrad --backend None [OPTIONS] PATH NAME
```
## Script Example
Using trajectories output from Smilei as an example:
```python
import h5py

um = 1e-6
fs = 1e-15

l0 = 0.8*um
omega0 = 2*pi*c / l0

# Read the 0th trajectory from the h5 file
with h5py.File("tests/test.h5", "r") as f:
    # Convert coordinates xyz and time t to SI units
    x = f['x'][:, 0] / 2/pi * l0
    y = f['y'][:, 0] / 2/pi * l0
    z = f['z'][:, 0] / 2/pi * l0
    ux = f['px'][:, 0]
    uy = f['py'][:, 0]
    uz = f['pz'][:, 0]
    t = np.arange(len(ux)) * f.attrs['dt'] / 2/pi * l0/c
```

Directly compute the radiation spectrum:
```python
from lwrad import get_lw_spectrum

# Determine the angular range
gamma0 = 5.0
theta_max = pi/2/gamma0
omega_max = 4*gamma0**2 * omega0 * 3

ntheta = 256
nomega = 256

theta_axis = np.linspace(-theta_max, theta_max, ntheta)
omega_axis = np.linspace(0, omega_max, nomega) 

I = np.zeros((ntheta, nomega))

for itheta, theta in enumerate(theta_axis):
    # Direction vector
    n = [np.cos(theta), 0, np.sin(theta)]
    I[itheta, :] = get_lw_spectrum(x, y, z, ux, uy, uz, t, n, omega_axis)
```
Plotting the results:
```python
fig, ax = plt.subplots(
    1, 1,
    tight_layout = True,
    figsize=(5, 4),
)


h = ax.pcolormesh(
    omega_axis/(4*gamma0**2*omega0), theta_axis,
    I,
    cmap='jet',
    shading='gouraud',
)
fig.colorbar(h, ax=ax)

axr = ax.twinx()
axr.plot(
    omega_axis/(4*gamma0**2*omega0), 
    I.sum(axis=0),
    color='w',
)
axr.set_yticks([])

ax.set_xlabel(r'$\omega/4\gamma_0^2\omega_0$')
ax.set_ylabel(r'$\theta$')
ax.set_title(rf'$a_0$={a0},$\gamma_0$={gamma0}')
```
![](tests/thomson.png)
Comparison with Figure 2.7(a) from [Richard (2012)](https://doi.org/10.5281/zenodo.843510)

# References
Richard (2012) ([DOI: 10.5281/zenodo.843510](https://doi.org/10.5281/zenodo.843510)) (2.32)、(2.36)、(2.39)式
