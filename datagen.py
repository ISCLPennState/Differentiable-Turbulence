# %%
import jax
import jaxlib
import jax.numpy as jnp
import jax_cfd.base as cfd
import jax_cfd.data as datax
from subgrid import *
from sgs import *
from utils import *
import numpy as np
import yaml
import optax
import flax.linen as nn
from functools import partial
import flax
from flax.training.common_utils import shard
import os, sys
import pickle
import wandb
from jax_cfd.base.forcings import *
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# %% Setup %% #
density = 1.
viscosity = 1 / 8000 #1000
seed = 10
inner_steps = 256

L = 2 * jnp.pi
max_velocity = 7.0
cfl_safety_factor = 0.5

size_DNS = 2048
grid_DNS = cfd.grids.Grid((size_DNS, size_DNS), domain=((0, L), (0, L)))

dt = cfd.equations.stable_time_step(
    max_velocity, cfl_safety_factor, viscosity, grid_DNS)


def simple_turbulence_forcing(
    grid: grids.Grid,
    constant_magnitude: float = 0,
    constant_wavenumber: int = 2,
    linear_coefficient: float = 0,
    swap_xy: bool = False,
    #forcing_type: str = 'kolmogorov',
) -> ForcingFn:
  linear_force = linear_forcing(grid, linear_coefficient)
  #constant_force_fn = kolmogorov_forcing(grid,swap_xy)
#   if constant_force_fn is None:
#     raise ValueError('Unknown `forcing_type`. '
#                      f'Expected one of {list(FORCING_FUNCTIONS.keys())}; '
#                      f'got {forcing_type}.')
  constant_force = kolmogorov_forcing(grid,constant_magnitude,constant_wavenumber,swap_xy=swap_xy) #constant_force_fn(grid, constant_magnitude,constant_wavenumber)
                                     
  return sum_forcings(linear_force, constant_force)

# %% Utils %% #
def dns_sim(v0, steps=25, inner_steps=inner_steps):
    forcing = simple_turbulence_forcing(grid_DNS, 2., 8, -0.1,swap_xy=True) # 1.,4,-0.1 ,swap_xy = False
    step_fn = cfd.funcutils.repeated(
        cfd.equations.semi_implicit_navier_stokes(
            density=density, viscosity=viscosity, dt=dt, grid=grid_DNS, forcing=forcing),
        steps=inner_steps)
    rollout_dns = jax.jit(cfd.funcutils.trajectory(step_fn, steps, start_with_input=True))
    _, traj = rollout_dns(array_to_IC(v0, grid_DNS))
    traj = jax.device_put(traj)
    return np.stack([traj[0].data,traj[1].data],axis=-1)

# %% Data %% #
dns = cfd.initial_conditions.filtered_velocity_field(
    jax.random.PRNGKey(seed), grid_DNS, max_velocity/100)
dns = jnp.stack([dns[i].data for i in range(grid_DNS.ndim)], axis=-1)
dns = dns_sim(dns, steps=2, inner_steps=int(40/dt))[-1]

for i in range(32):
    print('Iter', i)
    dns = dns_sim(dns, steps=256)
    jnp.save(f'spec_dataset/G4/data_{i}', dns)
    #np.save(f'plot_data/G4/data_{i}', dns)
    dns = dns[-1]