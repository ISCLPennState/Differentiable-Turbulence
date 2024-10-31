import jax
import jaxlib
import jax.numpy as jnp
import jax_cfd.base as cfd
import jax_cfd.data as datax
from subgrid import *
from sgs import *
import numpy as np
import xarray
import yaml
import optax
import flax.linen as nn
from functools import partial
import flax
from flax.training.common_utils import shard
import os, sys
import pickle
#import wandb
import matplotlib.pyplot as plt

def array_to_IC(v0, grid):
    velocity_components = []
    boundary_conditions = []
    for i in range(grid.ndim):
        velocity_components.append(v0[:,:,i])
        boundary_conditions.append(cfd.boundaries.periodic_boundary_conditions(grid.ndim))

    return cfd.initial_conditions.wrap_variables(velocity_components, grid, boundary_conditions)

def subsample_DNS(dns,size_LES=64):
    factor = dns.shape[1]//size_LES
    return dns[:,::factor,::factor,:]

def filter_DNS(dns, size_LES=64):
    filtered = []
    for i in range(dns.shape[0]):
        snap = []
        for j in range(dns.shape[-1]):
            vc = cfd.resize.downsample_staggered_velocity_component(dns[i,:,:,j],j,dns.shape[1]//size_LES)
            snap.append(vc)
        filtered.append(jnp.stack(snap, axis=-1))
    filtered = jnp.stack(filtered, axis=0)
    return filtered

def filter_DNS_cutoff(dns, size_LES=64):
    filtered = []
    #print(dns.shape)
    for j in range(dns.shape[-1]):
             vc = cfd.resize.downsample_staggered_velocity_component(dns[:,:,j],j%2,dns.shape[0]//size_LES)
             filtered.append(vc)
    filtered = jnp.stack(filtered, axis=-1)
    return filtered

def gauss_cutoff(dns, fac, L=2*jnp.pi):
    N = dns.shape[0]//fac
    width = 1*L/N

    kfreq = jnp.fft.fftfreq(N) * N
    kfreq2D = jnp.meshgrid(kfreq, kfreq)
    knrm = jnp.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
    g = jnp.expand_dims(jnp.exp(-knrm**2*width**2/24), -1)

    uh = g*jnp.fft.fftn(dns, axes=(0,1))[kfreq.astype(int),:,:][:,kfreq.astype(int),:] / fac**2
    u = jnp.fft.ifftn(uh, axes=(0,1)).real
    up = cfd.pressure.projection(array_to_IC(u, cfd.grids.Grid((N, N), domain=((0, L), (0, L)))))
    return jnp.stack([up[0].data,up[1].data],axis=-1)

def filter_gauss(dns, size_LES=64):
    gc = jax.jit(lambda dns: gauss_cutoff(dns, dns.shape[1]//size_LES))
    return jnp.stack([gc(dns[i]) for i in range(dns.shape[0])])

def espec(input):
    N = input.shape[0]
    fourier_image = jnp.fft.fftn(input, axes=(0,1))
    fourier_amplitudes = 0.5*jnp.sum(jnp.abs(fourier_image)**2, axis=-1)

    kfreq = jnp.fft.fftfreq(N) * N
    kfreq2D = jnp.meshgrid(kfreq, kfreq)
    knrm = jnp.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)

    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()

    kbins = jnp.arange(0.5, N//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])

    Abins = jnp.histogram(knrm, kbins, weights=fourier_amplitudes)[0] / jnp.histogram(knrm, kbins)[0]
    Abins *= kvals**2
    Abins /= jax.scipy.integrate.trapezoid(Abins, kvals)

    return Abins
espec = jax.vmap(jax.jit(espec))

correlation = jax.jit(jax.vmap(lambda x,y: jnp.corrcoef(x.flatten(), y.flatten())[0,1], (0, 0)))

def gen_ds(traj, grid, dt, inner_steps):
    ds = xarray.Dataset(
        {
            'u': (('time', 'x', 'y'), traj[:,:,:,0]),
            'v': (('time', 'x', 'y'), traj[:,:,:,1]),
        },
        coords={
            'x': grid.axes()[0],
            'y': grid.axes()[1],
            'time': dt * inner_steps * np.arange(traj.shape[0])
        }
    )
    return ds
def plot_traj(traj, grid, dt, inner_steps, title):
    ds = gen_ds(traj, grid, dt, inner_steps)
    def vorticity(ds):
        return (ds.v.differentiate('x') - ds.u.differentiate('y')).rename('vorticity')

    (ds.pipe(vorticity).thin(time=int(traj.shape[0]/4))
    .plot.imshow(col='time', cmap='viridis', robust=True, col_wrap=5));
    fig = plt.gcf()
    fig.suptitle(title, y=1.05, x=.415, fontsize=20)
def vortraj(traj, grid, dt, inner_steps):
    ds = gen_ds(traj, grid, dt, inner_steps)
    def vorticity(ds):
        return (ds.v.differentiate('x') - ds.u.differentiate('y')).rename('vorticity')
    return ds.pipe(vorticity).thin(time=int(traj.shape[0]/4))
