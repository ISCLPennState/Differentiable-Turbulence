import functools
from typing import Any, Callable, Mapping, Optional

import jax
from jax_cfd.base import boundaries
from jax_cfd.base import equations
from jax_cfd.base import finite_differences
from jax_cfd.base import forcings
from jax_cfd.base import grids
from jax_cfd.base import interpolation
import numpy as np
import jax.numpy as jnp


GridArray = grids.GridArray
GridArrayVector = grids.GridArrayVector
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
InterpolationFn = interpolation.InterpolationFn
ViscosityFn = Callable[[grids.GridArrayTensor, GridVariableVector],
                       grids.GridArrayTensor]

# Surrogate model
def nn_navier_stokes(dt, viscosity_fn, forcing, **kwargs):
  def step(v):
    f = forcing(v)
    dv = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(v), viscosity_fn(v, f))
    return jax.tree_map(lambda v, dv: v + dv * dt, v, dv) 
  return step


def nn_sg_navier_stokes(dt, viscosity_fn, forcing, **kwargs):
  smagorinsky_acceleration = functools.partial(
      evm_model, dt=dt, viscosity_fn=viscosity_fn)
  if forcing is None:
    forcing = smagorinsky_acceleration
  else:
    forcing = forcings.sum_forcings(forcing, smagorinsky_acceleration)
  return equations.semi_implicit_navier_stokes(dt=dt, forcing=forcing, **kwargs)


def evm_model(
    v: GridVariableVector,
    viscosity_fn: ViscosityFn,
    interpolate_fn: InterpolationFn = interpolation.linear,
    dt: Optional[float] = None
) -> GridArrayVector:

  if not boundaries.has_all_periodic_boundary_conditions(*v):
    raise ValueError('evm_model only valid for periodic BC.')
  grid = grids.consistent_grid(*v)
  bc = boundaries.periodic_boundary_conditions(grid.ndim)
  s_ij = grids.GridArrayTensor([
      [0.5 * (finite_differences.forward_difference(v[i], j) +  # pylint: disable=g-complex-comprehension
              finite_differences.forward_difference(v[j], i))
       for j in range(grid.ndim)]
      for i in range(grid.ndim)])
  w_ij = grids.GridArrayTensor([
      [0.5 * (finite_differences.forward_difference(v[i], j) -  # pylint: disable=g-complex-comprehension
              finite_differences.forward_difference(v[j], i))
       for j in range(grid.ndim)]
      for i in range(grid.ndim)])

  def wrapped_interp_fn(c, offset, v, dt):
    return interpolate_fn(grids.GridVariable(c, bc), offset, v, dt).array
  
  s_ij_offsets = [array.offset for array in s_ij.ravel()]
  unique_offsets = list(set(s_ij_offsets))
  cell_center = grid.cell_center
  interpolate_to_center = lambda x: wrapped_interp_fn(x, cell_center, v, dt)
  centered_s_ij = np.vectorize(interpolate_to_center)(s_ij)
  centered_w_ij = np.vectorize(interpolate_to_center)(w_ij)

  mod = viscosity_fn[0](centered_s_ij, centered_w_ij)
  
  # geometric average
  cutoff = np.prod(np.array(grid.step))**(1 / grid.ndim)
  viscosity = (mod[0] * cutoff)**2 * np.sqrt(
      2 * np.trace(centered_s_ij.dot(centered_s_ij)))
  viscosities_dict = {
      offset: wrapped_interp_fn(viscosity, offset, v, dt).data
      for offset in unique_offsets}
  viscosities = [viscosities_dict[offset] for offset in s_ij_offsets]
  viscosity = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(s_ij), viscosities)
  
  bases = viscosity_fn[1]
  ml_stresses = [coeff_tensor_mult(mod[1][i], centered_s_ij, centered_w_ij,
                                   bases[i],
                                   lambda x, o: wrapped_interp_fn(x, o, v, dt), s_ij) 
                                   for i in range(len(bases))]
  ml_tau = jax.tree_map(lambda *x: sum(x), *ml_stresses)

  tau = jax.tree_map(lambda x, y, z: -2. * x * y + z, viscosity, s_ij, ml_tau)
  return tuple(-finite_differences.divergence(  # pylint: disable=g-complex-comprehension
      tuple(grids.GridVariable(t, bc)  # use velocity bc to compute diverence
            for t in tau[i, :]))
               for i in range(grid.ndim))

def coeff_tensor_mult(c, s_ij, w_ij, basis_fn, interp_fn, offset_tensor):
  comps = [c*a for a in (basis_fn(s_ij, w_ij)).ravel()]
  offsets = [array.offset for array in offset_tensor.ravel()]
  icomps = [
    interp_fn(comps[i], offsets[i]).data for i in range(len(offsets))]
  tensor = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(offset_tensor), icomps)
  return tensor


def evm_offline(
    v: GridVariableVector,
    viscosity_fn: ViscosityFn,
    interpolate_fn: InterpolationFn = interpolation.linear,
    dt: Optional[float] = None
):

  if not boundaries.has_all_periodic_boundary_conditions(*v):
    raise ValueError('evm_model only valid for periodic BC.')
  grid = grids.consistent_grid(*v)
  bc = boundaries.periodic_boundary_conditions(grid.ndim)
  s_ij = grids.GridArrayTensor([
      [0.5 * (finite_differences.forward_difference(v[i], j) +  # pylint: disable=g-complex-comprehension
              finite_differences.forward_difference(v[j], i))
       for j in range(grid.ndim)]
      for i in range(grid.ndim)])
  w_ij = grids.GridArrayTensor([
      [0.5 * (finite_differences.forward_difference(v[i], j) -  # pylint: disable=g-complex-comprehension
              finite_differences.forward_difference(v[j], i))
       for j in range(grid.ndim)]
      for i in range(grid.ndim)])

  def wrapped_interp_fn(c, offset, v, dt):
    return interpolate_fn(grids.GridVariable(c, bc), offset, v, dt).array
  
  s_ij_offsets = [array.offset for array in s_ij.ravel()]
  unique_offsets = list(set(s_ij_offsets))
  cell_center = grid.cell_center
  interpolate_to_center = lambda x: wrapped_interp_fn(x, cell_center, v, dt)
  centered_s_ij = np.vectorize(interpolate_to_center)(s_ij)
  centered_w_ij = np.vectorize(interpolate_to_center)(w_ij)

  mod = viscosity_fn[0](centered_s_ij, centered_w_ij)
  
  # geometric average
  cutoff = np.prod(np.array(grid.step))**(1 / grid.ndim)
  viscosity = (mod[0] * cutoff)**2 * np.sqrt(
      2 * np.trace(centered_s_ij.dot(centered_s_ij)))
  viscosities_dict = {
      offset: wrapped_interp_fn(viscosity, offset, v, dt).data
      for offset in unique_offsets}
  viscosities = [viscosities_dict[offset] for offset in s_ij_offsets]
  viscosity = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(s_ij), viscosities)
  
  bases = viscosity_fn[1]
  ml_stresses = [coeff_tensor_mult(mod[1][i], centered_s_ij, centered_w_ij,
                                   bases[i],
                                   lambda x, o: wrapped_interp_fn(x, o, v, dt), s_ij) 
                                   for i in range(len(bases))]
  ml_tau = jax.tree_map(lambda *x: sum(x), *ml_stresses)

  tau = jax.tree_map(lambda x, y, z: -2. * x * y + z, viscosity, s_ij, ml_tau)
  vgt = jnp.stack(jax.tree_util.tree_flatten(centered_s_ij+centered_w_ij)[0], axis=-1)
  tau = jnp.stack(jax.tree_util.tree_flatten(np.vectorize(interpolate_to_center)(tau))[0], axis=-1)
  return vgt, tau