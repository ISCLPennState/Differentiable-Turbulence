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

# %% Config %% #
with open(sys.argv[1], 'r') as stream:
    config = yaml.safe_load(stream)

if config['gpus'] != -1:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpus'])

# %% Setup %% #
density = 1.
viscosity = 1e-3
inner_steps = 128 #128
inner_steps_test = 256 #128

L = 2 * jnp.pi
max_velocity = 7.0
cfl_safety_factor = 0.5

size_DNS = 2048
grid_DNS = cfd.grids.Grid((size_DNS, size_DNS), domain=((0, L), (0, L)))

dt = cfd.equations.stable_time_step(
    max_velocity, cfl_safety_factor, viscosity, grid_DNS)

les_fac = 32
size_LES = int(size_DNS/les_fac)
grid_LES = cfd.grids.Grid((size_LES, size_LES), domain=((0, L), (0, L)))

# %% Utils %% #
def les_sim(v0, vf, viscosity, fa, fl, steps=25, inner_steps=inner_steps, convect=cfd.advection.convect_linear):
    forcing = cfd.forcings.simple_turbulence_forcing(grid_LES, fa, 4, fl)
    dt_scale = les_fac//4
    step_fn = cfd.funcutils.repeated(
        nn_sg_navier_stokes(viscosity_fn=vf, forcing=forcing, convect=convect,
            density=density, viscosity=viscosity, dt=dt*dt_scale, grid=grid_LES),
        steps=inner_steps//dt_scale)
    rollout_les = jax.jit(cfd.funcutils.trajectory(step_fn, steps, start_with_input=True))
    _, traj = rollout_les(array_to_IC(v0, grid_LES))
    return jnp.stack([traj[0].data,traj[1].data],axis=-1)

# %% Data %% #
skip = config['skip']
chunk_steps = config['rollout']
dataset = []; viscosities=[]
for j in range(len(config['data_dirs'])):
    data_dir = config['data_dirs'][j]
    visc = 1/config['flow_conf'][j][0]
    for i in range(config['num_traj']):
        data = subsample_DNS(np.load(data_dir+f'data_{i}.npy')) #filter_DNS
        dataset.extend([data[skip*i:skip*i+chunk_steps] for i in range(1+int((data.shape[0]-chunk_steps)/skip))])
        viscosities.extend([visc for i in range(1+int((data.shape[0]-chunk_steps)/skip))])
dataset = np.stack(dataset, axis=0)
viscosities = np.array(viscosities)        
print('Train data: ', dataset.shape); ds_size = dataset.shape[0]
if np.isnan(dataset).any(): print('Warning: NaN in DNS')
val_data = subsample_DNS(np.load(config['val_data'])) #filter_DNS
val_dataset = np.stack([val_data[skip*i:skip*i+chunk_steps] for i in range(1+int((val_data.shape[0]-chunk_steps)/skip))], axis=0)
print('Val data:   ', val_dataset.shape)
test_dataset = []
for i in range(config['num_traj_test']):
    test_dataset.append(subsample_DNS(np.load(config['test_dir']+f'data_{i}.npy'))) #filter_DNS
test_dataset = np.stack(test_dataset, axis=0)
print('Test data:  ', test_dataset.shape)

# %% Loss %% #
def loss(params, label, visc, fa, fl):
    dns = label
    les = les_sim(dns[0], c_func(params), viscosity=visc, fa=fa, fl=fl, steps=dataset.shape[1])
    l = jnp.mean((dns[1:]-les[1:])**2)
    return l
bloss = lambda *args: jnp.mean(jax.vmap(loss, (None, 0, 0, None, None))(*args))

@partial(jax.pmap, axis_name='device')
def val_loss(params, label, visc):
    return jax.lax.pmean(bloss(params, label, visc, 1., -.1), axis_name='device')

correlation = jax.jit(jax.vmap(lambda x,y: jnp.corrcoef(x.flatten(), y.flatten())[0,1], (0, 0)))
@partial(jax.pmap, axis_name='device')
def test_loss(params, label, visc):
    dns = label
    les = les_sim(dns[0], c_func(params), viscosity=visc, fa=1., fl=-.1, steps=test_dataset.shape[1], inner_steps=inner_steps_test)
    lcorr = jnp.sum(correlation(dns, les)>0.99)
    l = lcorr*dt*inner_steps_test/(L/(4*np.mean(dns**2)**.5))
    return jax.lax.pmean(l, axis_name='device')

gloss = jax.jit(jax.value_and_grad(jax.jit(bloss)))
@partial(jax.pmap, axis_name='device')
def update(params, opt_state, label, visc):
    l, grads = gloss(params, label, visc, 0., -0.0) #1., -.1 - for G1       0.,-0.0 for DE
    grads = jax.lax.pmean(grads, axis_name='device')
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    l = jax.lax.pmean(l, axis_name='device')
    return l, params, opt_state

# %% Wandb %% #
lr = config['lr']
epochs = config['epochs']
epoch_size = config.get('epoch_size', ds_size)
batch_size = config['batch_size']
arch, model, num_in = config['arch'], config['model'], config.get('num_in', None)

run = wandb.init(#mode='disabled',
    project='jax-turb', job_type=config['job_type'],
    config={
        'architecture': arch,
        'model': model,
        'num_in': num_in,
        'rollout': chunk_steps,
        'dataset_size': ds_size,
        'epoch_size': epoch_size,
        'learning_rate': lr,
        'epochs': epochs,
        'batch_size': batch_size,
    })
wconf = wandb.config

# %% Training %% #
load = True if config['job_type'] in {'retrain', 'resume'} else False
params, c_func = get_model(wconf.get('architecture'), 
                           wconf.get('model'), wconf.get('num_in'),
                           load=load)
filename = wconf.get('architecture')+'_'+wconf.get('model')+'_'+str(wconf.get('num_in'))#+'_G1'#+'_gauss'#
# params, c_func = jnp.array([.25]), c_smag
# filename = 'smagorinsky'

params_swa = params; count = 0
lr_sch = optax.exponential_decay(lr, epoch_size//batch_size*50, 0.5, 
                                 transition_begin=epoch_size//batch_size*10,
                                 end_value=lr/5)
optimizer = optax.adam(lr_sch)
opt_state = optimizer.init(params)

if config['job_type'] == 'resume':
    with open('opt_state/'+filename, 'rb') as fp:
        opt_state = pickle.load(fp)

params = flax.jax_utils.replicate(params)
opt_state = flax.jax_utils.replicate(opt_state)

global_step = 0
trn_l = float('nan'); val_l = float('nan'); tst_l = float('nan');
for i in range(epochs): 
    # Backprop
    idxs = np.random.permutation(
        np.arange(ds_size))[:epoch_size//batch_size*batch_size].reshape(-1, batch_size)
    losses = []
    for j in range(idxs.shape[0]):
        batch = shard(dataset[idxs[j]]); visc_batch = shard(viscosities[idxs[j]])
        l, params, opt_state = update(params, opt_state, batch, visc_batch)
        global_step += 1
        
        l = flax.jax_utils.unreplicate(l)
        losses.append(l)
        if jnp.isnan(l): raise

        if global_step%np.minimum(16,idxs.shape[0]) == 0:
            wandb.log({'global_step': global_step,
                       'epoch': i,
                       'train_loss': l})
        if global_step%np.minimum(64,idxs.shape[0]) == 0:
            idxsv = np.random.permutation(
                np.arange(val_dataset.shape[0]))[:val_dataset.shape[0]//batch_size*batch_size].reshape(-1, batch_size)
            val_losses = []
            for j in range(idxsv.shape[0]):
                val_batch = shard(val_dataset[idxsv[j]]); val_visc = viscosity*np.ones([val_batch.shape[0],val_batch.shape[1]]) #1/8.* for G1
                l = val_loss(params, val_batch, val_visc)
                val_losses.append(flax.jax_utils.unreplicate(l))
            val_l = np.mean(val_losses)
            wandb.log({'global_step': global_step,
                       'epoch': i,
                       'lr': lr_sch(global_step),
                       'val_loss': val_l})

    # SWA
    params_swa = jax.tree_util.tree_map(lambda p, pn: (p*count + pn)/(count+1), params_swa, 
                                        flax.jax_utils.unreplicate(params))
    if i>int(.8*epochs):
        count += 1

    # Test
    device_test = shard(test_dataset).swapaxes(0,1); test_visc = viscosity*np.ones(device_test.shape[1]) #1/8.* for G1
    test_losses = []
    for j in range(device_test.shape[0]):
        l = test_loss(flax.jax_utils.replicate(params_swa), device_test[j], test_visc)
        test_losses.append(flax.jax_utils.unreplicate(l))
    tst_l = np.mean(test_losses)
    wandb.log({'global_step': global_step,
               'epoch': i, 
               'test_loss': tst_l})
    
    trn_l = np.mean(losses)
    print(f'Epoch {i}: ')
    print(f'    trn loss: {trn_l}')
    print(f'    val loss: {val_l}')
    print(f'    tst loss: {tst_l}')
    # print(f'          Cs: {params_swa[0]}')

    # with open('params/'+filename, 'wb') as fp:
    #     pickle.dump(params_swa, fp)
    # with open('opt_state/'+filename, 'wb') as fp:
    #     pickle.dump(flax.jax_utils.unreplicate(opt_state), fp)
    if i==200:
        optimizer = optax.sgd(learning_rate=1e-5)
        opt_state = optimizer.init(params)
    if i>=200:
        with open(f'stat_models/params/{filename}_{i}', 'wb') as fp:
            pickle.dump(params_swa, fp)
        with open(f'stat_models/opt_state/{filename}_{i}', 'wb') as fp:
            pickle.dump(flax.jax_utils.unreplicate(opt_state), fp)

