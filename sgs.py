from subgrid import *
import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import fno
import pickle
#from unet import UNet

def c_smag(cs):
    cmod = lambda x, y: (cs, [0])
    return cmod, basis_fns('none')

def get_model(arch, model, num_in=None, hh=64, load=False, net_only=False):
    k = 12
    if model == 'lin':
        num_out = 2; num_in = 2 if num_in is None else num_in
    elif model == 'nonlin':
        num_out = 3; num_in = 2 if num_in is None else num_in
    elif model == 'nonlin_asym':
        num_out = 4; num_in = 2 if num_in is None else num_in
    elif model == 'mf':
        num_out = 4; num_in = 3 if num_in is None else num_in
    elif model == 'mf_sym':
        num_out = 3; num_in = 3 if num_in is None else num_in
        h = hh#16
    elif model == 'nn_only':
        num_out = 2; num_in = 4
        hh = 256; h = 256; k = 24
    
    if arch == 'mlp':
        net = nn.Sequential([
            nn.Dense(hh), nn.relu,
            nn.Dense(hh), nn.relu,
            nn.Dense(hh), nn.relu,
            nn.Dense(hh), nn.relu,
            nn.Dense(hh), nn.relu,
            nn.Dense(hh), nn.relu,
            nn.Dense(num_out) 
        ])
    elif arch == 'cnn':
        net = nn.Sequential([
            nn.Conv(hh, [3,3], padding='CIRCULAR'), nn.relu,
            nn.Conv(hh, [3,3], padding='CIRCULAR'), nn.relu,
            nn.Conv(hh, [3,3], padding='CIRCULAR'), nn.relu,
            nn.Conv(hh, [3,3], padding='CIRCULAR'), nn.relu,
            nn.Conv(hh, [3,3], padding='CIRCULAR'), nn.relu,
            nn.Conv(hh, [3,3], padding='CIRCULAR'), nn.relu,
            nn.Conv(num_out, [3,3], padding='CIRCULAR')
        ])
    elif arch == 'fno':
        net = fno.modules.FNO2D(depth=4, modes1=k, modes2=k, width=32, channels_last_proj=hh, out_channels=num_out, activation=nn.relu)
    elif arch == 'pino':
        net = fno.modules.FNO2D(depth=4, modes1=24, modes2=24, width=32, channels_last_proj=64, out_channels=num_out, activation=nn.gelu)
        # net = nn.Sequential([
        #     fno.modules.FNO2D(depth=3, modes1=30, modes2=30, width=32, channels_last_proj=64, out_channels=64, activation=nn.gelu), 
        #     nn.Conv(64, [3,3], padding='CIRCULAR'), nn.gelu,
        #     nn.Conv(num_out, [3,3], padding='CIRCULAR')
        # ])
    elif arch == 'cnn+fno':
        net = nn.Sequential([
            fno.modules.FNO2D(depth=2, modes1=10, modes2=10, width=hh//16, channels_last_proj=hh, out_channels=hh, activation=nn.relu), 
            nn.Conv(hh, [3,3], padding='CIRCULAR'), nn.relu, #was h
            nn.Conv(hh, [3,3], padding='CIRCULAR'), nn.relu, #was h
            nn.Conv(hh, [3,3], padding='CIRCULAR'), nn.relu, #was h 
            nn.Conv(num_out, [3,3], padding='CIRCULAR')
        ])
    elif arch == 'unet':
        net = UNet(num_out)

    rng = jax.random.PRNGKey(np.random.randint(10000))
    x = jnp.ones([1, 64, 64, num_in])
    p_net = net.init(rng, x)
    p_cs = jnp.array([0.18])
    params = (p_cs, p_net)

    if num_in == 2:
        def c_func(params):
            def cmod(s_ij, w_ij):
                tensors = jnp.stack([y for x in [jax.tree_util.tree_flatten(a)[0] for a in [s_ij,w_ij]] for y in x], axis=-1)
                a = tensors[:,:,0]; b = tensors[:,:,1]; c = tensors[:,:,5]
                input = jnp.stack([2*(a**2+b**2),-2*c**2], axis=-1)
                cs = params[0]
                out = net.apply(params[1], jnp.expand_dims(input,axis=0)).squeeze(0) * 1e-3
                return cs, [out[:,:,i] for i in range(num_out)]
            return cmod, basis_fns(model)
    elif num_in == 3:
        def c_func(params):
            def cmod(s_ij, w_ij):
                tensors = jnp.stack([y for x in [jax.tree_util.tree_flatten(a)[0] for a in [s_ij,w_ij]] for y in x], axis=-1)
                a = tensors[:,:,0]; b = tensors[:,:,1]; c = tensors[:,:,5]
                input = jnp.stack([a,b,c], axis=-1)
                cs = params[0]
                out = net.apply(params[1], jnp.expand_dims(input,axis=0)).squeeze(0) * 1e-3
                return cs, [out[:,:,i] for i in range(num_out)]
            return cmod, basis_fns(model)
    elif num_in == 4:
        def c_func(params):
            def cmod(v, f):
                input = jnp.stack([y for x in [jax.tree_util.tree_flatten(a)[0] for a in [v,f]] for y in x], axis=-1)
                out = net.apply(params[1], jnp.expand_dims(input,axis=0)).squeeze(0) * 1e-3
                return [out[:,:,i] for i in range(num_out)]
            return cmod
    
    if load == False:
        return params, c_func
    else:
        if load == True:
            # filename = 'params/nn/'+arch+'_'+model+'_None'
            filename = 'params/'+arch+'_'+model+'_'+str(num_in)+'_gauss'#+'_G1'# +new for other model
        else:
            filename = load
        with open(filename, 'rb') as fp:
            params = pickle.load(fp)
        if net_only:
            return params, net, num_in
        else:
            return params, c_func
        
def basis_fns(model):
  i_t     = lambda s, w: (1/3-1/2)*np.eye(2) + 0*s
  s_t     = lambda s, w: s
  w_t     = lambda s, w: w

  sw_t    = lambda s, w: s.dot(w)
  swws_t  = lambda s, w: s.dot(w)-w.dot(s)

  i_t0    = lambda s, w: np.array([[1,0],[0,1]]) + 0*s
  a_t1    = lambda s, w: np.array([[0,1],[-1,0]]) + 0*s
  st_t2   = lambda s, w: np.array([[0,1],[1,0]]) + 0*s
  st_t3   = lambda s, w: np.array([[1,0],[0,-1]]) + 0*s

  if model == 'none':
    return i_t,
  elif model == 'lin':
    return i_t, s_t
  elif model == 'nonlin':
    return i_t, s_t, swws_t
  elif model == 'nonlin_asym':
    return i_t, s_t, w_t, sw_t
  elif model == 'mf':
    return i_t0, a_t1, st_t2, st_t3
  elif model == 'mf_sym':
    return i_t0, st_t2, st_t3
  else:
    print('Unknown model')
    return
