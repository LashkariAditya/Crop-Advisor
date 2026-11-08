"""Deterministic NumPy weight initializers."""
import numpy as np

def he_normal(shape,fan_in,seed=7): return np.random.default_rng(seed).normal(0,np.sqrt(2/max(1,fan_in)),shape)
def xavier_uniform(shape,fan_in,fan_out,seed=7):
    limit=np.sqrt(6/max(1,fan_in+fan_out)); return np.random.default_rng(seed).uniform(-limit,limit,shape)
def zeros(shape): return np.zeros(shape,dtype=float)
