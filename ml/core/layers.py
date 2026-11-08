"""Differentiable neural network layers implemented only with NumPy."""
from __future__ import annotations
import numpy as np


class Dense:
    def __init__(self, in_features: int, out_features: int, seed: int = 7):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, np.sqrt(2 / in_features), (in_features, out_features))
        self.b = np.zeros((1, out_features)); self.dW = np.zeros_like(self.W); self.db = np.zeros_like(self.b)
    def forward(self, x): self.x = x; return x @ self.W + self.b
    def backward(self, grad): self.dW = self.x.T @ grad; self.db = grad.sum(axis=0, keepdims=True); return grad @ self.W.T
    def parameters(self): return [(self.W, self.dW), (self.b, self.db)]


class ReLU:
    def forward(self, x): self.mask = x > 0; return x * self.mask
    def backward(self, grad): return grad * self.mask
    def parameters(self): return []


class Softmax:
    def forward(self, x):
        e = np.exp(x - x.max(axis=1, keepdims=True)); self.y = e / e.sum(axis=1, keepdims=True); return self.y
    def backward(self, grad): return self.y * (grad - (grad * self.y).sum(axis=1, keepdims=True))
    def parameters(self): return []


class Dropout:
    def __init__(self, rate: float = .2, seed: int = 7):
        if not 0 <= rate < 1: raise ValueError("rate must be in [0, 1)")
        self.rate = rate; self.rng = np.random.default_rng(seed)
    def forward(self, x, training: bool = True):
        self.mask = (self.rng.random(x.shape) >= self.rate) / (1-self.rate) if training and self.rate else np.ones_like(x)
        return x * self.mask
    def backward(self, grad): return grad * self.mask
    def parameters(self): return []


class BatchNorm:
    def __init__(self, features: int, eps: float = 1e-5):
        self.gamma = np.ones((1, features)); self.beta = np.zeros((1, features)); self.eps = eps
        self.dgamma = np.zeros_like(self.gamma); self.dbeta = np.zeros_like(self.beta)
    def forward(self, x, training: bool = True):
        self.x = x; self.mean = x.mean(0, keepdims=True); self.var = x.var(0, keepdims=True)
        self.xhat = (x-self.mean)/np.sqrt(self.var+self.eps); return self.gamma*self.xhat+self.beta
    def backward(self, grad):
        n = grad.shape[0]; std = np.sqrt(self.var+self.eps); self.dgamma=(grad*self.xhat).sum(0,keepdims=True); self.dbeta=grad.sum(0,keepdims=True)
        dxhat=grad*self.gamma
        return (1/n)/std*(n*dxhat-dxhat.sum(0,keepdims=True)-self.xhat*(dxhat*self.xhat).sum(0,keepdims=True))
    def parameters(self): return [(self.gamma,self.dgamma),(self.beta,self.dbeta)]


class Conv2D:
    """Valid convolution for NHWC arrays, with stride 1."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, seed: int = 7):
        self.k=kernel_size; rng=np.random.default_rng(seed); self.W=rng.normal(0,np.sqrt(2/(kernel_size**2*in_channels)),(kernel_size,kernel_size,in_channels,out_channels)); self.b=np.zeros(out_channels); self.dW=np.zeros_like(self.W); self.db=np.zeros_like(self.b)
    def forward(self, x):
        self.x=x; self.patches=np.lib.stride_tricks.sliding_window_view(x,(self.k,self.k),axis=(1,2)).transpose(0,1,2,4,5,3)
        return np.einsum('nxyijc,ijco->nxyo',self.patches,self.W,optimize=True)+self.b
    def backward(self, grad):
        dx=np.zeros_like(self.x); self.dW=np.einsum('nxyijc,nxyo->ijco',self.patches,grad,optimize=True); self.db=grad.sum((0,1,2))
        for i in range(self.k):
            for j in range(self.k):
                dx[:,i:i+grad.shape[1],j:j+grad.shape[2],:]+=np.einsum('nxyo,co->nxyc',grad,self.W[i,j],optimize=True)
        return dx
    def parameters(self): return [(self.W,self.dW),(self.b,self.db)]


class MaxPool2D:
    def __init__(self, size: int = 2): self.size=size
    def forward(self,x):
        self.x=x; n,h,w,c=x.shape; s=self.size; oh=h//s; ow=w//s; out=np.zeros((n,oh,ow,c)); self.argmax=np.zeros((n,oh,ow,c),dtype=int)
        for i in range(oh):
            for j in range(ow):
                p=x[:,i*s:(i+1)*s,j*s:(j+1)*s,:].reshape(n,s*s,c); self.argmax[:,i,j,:]=p.argmax(1); out[:,i,j,:]=p.max(1)
        return out
    def backward(self,grad):
        n,oh,ow,c=grad.shape; s=self.size; dx=np.zeros_like(self.x)
        for i in range(oh):
            for j in range(ow):
                for q in range(s*s):
                    mask=self.argmax[:,i,j,:]==q; dx[:,i*s+q//s,j*s+q%s,:]+=grad[:,i,j,:]*mask
        return dx
    def parameters(self): return []
