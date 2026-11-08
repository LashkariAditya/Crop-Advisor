"""SGD and Adam optimizers over layer parameter tuples."""
import numpy as np

class SGD:
    def __init__(self, lr=.01, momentum=0): self.lr=lr; self.momentum=momentum; self.v={}
    def step(self, layers):
        for layer in layers:
            for p,g in layer.parameters():
                key=id(p); self.v[key]=self.momentum*self.v.get(key,0)+g; p-=self.lr*self.v[key]

class Adam:
    def __init__(self, lr=.001, beta1=.9, beta2=.999, eps=1e-8): self.lr=lr; self.b1=beta1; self.b2=beta2; self.eps=eps; self.t=0; self.state={}
    def step(self,layers):
        self.t+=1
        for layer in layers:
            for p,g in layer.parameters():
                m,v=self.state.get(id(p),(np.zeros_like(p),np.zeros_like(p))); m=self.b1*m+(1-self.b1)*g; v=self.b2*v+(1-self.b2)*g*g; p-=self.lr*(m/(1-self.b1**self.t))/(np.sqrt(v/(1-self.b2**self.t))+self.eps); self.state[id(p)]=(m,v)
