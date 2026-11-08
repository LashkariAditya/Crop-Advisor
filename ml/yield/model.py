"""Small NumPy regression MLP shared by yield and price tasks."""
import numpy as np
from ml.core.layers import Dense,ReLU
from ml.core.losses import mse
from ml.core.optim import Adam

class MLPRegressor:
    def __init__(self,n_features,hidden=32,seed=7): self.layers=[Dense(n_features,hidden,seed),ReLU(),Dense(hidden,1,seed+1)]
    def forward(self,x):
        for layer in self.layers: x=layer.forward(x)
        return x
    def fit(self,x,y,epochs=120,lr=.001,seed=7,validation_data=None,patience=15):
        rng=np.random.default_rng(seed); opt=Adam(lr); hist=[]; self.validation_history=[]; best_params=None; best_loss=float('inf'); wait=0; self.best_validation_epoch=None; y=y.reshape(-1,1)
        for epoch in range(epochs):
            order=rng.permutation(len(x)); loss_sum=0
            for start in range(0,len(x),64):
                ix=order[start:start+64]; pred=self.forward(x[ix]); loss,grad=mse(pred,y[ix]); loss_sum+=loss*len(ix)
                for layer in reversed(self.layers): grad=layer.backward(grad)
                opt.step(self.layers)
            hist.append(loss_sum/len(x))
            if validation_data is not None:
                vx,vy=validation_data; val,_=mse(self.forward(vx),np.asarray(vy).reshape(-1,1)); self.validation_history.append(val)
                if val < best_loss-1e-8:
                    best_loss=val; wait=0; self.best_validation_epoch=epoch+1
                    best_params=[p.copy() for layer in self.layers for p,_ in layer.parameters()]
                else:
                    wait+=1
                    if wait>=patience: break
        if best_params is not None:
            i=0
            for layer in self.layers:
                for param,_ in layer.parameters(): param[...] = best_params[i]; i+=1
        self.history=hist
        return hist
    def predict(self,x): return self.forward(x).ravel()
