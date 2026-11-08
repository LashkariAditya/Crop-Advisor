"""Small MLP classifier trained from scratch with NumPy."""
from __future__ import annotations
import numpy as np
from ml.core.layers import Dense, ReLU
from ml.core.losses import cross_entropy
from ml.core.optim import Adam

class MLPClassifier:
    def __init__(self, input_size: int, hidden_size: int, classes: list[str], seed: int = 7):
        self.classes=classes; self.layers=[Dense(input_size,hidden_size,seed),ReLU(),Dense(hidden_size,len(classes),seed+1)]
    def forward(self,x):
        for layer in self.layers: x=layer.forward(x)
        return x
    def fit(self,x,y,epochs=100,lr=.001,seed=7,class_weights=None,validation_data=None,patience=15):
        rng=np.random.default_rng(seed); opt=Adam(lr); history=[]; self.validation_history=[]; best_params=None; best_loss=float('inf'); wait=0; self.best_validation_epoch=None
        for epoch in range(epochs):
            order=rng.permutation(len(x)); total=0
            for start in range(0,len(x),64):
                ix=order[start:start+64]; logits=self.forward(x[ix]); sw=class_weights[y[ix]] if class_weights is not None else None; loss,grad=cross_entropy(logits,y[ix],sw); total+=loss*len(ix)
                for layer in reversed(self.layers): grad=layer.backward(grad)
                opt.step(self.layers)
            history.append(total/len(x))
            if validation_data is not None:
                vx,vy=validation_data; val_loss=cross_entropy(self.forward(vx),vy)[0]; self.validation_history.append(val_loss)
                if val_loss < best_loss-1e-8:
                    best_loss=val_loss; wait=0; self.best_validation_epoch=epoch+1
                    best_params=[p.copy() for layer in self.layers for p,_ in layer.parameters()]
                else:
                    wait+=1
                    if wait>=patience: break
        if best_params is not None:
            i=0
            for layer in self.layers:
                for param,_ in layer.parameters(): param[...] = best_params[i]; i+=1
        self.history=history
        return history
    def predict_proba(self,x):
        logits=self.forward(x); e=np.exp(logits-logits.max(1,keepdims=True)); return e/e.sum(1,keepdims=True)
    def save(self,path,mean,std,features=None):
        np.savez(path,w1=self.layers[0].W,b1=self.layers[0].b,w2=self.layers[2].W,b2=self.layers[2].b,mean=mean,std=std,classes=np.array(self.classes),features=np.array(features or [],dtype=str))
    @classmethod
    def load(cls,path):
        d=np.load(path,allow_pickle=False); model=cls(len(d['mean']),d['w1'].shape[1],d['classes'].tolist()); model.layers[0].W=d['w1']; model.layers[0].b=d['b1']; model.layers[2].W=d['w2']; model.layers[2].b=d['b2']; model.mean=d['mean']; model.std=d['std']; model.features=d['features'].tolist() if 'features' in d else []; return model
    def rank(self,x):
        p=self.predict_proba((x-self.mean)/(self.std+1e-8))[0]; ix=np.argsort(p)[::-1]; return [{"crop":self.classes[i],"confidence":float(p[i])} for i in ix]
