"""A shallow NHWC CNN made only from CropAdvisor's NumPy layers."""
import numpy as np
from ml.core.layers import Conv2D,ReLU,MaxPool2D,Dense
from ml.core.losses import cross_entropy
from ml.core.optim import Adam

class SmallCNN:
    def __init__(self,classes,seed=7,image_size=32):
        self.classes=list(classes); self.image_size=image_size; self.conv=Conv2D(3,8,3,seed); self.r1=ReLU(); self.pool=MaxPool2D(2); pooled=(image_size-2)//2; self.d1=Dense(pooled*pooled*8,32,seed+1); self.r2=ReLU(); self.d2=Dense(32,len(classes),seed+2); self.layers=[self.conv,self.r1,self.pool,self.d1,self.r2,self.d2]
    def forward(self,x):
        x=self.pool.forward(self.r1.forward(self.conv.forward(x))); self.shape=x.shape; x=x.reshape(len(x),-1); return self.d2.forward(self.r2.forward(self.d1.forward(x)))
    def backward(self,g):
        g=self.d2.backward(g); g=self.r2.backward(g); g=self.d1.backward(g); g=g.reshape(self.shape); g=self.pool.backward(g); g=self.r1.backward(g); return self.conv.backward(g)
    def fit(self,x,y,epochs=5,lr=.001,seed=7,validation_data=None,class_weights=None,patience=15):
        rng=np.random.default_rng(seed); opt=Adam(lr); history=[]; self.validation_history=[]; best_params=None; best_loss=float('inf'); wait=0; self.best_validation_epoch=None
        for epoch in range(epochs):
            order=rng.permutation(len(x)); total=0
            for start in range(0,len(x),16):
                ix=order[start:start+16]; sw=class_weights[y[ix]] if class_weights is not None else None; loss,g=cross_entropy(self.forward(x[ix]),y[ix],sw); total+=loss*len(ix); self.backward(g); opt.step(self.layers)
            history.append(total/len(x))
            if validation_data is not None:
                vx,vy=validation_data; val_loss=cross_entropy(self.forward(vx),vy)[0]; self.validation_history.append(val_loss)
                if val_loss < best_loss-1e-8:
                    best_loss=val_loss; wait=0; self.best_validation_epoch=epoch+1
                    best_params=[(p.copy(),b.copy()) for layer in self.layers for p,b in layer.parameters()]
                else:
                    wait+=1
                    if wait>=patience: break
        if best_params is not None:
            i=0
            for layer in self.layers:
                for param,_ in layer.parameters(): param[...] = best_params[i][0]; i+=1
        self.history=history
        return history
    def predict_proba(self,x):
        z=self.forward(x); p=np.exp(z-z.max(1,keepdims=True)); return p/p.sum(1,keepdims=True)
    def save(self,path):
        np.savez(path,classes=np.array(self.classes),image_size=self.image_size,cw=self.conv.W,cb=self.conv.b,d1w=self.d1.W,d1b=self.d1.b,d2w=self.d2.W,d2b=self.d2.b)
    @classmethod
    def load(cls,path):
        d=np.load(path,allow_pickle=False); m=cls(d['classes'].tolist(),image_size=int(d['image_size']) if 'image_size' in d else 16); m.conv.W=d['cw']; m.conv.b=d['cb']; m.d1.W=d['d1w']; m.d1.b=d['d1b']; m.d2.W=d['d2w']; m.d2.b=d['d2b']; return m

def augment_image(image,rng):
    """Horizontal/vertical flips, right-angle rotations, and brightness jitter."""
    x=image.copy()
    if rng.random()<.5: x=x[:,::-1]
    if rng.random()<.5: x=x[::-1]
    x=np.rot90(x,int(rng.integers(0,4)))
    padded=np.pad(x,((2,2),(2,2),(0,0)),mode='reflect'); row=int(rng.integers(0,5)); col=int(rng.integers(0,5)); x=padded[row:row+image.shape[0],col:col+image.shape[1]]
    return np.clip(x*rng.uniform(.85,1.15),0,1).copy()
