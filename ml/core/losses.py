"""Loss functions returning scalar loss and derivative with respect to logits."""
import numpy as np

def cross_entropy(logits, labels, sample_weights=None):
    shifted=logits-logits.max(axis=1,keepdims=True); p=np.exp(shifted); p/=p.sum(axis=1,keepdims=True); y=np.asarray(labels,dtype=int); weights=np.ones(len(y)) if sample_weights is None else np.asarray(sample_weights,dtype=float); denom=max(weights.sum(),1e-12); loss=-(weights*np.log(np.maximum(p[np.arange(len(y)),y],1e-12))).sum()/denom; p[np.arange(len(y)),y]-=1; return float(loss),p*(weights/denom)[:,None]

def mse(prediction, target):
    d=prediction-target; return float(np.mean(d*d)), 2*d/d.size
