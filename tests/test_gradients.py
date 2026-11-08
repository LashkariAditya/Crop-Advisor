"""Numerical gradient checks for every differentiable NumPy layer."""
import numpy as np
from ml.core.layers import Dense,ReLU,Softmax,BatchNorm,Dropout,Conv2D,MaxPool2D

def check_input(layer,x,eps=1e-5,tol=2e-4):
    out=layer.forward(x); upstream=np.arange(out.size,dtype=float).reshape(out.shape)/out.size+.3; layer.backward(upstream); numeric=np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        old=x[idx]; x[idx]=old+eps; plus=(layer.forward(x)*upstream).sum(); x[idx]=old-eps; minus=(layer.forward(x)*upstream).sum(); x[idx]=old; numeric[idx]=(plus-minus)/(2*eps)
    layer.forward(x); analytic=layer.backward(upstream); assert np.allclose(analytic,numeric,atol=tol,rtol=tol)

def check_params(layer,x,eps=1e-5,tol=2e-4):
    y=layer.forward(x); up=np.ones_like(y)*.37; layer.backward(up)
    for param,grad in layer.parameters():
        for idx in np.ndindex(param.shape):
            old=param[idx]; param[idx]=old+eps; plus=(layer.forward(x)*up).sum(); param[idx]=old-eps; minus=(layer.forward(x)*up).sum(); param[idx]=old
            assert np.isclose(grad[idx],(plus-minus)/(2*eps),atol=tol,rtol=tol), (type(layer).__name__,idx,grad[idx])

def test_dense_and_parameter_gradients(): check_params(Dense(3,2),np.array([[.2,-.3,.4],[.6,.1,-.2]]))
def test_relu_gradient(): check_input(ReLU(),np.array([[-.7,.3],[.8,-.4]]))
def test_softmax_gradient(): check_input(Softmax(),np.array([[.2,-.3,.6],[.7,.1,-.5]]))
def test_batchnorm_gradients(): check_params(BatchNorm(3),np.array([[.1,.2,.4],[.3,.8,-.1],[.7,.2,.5]]))
def test_dropout_gradient():
    layer=Dropout(.3,3); x=np.arange(12,dtype=float).reshape(3,4)+1; y=layer.forward(x); mask=layer.mask.copy(); grad=np.ones_like(x)*.7; assert np.allclose(layer.backward(grad),mask*grad)
def test_conv_gradients():
    layer=Conv2D(1,2,2,4); x=np.arange(9,dtype=float).reshape(1,3,3,1)/10; check_params(layer,x,tol=8e-4); check_input(layer,x,tol=8e-4)
def test_maxpool_gradient(): check_input(MaxPool2D(2),np.array([[[[1.],[2.],[0.],[1.]],[[3.],[4.],[2.],[5.]],[[1.],[3.],[4.],[2.]],[[2.],[0.],[3.],[7.]]]]))
