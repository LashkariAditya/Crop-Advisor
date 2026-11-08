"""Price forecasting model, using lagged prices and calendar features via the shared NumPy regressor."""
from importlib import import_module
MLPRegressor=import_module('ml.yield.model').MLPRegressor

class PriceMLP(MLPRegressor):
    """MLP regression head for user-reviewed price time-series features."""
