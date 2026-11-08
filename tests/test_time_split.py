import numpy as np
from scripts.train_tabular import chronological_split

def test_chronological_split_keeps_future_out_of_train():
    train,val,test=chronological_split(100)
    assert train.max()<val.min()<val.max()<test.min()
    assert len(train)+len(val)+len(test)==100
