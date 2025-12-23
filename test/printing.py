import numpy as np

def print_details(data, prefix=''):
    if type(data) is tuple and len(data) == 2:
        val, cov = data
        print("%sValCovPair!"%prefix)
        print("%sValues:"%prefix)
        print_details(val, prefix=prefix+"  ")
        print("%sCov"%prefix)
        print_details(cov, prefix=prefix+"  ")
    elif isinstance(data, np.ndarray):
        print("%sArray! shape %s, dtype %s, sum %g"%(prefix, data.shape, data.dtype, np.sum(data)))
    else:
        print("%sUnknown type: %s"%(prefix, type(data)))

def assertions_valcov(data, target):
    assert type(data) is tuple, "Result should be a tuple!"
    assert len(data) == 2, "Result should be a ValCov pair!"
    assert isinstance(data[0], np.ndarray), "Values should be an ndarray!"
    assert isinstance(data[1], np.ndarray), "Covariance should be an ndarray!"

    assert data[0].shape == target[0].shape, "Values shape mismatch!"
    assert data[1].shape == target[1].shape, "Covariance shape mismatch!"  
    assert np.allclose(data[0], target[0]), "Values content mismatch!"
    assert np.allclose(data[1], target[1]), "Covariance content mismatch!"

    return True

def assertions_covmat(data, target):
    assert isinstance(data, np.ndarray), "Result should be a ndarray!"
    assert data.shape == target.shape, "Shape mismatch!"
    assert np.allclose(data, target), "Content mismatch!"

    return True