import numpy as np

'''
For time-series data only
win_len: how many samples do you want 
sliding_step: how many overlap do you want?

'''



def sliding_window(data,win_len,sliding_step,flatten=None):
    if len( data.shape )==1:
        # data = np.reshape(data,(data.shape[0],1))
        n_samples =  data.shape[0]
        dim=1
        n_windows = n_samples//sliding_step
        shape = (n_windows,sliding_step,)
        strides = data.itemsize*np.array([sliding_step*dim,dim,])
        result = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
        # result = np.expand_dims(result,axis=2)
        # result = np.reshape(result,(n_windows,sliding_step,1))

        return result
    if len(data.shape)!=1:
        n_samples,dim = data.shape
        n_windows = n_samples // sliding_step
        shape = (n_windows,sliding_step,dim)
        strides = data.itemsize * np.array([sliding_step * dim,dim,1])
        result = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
        return result
