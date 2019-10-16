from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

datax = np.ndarray(shape=[4000, 21*21*9])
for i in range(1, 4001):
    name = ''
    name += 'data5d/'
    name += 'data%d.txt' % i
    print(name)
    dataslice = np.loadtxt(name,dtype=float, delimiter=',')
    dataslice = np.reshape(dataslice, newshape=[21*21*9])
    datax[i - 1] = dataslice
datax=datax.reshape([-1,21,21,9])
datax=np.asarray(datax[:,:,:,3:9],dtype=np.float32)
print(datax.shape)
np.save('datax.npy',datax)