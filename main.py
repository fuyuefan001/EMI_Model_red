from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf

tmp = np.loadtxt("C:/Users/Fuyuefan/Desktop/data.txt", dtype=np.str, delimiter=",")
datay=tmp[:,:].astype(np.float32)
datax=np.ndarray(shape=[1000,6084])
for i in range(1,1001):
    name=''
    name+='C:/Users/Fuyuefan/Desktop/sample/datatxt/'
    name+='data%d.txt'%i
    print(name)
    dataslice=np.loadtxt(name,float,delimiter=',')
    dataslice=np.reshape(dataslice,newshape=[6084])
    datax[i-1]=dataslice
print(datax)
my_feature_columns=[tf.feature_column.numeric_column(key='x',shape=[6084])]
train_data = datax  # Returns np.array
train_labels = np.asarray(datay, dtype=np.float32)
print(train_labels)
train_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x':train_data},
                                                    y=train_labels,
                                                    batch_size=64,
                                                    num_epochs=100,
                                                    shuffle=True)
regressor = tf.estimator.DNNRegressor(feature_columns=my_feature_columns,
                                      hidden_units=[512,256,128],
                                      optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0000001),
                                      activation_fn=tf.nn.relu,
                                      label_dimension=18
                                      )
ev=regressor.train(input_fn=train_input_fn,max_steps=10000)




