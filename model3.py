from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf



def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  input_layer = tf.reshape(features["x"], [-1, 21, 21, 6])
  print(input_layer)
  input_layer=tf.cast(input_layer,dtype=tf.float32)
  input_norm=tf.layers.batch_normalization(input_layer)
  print(input_norm)
  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_norm,
      filters=64,
      kernel_size=[5,5],
      padding='valid',
      activation=tf.nn.relu)
  print(conv1)
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2)
  pool1=tf.layers.batch_normalization(pool1)
  print(pool1)
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  print(conv2)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
  pool2 = tf.layers.batch_normalization(pool2)
  print(pool2)
  pool2_flat = tf.reshape(pool2, [-1, 4*4*64])
  dense1 = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout1 = tf.layers.dropout(
      inputs=dense1, rate=0, training=mode == tf.estimator.ModeKeys.TRAIN)
  dense2 = tf.layers.dense(inputs=dropout1, units=1024, activation=tf.nn.relu)
  dense3 = tf.layers.dense(inputs=dense2, units=256, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense3, rate=0, training=mode == tf.estimator.ModeKeys.TRAIN)
  output = tf.layers.dense(inputs=dropout, units=5)
  predictions = {
      'val':output
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)

  loss = tf.losses.mean_squared_error(
      labels=labels, predictions=output)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    rate=tf.train.exponential_decay(learning_rate=0.01,global_step=tf.train.get_global_step(),decay_steps=100,decay_rate=0.98,staircase=False)
    train_op = tf.train.AdadeltaOptimizer(learning_rate=rate).minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  mse=tf.metrics.mean_squared_error(
      labels=labels, predictions=predictions["val"])
  eval_metric_ops = {
      "error": mse}

  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):

  datay = np.loadtxt('data5d.txt', dtype=np.float32, delimiter=',')

  datax = np.load('datax.npy')
  print(datax.shape)
  print(datay.shape)

  badarr=np.loadtxt('badfilename.txt')
  badarr=badarr.astype(dtype=np.int)
  print(badarr)
  progress=0;
  for index in badarr:
      print(datax[index-1-progress])
      datax=np.delete(datax,obj=index-1-progress,axis=0)
      datay = np.delete(datay, obj=index - 1 - progress, axis=0)
      progress=progress+1

  print(datax.shape)
  print(datay.shape)



  # for index in badarr:


  datain=np.asarray(a=datax,dtype=np.float32)
  dataout = np.asarray(a=datay, dtype=np.float32)
  # my_feature_columns = [tf.feature_column.numeric_column(key='x', shape=[6084])]
  train_data = datain[0:3500] # Returns np.array
  train_labels = dataout[0:3500]
  print(train_data.shape)
  print(train_labels.shape)

  eval_data =datain[3500:3975]
  eval_labels =dataout[3500:3975]

  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn)

  # Set up logging for predictions
  tensors_to_log = {"val"}
  logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=32,
      shuffle=True,
      num_threads=2,
      num_epochs=None)
  train_res=mnist_classifier.train(
      input_fn=train_input_fn,
      steps=10000)
  print('train result')
  print(train_res)
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=10,
      shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)
tf.logging.set_verbosity(tf.logging.INFO)

if __name__ == "__main__":
  tf.app.run()