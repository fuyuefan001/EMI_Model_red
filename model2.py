

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  print(features['x'])
  input_layer = tf.reshape(tensor=features['x'], shape=[-1, 21, 21,6])
  print(input_layer)
  input_norm=tf.layers.batch_normalization(input_layer)
  conv1 = tf.layers.conv2d(
      inputs=input_norm,
      filters=64,
      kernel_size=[5, 5],
      padding='valid',
      activation=tf.nn.relu)
  print(conv1)
  conv1 = tf.layers.batch_normalization(conv1)
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2)
  pool_flat=tf.reshape(pool1,[-1,8*8*64])
  dense1 = tf.layers.dense(inputs=pool_flat, units=512, activation=tf.nn.relu)
  dense2 = tf.layers.dense(inputs=dense1, units=256, activation=tf.nn.relu)
  dense3 = tf.layers.dense(inputs=dense2, units=128, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense3, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)

  logits = tf.layers.dense(inputs=dropout, units=1)
  predictions = {
      'val':logits
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.mean_squared_error(
      labels=labels, predictions=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    rate = tf.train.exponential_decay(0.001, tf.train.get_global_step(), 100, 0.8, True)
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=rate)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  mse=tf.metrics.mean_squared_error(
      labels=labels, predictions=predictions["val"])
  eval_metric_ops = {
      "accuracy": mse}

  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  datax=np.load('x_dataset.npy')
  datay=np.load('y_dataset.npy')
  datay=np.reshape(datay,[-1,1])
  print(datax)
  print(datay)



  datain=np.asarray(a=datax,dtype=np.float32)
  dataout = np.asarray(a=datay, dtype=np.float32)
  # my_feature_columns = [tf.feature_column.numeric_column(key='x', shape=[6084])]
  train_data = datain[0:3500] # Returns np.array
  train_labels = dataout[0:3500]
  print(train_data.shape)
  print(train_labels.shape)

  eval_data =datain[3600:4300]
  eval_labels =dataout[3600:4300]
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn)

  # Set up logging for predictions
  tensors_to_log = {"val"}
  logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=16,
      shuffle=True,
      num_threads=4,
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