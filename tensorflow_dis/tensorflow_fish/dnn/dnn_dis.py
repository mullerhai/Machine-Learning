# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Distributed MNIST training and validation, with model replicas.

A simple softmax model with one hidden layer is defined. The parameters
(weights and biases) are located on two parameter servers (ps), while the
ops are defined on a worker node. The TF sessions also run on the worker
node.
Multiple invocations of this script can be done in parallel, with different
values for --worker_index. There should be exactly one invocation with
--worker_index, which will create a master session that carries out variable
initialization. The other, non-master, sessions will wait for the master
session to finish the initialization before proceeding to the training stage.

The coordination between the multiple worker invocations occurs due to
the definition of the parameters on the same ps devices. The parameter updates
from one worker is visible to all other workers. As such, the workers can
perform forward computation and gradient calculation in parallel, which
should lead to increased training speed for the simple model.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import csv
import math
import sys
import tempfile
import time
import random
import collections
import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf.tensorflow_server_pb2 import ClusterDef
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import gfile
from sklearn.metrics import roc_auc_score
import reader

log_file = open("out.log","w")

flags = tf.app.flags
flags.DEFINE_string("data_dir", None,
                    "Directory for storing mnist data")
flags.DEFINE_string("train_dir", None,
                    "Directory for storing events")
#flags.DEFINE_string("cluster_spec", None,
#                    "ClusterSpec")
#flags.DEFINE_boolean("download_only", False,
#                     "Only perform downloading of data; Do not proceed to "
#                     "session preparation, model definition or training")
flags.DEFINE_integer("worker_index", 0,
                     "Worker task index, should be >= 0. worker_index=0 is "
                     "the master worker task the performs the variable "
                     "initialization ")
flags.DEFINE_integer("num_workers", None,
                     "Total number of workers (must be >= 1)")
flags.DEFINE_integer("num_parameter_servers", None,
                     "Total number of parameter servers (must be >= 1)")
flags.DEFINE_integer("replicas_to_aggregate", None,
                     "Number of replicas to aggregate before parameter update"
                     "is applied (For sync_replicas mode only; default: "
                     "num_workers)")
#flags.DEFINE_integer("grpc_port", 2222,
#                     "TensorFlow GRPC port")
flags.DEFINE_integer("hidden_units", 512,
                     "Number of units in the hidden layer of the NN")
flags.DEFINE_integer("train_steps", 4000,
                     "Number of (global) training steps to perform")
flags.DEFINE_integer("batch_size", 1000, "Training batch size")
flags.DEFINE_float("learning_rate", 0.00001, "Learning rate")
flags.DEFINE_string("worker_grpc_url", None,
                    "Worker GRPC URL (e.g., grpc://1.2.3.4:2222, or "
                    "grpc://tf-worker0:2222)")
flags.DEFINE_boolean("sync_replicas", False,
                     "Use the sync_replicas (synchronized replicas) mode, "
                     "wherein the parameter updates from workers are aggregated "
                     "before applied to avoid stale gradients")
FLAGS = flags.FLAGS


IMAGE_PIXELS = 28
INPUT_DIMENSION = 215
OUTPUT_DIMENSION = 1
SEED = 66478

#PARAM_SERVER_PREFIX = "tf-ps"  # Prefix of the parameter servers' domain names
#WORKER_PREFIX = "tf-worker"  # Prefix of the workers' domain names

def auc(sess, predict_op, y_op, repeat):
 predict, y = sess.run([predict_op,y_op]) 
 return roc_auc_score(y,predict)

def activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  tensor_name = x.op.name
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def get_device_setter(num_parameter_servers,task_index):
  """Get a device setter given number of servers in the cluster.

  Given the numbers of parameter servers and workers, construct a device
  setter object using ClusterSpec.

  Args:
    num_parameter_servers: Number of parameter servers
    num_workers: Number of workers

  Returns:
    Device setter object.
  """

  return tf.train.replica_device_setter(ps_tasks = num_parameter_servers,worker_device="/job:worker/task:%d" % task_index)

'''
  TrainFileNameList = []
  for idx in range(0,10):
      TrainFileNameList.append(Pre+"{0:03d}".format(idx))
  train_data = load_csv(TrainFileNameList, dtypes.int32, 0, False)
  length = len(train_data.target)
  assert length == len(train_data.data)
  print("Training data size: %d" % length)
'''
def main(unused_argv):
  
  #print("Worker GRPC URL: %s" % FLAGS.worker_grpc_url)
  log_file.write("Worker GRPC URL: %s" % FLAGS.worker_grpc_url+"\n")
  log_file.flush()
  #print("Worker index = %d" % FLAGS.worker_index)
  log_file.write("Worker index = %d" % FLAGS.worker_index+"\n")
  log_file.flush()
  #print("Number of workers = %d" % FLAGS.num_workers)
  log_file.write("Number of workers = %d" % FLAGS.num_workers+"\n")
  log_file.flush()

  # Sanity check on the number of workers and the worker index
  if FLAGS.worker_index >= FLAGS.num_workers:
    raise ValueError("Worker index %d exceeds number of workers %d " % (FLAGS.worker_index, FLAGS.num_workers))

  # Sanity check on the number of parameter servers
  if FLAGS.num_parameter_servers <= 0:
    raise ValueError("Invalid num_parameter_servers value: %d" % FLAGS.num_parameter_servers)

  is_chief = (FLAGS.worker_index == 0)

  if FLAGS.sync_replicas:
    if FLAGS.replicas_to_aggregate is None:
      replicas_to_aggregate = FLAGS.num_workers
    else:
      replicas_to_aggregate = FLAGS.replicas_to_aggregate

  # Construct device setter object
  device_setter = get_device_setter(FLAGS.num_parameter_servers, FLAGS.worker_index)

  # The device setter will automatically place Variables ops on separate
  # parameter servers (ps). The non-Variable ops will be placed on the workers.
  with tf.device(device_setter):
    global_step = tf.Variable(0, name="global_step", trainable=False)

    ws = []
    bs = []
    dims = [INPUT_DIMENSION,1024,1024,512,512,50,1]
    for i in range(1,len(dims)):
      w = tf.Variable(tf.truncated_normal([dims[i-1],dims[i]],stddev=1.0/math.sqrt(dims[i-1])),name=("hid_w" + str(i)))
      b = tf.Variable(tf.zeros([dims[i]]), name=("hid_b" + str(i)))
      ws.append(w)
      bs.append(b)


    # Ops: located on the worker specified with FLAGS.worker_index
    #x = tf.placeholder(tf.float32, [None, INPUT_DIMENSION])
    #y_ = tf.placeholder(tf.float32, [None, OUTPUT_DIMENSION])
    #train_x,train_y_ = reader.data_op(FLAGS.data_dir+"train/*",INPUT_DIMENSION)
    #eval_x,eval_y_ = reader.data_op(FLAGS.data_dir+"eval/*",INPUT_DIMENSION)
    train_x,train_y_ = reader.data_op("/home/fish.hy/dnn/data/*",INPUT_DIMENSION)
    eval_x,eval_y_ = reader.data_op("/home/fish.hy/dnn/data/*",INPUT_DIMENSION)

    is_add_dropout = [True,True,True,False,False,False]
    def model(input_data, train=False):
      prev = input_data
      for i in range(0,len(is_add_dropout)):
        hid_lin = tf.nn.xw_plus_b(prev,ws[i],bs[i])
        hid = tf.nn.elu(hid_lin) if i < len(is_add_dropout) else hid_lin
        if train and is_add_dropout[i]:
          hid = tf.nn.dropout(hid,0.5,seed = SEED)
        if is_chief:
          activation_summary(hid)
        prev = hid
      return hid

    logits = model(train_x, True)
    logits = tf.reshape(logits,shape=[50])
    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, train_y_) )
    if is_chief:
      tf.scalar_summary(cross_entropy.op.name+'/mean', cross_entropy)

    train_prediction = tf.sigmoid(logits)
    eval_prediction = tf.sigmoid(model(eval_x, False))

    opt = tf.train.AdamOptimizer(FLAGS.learning_rate)
    if FLAGS.sync_replicas:
      opt = tf.train.SyncReplicasOptimizer(
          opt,
          replicas_to_aggregate=replicas_to_aggregate,
          total_num_replicas=FLAGS.num_workers,
          replica_id=FLAGS.worker_index,
          name="mnist_sync_replicas")

#    train_op = opt.minimize(cross_entropy,global_step=global_step)

    grads = opt.compute_gradients(cross_entropy)
    train_op = opt.apply_gradients(grads, global_step=global_step)

    if is_chief:
      # Add histograms for trainable variables.
      for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)
      # Add histograms for gradients.
      for grad, var in grads:
        if grad is not None:
          tf.histogram_summary(var.op.name + '/gradients', grad)

      summary_op = tf.merge_all_summaries()


    if FLAGS.sync_replicas and is_chief:
      # Initial token and chief queue runners required by the sync_replicas mode
      chief_queue_runner = opt.get_chief_queue_runner()
      init_tokens_op = opt.get_init_tokens_op()

    init_op = tf.initialize_all_variables()
    train_dir = tempfile.mkdtemp()
    sv = tf.train.Supervisor(is_chief=is_chief,
                             logdir=train_dir,
                             summary_op = None,
                             init_op=init_op,
                             recovery_wait_secs=1,
                             global_step=global_step)

    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True
        #device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.worker_index])
	)

    # The chief worker (worker_index==0) session will prepare the session,
    # while the remaining workers will wait for the preparation to complete.
    if is_chief:
      #print("Worker %d: Initializing session..." % FLAGS.worker_index)
      log_file.write("Worker %d: Initializing session...\n" % FLAGS.worker_index)
      log_file.flush()
    else:
      #print("Worker %d: Waiting for session to be initialized..." %
      #      FLAGS.worker_index)
      log_file.write("Worker %d: Waiting for session to be initialized...\n" %
            FLAGS.worker_index)
      log_file.flush()

    sess = sv.prepare_or_wait_for_session(FLAGS.worker_grpc_url,
                                          config=sess_config)
    #tf.train.start_queue_runners(sess = sess)
    #sv.start_queue_runners(sess)

    #print("Worker %d: Session initialization complete." % FLAGS.worker_index)
    log_file.write("Worker %d: Session initialization complete.\n" % FLAGS.worker_index)
    log_file.flush()

    if FLAGS.sync_replicas and is_chief:
      # Chief worker will start the chief queue runner and call the init op
      #print("Starting chief queue runner and running init_tokens_op")
      log_file.write("Starting chief queue runner and running init_tokens_op\n")
      log_file.flush()
      sv.start_queue_runners(sess, [chief_queue_runner])
      sess.run(init_tokens_op)

#    summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)

    # Perform training
    time_begin = time.time()
    #print("Training begins @ %f" % time_begin)
    log_file.write("Training begins @ %f\n" % time_begin)
    log_file.flush()

    while True:

      _, step, loss = sess.run([train_op, global_step, cross_entropy])
      print(step)

      now = time.time()
      if step % 100 == 0:
        #print("%f: Worker %d: training step %d done (global step: %d), cross entropy = %g" % (now, FLAGS.worker_index, local_step, step, loss))
        log_file.write("%f: Worker %d: training step %d done (global step: %d), cross entropy = %g\n" % (now, FLAGS.worker_index, local_step, step, loss))
  	log_file.flush()
        if is_chief:
          sv.summary_computed(sess, sess.run(summary_op, feed_dict=train_feed))
 #        summary_writer.add_summary(summary_str, step)

      if step >= FLAGS.train_steps:
        break
      if step % 100 == 0:
    	#y_score = np.array(predict_results)
        #y_true = np.array(eval_y_real)
        auc(sess,eval_prediction,eval_y_,1)
        #print('Auc: %.3f, validation cross entropy =%g' % (auc, val_loss))
        log_file.write('Auc: %.3f\n' % (auc))
  	log_file.flush()

    time_end = time.time()
    #print("Training ends @ %f\n" % time_end)
    log_file.write("Training ends @ %f\n" % time_end)
    log_file.flush()
    training_time = time_end - time_begin
    #print("Training elapsed time: %f s" % training_time)
    log_file.write("Training elapsed time: %f s\n" % training_time)
    log_file.flush()

    auc(sess,eval_prediction,eval_y_,1)
    #print('Auc: %.3f' % auc)
    log_file.write('Auc: %.3f\n' % auc)
    log_file.flush()
    
    #auc, update_op = tf.contrib.metrics.streaming_auc(predicts, y_)
    #val_xent = sess.run(auc, feed_dict=val_feed)
    #print("After %d training step(s), validation cross entropy = %g" %  (FLAGS.train_steps, val_xent))


if __name__ == "__main__":
  tf.app.run()
  log_file.close()
