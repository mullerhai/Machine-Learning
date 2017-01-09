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
INPUT_DIMENSION = 428
OUTPUT_DIMENSION = 1
SEED = 66478

#PARAM_SERVER_PREFIX = "tf-ps"  # Prefix of the parameter servers' domain names
#WORKER_PREFIX = "tf-worker"  # Prefix of the workers' domain names


Dataset = collections.namedtuple('Dataset', ['data', 'target'])
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
def load_csv(FileNameList, target_dtype, target_column=0, has_header=True):
    data, target = [], []
    for idx in range(len(FileNameList)):
        #print (FileNameList[idx])
	log_file.write(FileNameList[idx]+"\n")
	log_file.flush()
	with gfile.Open(FileNameList[idx]) as csv_file:
            data_file = csv.reader(csv_file)
            for ir in data_file:
                target.append(int(ir.pop(target_column)))
                data.append(ir)

    target = np.array(target).reshape(len(target),1)
    data = np.array(data).reshape(len(data),428)
    return Dataset(data=data, target=target)

def random_shuffle(data, length):
    for i in range(1,length):
        pos = random.randint(0,i)
        data.data[i],data.data[pos] = data.data[pos],data.data[i]
        data.target[i],data.target[pos] = data.target[pos],data.target[i]

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

def parse_cluster_spec(cluster_spec):
  """Parse content of cluster_spec string and inject info into cluster protobuf.

  Args:
    cluster_spec: cluster specification string, e.g.,
          "work|localhost:2222;localhost:2223"

  Raises:
    ValueError: if the cluster_spec string is invalid.
  """

  cluster = ClusterDef()
  job_strings = cluster_spec.split(",")

  if not cluster_spec:
    raise ValueError("Empty cluster_spec string")

  for job_string in job_strings:
    job_def = cluster.job.add()

    if job_string.count("|") != 1:
      raise ValueError("Not exactly one instance of '|' in cluster_spec")

    job_name = job_string.split("|")[0]

    if not job_name:
      raise ValueError("Empty job_name in cluster_spec")

    job_def.name = job_name

    job_tasks = job_string.split("|")[1].split(";")
    for i in range(len(job_tasks)):
      if not job_tasks[i]:
        raise ValueError("Empty task string at position %d" % i)

      job_def.tasks[i] = job_tasks[i]

  return tf.train.ClusterSpec(cluster)

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

#  for j in range(num_parameter_servers):
#    ps_spec.append("%s%d:%d" % (PARAM_SERVER_PREFIX, j, FLAGS.grpc_port))

#  worker_spec = []
#  for k in range(num_workers):
#    worker_spec.append("%s%d:%d" % (WORKER_PREFIX, k, FLAGS.grpc_port))

#  cluster_spec = tf.train.ClusterSpec({
#      "ps": ps_spec,
#      "worker": worker_spec})

#  cluster_spec = parse_cluster_spec(FLAGS.cluster_spec)
  # Get device setter from the cluster spec
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
  
  TrainFileIdx = []
  for i in range(0,300):
    TrainFileIdx.append(i)

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

    # Variables of the first hidden layer
    hid_w1 = tf.Variable(tf.truncated_normal([INPUT_DIMENSION, FLAGS.hidden_units], stddev=1.0 / math.sqrt(INPUT_DIMENSION)), name="hid_w1")
    hid_b1 = tf.Variable(tf.zeros([FLAGS.hidden_units]), name="hid_b1")

    # Variables of the second hidden layer
    hid_w2 = tf.Variable(tf.truncated_normal([FLAGS.hidden_units, FLAGS.hidden_units], stddev=1.0 / math.sqrt(FLAGS.hidden_units)), name="hid_w2")
    hid_b2 = tf.Variable(tf.zeros([FLAGS.hidden_units]), name="hid_b2")

    # Variables of the third hidden layer
    hid_w3 = tf.Variable(tf.truncated_normal([FLAGS.hidden_units, FLAGS.hidden_units], stddev=1.0 / math.sqrt(FLAGS.hidden_units)), name="hid_w3")
    hid_b3 = tf.Variable(tf.zeros([FLAGS.hidden_units]), name="hid_b3")

    # Variables of the fourth hidden layer
    hid_w4 = tf.Variable(tf.truncated_normal([FLAGS.hidden_units, FLAGS.hidden_units], stddev=1.0 / math.sqrt(FLAGS.hidden_units)), name="hid_w4")
    hid_b4 = tf.Variable(tf.zeros([FLAGS.hidden_units]), name="hid_b4")

    # Variables of the fifth hidden layer
#    hid_w5 = tf.Variable(tf.truncated_normal([FLAGS.hidden_units, FLAGS.hidden_units], stddev=1.0 / math.sqrt(FLAGS.hidden_units)), name="hid_w5")
#    hid_b5 = tf.Variable(tf.zeros([FLAGS.hidden_units]), name="hid_b5")

    # Variables of the sixth hidden layer
#    hid_w6 = tf.Variable(tf.truncated_normal([FLAGS.hidden_units, FLAGS.hidden_units], stddev=1.0 / math.sqrt(FLAGS.hidden_units)), name="hid_w6")
#    hid_b6 = tf.Variable(tf.zeros([FLAGS.hidden_units]), name="hid_b6")

    # Variables of the seventh hidden layer
#    hid_w7 = tf.Variable(tf.truncated_normal([FLAGS.hidden_units, FLAGS.hidden_units], stddev=1.0 / math.sqrt(FLAGS.hidden_units)), name="hid_w7")
#    hid_b7 = tf.Variable(tf.zeros([FLAGS.hidden_units]), name="hid_b7")

    # Variables of the softmax layer
    sm_w = tf.Variable(tf.truncated_normal([FLAGS.hidden_units, OUTPUT_DIMENSION], stddev=1.0 / math.sqrt(FLAGS.hidden_units)),name="sm_w")
    sm_b = tf.Variable(tf.zeros([OUTPUT_DIMENSION]), name="sm_b")

    # Ops: located on the worker specified with FLAGS.worker_index
    x = tf.placeholder(tf.float32, [None, INPUT_DIMENSION])
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_DIMENSION])

    def model(data, train=False):
      hid_lin1 = tf.nn.xw_plus_b(x, hid_w1, hid_b1)
      hid1 = tf.nn.elu(hid_lin1)
      if train:
        hid1 = tf.nn.dropout(hid1, 0.5, seed=SEED)
      if is_chief:
        activation_summary(hid1)

      hid_lin2 = tf.nn.xw_plus_b(hid1, hid_w2, hid_b2)
      hid2 = tf.nn.elu(hid_lin2)
      if train:
        hid2 = tf.nn.dropout(hid2, 0.5, seed=SEED)
      if is_chief:
        activation_summary(hid2)

      hid_lin3 = tf.nn.xw_plus_b(hid2, hid_w3, hid_b3)
      hid3 = tf.nn.elu(hid_lin3)
      if train:
        hid3 = tf.nn.dropout(hid3, 0.5, seed=SEED)
      if is_chief:
        activation_summary(hid3)

      hid_lin4 = tf.nn.xw_plus_b(hid3, hid_w4, hid_b4)
      hid4 = tf.nn.elu(hid_lin4)
      if train:
        hid4 = tf.nn.dropout(hid4, 0.5, seed=SEED)
      if is_chief:
        activation_summary(hid4)
      '''
      hid_lin5 = tf.nn.xw_plus_b(hid4, hid_w5, hid_b5)
      hid5 = tf.nn.relu(hid_lin5)
      if train:
        hid5 = tf.nn.dropout(hid5, 0.5, seed=SEED)
      if is_chief:
        activation_summary(hid5)

      hid_lin6 = tf.nn.xw_plus_b(hid5, hid_w6, hid_b6)
      hid6 = tf.nn.relu(hid_lin6)
      if train:
        hid6 = tf.nn.dropout(hid6, 0.5, seed=SEED)
      if is_chief:
        activation_summary(hid6)

      hid_lin7 = tf.nn.xw_plus_b(hid6, hid_w7, hid_b7)
      hid7 = tf.nn.relu(hid_lin7)
      if train:
        hid7 = tf.nn.dropout(hid7, 0.5, seed=SEED)
      if is_chief:
        activation_summary(hid7)
      '''

      return tf.nn.xw_plus_b(hid4, sm_w, sm_b)

    logits = model(x, True)
    cross_entropy = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits, y_) )
    if is_chief:
      tf.scalar_summary(cross_entropy.op.name+'/mean', cross_entropy)

    train_prediction = tf.sigmoid(logits)
    eval_prediction = tf.sigmoid(model(x, False))

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

    # Validation feed
    EvalFileNameList = []
    Pre = "/home/chuanxin.tcx/Test/Data/eval_shuffled_part-00"
    for idx in range(0,12):
	EvalFileNameList.append(Pre+"{0:03d}".format(idx)) 
    test_data = load_csv(EvalFileNameList, dtypes.int32, 0, False)
    val_feed = {x: test_data.data, y_: test_data.target}

    # Perform training
    time_begin = time.time()
    #print("Training begins @ %f" % time_begin)
    log_file.write("Training begins @ %f\n" % time_begin)
    log_file.flush()

    local_step = 0
    start = 0
    length = -1
    CurFileIdx = len(TrainFileIdx)
    Pre = "/home/chuanxin.tcx/Test/Data/train_shuffled_part-00"
    while True:
      # Training feed
      end = start + FLAGS.batch_size
      if end > length:
        if CurFileIdx == len(TrainFileIdx):
          random.shuffle(TrainFileIdx)    
          CurFileIdx = 0
        TrainFileNameList = []
        TrainFileNameList.append(Pre+"{0:03d}".format(TrainFileIdx[CurFileIdx]))
        train_data = load_csv(TrainFileNameList, dtypes.int32, 0, False)
        length = len(train_data.target)
        assert length == len(train_data.data)
        #print("Current training file index: %d, Training data size: %d" % (TrainFileIdx[CurFileIdx],length))
        log_file.write("Current training file index: %d, Training data size: %d\n" % (TrainFileIdx[CurFileIdx],length))
  	log_file.flush()
#        random_shuffle(train_data,length)
        CurFileIdx += 1
        start = 0
        end = start + FLAGS.batch_size
      batch_xs = train_data.data[start:end]
      batch_ys = train_data.target[start:end]
      train_feed = {x: batch_xs,
                    y_: batch_ys}

      _, step, loss = sess.run([train_op, global_step, cross_entropy], feed_dict=train_feed)
      start = end
      local_step += 1

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
        predict_results, val_loss = sess.run([eval_prediction, cross_entropy], feed_dict=val_feed)
    	y_score = np.array(predict_results)
        y_true = np.array(test_data.target)
        auc = roc_auc_score(y_true, y_score)
        #print('Auc: %.3f, validation cross entropy =%g' % (auc, val_loss))
        log_file.write('Auc: %.3f, validation cross entropy =%g\n' % (auc, val_loss))
  	log_file.flush()

    time_end = time.time()
    #print("Training ends @ %f\n" % time_end)
    log_file.write("Training ends @ %f\n" % time_end)
    log_file.flush()
    training_time = time_end - time_begin
    #print("Training elapsed time: %f s" % training_time)
    log_file.write("Training elapsed time: %f s\n" % training_time)
    log_file.flush()

    # Validation feed
    predict_results = sess.run(eval_prediction, feed_dict=val_feed)  
    y_score = np.array(predict_results)
    y_true = np.array(test_data.target)
    auc = roc_auc_score(y_true, y_score)
    #print('Auc: %.3f' % auc)
    log_file.write('Auc: %.3f\n' % auc)
    log_file.flush()
    
    #auc, update_op = tf.contrib.metrics.streaming_auc(predicts, y_)
    #val_xent = sess.run(auc, feed_dict=val_feed)
    #print("After %d training step(s), validation cross entropy = %g" %  (FLAGS.train_steps, val_xent))


if __name__ == "__main__":
  tf.app.run()
  log_file.close()
