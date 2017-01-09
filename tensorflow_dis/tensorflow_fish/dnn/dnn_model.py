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

import numpy as np
import tensorflow as tf
import tempfile
import math
import time
from reader import data_op
from sklearn.metrics import roc_auc_score

flags = tf.app.flags
flags.DEFINE_string("data_dir", None,
                    "Directory for storing events")
flags.DEFINE_string("log_dir", None,
                    "Directory for storing events")
flags.DEFINE_string("report_file", None,
                    "Directory for storing events")

flags.DEFINE_integer("worker_index", 0,
                     "Worker task index, should be >= 0. worker_index=0 is "
                     "the master worker task the performs the variable "
                     "initialization ")
flags.DEFINE_integer("eval_secs", 0,
                     "eval seconds")
flags.DEFINE_integer("num_parameter_servers", None,
                     "Total number of parameter servers (must be >= 1)")
flags.DEFINE_float("learning_rate", 0.00001, "Learning rate")
flags.DEFINE_integer("batch_size", 50, "Training batch size")
flags.DEFINE_integer("eval_batch_size", 1000000, "Training batch size")
flags.DEFINE_integer("eval_iter", 1, "Training batch size")
flags.DEFINE_integer("train_steps", 100000000,
                     "Number of (global) training steps to perform")
flags.DEFINE_integer("report_step", 100,
                     "Number of (global) training steps to perform")
flags.DEFINE_string("master", None,
                    "Worker GRPC URL (e.g., grpc://1.2.3.4:2222, or "
                    "grpc://tf-worker0:2222)")

FLAGS = flags.FLAGS

INPUT_DIMENSION = 215
dims = [INPUT_DIMENSION,1024,1024,512,512,50,1]
is_add_dropout = [True,True,True,False,False,False]
SEED = 66478
log_dir = FLAGS.log_dir
learning_rate = FLAGS.learning_rate
train_steps = FLAGS.train_steps
worker_index = FLAGS.worker_index
is_chief = (worker_index == 0)
master = FLAGS.master

train_dir = FLAGS.data_dir + "train/*"
eval_dir = FLAGS.data_dir + "eval/*"

report_file = open(FLAGS.report_file,"w")

def log(msg):
    print(msg)
    report_file.write("%s\n" % msg)
    report_file.flush()

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

def log_auc(sess,predict_op, y_op,repeat):
    auc_start_time = time.time()
    log("AUC eval begins @ %f" % auc_start_time)
    auc_v = auc(sess,predict_op, y_op,repeat)
    log("Auc:%f" %auc_v)
    auc_end_time = time.time()
    log("AUC eval ends @ %f" % auc_end_time)
    auc_time = auc_end_time - auc_start_time
    log("AUC elapsed time : %f" %(auc_time))

def auc(sess, predict_op, y_op, repeat):
    predicts = []
    ys = []
    for i in xrange(repeat):
      log("AUC iter %d" % i)
      predict, y = sess.run([predict_op,y_op]) 
      predicts.append(predict)
      ys.append(y)
    eval_y = np.concatenate(predicts,axis=0)
    y_ = np.concatenate(ys,axis=0)
    return roc_auc_score(y_,eval_y)

def create_variables():
    global_step = tf.Variable(0, name="global_step", trainable=False)
    ws = []
    bs = []
    for i in range(1,len(dims)):
      w = tf.Variable(tf.truncated_normal([dims[i-1],dims[i]],stddev=1.0/math.sqrt(dims[i-1])),name=("hid_w" + str(i)))
      b = tf.Variable(tf.zeros([dims[i]]), name=("hid_b" + str(i)))
      ws.append(w)
      bs.append(b)
    return ws,bs,global_step

def model(input_data,ws,bs,train=False):
    prev = input_data
    for i in range(0,len(is_add_dropout)):
      hid_lin = tf.nn.xw_plus_b(prev,ws[i],bs[i])
      hid = tf.nn.elu(hid_lin) if i < len(is_add_dropout) else hid_lin
      if train and is_add_dropout[i]:
        hid = tf.nn.dropout(hid,0.5,seed = SEED)
      activation_summary(hid)
      prev = hid
    return hid

def train(logits, y_, global_step):
    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, y_) )
    tf.scalar_summary(cross_entropy.op.name+'/mean', cross_entropy)
    opt = tf.train.AdamOptimizer(learning_rate)
    grads = opt.compute_gradients(cross_entropy)
    train_op = opt.apply_gradients(grads, global_step=global_step)
    return train_op, cross_entropy
    

device_setter = tf.train.replica_device_setter(ps_tasks = FLAGS.num_parameter_servers,worker_device="/job:worker/task:%d" % worker_index)
with tf.device(device_setter):
    train_x,train_y_ = data_op(train_dir, INPUT_DIMENSION, batch_size = FLAGS.batch_size)
    ws,bs,global_step = create_variables()
    logits = model(train_x,ws,bs,True)
    logits = tf.reshape(logits,shape=[FLAGS.batch_size])
    train_op, loss = train(logits, train_y_,global_step)
    if is_chief:
      eval_x,eval_y_ = data_op(eval_dir, INPUT_DIMENSION, batch_size = FLAGS.eval_batch_size)
      eval_prediction = tf.sigmoid(model(eval_x,ws,bs, False))

sv =  tf.train.Supervisor(is_chief=is_chief,logdir = log_dir, global_step = global_step)
sess_config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True, device_filters=["/job:ps", "/job:worker/task:%d" % worker_index])

with sv.managed_session(master = master, config = sess_config) as sess:
    time_begin = time.time()
    log("Training begins @ %f" % time_begin)
    local_step = 0
    if is_chief:
      sv.loop(FLAGS.eval_secs, log_auc,(sess,eval_prediction, eval_y_, FLAGS.eval_iter))
    avg_step = 0
    avg_loss = 0.0
    while not sv.should_stop():
      _,loss_v,step = sess.run([train_op, loss, global_step])
      local_step += 1
      avg_step += 1
      avg_loss += loss_v
      now = time.time()
      if step % FLAGS.report_step ==0:
        log("%f: Worker %d: training step %d done (global step: %d),avg_step: %d, cross entropy = %g" % (now, worker_index, local_step, step,avg_step, avg_loss/avg_step))
        #log("%f: Worker %d: training step %d done (global step: %d),avg_step: %d, cross entropy = %g" % (now, worker_index, local_step, step,avg_step, avg_loss/avg_step/FLAGS.batch_size))
        avg_step = 0
        avg_loss = 0.0
      if step >= train_steps:
        break
    
    time_end = time.time()
    training_time = time_end - time_begin
    log("Training ends @ %f" % time_end)
    log("Training elapsed time: %f s" % training_time)
      #if is_chief and step % 100 == 0:
        #auc_v = auc(sess, eval_prediction, eval_y_, FLAGS.eval_iter)
        #print("Auc:%.3f" % (auc_v))
        #log("Auc:%.3f" % (auc_v))
    #if not sv.should_stop():
    #  auc_v = auc(sess, eval_prediction, eval_y_, FLAGS.eval_iter)
      #print("Auc:%.3f" % (auc_v))
      #log("Auc:%.3f" % (auc_v))
report_file.close()
    #print(sess.run([train_x,train_y_]))
    #print(sess.run(logits))
    #print(loss_v)
    #print(step)
    #print(sess.run(eval_prediction))

