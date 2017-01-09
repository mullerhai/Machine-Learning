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


def data_op(pattern, n_features, batch_size = 50, num_epochs = None, shuffle = True, num_threads = 8):
    #filenames = tf.train.match_filenames_once(pattern)
    filenames = tf.matching_files(pattern)
    filename_queue = tf.train.string_input_producer(filenames,num_epochs = num_epochs, shuffle = shuffle)

    reader = tf.TextLineReader()
    key,value = reader.read(filename_queue)

    label_features = tf.decode_csv(value,[[1.0]] * (1+n_features))
    label = label_features[0]
    features = tf.pack(label_features[1:])
    min_after_dequeue = 40000
    capacity = min_after_dequeue + 30 * batch_size
    if shuffle:
      features_batch, label_batch = tf.train.shuffle_batch(
        [features, label], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue, num_threads = num_threads) 
    else:
      features_batch, label_batch = tf.train.batch(
        [features, label], batch_size=batch_size, capacity=capacity,
        num_threads = num_threads) 
    return features_batch, label_batch

#device_setter = tf.train.replica_device_setter(ps_tasks = 1,worker_device="/job:worker/task:%d" % 0)
#with tf.device(device_setter):
#    x,y = data_op("/home/fish.hy/dnn/data/train/*",215)
#    global_step = tf.Variable(0, name="global_step", trainable=False)
#log_dir = tempfile.mkdtemp()
#print(log_dir)
#sv =  tf.train.Supervisor(is_chief=True,logdir = log_dir, global_step = global_step)
#sess_config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
#with sv.managed_session(master = "grpc://tf1.kgb.et2:3222", config = sess_config) as sess:
#    print(sess.run([x,y]))

#coord = tf.train.Coordinator()
#with  tf.Session() as sess:
#with  tf.Session(target = "grpc://tf1.kgb.et2:3222") as sess:
#    sess.run(init_op)
#    tf.train.start_queue_runners(coord = coord)
#    print(sess.run([x,y]))
#    coord.request_stop()
#    coord.join()
#    print(sess.run([x,y]))
