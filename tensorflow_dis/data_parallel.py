#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

import time

np.random.seed(0)
tf.set_random_seed(0)

def get_batch(n):
    x = np.random.random(n)
    y = np.exp(x)
    return x,y

def leaky_relu(x,alpha=0.2):
    return tf.maximum(alpha*x,x)

def get_variables():
    ri = tf.random_uniform_initializer(0,1) # min 0 max 1
    zi = tf.zeros_initializer
    W1 = tf.get_variable("W1",shape=[1,16],initializer=ri)
    W2 = tf.get_variable("W2",shape=[16,32],initializer=ri)
    W3 = tf.get_variable("W3",shape=[32,1],initializer=ri)
    b1 = tf.get_variable("b1",shape=[16],initializer=zi)
    b2 = tf.get_variable("b2",shape=[32],initializer=zi)
    b3 = tf.get_variable("b3",shape=[1],initializer=zi)
    return [W1,W2,W3,b1,b2,b3]

def loss(x,t):
    [W1,W2,W3,b1,b2,b3] = get_variables()
    h1 = leaky_relu(tf.matmul(x,W1)+b1)
    h2 = leaky_relu(tf.matmul(h1,W2)+b2)
    y  = leaky_relu(tf.matmul(h2,W3)+b3)
    e  = tf.nn.l2_loss(y-t)
    return e

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


with tf.device("/job:master/task:0"):
    with tf.variable_scope("ps") as scope:
        x = tf.placeholder(tf.float32, shape=[None, 1])
        t = tf.placeholder(tf.float32, shape=[None, 1])
        total_loss = loss(x,t)
        tf.add_to_collection("master_x",x)
        tf.add_to_collection("master_t",t)
        tf.add_to_collection("total_loss",total_loss)

workers= [ "worker","worker_"]
for workername in workers:
    with tf.device("/job:%s/task:0" % workername):
        with tf.variable_scope("ps",reuse=True):

            x = tf.placeholder(tf.float32, shape=[None, 1])
            t = tf.placeholder(tf.float32, shape=[None, 1])
            e = loss(x,t)
            opt = tf.train.AdamOptimizer()
            gv = opt.compute_gradients(e)

            tf.add_to_collection("x",x)
            tf.add_to_collection("t",t)
            tf.add_to_collection("e",e)
            tf.add_to_collection("gv",gv)

with tf.device("/job:master/task:0"):
    with tf.variable_scope("ps",reuse=True) as scope:
        grads=[]
        for gv in tf.get_collection("gv"):
            grads.append(gv)
        grad= average_gradients(grads)
        opt = tf.train.AdamOptimizer()
        ag  = opt.apply_gradients(grad)


num_workers=2
num_batches=100

with tf.Session("grpc://localhost:2222") as sess:

    sess.run(tf.initialize_all_variables())
    start = time.time()

    for i in range(1000):

        x,t = get_batch(num_batches)
        x = x.astype(np.float32).reshape(num_batches,1)
        t = t.astype(np.float32).reshape(num_batches,1)

        xs= np.split(x,num_workers)
        ts= np.split(t,num_workers)

        feed_dict={}
        for j in range(num_workers):
            feed_dict[tf.get_collection("x")[j]] = xs[j]
            feed_dict[tf.get_collection("t")[j]] = ts[j]

        sess.run(ag, feed_dict=feed_dict)

        mx = tf.get_collection("master_x")[0]
        mt = tf.get_collection("master_t")[0]
        e = tf.get_collection("total_loss")[0]
        loss = sess.run(e,feed_dict={mx: x, mt:t})

        if i%100==0:
            print loss

    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time)) + "[sec]"


