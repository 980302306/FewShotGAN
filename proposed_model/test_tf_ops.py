# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 11:27:23 2019
test tf ops
@author: liuhuaqing
"""

import tensorflow as tf

input=tf.constant([0,1,2,3],dtype=tf.float32)
output=tf.nn.softplus(input)

with tf.Session() as sess:
    print('input:')
    print(sess.run(input))
    print('output:')
    print(sess.run(output))
    sess.close()
    
input=tf.constant([0,1,2,3],dtype=tf.float32)
output=tf.reduce_logsumexp(input)

with tf.Session() as sess:
    print('input:')
    print(sess.run(input))
    print('output:')
    print(sess.run(output))
    sess.close()
    
import numpy as np
np.log(np.sum(np.exp([0,1,2,3])))