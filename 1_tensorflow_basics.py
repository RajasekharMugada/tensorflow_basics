# -*- coding: utf-8 -*-
"""
@author: Rajasekhar Mugada
@brief : Getting started with Tensorflow. 
In this example baisic tensorflow structure and datatypes are explained
"""

import tensorflow as tf


#STEP1 : Create graph with all the required variables and operations

#Tensorflow data types 

################## constants #########################################
'''
tf.constant(
    value,
    dtype=None,
    shape=None,
    name='Const'
)

Commonly used tensorflow dtype : 
    tf.int8/16/32/64, tf.uint8/16/32/64
    tf.bool, tf.string, tf.float16/32/64, tf.complex64/128
'''
c = tf.constant(value = 4.0, dtype= tf.float32, shape = (1,1) , name = 'c')

#############  place holders  #########################################
#place holders : Inputs that can be fed just before the execution
x = tf.placeholder(dtype = tf.float32, name = 'x')

###########  variables: to solve ######################################
'''
w = tf.Variable(<initial-value>, name = <optional-name>)
To assign a new value to the variable use 'assign()'
w.assign(W + 1.0)
'''
w = tf.Variable(1.0, name = 'w')

##########  equation to evaluate #####################################
y = w*x + c

########## Add an operation to initialize all the global variables ############
init_op = tf.global_variables_initializer()


#STEP2 : Launch the session to execute the above defined graph


################## Launch and  execute the session ##########################

#launch the graph in a session

#method 1 - using session.close() method:
sess = tf.Session()
sess.run(init_op)
result = sess.run(y, feed_dict = {x: [0.0, 1.0]})
sess.close()
print ('Method #1 - result :', result)


#method 2 - using context manager:
#In this mehtod we do not need to call session.close() explicitly
with tf.Session() as sess:
    #run the variable initializer
    sess.run(init_op)
    #save graph events
    writer = tf.summary.FileWriter("basic", sess.graph)
    
    #execute the graph
    result = sess.run(y, feed_dict = {x: [0.0, 1.0]})
    print('Method #2 - result :', result)