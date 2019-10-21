# -*- coding: utf-8 -*-
"""
@author: Rajasekhar Mugada
@brief: Solving an equation in tensorflow using gradint descent optimizer
"""

import tensorflow as tf

################## constants #########################################
c = tf.constant(value = 4.0, dtype= tf.float32, shape = (1,1) , name = 'c')

#############  place holders  #########################################
#place holders : Inputs that can be fed just before the execution
x = tf.placeholder(dtype = tf.float32, shape = (None,1), name = 'x')
y = tf.placeholder(dtype = tf.float32, shape = (None,1), name = 'y')

###########  variables: to solve ######################################
w1 = tf.Variable(0.0, name = 'w1')
w2 = tf.Variable(0.0, name = 'w2')

##########  equation/graph to solve #####################################
# equation to solve : y = 1*x^2 -4*x + 4
y_solve = w1*tf.square(x) + w2*x + c

########### Set equation solving method ######################### 
#Using gradient descent method to solve the equation. Ex - learning rate = 0.01
err = y_solve - y
loss = tf.square(err)
optim_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

########## Add an operation to initialize all the global variables ############
init_op = tf.global_variables_initializer()

################## Launch and  execute the graph ##########################

#launch and execute the graph in a session
with tf.Session() as sess:
    #run the variable initalizer
    sess.run(init_op)

    #run optimizer to solve the equation
    for i in range(5000):
       #Fedding x, y values at run time to solve the equation
       sess.run(optim_step, feed_dict = {x: [[0.0],[1.0],[2.0]], 
                                         y: [[4.0],[1.0],[0.0]]})

    print('w1 : ',sess.run(w1), ' w2 : ',sess.run(w2))
