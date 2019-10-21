# -*- coding: utf-8 -*-
"""
@author: Rajasekhar Mugada
@brief : Tensorflow tensorboard basics
In this example, visualizing tensorflow graph has been explained
"""

import tensorflow as tf

#################  display tensorflow graph #############################

#tensor flow graph can be visualized in tensorboard
#Tensorboard can be launched in webpage once we run through the code
#Taking simple addition example to visualize tensor graph       

#Clear previous graph if any
tf.reset_default_graph()     
       
#Create tensor flow graph
x1 = tf.constant(value = 1.0, name = 'x1_tf')
x2 = tf.constant(value = 2.0, name = 'x2_tf')
x3 = tf.add(x1, x2, name = 'result_tf')

#Launch tensor flow graph to execute operations
#Add logs to tensorflow event file, so that this file can be read by the 
#tensorboard later

#writer can be called from outside as well as inside of a session
writer = tf.summary.FileWriter("log_dir", tf.get_default_graph())
writer.close()  # close writer    
with tf.Session() as sess:
    res = sess.run(x3)
    print (res)
    
    #writer = tf.summary.FileWriter("log_dir", sess.graph())
    #writer.close()

"""
To launch tensorboard on web, 
First,  we have to specify the tensorflow event file that we have just created
(in "log_dir" folder) to tensorboard. We can also set the desired port. 
To do that execute the following line in cmd prompt also make sure the 
current directory as per your project folder
    
tensorboard --logdir "log_dir" --port 6008
    
Next, open the following url on your web browser to see the tensorboard graph   

"http://localhost:6008/"  

"""