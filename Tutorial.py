import tensorflow as tf 
import numpy as np 

x1 = tf.constant(5)
x2 = tf.constant(6)

result = tf.mul(x1, x2) #constant multiplication. this is faster than x1*x2.

print(result)

#sess = tf.Session()
#print(sess.run(result))
#sess.close()

#below is same as above. Use the one below.
with tf.Session() as sess:
	output = sess.run(result)
	print(output)

print (output)