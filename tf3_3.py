#coding:utf-8
import tensorflow as tf

x = tf.constant([[0.7,0.5]])
w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	print "y in/Users/taily/tf/tf3_4.py tf3_3.py is :\n", sess.run(y)

