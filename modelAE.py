import tensorflow as tf
import scipy.io as sio
from configparser import ConfigParser
from tensorflow.contrib.layers import xavier_initializer_conv2d
import matplotlib.pyplot as plt

b, h, w, c = 10, 256, 342, 2
hidden_dim = 512

x_  = tf.placeholder(dtype=tf.float32, shape = (b,h,w,c))
x = tf.contrib.layers.flatten(x_)
encode = {'weights': tf.Variable(tf.truncated_normal(
            [h*w*c, hidden_dim], dtype=tf.float32)),
            'biases': tf.Variable(tf.truncated_normal([hidden_dim],
                                                      dtype=tf.float32))}
decode = {'biases': tf.Variable(tf.truncated_normal([h*w*c], dtype=tf.float32)),'weights': tf.Variable(tf.truncated_normal(
            [hidden_dim, h*w*c], dtype=tf.float32))}

encoded = tf.nn.tanh(tf.matmul(x, encode['weights']) + encode['biases'])

decoded = tf.matmul(encoded, decode['weights'])+decode['biases']


loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(x,decoded))))     #recon loss (u can use crossentropy loss = -tf.reduce_mean(x_ * tf.log(decoded)))

train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        [out, _loss,_] = sess.run([decoded, loss,train], feed_dict={x_:inp})
        print(i, _loss)