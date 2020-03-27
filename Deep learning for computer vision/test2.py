import tensorflow as tf

num_input = 784
num_classes = 10
batch_size = 100
total_batch = 200

x = tf.placeholder(tf.float32, shape=[None, num_input])
y = tf.placeholder(tf.float32, shape=[None, num_classes])

w = tf.Variable(tf.random_normal([num_input, num_classes]))
b = tf.Variable(tf.random_normal([num_classes]))

logits = tf.matmul(x, w) + b