import tensorflow as tf

x_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_train = [[0], [1], [1], [0]]

x = tf.placeholder(tf.float32, shape=[None, 2]);
y = tf.placeholder(tf.float32, shape=[None, 1]);

w1 = tf.Variable(tf.random_normal([2, 4]))
b1 = tf.Variable(tf.random_normal([4]))

net1 = tf.matmul(x, w1) + b1;
logits1 = tf.nn.sigmoid(net1)

w2 = tf.Variable(tf.random_normal([4, 1]))
b2 = tf.Variable(tf.random_normal([1]))

net2 = tf.matmul(logits1, w2) + b2
logits2 = tf.nn.sigmoid(net2)

loss_operation = tf.reduce_mean(tf.square(logits2 - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss_operation)

x_test = [[1, 0], [0, 1], [1, 1], [0, 0], [1, 0], [0, 0]]
y_test = [[1], [1], [0], [0], [1], [0]]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(4000):
        _, loss_train = sess.run([optimizer, loss_operation], feed_dict={x: x_train, y: y_train})
        print("Train Loss (", i, ") : ", loss_train)

    prediction = tf.equal(logits2, y)
    accuracy_operation = tf.reduce_mean(tf.cast(prediction, tf.float32))
    logits2_test, loss_test = sess.run([logits2, loss_operation], feed_dict={x: x_test, y: y_test})

    # print("Test Accuracy : ", accuracy_test)
    print("Test Loss : " , loss_test)
    print("Test Logits : ", logits2_test)
    print("Test Target : ", y_test)
