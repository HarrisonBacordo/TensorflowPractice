import tensorflow as tf
import numpy as np


def generate_dataset():
    dat = np.zeros((10000, 2))
    answers = list()
    for n in range(10000):
        rand = np.random.randint(0, 2, 2)
        dat[n] = rand
        if 1 in rand and 0 in rand:
            answers.append(1)
        else:
            answers.append(0)
    dat = np.float32(dat)
    return dat, answers


num_input = 2
num_hidden = 20
num_output = 2
num_epochs = 5
batch_size = 100
learning_rate = 0.01

x = tf.placeholder(shape=[None, num_input], dtype=tf.float32)
y = tf.placeholder(shape=[None, num_output], dtype=tf.int32)

weights = {
    'in_h1': tf.Variable(tf.random_normal(shape=[num_input, num_hidden])),
    'h1_h2': tf.Variable(tf.random_normal(shape=[num_hidden, num_hidden])),
    'h2_out': tf.Variable(tf.random_normal(shape=[num_hidden, num_output]))
}

biases = {
    'in_h1': tf.Variable(tf.random_normal(shape=[num_hidden])),
    'h1_h2': tf.Variable(tf.random_normal(shape=[num_hidden])),
    'h2_out': tf.Variable(tf.random_normal(shape=[num_output]))
}

in_h1 = tf.add(tf.matmul(x, weights['in_h1']), biases['in_h1'])
in_h1 = tf.nn.relu(in_h1)

h1_h2 = tf.add(tf.matmul(in_h1, weights['h1_h2']), biases['h1_h2'])
h1_h2 = tf.nn.relu(h1_h2)

h2_out = tf.add(tf.matmul(h1_h2, weights['h2_out']), biases['h2_out'])

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=h2_out))
optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
init = tf.initialize_all_variables()

with tf.Session() as sess:
    data, labels = generate_dataset()
    sess.run(init)

    for i in range(data.size):
        feature = np.reshape(data[i], newshape=[1, 2])
        answer = sess.run(tf.one_hot(labels[i], depth=2))
        answer = np.reshape(answer, newshape=[1, 2])
        _, c = sess.run([optimizer, loss], feed_dict={x: feature, y: answer})
        print(c)
