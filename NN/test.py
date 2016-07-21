import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
from tf_utils import batch_norm

floatX = 'float32'
n_in = 10
n_hid = 10

xs = 30*np.random.randn(1000000, n_in) + 10
ys = np.sum(xs, 1)

def run_once(bn):
    '''
    def make_var(shape):
        return tf.Variable(np.random.random(shape), dtype=floatX)

    x = tf.placeholder(dtype=floatX, shape=[None,n_in])
    x2 = batch_norm(x, True) if bn else x
    W = make_var((n_in, n_hid))
    b = make_var(n_hid)
    V = make_var((n_hid, 1))

    h = tf.matmul(x2, W) + b
    if bn: h = batch_norm(h, True)

    out = tf.reshape(tf.matmul(tf.nn.relu(h), V), [-1])
    y = tf.placeholder(dtype=floatX, shape=[None])

    err = tf.reduce_mean(tf.square(y - out))
    train_step = tf.train.AdamOptimizer(0.001).minimize(err)

    n_minibatch = 100
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    errs = []
    for i in range(40000):
        sl = slice(n_minibatch*i, n_minibatch*(i+1), 1)
        fd = {x: xs[sl], y: ys[sl]}
        sess.run(train_step, feed_dict=fd)
        if i % 100 == 0:
            e = sess.run([err], feed_dict=fd)
            print i, e
            errs.append(e)

    return errs

    '''

    w1_initial = np.random.normal(size=(784,100)).astype(np.float32)
    w2_initial = np.random.normal(size=(100,100)).astype(np.float32)
    w3_initial = np.random.normal(size=(100,10)).astype(np.float32)

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    w1 = tf.Variable(w1_initial)
    b1 = tf.Variable(tf.zeros([100]))
    z1 = tf.matmul(x,w1)+b1
    if bn: z1 = batch_norm(z1, True)
    l1 = tf.nn.sigmoid(z1)

    w2 = tf.Variable(w2_initial)
    b2 = tf.Variable(tf.zeros([100]))
    z2 = tf.matmul(l1,w2)+b2
    if bn: z2 = batch_norm(z2, True)
    l2 = tf.nn.sigmoid(z2)

    w3 = tf.Variable(w3_initial)
    b3 = tf.Variable(tf.zeros([10]))
    y  = tf.nn.softmax(tf.matmul(l2,w3)+b3)

    cross_entropy = -tf.reduce_sum(y_*tf.log(y))

    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.arg_max(y,1),tf.arg_max(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    zs, acc = [], []

    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    for i in range(150000):
        batch = mnist.train.next_batch(60)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
        if i % 100 is 0:
            print i
            res = sess.run([accuracy,z2],
              feed_dict={x: mnist.test.images, y_: mnist.test.labels})
            acc.append(res[0])
            zs.append(np.mean(res[1],axis=0)) # record the mean value of z2 over the entire test set

    zs, acc = np.array(zs), np.array(acc)
    return zs, acc

(z, a), (z2, a2) = run_once(False), run_once(True)

fig, axes = plt.subplots(5, 2, figsize=(6,12))
fig.tight_layout()

for i, ax in enumerate(axes):
    ax[0].set_title("Without BN")
    ax[1].set_title("With BN")
    ax[0].plot(z[:,i])
    ax[1].plot(z2[:,i])
#plt.plot(a, 'g')
#plt.plot(a2, 'r')

plt.show()
