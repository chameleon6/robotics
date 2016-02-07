import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def s_const_grid(s):
    n = 1000
    xs = np.linspace(-100,100,n)[:,np.newaxis]
    return xs, np.concatenate((np.ones((n,1)) * s[np.newaxis,:], xs), 1)

class ControlNN:
    def __init__(self):
        tf_random_seed = 3
        self.n_s = 2
        self.n_a = 1
        self.n_sa = 3
        self.n_1 = 10
        self.n_2 = 10
        self.sa_learn = tf.placeholder("float", shape=[None,self.n_sa])
        self.W_sa_1 = tf.Variable(tf.random_normal([self.n_sa, self.n_1], seed=tf_random_seed))
        self.W_1_2 = tf.Variable(tf.random_normal([self.n_1, self.n_2], seed=tf_random_seed))
        self.W_2_q = tf.Variable(tf.random_normal([self.n_2,1], seed=tf_random_seed))
        self.q_learn = tf.matmul(tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(self.sa_learn, self.W_sa_1)),
            self.W_1_2)), self.W_2_q)

        self.s_query = tf.placeholder("float", shape=[1, self.n_s])
        self.a_query = tf.Variable(tf.random_normal([1, self.n_a], seed=tf_random_seed))
        self.sa_query = tf.concat(1, [self.s_query, self.a_query])
        self.q_query = tf.matmul(tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(self.sa_query, self.W_sa_1)),
            self.W_1_2)), self.W_2_q)

        query_opt = tf.train.AdamOptimizer(1.0) # adam
        query_grads_and_vars = query_opt.compute_gradients(self.q_query, [self.a_query])

        # grads_and_vars is a list of tuples (gradient, variable).
        query_grads_and_vars[0] = (-query_grads_and_vars[0][0], query_grads_and_vars[0][1])

        # Ask the optimizer to apply the capped gradients.
        self.apply_query_grads = query_opt.apply_gradients(query_grads_and_vars)

        self.init_op = tf.initialize_all_variables()

        self.sess = tf.Session()
        self.sess.run(self.init_op)

    def graph_output(self, s):
        assert self.n_a == 1
        xs, inputs = s_const_grid(s)
        outputs = self.sess.run(self.q_learn, feed_dict={self.sa_learn: inputs})
        plt.plot(xs, outputs)
        plt.show()
        return outputs

    def get_best_a(self, s):
        # Compute the gradients for a list of variables.

        for i in range(100):
            #if i%10 == 0:
            #    print i
            self.sess.run(self.apply_query_grads, feed_dict={self.s_query: s[np.newaxis,:]})
            #self.query_opt.run(feed_dict={self.s_query: s})

        return self.sess.run(self.a_query)
