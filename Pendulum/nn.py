import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def s_const_grid(s):
    n = 1000
    xs = np.linspace(-1,1,n)[:,np.newaxis]
    return xs, np.concatenate((np.ones((n,1)) * s[np.newaxis,:], xs), 1)

class ControlNN:
    def __init__(self):
        tf_random_seed = 6
        nonlinearity = tf.nn.relu
        self.keep_prob_train_val = 1.0

        def unif_fanin_mat(shape):
            b = np.sqrt(3 * self.keep_prob_train_val / shape[0])
            initial = tf.random_uniform(shape, minval=-b, maxval=b, seed=tf_random_seed)
            return tf.Variable(initial)

        def bias(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        self.n_s = 2
        self.n_a = 1
        self.n_sa = 3
        n_hidden = 5
        self.n_1 = n_hidden
        self.n_2 = n_hidden
        self.one_layer_only = True
        self.keep_prob = tf.placeholder('float')
        self.sa_learn = tf.placeholder('float', shape=[None,self.n_sa])
        self.W_sa_1 = unif_fanin_mat([self.n_sa, self.n_1])
        self.b_1 = bias([self.n_1])
        self.W_1_2 = unif_fanin_mat([self.n_1, self.n_2])
        self.b_2 = bias([self.n_2])
        self.W_2_q = unif_fanin_mat([self.n_2,1])
        self.b_q = bias([1])

        def q_from_input(i):
            o1 = tf.nn.dropout(nonlinearity(tf.matmul(i, self.W_sa_1) + self.b_1), self.keep_prob)
            if self.one_layer_only:
                return tf.matmul(o1, self.W_2_q) + self.b_q

            o2 = tf.nn.dropout(nonlinearity(tf.matmul(o1, self.W_1_2) + self.b_2), self.keep_prob)
            return tf.matmul(o2, self.W_2_q) + self.b_q

        self.q_learn = q_from_input(self.sa_learn)
        self.y_learn = tf.placeholder('float', shape = [None])
        self.learn_error = tf.reduce_mean(tf.square(self.y_learn - self.q_learn))
        self.learn_opt = tf.train.AdamOptimizer().minimize(self.learn_error)

        self.s_query = tf.placeholder('float', shape=[1, self.n_s])
        self.a_query = unif_fanin_mat([1, self.n_a])
        self.sa_query = tf.concat(1, [self.s_query, self.a_query])
        self.q_query = q_from_input(self.sa_query)

        query_opt = tf.train.AdamOptimizer(1.0)
        query_grads_and_vars = query_opt.compute_gradients(self.q_query, [self.a_query])
        # list of tuples (gradient, variable).
        query_grads_and_vars[0] = (-query_grads_and_vars[0][0], query_grads_and_vars[0][1])
        self.apply_query_grads = query_opt.apply_gradients(query_grads_and_vars)

        self.init_op = tf.initialize_all_variables()

        self.sess = tf.Session()
        self.sess.run(self.init_op)

    def print_params(self):
        for (name, param) in zip(['W_sa_1', 'b_1', 'W_1_2', 'b_2', 'W_2_q', 'b_q'],
                [self.W_sa_1, self.b_1, self.W_1_2, self.b_2, self.W_2_q, self.b_q]):
            print name
            print self.sess.run(param)

    def q_from_sa(self, sa_vals):
        return self.sess.run(self.q_learn, feed_dict={self.sa_learn: sa_vals, self.keep_prob: 1.0})

    def graph_output(self, s):
        assert self.n_a == 1
        xs, inputs = s_const_grid(s)
        outputs = self.q_from_sa(inputs)
        plt.plot(xs, outputs)
        plt.show()
        return outputs

    def get_best_a(self, s):
        for i in range(100):
            #if i%10 == 0:
            #    print i
            self.sess.run(self.apply_query_grads, feed_dict={self.s_query: s[np.newaxis,:], self.keep_prob: 1.0})
            #self.query_opt.run(feed_dict={self.s_query: s})

        return self.sess.run(self.a_query)

    def mse_q(self, sa_vals, y_vals):
        return self.sess.run(self.learn_error,
                feed_dict={self.sa_learn: sa_vals, self.y_learn: y_vals, self.keep_prob: 1.0})

    def train(self, sa_vals, y_vals):
        self.sess.run(self.learn_opt, feed_dict={self.y_learn: y_vals,
            self.sa_learn: sa_vals, self.keep_prob: self.keep_prob_train_val})
