import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from optimize.snopt7 import SNOPT_solver
from utils import *

class ControlNN:
    def __init__(self, load_file=None):
        tf_random_seed = 40
        nonlinearity = tf.nn.relu
        self.keep_prob_train_val = 1.0
        self.floatX = 'float32'

        def unif_fanin_mat(shape, name):
            b = np.sqrt(3 * self.keep_prob_train_val / shape[0])
            initial = tf.random_uniform(shape, minval=-b, maxval=b, seed=tf_random_seed, dtype=self.floatX)
            return tf.Variable(initial, name=name)

        def bias(shape, name):
            initial = tf.constant(0.1, shape=shape, dtype=self.floatX)
            return tf.Variable(initial, name=name)

        self.snopt = SNOPT_solver()
        self.snopt.setOption('Major print level', 0)

        self.profiler = Profiler()

        conf = read_conf('pendulum.conf')
        self.n_s = 2
        self.n_a = 1
        self.n_sa = 3
        n_hidden = conf['n_hidden']
        self.n_1 = n_hidden
        self.n_2 = n_hidden
        self.one_layer_only = True
        self.n_minibatch = conf['minibatch_size']
        self.n_batches = conf['num_batches']
        self.n_megabatch = self.n_minibatch * self.n_batches
        self.max_a_min_iters = 5
        self.max_abs_torque = conf['max_torque']
        self.max_torques = np.array([[self.max_abs_torque]], dtype=self.floatX)
        self.max_torques_p = np.ones((self.n_megabatch,1)) * np.array([[self.max_abs_torque]], dtype=self.floatX)
        self.min_torques = np.array([[-self.max_abs_torque]], dtype=self.floatX)
        self.min_torques_p = np.ones((self.n_megabatch,1)) * np.array([[-self.max_abs_torque]], dtype=self.floatX)

        self.sess = tf.Session()
        self.keep_prob = tf.placeholder(self.floatX)
        self.sa_learn = tf.placeholder(self.floatX, shape=[None,self.n_sa])
        self.W_sa_1 = unif_fanin_mat([self.n_sa, self.n_1], 'W_sa_1')
        self.b_1 = bias([self.n_1], 'b_1')
        self.W_1_2 = unif_fanin_mat([self.n_1, self.n_2], 'W_1_2')
        self.b_2 = bias([self.n_2], 'b_2')
        self.W_2_q = unif_fanin_mat([self.n_2,1], 'W_2_q')
        self.b_q = bias([1], 'b_q')
        name_var_pairs = zip(['W_sa_1', 'b_1', 'W_1_2', 'b_2', 'W_2_q', 'b_q'],
                [self.W_sa_1, self.b_1, self.W_1_2, self.b_2, self.W_2_q, self.b_q])
        self.name_var_dict = {i:j for (i,j) in name_var_pairs}

        def q_from_input(i):
            o1 = tf.nn.dropout(nonlinearity(tf.matmul(i, self.W_sa_1) + self.b_1), self.keep_prob)
            if self.one_layer_only:
                return tf.matmul(o1, self.W_2_q) + self.b_q

            o2 = tf.nn.dropout(nonlinearity(tf.matmul(o1, self.W_1_2) + self.b_2), self.keep_prob)
            return tf.matmul(o2, self.W_2_q) + self.b_q

        self.o1 = nonlinearity(tf.matmul(self.sa_learn, self.W_sa_1) + self.b_1)
        self.q_learn = q_from_input(self.sa_learn)
        self.y_learn = tf.placeholder(self.floatX, shape = [None, 1])
        self.learn_error = tf.reduce_mean(tf.square(self.y_learn - self.q_learn))

        self.max_a_time_limit = conf['max_a_time_limit']

        global_step = tf.Variable(0, trainable=False)
        self.learn_rate = tf.train.exponential_decay(conf['initial_learn_rate'], global_step,
                conf['learn_rate_half_life'], 0.5, staircase=False)
        self.learn_opt = tf.train.AdamOptimizer(self.learn_rate).minimize(self.learn_error, global_step=global_step)

        def query_setup(n_sample):
            s_query = tf.placeholder('float', shape=[n_sample, self.n_s])
            a_query = unif_fanin_mat([n_sample, self.n_a], 'a_query')
            min_cutoff = tf.matmul(np.ones((n_sample, 1), dtype=self.floatX), self.min_torques)
            max_cutoff = tf.matmul(np.ones((n_sample, 1), dtype=self.floatX), self.max_torques)
            # print "min cutoff:", self.sess.run(min_cutoff)
            # print "max cutoff:", self.sess.run(max_cutoff)
            a_query_clipped = tf.minimum(tf.maximum(min_cutoff, a_query), max_cutoff)

            #sa_query = tf.concat(1, [s_query, a_query_clipped])
            sa_query = tf.concat(1, [s_query, a_query])
            q_query = q_from_input(sa_query)
            q_query_mean = tf.reduce_mean(q_query)

            query_opt = tf.train.AdamOptimizer(0.1)
            query_grads_and_vars = query_opt.compute_gradients(q_query_mean, [a_query])
            # list of tuples (gradient, variable).
            query_grads_and_vars[0] = (-query_grads_and_vars[0][0], query_grads_and_vars[0][1])
            apply_query_grads = query_opt.apply_gradients(query_grads_and_vars)
            return s_query, a_query, a_query_clipped, sa_query, q_query, q_query_mean, apply_query_grads

        self.s_query, self.a_query, self.a_query_clipped, self.sa_query, self.q_query, \
                self.q_query_mean, self.apply_query_grads = query_setup(1)
        self.s_query_p, self.a_query_p, self.a_query_clipped_p, self.sa_query_p, self.q_query_p, \
                self.q_query_mean_p, self.apply_query_grads_p = query_setup(self.n_megabatch)

        self.sym_grad_p = tf.gradients(self.q_query_mean_p, self.a_query_p)
        self.sym_grad = tf.gradients(self.q_query_mean, self.a_query)

        self.saver = tf.train.Saver(self.name_var_dict)

        self.init_op = tf.initialize_all_variables()
        self.sess.run(self.init_op)

        if load_file != None:
            self.load_model(load_file)

    def __del__(self):
        self.sess.close()

    def print_params(self):
        for (name, param) in enumerate(self.name_var_dict):
            print name
            print self.sess.run(param)
        print

    def q_from_sa(self, sa_vals):
        return self.sess.run(self.q_learn, feed_dict={self.sa_learn: sa_vals, self.keep_prob: 1.0})

    def q_query_from_s(self, s_vals):
        return self.sess.run(self.q_query, feed_dict={self.s_query: s_vals[np.newaxis,:], self.keep_prob: 1.0})

    def q_query_from_s_p(self, s_vals):
        return self.sess.run(self.q_query_p, feed_dict={self.s_query_p: s_vals, self.keep_prob: 1.0})

    def o1_from_sa(self, sa_vals):
        return self.sess.run(self.o1, feed_dict={self.sa_learn: sa_vals, self.keep_prob: 1.0})

    def s_const_grid(self, s, xr, n=10000):
        # [[s x_1], [s x_2], ..., [s x_n]]
        s = np.array(s)
        xs = np.linspace(xr[0], xr[1], n)[:,np.newaxis]
        return xs, np.concatenate((np.ones((n,1)) * s[np.newaxis,:], xs), 1)

    def manual_max_a(self, s, xr=None):
        assert self.n_a == 1
        if xr == None:
            xr = self.max_abs_torque * np.array([-1.,1])
        xs, inputs = self.s_const_grid(s, xr)
        outputs = self.q_from_sa(inputs).flatten()
        best_input = np.argmax(outputs)
        return np.array([inputs[best_input][-1], outputs[best_input]])

    def manual_max_a_p(self, s, xr=None):
        assert self.n_a == 1
        ans = []
        for i in s:
            ans.append(self.manual_max_a(i, xr))
        return np.array(ans)

    def get_best_a_p(self, s, is_p, num_tries, init_a=None, tolerance=0.01):
        #TODO: benchmark different init methods

        assert (is_p and len(s.shape) == 2 and s.shape[0] == self.n_megabatch) or (not is_p and len(s.shape) == 1)
        n_batch = s.shape[0] if is_p else 1

        inf = 1.0e20
        obj_row = 1

        x_names = np.array(['x' + str(i) for i in range(n_batch)])
        F_names = np.array(['F1', 'F2'])
        xlow = np.array([-self.max_abs_torque for i in range(n_batch)])
        xupp = np.array([self.max_abs_torque for i in range(n_batch)])

        Flow = np.array([-inf, -inf])
        Fupp = np.array([inf, inf])

        A = np.array([[0 for _ in range(n_batch)], [1 for _ in range(n_batch)]])
        G = np.array([[2 for _ in range(n_batch)], [0 for _ in range(n_batch)]])

        ans_a, ans_q = None, None

        def check_timeout(start_time, time_limit):
            if time.time() - start_time > time_limit:
                err_msg = 'error!!! max a timeout: s=%s is_p=%s num_tries=%s init_a=%s' % (s, is_p, num_tries, init_a)
                print err_msg
                return True
            return False

        def inner_p(init_a, time_limit):
            start_time = time.time()
            if init_a == None:
                init_a = self.min_torques_p + (self.max_torques_p - self.min_torques_p) * \
                        np.random.random((self.n_megabatch, self.n_a)) / 2.0


            #def obj(a):
            #    output = self.sess.run(self.q_query_mean_p, feed_dict={self.s_query_p: s,
            #        self.a_query_p: a[:,np.newaxis], self.keep_prob: 1.0})
            #    return -output
            def grad(a):
                g = self.sess.run(self.sym_grad_p, feed_dict={self.s_query_p: s, self.a_query_p: a[:,np.newaxis],
                    self.keep_prob: 1.0})
                ret_grad = -g[0].flatten()
                return ret_grad

            def objFG(status,a,needF,needG,cu,iu,ru):
                F = np.array([-self.sess.run(self.q_query_mean_p, feed_dict={self.s_query_p: s,
                    self.a_query_p: a[:,np.newaxis], self.keep_prob: 1.0}), 0])
                G = grad(a)
                return status, F, G

            self.snopt.snopta(n=n_batch, nF=2, usrfun=objFG, x0=init_a, xlow=xlow, xupp=xupp,
                Flow=Flow, Fupp=Fupp, ObjRow=obj_row, A=A, G=G, xnames=x_names, Fnames=F_names)

            #print 'results'
            #print self.snopt.x
            #print self.snopt.major_itns
            #print self.snopt.objective

            res_x = self.snopt.x

            q_final = self.sess.run(self.q_query_p, feed_dict={self.s_query_p: s,
                self.a_query_p: res_x, self.keep_prob: 1.0})

            return res_x, q_final

        def inner(init_a, time_limit):
            start_time = time.time()
            count = [0]
            if init_a == None:
                #init_a = np.zeros([1, self.n_a])
                init_a = self.min_torques + (self.max_torques - self.min_torques) * \
                        np.random.random((1, self.n_a))

            def grad(a):
                g = self.sess.run(self.sym_grad, feed_dict={self.s_query: s[np.newaxis,:],
                    self.a_query: a[:,np.newaxis], self.keep_prob: 1.0})
                ret_grad = -g[0].flatten()
                return ret_grad

            def objFG(status,a,needF,needG,cu,iu,ru):
                F = np.array([-self.sess.run(self.q_query_mean, feed_dict={self.s_query: s[np.newaxis,:],
                    self.a_query: a[:,np.newaxis], self.keep_prob: 1.0}), 0])
                G = grad(a)
                count[0] += 1
                if count[0] > 10:
                    status = -1
                return status, F, G

            self.snopt.snopta(n=n_batch, nF=2, usrfun=objFG, x0=init_a, xlow=xlow, xupp=xupp,
                    Flow=Flow, Fupp=Fupp, ObjRow=obj_row, A=A, G=G)#, xnames=x_names, Fnames=F_names)

            res_x = self.snopt.x
            print init_a, res_x

            q_final = self.sess.run(self.q_query, feed_dict={self.s_query: s[np.newaxis,:],
                self.a_query: res_x, self.keep_prob: 1.0})

            return res_x, q_final

        inner_function = inner_p if is_p else inner
        iter_time_limit = self.max_a_time_limit / num_tries

        for i in range(num_tries):
            a, q = inner_function(init_a, iter_time_limit) if i == 0 else inner_function(None, iter_time_limit)
            if i == 0:
                ans_a, ans_q = a, q
                continue
            for i in range(len(q)):
                if q[i] > ans_q[i]:
                    ans_q[i] = q[i]
                    ans_a[i] = a[i]

        return ans_a, ans_q

    def mse_q(self, sa_vals, y_vals):
        print 'learn_rate', self.sess.run(self.learn_rate)
        return self.sess.run(self.learn_error,
                feed_dict={self.sa_learn: sa_vals, self.y_learn: y_vals, self.keep_prob: 1.0})

    def train(self, sa_vals, y_vals):
        self.sess.run(self.learn_opt, feed_dict={self.y_learn: y_vals,
            self.sa_learn: sa_vals, self.keep_prob: self.keep_prob_train_val})
        #print 'learn_rate', self.sess.run(self.learn_rate)

    def save_model(self, save_path):
        self.saver.save(self.sess, save_path)

    def load_model(self, load_path):
        self.saver.restore(self.sess, load_path)
