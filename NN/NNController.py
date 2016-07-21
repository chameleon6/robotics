import numpy as np
import os
import sys
import time
import random
import cPickle as pickle
import matplotlib.pyplot as plt
import scipy.io as sio
from utils import *
from nn import ControlNN
from TransitionContainer import TransitionContainer
from NetVisualizer import NetVisualizer
from matlab_rewards import *
from sklearn import gaussian_process
from scipy.stats import norm

prof = Profiler()

class NNController:

    def __init__(self, conf):
        self.profiler = Profiler()
        self.profiler.tic('total controller run time')

        self.matlab_state_file = os.getcwd() + '/matlab_state_file.out'
        self.python_action_file = os.getcwd() + '/python_action_file.out'
        self.last_no_matlab_net_path = 'last_no_matlab_net.out'

        conf = read_conf(conf)

        self.conf = conf
        self.max_torque = conf['max_torque']
        self.model_name = conf['name']
        self.bang_action = conf['bang_action']
        #self.old_net_update_time = conf['old_net_update_time']
        self.old_net_update_delay = conf['old_net_update_delay']
        if self.old_net_update_delay > 0:
            self.use_old_net = True
        else:
            self.use_old_net = False

        self.min_train_gap = conf['min_train_gap']
        self.train_per_iter = conf['train_per_iter']
        self.min_action_gap = conf['min_action_gap']
        self.final_epsilon = conf['final_epsilon']
        self.epsilon_anneal_time = conf['epsilon_anneal_time']
        self.no_op_time = conf['no_op_time']
        self.n_minibatch = conf['minibatch_size']
        self.n_batches = conf['num_batches']
        self.n_megabatch = self.n_minibatch * self.n_batches
        self.gamma = conf['gamma']
        self.sim_dt = conf['sim_dt']
        self.start_epsilon = conf['start_epsilon']
        self.mse_freq = conf['mse_freq']
        self.error_type = conf['error_type']
        self.RL_train = conf['RL_train']

        assert self.start_epsilon >= self.final_epsilon

        self.epsilon = self.start_epsilon
        #last_self.old_net_update_time = 0.0
        self.last_old_net_update_count = 0
        self.net_update_count = 0
        self.last_train_time = 0.0
        self.last_action_time = 0.0
        self.time_step = 1
        self.times_trained = 0
        self.last_state = None
        self.last_action = None
        self.succeeded = False
        self.train_t = -self.no_op_time
        #self.sim_start_time = self.train_t - 10
        self.sim_num = 0
        self.sim_t = 0

        self.sa_queue = None
        self.ys_queue = None
        self.minibatch_index = self.n_batches

        self.a_hists = []
        self.current_a_hist= []
        self.s_hists = []
        self.current_s_hist= []
        self.t_hists = []
        self.current_t_hist= []
        self.mse_hist = []
        self.mse_hist2 = []
        self.rmse_rel_hist = []
        self.rmse_rel_hist2 = []
        self.learn_rate_hist = []
        self.certainty_hist = []
        self.correct_rate_hist = []
        self.q_mean_hist = []
        self.global_correct_rates = []
        self.global_mse_hist = []

        self.action_counts = np.array([1,1,1,1.])

        self.transitions = TransitionContainer(conf['max_history_len'])

        if 'ref_transitions_file' in conf:
            with open(conf['ref_transitions_file'], 'rb') as f:
                self.all_ref_transitions = pickle.load(f)
            np.random.shuffle(self.all_ref_transitions)
            self.ref_transitions = self.all_ref_transitions[:400]

        if 'transitions_seed' in conf:
            self.transitions.container = pickle.load(open(conf['transitions_seed'], 'rb'))

        action_conf = conf.copy()
        action_conf['n_a'] = 0
        action_conf['n_q'] = 6
        action_conf['learn_rate'] = 0.01
        if 'action_net_file' in conf:
            print 'loading action net'
            self.action_net = ControlNN(action_conf, load_path=conf['action_net_file'])
        else:
            self.action_net = ControlNN(action_conf, load_path=None)

        '''
        sas_conf = conf.copy()
        sas_conf['n_s'] = conf['n_s'] + conf['n_a']
        sas_conf['n_q'] = conf['n_s']
        sas_conf['n_a'] = 0
        #sas_conf['n_hidden'] = 200
        sas_conf['one_layer_only'] = 0
        sas_conf['initial_learn_rate'] = 0.0001
        if 'sas_net_file' in conf:
            print 'loading sas net'
            self.sas_net = ControlNN(sas_conf, load_path=conf['sas_net_file'])
        else:
            self.sas_net = ControlNN(sas_conf, load_path=None)

        certainty_conf = conf.copy()
        certainty_conf['n_q'] = 1
        certainty_conf['n_a'] = 0
        if 'certainty_net_file' in conf:
            print 'loading certainty net'
            self.certainty_net = ControlNN(certainty_conf, load_path=conf['certainty_net_file'])
        else:
            self.certainty_net = ControlNN(certainty_conf, load_path=None)

        '''

        if 'q_net_file' in conf:
            self.load_path = conf['q_net_file']
        else:
            self.load_path = None

        self.initialize_nets()
        self.transfer_path = 'tmp/model_transfer'

        rand_int = int(time.time() * 10000000) % 100000000
        self.save_path = 'models/model_%s.out' % rand_int
        self.t_path = 'transitions/transitions_%s.out' % rand_int
        self.log_path = self.model_name + '.log'
        logging.basicConfig(filename=self.log_path,level=logging.INFO)
        logging.info('log for %s', self.model_name)
        print 'saving to', self.save_path, 'logging to', self.log_path

        self.b_dim = b_dim = 2
        self.d_dim = d_dim = 1
        self.b_params = []
        self.d_params = []
        self.r_by_params = []
        self.cur_b = np.ones(self.b_dim)
        self.cur_d = np.ones(self.d_dim)
        self.cur_r = 0

    def do_gp(self):
        X = np.concatenate((np.array(self.b_params), np.array(self.d_params)),1)
        gp = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
        gp.fit(X, self.r_by_params)
        sl = slice(-5, 5, 0.1)
        X_test = np.concatenate([x.flatten()[:,np.newaxis] for x in np.mgrid[sl, sl, sl]], 1)
        prof.tic('pred')
        ys, sigmas = gp.predict(X_test, eval_MSE=True)
        prof.toc('pred')
        y_max = max(self.r_by_params)
        us = (ys - y_max) / sigmas
        eis = sigmas * (us * norm.cdf(us) + norm.pdf(us))
        best_params = X_test[np.argmax(eis)]
        print 'best params', best_params
        return best_params[:2], best_params[2:]

    def b_transform(self, s):
        assert s.shape == (2,)
        return self.cur_b * s

    def d_transform(self, a):
        return np.maximum(-2., np.minimum(2., self.cur_d * a))

    def state_noise(self, s):
        return s * 2

    def initialize_nets(self):
        self.current_net = ControlNN(conf=self.conf, load_path=self.load_path)
        if self.use_old_net:
            self.old_net = ControlNN(conf=self.conf, load_path=self.load_path)
        else:
            self.old_net = self.current_net

    def should_update_old_net(self):
        return self.times_trained - self.last_old_net_update_count >= self.old_net_update_delay

    def will_update_old_net_next(self):
        return self.times_trained - self.last_old_net_update_count == self.old_net_update_delay-1

    def maybe_update_old_net(self):
        if not self.use_old_net:
            return

        if self.should_update_old_net():
            #self.profiler.tic('net transfer')
            self.current_net.save_model(self.transfer_path)
            self.old_net.load_model(self.transfer_path)
            #last_self.old_net_update_time = self.train_t
            self.last_old_net_update_count = self.times_trained
            self.net_update_count += 1
            #self.profiler.toc('net transfer')

    def test_mse(self, ts):
        #test_sa, test_ys = self.parse_transitions(self.ref_transitions)
        s, a, ys = self.parse_transitions(ts)
        #mse = self.current_net.mse_q(s, a, ys)
        qs = self.current_net.q_from_sa_discrete(s, a)
        #print qs
        #print a
        #print self.current_net.q_from_s_discrete(s)

        if self.error_type == 0:
            mse = np.sqrt(np.mean((ys - qs)**2))
        elif self.error_type == 1:
            mse = np.mean(np.abs(ys-qs))
        return mse, qs, ys

    def train_once(self, compute_mse=True):
        #self.profiler.tic('sa_and_ys_time')
        #sa, ys = self.get_sa_and_ys()
        ts = self.transitions.random_sample(self.n_minibatch)
        s, a, ys = self.parse_transitions(ts)
        #self.profiler.toc('sa_and_ys_time')

        if compute_mse:
            #self.profiler.tic('test_mse_time1')
            mse1, qs, test_ys = self.test_mse(self.ref_transitions)
            test_y_norm = np.linalg.norm(test_ys)
            rmse_rel1 = np.sqrt(mse1) / test_y_norm
            #print qs
            #print test_ys
            #print qs-test_ys
            print 'mse:', mse1 #, mse2
            print 'test_y_mean', np.mean(test_ys)
            #print 'rmse_rel:', rmse_rel1
            self.mse_hist.append(mse1)
            self.rmse_rel_hist.append(rmse_rel1)
            #self.profiler.toc('test_mse_time1')

        #self.profiler.tic('train_time')
        #self.current_net.train(sa, ys)
        self.current_net.train_discrete(s, a, ys)
        #self.profiler.toc('train_time')


        # self.profiler.tic('test_mse_time2')
        # mse2, qs, test_ys = test_mse()
        # self.profiler.toc('test_mse_time2')
        # rmse_rel2 = np.sqrt(mse2) / test_y_norm
        # self.mse_hist2.append(mse2)
        # self.rmse_rel_hist2.append(rmse_rel1)

        self.times_trained += 1
        self.epsilon = max(self.final_epsilon, self.start_epsilon - (self.start_epsilon-self.final_epsilon) / self.epsilon_anneal_time * self.train_t)
        self.last_train_time = self.train_t

    def get_sa_and_ys(self):
        self.profiler.tic('max_time_for_sa_and_ys')
        if self.minibatch_index == self.n_batches:
            ts = self.transitions.random_sample(self.n_megabatch)
            self.sa_queue, self.ys_queue = self.parse_transitions(ts)
            self.minibatch_index = 0
        self.profiler.toc('max_time_for_sa_and_ys')

        start_ind = self.minibatch_index * self.n_minibatch
        inds = slice(start_ind, start_ind + self.n_minibatch, 1)
        print 'using slice', inds
        self.minibatch_index += 1
        return self.sa_queue[inds], self.ys_queue[inds]

    def parse_transitions(self, ts):
        rs = np.array([t[2] for t in ts])[:, np.newaxis]
        terms = np.array([t[3] for t in ts])
        new_states = np.array([t[4] for t in ts])
        #best_q = self.old_net.get_best_a_p(new_states, is_p=True, num_tries=1)[1]
        all_q = self.old_net.q_from_s_discrete(new_states)[:,np.newaxis]
        best_q = self.old_net.get_best_q_discrete(new_states)[:,np.newaxis]
        #print 'rs', rs.shape, rs
        #print 'best_q', best_q.shape, best_q
        # TODO: fix for non-binary rs
        ys = rs.copy()
        for i in range(len(rs)):
            if not terms[i]:
                ys[i] += self.gamma * best_q[i]
            #else:
            #    print 'rs', rs
            #    print 'best_q', best_q
            #    print 'ys', ys
        # print 'rs', rs
        # print 'best_q', best_q
        # print 'all_q', all_q

        #sa = np.array([np.append(t[0], t[1]) for t in ts])
        s = np.array([t[0] for t in ts])
        a = [t[1] for t in ts]
        return s, a, ys

    def random_action(self):
        a = np.random.uniform(-self.max_torque,self.max_torque)
        print 'random_action', a
        return a

    def random_action_discrete(self):
        #p = 1.0 / np.sqrt(self.action_counts)
        #return np.random.choice(range(self.current_net.n_q), p=p/np.sum(p))
        return np.random.choice(range(self.current_net.n_q))

    def plot_sim(self, ind):
        aa = np.array(self.a_hists[ind])
        ss = np.array(self.s_hists[ind])
        plt.plot(ss)
        plt.plot(aa/self.max_torque)
        plt.show()

    def dump_transitions(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.transitions.container, f)

    def run_sas_train(self):
        self.run_net_train(self.sas_net, 'sas_train_data.p')

    def run_certainty_train(self):
        self.run_net_train(self.certainty_net, 'certainty_train_data.p')

    def run_action_train(self, xs, us):
        #self.run_net_train(self.action_net, 'action_train_data.p')
        xs = self.standardize_state(xs)
        us = self.standardize_action(us)
        self.run_net_train(self.action_net, xs, us)

    #def run_net_train(self, net, train_data_file):
    def run_net_train(self, net, xs, us):
        #self.profiler.tic('data load')
        #xs, us = pickle.load(open(train_data_file, 'rb'))
        #self.profiler.toc('data load')

        n_test = len(xs) / 10

        inds = np.array(range(len(xs)))
        np.random.shuffle(inds)
        xs = xs[inds]
        us = us[inds]
        #test_inds = np.random.choice(range(n_train), n_test, False)
        x_test, u_test = xs[:n_test], us[:n_test]
        xs, us = xs[n_test:], us[n_test:]
        assert len(xs) == len(us)

        # ms_x = mu_std(xs)
        # ms_u = mu_std(us)

        # net.set_standardizer(ms_x, ms_u)

        # xs = standardize(xs, ms_x)
        # x_test = standardize(x_test, ms_x)
        # us = standardize(us, ms_u)
        # u_test = standardize(u_test, ms_u)

        n_train = len(xs)
        n_batch = self.n_minibatch
        # print n_train

        # print ms_x, ms_u
        for ep in range(1000):
            self.profiler.tic('epoch')
            print 'epoch', ep
            mse = net.mse_q(x_test, u_test)
            self.mse_hist.append(mse)
            delta = net.learn_delta_q(x_test, u_test)
            #print 'test_us', u_test
            #print 'delta', delta, delta.shape
            #real_mse = np.mean(delta * (ms_u[1]**2))
            print 'mse', mse, '=', np.mean(delta), 'learn_rate', net.get_learn_rate()
            for j in range(n_train/n_batch):
                s = slice(j * n_batch, (j+1)*n_batch, 1)
                net.train(xs[s], us[s])
            self.profiler.toc('epoch')

    def run_dp_train(self):

        sa_costs = None
        with open('../Pendulum/dp_sa_cost.p', 'rb') as f:
            sa_costs = pickle.load(f).flatten()

        sa_samples = []
        a_grid = np.linspace(-2, 2, 9)
        s1_grid = np.linspace(0, 2*np.pi, 51)
        s2_grid = np.linspace(-10, 10, 51)

        for i, c in enumerate(sa_costs):
            a = i / (51 * 51)
            ss = i % (51 * 51)
            s2, s1 = ss / 51, ss % 51
            a = a_grid[a]
            s1 = s1_grid[s1]
            s2 = s2_grid[s2]
            sa_samples.append([s1, s2, a, c])

        sa_samples = np.array(sa_samples)
        n = len(sa_samples)
        for i in range(0):
            sample_size = n if i%100==0 else 50
            inds = np.random.choice(range(n), sample_size, False)
            #t = np.array(random.sample(sa_samples, sample_size))
            t = sa_samples[inds]

            if i%100 == 0:
                print 'iter', i
                mse = self.current_net.mse_q(t[:,0:3], -t[:,3][:,np.newaxis])
                learn_rate = self.current_net.get_learn_rate()
                print 'mse', mse, 'learn_rate', learn_rate
                self.mse_hist.append(mse)
                self.learn_rate_hist.append(learn_rate)
            else:
                self.current_net.train(t[:,0:3], -t[:,3][:,np.newaxis])

        #plt.plot(mses); plt.show()
        vis = NetVisualizer(self.current_net)
        vis.q_heat_map()

    def load_action_files(self, files):
        for fname in files:
            lines = open(fname, 'r').readlines()
            lines = map(lambda l: map(float, l.strip().split(' ')), lines)
            xs = np.array(lines[::3])
            us = np.array(lines[1::3])
            us_orig = np.array(lines[2::3])
            return xs, us, us_orig

    def load_action_files_old(self, files):
        for fname in files:
            lines = open(fname, 'r').readlines()
            lines = map(lambda l: map(float, l.strip().split(' ')), lines)
            xs = np.array(lines[::2])
            us = np.array(lines[1::2])
            return xs, us

    def load_simbicon_transitions(self, files):
        t = []
        for fname in files:
            lines = open(fname, 'r').readlines()
            lines = map(lambda l: map(float, l.strip().split(' ')), lines)
            r = np.array(lines[::3])
            s = np.array(lines[1::3])
            a = np.array(lines[2::3])
            r = np.array(map(float, r))
            a = np.array(map(int, a))
            t.extend([(s[i], a[i], r[i+1], False, s[i+1]) for i in range(len(s)-1)])
        return t

    def reflect_transition(self, t):
        return (reflect_state(t[0]), (t[1]+2) % 4, t[2], t[3], reflect_state(t[4]))

    def run_no_matlab(self):

        def count_dist(a):
            a = list(a)
            return [a.count(i) for i in range(4)]

        self.profiler.tic('no_matlab')
        if len(c.transitions.container) < 100:
            return

        mse_hist = []
        correct_rates = []
        np.random.shuffle(self.transitions.container)
        s = np.array([i[0] for i in self.transitions.container])

        for i in range(1500):
            if i % 100 == 0:
                self.profiler.tic('qs time')
                qs = self.current_net.q_from_s_discrete(s)
                self.profiler.toc('qs time')
                print qs[:20]
                print 'iteration', i
                self.profiler.tic('eval time')
                preds, actuals, rate = self.evaluate_simbicon(self.good_simbicon_transitions)
                self.profiler.toc('eval time')
                print 'actual_dist', count_dist(actuals)
                print 'pred_dist', count_dist(preds)
                print 'correct_rate', rate
                self.q_mean_hist.append(np.mean(qs))
                #if rate > 0.93:
                #    print 'good rate'
                #    break

            self.train_once(i % 100 == 0)
            if self.will_update_old_net_next() or i == 0:
                print 'testing mse, iter', i
                mse, qs, test_ys = self.test_mse(self.ref_transitions)
                mse_hist.append(mse)
                correct_rates.append(rate)
                self.correct_rate_hist.append(rate)
                self.mse_hist.append(mse)
                print 'mse_hist:', self.mse_hist
                if len(mse_hist) > 1 and mse_hist[-1] > mse_hist[-2]:
                    print 'mse_hist', mse_hist
                    print 'correct_rates', correct_rates
                    print 'loading best model'
                    self.current_net.load_model(self.last_no_matlab_net_path)
                    return
                self.current_net.save_model(self.last_no_matlab_net_path)
            self.maybe_update_old_net()
        self.profiler.toc('no_matlab')

    def whole_second_frac(self, n):
        return np.abs(self.sim_t*n - np.round(self.sim_t*n)) < 0.0001

    def run_matlab(self, algo):

        print 'ready for matlab'

        while True:
            start_time = time.time()
            #self.profiler.tic('wait')
            while not os.path.isfile(self.matlab_state_file):
                if time.time() - start_time > 300:
                    print "timeout"
                    #self.current_net.save_model(self.save_path)
                    self.profiler.toc('total controller run time')
                    self.profiler.print_time_stats()
                    sys.exit()
                pass

            f = open(self.matlab_state_file, 'r')
            lines = f.readlines()
            f.close()
            if len(lines) < 4: # file isn't fully written yet
                continue

            #self.profiler.toc('wait')

            reward_str = lines[0].rstrip()
            reward = float(reward_str)
            self.cur_r += reward
            # if reward > 0 and not self.succeeded and sim_t >= 0.05:
            #     logging.info('succeeded, reward=%s, sim_t=%s, state=%s', reward, sim_t, state)
            #     self.succeeded = True

            #if reward == 0.0:
            #    print 'sim already failed at time', self.sim_t

            term_str = lines[1].rstrip()
            term = int(term_str) > 0

            state_strs = lines[2].rstrip().split(' ')
            state = np.array(map(float, state_strs))
            if algo != 'pend_dp':
                state = self.standardize_state(state)

            last_sim_t = self.sim_t
            self.sim_t = float(lines[3].rstrip())

            if self.sim_t < last_sim_t:
                self.last_state = None
                self.sim_num += 1
                print 'new sim', self.sim_num, 'self.train_t', self.train_t
                #self.sim_start_time = self.train_t
                self.a_hists.append(self.current_a_hist[:])
                self.s_hists.append(self.current_s_hist[:])
                self.t_hists.append(self.current_t_hist[:])
                self.current_a_hist = []
                self.current_s_hist = []
                self.current_t_hist = []
                self.b_params.append(self.cur_b)
                self.d_params.append(self.cur_d)
                self.r_by_params.append(self.cur_r)
                self.cur_r = 0
                if len(self.b_params) > 1:
                    self.cur_b, self.cur_d = self.do_gp()
                else:
                    self.cur_b, self.cur_d = np.random.randn(self.b_dim), \
                            np.random.randn(self.d_dim)


            #if reward != 0.0 and self.whole_second_frac(10):
            if reward == 10.0:
                print 'received reward', reward, 'at time', self.sim_t #, 'x', state[0]

            #certainty = self.certainty_net.q_from_sa(state.reshape((1,-1)))[0][0]
            self.train_t += self.sim_dt

            #print 'sim_t', sim_t, 'state', state, 'certainty', certainty
            #self.certainty_hist.append(certainty)

            os.remove(self.matlab_state_file)
            if algo == 'RL':
                action = self.RL_train_and_action(state, reward, term)
                self.action_counts[action] += 1;
                write_str = '%s\n' % action
            elif algo == 'action_net':
                action = self.action_net_main(state)
                write_str = ' '.join(map(str, action)) + '\n'
            elif algo == 'pend_dp':
                state = np.array(state)
                state = self.state_noise(state)
                state = self.b_transform(state)
                output = self.current_net.manual_max_a_p(state[np.newaxis,:])
                action = output[0,0]
                action = self.d_transform(action)[0]
                write_str = '%s\n' % action
                print 'input', state
                print 'output', output
                print 'action', write_str
            else:
                raise ValueError('Not a valid algo')


            #print 'action', action
            f = open(self.python_action_file, 'w')
            #f.write('%s\n' % ' '.join(map(str, action)))
            f.write(write_str)
            f.close()

            if term:
                print 'sim failed at time', self.sim_t
                #self.initialize_nets()
                #self.run_no_matlab()

    def action_net_main(self, state):
        net_out = self.action_net.q_from_sa(state.reshape((1,-1)))[0]
        return self.unstandardize_action(net_out)

    def correct_rate(self, pred, actual):
        count = 0
        for x,y in zip(pred, actual):
            if x == y:
                count += 1
        return count/float(len(pred))

    def evaluate_simbicon(self, t):
        actual = [i[1] for i in t]
        pred = [self.current_net.get_best_a_discrete(i[0]) for i in t]
        mid = len(pred)/2
        # print 'first half', self.correct_rate(pred[:mid], actual[:mid])
        # print 'second half', self.correct_rate(pred[mid:], actual[mid:])
        return np.array(pred), np.array(actual), self.correct_rate(pred, actual)

    def standardize_action(self, u):
        return (u - self.u_mean) / self.u_std

    def unstandardize_action(self, u):
        return u * self.u_std + self.u_mean

    def standardize_state(self, s):
        return (s - self.s_mean) / self.s_std

    def unstandardize_state(self, s):
        return s * self.s_std + self.s_mean

    def standardize_transition(self, t):
        t_new = list(t)
        t_new[0] = self.standardize_state(t[0])
        t_new[-1] = self.standardize_state(t[-1])
        return t_new

    def unstandardize_transition(self, t):
        t_new = list(t)
        t_new[0] = self.unstandardize_state(t[0])
        t_new[-1] = self.unstandardize_state(t[-1])
        return t_new

    def truncate_transitions(self, t):
        return [(i[0][:-4], i[1], i[2], i[3], i[4][:-4]) for i in t]

    def summary_stats(self, t):
        t = np.array(t)
        return 'min: %f, max: %f, mean: %f, std: %f' % \
                (np.min(t), np.max(t), np.mean(t), np.std(t))

    def print_reward_summary(self):
        print self.summary_stats([x[2] for x in self.transitions.container])

    def change_rewards_unstandardized(self, t):
        t_new = t[:]
        old_rewards = []
        new_rewards = []
        for i in t_new:
            old_rewards.append(i[2])
            #i[2] = matlab_reward(i[-1])
            if i[2] == 10:
                i[2] = 2
            new_rewards.append(i[2])

        print 'old rewards', self.summary_stats(old_rewards)
        print 'new rewards', self.summary_stats(new_rewards)
        return t_new

    def change_rewards_standardized(self, t):
        t_unstandardized = [self.unstandardize_transition(i) for i in t]
        new_unstandardized = self.change_rewards_unstandardized(t_unstandardized)
        return [self.standardize_transition(i) for i in new_unstandardized]

    def transition_ratio(self):
        good_t = [x for x in self.transitions.container if x[2] > 0]
        return float(len(good_t))/len(self.transitions.container)

    def transition_summary(self):
        def t_to_tuple(t):
            def f(x):
                y = x
                return tuple(np.round(y))
            return (f(t[0]), t[1], t[2], t[3], f(t[4]))

        t = [t_to_tuple(x) for x in self.transitions.container]
        a = {}
        for i in t:
            if i not in a:
                a[i] = 0
            a[i] += 1

        return a, [x for x in a.items() if x[1] > 1]

    def duplicate_positive_rewards(self, t, ratio):
        good_t = [x for x in t if x[2] > 0]
        d = ratio * len(t) / (1-ratio) / len(good_t)
        print 'total len', len(t)
        print 'num good', len(good_t)
        print 'duplicating', d, 'times'
        ret_t = t[:]
        for _ in range(int(d)):
            ret_t.extend(good_t)
        return ret_t

    def set_standardizer(self, s):
        self.s_std = np.std(s, 0)
        self.s_mean = np.mean(s, 0)

    def set_u_standardizer(self, u):
        self.u_std = np.std(u, 0)
        self.u_mean = np.mean(u, 0)

    def RL_train_and_action(self, state, reward, term):

        #if self.time_step % 10 == 0:
        #self.profiler.toc('RL train and action')
        #self.profiler.tic('RL train and action')
        if self.whole_second_frac(10):
            print 'iter', self.time_step, 'sim_num', self.sim_num, \
                    'num transitions', len(self.transitions.container), \
                    'transition ratio', self.transition_ratio(), \
                    'train_t', self.train_t, 'sim_t', self.sim_t, \
                    'epsilon', self.epsilon, 'times_trained', self.times_trained, \
                    'net_update_count', self.net_update_count, \
                    'learn_rate', self.current_net.get_learn_rate()

        #self.profiler.tic('nn_cycle')
        if self.last_state != None:
            transition = (self.last_state, self.last_action, reward, term, state)
            self.transitions.append(transition)
            self.current_a_hist.append(self.last_action)
            self.current_s_hist.append(self.last_state)
            self.current_t_hist.append(transition)

        self.maybe_update_old_net()
        action = None
        if self.train_t < 0.0 or self.last_action == None:
            action = self.random_action_discrete()
        elif self.train_t - self.last_action_time > self.min_action_gap:
            self.last_action_time = self.train_t
            if np.random.random() > self.epsilon:
                action = self.current_net.get_best_a_discrete(state)

                # qs = c.current_net.q_from_s_discrete(state[np.newaxis,:])[0]
                # poss_next_action = (self.last_action + 1) % 4
                # if qs[poss_next_action] > qs[self.last_action]:
                #     action = poss_next_action
                # else:
                #     action = self.last_action

            else:
                #action = self.random_action()
                action = self.random_action_discrete()
        else:
            action = self.last_action

        # #print 'action', action
        # if self.bang_action == 1:
        #     if action > 0.0:
        #         action = self.max_torque
        #     else:
        #         action = -self.max_torque

        #self.profiler.tic('nn_cycle1')
        if self.train_t > 0.0 and self.train_t - self.last_train_time > self.min_train_gap \
                and len(self.transitions.container) > self.n_minibatch \
                and self.RL_train:
            #self.train_once(self.times_trained % self.mse_freq == 0)
            for i in range(self.train_per_iter):
                self.train_once(False)

        #self.profiler.toc('nn_cycle1')

        self.time_step += 1
        self.last_state = state
        self.last_action = action
        #self.profiler.toc('nn_cycle')
        if self.whole_second_frac(10):
            print 'action taken:', action
        return action

if __name__ == '__main__':

    c = NNController(conf='pendulum_new.conf')
    #c.run_dp_train()
    c.run_matlab('pend_dp')
    sys.exit()

    c = NNController(conf='simbicon.conf')
    #file_nums = [52211431, 41983419]
    #file_nums = [34403245, 26462535]
    f = open('simbicon_info', 'r')

    c.profiler.tic('loading simbicon files')
    #file_nums = map(int, f.readlines()[-1].strip().split(' '))[:5]
    #file_nums = [32366732, 14449125, 56594694, 39323664] # simbicon 0.8,1
    #file_nums = [56071150, 21714576, 16842298, 42847997, 24903202, 50801256, 33378301, 15476319, 23148165, 31165728] # simbicon, 0.4,0.6
    #file_nums = [10409118, 42863489, 46896633, 36285042, 119317, 35208865] # simb 0.6, 0.8
    #file_nums = [57956487, 51664368, 3794827, 10925401, 22669808, 23560151, 31258403] # NN 6-8

    #file_nums = [29155381, 20048806, 9022910, 57982100, 50309745, 41684616, 49912394, 43426867, 34486168, 26855010, 36171855] #NN2 6-8, with action
    #file_nums.extend([5076978, 57623861, 49306452, 55204125, 36616958, 110969, 47891551, 54853789, 1310811, 7504873, 13472364])

    #file_nums = [55692754, 6309521, 14761170, 56020394, 50383334, 59270192, 38319491] # NN both actions

    #file_nums = [35190956, 43266080, 30354380, 16114224, 2346631, 48551106, 18300972, 17168126, 2881742, 38799587, 24944375] # simbicon 0.8 1

    file_nums = [45784265, 31870569, 12752807, 36759048, 36294704, 59918304, 16822284, 59050432, 41944502, 24216744, 27941279] # NN 0.8, 1
    file_nums_noise = [44471764, 13875631, 321733, 51194756, 23131352, 7530505, 51562342, 38247513, 20316616, 2131893, 23873173] # NN 0.8, 1 + noise
    file_nums.extend(file_nums_noise)

    files = ['../KneedCompassGait/outputs/%d.out' % i for i in file_nums]
    action_files = ['../KneedCompassGait/action_outputs/%d.out' % i for i in file_nums]
    c.profiler.toc('loading simbicon files')

    t_orig = c.load_simbicon_transitions(files)

    np.random.shuffle(t_orig)
    #t_orig = c.truncate_transitions(t_orig)
    s = np.array([i[0] for i in t_orig])
    c.set_standardizer(s)
    t = [c.standardize_transition(i) for i in t_orig]

    c.good_simbicon_transitions = t[:2000]

    c.transitions.container = []
    #c.transitions.container = c.ref_transitions[:]
    #for _ in range(10):
    #c.transitions.container.extend(c.ref_transitions)

    c.transitions.container.extend(t)
    print 'training on %d transitions' % len(c.transitions.container)
    c.print_reward_summary()
    reflected_transitions = [c.reflect_transition(i) for i in c.transitions.container]
    #c.transitions.container = c.duplicate_positive_rewards(c.transitions.container, 0.1)
    #c.transitions.container.extend(reflected_transitions)

    #c.transitions.container = c.change_rewards_standardized(c.transitions.container)

    final_crs = []
    final_mses = []

    s = np.array([i[0] for i in t])
    qs = c.current_net.q_from_s_discrete(s)

    pred, actual, rate = c.evaluate_simbicon(t)
    print 'correct rate', rate
    #c.run_no_matlab()
    c.run_matlab('RL')

    action_xs, action_us, action_us_orig = c.load_action_files(action_files)
    #action_xs, action_us = c.load_action_files_old(action_files)
    c.set_u_standardizer(action_us)
    c.run_matlab('action_net')

    c.run_action_train(action_xs, action_us)
