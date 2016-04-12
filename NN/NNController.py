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

class NNController:

    def __init__(self, conf):
        self.profiler = Profiler()
        self.profiler.tic('total controller run time')

        self.matlab_state_file = os.getcwd() + '/matlab_state_file.out'
        self.python_action_file = os.getcwd() + '/python_action_file.out'

        conf = read_conf(conf)

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

        assert self.start_epsilon >= self.final_epsilon

        self.epsilon = self.start_epsilon
        #last_self.old_net_update_time = 0.0
        self.last_old_net_update_count = 0
        self.net_update_count = 0
        self.last_train_time = 0.0
        self.last_action_time = 0.0
        self.time_step = 1
        self.times_trained = 1
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
        self.mse_hist = []
        self.mse_hist2 = []
        self.rmse_rel_hist = []
        self.rmse_rel_hist2 = []
        self.learn_rate_hist = []
        self.certainty_hist = []
        self.correct_rate_hist = []
        self.q_mean_hist = []

        self.action_counts = np.array([1,1,1,1.])

        self.transitions = TransitionContainer(conf['max_history_len'])

        if 'ref_transitions_file' in conf:
            with open(conf['ref_transitions_file'], 'rb') as f:
                self.all_ref_transitions = pickle.load(f)
            np.random.shuffle(self.all_ref_transitions)
            self.ref_transitions = self.all_ref_transitions[:1000]

        if 'transitions_seed' in conf:
            self.transitions.container = pickle.load(open(conf['transitions_seed'], 'rb'))

        '''
        action_conf = conf.copy()
        action_conf['n_a'] = 0
        action_conf['n_q'] = 6
        if 'action_net_file' in conf:
            print 'loading action net'
            self.action_net = ControlNN(action_conf, load_path=conf['action_net_file'])
        else:
            self.action_net = ControlNN(action_conf, load_path=None)

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

        self.current_net = ControlNN(conf=conf, load_path=self.load_path)
        if self.use_old_net:
            self.old_net = ControlNN(conf=conf, load_path=self.load_path)
        else:
            self.old_net = self.current_net

        self.transfer_path = 'tmp/model_transfer'

        rand_int = int(time.time() * 10000000) % 100000000
        self.save_path = 'models/model_%s.out' % rand_int
        self.t_path = 'transitions/transitions_%s.out' % rand_int
        self.log_path = self.model_name + '.log'
        logging.basicConfig(filename=self.log_path,level=logging.INFO)
        logging.info('log for %s', self.model_name)
        print 'saving to', self.save_path, 'logging to', self.log_path

    def maybe_update_old_net(self):
        if not self.use_old_net:
            return

        if self.times_trained - self.last_old_net_update_count > self.old_net_update_delay:
            #self.profiler.tic('net transfer')
            self.current_net.save_model(self.transfer_path)
            self.old_net.load_model(self.transfer_path)
            #last_self.old_net_update_time = self.train_t
            self.last_old_net_update_count = self.times_trained
            self.net_update_count += 1
            #self.profiler.toc('net transfer')

    def train_once(self, compute_mse=True):

        def test_mse(ts):
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

        #self.profiler.tic('sa_and_ys_time')
        #sa, ys = self.get_sa_and_ys()
        ts = self.transitions.random_sample(self.n_minibatch)
        s, a, ys = self.parse_transitions(ts)
        #self.profiler.toc('sa_and_ys_time')

        if compute_mse:
            #self.profiler.tic('test_mse_time1')
            mse1, qs, test_ys = test_mse(self.ref_transitions)
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

    def run_action_train(self):
        self.run_net_train(self.action_net, 'action_train_data.p')

    def run_net_train(self, net, train_data_file):
        self.profiler.tic('data load')
        xs, us = pickle.load(open(train_data_file, 'rb'))
        self.profiler.toc('data load')

        n_test = len(xs) / 10

        inds = np.array(range(len(xs)))
        np.random.shuffle(inds)
        xs = xs[inds]
        us = us[inds]
        #test_inds = np.random.choice(range(n_train), n_test, False)
        x_test, u_test = xs[:n_test], us[:n_test]
        xs, us = xs[n_test:], us[n_test:]
        assert len(xs) == len(us)

        ms_x = mu_std(xs)
        ms_u = mu_std(us)

        net.set_standardizer(ms_x, ms_u)

        xs = standardize(xs, ms_x)
        x_test = standardize(x_test, ms_x)
        us = standardize(us, ms_u)
        u_test = standardize(u_test, ms_u)

        n_train = len(xs)
        n_batch = self.n_minibatch
        print n_train

        print ms_x, ms_u
        for ep in range(30):
            self.profiler.tic('epoch')
            print 'epoch', ep
            mse = net.mse_q(x_test, u_test)
            self.mse_hist.append(mse)
            delta = net.learn_delta_q(x_test, u_test)
            #print 'test_us', u_test
            print 'delta', delta, delta.shape
            real_mse = np.mean(delta * (ms_u[1]**2))
            print 'mse', mse, '=', np.mean(delta), 'real mse', real_mse, 'learn_rate', net.get_learn_rate()
            for j in range(n_train/n_batch):
                if j % 1000 == 0:
                    print j
                s = slice(j * n_batch, (j+1)*n_batch, 1)
                net.train(xs[s], us[s])
            self.profiler.toc('epoch')

    def run_dp_train(self):

        sa_costs = None
        with open('dp_sa_cost.p', 'rb') as f:
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
        for i in range(100000):
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

    def run_no_matlab(self, files, t):
        np.random.shuffle(self.transitions.container)
        s = np.array([i[0] for i in self.transitions.container])

        for i in range(20000):
            if i % 100 == 0:
                qs = self.current_net.q_from_s_discrete(s)
                print qs[:20]
                print 'iteration', i
                rate = self.evaluate_simbicon(t)[2]
                print 'correct_rate', rate
                self.q_mean_hist.append(np.mean(qs))
                self.correct_rate_hist.append(rate)
                if rate > 0.9:
                    print 'good rate'
                    break
            #self.train_once(i % 1000 == 0)
            self.train_once(i % 100 == 0)
            self.maybe_update_old_net()

    def run_matlab(self, algo):

        print "ready for matlab"
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
            # if reward > 0 and not self.succeeded and sim_t >= 0.05:
            #     logging.info('succeeded, reward=%s, sim_t=%s, state=%s', reward, sim_t, state)
            #     self.succeeded = True

            #if reward == 0.0:
            #    print 'sim already failed at time', self.sim_t

            term_str = lines[1].rstrip()
            term = int(term_str) > 0

            state_strs = lines[2].rstrip().split(' ')
            state = np.array(map(float, state_strs))
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
                self.current_a_hist = []
                self.current_s_hist = []

            if reward != 0.0:
                print 'received reward', reward, 'at time', self.sim_t #, 'x', state[0]

            #certainty = self.certainty_net.q_from_sa(state.reshape((1,-1)))[0][0]
            self.train_t += self.sim_dt

            #print 'sim_t', sim_t, 'state', state, 'certainty', certainty
            #self.certainty_hist.append(certainty)

            os.remove(self.matlab_state_file)
            if algo == 'RL':
                action = self.RL_train_and_action(state, reward, term)
            elif algo == 'action_net':
                action = self.action_net_main(state)
            else:
                raise ValueError('Not a valid algo')

            self.action_counts[action] += 1;

            #print 'action', action
            f = open(self.python_action_file, 'w')
            #f.write('%s\n' % ' '.join(map(str, action)))
            f.write('%s\n' % action)
            f.close()

            if term:
                print 'sim failed at time', self.sim_t

    def action_net_main(self, state):
        return self.action_net.q_from_sa(state.reshape((1,-1)))[0]

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
        print 'first half', self.correct_rate(pred[:mid], actual[:mid])
        print 'second half', self.correct_rate(pred[mid:], actual[mid:])
        return np.array(pred), np.array(actual), self.correct_rate(pred, actual)

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

    def change_rewards_unstandardized(self, t):
        t_new = t[:]
        for i in t_new:
            i[2] = matlab_reward(i[-1])
        return t_new

    def change_rewards_standardized(self, t):
        t_unstandardized = [self.unstandardize_transition(i) for i in t]
        new_unstandardized = self.change_rewards_unstandardized(t_unstandardized)
        return [self.standardize_transition(i) for i in new_unstandardized]

    def set_standardizer(self, s):
        self.s_std = np.std(s, 0)
        self.s_mean = np.mean(s, 0)

    def RL_train_and_action(self, state, reward, term):

        if self.time_step % 10 == 0:
            self.profiler.toc('test')
            self.profiler.tic('test')
            print 'iter', self.time_step, 'sim_num', self.sim_num, \
                    'num transitions', len(self.transitions.container), \
                    'train_t', self.train_t, 'sim_t', self.sim_t, \
                    'epsilon', self.epsilon, 'times_trained', self.times_trained, \
                    'net_update_count', self.net_update_count, \
                    'learn_rate', self.current_net.get_learn_rate()

        #self.profiler.tic('nn_cycle')
        if self.last_state != None:
            self.transitions.append((self.last_state, self.last_action, reward, term, state))
            self.current_a_hist.append(self.last_action)
            self.current_s_hist.append(self.last_state)

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
        if self.train_t > 0.0 and self.train_t - self.last_train_time > self.min_train_gap\
                and len(self.transitions.container) > self.n_minibatch:
            #self.train_once(self.times_trained % self.mse_freq == 0)
            self.train_once(False)

        #self.profiler.toc('nn_cycle1')

        self.time_step += 1
        self.last_state = state
        self.last_action = action
        #self.profiler.toc('nn_cycle')
        return action

if __name__ == '__main__':

    c = NNController(conf='simbicon.conf')
    #file_nums = [56385034, 9381058]
    #file_nums = [48519536, 37908959]
    file_nums = [52211431, 41983419]
    files = ['../KneedCompassGait/outputs/%d.out' % i for i in file_nums]
    t_orig = c.load_simbicon_transitions(files)
    s = np.array([i[0] for i in t_orig])
    c.set_standardizer(s)
    t = [c.standardize_transition(i) for i in t_orig]
    #t = c.change_rewards_standardized(t)

    #rs = [i[2] for i in ref_t_orig]
    #rs2 = [matlab_reward(i[-1]) for i in ref_t_orig]
    #diff = [x-y for x,y in zip(rs, rs2)]
    #print max(diff), min(diff)

    #c.transitions.container = []
    #c.transitions.container = c.all_ref_transitions[:]
    #c.transitions.container.extend(t)
    #c.transitions.container = c.change_rewards_standardized(c.transitions.container)
    #reflected_transitions = [c.reflect_transition(i) for i in c.transitions.container]
    ##c.transitions.container.extend(reflected_transitions)
    #c.run_no_matlab(files, t)

    s = np.array([i[0] for i in t])
    qs = c.current_net.q_from_s_discrete(s)

    pred, actual, rate = c.evaluate_simbicon(t)
    print 'correct rate', rate
    c.run_matlab('RL')
