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

class NNController:

    def __init__(self, conf, load_path=None):
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
        self.sim_start_time = self.train_t - 10
        self.sim_num = 0

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

        self.transitions = TransitionContainer(conf['max_history_len'])

        if 'ref_transitions_file' in conf:
            with open(conf['ref_transitions_file'], 'rb') as f:
                self.all_ref_transitions = pickle.load(f)
            np.random.shuffle(self.all_ref_transitions)
            self.ref_transitions = self.all_ref_transitions[:self.n_megabatch]

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

        # self.load_path = load_path
        # self.current_net = ControlNN(conf=conf, load_path=self.load_path)
        # self.old_net = ControlNN(conf=conf, load_path=self.load_path)
        # self.transfer_path = '/tmp/model_transfer'

        rand_int = int(time.time() * 10000000) % 100000000
        self.save_path = 'models/model_%s.out' % rand_int
        self.log_path = self.model_name + '.log'
        logging.basicConfig(filename=self.log_path,level=logging.INFO)
        logging.info('log for %s', self.model_name)
        print 'saving to', self.save_path, 'logging to', self.log_path

    def maybe_update_old_net(self):
        if self.times_trained - self.last_old_net_update_count > self.old_net_update_delay:
            self.profiler.tic('net transfer')
            self.current_net.save_model(self.transfer_path)
            self.old_net.load_model(self.transfer_path)
            #last_self.old_net_update_time = self.train_t
            self.last_old_net_update_count = self.times_trained
            self.net_update_count += 1
            self.profiler.toc('net transfer')

    def train_once(self, compute_mse=True):

        def test_mse():
            test_sa, test_ys = self.parse_transitions(self.ref_transitions)
            mse = self.current_net.mse_q(test_sa, test_ys)
            qs = self.current_net.q_from_sa(test_sa)
            return mse, qs, test_ys

        self.profiler.tic('sa_and_ys_time')
        sa, ys = self.get_sa_and_ys()
        self.profiler.toc('sa_and_ys_time')

        if compute_mse:
            self.profiler.tic('test_mse_time1')
            mse1, qs, test_ys = test_mse()
            test_y_norm = np.linalg.norm(test_ys)
            rmse_rel1 = np.sqrt(mse1) / test_y_norm
            print 'qs'
            print qs[1:30]
            print 'test_ys'
            print test_ys[1:30]
            print 'mse:', mse1 #, mse2
            print 'test_y_norm', test_y_norm
            print 'rmse_rel:', rmse_rel1
            self.mse_hist.append(mse1)
            self.rmse_rel_hist.append(rmse_rel1)
            self.profiler.toc('test_mse_time1')

        self.profiler.tic('train_time')
        self.current_net.train(sa, ys)
        self.profiler.toc('train_time')


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
        new_states = np.array([t[3] for t in ts])
        best_q = self.old_net.get_best_a_p(new_states, is_p=True, num_tries=1)[1]
        #aq = self.old_net.manual_max_a_p(new_states)
        #best_q = aq[:,1][:,np.newaxis]
        ys = rs + self.gamma * best_q
        #print 'new_states'
        #print new_states[:30]
        #print 'aq'
        #print aq[:30]
        #print 'best_q'
        #print best_q[:30]
        #print 'rs'
        #print rs[:30]
        #print 'ys'
        #print ys[:30]
        sa = np.array([np.append(t[0], t[1]) for t in ts])
        return sa, ys

    def random_action(self):
        a = np.random.uniform(-self.max_torque,self.max_torque)
        print 'random_action', a
        return a

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

        n_train = len(xs)
        n_batch = self.n_minibatch
        print n_train

        for ep in range(30):
            self.profiler.tic('epoch')
            print 'epoch', ep
            mse = net.mse_q(x_test, u_test)
            self.mse_hist.append(mse)
            print 'mse', mse, 'learn_rate', net.get_learn_rate()
            delta = net.learn_delta_q(x_test, u_test)
            #print 'test_us', u_test
            #print 'delta', delta, delta.shape
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


    def run_no_matlab(self, container_file):
        #self.transitions.container = self.all_ref_transitions
        with open(container_file, 'rb') as f:
            self.transitions.container = pickle.load(f)

        for i in range(4000):
            print 'iteration', i
            self.train_once(i % self.mse_freq == 0)
            self.maybe_update_old_net()

    def run_matlab(self, algo):

        print "ready for matlab"
        while True:
            start_time = time.time()
            while not os.path.isfile(self.matlab_state_file):
                if time.time() - start_time > 30:
                    print "timeout"
                    self.current_net.save_model(self.save_path)
                    self.profiler.toc('total controller run time')
                    self.profiler.print_time_stats()
                    sys.exit()
                pass

            f = open(self.matlab_state_file, 'r')
            lines = f.readlines()
            f.close()
            if len(lines) < 3: # file isn't fully written yet
                continue

            reward_str = lines[0].rstrip()
            reward = float(reward_str)
            # if reward > 0 and not self.succeeded and sim_t >= 0.05:
            #     logging.info('succeeded, reward=%s, sim_t=%s, state=%s', reward, sim_t, state)
            #     self.succeeded = True

            state_strs = lines[1].rstrip().split(' ')
            state = np.array(map(float, state_strs))

            sim_t = float(lines[2].rstrip())
            if sim_t < 0.001:
                self.last_state = None
                if self.train_t > self.sim_start_time + 0.05:
                    print 'new sim', self.sim_num, 'self.train_t', self.train_t
                    self.sim_start_time = self.train_t
                    self.sim_num += 1
                    self.a_hists.append(self.current_a_hist)
                    self.s_hists.append(self.current_s_hist)
                    self.current_a_hist = []
                    self.current_s_hist = []
                    # if not self.succeeded:
                    #     logging.info('didn\'t succeed :(')
                    # self.succeeded = False

            certainty = self.certainty_net.q_from_sa(state.reshape((1,-1)))[0][0]
            self.train_t += self.sim_dt

            print 'sim_t', sim_t, 'state', state, 'certainty', certainty
            self.certainty_hist.append(certainty)

            os.remove(self.matlab_state_file)
            if algo == 'rl':
                action = self.RL_train_and_action(state)
            elif algo == 'action_net':
                action = self.action_net_main(state)
            else:
                raise ValueError('Not a valid algo')

            print 'action', action
            f = open(self.python_action_file, 'w')
            f.write('%s\n' % ' '.join(map(str, action)))
            f.close()

    def action_net_main(self, state):
        return self.action_net.q_from_sa(state.reshape((1,-1)))[0]

    def RL_train_and_action(self, state):
        print 'iter', self.time_step, 'train_t', self.train_t, 'sim_t', sim_t, \
                'epsilon', self.epsilon, 'times_trained', self.times_trained, \
                'net_update_count', self.net_update_count, \
                'learn_rate', self.current_net.get_learn_rate()

        self.profiler.tic('nn_cycle')
        if self.last_state != None:
            self.transitions.append((self.last_state, self.last_action, reward, state))
            #self.current_a_hist.append(action)
            #self.current_s_hist.append(state)

        self.maybe_update_old_net()
        action = None
        if self.train_t < 0.0:
            action = 0.0
        elif self.train_t - self.last_action_time > self.min_action_gap:
            self.last_action_time = self.train_t
            if np.random.random() > self.epsilon:
                self.profiler.tic('action')

                action = self.current_net.get_best_a_p(state, is_p=False,
                        init_a = np.array([[self.last_action]]), num_tries=1)[0][0][0]

                #aq = self.old_net.manual_max_a(state)
                ##print aq
                ##best_q = aq[:,1][:,np.newaxis]
                #action = aq[0]

                #i = int(state[0] / (2*np.pi /50))
                #j = int((state[1] + 10) / 0.4)
                #action = -2 + 0.5 * np.argmin(sa_costs[:,51*j + i])
                #print state, i, j, action

                self.profiler.toc('action')
            else:
                action = self.random_action()
        else:
            action = self.last_action

        #print 'action', action
        if self.bang_action == 1:
            if action > 0.0:
                action = self.max_torque
            else:
                action = -self.max_torque

        self.profiler.tic('nn_cycle1')
        if self.train_t > 0.0 and self.train_t - self.last_train_time > self.min_train_gap:
            self.train_once(self.times_trained % self.mse_freq == 0)

        self.profiler.toc('nn_cycle1')

        self.time_step += 1
        self.last_state = state
        self.last_action = action
        self.profiler.toc('nn_cycle')
        return action

if __name__ == '__main__':
    #c = NNController(load_path=None, conf='pendulum.conf')
    #c.run_dp_train()

    c = NNController(load_path=None, conf='simbicon.conf')

    #c.run_no_matlab('t1.p')

    #c = NNController(load_path='good_models/dp_trained_pendulum_net.out', conf='exploit_pendulum.conf')

    c.run_matlab('action_net')
    #c.run_certainty_train()
    #c.run_action_train()
