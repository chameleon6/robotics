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

        self.transitions = TransitionContainer()
        with open('reference_transitions.p', 'rb') as f:
            self.all_ref_transitions = pickle.load(f)
        np.random.shuffle(self.all_ref_transitions)
        self.ref_transitions = self.all_ref_transitions[:self.n_megabatch]
        #print self.ref_transitions


        #self.load_path = 'models/model_31275.out'
        #self.load_path = 'models/model_77422.out'
        self.load_path = load_path
        self.current_net = ControlNN(self.load_path)
        self.old_net = ControlNN(self.load_path)

        self.transfer_path = '/tmp/model_transfer'
        self.save_path = 'models/model_%s.out' % (int(time.time() * 10000) % 100000)
        print 'saving to', self.save_path

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
        return np.random.uniform(-self.max_torque,self.max_torque)

    def plot_sim(self, ind):
        aa = np.array(self.a_hists[ind])
        ss = np.array(self.s_hists[ind])
        plt.plot(ss)
        plt.plot(aa/self.max_torque)
        plt.show()

    def dump_transitions(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.transitions.container, f)

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
            s1, s2 = ss / 51, ss % 51
            a = a_grid[a]
            s1 = s1_grid[s1]
            s2 = s2_grid[s2]
            sa_samples.append([s1, s2, a, c])

        mses = []
        for i in range(10000):
            t = np.array(random.sample(sa_samples, 100))

            if i%100 == 0:
                print 'iter', i
                mse = self.current_net.mse_q(t[:,0:3], -t[:,3][:,np.newaxis])
                print 'mse', mse
                mses.append(mse)
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
            self.train_once(i % 50 == 0)
            self.maybe_update_old_net()

    def run_matlab(self):
        print "ready for matlab"
        while True:
            start_time = time.time()
            while not os.path.isfile(self.matlab_state_file):
                if time.time() - start_time > 10:
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

            #print lines
            reward_str = lines[0].rstrip()
            reward = float(reward_str)

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

            self.train_t += self.sim_dt

            os.remove(self.matlab_state_file)

            print 'iter', self.time_step, 'train_t', self.train_t, 'sim_t', sim_t, 'epsilon', self.epsilon, \
                    'times_trained', self.times_trained, 'net_update_count', self.net_update_count

            self.profiler.tic('nn_cycle')
            if self.last_state != None:
                self.transitions.append((self.last_state, self.last_action, reward, state))
                self.current_a_hist.append(action)
                self.current_s_hist.append(state)

            #if self.train_t - last_self.old_net_update_time > self.old_net_update_time:

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

                    '''
                    aq = self.old_net.manual_max_a(state)
                    #print aq
                    #best_q = aq[:,1][:,np.newaxis]
                    action = aq[0]
                    '''

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
                self.train_once(self.times_trained % 20 == 0)

            f = open(self.python_action_file, 'w')
            f.write('%s\n' % action)
            f.close()
            self.profiler.toc('nn_cycle1')

            self.time_step += 1
            self.last_state = state
            self.last_action = action
            self.profiler.toc('nn_cycle')
            print

if __name__ == '__main__':
    #c = NNController(load_path=None, conf='pendulum.conf')
    ##c.run_dp_train()
    #c.run_no_matlab('t1.p')

    #c = NNController(load_path='models/model_48657.out', conf='exploit_pendulum.conf')
    c = NNController(load_path='models/model_39574.out', conf='exploit_pendulum.conf')
    c.run_matlab()
