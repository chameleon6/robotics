import numpy as np
import os
import sys
import time
import cPickle as pickle
import matplotlib.pyplot as plt
from utils import *
from nn import ControlNN
from TransitionContainer import TransitionContainer

MATLAB = False

profiler = Profiler()
profiler.tic('total controller run time')

matlab_state_file = os.getcwd() + '/matlab_state_file.out'
python_action_file = os.getcwd() + '/python_action_file.out'

transitions = TransitionContainer()
all_ref_transitions = pickle.load(open('reference_transitions.p', 'rb'))
np.random.shuffle(all_ref_transitions)
ref_transitions = all_ref_transitions[:20]
print ref_transitions

conf = read_conf('pendulum.conf')

max_torque = conf['max_torque']
bang_action = conf['bang_action']
#old_net_update_time = conf['old_net_update_time']
old_net_update_delay = conf['old_net_update_delay']
min_train_gap = conf['min_train_gap']
min_action_gap = conf['min_action_gap']
final_epsilon = conf['final_epsilon']
epsilon_anneal_time = conf['epsilon_anneal_time']
no_op_time = conf['no_op_time']
minibatch_size = conf['minibatch_size']
gamma = conf['gamma']
sim_dt = conf['sim_dt']
start_epsilon = conf['start_epsilon']

assert start_epsilon >= final_epsilon

epsilon = start_epsilon
#last_old_net_update_time = 0.0
last_old_net_update_count = 0
net_update_count = 0
last_train_time = 0.0
last_action_time = 0.0
ready_to_train = False
time_step = 1
times_trained = 0
last_state = None
last_action = None
train_t = -no_op_time
sim_start_time = train_t - 10
sim_num = 0

a_hists = []
current_a_hist= []
s_hists = []
current_s_hist= []
mse_hist = []
mse_hist2 = []
rmse_rel_hist = []
rmse_rel_hist2 = []

#load_path = 'models/model_31275.out'
#load_path = 'models/model_77422.out'
load_path = None
current_net = ControlNN(load_path)
old_net = ControlNN(load_path)

transfer_path = '/tmp/model_transfer'
save_path = 'models/model_%s.out' % (int(time.time() * 10000) % 100000)
print 'saving to', save_path

def maybe_update_old_net():
    if times_trained - last_old_net_update_count > old_net_update_delay:
        ready_to_train = True
        profiler.tic('net transfer')
        current_net.save_model(transfer_path)
        old_net.load_model(transfer_path)
        #last_old_net_update_time = train_t
        last_old_net_update_count = times_trained
        net_update_count += 1
        profiler.toc('net transfer')

def train_once():
    ts = transitions.random_sample(minibatch_size)

    def test_mse():
        test_sa, test_ys = parse_transitions(ref_transitions)
        mse = current_net.mse_q(test_sa, test_ys)
        qs = current_net.q_from_sa(test_sa)
        return mse, qs, test_ys

    profiler.tic('train_max_time')
    sa, ys = parse_transitions(ts)
    profiler.toc('train_max_time')

    profiler.tic('test_mse_time1')
    mse1, qs, test_ys = test_mse()
    test_y_norm = np.linalg.norm(test_ys)
    rmse_rel1 = np.sqrt(mse1) / test_y_norm
    print 'qs'
    print qs
    print 'test_ys'
    print test_ys
    print 'MSE:', mse1 #, mse2
    print 'test_y_norm', test_y_norm
    print 'RMSE_rel:', rmse_rel1
    mse_hist.append(mse1)
    rmse_rel_hist.append(rmse_rel1)
    profiler.toc('test_mse_time1')

    profiler.tic('train_time')
    current_net.train(sa, ys)
    profiler.toc('train_time')


    # profiler.tic('test_mse_time2')
    # mse2, qs, test_ys = test_mse()
    # profiler.toc('test_mse_time2')
    # rmse_rel2 = np.sqrt(mse2) / test_y_norm
    # mse_hist2.append(mse2)
    # rmse_rel_hist2.append(rmse_rel1)

    times_trained += 1
    epsilon = max(final_epsilon, start_epsilon - (start_epsilon-final_epsilon) / epsilon_anneal_time * train_t)
    last_train_time = train_t

def parse_transitions(ts):
    rs = np.array([t[2] for t in ts])[:, np.newaxis]
    new_states = np.array([t[3] for t in ts])
    ys = rs + gamma * old_net.get_best_a_p(new_states, is_p=True, num_tries=2)[1]
    sa = np.array([np.append(t[0], t[1]) for t in ts])
    return sa, ys

def random_action():
    return np.random.uniform(-max_torque,max_torque)

def plot_sim(ind):
    aa = np.array(a_hists[ind])
    ss = np.array(s_hists[ind])
    plt.plot(ss)
    plt.plot(aa/max_torque)
    plt.show()

if not MATLAB:
    transitions.container = all_ref_transitions
    for i in range(1000):
        train_once()
        maybe_update_old_net()

    sys.exit()

print "ready for matlab"
while True:
    start_time = time.time()
    while not os.path.isfile(matlab_state_file):
        if time.time() - start_time > 10:
            print "timeout"
            current_net.save_model(save_path)
            profiler.toc('total controller run time')
            profiler.print_time_stats()
            sys.exit()
        pass

    f = open(matlab_state_file, 'r')
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
        last_state = None
        if train_t > sim_start_time + 0.05:
            print 'new sim', sim_num, 'train_t', train_t
            sim_start_time = train_t
            sim_num += 1
            a_hists.append(current_a_hist)
            s_hists.append(current_s_hist)
            current_a_hist = []
            current_s_hist = []

    train_t += sim_dt

    os.remove(matlab_state_file)

    print 'iter', time_step, 'train_t', train_t, 'sim_t', sim_t, 'epsilon', epsilon, \
            'times_trained', times_trained, 'net_update_count', net_update_count

    profiler.tic('nn_cycle')
    if last_state != None:
        transitions.append((last_state, last_action, reward, state))
        current_a_hist.append(action)
        current_s_hist.append(state)

    #if train_t - last_old_net_update_time > old_net_update_time:

    maybe_update_old_net()
    action = None
    if train_t < 0.0:
        action = 0.0
    elif train_t - last_action_time > min_action_gap:
        last_action_time = train_t
        if np.random.random() > epsilon:
            profiler.tic('action')
            action = current_net.get_best_a_p(state, is_p=False, num_tries=2)[0][0][0]
            profiler.toc('action')
        else:
            action = random_action()
    else:
        action = last_action

    #print 'action', action
    if bang_action == 1:
        if action > 0.0:
            action = max_torque
        else:
            action = -max_torque

    profiler.tic('nn_cycle1')
    if ready_to_train and train_t - last_train_time > min_train_gap:
        train_once()

    f = open(python_action_file, 'w')
    f.write('%s\n' % action)
    f.close()
    profiler.toc('nn_cycle1')

    time_step += 1
    last_state = state
    last_action = action
    profiler.toc('nn_cycle')
    print
