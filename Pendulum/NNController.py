import numpy as np
import os
import sys
import time
import copy
from utils import *
from nn import ControlNN
from TransitionContainer import TransitionContainer

matlab_state_file = os.getcwd() + '/matlab_state_file.out'
python_action_file = os.getcwd() + '/python_action_file.out'

time_step = 1
times_trained = 0
last_state = None
last_action = None
transitions = TransitionContainer()

current_net = ControlNN()
old_net = ControlNN()
old_net_update_time = 100
epsilon = 1.0
final_epsilon = 0.1
epsilon_anneal_time = 1000
minibatch_size = 20
gamma = 0.99
ready_to_train = False

transfer_path = '/tmp/model_transfer'
save_path = 'models/model_%s.out' % (int(time.time() * 10000) % 100000)
print 'saving to', save_path

max_torque = 3

train_mod = 10

def random_action():
    return np.random.uniform(-max_torque,max_torque)

print "ready for matlab"
while True:
    start_time = time.time()
    while not os.path.isfile(matlab_state_file):
        if time.time() - start_time > 10:
            print "timeout"
            current_net.save_model(save_path)
            sys.exit()
        pass

    f = open(matlab_state_file, 'r')
    lines = f.readlines()
    f.close()
    if len(lines) < 2: # file isn't fully written yet
        continue

    print "iter", time_step, "epsilon", epsilon, "times_trained", times_trained
    #print lines
    reward_str = lines[0].rstrip()
    reward = float(reward_str)

    state_strs = lines[1].rstrip().split(' ')
    state = np.array(map(float, state_strs))
    os.remove(matlab_state_file)

    if last_state != None:
        transitions.append((last_state, last_action, reward, state))

    if time_step % old_net_update_time == 0:
        ready_to_train = True
        current_net.save_model(transfer_path)
        old_net.load_model(transfer_path)

    action = current_net.get_best_a(state)[0][0][0] \
            if np.random.random() > epsilon else random_action()
    print 'action', action

    if ready_to_train and time_step % train_mod == 0:
        ts = transitions.random_sample(minibatch_size)
        #print ts
        def y_from_sample(sample):
            #start = time.time()
            y = sample[2] + gamma * old_net.get_best_a(sample[3])[1][0][0]
            #print "maximizing time", time.time() - start
            return y

        # tic('total max time')
        # ys = np.array(map(y_from_sample, ts))[:, np.newaxis]
        # toc('total max time')

        tic('total max time p')
        rs = np.array([t[2] for t in ts])[:, np.newaxis]
        new_states = np.array([t[3] for t in ts])
        ys = rs + gamma * old_net.get_best_a_p(new_states)[1]
        toc('total max time p')

        sa = np.array([np.append(t[0], t[1]) for t in ts])

        # print 'ys'
        # print ys
        # print 'sa'
        # print sa
        # print

        tic('train time')
        current_net.train(sa, ys)
        toc('train time')
        times_trained += 1
        epsilon = max(final_epsilon, 1 - (1-final_epsilon) / epsilon_anneal_time * time_step)

    f = open(python_action_file, 'w')
    f.write('%s\n' % action)
    f.close()

    time_step += 1
    last_state = state
    last_action = action
