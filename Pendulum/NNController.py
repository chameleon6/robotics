import numpy as np
import os
import sys
import time
import copy
from nn import ControlNN
from TransitionContainer import TransitionContainer

matlab_state_file = os.getcwd() + '/matlab_state_file.out'
python_action_file = os.getcwd() + '/python_action_file.out'

time_step = 1
last_state = None
last_action = None
transitions = TransitionContainer()

current_net = ControlNN()
old_net = ControlNN()
old_net_update_time = 100
epsilon = 1.0
final_epsilon = 0.1
epsilon_anneal_time = 10000
minibatch_size = 20
gamma = 0.99
ready_to_train = False

transfer_path = '/tmp/model_transfer'

def random_action():
    return np.random.uniform(-10,10)

while True:
    start_time = time.time()
    while not os.path.isfile(matlab_state_file):
        if time.time() - start_time > 10:
            sys.exit()
        pass

    f = open(matlab_state_file, 'r')
    lines = f.readlines()
    f.close()
    if len(lines) < 2: # file isn't fully written yet
        continue

    print "iter", time_step
    print lines
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

    action = current_net.get_best_a(state)[0] if np.random.random() > epsilon else random_action()
    print 'action', action

    if ready_to_train:
        ts = transitions.random_sample(minibatch_size)
        print ts
        def y_from_sample(sample):
            return sample[2] + gamma * old_net.get_best_a(sample[3])[1]
        ys = map(y_from_sample, ts)
        sa = np.array([t[0] + t[1] for t in ts])
        current_net.train(sa, ys)
        epsilon = max(final_epsilon, 1 - (1-final_epsilon) / epsilon_anneal_time * time_step)

    f = open(python_action_file, 'w')
    f.write('%s\n' % action)
    f.close()

    time_step += 1
    last_state = state
    last_action = action

    if time_step % 10 == 0:
        print transitions
