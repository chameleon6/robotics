import sys
sys.path.append('..')

import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle
from NN.utils import *
from NN.nn import ControlNN

#TODO: import utils and use conf for certainty net
certainty_thresh = 0.5
min_ind = 200
conf = read_conf('../NN/simbicon.conf')
conf['n_q'] = 1
conf['n_a'] = 0
certainty_net_file = conf['certainty_net_file']
certainty_net = ControlNN(conf, certainty_net_file)

#good_files = sys.argv[1]
#output = sys.argv[1]
output = 'sas_train_data.p'
print 'saving to', output

profiler = Profiler()

inputs = np.zeros((0, 18+6))
outputs = np.zeros((0, 18))
for line in open('all_simbicon_files.out', 'r'):
#for line in open(good_files, 'r'):
    fname = line.strip()
    print fname
    lines = open(fname, 'r').readlines()
    lines = map(lambda l: map(float, l.strip().split(' ')), lines)
    x = np.array(lines[::2])
    u = np.array(lines[1::2])

    #profiler.tic('c')
    certainties = certainty_net.q_from_sa(x)
    #profiler.toc('c')
    # ind of first less than thresh
    for i in range(min_ind, len(certainties)):
        if certainties[i] < certainty_thresh:
            last_ind = i
            break

    print len(x), last_ind
    xmax, umax = np.max(np.abs(x)), np.max(np.abs(u))

    if xmax > 50.1 or umax > 50.1:
        print xmax, umax
        print 'skipping', fname
        continue

    x = x[:last_ind]
    u = u[:last_ind]
    sa = np.concatenate((x,u), 1)[:-1]
    s = x[1:]

    inputs = np.concatenate((inputs, sa), 0)
    outputs = np.concatenate((outputs, s), 0)

print 'max/min u', np.max(outputs), np.min(outputs)
print 'max/min x', np.max(inputs), np.min(inputs)
print outputs.shape, inputs.shape
#pickle.dump((inputs, outputs), open('../NN/simbicon_train_data.p', 'wb'))
pickle.dump((inputs, outputs), open('../NN/' + output, 'wb'))
