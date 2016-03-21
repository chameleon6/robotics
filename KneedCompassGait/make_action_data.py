import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle

#good_files = sys.argv[1]
#output = sys.argv[1]
output = 'action_train_data.p'
print 'saving to', output

xs = np.zeros((0, 18))
us = np.zeros((0, 6))
for line in open('good_simbicon_files.out', 'r'):
#for line in open(good_files, 'r'):
    fname = line.strip()
    lines = open(fname, 'r').readlines()
    lines = map(lambda l: map(float, l.strip().split(' ')), lines)
    x = np.array(lines[::2])
    u = np.array(lines[1::2])
    xs = np.concatenate((xs, x), 0)
    us = np.concatenate((us, u), 0)

print 'max/min u', np.max(us), np.min(us)
print 'max/min x', np.max(xs), np.min(xs)
print us.shape, xs.shape
#pickle.dump((xs, us), open('../NN/simbicon_train_data.p', 'wb'))
pickle.dump((xs, us), open('../NN/' + output, 'wb'))
