# function fitting test and max finding test
from nn import *
from utils import *
import time
import sys

profiler = Profiler()

start_time = time.time()
save_path = None #'/tmp/model.ckpt'
net = ControlNN(save_path)
print "compile time", time.time() - start_time

s = np.array([0.5, 0.5])
mb_size = 20
num_mb = 1000
n_test = 1000

def ys_from_xs(xs):
    #return np.sum(xs,1)[:,np.newaxis]
    ys = -(xs[:,0]**2 + xs[:,1] + (xs[:,2] - 0.5)**2)
    return ys[:, np.newaxis]

# xs = np.random.uniform(-5,5, (mb_size*num_mb, 3))
# ys = ys_from_xs(xs)
#
# xs_test = np.random.uniform(-5,5, (n_test, 3))
# ys_test = ys_from_xs(xs_test)
#
# start_time = time.time()
#
# for i in range(num_mb):
#     sample_x, sample_y = xs[mb_size*i:mb_size*(i+1)], ys[mb_size*i:mb_size*(i+1)]
#     if i%1000 == 0:
#         print i, net.mse_q(xs_test, ys_test), net.mse_q(sample_x, sample_y)
#     net.train(sample_x, sample_y)
#     if i%1000 == 0:
#         print i, net.mse_q(xs_test, ys_test), net.mse_q(sample_x, sample_y)
#
# #net.save_model(save_path)
# print "train time", time.time() - start_time

# print net.manual_max(s, [-5,5])
# print net.get_best_a(s)
# net.graph_output(s, (-5,5))

s_max = 50
s = s_max * np.random.random((20,2))
print "manual max"
manual_out = net.manual_max_a_p(s, (-s_max,s_max))
print manual_out

profiler.tic('net max')
print "net max"
net_out = np.concatenate(net.get_best_a_p(s), 1)
#print net_out
profiler.toc('net max')

print "difference"
diff = manual_out - net_out
#print diff

print 'summary'
print np.concatenate((net_out, manual_out, diff), 1)
