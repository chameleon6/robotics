from nn import *
import time
import sys

start_time = time.time()
net = ControlNN()
print "compile time", time.time() - start_time

s = np.array([1,1])
mb_size = 20
num_mb = 5000
n_test = 1000

def ys_from_xs(xs):
    #return np.sum(xs,1)[:,np.newaxis]
    ys = xs[:,0]**2 + xs[:,1] + xs[:,2]**2
    return ys[:, np.newaxis]

xs = np.random.uniform(-1,1, (mb_size*num_mb, 3))
ys = ys_from_xs(xs)

xs_test = np.random.uniform(-1,1, (n_test, 3))
ys_test = ys_from_xs(xs_test)

#net.graph_output(s)

start_time = time.time()

# print xs[0:2], ys[0:2]
# for i in range(1000):
#     net.train(xs[0:2], ys[0:2])
#     if i%100 == 0:
#         num_eval = 2
#         net_out = net.q_from_sa(xs[0:num_eval])
#         print net_out, ys[0:num_eval]
#         print np.mean((net_out - ys[0:num_eval])**2), net.mse_q(xs[0:num_eval], ys[0:num_eval])
#         print net.o1_from_sa(xs[0:num_eval])
#         net.print_params()
#
# sys.exit()

for i in range(num_mb):
    sample_x, sample_y = xs[mb_size*i:mb_size*(i+1)], ys[mb_size*i:mb_size*(i+1)]
    if i%1000 == 0:
        print i, net.mse_q(xs_test, ys_test), net.mse_q(sample_x, sample_y)
    net.train(sample_x, sample_y)
    if i%1000 == 0:
        print i, net.mse_q(xs_test, ys_test), net.mse_q(sample_x, sample_y)

print "train time", time.time() - start_time

print net.get_best_a(s)
net.graph_output(s)
