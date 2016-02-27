# function fitting test and max finding test
from nn import *
from utils import *
from NetVisualizer import NetVisualizer
import scipy.io as sio
import time
import sys

profiler = Profiler()
conf = read_conf('pendulum.conf')

start_time = time.time()
#save_path = 'models/model_20247.out'
save_path = 'models/model_35202.out'
net = ControlNN(save_path)
vis = NetVisualizer(net)
print "compile time", time.time() - start_time

s = np.array([0.5, 0.5])
n_minibatch = conf['minibatch_size'] * conf['num_batches']
num_mb = 1000
n_test = 1000

def ys_from_xs(xs):
    #return np.sum(xs,1)[:,np.newaxis]
    ys = -(xs[:,0]**2 + xs[:,1] + (xs[:,2] - 0.5)**2)
    return ys[:, np.newaxis]

def train_test():
    xs = np.random.uniform(-5,5, (mb_size*num_mb, 3))
    ys = ys_from_xs(xs)

    xs_test = np.random.uniform(-5,5, (n_test, 3))
    ys_test = ys_from_xs(xs_test)

    start_time = time.time()

    for i in range(num_mb):
        sample_x, sample_y = xs[mb_size*i:mb_size*(i+1)], ys[mb_size*i:mb_size*(i+1)]
        if i%1000 == 0:
            print i, net.mse_q(xs_test, ys_test), net.mse_q(sample_x, sample_y)
        net.train(sample_x, sample_y)
        if i%1000 == 0:
            print i, net.mse_q(xs_test, ys_test), net.mse_q(sample_x, sample_y)

    #net.save_model(save_path)
    print "train time", time.time() - start_time

def graph_max_a_test():
    print net.manual_max(s, [-5,5])
    print net.get_best_a(s)
    net.graph_output(s, (-5,5))

def max_verification_test(test_p):
    u_max = conf['max_torque']
    s_max = 2
    s = s_max * np.random.random((n_minibatch,2))
    #s1 = s_max * np.random.random(2)

    profiler.tic('manual max')
    print "manual max"
    manual_out = vis.manual_max_a_p(s, (-u_max,u_max)).reshape(-1,2)
    print manual_out
    profiler.toc('manual max')

    profiler.tic('net max')
    print "net max"

    if test_p:
        res_a, res_q = net.get_best_a_p(s, is_p=True, num_tries=1)
    else:
        res_a = []
        res_q = []
        for i in s:
            ra, rq = net.get_best_a_p(i, is_p=False, num_tries=1)
            res_a.append(ra.flatten())
            res_q.append(rq.flatten())

        res_a = np.array(res_a)
        res_q = np.array(res_q)

    net_out = np.concatenate((res_a, res_q), 1)
    #print net_out
    profiler.toc('net max')

    print "difference"
    diff = manual_out - net_out
    #print diff

    summary = np.concatenate((s, net_out, manual_out, diff), 1)
    print 'summary'
    print summary
    return summary

def dp_plot_test():
    C = None
    with open('dp_C.out', 'r') as f:
        C = np.array(map(float, f.read().strip().split('\n')))

    J = None
    with open('dp_J.out', 'r') as f:
        J = np.array(map(float, f.read().strip().split('\n')))

    T = sio.loadmat('dp_T.mat')['t'][0]
    B = np.array([T[i] * J for i in range(9)])
    C = C.reshape((9,-1))
    J2 = np.min(C+B, 0).reshape(51,51)

    vis = NetVisualizer()
    xr = (0, 2*np.pi, 51)
    xdr = (-10., 10., 51)
    vis.plot_heat_map(xr, xdr, J2)

#summary = max_verification_test(False)
vis.q_heat_map()
