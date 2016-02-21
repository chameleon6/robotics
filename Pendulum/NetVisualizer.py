import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np

from nn import *
from utils import *

class NetVisualizer:
    def __init__(self, net):
        self.net = net
        self.profiler = Profiler()

    def s_const_grid(self, s, xr, n=10000):
        # [[s x_1], [s x_2], ..., [s x_n]]
        s = np.array(s)
        xs = np.linspace(xr[0], xr[1], n)[:,np.newaxis]
        return xs, np.concatenate((np.ones((n,1)) * s[np.newaxis,:], xs), 1)

    # over range of a
    def graph_output(self, s, xr):
        assert self.net.n_a == 1
        xs, inputs = self.s_const_grid(s, xr)
        outputs = self.net.q_from_sa(inputs)
        plt.plot(xs, outputs)
        plt.show()

    # first coord constant -- gridworld
    def graph_max_qs(self, sr):
        assert self.net.n_s == 2
        xs, inputs = self.s_const_grid([0.0], sr, self.net.n_minibatch)
        outputs = self.net.get_best_a_p(inputs)[1]
        plt.plot(xs, outputs)
        plt.show()

    def manual_max_a(self, s, xr):
        assert self.net.n_a == 1
        xs, inputs = self.s_const_grid(s, xr)
        outputs = self.net.q_from_sa(inputs).flatten()
        best_input = np.argmax(outputs)
        return np.array([inputs[best_input][-1], outputs[best_input]])

    def manual_max_a_p(self, s, xr):
        assert self.net.n_a == 1
        ans = []
        for i in s:
            ans.append(self.manual_max_a(i, xr))
        return np.array(ans)

    def q_heat_map(self):

        #y, x = np.mgrid[slice(-3, 3., 2*3./20), slice(0, 2*np.pi, 2*np.pi/20)]
        th_r = (1, 4.)
        th_dot_r = (-0.5, 0.5)
        y, x = np.mgrid[slice(th_dot_r[0], th_dot_r[1], (th_dot_r[1] - th_dot_r[0])/20),
                slice(th_r[0], th_r[1], (th_r[1] - th_r[0])/20)]

        #z = np.sin(x)**10 + np.cos(10 + y*x) * np.cos(x)
        z = []
        self.profiler.tic('heat map max calculations')
        for i,j in zip(x,y):
            s = np.concatenate((i[:,np.newaxis], j[:,np.newaxis]), 1)
            q = self.net.get_best_a_p(s, is_p=True, num_tries=1)[1].flatten()
            #print q
            z.append(q)
        self.profiler.toc('heat map max calculations')

        z = np.array(z)
        print z

        for _ in range(6):
            i = np.array([np.random.uniform(*th_r), np.random.uniform(*th_dot_r)])
            q = self.net.get_best_a_p(i, is_p=False, num_tries=3)[1][0][0]
            print i, q

        z = z[:-1, :-1]
        levels = MaxNLocator(nbins=15).tick_values(z.min(), z.max())

        cmap = plt.get_cmap('PiYG')
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        im = plt.pcolormesh(x, y, z, cmap=cmap, norm=norm)
        plt.colorbar(im)
        plt.title('pcolormesh with levels')
        plt.show()
