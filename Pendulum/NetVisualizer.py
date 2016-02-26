import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np

from nn import *
from utils import *

class NetVisualizer:
    def __init__(self, net=None):
        self.net = net
        self.profiler = Profiler()

    def s_const_grid(self, s, xr, n=10000):
        # [[s x_1], [s x_2], ..., [s x_n]]
        s = np.array(s)
        xs = np.linspace(xr[0], xr[1], n)[:,np.newaxis]
        return xs, np.concatenate((np.ones((n,1)) * s[np.newaxis,:], xs), 1)

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

    def q_heat_map(self):

        #y, x = np.mgrid[slice(-3, 3., 2*3./20), slice(0, 2*np.pi, 2*np.pi/20)]
        th_r = (2.0, 4.2, 40)
        th_dot_r = (-5.5, 5.5, 40)
        x, y = self.xy_grid(th_r, th_dot_r)

        #z = np.sin(x)**10 + np.cos(10 + y*x) * np.cos(x)
        z = np.zeros((0,40))
        self.profiler.tic('heat map max calculations')
        current_buf = np.zeros((0,2))
        for i,j in zip(x,y):
            s = np.concatenate((i[:,np.newaxis], j[:,np.newaxis]), 1)
            current_buf = np.concatenate((current_buf, s), 0)
            if current_buf.shape[0] == 200:
                #q = self.net.get_best_a_p(current_buf, is_p=True, num_tries=3, tolerance=0.01)[1].flatten()
                q = self.net.manual_max_a_p(current_buf)[:,1][:,np.newaxis]
                q_r = q.reshape((5,40))
                print q
                print q_r
                z = np.concatenate((z, q_r),0)
                current_buf = np.zeros((0,2))

        self.profiler.toc('heat map max calculations')
        print z
        for _ in range(6):
            i = np.array([np.random.uniform(th_r[0], th_r[1]),
                np.random.uniform(th_dot_r[0], th_dot_r[1])])
            q = self.net.get_best_a_p(i, is_p=False, num_tries=3)[1][0][0]
            print i, q
        self.plot_heat_map(th_r, th_dot_r, z)


    def xy_grid(self, xr, yr):
        assert len(xr) == 3 and len(yr) == 3
        y, x = np.mgrid[slice(yr[0], yr[1], (yr[1] - yr[0])/yr[2]),
                slice(xr[0], xr[1], (xr[1] - xr[0])/xr[2])]
        return x,y

    def plot_heat_map(self, xr, yr, z):
        z = z[:-1, :-1]
        x, y = self.xy_grid(xr, yr)
        levels = MaxNLocator(nbins=15).tick_values(z.min(), z.max())

        cmap = plt.get_cmap('PiYG')
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        im = plt.pcolormesh(x, y, z, cmap=cmap, norm=norm)
        plt.colorbar(im)
        plt.title('pcolormesh with levels')
        plt.show()
