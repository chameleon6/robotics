from nn import *
from utils import *

class NetVisualizer:
    def __init__(self, net):
        self.net = net

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
