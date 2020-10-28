import numpy as np

# Unfinished port of dp45 from Harder's class.
# Edits are likely to revolve around porting matlab's native matrix interpretation
# to explicit numpy statements, and indexing quirks.

rk4 = {
    'A': np.asarray([[0.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]),
    'b': np.asarray([1/3, 1/6, 1/6, 1/3]),
    'b*': None,
    'c': np.asarray([0.0, 0.5, 0.5, 1.0])
    }

dp45 = {
    'A': np.asarray([[0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0], [1/4, 3/4, 0, 0, 0, 0, 0], [11/9, -14/3, 40/9, 0, 0, 0, 0],[4843/1458, -3170/243, 8056/729, -53/162, 0, 0, 0],[9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0], [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]]),
    'b': np.asarray([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]),
    'b*': None,
    'c': np.asarray([0.0, 0.5, 0.5, 1.0]),
    }

class RungeKutta:
    # 4th order by default
    A = np.asarray([[0.0, 0.0, 0.0, 0.0], # Matrix of slopes for each approximation at each step
                    [0.5, 0.0, 0.0, 0.0],
                    [0.0, 0.5, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0]])
    b  = np.asarray([1/3, 1/6, 1/6, 1/3] ) # Weight vector for linear approximations.
    b_ = np.asarray([1/3, 1/6, 1/6, 1/3] )
    c = np.asarray( [0.0, 0.5, 0.5, 1.0] ) # Sizes of each partial approximation step, relative to h
    s = 4

    def __init__(self, f, t0, y0, h):
        self.f = f
        self.t, self.y = t0, y0
        self.h = h

    def __iter__(self):
        """Docstring Inherited."""
        return self

    def __next__(self):
        """"""

        k = np.zeros(self.s)

        for j in range(self.s):
            # Read c to for relative spacing of t for approx j
            dt_j = self.h * self.c[j]
            # Read A for weightings of past k values to calculate dy for approx j
            dy_j = dt_j * np.dot(k, self.A[:, j])
            # Calculate actual value k of f for approx j
            k[j] = self.f(self.t + dt_j, self.y + dy_j)


        self.y = self.y + self.h * np.dot(k, self.b)
        self.t += self.h

        return self.y, self.t


def rk4(f, t_rng, y0, n):
    t0, tf = t_rng
    h = (tf - t0) / (n - 1)

    t_out = np.zeros(n)
    y_out = np.zeros(n)

    t_out[0], y_out[0] = t0, y0

    A = np.asarray([[0.0, 0.0, 0.0, 0.0],
                    [0.5, 0.0, 0.0, 0.0],
                    [0.0, 0.5, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0]])
    b = np.asarray( [1/3, 1/6, 1/6, 1/3] )
    c = np.asarray( [0.0, 0.5, 0.5, 1.0] )


    for k in range(n):

        n_k = 4
        k = np.zeros((1, n_k))

        for m in range(n_k):
            dt = h * c[m]
            dy = dt * np.dot(k, A[:, m])
            k[m] = f(t_out[k] + dt, y_out[k] + dy)

        y_tmp = y_out[k] + h * np.dot(k, b)

        y_out[k + 1] = y_tmp
        t_out[k + 1] = t_out[k] + h

    return t_out, y_out