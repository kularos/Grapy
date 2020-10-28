import numpy as np

from .Spatial import FDGenerator


class RKIterator:
    """
    Runge Kutta Baby!!!
    """

    #
    a = [[0.0, 0.0, 0.0, 0.0],
         [0.5, 0.0, 0.0, 0.0],
         [0.0, 0.5, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0]]
    #
    b = [1/3, 1/6, 1/6, 1/3]
    #
    c = [0.0, 0.5, 0.5, 1.0]
    # number of steps
    s = 4

    def __init__(self, func, dt, y0):

        self.t = 0
        self.y = y0

        self.func = func

        self.dt = dt
        self.k = [0 for _ in range(self.s)]

    def _generate_k(self):
        for i in range(self.s):
            t_i = self.t + self.dt * self.c[i]
            # needs to be modified to allow implicit methods
            y_i = self.y + self.dt * sum((self.a[i][j] * self.k[j] for j in range(i)))

            self.k[i] = self.func(t_i, y_i)

    def __next__(self):
        self._generate_k()

        self.y = self.y + self.dt * sum((self.b[i] * self.k[i] for i in range(self.s)))
        self.t += self.dt

        return self.y

    def __iter__(self):
        return self


class euler:
    def __init__(self, domain, equation, dt, u0=None, dtype=None):
        self.domain = domain
        self.dt = dt
        self.dtype = dtype

        self.fd = FDGenerator(domain, equation, dtype=self.dtype)
        self._generate_matrices()

        # if u0 is none, assume u was initialized upstream.
        if u0 is not None:
            self.u = u0

    def _generate_matrices(self):
        # generate the kernel specifications

        matrix = self.fd.matrix
        identity = np.identity(self.domain.cardinality)

        self.A_imp = matrix * self.dt / 2 - identity
        self.A_exp = matrix * self.dt / 2 + identity

    def __iter__(self):
        return self

    def __next__(self):
        # I cannot for the life of me figure out where this negative sign came from,
        # but the example breaks when I remove it.
        self.u = -np.linalg.solve(self.A_imp, self.A_exp.dot(self.u))
        return self.u

if __name__ == '__main__':

    def F1(t, v):
        return t * v

    V = 2

    iterator = RKIterator(F1, V, 0.01)

    for i in range(10):
        print(iterator.__next__())