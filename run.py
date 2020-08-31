import numpy as np
import matplotlib.pyplot as plt

import Graphic
import Core


class Calculator_legacy:

    def __init__(self, domain, equation, dt, u0, dtype=None):
        # iterator
        self.domain = domain
        self.dt = dt

        u0 = domain.to_points(u0)
        self.PDE = Core.euler(domain, equation, dt, u0, dtype=dtype)

        # figure:
        self.fig = plt.figure(facecolor="#000000")
        self.fig.labelcolor = '#FFFFFF'
        self.ax1 = self.fig.add_subplot()
        self.ax1.axis('off')
        self.im1 = self.ax1.imshow(np.zeros(domain.shape))

        # cspace
        self.cspace = Graphic.ComplexMappable()

    def run(self):
        tc = 0
        while plt.fignum_exists(1):
            u = next(self.PDE)

            # animator step
            u_rect = self.domain.to_array(u)
            rgb_image = self.cspace.to_rgba(u_rect)
            self.im1.set_array(rgb_image)

            self.fig.suptitle("Time = {0:.3f}".format(tc * self.dt))

            # Pause
            plt.pause(self.dt)
            tc += 1


if __name__ == "__main__":
    def gaussian(x, s, m):
        return np.exp(-(((x - m * np.ones(x.shape, dtype=x.dtype)) / s) ** 2))


    def momentum(x, p):
        return np.exp(x * p * 1j)


    def angular(x, y, ang, s):
        return (x + s * y * 1j) ** ang


    # Set xy grid
    y0 = np.linspace(-0.5, 0.5, 30)
    x0 = np.linspace(-0.5, 0.5, 30)

    grid = np.meshgrid(x0, y0, indexing='ij')
    grid_array = np.stack(grid, axis=-1)

    Domain = Core.RectangularDomain(grid_array, wrapped=True)
    X, Y = grid  # X and Y are matrices, while x0 and y0 are vectors

    # Set initial psi and V w
    d = 0.0
    S = 0.1

    psi0 = (angular(X, Y, 1, 1) + angular(X, Y, 1, -1)) * gaussian(X, S, 0) * gaussian(Y, S, 0)
    V = 16 * (X ** 2 + Y ** 2) ** 0.5

    hbar = 0.05
    mass = 0.5
    schrodinger_eq = [-hbar / mass * 0.5j, 0, -hbar / mass * 0.5j, 0, 1j / hbar * V.flatten()]

    # build and run simulator
    simulator = Calculator_legacy(Domain, schrodinger_eq, 0.005, psi0, dtype='complex')
    simulator.run()
    plt.show()
