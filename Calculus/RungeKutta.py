import numpy as np
import pandas as pd


def load_butcher(butcher_name, **kwargs):
    path = 'dir/' + butcher_name + '.csv'

    def from_csv(path, delimiter=','):
        """ Provides an interface for reading a template array from CSV files.
        The standard format for these files is header-less.
        @param path: path to csv file
        @param delimiter: delimiter for parsing CSV file, defaults to ','
        """
        butcher_raw = pd.read_csv(path, sep=delimiter, header=None)
        butcher_array = butcher_raw.apply(pd.eval).to_numpy()
        return butcher_array

    butcher_array = from_csv(path, **kwargs)

    #get p from dict (can calculate, but the math is FUCKED)
    p_dict = {'dp45' :5,
              'euler':2,
              'heun' :2,
              'rk4'  :4,
              'rk375':4}

    #Initialize create dict based upon a provided butcher array.
    butcher_dict = {}
    rows, cols = butcher_array.shape
    butcher_dict["A"] = butcher_array[:cols] # Square matrix
    butcher_dict["b"] = butcher_array[cols]
    butcher_dict["b_"] = None  # Assume non-adaptive case.
    butcher_dict["c"] = butcher_array[-1]
    butcher_dict["s"] = cols
    butcher_dict["p"] = p_dict[butcher_name]

    # check to see if b_ is present, and react accordingly
    if rows - cols == 3:
        butcher_dict["b_"] = butcher_array[-2]
    elif rows - cols != 2:
        raise ValueError('Butcher tableau is improperly formatted:'
                         'Could not infer from shape({0}, {1})'.format(rows,
                                                                       cols))

    if len(butcher_array.shape) != 2:
        raise ValueError('Butcher tableau is improperly formatted:'
                         'template array must be a Matrix.')

    return butcher_dict

class RungeKutta:
    # define signature
    A, b, b_, c, s, p = None, None, None, None, None, None

    def __init__(self, func, t0, y0, dt, name='rk4', e_drift=0.1, **kwargs):
        # Instance properties for IVP.
        self.y = y0
        self.t = t0
        self.func = func

        # Finite differential quantities
        self.dt = dt
        self._e_drift = e_drift
        self.__dict__.update(load_butcher(name))

    def __iter__(self):
        return self

    def __next__(self):
        y_new, t_new = self.time_step(self.y, self.t)
        self.y, self.t = y_new, t_new
        return y_new, t_new

    def time_step(self, y0, t0):
        k = np.zeros(self.s)
        t1 = t0 + self.dt

        for j in range(self.s):
            # Explicitly calculate approximations of derivative.
            dt_j = self.dt * self.c[j]
            dy_j = dt_j * np.dot(k, self.A[:, j])
            k[j] = self.func(t0 + dt_j, y0 + dy_j)

        y1 = y0 + self.dt * np.dot(k, self.b)

        # End non-adaptive schemes here
        if self.b_ is None:
            return y1, t1

        # Weird issue where y1 and y1_ are way closer than they should be.
        # This causes adaptive schemes to exponentially grow in step size.
        y1_ = y0 + self.dt*np.dot(k, self.b_)
        sigma = (self._e_drift * self.dt / abs(y1 - y1_)) ** (1 / (self.p - 1))

        # Look into seeing if this segment can be tuned to get a desired
        # speed/accuracy relationship.
        if sigma >= 2:
            self.dt *= 2
            return y1, t1
        elif sigma <= 1:
            self.dt /= 2
            return self.time_step(y0, t0)
        else:
            return y1, t1


def harder_dp45(f, t_rng, y0, h, eps_abs):
    t0, tf = t_rng

    # Follows Harder's 0.5 safety factor
    e_drift = eps_abs / (tf - t0) / 2

    iterator = RungeKutta(f, t0, y0, h, name='heun', e_drift=e_drift)

    y_out = [y0]
    t_out = [t0]

    while iterator.t < tf:

        y, t = iterator.__next__()

        y_out.append(y)
        t_out.append(t)

        #if iterator.t + iterator.dt > tf:
        #    iterator.dt = tf - iterator.t

    y_out = np.asarray(y_out)
    t_out = np.asarray(t_out)
    return t_out, y_out


if __name__ == '__main__':

    def f4a(t, y):
        return y*(2-t) * t + (t - 1)

    t2a, y2a = harder_dp45(f4a, [0, 5], 1, 0.1, 10e-3)

    import matplotlib.pyplot as plt

    n = np.linspace(0, 1, t2a.size)

    plt.plot(n, t2a, '.')
    plt.show()