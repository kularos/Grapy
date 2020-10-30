import numpy as np
import pandas as pd

def load_butcher(butcher_name, **kwargs):
    """Provides an interface for reading a template array from CSV files.
    The standard format for these files is as a header-less
    @param delimiter: delimiter for parsing CSV file, defaults to ','
    @param path: path to file.
    @return: _ButcherTableau object.
    """
    path = 'dir/' + butcher_name + '.csv'

    def from_csv(path, delimiter=','):
        butcher_raw = pd.read_csv(path, sep=delimiter, header=None)
        butcher_array = butcher_raw.apply(pd.eval).to_numpy()
        return butcher_array

    return from_csv(path, **kwargs)

class ODESolver:
    """Base class for iteratively solving systems of ODEs."""

    def __init__(self, func, y0, t0, dt = 0.01, epsilon = 0.001, **kwargs):
        self.y = y0
        self.t = t0
        self.func = func

        # Finite differential quantities
        self._dt  = dt
        self._epsilon = epsilon

    def __iter__(self):
        return self

    def time_step(self):
        y_new, t_new = self.y, self.t
        return y_new, t_new

    def __next__(self):
        y_new, t_new = self.time_step()
        self.y, self.t = y_new, t_new
        return y_new, t_new


class RungeKutta(ODESolver):

    def __init__(self, func, t0, y0, name='rk4', **kwargs):
        ODESolver.__init__(self, func, y0, t0, **kwargs)
        butcher_array = load_butcher(name)
        self._init(butcher_array)

    def _init(self, butcher_array):
        """Initialize internal machinery based upon a provided butcher array."""
        rows, cols = butcher_array.shape
        self._A = butcher_array[:cols]  # Square matrix
        self._b = butcher_array[cols]
        self._c = butcher_array[-1]
        self._s = cols
        self._b_ = None  # Assume non-adaptive case.

        # check to see if b_ is present, and react accordingly
        if rows - cols == 3:
            self.b_ = butcher_array[-2]
        elif rows - cols != 2:
            raise ValueError(
                'Butcher tableau is improperly formatted:'
                'Could not infer from shape({0}, {1})'.format(rows, cols))

        if len(butcher_array.shape) != 2:
            raise ValueError(
                'Butcher tableau is improperly formatted:'
                'template array must be a Matrix.')

    def time_step(self):
        k = np.zeros(self._s)

        for j in range(self._s):
            dt_j = self._dt * self._c[j]
            dy_j = dt_j * np.dot(k, self._A[:, j])
            k[j] = self.func(self.t + dt_j, self.y + dy_j)

        y_new = self.y + self._dt * np.dot(k, self._b)
        t_new = self.t + self._dt

        return y_new, t_new



if __name__ == '__main__':



    runge = RungeKutta()
