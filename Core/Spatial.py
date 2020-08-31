import numpy as np
import math

from .Expression import sanitize_functions


def position_array(*xi, **kwargs):
    """ Copies the numpy meshgrid interface, but returns a Domain-friendly array of point positions.
    :returns: (npoints, ndim) point array.
    """
    try:
        if kwargs["sparse"]:
            raise NotImplementedError("Sparse position arrays are not permitted.")
    except KeyError:
        pass

    array = np.stack(np.meshgrid(*xi, **kwargs), axis=-1)

    npoints = math.prod(x.size for x in xi)
    ndim = len(xi)

    return array.reshape((npoints, ndim))


class Domain:
    """Domain class which uses numpy arrays to contain fields"""

    # _neighbor_array is implemented for compatibility with the qhull interface, and is useful
    # when the total number of neighbors that a point has is the same for most points.

    def __init__(self, point_array, dtype=None):

        # Expand point_array if only 1D
        if point_array.ndim == 1:
            point_array = point_array[:, None]

        # Cast to provided dtype if necessary, and set internal point array
        if dtype is not None:
            self._dtype = dtype
            self._point_array = point_array.astype(dtype)
        else:
            self._dtype = point_array.dtype
            self._point_array = point_array

        # self.points will take the shape of [cardinality, dimension],
        # essentially a collection of position vectors.
        self.cardinality, self.dimension = self._point_array.shape
        self._connectivity_array = self._gen_connectivity_array()

    def _gen_connectivity_array(self):
        raise NotImplementedError("Abstract base class")

    @property
    def points(self):
        """Generates every point vector in *self._point_array*"""
        for i in range(self.cardinality):
            yield self._point_array[i]

    @property
    def connectivity(self):
        """Generates the indices of all points that are connected to each central point"""
        for i in range(self.cardinality):
            indices = self._connectivity_array[i]
            valid_indices = np.extract(indices != -1, indices)
            yield i, *valid_indices

    @property
    def subdomains(self):
        """Generates all subdomains of connected points, with the central point at index 0"""
        for subdomain_index in self.connectivity:
            subdomain = self._point_array[subdomain_index, :]
            subdomain -= subdomain_index[0]
            yield subdomain


class RegularDomain(Domain):

    def __init__(self, point_array, wrapped=False, **kwargs):
        self.wrapped = wrapped
        Domain.__init__(self, point_array, **kwargs)

    def _gen_connectivity_array(self):
        if self.wrapped:
            return self._toroidal_connectivity()
        else:
            return self._closed_connectivity()

    def _toroidal_connectivity(self):
        """Generates connectivity for array, assuming topology is that of an N-torus"""
        raise NotImplementedError("Abstract base class")

    def _closed_connectivity(self):
        """Generates connectivity for array, assuming topology is that of an N-torus"""
        raise NotImplementedError("Abstract base class")

    def to_points(self, array):
        """Reshapes an array to match the indexing of self.points"""
        raise NotImplementedError("Abstract base class")

    def to_array(self, points):
        """Reshapes a vector to match self.shape"""
        raise NotImplementedError("Abstract base class")


class RectangularDomain(RegularDomain):
    """This domain subclass is designed to take advantage of the structural properties of arrays.
    Each regular domain has a pair of functions to_points, and to_array which ravel and unravel external
    arrays respectively. This allows regular (n-rectangular) arrays to be used as domains.
    """

    def __init__(self, domain_array, wrapped=False, **kwargs):
        # we can now explicitly state whether the topology should be interpreted as wrapped.
        self.wrapped = wrapped

        # By convention, the 'vector' dimension of the array is -1.
        self.shape = domain_array.shape[:-1]
        npoints, ndim = math.prod(self.shape), domain_array.shape[-1]

        self._to_array = np.arange(npoints).reshape(self.shape)

        index_vectors = position_array(*(np.arange(dim) for dim in self.shape), indexing='ij')
        points_index = index_vectors[np.argsort(self._to_array.flatten()), ...]
        self._to_points = tuple(points_index[..., i] for i in range(ndim))

        point_array = domain_array[self._to_points]

        Domain.__init__(self, point_array, **kwargs)

    def _toroidal_connectivity(self):
        """Generates connectivity for array, assuming topology is that of an N-torus."""
        connectivity = -np.ones((self.cardinality, 2 * self.dimension), dtype='int')
        j, k = 0, 1

        for i in range(self.dimension):
            neg_roll = np.roll(self._to_array, 1, axis=i)
            connectivity[:, j] = neg_roll[self._to_points]
            j += 2

            pos_roll = np.roll(self._to_array, -1, axis=i)
            connectivity[:, k] = pos_roll[self._to_points]
            k += 2
        return connectivity

    def _closed_connectivity(self):  # untested
        """Generates connectivity for array, assuming topology is closed."""
        connectivity = -np.ones((self.cardinality, 2 * self.dimension), dtype='int')
        j, k = 0, 1

        invalid_indices = -np.ones(self.shape, dtype="int")
        upper_slice = slice(None, -1, None)
        lower_slice = slice(1,  None, None)

        for i in range(self.dimension):
            shifted_up, shifted_down = invalid_indices.copy(), invalid_indices.copy()

            upper_array = np.take(self._to_array, upper_slice, axis=i)
            shifted_up[lower_slice] = upper_array
            connectivity[:, j] = self.to_points(shifted_up)
            j += 2

            lower_array = np.take(self._to_array, lower_slice, axis=i)
            shifted_down[upper_slice] = lower_array
            connectivity[:, k] = self.to_points(shifted_down)
            k += 2

    def to_points(self, array):
        """Reshapes an array to match the indexing of self.points"""
        return array[self._to_points]

    def to_array(self, points):
        """Reshapes a vector to match self.shape"""
        return points[self._to_array]

    @property
    def subdomains(self):
        """Generates all subdomains of connected points, with the central point at index 0"""
        for subdomain_index in self.connectivity:
            subdomain = self._point_array[subdomain_index, :]
            subdomain -= subdomain[0, :]
            yield subdomain


class FDGenerator:
    """
    a factory class, designed to create and return a set of equations to be used in finite difference schemes.

    This class will likely be drawn and quartered, with the bits being distributed among domain and operator classes
    """

    def __init__(self, domain, equation, dtype=None):
        self.dtype = dtype
        self.domain = domain
        self.equation = equation
        self.order = len(self.equation)

        self._disc_funcs = sanitize_functions(equation, domain)

        self._create_fd_matrix()

    def inverse_taylor_nd(self, displacement):
        # This current scheme is bupkis,
        orders = np.asarray([(2, 0), (1, 0), (0, 2), (0, 1), (0, 0)]).T

        n = displacement.shape[0]

        # use the generated power values to raise each offset to the correct
        # power for each differential basis operator.
        coefficients = np.ones((n, n), dtype=self.dtype)

        for i in range(n):
            powered = np.apply_along_axis(np.power, 1, displacement, orders[:, i])
            prodded = np.prod(powered, axis=1)
            coefficients[:, i] = prodded

        coefficients = np.linalg.inv(coefficients)
        for i in range(n):
            coefficients[i, :] *= math.prod((math.factorial(p) for p in orders[:, i]))

        return coefficients

    def _create_fd_matrix(self):
        domain = self.domain
        matrix = np.zeros((domain.cardinality, domain.cardinality), dtype=self.dtype)

        for index, subdomain in zip(domain.connectivity, domain.subdomains):

            fd_matrix = self.inverse_taylor_nd(subdomain)
            fd_equation = np.dot(self._disc_funcs[index[0], :], fd_matrix)

            matrix[index[0], index] = fd_equation

        self.matrix = matrix
