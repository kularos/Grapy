import numpy as np
from matplotlib.transforms import Transform
from matplotlib.pyplot import imread
from matplotlib.colors import Normalize

directory_path = ''


__all__ = ['ColorSpace', 'VectorMappable', 'ComplexMappable']


class ColorSpace:
    """
    For a 1D ColorSpace (mpl.Colormap), The scalar-to RGBA mapping pipeline is more direct,
    flowing from input to output in a single object.
    In higher dimensions, this task becomes more involved.
    The ColorSpace object manages mapping from an input vector space, controlling a set of ColorAxis objects,
    each of which produces a scalar coordinate in the output ColorSpace. These scalar values are then recombined to form
    a unique co-ordinate in the output ColorSpace, which is then delivered.
    """

    vector_axis = -1

    def __init__(self, name, N):
        self.name = name
        self.N = tuple(int(n) for n in N)
        self._ndim = len(N)

        self._rgba_bad = (1., 1., 1., 1.)
        self._i_under = tuple(n for n in self.N)
        self._i_over = tuple(n + 1 for n in self.N)
        self._index_bad = tuple([-1] * self._ndim)

        self._isinit = False

    def __call__(self, X):

        if not self._isinit:
            self._init()

        xa = np.ma.masked_invalid(X)

        for i in range(self._ndim):
            xi = xa[..., i] * self.N[i]

            xi[xi < 0] = -1
            xi[xi == self.N[i]] = self.N[i] - 1
            np.clip(xi, -1, self.N[i], out=xi)

            xi[xi > self.N[i]] = self._i_over[i]
            xi[xi < 0] = self._i_under[i]

            xa[..., i] = xi
        xa = xa.astype(int)

        if xa.mask.shape == xa.shape:
            bad_vector = np.any(xa.mask, axis=-1)[..., None]
            xa = np.ma.where(bad_vector, -1, xa)

        # Convert xa into a tuple of entries along each axis,
        # Use each of these as a respective axis' indexing array
        rgba = self._lut[tuple(xa[..., i] for i in range(self._ndim))]

        return rgba

    def _init(self):
        self._lut = None
        """Generate the lookup table, self._lut"""
        raise NotImplementedError("Abstract class only")

    @staticmethod
    def from_png(file_name):
        img = imread(directory_path + file_name + '.png')
        return ColorSpace.from_array(img, file_name)

    @staticmethod
    def from_array(a, name=None):
        """
        after __init__ is passed, the stored lut_data is raw, and does not contain values for denormalized/invalid
        inputs. This method takes the raw lut_data, and pads each axis with duplicate entries for the max/min entry of
        lut_data across that respective axis. This creates a 'pocket' of unassigned values at the end of lut.
        we can use the index [-1,-1,...,-1] to call this value. This has the unfortunate effect that if a value is
        simultaneously denormalized in the positive direction in all axes, it will appear as a 'bad' value in the
        rgb mapping.
        """
        dims = tuple(n-1 for n in a.shape[:-1])
        if name is None:
            name = a.__name__
        cs = ColorSpace(name, dims)

        # if an alpha channel isn't present in img_data, add one at a=1.0
        if a.shape[-1] < 4:
            pad = np.ones((*a.shape[:-1], 4 - a.shape[-1]), dtype=a.dtype)
            a = np.concatenate((a, pad), axis=-1)

        # takes img data and sets copies of the first and last array to each axis.
        # this is done in the same method as the ADI class (Shout out to Karim).
        for i in range(cs._ndim):
            # Create copies of the last and first entries along axis 0 for the over and under lookup tables
            over = a[None, -1]
            under = a[None, 0]

            # Concatenate arrays into the expected indexing order along axis 0
            a = np.concatenate((a, under, over), axis=0)

            # Rotate img array such that the (i+1)-th axis is in the 0 position
            a = np.moveaxis(a, 0, -2)

        # in the intersection of the i_over/i_under planes, insert the rgba_bad color values
        a[cs._index_bad] = cs._rgba_bad

        cs._lut = a
        cs._isinit = True

        return cs

########################################################################################################################


class VectorMappable:
    """
    This class serves as a generalization of the ScalarMappable offered by matplotlib, but generalized.
    Instead of mapping the internal _A of scalars into a Colormap object, this will treat _B as an array of vectors in a
    _B.__len__()[-1]-Dimensional vector space, and perform the mapping between this vector space and an arbitrary
    ColorSpace.
    """

    def __init__(self, transform, norms, cspace):
        self.transform = transform
        self.cspace = cspace

        # self.norms is a collection of None and Normalize objects. If an output axis in self.transform is constrained,
        # it need not be normalized in some cases
        self.norms = norms

        self.input_dims = transform.input_dims

        # ensure that the transform can validly map to the colorspace
        assert transform.output_dims == len(cspace.N) == len(norms)
        self.output_rank = transform.output_dims

        self._A = None

    def to_rgba(self, V, alpha=None, bytes=None):
        """
        this function call occurs in 3 main steps.
        1 self.transform performs the vectorspace mapping from V into the colorspace
        2 the list of self.norms are called on each axis of the colorspace
        3 rgba is pulled from the self.cspace
        """
        # arbitrary mapping
        colorvector = self.transform.transform_non_affine(V, )

        # normalize each axis
        for i in range(self.output_rank):
            norm = self.norms[i]
            if norm is not None:
                colorvector[..., i] = norm(colorvector[..., i])

        # pull from cspace
        rgba = self.cspace(colorvector)
        return rgba


class complex_to_polar(Transform):
    """
    Transforms a complex array to a scalar into a 2-dimensional vector of [r, φ] values
    """

    input_dims = 1
    output_dims = 2
    has_inverse = False

    def __copy__(self, *args):
        pass

    def inverted(self):
        pass

    @staticmethod
    def transform_non_affine(values, **kwargs):
        r = np.abs(values)
        th = np.angle(values)
        return np.stack((r, th), axis=-1)


class complex_to_cartesian(Transform):
    """
    Transforms a complex array to a scalar into a 2-dimensional vector of [r, φ] values
    """

    input_dims = 1
    output_dims = 2
    has_inverse = False

    def __copy__(self, *args):
        pass

    def inverted(self):
        pass

    @staticmethod
    def transform_non_affine(values, **kwargs):
        x = np.real(values)
        y = np.imag(values)
        return np.stack((x, y), axis=-1)


class ComplexMappable(VectorMappable):
    """
    special case for complex mapping
    """
    def __init__(self, cmap_name='large'):
        r_norm = Normalize(vmin=0, vmax=None)
        th_norm = Normalize(vmin=-np.pi, vmax=np.pi)
        cspace = ColorSpace.from_png(cmap_name)
        super().__init__(complex_to_polar, (r_norm, th_norm), cspace)
