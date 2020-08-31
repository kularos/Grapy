import numpy as np
from matplotlib.image import AxesImage
from matplotlib.animation import FuncAnimation

from .Mapping import *


class ComplexImage(ComplexMappable, AxesImage):
    """
    not an exact copy of the AxesImage, maps complex to scalar, and then scalar to RGB
    """
    def __init__(self, fig, imdata, cmap_name="large"):
        ComplexMappable.__init__(self, cmap_name)

        # setup figure and axes
        self.ax1 = fig.add_subplot()
        self.ax1.axis('off')
        self.im1 = self.ax1.imshow(np.zeros(imdata.shape))

        self._animated = False

    def set_data(self, A):
        self._A = A
        self.plot(self._A)

    def plot(self, imdata):
        rgb_image1 = self.to_rgba(imdata)
        self.im1.set_array(rgb_image1)


class FunctionAnimation(FuncAnimation):

    def __init__(self, fig, domain, function, cache=False, **kwargs):

        plotting_function = self.mesh_plot
        if domain.is_regular:
            plotting_function = self.image_plot

        FuncAnimation.__init__(self, fig, plotting_function, frames=function)

    def image_plot(self):
        pass

    def mesh_plot(self):
        pass
