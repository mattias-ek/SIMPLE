import functools
from idlelib.multicall import r

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.axes import Axes
import numpy as np
import warnings

import simple.utils, simple.models

import logging
logger = logging.getLogger('SIMPLE.plot')

__all__ = ['create_rose_plot']

class EndlessList(list):
    """
    A subclass of ``list`` that where the index will never go out of bounds. If a requested
    index is out of bounds, it will cycle around to the start of the list.

    Examples:
        >>> ls = simple.plot.Endlesslist(["a", "b", "c"])
        >>> ls[3]
        "a"
    """
    # Index will never go out of bounds. It will just start from the beginning if larger than the initial list.
    def __getitem__(self, index):
        value = super().__getitem__(index % len(self))
        if type(index) == slice:
            return EndlessList(value)
        else:
            return value


# colours appropriate for colour blindness
# Taken from https://davidmathlogic.com/colorblind/#%23000000-%23E69F00-%2356B4E9-%23009E73-%23F0E442-%230072B2-%23D55E00-%23CC79A7
all_colors=EndlessList(["#D55E00", "#56B4E9", "#009E73", "#E69F00", "#CC79A7", "#0072B2", "#F0E442"])
"""
[``Endlesslist``][simple.plot.EndlessList] containing the default colors used by simple plotting functions.
"""

all_linestyles = EndlessList(['-', (0, (4, 4)), (0, (2,1)),
                              (0, (4,2,1,2)), (0, (4,2,1,1,1,2)), (0, (4,2,1,1,1,1,1,2)),
                              (0, (2,1,2,2,1,2)), (0, (2,1,2,2,1,1,1,2)), (0, (2,1,2,2,1,1,1,1,1,2)),
                              (0, (2,1,2,1,2,2,1,2)), (0, (2,1,2,1,2,2,1,1,1,2)), (0, (2,1,2,1,2,2,1,1,1,1,1,2))])
"""
[``Endlesslist``][simple.plot.EndlessList] default line styles used by simple plotting functions.
"""

all_markers = EndlessList(["o", "s", "^", "D", "P","X", "v", "<", ">",  "*", "p", "d", "H"])
"""
[``Endlesslist``][simple.plot.EndlessList] default marker styles used by simple plotting functions.
"""

def get_axes(axes, projection=None):
    """
    Return the axes that should be used for plotting.

    Args:
        axes (): Must either be ``None``, in which case ``plt.gca()`` will be returned, a matplotlib axes instance, or
            any object that has a ``.gca()`` method.
        projection (): If given, an exception is raised if ``axes`` does not have this projection.
    """
    if axes is None:
        axes = plt.gca()
    elif isinstance(axes, Axes):
        pass
    elif hasattr(axes, 'gca'):
        axes = axes.gca() # if axes = plt
    else:
        raise ValueError('axes must be an Axes, Axes instance or have a gca() method that return an Axes')

    if projection is not None and axes.name != projection:
        raise TypeError(f'The selected axes has the wrong projection. Expected {projection} got {axes.name}.')
        
    return axes

def get_lscm(linestyle = False, color = False, marker=False):
    """
    Convert the ``linestyle``, ``color`` and ``marker`` arguments into [EndlessList][simple.plot.EndlessList]
    objects for plotting.

    Args:
        linestyle (): Either a single line style or a list of line styles. If ``True`` then the
            [default line styles][simple.plot.all_linestyles] is returned. If ``False`` or ``None`` a list
            containing only the no line sentinel is returned.
        color (): Single colour or a list of colours. If ``True`` then the
            [default colors][simple.plot.all_colors] is returned. If ``False`` or ``None`` a list
            containing only the color black is returned.
        marker (): Either a marker shape or a list of marker shapes. If ``True`` then the
            [default marker shapes][simple.plot.all_markers] shapes is returned. If ``False`` or ``None`` a
            list containing only the no marker sentinel is returned.

    Returns:
        (EndlessList, EndlessList, EndlessList): linestyles, colors, markers
    """
    if color is False or color is None:
        colors = EndlessList(["#000000"])
    elif color is True:
        colors = all_colors
    else:
        colors = EndlessList(color)

    if linestyle is False or linestyle is None:
        linestyles = EndlessList([""])
    elif linestyle is True:
        linestyles = all_linestyles
    else:
        linestyles = EndlessList(linestyle)

    if marker is False or marker is None:
        markers = EndlessList([""])
    elif marker is True:
        markers = all_markers
    else:
        markers = EndlessList(marker)

    return linestyles, colors, markers

def get_models(models, where=None, where_kwargs={}):
    """
    Return a selection of models.

    Args:
        models (ModelCollection): A collection of models.
        where (str): Only models fitting this criteria will be selected. If not given all models are selected.
        where_kwargs (): Keyword arguments to go with the ``where`` string.

    Returns:
        ModelCollection: The selected models
    """
    if not isinstance(models, simple.models.ModelCollection):
        raise TypeError(f'models must be a ModelCollection object not {type(models)}.')

    # select models
    if where is not None:
        models = models.where(where, **where_kwargs)

    return models

#################
### Rose plot ###
#################
def as1darray(*a, dtype=np.float64):
    size = None

    out = [np.asarray(x, dtype=dtype) if x is not None else x for x in a]

    for o in out:
        if o is None:
            continue
        if o.ndim == 1 and o.size != 1:
            if size is None:
                size = o.size
            elif o.size != size:
                raise ValueError('Size of arrays do not match')
        elif o.ndim > 1:
            raise ValueError('array cannot have more than 1 dimension')

    out = tuple(o if (o is None or (o.ndim == 1 and o.size != 1)) else np.full(size or 1, o) for o in out)
    if len(out) == 1:
        return out[0]
    else:
        return out


def as0darray(*a, dtype=np.float64):
    out = [np.asarray(x, dtype=dtype) if x is not None else x for x in a]

    for o in out:
        if o is None:
            continue
        if o.ndim >= 1 and o.size != 1:
            raise ValueError('Size of arrays must be 1')

    out = tuple(o if (o is None or o.ndim == 0) else o.reshape(tuple()) for o in out)
    if len(out) == 1:
        return out[0]
    else:
        return out


def xy2rad(x, y, xscale=1.0, yscale=1.0):
    def calc(x, y):
        if x == 0:
            return 0
        if y == 0:
            return np.pi / 2

        if x > 0:
            return np.pi * 0.5 - np.arctan(y / x)
        else:
            return np.pi * 1.5 - np.arctan(y / x)

    x, y = as1darray(x, y)
    return np.array([calc(x[i] / xscale, y[i] / yscale) for i in range(x.size)])


def xy2deg(x, y, xscale=1.0, yscale=1.0):
    rad = xy2rad(x, y, xscale=xscale, yscale=yscale)
    return rad2deg(rad)


def deg2rad(deg):
    return np.deg2rad(deg)


def rad2deg(rad):
    return np.rad2deg(rad)


def get_cmap(name):
    try:
        return mpl.colormaps[name]
    except:
        return mpl.cm.get_cmap(name)


def create_rose_plot(ax=None, *, vmin= None, vmax=None, log = False, cmap='turbo',
                     colorbar_show=True, colorbar_label=None, colorbar_fontsize=None,
                     xscale=1, yscale=1,
                     segment = None, rres=None,
                     **fig_kw):
    """
    Create a plot with a [rose projection](simple.plot.RoseAxes).

    The rose axes is a subclass of matplotlibs
    [polar axes](https://matplotlib.org/stable/api/projections/polar.html#matplotlib.projections.polar.PolarAxes).

    Args:
        ax (): If no preexisting axes is given then a new figure with a single rose axes is created. If an existing
        axes is passed this axes will be deleted and replaced with a [RoseAxes][#RoseAxes].
        vmin (float): The lower limit of the colour map. If no value is given the minimum value is ``0`` (or ``1E-10`` if
        ``log=True``)
        vmax (float): The upper limit of the colour map. If no value is given then ``vmax`` is set to ``1`` and all bin
            weights are divided by the heaviest bin weight in each histogram.
        log (bool): Whether the color map scale is logarithmic or not.
        cmap (): The name of the colormap to use. See,
                [matplotlib documentation][https://matplotlib.org/stable/users/explain/colors/colormaps.html]
                 for a list of available colormaps.
        colorbar_show (): Whether to add a colorbar to the right of the axes.
        colorbar_label (): The label given to the colorbar.
        colorbar_fontsize (): The fontsize of the colorbar label.
        xscale (): The scale of the x axis.
        yscale (): The scale of the y axis.
        segment (): Which segment of the rose diagram to show. Options are ``N``, ``E``, ``S``, ``W``, ``None``.
            If ``None`` the entire circle is shown.
        rres (): The resolution of lines drawn along the radius ``r``. The number of points in a line is calculated as
        ``r*rres+1`` (Min. 2).
        **fig_kw (): Additional figure keyword arguments passed to the ``pyplot.figure`` call. Only used when ``ax``
            is not given.

    Returns:
        RoseAxes : The new rose axes.
    """
    if ax is None:
        figure_kwargs = {'layout': 'constrained'}
        figure_kwargs.update(fig_kw)
        fig, ax = plt.subplots(subplot_kw={'projection': 'rose'}, **figure_kwargs)
    else:
        ax = get_axes(ax)
        fig = ax.get_figure()
        rows, cols, start, stop = ax.get_subplotspec().get_geometry()

        ax.remove()
        ax = fig.add_subplot(rows, cols, start + 1, projection='rose')


    ax.set_colorbar(vmin=vmin, vmax=vmax, log=log, cmap=cmap,
                    label=colorbar_label, show=colorbar_show, fontsize=colorbar_fontsize)
    ax.set_xyscale(xscale, yscale)

    if segment:
        ax.set_segment(segment)

    if rres:
        ax.set_rres(rres)

    return ax

class RoseAxes(mpl.projections.polar.PolarAxes):
    """
    A subclass of matplotlibs [Polar Axes](https://matplotlib.org/stable/api/projections/polar.html#matplotlib.projections.polar.PolarAxes).

    Rose plots can be created using the [create_rose_plot](simple.plot.create_rose_plot) function or by
    specifying the projection ``'rose'`` using matplotlib functions.

    Only custom and reimplemented methods are described here. See matplotlibs documentation for more methods. Note
    however, that these method might not behave as the reimplemented version below. For example the matplotlib methods
    will not take into account the ``xscale`` and ``yscale``.

    **Note** that some features, like axlines, might require an updated version of matplotlib to work.
    """
    name = 'rose'

    def __init__(self, *args, **kwargs):
        self._xysegment = None
        self._yscale = 1
        self._xscale = 1
        self._rres = 720

        self._vrel, self._vmin, self._vmax = True, 0, 1
        self._norm = mpl.colors.Normalize(vmin=self._vmin, vmax=self._vmax)
        self._cmap = get_cmap('turbo')
        self._colorrbar = None

        super().__init__(*args,
                         theta_offset=np.pi * 0.5, theta_direction=-1,
                         **kwargs)

    def clear(self):
        """
        Clear the axes.
        """
        super().clear()
        self.set_xysegment(self._xysegment)
        self.axes.set_yticklabels([])

    def set_colorbar(self, vmin=None, vmax=None, log=False, cmap='turbo',
                     label=None, fontsize=None, show=True, clear=True):
        """
        Define the colorbar used for histograms.

        Currently, there is no way to delete any existing colorbars. Thus, everytime this function is called a new
        colorbar is created. Therefore, It's advisable to only call this method once. Note that it is always called
        by the [create_rose_plot](simple.plot.create_rose_plot) function.

        Args:
            vmin (float): The lower limit of the colour map. If no value is given the minimum value is ``0`` (or ``1E-10`` if
                ``log=True``)
            vmax (float): The upper limit of the colour map. If no value is given then ``vmax`` is set to ``1`` and all bin
                weights are divided by the heaviest bin weight in each histogram.
            log (bool): Whether the color map scale is logarithmic or not.
            cmap (): The name of the colormap to use. See,
                [matplotlib documentation][https://matplotlib.org/stable/users/explain/colors/colormaps.html]
                for a list of available colormaps.
            label (): The label given to the colorbar.
            fontsize (): The fontsize of the colorbar label.
            show (): Whether to add a colorbar to the right of the axes.
            clear (): If ``True`` the axes will be cleared.
        """
        self._vrel = True if vmax is None else False
        if log:
            self._vmin, self._vmax = vmin or 1E-10, vmax or 1
            self._norm = mpl.colors.LogNorm(vmin=self._vmin, vmax=self._vmax)
        else:
            self._vmin, self._vmax = vmin or 0, vmax or 1
            self._norm = mpl.colors.Normalize(vmin=self._vmin, vmax=self._vmax)

        self._cmap = get_cmap(cmap)
        if show:
            self._colorbar = self.get_figure().colorbar(mpl.cm.ScalarMappable(norm=self._norm, cmap=self._cmap),
                                                        ax=self)
            self._colorbar.set_label(label, fontsize=fontsize)

        if clear:
            self.clear()

    def set_xyscale(self, xscale, yscale):
        """
        Set the scale of the *x* and *y* dimensions of the rose diagram.

        This can be used to distort the diagram to e.g. better show large or small slopes.

        **Note** Should not be confused with matplotlibs ``set_xscale`` and the ``set_yscale`` methods. They have
        are used to set the type of scale, e.g. log, linear etc., used for the different axis.

        Args:
            xscale (float): The scale of the *x* dimension of the rose diagram.
            yscale (float): The scale of the *y* dimension of the rose diagram.
        """
        # Not to be confused with set_xscale. This does something different.
        self._xscale = xscale
        self._yscale = yscale
        self.clear()

    def get_xyscale(self):
        """
        Return a tuple of the scale of the *x* and *y* dimensions of the rose diagram.
        """
        return (self._xscale, self._yscale)

    def set_xysegment(self, segment):
        """
        Define which segment of the rose diagram to show.

        Args:
            segment (): Options are ``N``, ``E``, ``S``, ``W``, ``None``.
                If ``None`` the entire circle is shown.
        """
        if segment is None:
            self.axes.set_thetagrids((0, 90, 180, 270),
                                     (self._yscale, self._xscale, self._yscale * -1, self._xscale * -1))
        elif type(segment) is not str:
            raise TypeError('segment must be a string')
        elif segment.upper() == 'N':
            self.axes.set_thetalim(-np.pi * 0.5, np.pi * 0.5)
            self.axes.set_thetagrids((-90, 0, 90), (-self._xscale, self._yscale, self._xscale))
        elif segment.upper() == 'S':
            self.axes.set_thetalim((np.pi * 0.5, np.pi * 1.5))
            self.axes.set_thetagrids((90, 180, 270), (-self._xscale, -self._yscale, self._xscale))
        elif segment.upper() == 'E':
            self.axes.set_thetalim(0, np.pi)
            self.axes.set_thetagrids((0, 90, 180), (self._yscale, self._xscale, -self._yscale))
        elif segment.upper() == 'W':
            self.axes.set_thetalim(np.pi, np.pi * 2)
            self.axes.set_thetagrids((180, 270, 360), (-self._yscale, -self._xscale, self._yscale))
        else:
            raise ValueError(f'Unknown segment: {segment}')
        self._xysegment = segment

    def _xy2rad(self, x, y):
        """
        Convert x, y coordinated to a theta value in relation to the *x* and *y* dimensions of the diagram.
        """
        return xy2rad(x, y, self._xscale, self._yscale)

    def set_rres(self, rres):
        """
        Set the resolution of lines drawn along the radius ``r``. The number of points in a line is calculated as
        ``r*rres+1`` (Min. 2).
        """
        self._rres = rres

    def get_rres(self):
        """
        Return the resolution of lines drawn along the radius ``r``.
        """
        return self._rres

    def _rplot(self, theta1, theta2, r, **kwargs):
        # Plot lines along the radius
        theta1, theta2, r = as0darray(theta1, theta2, r)

        if theta2 < theta1: theta1 -= np.pi * 2
        diff = (theta2 - theta1) / (np.pi * 2)
        nr = int(np.max([1, diff * self._rres * r])) + 1

        self.axes.plot(np.linspace(theta1, theta2, nr), np.full(nr, r), **kwargs)

    def _rfill(self, theta1, theta2, r1, r2, **kwargs):
        # Create a shaded segment.
        theta1, theta2, r1, r2 = as0darray(theta1, theta2, r1, r2)

        if theta2 < theta1: theta1 -= np.pi * 2
        diff = (theta2 - theta1) / (np.pi * 2)

        nr1 = int(np.max([1, diff * self._rres * r1])) + 1
        nr2 = int(np.max([1, diff * self._rres * r2])) + 1
        theta = np.append(np.linspace(theta1, theta2, nr1),
                          np.linspace(theta2, theta1, nr2))
        r = np.append(np.full(nr1, r1),
                      np.full(nr2, r2))

        self.fill(theta, r, **kwargs)

    #####################
    ### Point methods ###
    #####################
    def merrorbar(self, m, r=1, merr=None,
                  antipodal=None,
                  **kwargs):
        """
        Plot data points with errorbars.

        This is an adapted version of matplotlibs
        [errorbar](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.errorbar.html) method.

        **Note** it is currently not possible to add bar ends to the error bars.

        Args:
            m (float, (float, float)): Either a single array of floats representing a slope or a tuple of *x* and *y*
                coordinates from which a slope will be calculated.
            r (): The radius at which the data points will be drawn.
            merr (): The uncertainty of the slope.
            antipodal (): Whether the antipodal data points will be drawn. By default, ``antipodal=True`` when ``m`` is
                a slope and ``antipodal=False`` when ``m`` is *x,y* coordinates.
            **kwargs (): Additional keyword arguments passed to matplotlibs
            [errorbar](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.errorbar.html) method.
        """

        kwargs.setdefault('linestyle', '')
        kwargs.setdefault('marker', 'o')
        if type(m) is tuple and len(m) == 2:
            x, y = m
            if antipodal is None:
                antipodal = False
        else:
            x, y = 1, m
            if antipodal is None:
                antipodal = True

        x, y, r, merr = as1darray(x, y, r, merr)

        if merr is not None:
            # Makes errorbar show up in legend
            yerr = np.nan
        else:
            yerr = None

        kwargs['capsize'] = 0  # Because we cannot show these
        with warnings.catch_warnings():
            # Supresses warning that comes from having yerr as nan
            warnings.filterwarnings('ignore', message='All-NaN axis encountered', category=RuntimeWarning)
            data_line, caplines, barlinecols = self.errorbar(self._xy2rad(x, y), r, yerr=yerr, **kwargs)

        # print(len(barlinecols), barlinecols, barlinecols[0].get_colors(), barlinecols[0].get_linewidth())
        if merr is not None:
            colors = barlinecols[0].get_colors()
            if len(colors) == 1: colors = [colors[0]] * x.size

            linestyles = barlinecols[0].get_linestyles()
            if len(linestyles) == 1: linestyles = [linestyles[0]] * x.size

            linewidths = barlinecols[0].get_linewidths()
            if len(linewidths) == 1: linewidths = [linewidths[0]] * x.size

            zorder = barlinecols[0].get_zorder()

            for i in range(x.size):
                m = y[i] / x[i]
                self._rplot(self._xy2rad(x[i], (m + merr[i]) * x[i]), self._xy2rad(x[i], (m - merr[i]) * x[i]), r[i],
                            color=colors[i], linestyle=linestyles[i], linewidth=linewidths[i],
                            zorder=zorder, marker="")
        if antipodal:
            kwargs.pop('label', None)
            kwargs.pop('color', None)
            self.merrorbar((x * -1, y * -1), r, merr=merr, antipodal=False, color=data_line.get_color(), **kwargs)

    ####################
    ### Line methods ###
    ####################
    def _mline(self, m, r=1, merr=None,
               ecolor=None, elinestyle=":", elinewidth=None, ezorder=None,
               ealpha=0.5, efill=False, eline=True, axline=False,
               antipodal=None, **kwargs):
        # Does the heavy lifting for the axmline and mline methods.

        if type(m) is tuple and len(m) == 2:
            x, y = m
            if antipodal is None:
                antipodal = False
        else:
            x, y = 1, m
            if antipodal is None:
                antipodal = True

        if type(r) is tuple and len(r) == 2:
            rmin, rmax = r
        else:
            rmin, rmax = 0, r

        theta = self._xy2rad(x, y)[0]
        if axline:
            line = self.axvline(theta, rmin, rmax, **kwargs)
        else:
            line = self.plot([theta, theta], [rmin, rmax], **kwargs)[0]

        if merr is not None:
            if type(line) is list: line = line[0]
            ezorder = ezorder or line.get_zorder() - 0.001
            ecolor = ecolor or line.get_color()
            elinewidth = elinewidth or line.get_linewidth()

            m = y / x
            lowerlim = self._xy2rad(x, (m - merr) * x)
            upperlim = self._xy2rad(x, (m + merr) * x)

            # Do this first so it's beneath the lines.
            if efill:
                self._rfill(upperlim, lowerlim, rmin, rmax, color=ecolor, alpha=ealpha, zorder=ezorder)

            if eline:
                if axline:
                    self.axvline(lowerlim, rmin, rmax,
                                 color=ecolor, linestyle=elinestyle, linewidth=elinewidth, zorder=ezorder)
                    self.axvline(upperlim, rmin, rmax,
                                 color=ecolor, linestyle=elinestyle, linewidth=elinewidth, zorder=ezorder)
                else:
                    self.plot([lowerlim, lowerlim], [rmin, rmax],
                              color=ecolor, linestyle=elinestyle, linewidth=elinewidth, zorder=ezorder)
                    self.plot([upperlim, upperlim], [rmin, rmax],
                              color=ecolor, linestyle=elinestyle, linewidth=elinewidth, zorder=ezorder)

        if antipodal:
            kwargs.pop('label', None)
            kwargs['color'] = line.get_color()
            self._mline(m=(x * -1, y * -1), r=(rmin, rmax), merr=merr,
                        ecolor=ecolor, elinestyle=elinestyle, elinewidth=elinewidth,
                        ezorder=ezorder, ealpha=ealpha, axline=axline, antipodal=False, **kwargs)

    def mline(self, m, r=1, merr=None, antipodal=None, *, eline=True, efill=False,
              ecolor=None, elinestyle=":", elinewidth=None, ezorder=None, ealpha=0.1,
              **kwargs):
        """
        Draw a line along a slope.

        Used matplotlibs [plot](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html) method to
        draw the line(s).

        Args:
            m (float, (float, float)): Either a single slope or a tuple of *x* and *y*
                coordinates from which a slope will be calculated.
            r (float, (float, float)): If a single value is given a line will be drawn between the ``0`` and ``r``.
                if a tuple of two values are given the line will be drawn between ``r[0]`` and ``r[1]``. Note these
                are absolute coordinates.
            merr (): The uncertainty of the slope.
            eline (): If ``True`` lines will also be drawn for the uncertainty of the slope.
            efill (): If ``True`` the area defined by the uncertainty of the slope will be shaded.
            ecolor (): The color used for the uncertainty lines and/or the shaded area.
            elinestyle (): The line style used for the uncertainty lines.
            elinewidth (): The line width used for the uncertainty lines.
            ezorder (): The z order width used for the uncertainty lines.
            ealpha (): The alpha value for the shaded area.
            antipodal (): Whether the antipodal data points will be drawn. By default, ``antipodal=True`` when ``m`` is
                a slope and ``antipodal=False`` when ``m`` is *x,y* coordinates.
            **kwargs (): Additional keyword arguments passed to matplotlibs
                [plot](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html) method.
        """

        kwargs.setdefault('linestyle', '-')
        return self._mline(m, r, merr,
                           ecolor=ecolor, elinestyle=elinestyle, elinewidth=elinewidth,
                           ezorder=ezorder, ealpha=ealpha, efill=efill, eline=eline,
                           antipodal=antipodal, axline=False, **kwargs)

    def axmline(self, m, r=1, merr=None, eline=True,
                ecolor=None, elinestyle=":", elinewidth=None, ezorder=None,
                antipodal=None, **kwargs):
        """
        Draw a line along a slope.

        Used matplotlibs [axvline](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.axvline.html) method
        to draw the line(s).

        Args:
            m (float, (float, float)): Either a single slope or a tuple of *x* and *y*
                coordinates from which a slope will be calculated.
            r (float, (float, float)): If a single value is given a line will be drawn between the ``0`` and ``r``.
                if a tuple of two values are given the line will be drawn between ``r[0]`` and ``r[1]``. Note these
                are relative coordinates.
            merr (): The uncertainty of the slope.
            eline (): If ``True`` lines will also be drawn for the uncertainty of the slope.
            ecolor (): The color used for the uncertainty lines.
            elinestyle (): The line style used for the uncertainty lines.
            elinewidth (): The line width used for the uncertainty lines.
            ezorder (): The z order width used for the uncertainty lines.
            antipodal (): Whether the antipodal data points will be drawn. By default, ``antipodal=True`` when ``m`` is
                a slope and ``antipodal=False`` when ``m`` is *x,y* coordinates.
            **kwargs (): Additional keyword arguments passed to matplotlibs
                [axvline](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.axvline.html) method.
        """

        return self._mline(m, r, merr,
                           ecolor=ecolor, elinestyle=elinestyle, elinewidth=elinewidth,
                           ezorder=ezorder, ealpha=0, eline=eline, efill=False,
                           antipodal=antipodal, axline=True, **kwargs)

    def axrline(self, r, tmin=0, tmax=1, **kwargs):
        """
        Plot a line along a given radius.

        Used matplotlibs [axhline](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.axhline.html) method
        to draw the line.

        Args:
            r (): The radius at which to draw the line.
            tmin (): The starting angle of the line. In relative coordinates.
            tmax (): The stopping angle of the line. In relative coordinates.
            **kwargs (): Additional keyword arguments passed to matplotlibs
                [axhline](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.axhline.html) method.

        Returns:

        """
        return self.axhline(r, tmin, tmax, **kwargs)

    ####################
    ### Hist Methods ###
    ####################
    def mhist(self, m, r=1, weights=1, rwidth=0.9, rscale=True, rescale=True, antipodal=None,
              bins=72, label=None, label_pos=0, label_roffset=None, label_kwargs={}):
        """
        Create a histogram of the given slopes.

        Args:
            m (float, (float, float)): Either a single array of floats representing a slope or a tuple of *x* and *y*
                coordinates from which a slope will be calculated.
            r (): The radius at which the histogram will be drawn.
            weights (): The weight assigned to each slope.
            rwidth (): The width of the histogram.
            rscale (): If ``True`` width of the individual bins will be scaled to their weight. Otherwise all bins
                will have the same width.
            rescale (): If ``True`` all bin widths will be scaled relative to the heaviest bin. Otherwise, they will
                be scaled to the color map.
            antipodal (): Whether the antipodal data points will be included in the histogram. By default,
            ``antipodal=True`` when ``m`` is a slope and ``antipodal=False`` when ``m`` is *x,y* coordinates.
            bins (): The number of even sized bin in the histogram.
            label (): A label for the histogram.
            label_pos (): The position of the label in the histogram. In degrees.
            label_roffset (): The offset from ``r`` where the label will be shown. By default, ``rwidth/2``.
            label_kwargs (): A dictionary with additional keyword arguments passed to matplotlibs
                [annotate](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Annotate.html) method which is
                used to draw the label.
        """
        if type(m) is tuple and len(m) == 2:
            x, y = m
            if antipodal is None:
                antipodal = False
        else:
            x, y = 1, m
            if antipodal is None:
                antipodal = True

        x, y, weights = as1darray(x, y, weights)
        theta = self._xy2rad(x, y)
        if antipodal:
            theta = np.append(theta, self._xy2rad(x * -1, y * -1))
            weights = np.append(weights, weights)

        bin_weights, bin_edges = np.histogram(theta, bins=bins, range=(0, np.pi * 2), weights=weights, density=False)

        bin_weights = bin_weights / (np.max(bin_weights) if self._vrel else self._vmax)
        bin_colors = np.array([self._cmap(self._norm(bw)) for bw in bin_weights])

        if rscale:
            bin_widths = np.array([self._norm(bw, clip=True) for bw in bin_weights])
            if rescale: bin_widths = bin_widths / np.max(bin_widths)
            bin_widths = bin_widths * rwidth
        else:
            bin_widths = np.full(bin_weights.size, rwidth)

        for i in range(bin_weights.size):
            self._rfill(bin_edges[i], bin_edges[i + 1], r, r + bin_widths[i],
                        alpha=1, facecolor=bin_colors[i])

        if label:
            label_pos = label_pos % 360

            if label_roffset is None:
                label_roffset = rwidth / 2

            label_rot = label_pos * -1
            if label_pos > 90 and label_pos < 270:
                label_rot = (label_rot + 180) % 360

            label_kw = {'text': label, 'xy': (deg2rad(label_pos), r + label_roffset),
                             'ha': 'center', 'va': 'center', 'rotation': label_rot}
            label_kw.update(label_kwargs)
            self.annotate(**label_kw)


mpl.projections.register_projection(RoseAxes)

