import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.ticker import AutoMinorLocator
import functools

import numpy as np
import warnings

import simple.models
from simple import utils

import logging
logger = logging.getLogger('SIMPLE.plot')

__all__ = ['create_rose_plot',
           'mhist_abundance', 'mhist_intnorm', 'mhist_simplenorm',
           'plot_intnorm', 'plot_simplenorm', 'plot_abundance']


# colours appropriate for colour blindness
# Taken from https://davidmathlogic.com/colorblind/#%23000000-%23E69F00-%2356B4E9-%23009E73-%23F0E442-%230072B2-%23D55E00-%23CC79A7
all_colors=utils.EndlessList(["#D55E00", "#56B4E9", "#009E73", "#E69F00", "#CC79A7", "#0072B2", "#F0E442"])
"""
[``Endlesslist``][simple.plot.EndlessList] containing the default colors used by simple plotting functions.
"""

all_linestyles = utils.EndlessList(['-', (0, (4, 4)), (0, (2,1)),
                              (0, (4,2,1,2)), (0, (4,2,1,1,1,2)), (0, (4,2,1,1,1,1,1,2)),
                              (0, (2,1,2,2,1,2)), (0, (2,1,2,2,1,1,1,2)), (0, (2,1,2,2,1,1,1,1,1,2)),
                              (0, (2,1,2,1,2,2,1,2)), (0, (2,1,2,1,2,2,1,1,1,2)), (0, (2,1,2,1,2,2,1,1,1,1,1,2))])
"""
[``Endlesslist``][simple.plot.EndlessList] default line styles used by simple plotting functions.
"""

all_markers = utils.EndlessList(["o", "s", "^", "D", "P","X", "v", "<", ">",  "*", "p", "d", "H"])
"""
[``Endlesslist``][simple.plot.EndlessList] default marker styles used by simple plotting functions.
"""

def get_axes(axes, projection=None):
    """
    Return the ax that should be used for plotting.

    Args:
        axes (): Must either be ``None``, in which case ``plt.gca()`` will be returned, a matplotlib ax instance, or
            any object that has a ``.gca()`` method.
        projection (): If given, an exception is raised if ``ax`` does not have this projection.
    """
    if axes is None:
        axes = plt.gca()
    elif isinstance(axes, Axes):
        pass
    elif hasattr(axes, 'gca'):
        axes = axes.gca() # if ax = plt
    else:
        raise ValueError('ax must be an Axes, Axes instance or have a gca() method that return an Axes')

    if projection is not None and axes.name != projection:
        raise TypeError(f'The selected ax has the wrong projection. Expected {projection} got {axes.name}.')
        
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
        colors = utils.EndlessList(["#000000"])
    elif color is True:
        colors = all_colors
    else:
        colors = utils.EndlessList(color)

    if linestyle is False or linestyle is None:
        linestyles = utils.EndlessList([""])
    elif linestyle is True:
        linestyles = all_linestyles
    else:
        linestyles = utils.EndlessList(linestyle)

    if marker is False or marker is None:
        markers = utils.EndlessList([""])
    elif marker is True:
        markers = all_markers
    else:
        markers = utils.EndlessList(marker)

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

def update_axes(ax, kwargs, *delay, delay_all=False):
    """
    Updates the axes and figure objects.

    Keywords beginning with ``ax_<name>`` and ``fig_<name>`` will be stripped from kwargs. These will then be used
    to call the ``set_<name>`` or ``<name>`` method of the axes or figure object.

    If the value mapped to ``ax_<name>`` or ``fig_<name>`` is:
    - A boolean it is used to determine whether to call the method. The boolean itself will not be passed to
        the method. To pass a boolean to a method place it in a tuple, e.g. ``(True, )``.
    - A tuple then the contents of the tuple is unpacked as arguments for the method call.
    - A dictionary then the contents of the dictionary is unpacked as keyword arguments for the method call.
    - Any other value will be passed as the first argument to the method call.

    Additional keyword arguments can be passed to methods by mapping ``ax_kw_<name>_<keyword>`` or
    ``fig_kw_<name>_<keyword>`` kwargs to the value. These additional keyword arguments are only used if the
    ``ax_<name>`` or ``fig_<name>`` kwargs exist. Note however that they are always stripped from ``kwargs``.

    It is possible to delay calling certain method adding ``ax_<name>`` or ``fig_<name>`` to ``*delay``. Keywords
    associated with these method will then be included in the returned dictionary. This dictionary can be passed back
    to the function at a later time. To delay all calls, i.e. just strip out all the relevant kwargs use
    ``delay_all=True``.

    Returns
        dict: A dictionary containing the delayed method calls.
    """
    axes_meth = utils.extract_kwargs(kwargs, prefix='ax')
    axes_kw = utils.extract_kwargs(axes_meth, prefix='kw')

    figure_meth = utils.extract_kwargs(kwargs, prefix='fig')
    figure_kw = utils.extract_kwargs(figure_meth, prefix='kw')

    # Special cases
    if 'size' in figure_meth: figure_meth.setdefault('size_inches', figure_meth.pop('size'))

    delayed_kwargs = {}

    def update(obj, name, meth_kwargs, kw_kwargs):
        for var, arg in meth_kwargs.items():
            var_kwargs = utils.extract_kwargs(kw_kwargs, prefix=var)
            try:
                method = getattr(obj, f'set_{var}')
            except:
                try:
                    method = getattr(obj, var)
                except:
                    raise AttributeError(f'The {name} object has no method called ``set_{var}`` or ``{var}``')

            if f'{name}_{var}' in delay or delay_all:
                delayed_kwargs[f'{name}_{var}'] = arg
                delayed_kwargs.update({f'{name}_kw_{var}_{k}': v for k,v in var_kwargs.items()})
                continue

            elif arg is False:
                continue
            elif arg is True:
                arg = ()
            elif type(arg) is dict:
                var_kwargs.update(arg)
                arg = ()

            method(*(arg if type(arg) is tuple else (arg,)), **var_kwargs)

    update(ax, 'ax', axes_meth, axes_kw)
    update(ax.get_figure(), 'fig', figure_meth, figure_kw)

    return delayed_kwargs


def create_rose_plot(ax=None, *, vmin= None, vmax=None, log = False, cmap='turbo',
                     colorbar_show=True, colorbar_label=None, colorbar_fontsize=None,
                     xscale=1, yscale=1,
                     segment = None, rres=None,
                     **fig_kw):
    """
    Create a plot with a [rose projection](simple.plot.RoseAxes).

    The rose ax is a subclass of matplotlibs
    [polar ax](https://matplotlib.org/stable/api/projections/polar.html#matplotlib.projections.polar.PolarAxes).

    Args:
        ax (): If no preexisting ax is given then a new figure with a single rose ax is created. If an existing
        ax is passed this ax will be deleted and replaced with a [RoseAxes][#RoseAxes].
        vmin (float): The lower limit of the colour map. If no value is given the minimum value is ``0`` (or ``1E-10`` if
        ``log=True``)
        vmax (float): The upper limit of the colour map. If no value is given then ``vmax`` is set to ``1`` and all bin
            default_weight are divided by the heaviest bin weight in each histogram.
        log (bool): Whether the color map scale is logarithmic or not.
        cmap (): The prefixes of the colormap to use. See,
                [matplotlib documentation][https://matplotlib.org/stable/users/explain/colors/colormaps.html]
                 for a list of available colormaps.
        colorbar_show (): Whether to add a colorbar to the right of the ax.
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
        RoseAxes : The new rose ax.
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
        self._last_hist_r = 0

        super().__init__(*args,
                         theta_offset=np.pi * 0.5, theta_direction=-1,
                         **kwargs)

    def clear(self):
        """
        Clear the ax.
        """
        super().clear()
        self.set_xysegment(self._xysegment)
        self.axes.set_yticklabels([])
        self._last_hist_r = 0

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
                default_weight are divided by the heaviest bin weight in each histogram.
            log (bool): Whether the color map scale is logarithmic or not.
            cmap (): The prefixes of the colormap to use. See,
                [matplotlib documentation][https://matplotlib.org/stable/users/explain/colors/colormaps.html]
                for a list of available colormaps.
            label (): The label given to the colorbar.
            fontsize (): The fontsize of the colorbar label.
            show (): Whether to add a colorbar to the right of the ax.
            clear (): If ``True`` the ax will be cleared.
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
    def mhist(self, m, r=None, weights=1, rwidth=0.9, rscale=True, rescale=True, antipodal=None,
              bins=72, label=None, label_pos=0, label_roffset=None, label_kwargs={}):
        """
        Create a histogram of the given slopes.

        Args:
            m (float, (float, float)): Either a single array of floats representing a slope or a tuple of *x* and *y*
                coordinates from which a slope will be calculated.
            r (): The radius at which the histogram will be drawn. If 'None' it will be plotted ``1`` above the
                previous histogram, or at 1 if no histogram have been drawn.
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

        if r is None:
            r = self._last_hist_r + 1
        self._last_hist_r = r

        x, y, weights = as1darray(x, y, weights)
        theta = self._xy2rad(x, y)
        if antipodal:
            theta = np.append(theta, self._xy2rad(x * -1, y * -1))
            weights = np.append(weights, weights)

        bin_weights, bin_edges = np.histogram(theta, bins=bins, range=(0, np.pi * 2), weights=weights, density=False)
        if self._vrel: bin_weights = bin_weights / np.max(bin_weights)
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

#################################
### helper plotting functions ###
#################################
class GetterTemplate:
    """
    Helper object for extracting data from models. To be used together with the helper plotting functions.
    """
    def __init__(self):
        # This caches the data of the last used model only.
        # This approach shouldn't cause any memory leaks
        self._get_data_ = functools.lru_cache(maxsize=1)(self._get_data_)

    def get_data(self, model, key=None):
        """
        Extract the data from model.

        Args:
            model (): Model from which to extract the desired data
            key (): The data column ``key`` to be extracted from the predefined attribute. If ``None`` the predefined
                attribute is used.
        """
        raise NotImplementedError()

    def _get_data_(self, model):
        raise NotImplementedError()

    def get_label(self, model, key=None):
        raise NotImplementedError()

    def get_weights(self, model, weights=None, desired_unit=None):
        raise NotImplementedError()

class DataGetter(GetterTemplate):
    def __init__(self, attrname, key = None, desired_unit=None, default_weight=1):
        super().__init__()
        self.attrname = attrname
        self.key = key
        self.desired_unit = desired_unit
        self.default_weight = default_weight

    def get_data(self, model, key=None):
        data, data_unit = self._get_data_(model)

        if key is None: key = self.key

        try:
            key = utils.asisotope(key)
        except ValueError:
            pass
        else:
            return data[key]

        try:
            key = utils.asratio(key)
        except ValueError:
            raise ValueError(f'Unable to parse {key} into an isotope or ratio string')
        else:
            return (data[key.numer] / data[key.denom])

    def _get_data_(self, model, desired_unit=None):
        desired_unit = desired_unit or self.desired_unit

        data = getattr(model, self.attrname)
        data_unit = getattr(model, f"{self.attrname}_unit", None)

        data = model.convert_keyarray(data, data_unit, desired_unit)
        return data, self.desired_unit or data_unit

    def get_label(self, model, key=None, prefix = '',
                  key_in_label= True, numer_in_label=True, denom_in_label=True,
                  model_in_label=False, unit_in_label=True):
        if key is None: key = self.key

        label = prefix
        if key_in_label:
            try:
                key = utils.asisotope(key)
            except ValueError:
                try:
                    key = utils.asratio(key)
                except ValueError:
                    raise ValueError(f'Unable to parse {key} into an isotope or ratio string')
                else:
                    if numer_in_label and denom_in_label:
                        label += f'{self._get_label_(model, key.numer)}/{self._get_label_(model, key.denom)} '
                    elif numer_in_label:
                        label += self._get_label_(model, key.numer) + ' '
                    elif denom_in_label:
                        label += self._get_label_(model, key.denom) + ' '
            else:
                label += self._get_label_(model, key) + ' '

            if unit_in_label:
                _, data_unit = self._get_data_(model)

                if data_unit:
                    label += f'[{data_unit}] '

        if model_in_label:
            return (label + model.name).strip()
        else:
            return label.strip()

    def _get_label_(self, model, isotope):
        return rf'${{}}^{{{isotope.mass}}}\mathrm{{{isotope.element}}}$'

    def get_weights(self, model, weights=None):
        if weights is None: weights = self.default_weight

        if type(weights) is str:
            return self._get_weight_(model, weights)
        else:
            return weights

        data = getattr(model, self.attrname)
        data_unit = getattr(model, f"{self.attrname}_unit", None)

        data = model.convert_keyarray(data, data_unit)

        return self._get_weight_(data, weights)

    def _get_weights_(self, model, weights):
        data, unit = self._get_data_(model, self.desired_unit)
        return self._calc_weight_(data, weights)

    def _calc_weight_(self, data, string):
        try:
            isotopes = utils.asisotopes(string)
        except ValueError:
            try:
                element = utils.aselement(string)
            except ValueError:
                raise ValueError(f'Unable to parse "{string}" into a list of isotope string or an element string')
            else:
                isotopes = utils.get_isotopes_of_element(data.dtype.names, element)

        weights = 0
        for iso in isotopes:
            if iso in data.dtype.names:
                weights += data[iso]

        return weights

class NormGetter(DataGetter):
    def __init__(self, attrname, Rvalname, key=None,
                 default_weight=1, weights_attrname=None, weights_desired_unit=None):
        super().__init__(attrname, key, default_weight=default_weight)
        self.Rvalname = Rvalname
        self.weights_attrname = weights_attrname
        self.weights_desired_unit = weights_desired_unit

    def _get_data_(self, model):
        norm = getattr(model, self.attrname)
        data = getattr(norm, self.Rvalname)

        return data, None

    def _get_label_(self, model, isotope=None):
        return getattr(model, self.attrname).label_latex[isotope]

    def _get_weights_(self, model, weights):
        if self.weights_attrname is None:
            data, data_unit = self._get_data_(model, self.weights_desired_unit)
        else:
            data = model.get_keyarray(self.weights_attrname, self.weights_desired_unit)

        return self._calc_weight_(data, weights)


################
### xy plots ###
################
@utils.set_default_kwargs()
def plot_abundance(models, xkey, ykey, *,
                 attrname = 'abundance', unit=None,
                 mask = None, ax = None, where=None, where_kwargs={},
                 **kwargs):
    """
    Plots *xkey* against *ykey* from a data array.

    Args:
        models (): The collection of models to be plotted.
        xkey (): Can either an isotope or a ratio of two isotopes.
        ykey (): Can either an isotope or a ratio of two isotopes.
        attrname (): The name of the attribute storing the data array.
        unit (str): The unit the data should be plotted in. If the data is stored in a different unit an attempt
            to convert the data to *unit* is made before plotting.
        ax (Axes): Axes on which to plot the data.
        where (): A string to select which models to plot. See
            [``ModelCollection.where``](simple.models.ModelCollection.where) for more details.
        where_kwargs (): Keyword arguments to go with *where*.
        kwargs: See section below for a description of acceptable keywords.

    Keyword Arguments:
        - ``model_in_legend`` Whether to add the model name to the legend label. If ``None`` the model name is
        only added if more than one model is being plotted.

        - ``linestyle`` Can be a list of linestyles that will be iterated through for each item plotted. If ``True``
        the default list of linestyles is used.
        - ``color`` Can be a list of colors that will be iterated through for each item plotted. If ``True``
        the default list of colors is used.
        - ``marker`` Can be a list of markers that will be iterated through for each item plotted. If ``True``
        the default list of markers is used.
        - Any keyword argument accepted by matplotlibs
        [``plot``](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html) method.

        - Keywords in the style of ``ax_<name>`` and ``fig_<name>`` can be used to update the axes and figure object.
        Additional keyword arguments for these method calls can be set using ``ax_kw_<name>_<keyword>`` and
        ``fig_kw_<name>_<keyword>``. See [here](simple.plot.update_axes) for more details.

    """
    xygetter = DataGetter(attrname, desired_unit=unit)
    ax, model = template_plot_y(models, xygetter, xygetter, xkey, ykey,
                                mask=mask, ax=ax, where=where, where_kwargs=where_kwargs,
                                **kwargs)
    return ax
@utils.set_default_kwargs()
def plot_simplenorm(models, xkey, ykey, *,
                 attrname = 'simplenorm',
                 mask = None, ax = None, where=None, where_kwargs={},
                 **kwargs):
    """
    Plots *xkey* against *ykey* from the simply normalised Ri compositions.

    Args:
        models (): The collection of models to be plotted.
        xkey (): Can either an isotope or a ratio of two isotopes.
        ykey (): Can either an isotope or a ratio of two isotopes.
        attrname (): The name of the attribute storing the internally normalised data.
        ax (Axes): Axes on which to plot the data.
        where (): A string to select which models to plot. See
            [``ModelCollection.where``](simple.models.ModelCollection.where) for more details.
        where_kwargs (): Keyword arguments to go with *where*.
        kwargs: See section below for a description of acceptable keywords.

    Keyword Arguments:
        - ``model_in_legend`` Whether to add the model name to the legend label. If ``None`` the model name is
        only added if more than one model is being plotted.

        - ``linestyle`` Can be a list of linestyles that will be iterated through for each item plotted. If ``True``
        the default list of linestyles is used.
        - ``color`` Can be a list of colors that will be iterated through for each item plotted. If ``True``
        the default list of colors is used.
        - ``marker`` Can be a list of markers that will be iterated through for each item plotted. If ``True``
        the default list of markers is used.
        - Any keyword argument accepted by matplotlibs
        [``plot``](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html) method.

        - Keywords in the style of ``ax_<name>`` and ``fig_<name>`` can be used to update the axes and figure object.
        Additional keyword arguments for these method calls can be set using ``ax_kw_<name>_<keyword>`` and
        ``fig_kw_<name>_<keyword>``. See [here](simple.plot.update_axes) for more details.

    """
    xygetter = NormGetter(attrname, 'eRi')
    ax, model = template_plot_y(models, xygetter, xygetter, xkey, ykey,
                                mask=mask, ax=ax, where=where, where_kwargs=where_kwargs,
                                **kwargs)
    return ax

@utils.set_default_kwargs()
def plot_intnorm(models, xkey, ykey, *,
                 attrname = 'intnorm',
                 mask = None, ax = None, where=None, where_kwargs={},
                 **kwargs):
    """
    Plots *xkey* against *ykey* from the internally normalised eRi compositions.

    Args:
        models (): The collection of models to be plotted.
        xkey (): Can either an isotope or a ratio of two isotopes.
        ykey (): Can either an isotope or a ratio of two isotopes.
        attrname (): The name of the attribute storing the internally normalised data.
        ax (Axes): Axes on which to plot the data.
        where (): Can be used to select only a subset of the models to plot.
        where (): A string to select which models to plot. See
            [``ModelCollection.where``](simple.models.ModelCollection.where) for more details.
        kwargs: See section below for a description of acceptable keywords.

    Keyword Arguments:
        - ``model_in_legend`` Whether to add the model name to the legend label. If ``None`` the model name is
        only added if more than one model is being plotted.

        - ``linestyle`` Can be a list of linestyles that will be iterated through for each item plotted. If ``True``
        the default list of linestyles is used.
        - ``color`` Can be a list of colors that will be iterated through for each item plotted. If ``True``
        the default list of colors is used.
        - ``marker`` Can be a list of markers that will be iterated through for each item plotted. If ``True``
        the default list of markers is used.
        - Any keyword argument accepted by matplotlibs
        [``plot``](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html) method.

        - Keywords in the style of ``ax_<name>`` and ``fig_<name>`` can be used to update the axes and figure object.
        Additional keyword arguments for these method calls can be set using ``ax_kw_<name>_<keyword>`` and
        ``fig_kw_<name>_<keyword>``. See [here](simple.plot.update_axes) for more details.

    """
    xygetter = NormGetter(attrname, 'eRi')
    ax, model = template_plot_y(models, xygetter, xygetter, xkey, ykey,
                                mask=mask, ax=ax, where=where, where_kwargs=where_kwargs,
                                **kwargs)
    return ax

##################
### rose plots ###
##################

@utils.set_default_kwargs()
def mhist_intnorm(models, xisotope, yisotope, *,
                 attrname='intnorm',
                 weights = 1, weights_attrname='abundance', weights_desired_unit='mole',
                 mask = None, ax = None, where=None, where_kwargs={},
                 **kwargs):

    xygetter = NormGetter(attrname, 'eRi',
                          default_weight=weights, weights_attrname=weights_attrname, weights_desired_unit=weights_desired_unit)

    ax, model = template_mhist_xy(models, xygetter, xisotope, yisotope,
                                  mask=mask, ax=ax, where=where, where_kwargs=where_kwargs,
                                  **kwargs)
    return ax

@utils.set_default_kwargs()
def mhist_simplenorm(models, xisotope, yisotope, *,
                 attrname='simplenorm',
                 weights = 1, weights_attrname='abundance', weights_desired_unit='mole',
                 mask = None, ax = None, where=None, where_kwargs={},
                 **kwargs):

    xygetter = NormGetter(attrname, 'eRi',
                          default_weight=weights, weights_attrname=weights_attrname, weights_desired_unit=weights_desired_unit)

    ax, model = template_mhist_xy(models, xygetter, xisotope, yisotope,
                                  mask=mask, ax=ax, where=where, where_kwargs=where_kwargs,
                                  **kwargs)
    return ax

@utils.set_default_kwargs()
def mhist_abundance(models, xisotope, yisotope, *,
                    attrname='intnorm', unit=None, default_weight=1,
                    mask = None, ax = None, where=None, where_kwargs={},
                    **kwargs):

    xygetter = DataGetter(attrname, desired_unit=unit, default_weight=default_weight)

    #numerators = utils.get_isotopes_of_element(abu.dtype.names, rat.denom.element, rat.denom.suffix)


    ax, model = template_mhist_xy(models, xygetter, xisotope, yisotope,
                                  mask=mask, ax=ax, where=where, where_kwargs=where_kwargs,
                                  **kwargs)
    return ax

###############################
### Template plot functions ###
###############################
# These aid in plotting the same thing for different models.
@utils.set_default_kwargs(
    linestyle=True, color=True, marker=False,
    ax_kw_xlabel_fontsize=15,
    ax_kw_ylabel_fontsize=15,
    markersize=4,
    ax_legend=dict(loc='upper right'),
    model_in_legend=None,
    ax_tick_params=dict(axis='both', left=True, right=True, top=True)
    )

def template_plot_xy(models, xgetter, ygetter, xkey, ykey,
                     mask = None, ax = None, where=None, where_kwargs={},
                     **kwargs):
    """
    Template function for a xy plot.

    Args:
        models (): The collection of models to be plotted.
        xgetter (): Getter object for retrieving the *xkey* data from each model.
        ygetter (): Getter object for retrieving the *ykey* data from each model.
        xkey (): Can either an isotope or a ratio of two isotopes.
        ykey (): Can either an isotope or a ratio of two isotopes.
        ax (Axes): Axes on which to plot the data.
        where (): A string to select which models to plot. See
            [``ModelCollection.where``](simple.models.ModelCollection.where) for more details.
        where_kwargs (): Keyword arguments to go with *where*.
        kwargs: See section below for a description of acceptable keywords.

    Keyword Arguments:
        - ``model_in_legend`` Whether to add the model name to the legend label. If ``None`` the model name is
        only added if more than one model is being plotted.

        - ``linestyle`` Can be a list of linestyles that will be iterated through for each item plotted. If ``True``
        the default list of linestyles is used.
        - ``color`` Can be a list of colors that will be iterated through for each item plotted. If ``True``
        the default list of colors is used.
        - ``marker`` Can be a list of markers that will be iterated through for each item plotted. If ``True``
        the default list of markers is used.
        - Any keyword argument accepted by matplotlibs
        [``plot``](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html) method.

        - Keywords in the style of ``ax_<name>`` and ``fig_<name>`` can be used to update the axes and figure object.
        Additional keyword arguments for these method calls can be set using ``ax_kw_<name>_<keyword>`` and
        ``fig_kw_<name>_<keyword>``. See [here](simple.plot.update_axes) for more details.

    """
    where_kwargs.update(utils.extract_kwargs(kwargs, prefix='where'))
    lscm_kwargs = utils.extract_kwargs(kwargs, 'linestyle', 'color', 'marker',
                                       linestyle=True, color=True, marker=False)

    model_in_legend = kwargs.pop('model_in_legend', None)

    ax = get_axes(ax)  # We are working on the axes object proper
    models = get_models(models, where=where, where_kwargs=where_kwargs)
    linestyles, colors, markers = get_lscm(**lscm_kwargs)

    # If there is only one model1 it is set as the title to make the legend shorter
    if len(models) == 1:
        if model_in_legend is None: model_in_legend = False
        kwargs.setdefault('ax_title', models[0].name)
    else:
        if model_in_legend is None: model_in_legend = True

    kwargs.setdefault('ax_xlabel', xgetter.get_label(models[0], xkey))
    kwargs.setdefault('ax_ylabel', ygetter.get_label(models[0], ykey))

    delayed_kwargs = update_axes(ax, kwargs, 'ax_legend')

    # Get the linestyle, color and marker for each thing to be plotted.
    lscm = [(linestyles[i], colors[i], markers[i]) for i in range(len(models))]

    label = kwargs.pop('label', '')
    mfc = kwargs.pop('markerfacecolor', None)
    for i, model in enumerate(models):
        ls, c, m = lscm.pop(0)

        if model_in_legend:
            legend = f'{label} {model.name}'
        else:
            legend = label

        xval = xgetter.get_data(model, xkey)
        yval = ygetter.get_data(model, ykey)

        if mask is not None:
            mask = model.get_mask(mask, y=yval, x=xval)
            xval = xval[mask]
            yval = yval[mask]

        ax.plot(xval, yval,
                color=c, ls=ls, marker=m,
                markerfacecolor=mfc or c,
                label=legend.strip() or None, **kwargs)

    update_axes(ax, delayed_kwargs)

    return ax, models

@utils.set_default_kwargs(
    linestyle=True, color=True, marker=False,
    ax_kw_xlabel_fontsize=15,
    ax_kw_ylabel_fontsize=15,
    markersize=4,
    ax_legend=dict(loc='upper right'),
    y_in_legend=None, model_in_legend=None,
    ax_tick_params=dict(axis='both', left=True, right=True, top=True)
    )

def template_plot_y(models, xgetter, ygetter, ykeys, *,
                    mask = None, ax = None, where=None, where_kwargs={},
                    **kwargs):
    """
    Template function for plotting different y values against a static x value.

    Args:
        models (): The models to be plotted
        xgetter (): Getter object for data on the x-axis
        ygetter (): Getter object for data on the y-axis
        ykeys (): A list of isotopes or ratios that will be plotted on the y-axis.
        ax (): The axes for the plotting.
        where (): A string to select which models to plot. See
            [``ModelCollection.where``](simple.models.ModelCollection.where) for more details.
        where_kwargs (): Arguments used to evaluate ``where``.
        **kwargs (): See section below for a description of acceptable keywords.

    Keyword Arguments:
        - ``y_in_legend`` Whether to add the y-axis data label to the legend label. If ``None`` only the
        unique part of the y-axis data label is included and common-to-all parts of the y-axis data labels is
        included in the default to the y-axis label.
        - ``model_in_legend`` Whether to add the model name to the legend label. If ``None`` the model name is
        only added if more than one model is being plotted.

        - ``linestyle`` Can be a list of linestyles that will be iterated through for each item plotted. If ``True``
        the default list of linestyles is used.
        - ``color`` Can be a list of colors that will be iterated through for each item plotted. If ``True``
        the default list of colors is used.
        - ``marker`` Can be a list of markers that will be iterated through for each item plotted. If ``True``
        the default list of markers is used.
        - Any keyword argument accepted by matplotlibs
        [``plot``](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html) method.

        - Keywords in the style of ``ax_<name>`` and ``fig_<name>`` can be used to update the axes and figure object.
        Additional keyword arguments for these method calls can be set using ``ax_kw_<name>_<keyword>`` and
        ``fig_kw_<name>_<keyword>``. See [here](simple.plot.update_axes) for more details.

    Returns:
        Axes, ModelCollection: The axes and the models plotted.

    """
    where_kwargs.update(utils.extract_kwargs(kwargs, prefix='where'))
    lscm_kwargs = utils.extract_kwargs(kwargs, 'linestyle', 'color', 'marker',
                                       linestyle=True, color=True, marker=False)

    y_in_legend = kwargs.pop('y_in_legend', None)
    model_in_legend = kwargs.pop('model_in_legend', None)

    ax = get_axes(ax) # We are working on the axes object proper
    models = get_models(models, where=where, where_kwargs=where_kwargs)
    linestyles, colors, markers = get_lscm(**lscm_kwargs)

    try:
        ykeys = simple.asratios(ykeys)
    except ValueError:
        try:
          ykeys = simple.asisotopes(ykeys)
        except ValueError:
            raise ValueError(f'Unable to convert {ykeys} into isotope or ratio strings')
        else:
            plot_ratio=False
    else:
        plot_ratio = True

    # Create a label for the y-axis
    if plot_ratio:
        # Figure out ylabel and what should go in the legend label.
        n = {r.numer for r in ykeys}
        d = {r.denom for r in ykeys}
        key_in_legend = True
        numer_in_legend = y_in_legend
        denom_in_legend = y_in_legend
        if len(n) == 1:
            ylabel = f"Slope of {ygetter.get_label(models[0], ykeys[0].numer)}"
            if numer_in_legend is None: numer_in_legend = False
        else:
            ylabel = 'Slope of A'
            if numer_in_legend is None:numer_in_legend = True

        if len(d) == 1:
            ylabel = f"{ylabel} / {ygetter.get_label(models[0], ykeys[0].denom)}"
            if denom_in_legend is None: denom_in_legend = False
        else:
            ylabel = f'{ylabel} / B'
            if denom_in_legend is None: denom_in_legend = True
    else:
        key_in_legend = y_in_legend
        numer_in_legend = False
        denom_in_legend = False
        if len(ykeys) == 1:
            ylabel = f"{ygetter.get_label(models[0], ykeys[0])}"
            if key_in_legend is None: key_in_legend = False
        else:
            ylabel = r'${\mathrm{R}}_{\mathrm{i}}$'
            if key_in_legend is None: key_in_legend = True
    kwargs.setdefault('ax_ylabel', ylabel)
    kwargs.setdefault('ax_xlabel', xgetter.get_label(models[0], None))

    # If there is only one model1 it is set as the title to make the legend shorter
    if len(models) == 1:
        if model_in_legend is None: model_in_legend = False
        kwargs.setdefault('ax_title', models[0].name)
    else:
        if model_in_legend is None: model_in_legend = True

    delayed_kwargs = update_axes(ax, kwargs, 'ax_legend')

    # Get the linestyle, color and marker for each thing to be plotted.
    if (len(models) == 1 or len(ykeys) == 1):
        #Everything get a different colour and linestyle
        lscm = [(linestyles[i], colors[i], markers[i]) for i in range(len(ykeys) * len(models))]
    else:
        # Each model has the same linestyle and each ykeys a different color
        lscm = [(linestyles[i // len(ykeys)], colors[i % len(ykeys)], markers[i // len(ykeys)])
                for i in range(len(ykeys) * len(models))]

    label = kwargs.pop('label', '')
    mfc = kwargs.pop('markerfacecolor', None)
    for i, model in enumerate(models):
        for ykey in ykeys:
            ls, c, m = lscm.pop(0)

            legend = ygetter.get_label(model, ykey, prefix=label, key_in_label=key_in_legend,
                                       numer_in_label=numer_in_legend, denom_in_label=denom_in_legend,
                                       model_in_label=model_in_legend)
            yval = ygetter.get_data(model, ykey)
            xval = xgetter.get_data(model)

            if mask is not None:
                parsed_mask = model.get_mask(mask, y=yval, x=xval)
                xval = xval[parsed_mask]
                yval = yval[parsed_mask]

            ax.plot(xval, yval,
                    color=c, ls=ls, marker=m,
                    markerfacecolor=mfc or c,
                    label=legend.strip() or None, **kwargs)

    update_axes(ax, delayed_kwargs)

    return ax, models


@utils.set_default_kwargs(
                        ax_kw_ylabel_labelpad=20,
                         )
def template_mhist_xy(models, xygetter, xisotope, yisotope,
                      mask = None, ax = None, where=None, where_kwargs={},
                      **kwargs):
    where_kwargs.update(utils.extract_kwargs(kwargs, prefix='where'))
    rose_kwargs = utils.extract_kwargs(kwargs, prefix='rose')

    try:
        xisotope = simple.asisotope(xisotope)
    except ValueError as e:
        raise ValueError(f'Unable to convert {xisotope} into isotope string') from e
    try:
        yisotope = simple.asisotope(yisotope)
    except ValueError as e:
        raise ValueError(f'Unable to convert {yisotope} into isotope string') from e


    ax = get_axes(ax)
    models = get_models(models, where=where, where_kwargs=where_kwargs)

    if ax.name != 'rose':
        ax = create_rose_plot(ax, **rose_kwargs)

    kwargs.setdefault('ax_xlabel', xygetter.get_label(models[0], xisotope))
    kwargs.setdefault('ax_ylabel', xygetter.get_label(models[0], yisotope))
    delayed_kwargs = update_axes(ax, kwargs, 'legend')

    for model in models:
        xval = xygetter.get_data(model, xisotope)
        yval = xygetter.get_data(model, yisotope)
        w = xygetter.get_weights(model)

        if mask is not None:
            mask = model.get_mask(mask, y=yval, x=xval, slope=yval/xval, w=w)
            xval = xval[mask]
            yval = yval[mask]
            w = w[mask]

        ax.mhist((xval, yval), weights=w, **kwargs)

    update_axes(ax, delayed_kwargs)


    return ax, models





