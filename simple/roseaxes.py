import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools

import numpy as np
import warnings

from simple import utils

import logging
logger = logging.getLogger('SIMPLE.roseaxes')

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
    """
    Convert *x*, *y* coordinated into a angle value given in radians.
    """

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
    """
        Convert *x*, *y* coordinated into a angle value given in degrees.
        """
    rad = xy2rad(x, y, xscale=xscale, yscale=yscale)
    return rad2deg(rad)


def deg2rad(deg):
    """Convert a degree angle into a radian angle`value"""
    return np.deg2rad(deg)


def rad2deg(rad):
    """Convert a degree angle into a radian angle`value"""
    return np.rad2deg(rad)


def get_cmap(name):
    """Return the matplotlib colormap with the given name."""
    try:
        return mpl.colormaps[name]
    except:
        return mpl.cm.get_cmap(name)


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
        self._colorbar = None

        super().__init__(*args,
                         theta_offset=np.pi * 0.5, theta_direction=-1,
                         **kwargs)

        self.tick_params(axis='y', which='major', labelleft=False, labelright=False)
        self.tick_params(axis='y', which='minor')
        self.tick_params(axis='x', which='major', direction='out')
        self.margins(y=0.1)

    def clear(self):
        """
        Clear the ax.
        """
        super().clear()

        self._last_hist_r = 0
        self._bar_color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

        # Grid stuff
        self.grid(True, axis='y', which='minor')
        self.grid(True, axis='y', which='major', color='black')
        self.grid(False, axis='x', which='minor')
        self.grid(True, axis='x', which='major')

        self.set_rlim(rmin=0)
        self.set_rticks([], minor=True)
        self.set_rticks([], minor=False)

        # Segment
        self.set_xysegment(self._xysegment)
        # self.axes.set_yticklabels([])

    def set_colorbar(self, vmin=None, vmax=None, log=False, cmap='turbo',
                     label=None, fontsize=None, show=True, ax=None, clear=True):
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
            show (): Whether to add a colorbar to the figure.
            ax (): The axis where the colorbar is drawn. If ``None`` it will be drawn on the right of the current axes.
            clear (): If ``True`` the current axes will be cleared.
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
                                                        ax=self, cax=ax, pad=0.1)
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
        elif segment.upper() == 'NE':
            self.axes.set_thetalim(0, np.pi * 0.5)
            self.axes.set_thetagrids((0, 90), (self._yscale, self._xscale))
        elif segment.upper() == 'SE':
            self.axes.set_thetalim(np.pi * 0.5, np.pi)
            self.axes.set_thetagrids((90, 180), (self._xscale, -self._yscale))
        elif segment.upper() == 'SW':
            self.axes.set_thetalim(np.pi, np.pi * 1.5)
            self.axes.set_thetagrids((180, 270), (-self._yscale, -self._xscale))
        elif segment.upper() == 'NW':
            self.axes.set_thetalim(np.pi * 1.5, np.pi * 2)
            self.axes.set_thetagrids((270, 360), (-self._xscale, self._yscale))
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

    def _rline(self, theta1, theta2, r):
        theta1, theta2, r = as0darray(theta1, theta2, r)

        if theta2 < theta1: theta1 -= np.pi * 2
        diff = (theta2 - theta1) / (np.pi * 2)

        nr = int(np.max([1, diff * self._rres * r])) + 1
        return np.linspace(theta1, theta2, nr), np.full(nr, r)

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
                        ezorder=ezorder, ealpha=ealpha, efill=efill, eline=eline, axline=axline,
                        antipodal=False, **kwargs)

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

    def rtext(self, text, r, deg=0, rotation=None, **kwargs):
        deg = deg % 360

        if rotation is None:
            rotation = deg * -1
            if deg > 90 and deg < 270:
                rotation = (rotation + 180) % 360

        label_kw = {'text': text, 'xy': (deg2rad(deg), r),
                    'ha': 'center', 'va': 'center', 'rotation': rotation}
        label_kw.update(kwargs)
        self.annotate(**label_kw)

    ####################
    ### Hist Methods ###
    ####################
    def _mbins(self, m, r, weights, rwidth, rscale, rescale, antipodal, bins, update_rticks, minor_rticks):
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

        if rescale:
            bin_weights = bin_weights / np.max(bin_weights)
        elif self._vrel:
            bin_weights = bin_weights / np.sum(bin_weights)

        if rscale:
            bin_heights = np.array([self._norm(bw, clip=True) for bw in bin_weights])
            bin_heights = bin_heights * rwidth
        else:
            bin_heights = np.full(bin_weights.size, rwidth)

        if update_rticks:
            major_ticks = list(self.get_yticks(minor=False))
            if r not in major_ticks:
                major_ticks.append(r)
                self.set_yticks(major_ticks, minor=False)
            self.yaxis.set_view_interval(r, r + rwidth * 1.15)

            minor_ticks = list(self.get_yticks(minor=True))
            update_minor = False
            for minor_r in list(np.linspace(r, r + rwidth, minor_rticks + 1))[1:]:
                if minor_r not in minor_ticks:
                    minor_ticks.append(minor_r)
                    update_minor = True
            if update_minor:
                self.set_yticks(minor_ticks, minor=True)

        return bin_weights, bin_edges, bin_heights

    def mhist(self, m, r=None, weights=1, rheight=0.9, rscale=True,
              rescale=False, antipodal=None, update_rticks=True, minor_rticks=2,
              bins=72, rtext=None, fill=True, outline=None, cmap=False, **kwargs):
        """
        Create a histogram of the given slopes.

        Args:
            m (float, (float, float)): Either a single array of floats representing a slope or a tuple of *x* and *y*
                coordinates from which a slope will be calculated.
            r (): The radius at which the histogram will be drawn. If 'None' it will be plotted ``1`` above the
                previous histogram, or at 1 if no histogram have been drawn.
            weights (): The weight assigned to each slope.
            rheight (): The height of the histogram bars.
            rscale (): If ``True`` height of the individual bins will be scaled to their weight. Otherwise all bins
                will have the same height.
            rescale (): If ``True`` all bin heights will be scaled relative to the heaviest bin. Otherwise, they are
                scaled relative to the sum of all bin weights or the range set by the colormap.
            antipodal (): Whether the antipodal data points will be included in the histogram. By default,
            ``antipodal=True`` when ``m`` is a slope and ``antipodal=False`` when ``m`` is *x,y* coordinates.
            bins (): The number of even sized bin in the histogram.
            rtext (): A text label for the histogram in the plot. Created using the ``rtext`` method. Addtional keywords
                arguments can be passed using the prefix ``rtext_``.

        """
        kwargs = utils.DefaultKwargs.Dict(kwargs)
        if r is None:
            r = self._last_hist_r + 1
        self._last_hist_r = r

        if outline is None:
            if fill and not cmap:
                outline = False
            else:
                outline = True

        color = kwargs.pop('color', None)
        if color is None:
            color = next(self._bar_color_cycle)

        rtext_kwargs = kwargs.pop_many(prefix='rtext')

        zorder = kwargs.pop('zorder', 1)
        fill_kwargs = kwargs.pop_many(prefix='fill', linestyle='', zorder=zorder)
        outline_kwargs = kwargs.pop_many(prefix='outline',
                                         color=color,
                                         linestyle=kwargs.pop('linestyle', '-'),
                                         linewidth=kwargs.pop('linewidth', 1),
                                         zorder=zorder + 0.001)
        baseline_kwargs = kwargs.pop_many(prefix='baseline', **outline_kwargs)

        bin_weights, bin_edges, bin_heights = self._mbins(m, r, weights, rheight, rscale, rescale, antipodal, bins,
                                                          update_rticks, minor_rticks)
        label = kwargs.pop('label', None)
        if (fill and not cmap) or outline:
            label_fill_kwargs = fill_kwargs.copy()
            if outline:
                label_fill_kwargs['linestyle'] = outline_kwargs['linestyle']
                label_fill_kwargs['linewidth'] = outline_kwargs['linewidth']
                label_fill_kwargs['edgecolor'] = outline_kwargs['color']
            if (fill and not cmap):
                label_fill_kwargs['fill'] = True
            else:
                label_fill_kwargs['fill'] = False

            self.fill([np.nan, np.nan], facecolor=color, label=label, **label_fill_kwargs)
        elif (fill and cmap) and rtext is None:
            rtext = label

        if fill:
            for i in range(bin_weights.size):
                if cmap:
                    bin_color = self._cmap(self._norm(bin_weights[i]))
                else:
                    bin_color = color

                self._rfill(bin_edges[i], bin_edges[i + 1], r, r + bin_heights[i],
                            color=bin_color, **fill_kwargs)

        if outline:
            theta_, r_ = np.array([]), np.array([])
            for i in range(bin_weights.size):
                tr = self._rline(bin_edges[i], bin_edges[i + 1], r + bin_heights[i])
                theta_, r_ = np.append(theta_, tr[0]), np.append(r_, tr[1])
            theta_, r_ = np.append(theta_, [theta_[0]]), np.append(r_, [r_[0]])
            self.axes.plot(theta_, r_, **outline_kwargs)

            theta_, r_ = self._rline(bin_edges[0], bin_edges[-1], r)
            self.axes.plot(theta_, r_, **baseline_kwargs)

        # self.axrline(r, 0, np.pi*2, color='black', linewidth = 0.2)

        if rtext is not None:
            rtext_kwargs.setdefault('r', r + rheight / 2)
            self.rtext(rtext, **rtext_kwargs)


mpl.projections.register_projection(RoseAxes)