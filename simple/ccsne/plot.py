import numpy as np
import logging

import simple.plot
from simple import models, utils, plot

logger = logging.getLogger('SIMPLE.CCSNe.plotting')

__all__ = ['plot_abundance', 'plot_intnorm', 'plot_simplenorm',]

# For line, text and fill there is additional keyword - show - which determines if the thing is drawn
# For text there is also an additional keyword - y - which is the y position in xy



def get_onion_layer_mask(model, layer):
    # Returns a mask that can be used to select only data from a certain layer.
    # If there is no onion structure then the mask will not select any data.
    mask = np.full(model.masscoord.size, False, dtype=np.bool)

    lower_bounds = getattr(model, 'onion_lbounds', None)
    if lower_bounds is None:
        logger.error('No onion structure defined for this model')
        return mask

    def check(desired_layer, current_layer, lbound, ubound):
        if lbound>0:
            if desired_layer.lower() == current_layer:
                mask[lbound:ubound] = True
            return lbound
        else:
            return ubound

    layers = layer.split(',')
    for desired_layer in layers:
        ubound = len(mask.size)
        ubound = check(desired_layer, 'h', lower_bounds['H'][0], ubound)
        ubound = check(desired_layer, 'he/n', lower_bounds['He/N'][0], ubound)
        ubound = check(desired_layer, 'he/c', lower_bounds['He/C'][0], ubound)
        ubound = check(desired_layer, 'o/c', lower_bounds['O/C'][0], ubound)
        ubound = check(desired_layer, 'o/ne', lower_bounds['O/Ne'][0], ubound)
        ubound = check(desired_layer, 'o/si', lower_bounds['O/Si'][0], ubound)
        ubound = check(desired_layer, 'si', lower_bounds['Si'][0], ubound)
        ubound = check(desired_layer, 'ni', lower_bounds['Ni'][0], ubound)

    return mask

# TODO dont plot if smaller than x
@utils.set_default_kwargs(
    # Default settings for line, text and fill
    default_line_color='black', default_line_linestyle='--', default_line_lw=2, default_line_alpha=0.75,
    default_text_fontsize=10., default_text_color='black',
    default_text_horizontalalignment='center', default_text_xycoords=('data', 'axes fraction'), default_text_y = 1.01,
    default_fill_color='lightblue', default_fill_alpha=0.25,

    # For the rest we only need to give the values that differ from the default
   remnant_line_linestyle=':', remnant_fill_color='gray', remnant_fill_alpha=0.5,
   HeN_fill_show=False,
   OC_fill_show=False,
   OSi_fill_show=False,
   Ni_fill_show=False)
def plot_onion_structure(model, *, ax=None, **kwargs):
    if not isinstance(model, models.ModelTemplate):
        raise ValueError(f'model must be an Model object not {type(model)}')

    ax = plot.get_axes(ax)
    delayed_kwargs = plot.update_axes(ax, kwargs, 'ax_legend')

    lower_bounds = getattr(model, 'onion_lbounds', None)
    if lower_bounds is None:
        logger.error('No onion structure defined for this model')
        return

    masscoord = model.masscoord
    lbound_H = masscoord[lower_bounds['H'][0]]
    lbound_HeN = masscoord[lower_bounds['He/N'][0]]
    lbound_HeC = masscoord[lower_bounds['He/C'][0]]
    lbound_OC = masscoord[lower_bounds['O/C'][0]]
    lbound_ONe = masscoord[lower_bounds['O/Ne'][0]]
    lbound_OSi = masscoord[lower_bounds['O/Si'][0]]
    lbound_Si = masscoord[lower_bounds['Si'][0]]
    lbound_Ni = masscoord[lower_bounds['Ni'][0]]

    default_line = utils.extract_kwargs(kwargs, prefix='default_line')
    default_text = utils.extract_kwargs(kwargs, prefix='default_text')
    default_fill = utils.extract_kwargs(kwargs, prefix='default_fill')

    def add_line(name, x):
        line_kwargs = utils.extract_kwargs(kwargs, prefix='{name}_line', **default_line)
        if line_kwargs.pop('show', True):
            ax.axvline(x, **line_kwargs)

    def add_text(name, text, x):
        text_kwargs = utils.extract_kwargs(kwargs, prefix=f'{name}_text', **default_text)
        if text_kwargs.pop('show', True):
            # Using annotate instead of text as we can then specify x in absolute, and y coordinates relative, in space.
            ax.annotate(text_kwargs.pop('xytext', text), (x, text_kwargs.pop('y', 1.01)),
                        **text_kwargs)

    def add_fill(name, x):
        fill_kwargs = utils.extract_kwargs(kwargs, prefix=f'{name}_fill', **default_fill)
        if fill_kwargs.pop('show', True):
            ax.fill_between(x, fill_kwargs.pop('y1', [ylim[0], ylim[0]]),
                             fill_kwargs.pop('y2', [ylim[1], ylim[1]]),
                            **fill_kwargs)

    def add(name, text, lbound):
        if kwargs.get(f'{name}_show', True) is False:
            return

        if lbound<0 or not (ubound > xlim[0]) or not (lbound < ubound): return ubound

        if not lbound > xlim[0]:
            lbound = xlim[0]
        else:
            add_line(name, lbound)

        add_text(name, text, (lbound + ubound)/2)
        add_fill(name, [lbound, ubound])
        return lbound

    ylim = ax.get_ylim()
    xlim = ax.get_xlim()

    # Outside-in since the last lower limit is the upper limit of the next one

    ubound = np.min([model.masscoord[-1], xlim[1]])

    # H - Envelope
    ubound = add('H', 'H', lbound_H)

    # He/N
    ubound = add('HeN', 'He/N', lbound_HeN)

    # He/C
    ubound = add('HeC', 'He/C', lbound_HeC)

    # O/C
    ubound = add('OC', 'O/C',lbound_OC)

    # O/Ne
    ubound = add('ONe', 'O/Ne',lbound_ONe)

    # O/Si
    ubound = add('OSi', 'O/Si', lbound_OSi)

    # Si
    ubound = add('Si', 'Si', lbound_Si)

    # Ni
    ubound = add('Ni', 'Ni', lbound_Ni)

    # Remnant
    masscut = model.masscoord[0]
    if xlim[0] < masscut:
        lbound_rem = np.max([0,xlim[0]])
        add_text('remnant', r'M$_{\rm rem}$', ((lbound_rem + masscut) / 2))
        add_fill('remnant', [lbound_rem, masscut])

    plot.update_axes(ax, delayed_kwargs)


class MasscoordGetter(plot.GetterTemplate):
    def get_data(self, model, isotope=None):
        return model.masscoord

    def get_label(self, model=None, isotope=None):
        return 'Mass coordinate M${}_{\\odot}$'

@utils.set_default_kwargs()
def plot_abundance(models, isotopes_or_ratios, *,
                   semilog = False,
                   attrname='abundance', unit=None,
                   onion=None,
                   ax = None, where=None, where_kwargs={},
                   **kwargs):
    """
    Plots *ykey* from a data array against the mass coordinate for different CCSNe models.

    Args:
        models (): The collection of models to be plotted.
        ykey (): Can either an isotope or a ratio of two isotopes. Accepts multiple keys seperated by ``,``.
        semilog (bool): Whether to plot the data on the yaxis in a logarithmic scale.
        attrname (): The name of the attribute storing the data array.
        unit (str): The unit the data should be plotted in. If the data is stored in a different unit an attempt
            to convert the data to *unit* is made before plotting.
        onion (bool): Whether to plot the onion shell structure. Will only be shown in a single model is plotted.
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

    if semilog: kwargs.detdefault('ax_yscale', 'log')
    xgetter = MasscoordGetter()
    ygetter = plot.DataGetter(attrname, desired_unit=unit)
    ax, model = template_plot_y(models, xgetter, ygetter, isotopes_or_ratios, onion=onion,
                                ax=ax, where=where, where_kwargs=where_kwargs,
                                **kwargs)
    return ax

@utils.set_default_kwargs()
def plot_intnorm(models, ykey, *,
                 attrname = 'intnorm', onion=None,
                 ax = None, where=None, where_kwargs={},
                 **kwargs):
    """
    Plots *ykey* from the internally normalised eRi compositions against the mass coordinate for
    different CCSNe models.

    Args:
        models (): The collection of models to be plotted.
        ykey (): Can either an isotope or a ratio of two isotopes. Accepts multiple keys seperated by ``,``.
        attrname (): The name of the attribute storing the internally normalised data.
        ax (Axes): Axes on which to plot the data.
        onion (bool): Whether to plot the onion shell structure. Will only be shown in a single model is plotted.
        where (): A string to select which models to plot. See
            [``ModelCollection.where``](simple.models.ModelCollection.where) for more details.
        where_kwargs (): Keyword arguments to go with *where*.

    """
    xgetter = MasscoordGetter()
    ygetter = plot.NormGetter(attrname, 'eRi')
    ax, model = template_plot_y(models, xgetter, ygetter, ykey, onion=onion,
                                ax=ax, where=where, where_kwargs=where_kwargs,
                                **kwargs)
    return ax



@utils.set_default_kwargs()
def plot_simplenorm(models, isotopes_or_ratios, *,
                    attrname = 'simplenorm', onion=None,
                    ax = None, where=None, where_kwargs={}, **kwargs):
    """
    Plots *ykey* from the simply normalised Ri compositions against the mass coordinate for
    different CCSNe models.

    Args:
        models (): The collection of models to be plotted.
        isotopes_or_ratios (): Can either be a single ykeys or multiple ratios.

    """
    xgetter = MasscoordGetter()
    ygetter = plot.NormGetter(attrname, 'Ri')
    ax, model = template_plot_y(models, xgetter, ygetter, isotopes_or_ratios, onion=onion,
                                ax=ax, where=where, where_kwargs=where_kwargs,
                                **kwargs)
    return ax


def template_plot_y(models, xgetter, ygetter, isotopes_or_ratios, onion=False, **kwargs):
    # Wrapper that adds the option to plot the onion structure of CCSNe models
    onion_kwargs = utils.extract_kwargs(kwargs, prefix='onion')

    ax, models = plot.template_plot_y(models, xgetter, ygetter, isotopes_or_ratios, **kwargs)

    if onion or (onion is None and len(models) == 1):
        ax.set_title(None)
        if len(models) > 1:
            raise ValueError(f"Can only plot onion structure for a single model")
        else:
            plot_onion_structure(models[0], ax=ax, **onion_kwargs)

    return ax, models
