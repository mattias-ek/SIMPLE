import numpy as np
import logging

import simple.plot
from simple import models, utils, plot
import simple.ccsne.utils as ccsneutils

logger = logging.getLogger('SIMPLE.CCSNe.plotting')

__all__ = ['plot_abundance', 'plot_intnorm', 'plot_simplenorm', 'mhist_intnorm']

# Stores default kwargs for plots. Should be mapped to the prefixes of the function.
default_kwargs = dict()


# For line, text and fill there is additional keyword - show - which determines if the thing is drawn
# For text there is also an additional keyword - y - which is the y position in xy
default_kwargs['plot_onion_structure'] = dict(
    default = dict(line=dict(color='black', linestyle='--', lw=2, alpha=0.75),
                   text=dict(fontsize=10., color='black',
                             horizontalalignment='center',
                             xycoords=('data', 'axes fraction'), y = 1.01),
                   fill=dict(color='lightblue', alpha=0.25)),

                    # For the rest we only need to give the values that differ from the default
                   remnant = dict(line=dict(linestyle=':'),
                                  fill=dict(color='gray', alpha=0.5)),
                   HeN = dict(fill=dict(show=False)),
                   OC = dict(fill=dict(show=False)),
                   OSi = dict(fill=dict(show=False)),
                   Ni = dict(fill=dict(show=False)))

@utils.set_default_kwargs(default_kwargs,
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

def plot_onion_structure(model, *, xlim = None, ylim=None, ax=None, **kwargs):
    if not isinstance(model, models.ModelTemplate):
        raise ValueError(f'model must be an Model object not {type(model)}')

    ax = plot.get_axes(ax)

    lower_bounds = getattr(model, 'onion_lower_bounds', None)
    if lower_bounds is None:
        lower_bounds = ccsneutils.get_onion_structure(model)

    lbound_H = lower_bounds['H'][0]
    lbound_HeN = lower_bounds['He/N'][0]
    lbound_HeC = lower_bounds['He/C'][0]
    lbound_OC = lower_bounds['O/C'][0]
    lbound_ONe = lower_bounds['O/Ne'][0]
    lbound_OSi = lower_bounds['O/Si'][0]
    lbound_Si = lower_bounds['Si'][0]
    lbound_Ni = lower_bounds['Ni'][0]

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

        if np.isnan(lbound) or not (ubound > xlim[0]) or not (lbound < ubound): return ubound

        if not lbound > xlim[0]:
            lbound = xlim[0]
        else:
            add_line(name, lbound)

        add_text(name, text, (lbound + ubound)/2)
        add_fill(name, [lbound, ubound])
        return lbound

    if ylim is None:
        ylim = ax.get_ylim()
    if xlim is None:
        xlim = ax.get_xlim()

    # Outside-in since the last lower limit is the upper limit of the next one

    # @ Gabor - I dont understand all the additional limits you imposed here?
    # Were they just cosmetic or were they part of the limit determinations.
    # If its the latter it should go in the function that determines the onion structure.
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

class AbuGetter:
    def __init__(self, attrname, unit, isotope = None):
        self.attrname = attrname
        self.isotope = isotope

        for k, v in utils.UNITS.items():
            if unit in v:
                self.unit = k
                break
        else:
            raise ValueError(f'Unit not recognised')


    def get_data(self, model, isotope=None):
        if isotope is None: isotope = self.isotope
        data = getattr(model, self.attrname)[isotope]
        data_unit = getattr(model, f"{self.attrname}_unit", None)

        if data_unit is None:
            logger.warning('Data does not have a specified unit. Assuming it has the requested unit')
            data_unit = self.unit

        if data_unit not in utils.UNITS[self.unit]:
            if self.unit == 'mass' and data_unit in utils.UNITS['mole']:
                logger.info(f'Multiplying data by the isotope mass number to convert from mass to moles')
                data = data * float(isotope.mass)
            elif self.unit == 'mole' and data_unit in utils.UNITS['mass']:
                logger.info(f'Dividing data by the isotope mass number to convert from moles to mass')
                data = data / float(isotope.mass)
            else:
                raise ValueError(f'Unable to convert data from {data_unit} to {self.unit})')

        return data

    def get_label(self, model, isotope=None):
        if isotope is None: isotope = self.isotope
        return rf'${{}}^{{{isotope.mass}}}\mathrm{{{isotope.element}}}$'

class NormGetter:
    def __init__(self, attrname, Rvalname, isotope=None):
        self.attrname = attrname
        self.Rvalname = Rvalname
        self.isotope = isotope

    def get_data(self, model, isotope=None):
        if isotope is None: isotope = self.isotope
        norm = getattr(model, self.attrname)
        Rval = getattr(norm, self.Rvalname)
        return Rval[isotope]

    def get_label(self, model, isotope=None):
        if isotope is None: isotope = self.isotope
        return getattr(model, self.attrname).label_latex[isotope]

class MasscoordGetter:
    def get_data(self, model, isotope=None):
        return model.masscoord

    def get_label(self, model=None, isotope=None):
        return 'Mass coordinate M$_{\odot}$',

@utils.set_default_kwargs(default_kwargs,)
def plot_abundance(models, isotopes_or_ratios, *,
                   onion=None, semilog = False, unit='mass',
                   ax = None, where=None, where_kwargs={},
                   **kwargs):

    if semilog: kwargs['yscale'] = 'log'
    xgetter = MasscoordGetter()
    ygetter = AbuGetter('abundance', unit)
    ax, model = helper_plot_multiy_fixedx(models, xgetter, ygetter, isotopes_or_ratios, onion=onion,
                                          ax=ax, where=where, where_kwargs=where_kwargs,
                                          **kwargs)
    return ax

@utils.set_default_kwargs(default_kwargs,)
def plot_intnorm(models, isotopes_or_ratios, *,
                 attrname = 'intnorm', onion=None,
                 ax = None, where=None, where_kwargs={},
                 **kwargs):
    """
    Plots the slope of two internally normalised eRi compositions against the mass coordinates.

    Args:
        models (): The collection of models to be plotted.
        isotopes_or_ratios (): Can either be a single isotopes_or_ratios or multiple ratios.
        where (): Can be used to select only a subset of the models to plot.
        where_kwargs ():

    """
    xgetter = MasscoordGetter()
    ygetter = NormGetter(attrname, 'eRi')
    ax, model = helper_plot_multiy_fixedx(models, xgetter, ygetter, isotopes_or_ratios, onion=onion,
                                          ax=ax, where=where, where_kwargs=where_kwargs,
                                          **kwargs)
    return ax

@utils.set_default_kwargs(default_kwargs,)
def plot_simplenorm(models, isotopes_or_ratios, *,
                    attrname = 'simplenorm', onion=None,
                    ax = None, where=None, where_kwargs={}, **kwargs):
    """
    Plots the slope of two simply normalised Ri compositions against the mass coordinates.

    Args:
        models (): The collection of models to be plotted.
        isotopes_or_ratios (): Can either be a single isotopes_or_ratios or multiple ratios.

    """
    xgetter = MasscoordGetter()
    ygetter = NormGetter(attrname, 'Ri')
    ax, model = helper_plot_multiy_fixedx(models, xgetter, ygetter, isotopes_or_ratios, onion=onion,
                                          ax=ax, where=where, where_kwargs=where_kwargs,
                                          **kwargs)
    return ax

@utils.set_default_kwargs(default_kwargs,)
def mhist_intnorm(models, xisotope, yisotope, *,
                 attrname='intnorm',
                 ax = None, where=None, where_kwargs={},
                 **kwargs):

    xygetter = NormGetter(attrname, 'eRi')
    ax, model = simple.plot.helper_mhist_singlex_singley(models, xygetter, xygetter,
                                                         xisotope, yisotope,
                                                         ax=ax, where=where, where_kwargs=where_kwargs,
                                                         **kwargs)
    return ax

@utils.set_default_kwargs(default_kwargs,)
def mhist_abundance(models, xisotope, yisotope, *,
                    unit = 'mass',
                    ax = None, where=None, where_kwargs={},
                    **kwargs):

    xygetter = AbuGetter('abundance', unit)
    ax, model = simple.plot.helper_mhist_singlex_singley(models, xygetter, xygetter,
                                                         xisotope, yisotope,
                                                         ax=ax, where=where, where_kwargs=where_kwargs,
                                                         **kwargs)
    return ax


def helper_plot_multiy_fixedx(models, xgetter, ygetter, isotopes_or_ratios, onion=False, **kwargs):
    # Wrapper that adds the option to plot the onion structure of CCSNe models
    onion_kwargs = utils.extract_kwargs(kwargs, prefix='onion')

    ax, models = plot.helper_plot_multiy_fixedx(models, xgetter, ygetter, isotopes_or_ratios, **kwargs)

    if onion or (onion is None and len(models) == 1):
        ax.set_title(None)
        if len(models) > 1:
            raise ValueError(f"Can only plot onion structure for a single model")
        else:
            plot_onion_structure(models[0], ax=ax, **onion_kwargs)

    return ax, models
