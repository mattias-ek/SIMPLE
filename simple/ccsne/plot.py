import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import logging

import simple
from simple import models
from simple.plot import get_axes, get_models, get_lscm
import simple.ccsne.utils as ccsneutils


logger = logging.getLogger('SIMPLE.CCSNe.plotting')

def plot_onion_structure(model, *, axes=None, text_ypos = 1.01):
    if not isinstance(model, models.ModelTemplate):
        raise ValueError(f'model must be an Model object not {type(model)}')

    axes = get_axes(axes)

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

    mass = model.masscoord
    masscut = mass[0]
    massmax = mass[-1]

    # plot limits
    xl1 = masscut - 0.25
    xl2 = np.round(massmax, 1)
    xln = np.round(massmax, 1)

    # Using annotate instead of text as we can then sp

    # remnant
    axes.axvline(x=masscut, color='black', linestyle=':', lw=2, alpha=0.75)
    axes.annotate(r'M$_{\rm rem}$', (((xl1 + masscut) / 2), text_ypos), fontsize=10., color='black',
                   horizontalalignment='center', xycoords=('data', 'axes fraction'))
    axes.fill_between([xl1, masscut], [-1.e10, -1.e10], [1.e10, 1.e10], color='gray', alpha=0.5)

    # Convective envelope
    if (lbound_H + 1) < massmax:
        xl2 = np.round(lbound_H + 0.5, 1)
        axes.axvline(x=lbound_H, color='black', linestyle='--', lw=2, alpha=0.75)
        axes.fill_between([lbound_H, xl2], [-1.e10, -1.e10], [1.e10, 1.e10], color='lightblue', alpha=0.25)
        if np.round(lbound_H, 2) != np.round(massmax, 2):
            axes.annotate('H', (((lbound_H + xl2) / 2), text_ypos), fontsize=10., color='black',
                           horizontalalignment='center', xycoords=('data', 'axes fraction'))

    # He/N layer
    if (lbound_HeC + 0.05) < massmax:
        axes.axvline(x=lbound_HeC, color='black', linestyle='--', lw=2, alpha=0.75)
        if lbound_H == massmax:
            xl2 = np.round(lbound_HeN + 0.3, 1)
            axes.annotate('He/N', (((xl2 + lbound_HeN) / 2), text_ypos), fontsize=10., color='black',
                           horizontalalignment='center', xycoords=('data', 'axes fraction'))
        else:
            axes.annotate('He/N', (((lbound_H + lbound_HeN) / 2), text_ypos), fontsize=10., color='black',
                           horizontalalignment='center', xycoords=('data', 'axes fraction'))

    # He/C layer
    if (lbound_HeN + 0.04) < massmax:
        axes.axvline(x=lbound_HeN, color='black', linestyle='--', lw=2, alpha=0.75)
        axes.annotate('He/C', (((lbound_HeC + lbound_HeN) / 2), text_ypos), fontsize=10., color='black',
                       horizontalalignment='center', xycoords=('data', 'axes fraction'))

        if (lbound_H + 0.05) > massmax and lbound_HeC + 0.05 < massmax:
            if (lbound_HeC - lbound_OC) < 0.2:
                xl2 = np.round(lbound_HeN + 0.5, 1)
                axes.annotate('He/C', (((int(xl2) + lbound_OC) / 2), text_ypos), fontsize=10., color='black',
                               horizontalalignment='center', xycoords=('data', 'axes fraction'))

    axes.axvline(x=lbound_OC, color='black', linestyle='--', lw=2, alpha=0.75)

    # merger
    # max_value = max([mass[ic2], masscut, mass[ine]])
    if np.abs(lbound_OC - lbound_ONe) > 0.05:
        axes.axvline(x=lbound_ONe, color='black', linestyle='--', lw=2, alpha=0.75)
        axes.annotate('O/Ne', (((lbound_OC + max(lbound_ONe, masscut)) / 2), text_ypos),  fontsize=10., color='black',
                       horizontalalignment='center', xycoords=('data', 'axes fraction'))
        axes.fill_between([max(lbound_ONe, masscut), lbound_OC], [-1.e10, -1.e10], [1.e10, 1.e10],
                               color='lightblue', alpha=0.25)

    axes.axvline(x=lbound_OSi, color='black', linestyle='--', lw=2, alpha=0.75)
    axes.axvline(x=lbound_Si, color='black', linestyle='--', lw=2, alpha=0.75)

    axes.fill_between([lbound_HeC, lbound_HeN], [-1.e10, -1.e10], [1.e10, 1.e10], color='lightblue', alpha=0.25)

    axes.annotate('O/C', (((max(lbound_OC, masscut) + lbound_HeC) / 2), text_ypos),  fontsize=10., color='black',
                   horizontalalignment='center', xycoords=('data', 'axes fraction'))

    if (lbound_OSi > masscut and (lbound_OSi - masscut) > 0.05):
        axes.annotate('O/Si', (((lbound_ONe + lbound_OSi) / 2), text_ypos),  fontsize=10., color='black',
                       horizontalalignment='center', xycoords=('data', 'axes fraction'))
    elif (lbound_ONe - masscut) > 0.05:
        axes.annotate('O/Si', (((lbound_ONe + masscut) / 2), text_ypos), fontsize=10., color='black',
                       horizontalalignment='center', xycoords=('data', 'axes fraction'))

    # Si layer
    if lbound_OSi > masscut:
        if not np.isnan(lbound_Si):
            axes.annotate('Si', (((lbound_OSi + lbound_Si) / 2), text_ypos), fontsize=10., color='black',
                           horizontalalignment='center', xycoords=('data', 'axes fraction'))
            axes.fill_between([lbound_Si, lbound_OSi], [-1.e10, -1.e10], [1.e10, 1.e10], color='lightblue',
                                   alpha=0.25)  # silicon
        elif (lbound_OSi - masscut) > 0.05:
            axes.annotate('Si', (((lbound_OSi + masscut) / 2), text_ypos),  fontsize=10., color='black',
                           horizontalalignment='center', xycoords=('data', 'axes fraction'))
            axes.fill_between([lbound_OSi, masscut], [-1.e10, -1.e10], [1.e10, 1.e10], color='lightblue',
                                   alpha=0.25)  # silicon

    # Ni layer
    if not np.all(np.isnan(lbound_Si)):
        axes.annotate('Ni', (((lbound_Ni + lbound_Si) / 2), text_ypos), fontsize=10., color='black',
                       horizontalalignment='center', xycoords=('data', 'axes fraction'))

    if (xl2 - xln) > 0.3:
        xl2 = np.round(massmax, 1) + 0.1

    return (xl1, xl2)


def plot_intnorm(models, isotopes_or_ratios, *, attrname = 'intnorm', onion=False,
                 axes = None, where = None, where_kwargs={}, **kwargs):
    """
    Plots the slope of two internally normalised eRi compositions against the mass coordinates.

    Args:
        models (): The collection of models to be plotted.
        isotopes_or_ratios (): Can either be a single isotopes_or_ratios or multiple ratios.
        where (): Can be used to select only a subset of the models to plot.
        where_kwargs ():

    """
    if where is not None:
        models = models.where(where, **where_kwargs)

    axes = get_axes(axes)

    plot_norm(models, attrname, 'eRi', isotopes_or_ratios, **kwargs)

    if onion:
        axes.set_title(None)
        if len(models) > 1:
            raise ValueError(f"Can only plot onion structure for a single model")
        else:
            plot_onion_structure(models[0], axes=axes)

def plot_simplenorm(models, isotopes_or_ratios, *, attrname = 'simplenorm', onion=False,
                 axes = None, where = None, where_kwargs={}, **kwargs):
    """
    Plots the slope of two internally normalised eRi compositions against the mass coordinates.

    Args:
        models (): The collection of models to be plotted.
        isotopes_or_ratios (): Can either be a single isotopes_or_ratios or multiple ratios.
        where (): Can be used to select only a subset of the models to plot.
        where_kwargs ():

    """
    if where is not None:
        models = models.where(where, **where_kwargs)

    axes = get_axes(axes)

    plot_norm(models, attrname, 'Ri', isotopes_or_ratios, **kwargs)

    if onion:
        axes.set_title(None)
        if len(models) > 1:
            raise ValueError(f"Can only plot onion structure for a single model")
        else:
            plot_onion_structure(models[0], axes=axes)


def plot_norm(models, normname, Rvalname, isotopes_or_ratios, *,
              linestyle=True, color=True, marker=False,
              axes = None, use_title = True,
              where=None, where_kwargs={},
              **kwargs):
    """
    Plots the slope of two internally normalised eRi compositions against the mass coordinates.

    Args:
        models (): The collection of models to be plotted.
        isotopes_or_ratios (): Can either be a single isotopes_or_ratios or multiple ratios.
        where (): Can be used to select only a subset of the models to plot.
        where_kwargs ():

    """
    # Work on the axes object. That way it will work for subplots to

    axes = get_axes(axes)
    models = get_models(models, where=where, where_kwargs=where_kwargs)
    linestyles, colors, markers = get_lscm(linestyle, color, marker)

    try:
        isotopes_or_ratios = simple.asratios(isotopes_or_ratios)
    except ValueError:
        try:
          isotopes_or_ratios = simple.asisotopes(isotopes_or_ratios)
        except ValueError:
            raise ValueError(f'Unable to convert {isotopes_or_ratios} into isotope or isotopes_or_ratios strings')
        else:
            plot_ratio=False
    else:
        plot_ratio = True

    if plot_ratio:
        # Figure out ylabel and what should go in the legend label.
        n = {r.numer for r in isotopes_or_ratios}
        d = {r.denom for r in isotopes_or_ratios}
        if len(n) == 1:
            ylabel = f"Slope of {getattr(models[0], normname).label_latex[isotopes_or_ratios[0].numer]}"
            nlegend = False
        else:
            ylabel = 'Slope of A'
            nlegend = True

        if len(d) == 1:
            ylabel = f"{ylabel} / {getattr(models[0], normname).label_latex[isotopes_or_ratios[0].denom]}"
            dlegend = False
        else:
            ylabel = f'{ylabel} / B'
            dlegend = True
    else:
        if len(isotopes_or_ratios) == 1:
            ylabel = f"{getattr(models[0], normname).label_latex[isotopes_or_ratios[0]]}"
            ilegend = False
        else:
            ylabel = r'${\mathrm{R}}_{\mathrm{i}}}$'
            ilegend = True

    axes.set_ylabel(ylabel, fontsize=15)
    axes.set_xlabel('Mass coordinate M$_{\odot}$', fontsize=15)

    # If there is only one model1 it is set as the title to make the legend shorter
    if len(models) == 1 and use_title:
        mlegend = False
        axes.set_title(models[0].name)
    else:
        mlegend = True

    if (len(models) == 1 or len(isotopes_or_ratios) == 1):
        #Everything get a different colour and linestyle
        lscm = [(linestyles[i], colors[i], markers[i]) for i in range(len(isotopes_or_ratios)*len(models))]
    else:
        # Each model1 has the same linesyle and each isotopes_or_ratios a different color
        lscm = [(linestyles[i//len(models)], colors[i%len(isotopes_or_ratios)], markers[i%len(isotopes_or_ratios)])
                for i in range(len(isotopes_or_ratios) * len(models))]

    min_masscut, max_masscut = [], []
    label = kwargs.pop('label', '')
    mfc = kwargs.pop('markerfacecolor', None)
    for iso_or_rat in isotopes_or_ratios:
        for i, model in enumerate(models):
            norm = getattr(model, normname)
            Rval = getattr(model, Rvalname)
            ls, c, m = lscm.pop(0)

            legend = label
            if plot_ratio:
                if nlegend and dlegend: legend += f'{norm.label_latex[iso_or_rat.numer]}/{norm.label_latex[iso_or_rat.denom]}'
                elif nlegend: legend += f"{norm.label_latex[iso_or_rat.numer]}"
                elif dlegend: legend += f" {norm.label_latex[iso_or_rat.denom]}"
                yval = Rval[iso_or_rat.numer] / Rval[iso_or_rat.denom]
            else:
                if ilegend: legend += f'{norm.label_latex[iso_or_rat]}'
                yval = Rval[iso_or_rat]

            if mlegend: legend += f' {model.name}'
            axes.plot(model.masscoord, yval,
                      color=c, markersize=4, ls=ls, marker=m,
                      markerfacecolor=mfc or c,
                      label=legend.strip() or None, **kwargs)
            min_masscut.append(np.min(model.masscoord))
            max_masscut.append(np.max(model.masscoord))

    axes.legend(loc='upper right')
    axes.set_xlim(np.min(min_masscut), np.max(max_masscut))
    axes.tick_params(left=True, right=True, top=True, labelleft=True, which='both')  # ,labelright=True)

    axes.xaxis.set_minor_locator(AutoMinorLocator())
    axes.yaxis.set_minor_locator(AutoMinorLocator())

    return axes