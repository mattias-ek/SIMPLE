import logging

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import simple
import logging

logger = logging.getLogger('SIMPLE.CCSNe.plotting')


class EndlessList:
    # Index will never go out of bounds. It will just start from the beginning if larger than the initial list.
    def __init__(self, items):
        if type(items) is list:
            self.items = items
        elif type(items) is self.__class__:
            self.items = items.items
        else:
            self.items = [items]

    def __getitem__(self, index):
        return self.items[index % len(self.items)]

    def __len__(self):
        return len(self.items)



# colours appropriate for colour blindness
# Taken from https://davidmathlogic.com/colorblind/#%23000000-%23E69F00-%2356B4E9-%23009E73-%23F0E442-%230072B2-%23D55E00-%23CC79A7
all_colors=EndlessList(["#D55E00", "#56B4E9", "#009E73", "#E69F00", "#CC79A7", "#0072B2", "#F0E442"])
all_linestyles = EndlessList(['-', (0, (4, 4)), (0, (2,1)),
                          (0, (4,2,1,2)), (0, (4,2,1,1,1,2)), (0, (4,2,1,1,1,1,2)),
                          (0, (2,1,2,2,1,2)), (0, (2,1,2,2,1,1,1,2)), (0, (2,1,2,2,1,1,1,1,1,2)),
                          (0, (2,1,2,1,2,2,1,2)), (0, (2,1,2,1,2,2,1,1,1,2)), (0, (2,1,2,1,2,2,1,1,1,1,1,2))])
all_markers = EndlessList(["o", "s", "^", "D", "P","X", "v", "<", ">",  "*", "p", "d", "H"])

def get_onion_structure(model):
    """
    Return a key array with the lower boundaries of the H, He/N, He/C, O/C, O/Ne, O/Si, Si and Ni shells/layers.
    """
    mass = model.masscoord
    he4 = model.abundance['He-4']
    c12 = model.abundance['C-12']
    ne20 = model.abundance['Ne-20']
    o16 = model.abundance['O-16']
    si28 = model.abundance['Si-28']
    n14 = model.abundance['N-14']
    ni56 = model.abundance['Ni-56']
    masscut = mass[0]
    massmax = mass[-1]
    logging.INFO("m_cut: " + str(masscut))
    logging.INFO("massmax: " + str(massmax))

    shells = 'H He/N He/C O/C O/Ne O/Si Si Ni'.split()
    boundaries = []

    # definition of borders
    try:
        ih = np.where((he4 > 0.5))[0][-1]
    except IndexError:
        logging.INFO("No lower boundary of H shell")
        boundaries.append(np.nan)
    else:
        logging.INFO("Lower boundary of the H shell: " + str(mass[ih]))
        boundaries.append(mass[ih])

    try:
        ihe1 = np.where((n14 > o16) & (n14 > c12) & (n14 > 1.e-3))[0][0]
    except IndexError:
        logging.INFO("No lower boundary of He/N shell")
        boundaries.append(np.nan)
    else:
        logging.INFO("Lower boundary of the He/N shell: " + str(mass[ihe1]))
        boundaries.append(mass[ihe1])
    try:
        ihe = np.where((c12 > he4) & (mass <= mass[ih]))[0][-1]
    except IndexError:
        logging.INFO("No lower boundary of He/C shell")
        boundaries.append(np.nan)
    else:
        logging.INFO("Lower boundary of the He/C shell: " + str(mass[ihe]))
        boundaries.append(mass[ihe])

    try:
        ic2 = np.where((c12 > ne20) & (si28 < c12) & (c12 > 8.e-2))[0][0]
    except IndexError:
        logging.INFO("No lower boundary of O/C shell")
        boundaries.append(np.nan)
        ic2 = np.nan
    else:
        logging.INFO("Lower boundary of the O/C shell: " + str(mass[ic2]))
        boundaries.append(mass[ic2])

    try:
        ine = np.where((ne20 > 1.e-3) & (si28 < ne20) & (ne20 > c12))[0][0]
    except IndexError:
        logging.INFO("No lower boundary of O/Ne shell")
        boundaries.append(np.nan)
    else:
        if ine > ic2:
            ine = ic2
        logging.INFO("Lower boundary of the O/Ne shell: " + str(mass[ine]))
        boundaries.append(mass[ine])

    try:
        io = np.where((si28 < o16) & (o16 > 5.e-3))[0][0]
    except IndexError:
        logging.INFO("No lower boundary of O/Si layer")
        boundaries.append(np.nan)
    else:
        logging.INFO("Lower boundary of the O/Si layer: " + str(mass[io]))
        boundaries.append(mass[io])

    try:
        isi = np.where((ni56 > si28))[0][-1]
    except IndexError:
        logging.INFO("No lower boundary of Si layer")
        boundaries.append(np.nan)
    else:
        logging.INFO("Lower boundary of the Si layer: " + str(mass[isi]))
        boundaries.append(mass[isi])

    try:
        ini = np.where((ni56 > si28) & (mass > masscut))[0][0]
    except IndexError:
        logging.INFO("No lower boundary of Ni layer")
        boundaries.append(np.nan)
    else:
        logging.INFO("Lower boundary of the Ni layer: " + str(mass[ini]))
        boundaries.append(mass[ini])

    return simple.askeyarray(boundaries, shells)


def plot_onion_structure(model):
    pass


# TODO change this so that the first item is axes where to plot the data
# if its plt then use gca() on it. But then all the method calls need to be the axes
# ones and not the plt ones. But this way it will work if you have subplots etc.
def plot_slopes(models, ratio, linestyle=True, color=True, marker=False, axes = None, where=None, where_kwargs={}):
    """
    Plots the slope of two internally normalised eRi compositions.

    Args:
        models (): The collection of models to be plotted.
        ratio (): Can either be a single ratio or multiple ratios.
        where (): Can be used to select only a subset of the models to plot.
        where_kwargs ():

    """
    # Work on the axes object. That way it will work for subplots to
    if axes is None:
        axes = plt.gca()
    elif isinstance(axes, Axes):
        pass
    elif hasattr(axes, 'gca'):
        axes = axes.gca() # if axes = plt
    else:
        raise ValueError('axes must be an Axes, Axes instance or have a gca() method that return an Axes')

    # select models
    if where is not None:
        models = models.where(where, **where_kwargs)

    if len(models) == 0:
        logger.warning('No models to plot')
        return

    ratios = simple.asratios(ratio)

    # Figure out ylabel and what should go in the legend label.
    n = {r.numer for r in ratios}
    d = {r.denom for r in ratios}
    if len(n) == 1:
        ylabel = f"Slope of {models[0].intnorm.label_latex[ratios[0].numer]}"
        nlegend = False
    else:
        ylabel = 'Slope of A'
        nlegend = True

    if len(d) == 1:
        ylabel = f"{ylabel} / {models[0].intnorm.label_latex[ratios[0].denom]}"
        dlegend = False
    else:
        ylabel = f'{ylabel} / B'
        dlegend = True

    if len(models) == 1:
        mlegend = False
        title = models[0].name
    else:
        mlegend = True
        title = None

    # Sorts our the linestyle, color and marker for each plot
    if color is False:
        colors = EndlessList("#000000")
    elif color is True:
        colors = all_colors
    else:
        colors = EndlessList(color)

    if linestyle is False:
        linestyles = EndlessList("")
    elif linestyle is True:
        linestyles = all_linestyles
    else:
        linestyles = EndlessList(linestyle)

    if marker is False:
        markers = EndlessList("")
    elif marker is True:
        markers = all_markers
    else:
        markers = EndlessList(marker)

    if (len(models) == 1 or len(ratios) == 1):
        #Everything get a different colour and linestyle
        lscm = [(linestyles[i], colors[i], markers[i]) for i in range(len(ratios)*len(models))]
    else:
        # Each model has the same linesyle and each ratio a different color
        lscm = [(linestyles[i//len(models)], colors[i%len(ratios)], markers[i%len(ratios)])
                for i in range(len(ratios) * len(models))]

    masscut = []
    for rat in ratios:
        for i, model in enumerate(models):
            legend = ""
            if nlegend and dlegend: legend += f'{model.intnorm.label_latex[rat.numer]}/{model.intnorm.label_latex[rat.denom]}'
            elif nlegend: legend += f"{model.intnorm.label_latex[rat.numer]}"
            elif dlegend: legend += f" {model.intnorm.label_latex[rat.denom]}"
            if mlegend: legend += f' {model.name}'

            ls, c, m = lscm.pop(0)
            axes.plot(model.masscoord, model.intnorm.eRi[rat.numer] / model.intnorm.eRi[rat.denom],
                     color=c, markersize=4, ls=ls, marker = m, markerfacecolor = c, label=legend.strip())
            masscut.append(np.min(model.masscoord))

    # If there is only one model it is set as the title to make the legend shorter
    if title is not None:
        axes.set_title(title)
    axes.set_ylabel(ylabel, fontsize=15)
    axes.set_xlabel('Mass coordinate M$_{\odot}$', fontsize=15)

    axes.legend(loc='upper right')
    axes.set_xlim(np.min(masscut), 9)
    axes.tick_params(left=True, right=True, top=True, labelleft=True, which='both')  # ,labelright=True)

    axes.xaxis.set_minor_locator(AutoMinorLocator())
    axes.yaxis.set_minor_locator(AutoMinorLocator())