import logging

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import simple


class EndlessList:
    # Index will never go out of bounds. It will just start from the beginning if larger than the initial list.
    def __init__(self, items):
        self.list = list(items)

    def __getitem__(self, index):
        return self.list[index % len(self.list)]

colours=EndlessList(["red", "blue", "green", "orange", "black", "blueviolet", "darkgoldenrod", "mediumvioletred"])
linestyles = EndlessList(['-', '--', ':', '-.'])
markers = EndlessList(["o", "s", "^", "D", "P"])

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
def plot_slopes(models, ratio, axes = None, where=None, where_kwargs={}):
    """
    Plots the slope of two internally normalised eRi compositions.

    Args:
        models (): The collection of models to be plotted.
        ratio (): Can either be a single ratio or multiple ratios.
        where (): Can be used to select only a subset of the models to plot.
        where_kwargs ():

    """
    if axes is None:
        axes = plt.gca()

    if where is not None:
        models = models.where(where, **where_kwargs)

    if len(models) == 0:
        print('No models to plot')
        return

    ratios = simple.asratios(ratio)

    if len(ratios) == 1:
        n = models[0].intnorm.label_latex[ratios[0].numer]
        d = models[0].intnorm.label_latex[ratios[0].denom]
        ylabel = f'Slope of {n}/{d}'
        simple_legend = True
    else:
        ylabel = f'Slope of A/B'
        simple_legend = False

    plt.figure(figsize=(9, 6))

    masscut = []
    for rat in ratios:
        for i, model in enumerate(models):
            if simple_legend:
                legend = model.name
            else:
                n = models[0].intnorm.label_latex[rat.numer]
                d = models[0].intnorm.label_latex[rat.denom]
                legend = f'{n}/{d} {model.name}'

            plt.plot(model.masscoord, model.intnorm.eRi[rat.numer] / model.intnorm.eRi[rat.denom],
                     color=colours[i], markersize=4, ls='-', label=legend)  # label=label_for_legend+ ' '+label_models[i]
            masscut.append(np.min(model.masscoord))

    plt.ylabel(ylabel, fontsize=15)
    plt.xlabel('Mass coordinate M$_{\odot}$', fontsize=15)

    plt.legend(loc='upper right')
    plt.xlim(np.min(masscut), 9)
    plt.ylim(-10, 15)
    plt.tick_params(left=True, right=True, top=True, labelleft=True, which='both')  # ,labelright=True)

    plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
    plt.gcf().subplots_adjust(left=0.25)

    return plt