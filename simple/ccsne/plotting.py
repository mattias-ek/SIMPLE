import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import simple


class EndlessList:
    # Index will never go out of bounds. It will just start from the beginning if larger than the inital list.
    def __init__(self, items):
        self.list = list(items)

    def __getitem__(self, index):
        return self.list[index % len(self.list)]

colours=EndlessList(["red", "blue", "green", "orange", "black", "blueviolet", "darkgoldenrod", "mediumvioletred"])
linestyles = EndlessList(['-', '--', ':', '-.'])
markers = EndlessList(["o", "s", "^", "D", "P"])

def plot_slopes(models, ratio, where=None, where_kwargs={}):
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