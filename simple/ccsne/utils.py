import simple
import numpy as np
import logging

logger = logging.getLogger('SIMPLE.CCSNe.utils')

def get_onion_structure(model):
    """
    Return a key array with the lower boundaries of the H, He/N, He/C, O/C, O/Ne, O/Si, Si and Ni shells/layers.
    """
    logging.info(f'Calculating the onion structure for: {model.name}')
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
    logging.info("m_cut: " + str(masscut))
    logging.info("massmax: " + str(massmax))

    shells = 'H He/N He/C O/C O/Ne O/Si Si Ni'.split()
    boundaries = []

    # definition of borders
    ih = np.where((he4 > 0.5))[0][-1]
    logging.info("Lower boundary of the H shell: " + str(mass[ih]))
    boundaries.append(mass[ih])

    ihe1 = np.where((n14 > o16) & (n14 > c12) & (n14 > 1.e-3))[0][0]
    logging.info("Lower boundary of the He/N shell: " + str(mass[ihe1]))
    boundaries.append(mass[ihe1])

    ihe = np.where((c12 > he4) & (mass <= mass[ih]))[0][-1]
    logging.info("Lower boundary of the He/C shell: " + str(mass[ihe]))
    boundaries.append(mass[ihe])

    ic2 = np.where((c12 > ne20) & (si28 < c12) & (c12 > 8.e-2))[0][0]
    logging.info("Lower boundary of the O/C shell: " + str(mass[ic2]))
    boundaries.append(mass[ic2])

    ine = np.where((ne20 > 1.e-3) & (si28 < ne20) & (ne20 > c12))[0][0]
    if ine > ic2:
        ine = ic2
    logging.info("Lower boundary of the O/Ne shell: " + str(mass[ine]))
    boundaries.append(mass[ine])

    io = np.where((si28 < o16) & (o16 > 5.e-3))[0][0]
    logging.info("Lower boundary of the O/Si layer: " + str(mass[io]))
    boundaries.append(mass[io])

    try:
        isi = np.where((ni56 > si28))[0][-1]
    except IndexError:
        logging.info("No lower boundary of Si layer")
        boundaries.append(np.nan)
    else:
        logging.info("Lower boundary of the Si layer: " + str(mass[isi]))
        boundaries.append(mass[isi])

    try:
        ini = np.where((ni56 > si28) & (mass > masscut))[0][0]
    except IndexError:
        logging.info("No lower boundary of Ni layer")
        boundaries.append(np.nan)
    else:
        logging.info("Lower boundary of the Ni layer: " + str(mass[ini]))
        boundaries.append(mass[ini])

    return simple.askeyarray(boundaries, shells)

