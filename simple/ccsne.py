from simple import utils, models, plotting
import numpy as np
import h5py
import re
import logging

logger = logging.getLogger('SIMPLE.ccsne')

__all__ = ['plot_ccsne', 'hist_ccsne', 'add_weights_ccsne']

ccsne_zones = ('Mrem', 'Ni', 'Si', 'O/Si', 'O/Ne', 'O/C', 'He/C', 'He/N', 'H')
"""The different shells of the onion structure going outwards."""

#############
### Utils ###
#############
z_names = ['Neut', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al',
           'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co',
           'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb',
           'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs',
           'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm',
           'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
           'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U']
"""The element symbol for each atomic number (z) up to U"""

##############
### Models ###
##############

class CCSNe(models.ModelBase):
    """
    Model for CCSNe yields and their mass coordinates.
    """
    REQUIRED_ATTRS = ['type', 'dataset', 'citation', 'mass', 'masscoord', 'masscoord_mass',
                      'abundance_values', 'abundance_keys', 'abundance_unit',
                      'refid_isoabu', 'refid_isomass']
    REPR_ATTRS = ['name', 'type', 'dataset', 'mass']
    ABUNDANCE_KEYARRAY = 'abundance'
    masscoord_label = 'Mass Coordinate [solar masses]'
    masscoord_label_latex = 'Mass Coordinate [M${}_{\\odot}$]'
    masscoord_mass_label = 'Coordinate Mass [solar masses]'
    masscoord_mass_label_latex = 'Coordinate Mass [M${}_{\\odot}$]'


# These functions cannot be tested without the original data so there
# are no automatic tests, and they are ignored in coverage.
# They should therefore be used with caution!
def fudge_masscord_mass(masscoord): # pragma: no cover
    """
    Estimate the mass at each mass coordinate.

    The mass associated with a coordinate is approximated by the difference
    between successive coordinates. The last value is duplicated so the output
    has the same length as the input array.
    """
    logging.info('Fudging the masscoord mass from masscoord')
    masscoord = np.asarray(masscoord)
    masscoord = masscoord[1:] - masscoord[:1]
    return np.append(masscoord, masscoord[-1])

def calc_default_onion_structure(abundance, keys, masscoord): # pragma: no cover
    """
    Calculated the boundaries of different layers within the CCSNe onion structure.

    **Note** This function is calibrated for the initial set of CCSNe models and might not be applicable to
    other models.

    The returned array contains the index of the lower bound of the given layer. If a layer is not found the index is
    given as np.nan.
    """
    abundance = np.asarray(abundance)
    masscoord = np.asarray(masscoord)
    mass = masscoord
    he4 = abundance[:, keys.index('He-4')]
    c12 = abundance[:, keys.index('C-12')]
    ne20 = abundance[:, keys.index('Ne-20')]
    o16 = abundance[:, keys.index('O-16')]
    si28 = abundance[:, keys.index('Si-28')]
    n14 = abundance[:, keys.index('N-14')]
    ni56 = abundance[:, keys.index('Ni-56')]

    masscut = mass[0]
    massmax = mass[-1]
    logging.info('Calculating Default Onion Structure')
    logging.info("m_cut: " + str(masscut))
    logging.info("massmax: " + str(massmax))

    zones = 'H He/N He/C O/C O/Ne O/Si Si Ni'.split()
    boundaries = []

    # This code works for most but not all of the 18 models in the original release
    # So use with caution

    # definition of borders
    ih = np.where((he4 > 0.5))[0][-1]
    logging.info("Lower boundary of the H shell: " + str(mass[ih]))
    boundaries.append(ih)

    ihe1 = np.where((n14 > o16) & (n14 > c12) & (n14 > 1.e-3))[0][0]
    ihe_check = np.searchsorted(mass, mass[ihe1] + 0.005)
    if ihe_check < len(mass) and not (
            (n14[ihe_check] > o16[ihe_check]) and (n14[ihe_check] > c12[ihe_check]) and (n14[ihe_check] > 1.e-3)):
        ihe1 = np.where((n14 > o16) & (n14 > c12) & (n14 > 1.e-3) & (mass >= mass[ihe1] + 0.005))[0][0]
    logging.info("Lower boundary of the He/N shell: " + str(mass[ihe1]))
    boundaries.append(ihe1)

    ihe = np.where((c12 > he4) & (mass <= mass[ih]))[0][-1]
    logging.info("Lower boundary of the He/C shell: " + str(mass[ihe]))
    boundaries.append(ihe)

    ic2 = np.where((c12 > ne20) & (si28 < c12) & (c12 > 8.e-2))[0][0]
    logging.info("Lower boundary of the O/C shell: " + str(mass[ic2]))
    boundaries.append(ic2)

    ine = np.where((ne20 > 1.e-3) & (si28 < ne20) & (ne20 > c12))[0][0]
    if ine > ic2:
        ine = ic2
    logging.info("Lower boundary of the O/Ne shell: " + str(mass[ine]))
    boundaries.append(ine)

    io = np.where((si28 < o16) & (o16 > 5.e-3))[0][0]
    logging.info("Lower boundary of the O/Si layer: " + str(mass[io]))
    boundaries.append(io)

    try:
        indices = np.where(ni56 > si28)[0]
        if indices.size > 0:
            isi = indices[-1]
        else:
            if ni56[1] < si28[1] and si28[1] > o16[1]:
                isi = 0
            else:
                raise IndexError("No suitable boundary found")
    except IndexError:
        logging.info("No lower boundary of Si layer")
        boundaries.append(-1)
    else:
        if len(mass[isi:io]) < 2:
            boundaries.append(-1)
            logging.info("No lower boundary of Si layer")
        else:
            logging.info(f"Lower boundary of the Si layer: {mass[isi]}")
            boundaries.append(isi)

    try:
        ini = np.where((ni56 > si28))[0][0]
    except IndexError:
        logging.info("No lower boundary of Ni layer")
        boundaries.append(-1)
    else:
        logging.info("Lower boundary of the Ni layer: " + str(mass[ini]))
        boundaries.append(ini)

    onion_lbounds =  utils.askeyarray(boundaries, zones, dtype=np.int64)

    zone = np.full(masscoord.shape, 'Undefined')
    keys = onion_lbounds.dtype.names
    ubound = None
    for key in keys:
        lbound = int(onion_lbounds[key][0])
        if lbound >= 0:
            zone[slice(lbound, ubound)] = key
            ubound = lbound

        # Remnant is everything inside the lowermost shell.
        zone[slice(None, ubound)] = 'Mrem'

    return onion_lbounds, zone

def load_Ri18(fol2mod, ref_isoabu, ref_isomass): # pragma: no cover
    """Load the CCSNe models from Ritter et al. (2018)."""
    from nugridpy import nugridse as mp
    def load(emass, modelname):
        pt_exp = mp.se(fol2mod, modelname, rewrite=True)
        cyc = pt_exp.se.cycles[-1]
        t9_cyc = pt_exp.se.get(cyc, 'temperature')
        mass = pt_exp.se.get(cyc, 'mass')

        ejected = np.where(np.array(t9_cyc) > 1.1e-9)[0][0]

        masscoord = pt_exp.se.get(cyc, 'mass')[ejected:]
        abu = np.array(pt_exp.se.get(cyc, 'iso_massf'))[ejected:]
        unit = 'mass'
        keys = utils.asisotopes(pt_exp.se.isotopes, allow_invalid=True)

        masscoord_mass = fudge_masscord_mass(masscoord)

        data = dict(type='CCSNe', dataset=dataset, citation=citation,
                    refid_isoabu=ref_isoabu, refid_isomass=ref_isomass,
                    mass=int(emass), masscoord=np.asarray(masscoord),
                    masscoord_mass = np.asarray(masscoord_mass),
                    abundance_values=np.asarray(abu), abundance_keys=keys,
                    abundance_unit=unit)

        data['onion_lbounds'], data['zone'] = calc_default_onion_structure(abu, keys, masscoord)

        models[f'{dataset}_m{emass}'] = data
        return data

    dataset = 'Ri18'
    citation = ''
    models = {}

    # 15Msun
    load('15', 'M15.0Z2.0e-02.Ma.0020601.out.h5')

    # 20Msun
    load('20', 'M20.0Z2.0e-02.Ma.0021101.out.h5')

    # 25Msun
    load('25', 'M25.0Z2.0e-02.Ma.0023601.out.h5')

    return models

def load_Pi16(fol2mod, ref_isoabu, ref_isomass): # pragma: no cover
    """Load the CCSNe models from Pignatari et al. (2016)."""
    from nugridpy import nugridse as mp
    def load(emass, modelname):
        pt_exp = mp.se(fol2mod, modelname, rewrite=True)
        cyc = pt_exp.se.cycles[-1]
        t9_cyc = pt_exp.se.get(cyc, 'temperature')
        mass = pt_exp.se.get(cyc, 'mass')

        ejected = np.where(t9_cyc < 1.1e-10)[0][0]

        masscoord = pt_exp.se.get(cyc, 'mass')[ejected:]
        abu = pt_exp.se.get(cyc, 'iso_massf')[ejected:]
        unit = 'mass'
        keys = utils.asisotopes(pt_exp.se.isotopes, allow_invalid=True)

        masscoord_mass = fudge_masscord_mass(masscoord)

        data = dict(type='CCSNe', dataset=dataset, citation=citation,
                    refid_isoabu=ref_isoabu, refid_isomass=ref_isomass,
                    mass=int(emass), masscoord=np.asarray(masscoord),
                    masscoord_mass=np.asarray(masscoord_mass),
                    abundance_values=np.asarray(abu), abundance_keys=keys,
                    abundance_unit=unit)

        data['onion_lbounds'], data['zone'] = calc_default_onion_structure(abu, keys, masscoord)

        models[f'{dataset}_m{emass}'] = data
        return data

    dataset = 'Pi16'
    citation = ''
    models = {}

    # 15Msun
    load('15', 'M15.0')

    # 20Msun
    load('20', 'M20.0')

    # 25Msun
    load('25', 'M25.0')

    return models

def load_La22(data_dir, ref_isoabu, ref_isomass): # pragma: no cover
    """Load the CCSNe models from Lawson et al. (2022)."""
    def load(emass, model_name, default_onion_structure=True):
        mass_lines = []
        with open(data_dir + model_name, "rt") as f:
            for ln, line in enumerate(f):
                if 'mass enclosed' in line:
                    mass_lines.append(line)
        mass = [float(row.split()[3]) for row in mass_lines]
        numpart = [int(row.split()[0][1:]) for row in mass_lines]
        number_of_parts = len(numpart)  # number of particles (it may change from model1 to model1)
        # print('# particles = ',number_of_parts)

        # open and read abundances for all trajectories
        a, x, z, iso_name = [], [], [], []
        with open(data_dir + model_name, "rt") as f:
            i = 0
            while i < number_of_parts:
                f.readline();
                f.readline();
                j = 0
                a_i, x_i, z_i, iso_i = [], [], [], []
                while j < num_species:
                    line = f.readline().split()
                    a_i.append(int(line[0]))
                    z_i.append(int(line[1]))
                    x_i.append(float(line[2]))
                    iso_i.append(f"{z_names[int(line[1])]}-{line[0]}")
                    j += 1
                a.append(a_i);
                z.append(z_i);
                x.append(x_i);
                iso_name.append(iso_i)
                i += 1

                # Assumes all trajectories have the same isotope list but not necessarily ordered the same
        y = {}
        for i in range(number_of_parts):
            for j, iso in enumerate(iso_name[i]):
                y.setdefault(iso, [])
                y[iso].append(x[i][j])
        dum_ab = np.array([list(v) for v in y.values()])

        # If iso is identical for all trajectories this is another way to do it
        # y = np.transpose([[x[i][j] for j in range(len(iso[i])] for i in range(number_of_parts)])

        masscoord = mass
        keys = utils.asisotopes(y.keys(), allow_invalid=True)
        abu = np.transpose([list(v) for v in y.values()])
        unit = 'mass'

        masscoord_mass = fudge_masscord_mass(masscoord)

        data = dict(type='CCSNe', dataset=dataset, citation=citation,
                    refid_isoabu=ref_isoabu, refid_isomass=ref_isomass,
                    mass=int(emass), masscoord=np.asarray(masscoord),
                    masscoord_mass=np.asarray(masscoord_mass),
                    abundance_values=np.asarray(abu), abundance_keys=keys,
                    abundance_unit=unit)

        if default_onion_structure:
            data['onion_lbounds'], data['zone'] = calc_default_onion_structure(abu, keys, masscoord)

        models[f'{dataset}_m{emass}'] = data
        return data

    num_species = 5209
    dataset = 'La22'
    citation = ''
    models = {}


    # 15
    load('15','M15s_run15f1_216M1.3bgl_mp.txt')

    # 20
    load('20', 'M20s_run20f1_300M1.56jl_mp.txt')

    # 25
    load('25', 'M25s_run25f1_280M1.83rrl_mp.txt')

    return models

def load_Si18(data_dir, ref_isoabu, ref_isomass, decayed=False): # pragma: no cover
    """Load the CCSNe models from Sieverding et al. (2018)."""
    def load(emass, file_sie):
        with h5py.File(data_dir + file_sie) as data_file:
            data = data_file["post-sn"]

            # need to decode binary isotope names to get strings
            iso_list_sie = [name.decode() for name in data["isotopes"]]
            mr = list(data["mass_coordinates_sun"])

            results = {}
            for jiso, iso in enumerate(iso_list_sie):
                if decayed:
                    results[iso] = data["mass_fractions_decayed"][:, jiso]
                else:
                    results[iso] = data["mass_fractions"][:, jiso]

        masscoord = np.array(mr)
        keys = utils.asisotopes(results.keys(), allow_invalid=True)
        abu = np.transpose([list(v) for v in results.values()])
        unit = 'mass'

        masscoord_mass = fudge_masscord_mass(masscoord)

        data = dict(type='CCSNe', dataset=dataset, citation=citation,
                    refid_isoabu=ref_isoabu, refid_isomass=ref_isomass,
                    mass=int(emass), masscoord=np.asarray(masscoord),
                    masscoord_mass=np.asarray(masscoord_mass),
                    abundance_values=np.asarray(abu), abundance_keys=keys,
                    abundance_unit=unit)

        data['onion_lbounds'], data['zone'] = calc_default_onion_structure(abu, keys, masscoord)

        models[f'{dataset}_m{emass}'] = data
        return data

    dataset = 'Si18'
    citation = ''
    models = {}

    # 15
    load('15', "s15_data.hdf5")

    # 20
    load('20', "s20_data.hdf5")

    # 25
    load('25', "s25_data.hdf5")

    return models

def load_Ra02(data_dir, ref_isoabu, ref_isomass): # pragma: no cover
    """Load the CCSNe models from Rauscher et al. (2002)."""
    def load(emass, model_name):
        filename = data_dir + model_name
        # print(filename)
        with open(filename, 'r') as f:
            head = f.readline();
            isos_dum = head.split()[5:]  # getting isotopes, not first header names
            dum_a = [re.findall('\d+', ik)[0] for ik in isos_dum]  # getting the A from isotope prefixes
            dum_el = [re.sub(r'[0-9]+', '', ik) for ik in
                      isos_dum]  # getting the element prefixes from the isotope prefixes
            dum_new_iso = [dum_el[ik].capitalize() + '-' + dum_a[ik] for ik in range(len(isos_dum))]

            # isotope prefixes that we can use around, just neutron prefixes is different, but not care
            keys = utils.asisotopes(dum_new_iso, allow_invalid=True)

            data = f.readlines()[:-2]  # getting the all item, excepting the last two lines
            # rau_mass.append(dum) # converting in Msun too.
            abu = np.asarray([row.split()[3:] for row in data], np.float64)
            unit = 'mass'

            masscoord = np.array([float(ii.split()[1]) / 1.989e+33 for ii in data])
            masscoord_mass = np.array([float(ii.split()[2]) / 1.989e+33 for ii in data])
            #masscoord_mass = fudge_masscord_mass(masscoord)

            data = dict(type='CCSNe', dataset=dataset, citation=citation,
                        refid_isoabu=ref_isoabu, refid_isomass=ref_isomass,
                        mass=int(emass), masscoord=np.asarray(masscoord),
                        masscoord_mass=np.asarray(masscoord_mass),
                        abundance_values=np.asarray(abu), abundance_keys=keys,
                        abundance_unit=unit)

            data['onion_lbounds'], data['zone'] = calc_default_onion_structure(abu, keys, masscoord)

            models[f"{dataset}_m{emass}"] = data
            return data

    dataset = 'Ra02'
    citation = ''
    models = {}

    # 15
    load('15', 's15a28c.expl_comp')

    # 20
    load('20', 's20a28n.expl_comp')

    # 25
    load('25', 's25a28d.expl_comp')

    return models

def load_LC18(data_dir, ref_isoabu, ref_isomass): # pragma: no cover
    """Load the CCSNe models from Limongi & Chieffi (2018)."""
    def load(emass, model_name):
        filename = data_dir + model_name
        # print(filename)
        with open(filename, 'r') as f:
            # getting isotopes, not first header names, and final ye and spooky abundances (group of isolated isotopes,
            # probably sorted with artificial reactions handling mass conservation or sink particles approach)
            head = f.readline();
            isos_dum = head.split()[4:-skip_heavy_]
            # correcting names to get H1 (and the crazy P and A)
            isos_dum[0] = isos_dum[0] + '1';
            isos_dum[1] = isos_dum[1] + '1';
            isos_dum[6] = isos_dum[6] + '1'
            dum_a = [re.findall('\d+', ik)[0] for ik in isos_dum]  # getting the A from isotope prefixes
            dum_el = [re.sub(r'[0-9]+', '', ik) for ik in
                      isos_dum]  # getting the element prefixes from the isotope prefixes
            dum_new_iso = [dum_el[ik].capitalize() + '-' + dum_a[ik] for ik in range(len(isos_dum))]

            data = f.readlines()[:-1]  # getting the all item, excepting the last fake line (bounch of zeros)

            masscoord = np.array([float(ii.split()[0]) for ii in data])

            # isotope prefixes that we can use around, just neutron prefixes is different, but not care
            keys = utils.asisotopes(dum_new_iso, allow_invalid=True)

            # done reading, just closing the file now
            # converting in Msun too.
            abu = np.asarray([row.split()[4:-skip_heavy_] for row in data], dtype=np.float64)
            unit = 'mass'

            masscoord_mass = fudge_masscord_mass(masscoord)

            data = dict(type='CCSNe', dataset=dataset, citation=citation,
                        refid_isoabu=ref_isoabu, refid_isomass=ref_isomass,
                        mass=int(emass), masscoord=np.asarray(masscoord),
                        masscoord_mass=np.asarray(masscoord_mass),
                        abundance_values=np.asarray(abu), abundance_keys=keys,
                        abundance_unit=unit)

            data['onion_lbounds'], data['zone'] = calc_default_onion_structure(abu, keys, masscoord)

            models[f"{dataset}_m{emass}"] = data
            return data

    skip_heavy_ = 43  # used to skip final ye and spooky abundances (see below)
    dataset = 'LC18'
    citation = ''
    models = {}

    # 15
    load('15', '015a000.dif_iso_nod')

    # 20
    load('20', '020a000.dif_iso_nod')

    # 25
    load('25', '025a000.dif_iso_nod')

    return models

################
### Plotting ###
################
# TODO dont plot if smaller than x
@utils.set_default_kwargs(
    # Default settings for line, text and fill
    ax_kw_title_pad = 20,
    default_line_color='black', default_line_linestyle='--', default_line_lw=2, default_line_alpha=0.75,
    default_text_fontsize=10., default_text_color='black',
    default_text_horizontalalignment='center', default_text_xycoords=('data', 'axes fraction'), default_text_y = 1.01,
    default_fill_color='lightblue', default_fill_alpha=0.25,

    # For the rest we only need to give the values that differ from the default
   remnant_line_linestyle=':', remnant_fill_color='gray', remnant_fill_alpha=0.5,
   HeN_fill_show=False,
   OC_fill_show=False,
   OSi_fill_show=False,
   Ni_fill_show=False, )
def plot_zonal_structure(model, *, ax=None, update_ax=True, update_fig=True, kwargs=None):
    """Visualise the onion-shell structure for a single CCSNe model."""
    if not isinstance(model, models.ModelBase):
        raise ValueError(f'model must be an Model object not {type(model)}')

    ax = plotting.get_axes(ax)
    title = ax.get_title()
    if title:
        kwargs.setdefault('ax_title', title)
    plotting.update_axes(ax, kwargs, update_ax=update_ax, update_fig=update_fig)

    lower_bounds = getattr(model, 'onion_lbounds', None)
    if lower_bounds is None:
        logger.error('No zonal structure defined for this model')
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

    default_line = kwargs.pop_many(prefix='default_line')
    default_text = kwargs.pop_many(prefix='default_text')
    default_fill = kwargs.pop_many(prefix='default_fill')

    def add_line(name, x):
        line_kwargs = kwargs.pop_many(prefix=f'{name}_line', **default_line)
        if line_kwargs.pop('show', True):
            ax.axvline(x, **line_kwargs)

    def add_text(name, text, x):
        text_kwargs = kwargs.pop_many(prefix=f'{name}_text', **default_text)
        if text_kwargs.pop('show', True):
            # Using annotate instead of text as we can then specify x in absolute, and y coordinates relative, in space.
            ax.annotate(text_kwargs.pop('xytext', text), (x, text_kwargs.pop('y', 1.01)),
                        **text_kwargs)

    def add_fill(name, x):
        fill_kwargs = kwargs.pop_many(prefix=f'{name}_fill', **default_fill)
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
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

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

    return ax


@utils.set_default_kwargs(inherits_=plotting.add_weights)
def add_weights_ccsne(modeldata, axis, weights = 1, kwargs=None):
    """
    Add weights to the specified axis of each CCSNe datapoint in the modeldata dictionary.

    Before normalisation, if applied, the weight of each datapoint will be multiplied by the
    mass of each mass coordinate.

    This function appends a new array of weights (under `axisname`) to each datapoint
    in `modeldata`. The weights can be a constant or a string referring to data to be indvidually
    retrieved from each model. Optionally, the weights can be summed,
    normalized, and masked for missing data.

    The `mask` and `mask_na` arguments should be the same as those used to generate 'modeldata' to ensure
    conistent results.

    Add weights to CCSNe datapoints by combining standard weighting with mass coordinate scaling.

    This function extends `add_weights` by additionally multiplying the resulting weights
    by the mass associated with each mass coordinate in CCSNe models.

    The initial weighting follows the same logic as `add_weights`, accepting either a scalar
    or a string referring to model attributes. The 'mask' and 'mask_na' arguments should
    match those used when creating `modeldata` to ensure consistency.

    Args:
        modeldata (dict): The data dictionary as returned by `get_data`, structured as
            {model_name: list of datapoints}. Each datapoint is a dictionary.
        axis (str): The axis key to which the weights apply (e.g., 'x', 'y').
        weights (int, float, or str): The weight specification. Can be:
            - A scalar to apply uniformly across all datapoints,
            - A string key to retrieve values from each model individually.
        **kwargs: Any valid keyword arguments for the `add_weights` function.

    Returns:
        dict: The modified `modeldata`, with weight arrays added to each datapoint.
    """

    mask = kwargs.get('mask', None)
    mask_na = kwargs.get('mask_na', True)
    axisname = kwargs.get('axisname', 'w')

    norm_weights = kwargs.pop('norm_weights', True) # Defaults to the value of add_weights
    kwargs['norm_weights'] = False
    modeldata = plotting.add_weights(modeldata, axis, weights, kwargs=kwargs)

    logger.info('Multiplying all weights by the mass coordinate mass')

    for model, datapoints in modeldata.items():
        masscoord_mass = model.masscoord_mass
        if mask:
            imask = model.get_mask(mask)

        for ki, datapoint in enumerate(datapoints):
            if mask and not mask_na:
                datapoint[axisname] = datapoint[axisname] * masscoord_mass[imask]
            else:
                datapoint[axisname] = datapoint[axisname] * masscoord_mass

    if norm_weights:
        plotting._norm_weights(modeldata, axisname)

    return modeldata

@utils.set_default_kwargs(inherits_=plotting.plot,
                                  linestyle=True, marker=False, fig_size=(10,5),
                                  xhist=False)
def plot_ccsne(models, ykey, *,
         semilog = False, onion=None,
         kwargs=None):
    """
    CCSNe implementation of the [`plot`][simple.plot] function where you specify the data on the y-axis which is
    automatically plotted against the mass coordinated on the x-axis. If a single model is shown then by default
    the onion shell structure is also drawn.
    """

    onion_kwargs = kwargs.pop_many(prefix=['onion', 'zone'])
    if semilog: kwargs.setdefault('ax_yscale', 'log')
    kwargs.setdefault('_SIMPLE_add_weights', add_weights_ccsne)

    modeldata, axis_labels = plotting.plot_get_data(models, '.masscoord', ykey,
                                           xunit=None, kwargs=kwargs)

    ax = plotting.plot_draw(modeldata, axis_labels, kwargs=kwargs)

    if onion or (onion is None and len(modeldata) == 1):
        if len(modeldata) > 1:
            raise ValueError(f"Can only plot onion structure for a single model")
        else:
            plot_zonal_structure(list(modeldata.keys())[0], ax=ax, **onion_kwargs)

    return ax

@utils.set_default_kwargs(inherits_=plotting.hist,
                                  weights_default_attrname='abundance', weights_unit='mass',
                                  )
def hist_ccsne(models, xkey=None, ykey=None, weights=1, r=None, kwargs=None):
    """
    CCSNe implementation of [`hist`][simple.hist] where all weights are multiplied by the mass coordinate mass
    (see [`add_ccsne_weights`][simple.add_ccsne_weights]).
    """
    kwargs.setdefault('_SIMPLE_add_weights', add_weights_ccsne)
    return plotting.hist(models, xkey, ykey, weights=weights, r=r, kwargs=kwargs)



