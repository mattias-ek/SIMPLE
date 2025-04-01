from simple import utils, models, plotting
import numpy as np
import h5py
import re
from nugridpy import nugridse as mp
import logging

logger = logging.getLogger('SIMPLE.ccsne')

__all__ = ['plot_ccsne', 'mhist_ccsne', 'mcontour_ccsne']

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

def calc_default_onion_structure(abundance, keys, masscoord):
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

    shells = 'H He/N He/C O/C O/Ne O/Si Si Ni'.split()
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

    return utils.askeyarray(boundaries, shells, dtype=np.int64)


##############
### Models ###
##############

class CCSNe(models.ModelTemplate):
    """
    Model specifically for CCSNe yields and their mass coordinates.

    Attributes:
        type (str): The type of data stored in the model. **Required at initialisation**
        citation (str): A citation for the data. **Required at initialisation**
        mass (): The initial mass of the CCSNe modelled. **Required at initialisation**
        masscoord (): The mass coordinates of the yields. **Required at initialisation**
        abundance (): A key array containing the isotope yields. Is created upon model initiation from the
        ``abundance_values`` and ``abundance_keys`` attributes.
        abundance_values (): A 2dim array containing the isotope yields. **Required at initialisation**
        abundance_keys (): The isotope key for each column in ``abundance_values``. **Required at initialisation**
        abundance_unit (): Unit for the yields. Should typically be either ``mol`` or ``mass`` for molar and mass
            fractions respectively. **Required at initialisation**
        refid_isoabu (str): Name of the reference model containing the reference isotope abundances
            used for normalisations. **Required at initialisation**
        refid_isomass (str): Name of the reference model containing the reference isotope masses
            used for normalisations. **Required at initialisation**
    """
    REQUIRED_ATTRS = ['type', 'dataset', 'citation', 'mass', 'masscoord',
                      'abundance_values', 'abundance_keys', 'abundance_unit',
                      'refid_isoabu', 'refid_isomass']
    REPR_ATTRS = ['name', 'type', 'dataset', 'mass']
    ABUNDANCE_KEYARRAY = 'abundance'
    masscoord_label = 'Mass Coordinate [solar masses]'
    masscoord_label_latex = 'Mass Coordinate [M${}_{\\odot}$]'

    def get_mask(self, mask, shape = None, **mask_attrs):
        """
        Returns a selection mask for an array with ``shape``.

        This function is used by plotting functions to plot only a sub selction of the data. The mask string
        can an integer representing an index, a slice or a condition that generates a mask. Use ``&`` or ``|``
        to combine multiple indexes and/or conditions.

        Supplied attributes can be accesed by putting a dot infront of the name, e.g. ``.data > 1``. The available
        operators for mask conditions are ``==``, ``!=``, ``>``, ``>=``, ``<``, ``<=``.

        The result of the mask evaluation must be broadcastable with ``shape``. If it is not an all ``False`` mask is
        returned.

        Masks for the different onion layers are included as attributes.

        **Note**
        - It is not possible to mix ``&`` and ``|`` seperators. Doing so will raise an exception.
        - Any text not precceded by a dot will be evaluated as text. Text on its own will always be evaluated
        as ``False``.
        - An empty string will be evaluated as ``True``


        Args:
            mask (): String or object that will be evaluated to create a mask.
            shape (): Shape of the returned mask. If omitted the shape of the default abundance array is used.
            **mask_attrs (): Attributes to be used during the evaluation.

        Examples:
            >>> a = np.array([0,1,2,3,4])
            >>> model.get_mask('3', a.shape)
            array([False, False, False,  True,  False])

            >>> model.get_mask('1:3', a.shape)
            array([False, True, True,  False,  False])

            >>> model.get_mask('.data >= 1 & .data < 3', a.shape, data=a)
            array([False, True, True,  False,  False])

            >>> model.get_mask('.data >= 1 | .data > 3', a.shape, data=a)
            rray([True, True, False,  False,  True])

        Returns:
            A boolean numpy array with ``shape``.
        """
        onion_lbounds = getattr(self, 'onion_lbounds', None)
        shell_a = np.full(self.abundance.shape, 'undefined')
        if onion_lbounds is not None:
            shell_d = {}
            keys = onion_lbounds.dtype.names
            ubound = None
            for key in keys:
                lbound = int(onion_lbounds[key][0])
                if lbound >= 0:
                    i = slice(lbound, ubound)
                    shell_d[key] = i
                    shell_a[i] = key
                    ubound = lbound
                else:
                    shell_d[key] = slice(0, 0)

                # Remnant is everything inside the lowermost shell.
                i = slice(None, ubound)
                shell_d['Mrem'] = i
                shell_a[i] = 'Mrem'

            shell_d.update(mask_attrs)
            mask_attrs = shell_d

        return super().get_mask(mask, shape, shell = shell_a, **mask_attrs)

def load_Ri18(fol2mod, ref_isoabu, ref_isomass):
    def load(emass, modelname, default_onion_structure=True):
        pt_exp = mp.se(fol2mod, modelname, rewrite=True)
        cyc = pt_exp.se.cycles[-1]
        t9_cyc = pt_exp.se.get(cyc, 'temperature')
        mass = pt_exp.se.get(cyc, 'mass')

        ejected = np.where(np.array(t9_cyc) > 1.1e-9)[0][0]

        masscoord = pt_exp.se.get(cyc, 'mass')[ejected:]
        abu = np.array(pt_exp.se.get(cyc, 'iso_massf'))[ejected:]
        unit = 'mass'
        keys = utils.asisotopes(pt_exp.se.isotopes, allow_invalid=True)

        data = dict(type='CCSNe', dataset=dataset, citation=citation,
                    refid_isoabu=ref_isoabu, refid_isomass=ref_isomass,
                    mass=int(emass), masscoord=masscoord,
                    abundance_values=abu, abundance_keys=keys,
                    abundance_unit=unit)

        if default_onion_structure:
            data['onion_lbounds'] = calc_default_onion_structure(abu, keys, masscoord)

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

def load_Pi16(fol2mod, ref_isoabu, ref_isomass):
    def load(emass, modelname, default_onion_structure=True):
        pt_exp = mp.se(fol2mod, modelname, rewrite=True)
        cyc = pt_exp.se.cycles[-1]
        t9_cyc = pt_exp.se.get(cyc, 'temperature')
        mass = pt_exp.se.get(cyc, 'mass')

        ejected = np.where(t9_cyc < 1.1e-10)[0][0]

        masscoord = pt_exp.se.get(cyc, 'mass')[ejected:]
        abu = pt_exp.se.get(cyc, 'iso_massf')[ejected:]
        unit = 'mass'
        keys = utils.asisotopes(pt_exp.se.isotopes, allow_invalid=True)


        data = dict(type='CCSNe', dataset=dataset, citation=citation,
                    refid_isoabu=ref_isoabu, refid_isomass=ref_isomass,
                    mass=int(emass), masscoord=masscoord,
                    abundance_values=abu, abundance_keys=keys,
                    abundance_unit=unit)

        if default_onion_structure:
            data['onion_lbounds'] = calc_default_onion_structure(abu, keys, masscoord)

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

def load_La22(data_dir, ref_isoabu, ref_isomass):
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

        data = dict(type='CCSNe', dataset=dataset, citation=citation,
                    refid_isoabu=ref_isoabu, refid_isomass=ref_isomass,
                    mass=int(emass), masscoord=masscoord,
                    abundance_values=abu, abundance_keys=keys,
                    abundance_unit=unit)

        if default_onion_structure:
            data['onion_lbounds'] = calc_default_onion_structure(abu, keys, masscoord)

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

def load_Si18(data_dir, ref_isoabu, ref_isomass, decayed=False):
    def load(emass, file_sie, default_onion_structure=True):
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

        data = dict(type='CCSNe', dataset=dataset, citation=citation,
                    refid_isoabu=ref_isoabu, refid_isomass=ref_isomass,
                    mass=int(emass), masscoord=masscoord,
                    abundance_values=abu, abundance_keys=keys,
                    abundance_unit=unit)

        if default_onion_structure:
            data['onion_lbounds'] = calc_default_onion_structure(abu, keys, masscoord)

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

def load_Ra02(data_dir, ref_isoabu, ref_isomass):
    def load(emass, model_name, default_onion_structure=True):
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

            data = dict(type='CCSNe', dataset=dataset, citation=citation,
                        refid_isoabu=ref_isoabu, refid_isomass=ref_isomass,
                        mass=int(emass), masscoord=masscoord,
                        abundance_values=abu, abundance_keys=keys,
                        abundance_unit=unit)

            if default_onion_structure:
                data['onion_lbounds'] = calc_default_onion_structure(abu, keys, masscoord)

            models[f"{dataset}_m{emass}"] = data
            return data

    dataset = 'Ra02'
    citation = ''
    models = {}

    # 15
    load('15', 's15a28c.expl_yield')

    # 20
    load('20', 's20a28n.expl_yield')

    # 25
    load('25', 's25a28d.expl_yield')

    return models

def load_LC18(data_dir, ref_isoabu, ref_isomass):
    def load(emass, model_name, default_onion_structure=True):
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

            data = dict(type='CCSNe', dataset=dataset, citation=citation,
                                                 refid_isoabu=ref_isoabu, refid_isomass=ref_isomass,
                                                 mass=int(emass), masscoord=masscoord,
                                                 abundance_values=abu, abundance_keys=keys,
                                                 abundance_unit=unit)

            if default_onion_structure:
                data['onion_lbounds'] = calc_default_onion_structure(abu, keys, masscoord)

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
# TODO just create a onionshell attr.
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
def plot_onion_structure(model, *, ax=None, update_ax=True, update_fig=True, **kwargs):
    if not isinstance(model, models.ModelTemplate):
        raise ValueError(f'model must be an Model object not {type(model)}')

    ax = plotting.get_axes(ax)
    title = ax.get_title()
    if title:
        kwargs.setdefault('ax_title', title)
    delayed_kwargs = plotting.update_axes(ax, kwargs, delay='ax_legend', update_ax=update_ax, update_fig=update_fig)

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

    plotting.update_axes(ax, delayed_kwargs, update_ax=update_ax, update_fig=update_fig)


@utils.add_shortcut('abundance', default_attrname='abundance', unit='mass')
@utils.add_shortcut('intnorm', default_attrname='intnorm.eRi', unit=None)
@utils.add_shortcut('stdnorm', default_attrname='stdnorm.Ri', unit=None)
@utils.set_default_kwargs(
    linestyle = True, color=True, marker=False,
    fig_size= (10,5))
def plot_ccsne(models, ykey, *,
         semilog = False, onion=None,
         **kwargs):
    """
    Plot for CCSNe models. Plots the mass coordinates on the x-axis.
    """
    # Wrapper that adds the option to plot the onion structure of CCSNe models
    onion_kwargs = utils.extract_kwargs(kwargs, prefix='onion')

    # Do this here since we need to know the number of models for the onion shell
    where = kwargs.pop('where', None)
    where_kwargs = kwargs.pop('where_kwargs', {})
    where_kwargs.update(utils.extract_kwargs(kwargs, prefix='where'))
    models = plotting.get_models(models, where=where, where_kwargs=where_kwargs)

    if semilog: kwargs.setdefault('ax_yscale', 'log')

    ax = plotting.plot(models, '.masscoord', ykey, xunit=None, **kwargs)

    if onion or (onion is None and len(models) == 1):
        if len(models) > 1:
            raise ValueError(f"Can only plot onion structure for a single model")
        else:
            plot_onion_structure(models[0], ax=ax, **onion_kwargs)

    return ax

def _mweights(models, modeldata_w):
    logger.info('Multiplying all weights by the mass coordinate mass')
    modeldata_m, axis_labels_m = plotting.get_data(models, {'m': '.masscoord'},
                                                       default_value=np.nan,
                                                       latex_labels=False)

    for model_name, model_data_w in modeldata_w.items():
        masscoord_mass = modeldata_m[model_name][0]['m']

        # Temporary as the mass coord mass doesnt exist yet
        masscoord_mass = masscoord_mass[1:] - masscoord_mass[:1]
        masscoord_mass = np.insert(masscoord_mass, 0, masscoord_mass[0])
        for ki, data_w in enumerate(model_data_w):
            data_w['w'] *= masscoord_mass

    return modeldata_w

@utils.add_shortcut('abundance', default_attrname='abundance', unit='mass', xunit=None)
@utils.add_shortcut('intnorm', default_attrname='intnorm.eRi', unit=None, xunit=None)
@utils.add_shortcut('stdnorm', default_attrname='stdnorm.Ri', unit=None, xunit=None)
@utils.set_default_kwargs(
    weights_default_attrname='abundance', weights_unit='mass',
)
def mhist_ccsne(models, xkey, ykey, r=None, weights=1, **kwargs):
    """
    Histogram plot on a rose diagram for CCNSe models.
    """
    kwargs_ = plotting.mhist.default_kwargs.copy()
    kwargs_.update(kwargs)

    ax, models, r, modeldata_xy, modeldata_w, kwargs = plotting._mprep(models, xkey, ykey, r, weights, **kwargs_)
    modeldata_w = _mweights(models, modeldata_w)
    return plotting._mhist(ax, r, modeldata_xy, modeldata_w, **kwargs)

@utils.add_shortcut('abundance', default_attrname='abundance', unit='mass', xunit=None)
@utils.add_shortcut('intnorm', default_attrname='intnorm.eRi', unit=None, xunit=None)
@utils.add_shortcut('stdnorm', default_attrname='stdnorm.Ri', unit=None, xunit=None)
@utils.set_default_kwargs(
    weights_default_attrname='abundance', weights_unit='mass',
)
def mcontour_ccsne(models, xkey, ykey, r=None, weights=1, **kwargs):
    """
    Contour plot on a rose diagram for CCNSe models.
    """
    kwargs_ = plotting.mcontour.default_kwargs.copy()
    kwargs_.update(kwargs)

    ax, models, r, modeldata_xy, modeldata_w, kwargs = plotting._mprep(models, xkey, ykey, r, weights, **kwargs_)
    modeldata_w = _mweights(models, modeldata_w)
    return plotting._mcontour(ax, r, modeldata_xy, modeldata_w, **kwargs)

########################
### Deprecated stuff ###
########################
@utils.deprecation_warning('``ccsne.plot_abundance`` has been deprecated: Use ``plot_ccsne.abundance`` instead')
def plot_abundance(*args, **kwargs):
    return plot_ccsne.abundance(*args, **kwargs)

@utils.deprecation_warning('``ccsne.plot_intnorm`` has been deprecated: Use ``plot_ccsne.intnorm`` instead')
def plot_intnorm(*args, **kwargs):
    return plot_ccsne.intnorm(*args, **kwargs)

@utils.deprecation_warning('``ccsne.plot_simplenorm`` has been deprecated: Use ``plot_ccsne.stdnorm`` instead')
def plot_simplenorm(*args, **kwargs):
    return plot_ccsne.stdnorm(*args, **kwargs)
