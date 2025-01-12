from simple import utils, models
import numpy as np
import h5py
import re
from nugridpy import nugridse as mp
import logging

logger = logging.getLogger('SIMPLE.CCSNe.models')

__all__ = []

#############
### Utils ###
#############
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

    # TODO This code works for most but not all of the 18 models in the original release
    # So use with caution

    # definition of borders
    ih = np.where((he4 > 0.5))[0][-1]
    logging.info("Lower boundary of the H shell: " + str(mass[ih]))
    boundaries.append(ih)

    ihe1 = np.where((n14 > o16) & (n14 > c12) & (n14 > 1.e-3))[0][0]
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
        isi = np.where((ni56 > si28))[0][-1]
    except IndexError:
        logging.info("No lower boundary of Si layer")
        boundaries.append(-1)
    else:
        logging.info("Lower boundary of the Si layer: " + str(mass[isi]))
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


###################
### Load models ###
###################

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
        if onion_lbounds is not None:
            shells = {}
            keys = onion_lbounds.dtype.names
            ubound = None
            for key in keys:
                lbound = int(onion_lbounds[key][0])
                if lbound >= 0:
                    shells[key] = slice(lbound, ubound)
                    ubound = lbound
                else:
                    shells[key] = slice(0, 0)

            shells.update(mask_attrs)
            mask_attrs = shells

        return super().get_mask(mask, shape, **mask_attrs)



z_names = ['Neut', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al',
           'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co',
           'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb',
           'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs',
           'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm',
           'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
           'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U']


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

def load_LC18(data_dir, ref_isoabu, ref_isomass, divide_by_isomass = True):
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
