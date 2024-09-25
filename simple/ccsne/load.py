import simple.utils as utils
from nugridpy import nugridse as mp
import numpy as np
import h5py
import re

import logging

logger = logging.getLogger('SIMPLE.CCSNe.load')


###################
### Load models ###
###################
def AGB_test(data_dir):
    raise NotImplementedError()
    # loading AGB models
    # test case, M=3Msun, Z=0.03, Battino et al., getting only the last surf file
    pt_3 = mp.se(data_dir + 'agb_surf_m3z2m3/', '96101.surf.h5', rewrite=True)

def Ri18(data_dir, ref_isoabu, ref_isomass, divide_by_isomass = True):
    # loading Ritter+18 model
    fol2mod = data_dir + 'R18/'
    # load instances of models

    # 15Msun
    pt_15 = mp.se(fol2mod, 'M15.0Z2.0e-02.Ma.0020601.out.h5', rewrite=True)
    cyc_15 = pt_15.se.cycles[-1]
    # pt_15.se.get('temperature')
    t9_cyc_15 = pt_15.se.get(cyc_15, 'temperature')
    mass_15 = pt_15.se.get(cyc_15, 'mass')

    # 20Msun
    pt_20 = mp.se(fol2mod, 'M20.0Z2.0e-02.Ma.0021101.out.h5', rewrite=True)
    cyc_20 = pt_20.se.cycles[-1]
    # pt_20.se.get('temperature')
    t9_cyc_20 = pt_20.se.get(cyc_20, 'temperature')
    mass_20 = pt_20.se.get(cyc_20, 'mass')

    # 25Msun
    pt_25 = mp.se(fol2mod, 'M25.0Z2.0e-02.Ma.0023601.out.h5', rewrite=True)
    cyc_25 = pt_25.se.cycles[-1]
    # pt_25.se.get('temperature')
    t9_cyc_25 = pt_25.se.get(cyc_25, 'temperature')
    mass_25 = pt_25.se.get(cyc_25, 'mass')

    dataset = 'Ri18'
    citation = ''
    models = {}
    for emass, cyc_, pt_exp in [('15', cyc_15, pt_15), ('20', cyc_20, pt_20), ('25', cyc_25, pt_25)]:
        masscoord = np.array(pt_exp.se.get(cyc_, 'mass'))
        keys = utils.asisotopes(pt_exp.se.isotopes, allow_invalid=True)
        abu = pt_exp.se.get(cyc_, 'iso_massf')
        if divide_by_isomass:
            abu = np.asarray(abu) / np.array([float(iso.mass) if type(iso) is utils.Isotope else 1.0
                                              for iso in keys])
            unit = 'mol'
        else:
            unit = 'wt'

        models[f"{dataset}_m{emass}"] = dict(type='CCSNe', dataset=dataset, citation=citation,
                                                   refid_isoabu=ref_isoabu, refid_isomass=ref_isomass,
                                                   mass=int(emass), masscoord=masscoord,
                                                   abundance_values=abu, abundance_keys=keys,
                                                   abundance_unit=unit)
    return models

def Pi16(data_dir, ref_isoabu, ref_isomass, divide_by_isomass = True):
    # loading Pignatari+16 model
    fol2mod = data_dir + 'P16/'
    # load instances of models

    # 15Msun
    P16_15 = mp.se(fol2mod, 'M15.0', rewrite=True)
    cyc_P16_15 = P16_15.se.cycles[-1]
    # pt_15.se.get('temperature')
    t9_cyc_P16_15 = P16_15.se.get(cyc_P16_15, 'temperature')
    mass_P16_15 = P16_15.se.get(cyc_P16_15, 'mass')

    # 20Msun
    P16_20 = mp.se(fol2mod, 'M20.0', rewrite=True)
    cyc_P16_20 = P16_20.se.cycles[-1]
    # pt_20.se.get('temperature')
    t9_cyc_P16_20 = P16_20.se.get(cyc_P16_20, 'temperature')
    mass_P16_20 = P16_20.se.get(cyc_P16_20, 'mass')

    # 25Msun
    P16_25 = mp.se(fol2mod, 'M25.0', rewrite=True)
    cyc_P16_25 = P16_25.se.cycles[-1]
    # pt_25.se.get('temperature')
    t9_cyc_P16_25 = P16_25.se.get(cyc_P16_25, 'temperature')
    mass_P16_25 = P16_25.se.get(cyc_P16_25, 'mass')

    dataset = 'Pi16'
    citation = ''
    models = {}
    for emass, cyc_, pt_exp in [('15', cyc_P16_15, P16_15), ('20', cyc_P16_20, P16_20), ('25', cyc_P16_25, P16_25)]:
        masscoord = np.array(pt_exp.se.get(cyc_, 'mass'))
        keys = utils.asisotopes(pt_exp.se.isotopes, allow_invalid=True)
        abu = pt_exp.se.get(cyc_, 'iso_massf')
        if divide_by_isomass:
            abu = np.asarray(abu) / np.array([float(iso.mass) if type(iso) is utils.Isotope else 1.0
                                              for iso in keys])
            unit = 'mol'
        else:
            unit = 'wt'

        models[f"{dataset}_m{emass}"] = dict(type='CCSNe', dataset=dataset, citation=citation,
                                                   refid_isoabu=ref_isoabu, refid_isomass=ref_isomass,
                                                   mass=int(emass), masscoord=masscoord,
                                                   abundance_values=abu, abundance_keys=keys,
                                                   abundance_unit=unit)
    return models

def La22(data_dir, ref_isoabu, ref_isomass, divide_by_isomass = True):
    # Loading Lawson+22 - 1 peformance upgrade
    dir_law = data_dir + 'LAW22/'
    models_list = {'15': 'M15s_run15f1_216M1.3bgl_mp.txt',
                   '20': 'M20s_run20f1_300M1.56jl_mp.txt',
                   '25': 'M25s_run25f1_280M1.83rrl_mp.txt'}
    num_species = 5209

    dataset = 'La22'
    citation = ''
    models = {}
    for emass, model_name in models_list.items():
        mass_lines = []
        with open(dir_law + model_name, "rt") as f:
            for ln, line in enumerate(f):
                if 'mass enclosed' in line:
                    mass_lines.append(line)
        mass = [float(row.split()[3]) for row in mass_lines]
        numpart = [int(row.split()[0][1:]) for row in mass_lines]
        number_of_parts = len(numpart)  # number of particles (it may change from model to model)
        # print('# particles = ',number_of_parts)

        # open and read abundances for all trajectories
        a, x, z, iso_name = [], [], [], []
        with open(dir_law + model_name, "rt") as f:
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
                    iso_i.append(f"{utils.z_names[int(line[1])]}-{line[0]}")
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
        abu =  np.transpose([list(v) for v in y.values()])
        if divide_by_isomass:
            abu = np.asarray(abu) / np.array([float(iso.mass) if type(iso) is utils.Isotope else 1.0
                                              for iso in keys])
            unit = 'mol'
        else:
            unit = 'wt'

        models[f"{dataset}_m{emass}"] = dict(type='CCSNe', dataset=dataset, citation=citation,
                                                   refid_isoabu=ref_isoabu, refid_isomass=ref_isomass,
                                                   mass=int(emass), masscoord=masscoord,
                                                   abundance_values=abu, abundance_keys=keys,
                                                   abundance_unit=unit)
    return models

def Si18(data_dir, ref_isoabu, ref_isomass, divide_by_isomass = True, decayed=False):
    # dir where Sieverdin models are located
    dir_sie = data_dir + 'SIE18/'

    file_sie_all = {'15': "s15_data.hdf5",
                    '20': "s20_data.hdf5",
                    '25': "s25_data.hdf5"}

    dataset = 'Si18'
    citation = ''
    models = {}
    for emass, file_sie in file_sie_all.items():
        with h5py.File(dir_sie + file_sie) as data_file:
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
        if divide_by_isomass:
            abu = np.asarray(abu) / np.array([float(iso.mass) if type(iso) is utils.Isotope else 1.0
                                              for iso in keys])
            unit = 'mol'
        else:
            unit = 'wt'

        models[f"{dataset}_m{emass}"] = dict(type='CCSNe', dataset=dataset, citation=citation,
                                                   refid_isoabu=ref_isoabu, refid_isomass=ref_isomass,
                                                   mass=int(emass), masscoord=masscoord,
                                                   abundance_values=abu, abundance_keys=keys,
                                                   abundance_unit=unit)
    return models

def Ra02(data_dir, ref_isoabu, ref_isomass, divide_by_isomass = True):
    # Rauscher - 1 peformance upgrade
    dir_rau = data_dir + 'R02/'

    # models_rau = ['s15a28c.expl_yield']
    models_rau = {'15': 's15a28c.expl_yield',
                  '20': 's20a28n.expl_yield',
                  '25': 's25a28d.expl_yield'}

    dataset = 'Ra02'
    citation = ''
    models = {}
    for emass, model_name in models_rau.items():
        filename = dir_rau + model_name
        # print(filename)
        with open(filename, 'r') as f:
            head = f.readline();
            isos_dum = head.split()[5:]  # getting isotopes, not first header names
            dum_a = [re.findall('\d+', ik)[0] for ik in isos_dum]  # getting the A from isotope name
            dum_el = [re.sub(r'[0-9]+', '', ik) for ik in isos_dum]  # getting the element name from the isotope name
            dum_new_iso = [dum_el[ik].capitalize() + '-' + dum_a[ik] for ik in range(len(isos_dum))]


            # isotope name that we can use around, just neutron name is different, but not care
            keys = utils.asisotopes(dum_new_iso, allow_invalid=True)

            data = f.readlines()[:-2]  # getting the all item, excepting the last two lines
            # rau_mass.append(dum) # converting in Msun too.
            abu = np.asarray([row.split()[3:] for row in data], np.float64)
            if divide_by_isomass:
                abu = np.asarray(abu) / np.array([float(iso.mass) if type(iso) is utils.Isotope else 1.0
                                                  for iso in keys])
                unit = 'mol'
            else:
                unit = 'wt'

            masscoord = np.array([float(ii.split()[1]) / 1.989e+33 for ii in data])

            models[f"{dataset}_m{emass}"] = dict(type='CCSNe', dataset=dataset, citation=citation,
                                                       refid_isoabu=ref_isoabu, refid_isomass=ref_isomass,
                                                       mass=int(emass), masscoord=masscoord,
                                                       abundance_values=abu, abundance_keys=keys,
                                                   abundance_unit=unit)

    return models

def LC18(data_dir, ref_isoabu, ref_isomass, divide_by_isomass = True):
    # item from LC18
    dir_lc18 = data_dir + 'LC18/'

    models_lc18 = {'15': '015a000.dif_iso_nod',
                   '20': '020a000.dif_iso_nod',
                   '25': '025a000.dif_iso_nod'}

    skip_heavy_ = 43  # used to skip final ye and spooky abundances (see below)

    dataset = 'LC18'
    citation = ''
    models = {}
    for emass, model_name in models_lc18.items():
        filename = dir_lc18 + model_name
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
            dum_a = [re.findall('\d+', ik)[0] for ik in isos_dum]  # getting the A from isotope name
            dum_el = [re.sub(r'[0-9]+', '', ik) for ik in isos_dum]  # getting the element name from the isotope name
            dum_new_iso = [dum_el[ik].capitalize() + '-' + dum_a[ik] for ik in range(len(isos_dum))]

            data = f.readlines()[:-1]  # getting the all item, excepting the last fake line (bounch of zeros)

            masscoord = np.array([float(ii.split()[0]) for ii in data])

            # isotope name that we can use around, just neutron name is different, but not care
            keys = utils.asisotopes(dum_new_iso, allow_invalid=True)

            # done reading, just closing the file now
            # converting in Msun too.
            abu = np.asarray([row.split()[4:-skip_heavy_] for row in data], dtype=np.float64)
            if divide_by_isomass:
                abu = np.asarray(abu) / np.array([float(iso.mass) if type(iso) is utils.Isotope else 1.0
                                                  for iso in keys])
                unit = 'mol'
            else:
                unit = 'wt'

            models[f"{dataset}_m{emass}"] = dict(type='CCSNe', dataset=dataset, citation=citation,
                                                       refid_isoabu=ref_isoabu, refid_isomass=ref_isomass,
                                                       mass=int(emass), masscoord=masscoord,
                                                       abundance_values=abu, abundance_keys=keys,
                                                   abundance_unit=unit)
    return models