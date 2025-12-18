import pytest

import numpy as np
import simple

@pytest.fixture
def input_values():
    stdkeys = simple.utils.asisotopes("92Mo, 94Mo, 95Mo, 96Mo, 97Mo, 98Mo, 100Mo, "
                                      "96Ru, 98Ru, 99Ru, 100Ru, 101Ru, 102Ru, 104Ru, "
                                      "103Rh, "
                                      "102Pd, 104Pd, 105Pd, 106Pd, 108Pd, 110Pd")

    isomass = np.array([91.90680716, 93.90508359, 94.90583744, 95.90467477, 96.9060169, 97.90540361, 99.907468,
                        95.90758891, 97.905287, 98.9059303, 99.9042105, 100.9055731, 101.9043403, 103.9054254,
                        102.9054941,
                        101.9056321, 103.9040304, 104.9050795, 105.9034803, 107.9038918, 109.9051729])

    std_abu = np.array([0.37, 0.233, 0.404, 0.425, 0.245, 0.622, 0.25,
                        0.099, 0.033, 0.227, 0.224, 0.304, 0.562, 0.332,
                        0.37,
                        0.0139, 0.1513, 0.3032, 0.371, 0.359, 0.159])

    abukeys = simple.utils.asisotopes("92Mo*, 94Mo, 95Mo, 96Mo, 97Mo, 98Mo, 100Mo, "
                                       "96Ru, 98Ru*, 99Ru*, 100Ru*, 101Ru*, 102Ru*, 104Ru*, "
                                       "103Rh, "
                                       "102Pd, 104Pd, 105Pd, 106Pd, 108Pd, 110Pd")

    abu = np.array([0, 0.002097, 0.281184, 0.509575, 0.156065, 0.511284, 0.01125,
                     0, 0, 0.075137, 0.246176, 0.053808, 0.281, 0.0083,
                     0.05624,
                     0, 0.1839808, 0.0476024, 0.216664, 0.267814, 0.00477])

    return dict(stdkeys=stdkeys, isomass=isomass, std_abu=std_abu, abukeys=abukeys, abu=abu)

@pytest.fixture
def collection(input_values):
    collection = simple.new_collection()
    collection.new_ref('IsoRef', 'mass', type='MASS', citation='',
                         data_values=input_values['isomass'], data_keys=input_values['stdkeys'], data_unit='Da')
    collection.new_ref('IsoRef', 'abu', type='ABU', citation='',
                         data_values=input_values['std_abu'], data_keys=input_values['stdkeys'], data_unit='mol')
    return collection

@pytest.fixture
def model1(collection, input_values):
    model = collection.new_model('CCSNe', 'model1',
                                 refid_isomass='mass', refid_isoabu='abu',
                                 type='Test', dataset='Testing', citation = '',
                                 mass='-1', masscoord=np.array([1], dtype=np.float64),
                                 masscoord_mass=np.array([10], dtype=np.float64),
                                 abundance_values=input_values['abu'], abundance_keys=input_values['abukeys'],
                                 abundance_unit='mol')
    return model

@pytest.fixture
def model2(collection, input_values):
    model = collection.new_model('CCSNe', 'model2',
                                 refid_isomass='mass', refid_isoabu='abu',
                                 type='Test', dataset='Testing', citation = '',
                                 mass='-1', masscoord=np.array([1], dtype=np.float64),
                                 masscoord_mass=np.array([10], dtype=np.float64),
                                 abundance_values=input_values['abu']*0.1, abundance_keys=input_values['stdkeys'],
                                 abundance_unit='mol')
    return model

@pytest.fixture
def model3a(collection, input_values):
    sabu = input_values['abu']
    abu = np.concatenate([[sabu],
                          [sabu * 0.01],
                          [sabu * 0.1],
                          [sabu * 0.5]], axis=0)
    abu[:, 2] = abu[:, 2] * [1, 2, 3, 4]

    model = collection.new_model('CCSNe', 'model3a',
                                 refid_isomass='mass', refid_isoabu='abu',
                                 type='Test', dataset='Testing', citation='',
                                 mass='-1', masscoord=np.array([1,2,3,4], dtype=np.float64),
                                 masscoord_mass=np.array([10, 20, 30, 40], dtype=np.float64),
                                 abundance_values=abu, abundance_keys=input_values['abukeys'],
                                 abundance_unit='mol')
    return model

@pytest.fixture
def model3b(collection, input_values):
    sabu = input_values['abu']
    abu = np.concatenate([[sabu],
                          [sabu * 0.000001],
                          [sabu * 0.0000001],
                          [sabu * 0.0001]], axis=0)
    abu[:, 2] = abu[:, 2] * [1, 2, 3, 4]  # Largest offset has df larger smaller than 0.1 for row 1 and 2

    model = collection.new_model('CCSNe', 'model3b',
                                 refid_isomass='mass', refid_isoabu='abu',
                                 type='Test', dataset='Testing', citation='',
                                 mass='-1', masscoord=np.array([1,2,3,4], dtype=np.float64),
                                 masscoord_mass=np.array([10, 20, 30, 40], dtype=np.float64),
                                 abundance_values=abu, abundance_keys=input_values['abukeys'],
                                 abundance_unit='mol')
    return model


@pytest.fixture
def ccsne_models():
    return simple.load_collection('tests/data/CCSNe_FeNi.hdf5')