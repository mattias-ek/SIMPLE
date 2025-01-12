import pytest
import numpy as np
import simple
from simple import utils, models
import logging

logging.basicConfig(level=logging.INFO)

# The `norm_calc.xlsx` spreadsheet contains the calculation of the correct values used here.

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
    collection.new_model('IsoRef', 'mass', type='MASS', citation='',
                         data_values=input_values['isomass'], data_keys=input_values['stdkeys'], data_unit='Da')
    collection.new_model('IsoRef', 'abu', type='ABU', citation='',
                         data_values=input_values['std_abu'], data_keys=input_values['stdkeys'], data_unit='mol')
    return collection

@pytest.fixture
def model1(collection, input_values):
    model = collection.new_model('CCSNe', 'model1',
                                 refid_isomass='mass', refid_isoabu='abu',
                                 type='Test', dataset='Testing', citation = '',
                                 mass='-1', masscoord=[],
                                 abundance_values=input_values['abu'], abundance_keys=input_values['abukeys'],
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
                                 mass='-1', masscoord=[],
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
                                 mass='-1', masscoord=[],
                                 abundance_values=abu, abundance_keys=input_values['abukeys'],
                                 abundance_unit='mol')
    return model

# Internal normalisation
def test_intnorm_largest_offset1(collection, model1):
    # Test 1 - Pd
    collection.internal_normalisation('108pd/105pd')
    result = model1.intnorm

    correct = 12593.65716
    assert np.allclose(result['dilution_factor'], correct, rtol=1e-3, atol=0)

    correct = np.array([0.356688389, 1.000003237, 0, 0.18187762, 0, -0.873417299])
    assert np.allclose(result['eRi_values'], correct, rtol=0, atol=1E-4)

    # Test 2 - Mo, Ru, Pd
    collection.internal_normalisation(('98mo/96mo', '99ru*/101ru*', '108pd/105pd'))
    result = model1.intnorm

    correct = 15745.78012
    assert np.allclose(result['dilution_factor'], correct, rtol=1e-3, atol=0)

    correct = np.array([-1.000000019, -0.440821659, 0, -0.236455613, 0, -0.258723091,
                        -0.259895723, 0, 0.536812512, 0, 0.253259338, 0.046613434,
                        0.285285122, 0.799814409, 0, 0.145468255, 0, -0.698579925])
    assert np.allclose(result['eRi_values'], correct, rtol=0, atol=1E-4)

    # Test 3 - Mo1, Ru2, Pd3
    collection.internal_normalisation(('98mo/96mo', '99ru*/101ru*', '108pd/105pd'), enrichment_factor=(1, 2, 3))
    result = model1.intnorm

    correct = 37780.97147
    assert np.allclose(result['dilution_factor'], correct, rtol=1e-3, atol=0)

    correct = np.array([-0.416787848, -0.183727891, 0, -0.098550241, 0, -0.107827139,
                        -0.21663168, 0, 0.447450333, 0, 0.211099444, 0.038853899,
                        0.356688389, 1.000003237, 0, 0.18187762, 0, -0.873417299])
    assert np.allclose(result['eRi_values'], correct, rtol=0, atol=1E-4)

    # Test 4 - abs Mo1, Ru2, Pd3
    collection.internal_normalisation(('98mo/96mo', '99ru*/101ru*', '108pd/105pd'), enrichment_factor=(1, 2, 3), relative_enrichment=False)
    result = model1.intnorm

    correct = 52413.23777
    assert np.allclose(result['dilution_factor'], correct, rtol=1e-3, atol=0)

    correct = np.array([-0.204177996, -0.090005268, 0, -0.048277974, 0, -0.052821885,
                        -0.235022799, 0, 0.485437284, 0, 0.229021171, 0.042152404,
                        0.356687204, 0.999999913, 0, 0.181877016, 0, -0.873414396])

    assert np.allclose(result['eRi_values'], correct, rtol=0, atol=1E-4)

def test_intnorm_largest_offset3(collection, model3a, model3b):
    # Test 5 - Multirow large df
    collection.internal_normalisation('98mo/96mo')
    result = model3a.intnorm

    correct = np.array([15745.78012, 157.4578012, 1574.578012, 7872.890098])
    assert np.allclose(result['dilution_factor'], correct, rtol=1e-3, atol=0)

    correct = np.array([[-1.000000019, -0.440821659, 0, -0.236455613, 0, -0.258723091],
                        [-1.000000019, 0.001162511, 0, -0.236455613, 0, -0.258723091],
                        [-1.000000019, 0.443146682, 0, -0.236455613, 0, -0.258723091],
                        [-1.000000014, 0.885130848, 0, -0.236455611, 0, -0.25872309]])
    assert np.allclose(result['eRi_values'], correct, rtol=0, atol=1E-4)

    # Test 6.1 - Multirow small df
    result = model3b.intnorm

    # correct = np.values([15745.78042, 0.015745781, 0.001574578, 1.574578055])
    correct = np.array([15745.78042, np.nan, np.nan, 1.574578055])
    assert np.allclose(result['dilution_factor'], correct, rtol=1e-4, atol=0, equal_nan=True)

    correct = np.array([[-1, -0.440821651, 0, -0.236455608, 0, -0.258723086],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [-0.999999991, 0.885130828, 0, -0.236455606, 0, -0.258723084]])
    assert np.allclose(result['eRi_values'], correct, rtol=0, atol=1E-4, equal_nan=True)

    # Test 6.2 - Multirow small df, lower min_dilution_factor
    collection.internal_normalisation('98mo/96mo', min_dilution_factor=0.01)
    result = model3b.intnorm

    # correct = np.values([15745.78042, 0.015745781, 0.001574578, 1.574578055])
    correct = np.array([15745.78042, 0.015745781, np.nan, 1.574578055])
    assert np.allclose(result['dilution_factor'], correct, rtol=1e-4, atol=0, equal_nan=True)

    correct = np.array([[-1, -0.440821651, 0, -0.236455608, 0, -0.258723086],
                        [-0.999999992, 0.001162511, 0, -0.236455606, 0, -0.258723084],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [-0.999999991, 0.885130828, 0, -0.236455606, 0, -0.258723084]])
    assert np.allclose(result['eRi_values'], correct, rtol=0, atol=1E-4, equal_nan=True)

    # Test 6.3 - Multirow small df, even lower min_dilution_factor
    collection.internal_normalisation('98mo/96mo', min_dilution_factor=0.001)
    result = model3b.intnorm

    correct = np.array([15745.78042, 0.015745781, 0.001574578, 1.574578055])
    assert np.allclose(result['dilution_factor'], correct, rtol=1e-4, atol=0, equal_nan=True)

    correct = np.array([[-1, -0.440821651, 0, -0.236455608, 0, -0.258723086],
                        [-0.999999992, 0.001162511, 0, -0.236455606, 0, -0.258723084],
                        [-0.999999992, 0.44314667, 0, -0.236455606, 0, -0.258723084],
                        [-0.999999991, 0.885130828, 0, -0.236455606, 0, -0.258723084]])
    assert np.allclose(result['eRi_values'], correct, rtol=0, atol=1E-4, equal_nan=True)

    # Test 7.1 - Multirow small df, lower largest offset
    collection.internal_normalisation('98mo/96mo', largest_offset=0.1)
    result = model3b.intnorm

    correct = np.array([157471.2203, 0.15747122, np.nan, 15.74712203])
    assert np.allclose(result['dilution_factor'], correct, rtol=1e-4, atol=0, equal_nan=True)

    correct = np.array([[-0.1, -0.044081715, 0, -0.023644975, -1.11022E-12, -0.025870272],
                        [-0.1, 0.000116448, 0, -0.023644975, 0, -0.025870272],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [-0.1, 0.088512772, 0, -0.023644975, 0, -0.025870272]])
    assert np.allclose(result['eRi_values'], correct, rtol=0, atol=1E-4, equal_nan=True)

    # Test 7.2 - Multirow small df, lower largest offset and min_dilution_factor
    collection.internal_normalisation('98mo/96mo', largest_offset=0.1, min_dilution_factor=0.01)
    result = model3b.intnorm

    correct = np.array([157471.2203, 0.15747122, 0.015747122, 15.74712203])
    assert np.allclose(result['dilution_factor'], correct, rtol=1e-4, atol=0, equal_nan=True)

    correct = np.array([[-0.1, -0.044081715, 0, -0.023644975, -1.11022E-12, -0.025870272],
                        [-0.1, 0.000116448, 0, -0.023644975, 0, -0.025870272],
                        [-0.1, 0.04431461, 0, -0.023644975, -1.11022E-12, -0.025870272],
                        [-0.1, 0.088512772, 0, -0.023644975, 0, -0.025870272]])
    assert np.allclose(result['eRi_values'], correct, rtol=0, atol=1E-4, equal_nan=True)

def test_intnorm_better_linear1(collection, model1):
    # Test 1 - Pd
    collection.internal_normalisation('108pd/105pd',
                                          method='better_linear')
    result = model1.intnorm

    correct = np.array([28612.34852, 80214.76022, 0, 14589.4154, 0, -70065.96604])
    assert np.allclose(result['eRi_values'], correct, rtol=0, atol=1E-4)

    # Test 2 - Mo, Ru, Pd
    collection.internal_normalisation(('98mo/96mo', '99ru*/101ru*','108pd/105pd'),
                                          method='better_linear')
    result = model1.intnorm

    correct = np.array([-13133.67064, -5789.540656, 0, -3105.444561, 0, -3397.686672,
                        -23120.68645, 0, 47755.17572, 0, 22529.92662, 4146.834915,
                        28612.34852, 80214.76022, 0, 14589.4154, 0, -70065.96604])
    assert np.allclose(result['eRi_values'], correct, rtol=0, atol=1E-4)

    # Test 3 - Mo1, Ru2, Pd3
    collection.internal_normalisation(('98mo/96mo', '99ru*/101ru*', '108pd/105pd'),
                                          enrichment_factor=(1, 2, 3), method='better_linear')
    result = model1.intnorm

    correct = np.array([-13133.67064, -5789.540656, 0, -3105.444561, 0, -3397.686672,
                        -23120.68645, 0, 47755.17572, 0, 22529.92662, 4146.834915,
                        28612.34852, 80214.76022, 0, 14589.4154, 0, -70065.96604])
    assert np.allclose(result['eRi_values'], correct, rtol=0, atol=1E-4)

    # Test 4 - abs Mo1, Ru2, Pd3
    collection.internal_normalisation(('98mo/96mo', '99ru*/101ru*', '108pd/105pd'),
                                          enrichment_factor=(1, 2, 3), relative_enrichment=False, method='better_linear')
    result = model1.intnorm

    correct = np.array([-13133.67064, -5789.540656, 0, -3105.444561, 0, -3397.686672,
                        -23120.68645, 0, 47755.17572, 0, 22529.92662, 4146.834915,
                        28612.34852, 80214.76022, 0, 14589.4154, 0, -70065.96604])

    assert np.allclose(result['eRi_values'], correct, rtol=0, atol=1E-4)

def test_intnorm_better_linear3(collection, model3a, model3b):
    # Test 5 - Multirow large df
    collection.internal_normalisation('98mo/96mo', method='better_linear')

    correct = np.array([[-13133.67064, -5789.540656, 0, -3105.444561, 0, -3397.686672],
                        [-13133.67064, 15.29670887, 0, -3105.444561, 0, -3397.686672],
                        [-13133.67064, 5820.134073, 0, -3105.444561, 0, -3397.686672],
                        [-13133.67064, 11624.97144, 0, -3105.444561, 0, -3397.686672]])
    assert np.allclose(model3a.intnorm['eRi_values'], correct, rtol=0, atol=1E-4)
    assert np.allclose(model3b.intnorm['eRi_values'], correct, rtol=0, atol=1E-4)

    # linear no mass_coeff
    collection.internal_normalisation('98mo/96mo', method='linear')

    correct = np.array([[-13133.67064, -5789.540656, 0, -3105.444561, 0, -3397.686672],
                        [-13133.67064, 15.29670887, 0, -3105.444561, 0, -3397.686672],
                        [-13133.67064, 5820.134073, 0, -3105.444561, 0, -3397.686672],
                        [-13133.67064, 11624.97144, 0, -3105.444561, 0, -3397.686672]])

    assert np.allclose(model3a.intnorm['eRi_values'], correct, rtol=0, atol=1E-4)
    assert np.allclose(model3b.intnorm['eRi_values'], correct, rtol=0, atol=1E-4)

    # linear mass_coeff
    collection.internal_normalisation('98mo/96mo', method='linear', mass_coef='better')

    correct = np.array([[-13133.67064, -5789.540656, 0, -3105.444561, 0, -3397.686672],
                        [-13133.67064, 15.29670887, 0, -3105.444561, 0, -3397.686672],
                        [-13133.67064, 5820.134073, 0, -3105.444561, 0, -3397.686672],
                        [-13133.67064, 11624.97144, 0, -3105.444561, 0, -3397.686672]])
    assert np.allclose(model3a.intnorm['eRi_values'], correct, rtol=0, atol=1E-4)
    assert np.allclose(model3b.intnorm['eRi_values'], correct, rtol=0, atol=1E-4)

def test_intnorm_simplified_linear1(collection, model1):

    # Test 1 - Pd
    collection.internal_normalisation('108pd/105pd',
                                          method='simplified_linear')
    result = model1.intnorm

    correct = np.array([27523.86883, 79975.61448, 0, 14707.19797, 0, -70641.64377])
    assert np.allclose(result['eRi_values'], correct, rtol=0, atol=1E-4)

    # Test 2 - Mo, Ru, Pd
    collection.internal_normalisation(('98mo/96mo', '99ru*/101ru*','108pd/105pd'),
                                          method='simplified_linear')
    result = model1.intnorm

    correct = np.array([-13067.43644, -5764.906158, 0 ,-3113.559373, 0, -3334.014517,
                        -23054.42359, 0, 47733.40714, 0, 22594.28317, 4464.965482,
                        27523.86883, 79975.61448, 0, 14707.19797, 0, -70641.64377])
    assert np.allclose(result['eRi_values'], correct, rtol=0, atol=1E-4)

    # Test 3 - Mo1, Ru2, Pd3
    collection.internal_normalisation(('98mo/96mo', '99ru*/101ru*', '108pd/105pd'),
                                          enrichment_factor=(1, 2, 3), method='simplified_linear')
    result = model1.intnorm

    correct = np.array([-13067.43644, -5764.906158, 0, -3113.559373, 0, -3334.014517,
                        -23054.42359, 0, 47733.40714, 0, 22594.28317, 4464.965482,
                        27523.86883, 79975.61448, 0, 14707.19797, 0, -70641.64377])
    assert np.allclose(result['eRi_values'], correct, rtol=0, atol=1E-4)

    # Test 4 - abs Mo1, Ru2, Pd3
    collection.internal_normalisation(('98mo/96mo', '99ru*/101ru*', '108pd/105pd'),
                                          enrichment_factor=(1, 2, 3), relative_enrichment=False,
                                          method='linear', mass_coef='simplified')
    result = model1.intnorm

    correct = np.array([-13067.43644, -5764.906158, 0, -3113.559373, 0, -3334.014517,
                        -23054.42359, 0, 47733.40714, 0, 22594.28317, 4464.965482,
                        27523.86883, 79975.61448, 0, 14707.19797, 0, -70641.64377])

    assert np.allclose(result['eRi_values'], correct, rtol=0, atol=1E-4)

def test_intnorm_simplified_linear3(collection, model3a, model3b):
    # Test 5 - Multirow large df
    collection.internal_normalisation('98mo/96mo', method='simplified_linear')

    correct = np.array([[-13067.43644, -5764.906158, 0, -3113.559373, 0, -3334.014517],
                        [-13067.43644, 39.93120603, 0, -3113.559373, 0, -3334.014517],
                        [-13067.43644, 5844.76857, 0, -3113.559373, 0, -3334.014517],
                        [-13067.43644, 11649.60593, 0, -3113.559373, 0, -3334.014517]])
    assert np.allclose(model3a.intnorm['eRi_values'], correct, rtol=0, atol=1E-4)
    assert np.allclose(model3b.intnorm['eRi_values'], correct, rtol=0, atol=1E-4)

    # linear mass_coeff
    collection.internal_normalisation('98mo/96mo', method='linear', mass_coef='simplified')

    correct = np.array([[-13067.43644, -5764.906158, 0, -3113.559373, 0, -3334.014517],
                        [-13067.43644, 39.93120603, 0, -3113.559373, 0, -3334.014517],
                        [-13067.43644, 5844.76857, 0, -3113.559373, 0, -3334.014517],
                        [-13067.43644, 11649.60593, 0, -3113.559373, 0, -3334.014517]])
    assert np.allclose(model3a.intnorm['eRi_values'], correct, rtol=0, atol=1E-4)
    assert np.allclose(model3b.intnorm['eRi_values'], correct, rtol=0, atol=1E-4)

def test_intnorm_convert_unit(collection, model1):
    # This test does not need to be done for all methods as it is done in the step before
    # the method

    collection['abu'].setattr('data_unit', 'mol', hdf5_compatible=True, overwrite=True)
    collection['model1'].setattr('abundance_unit', 'mass', hdf5_compatible=True, overwrite=True)

    # Test 8 - convert_unit of abu
    collection.internal_normalisation('108pd/105pd')
    result = model1.intnorm

    correct = 120.3821299
    assert np.allclose(result['dilution_factor'], correct, rtol=1e-3, atol=0)

    correct = np.array([0.33850447, 1, 0, 0.182359801, 0, -0.844206277])
    assert np.allclose(result['eRi_values'], correct, rtol=0, atol=1E-4)


    # Test 1 - Pd - convert_unit=False
    collection.internal_normalisation('108pd/105pd', convert_unit=False)
    result = model1.intnorm

    correct = 12593.65716
    assert np.allclose(result['dilution_factor'], correct, rtol=1e-3, atol=0)

    correct = np.array([0.356688389, 1.000003237, 0, 0.18187762, 0, -0.873417299])
    assert np.allclose(result['eRi_values'], correct, rtol=0, atol=1E-4)

    collection['abu'].setattr('data_unit', 'mass', hdf5_compatible=True, overwrite=True)
    collection['model1'].setattr('abundance_unit', 'mol', hdf5_compatible=True, overwrite=True)

    # Test 9- convert_unit of stdabu
    collection.internal_normalisation('108pd/105pd')
    result = model1.intnorm

    correct = 1317792.478
    assert np.allclose(result['dilution_factor'], correct, rtol=1e-3, atol=0)

    correct = np.array([0.375396484, 0.999999621, 0, 0.181228699, 0, -0.903340835])
    assert np.allclose(result['eRi_values'], correct, rtol=0, atol=1E-4)


    # Test 1 - Pd - convert_unit=False
    collection.internal_normalisation('108pd/105pd', convert_unit=False)
    result = model1.intnorm

    correct = 12593.65716
    assert np.allclose(result['dilution_factor'], correct, rtol=1e-3, atol=0)

    correct = np.array([0.356688389, 1.000003237, 0, 0.18187762, 0, -0.873417299])
    assert np.allclose(result['eRi_values'], correct, rtol=0, atol=1E-4)

def test_intnorm_attrname(collection, model1):
    correct = np.array([0.356688389, 1.000003237, 0, 0.18187762, 0, -0.873417299])
    correct_lin = np.array([28612.34852, 80214.76022, 0, 14589.4154, 0, -70065.96604])

    collection.internal_normalisation('108pd/105pd') #default normname=intnorm
    collection.internal_normalisation('108pd/105pd',
                                      method='better_linear', attrname='intnorm_lin')

    assert np.allclose(model1.intnorm['eRi_values'], correct, rtol=0, atol=1E-4)
    assert np.allclose(model1.intnorm_lin['eRi_values'], correct_lin, rtol=0, atol=1E-4)

# Linear normalisation
def test_simplenorm1(collection, model1):
    # Test 1 - Pd
    collection.simple_normalisation('105pd')

    correct = np.array([-1, 6.74522293, 0, 2.719745223, 3.751592357, -0.808917197])
    assert np.allclose(model1.simplenorm['Ri_values'], correct, rtol=0, atol=1E-4)

    # Test 2 - Mo, Ru, Pd
    collection.simple_normalisation(('96mo', '101ru*', '105pd'))

    correct = np.array([-0.992493745, -0.419516264, 0, -0.468723937, -0.314428691, -0.962468724,
                        -1, 0.870056497, 5.209039548, 0, 1.824858757, -0.858757062,
                        -1, 6.74522293, 0, 2.719745223, 3.751592357, -0.808917197])
    assert np.allclose(model1.simplenorm['Ri_values'], correct, rtol=0, atol=1E-4)

    # Test 3 - Mo1, Ru2, Pd3
    collection.simple_normalisation(('96mo', '101ru*', '105pd'),
                                      enrichment_factor=(1, 2, 3))

    correct = np.array([-0.992493745, -0.419516264, 0, -0.468723937, -0.314428691, -0.962468724,
                        -1, 0.870056497, 5.209039548, 0, 1.824858757, -0.858757062,
                        -1, 6.74522293, 0, 2.719745223, 3.751592357, -0.808917197])
    assert np.allclose(model1.simplenorm['Ri_values'], correct, rtol=0, atol=1E-4)

    # Test 4 - abs Mo1, Ru2, Pd3
    collection.simple_normalisation(('96mo', '101ru*', '105pd'),
                                      enrichment_factor=(1, 2, 3), relative_enrichment=False)

    correct = np.array([-0.992493745, -0.419516264, 0, -0.468723937, -0.314428691, -0.962468724,
                        -1, 0.870056497, 5.209039548, 0, 1.824858757, -0.858757062,
                        -1, 6.74522293, 0, 2.719745223, 3.751592357, -0.808917197])

    assert np.allclose(model1.simplenorm['Ri_values'], correct, rtol=0, atol=1E-4)

def test_simplenorm3(collection, model3a, model3b):
    # Test 5 - Multirow large df
    collection.simple_normalisation('96mo')

    correct = np.array([[-0.992493745, -0.419516264, 0, -0.468723937, -0.314428691, -0.962468724],
                        [-0.992493745, 0.160967473, 0, -0.468723937, -0.314428691, -0.962468724],
                        [-0.992493745, 0.741451209, 0, -0.468723937, -0.314428691, -0.962468724],
                        [-0.992493745, 1.321934946, 0, -0.468723937, -0.314428691, -0.962468724]])
    assert np.allclose(model3a.simplenorm['Ri_values'], correct, rtol=0, atol=1E-4)
    assert np.allclose(model3b.simplenorm['Ri_values'], correct, rtol=0, atol=1E-4)

def test_simplenorm_attrname(collection, model1):
    # Test 1 - Pd
    correct = np.array([-1, 6.74522293, 0, 2.719745223, 3.751592357, -0.808917197])

    collection.simple_normalisation('105pd', attrname='simplenorm2')
    assert np.allclose(model1.simplenorm2['Ri_values'], correct, rtol=0, atol=1E-4)
