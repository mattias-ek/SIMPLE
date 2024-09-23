import pytest
import numpy as np
import simple

isokeys = simple.utils.asisotopes("92Mo, 94Mo, 95Mo, 96Mo, 97Mo, 98Mo, 100Mo, "
                                  "96Ru, 98Ru, 99Ru, 100Ru, 101Ru, 102Ru, 104Ru, "
                                  "103Rh, "
                                  "102Pd, 104Pd, 105Pd, 106Pd, 108Pd, 110Pd")

isokeys2 = simple.utils.asisotopes("92Mo*, 94Mo, 95Mo, 96Mo, 97Mo, 98Mo, 100Mo, "
                                   "96Ru, 98Ru*, 99Ru*, 100Ru*, 101Ru*, 102Ru*, 104Ru*, "
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

sabu = np.array([0, 0.002097, 0.281184, 0.509575, 0.156065, 0.511284, 0.01125,
                 0, 0, 0.075137, 0.246176, 0.053808, 0.281, 0.0083,
                 0.05624,
                 0, 0.1839808, 0.0476024, 0.216664, 0.267814, 0.00477])

result123_rel = np.array([])


def test_intnorm_largest_offset1():
    abu = sabu


    collection = simple.ModelCollection()
    collection.new_model('IsoRef', 'mass', type='MASS', citation='', data_values = isomass, data_keys=isokeys)
    collection.new_model('IsoRef', 'abu', type='ABU', citation='', data_values=std_abu, data_keys=isokeys)
    model = collection.new_model('Test', 'testing',
                                 refid_isomass='mass', refid_isoabu='abu',
                                 abundance=simple.askeyarray(abu, isokeys2), abundance_keys = isokeys2,
                                )
    # If the test suddenly start failing it could be float point issues. Maybe the tolerances are to high?
    # The default rtol of the largest offset is 1E-4 and the atol on the eps values is 1E-4 (0.01 ppm)

    # Test 1 - Pd
    result = model.internal_normalisation('108pd/105pd')

    correct = 12593.65716
    assert np.allclose(result['dilution_factor'], correct, rtol=1e-3, atol=0)

    correct = np.array([0.356688389, 1.000003237, 0, 0.18187762, 0, -0.873417299])
    assert np.allclose(result['eRi_values'], correct, rtol=0, atol=1E-4)

    # Test 2 - Mo, Ru, Pd
    result = model.internal_normalisation(('98mo/96mo', '99ru*/101ru*','108pd/105pd'))

    correct = 15745.78012
    assert np.allclose(result['dilution_factor'], correct, rtol=1e-3, atol=0)

    correct = np.array([-1.000000019, -0.440821659, 0, -0.236455613, 0, -0.258723091,
                        -0.259895723, 0, 0.536812512, 0, 0.253259338, 0.046613434,
                        0.285285122, 0.799814409, 0, 0.145468255, 0, -0.698579925])
    assert np.allclose(result['eRi_values'], correct, rtol=0, atol=1E-4)

    # Test 3 - Mo1, Ru2, Pd3
    result = model.internal_normalisation(('98mo/96mo', '99ru*/101ru*', '108pd/105pd'), enrichment_factor=(1, 2, 3))

    correct = 37780.97147
    assert np.allclose(result['dilution_factor'], correct, rtol=1e-3, atol=0)

    correct = np.array([-0.416787848, -0.183727891, 0, -0.098550241, 0, -0.107827139,
                        -0.21663168, 0, 0.447450333, 0, 0.211099444, 0.038853899,
                        0.356688389, 1.000003237, 0, 0.18187762, 0, -0.873417299])
    assert np.allclose(result['eRi_values'], correct, rtol=0, atol=1E-4)

    # Test 4 - abs Mo1, Ru2, Pd3
    result = model.internal_normalisation(('98mo/96mo', '99ru*/101ru*', '108pd/105pd'), enrichment_factor=(1, 2, 3), relative_enrichment=False)

    correct = 52413.23777
    assert np.allclose(result['dilution_factor'], correct, rtol=1e-3, atol=0)

    correct = np.array([-0.204177996, -0.090005268, 0, -0.048277974, 0, -0.052821885,
                        -0.235022799, 0, 0.485437284, 0, 0.229021171, 0.042152404,
                        0.356687204, 0.999999913, 0, 0.181877016, 0, -0.873414396])

    assert np.allclose(result['eRi_values'], correct, rtol=0, atol=1E-4)

def test_intnorm_largest_offset2():
    abu = np.concatenate([[sabu],
                          [sabu * 0.01],
                          [sabu * 0.1],
                          [sabu * 0.5]], axis=0)
    abu[:, 2] = abu[:, 2] * [1, 2, 3, 4]

    collection = simple.ModelCollection()
    collection.new_model('IsoRef', 'mass', type='MASS', citation='', data_values = isomass, data_keys=isokeys)
    collection.new_model('IsoRef', 'abu', type='ABU', citation='', data_values=std_abu, data_keys=isokeys)
    model = collection.new_model('Test', 'testing',
                                 refid_isomass='mass', refid_isoabu='abu',
                                 abundance=simple.askeyarray(abu, isokeys2), abundance_keys = isokeys2,
                                )

    # Test 5 - Multirow large df
    result = model.internal_normalisation('98mo/96mo')

    correct = np.array([15745.78012, 157.4578012, 1574.578012, 7872.890098])
    assert np.allclose(result['dilution_factor'], correct, rtol=1e-3, atol=0)

    correct = np.array([[-1.000000019, -0.440821659, 0, -0.236455613, 0, -0.258723091],
                        [-1.000000019, 0.001162511, 0, -0.236455613, 0, -0.258723091],
                        [-1.000000019, 0.443146682, 0, -0.236455613, 0, -0.258723091],
                        [-1.000000014, 0.885130848, 0, -0.236455611, 0, -0.25872309]])
    assert np.allclose(result['eRi_values'], correct, rtol=0, atol=1E-4)

def test_intnorm_largest_offset3():
    abu = np.concatenate([[sabu],
                          [sabu * 0.000001],
                          [sabu * 0.0000001],
                          [sabu * 0.0001]], axis=0)
    abu[:, 2] = abu[:, 2] * [1, 2, 3, 4] # Largest offset has df larger smaller than 0.1 for row 1 and 2

    collection = simple.ModelCollection()
    collection.new_model('IsoRef', 'mass', type='MASS', citation='', data_values = isomass, data_keys=isokeys)
    collection.new_model('IsoRef', 'abu', type='ABU', citation='', data_values=std_abu, data_keys=isokeys)
    model = collection.new_model('Test', 'testing',
                                 refid_isomass='mass', refid_isoabu='abu',
                                 abundance=simple.askeyarray(abu, isokeys2), abundance_keys = isokeys2,
                                )

    # Test 6.1 - Multirow small df
    result = model.internal_normalisation('98mo/96mo')

    #correct = np.array([15745.78042, 0.015745781, 0.001574578, 1.574578055])
    correct = np.array([15745.78042, np.nan, np.nan, 1.574578055])
    assert np.allclose(result['dilution_factor'], correct, rtol=1e-4, atol=0, equal_nan=True)

    correct = np.array([[-1, -0.440821651, 0, -0.236455608, 0, -0.258723086],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [-0.999999991, 0.885130828, 0, -0.236455606, 0, -0.258723084]])
    assert np.allclose(result['eRi_values'], correct, rtol=0, atol=1E-4, equal_nan=True)

    # Test 6.2 - Multirow small df, lower min_dilution_factor
    result = model.internal_normalisation('98mo/96mo', min_dilution_factor=0.01)

    # correct = np.array([15745.78042, 0.015745781, 0.001574578, 1.574578055])
    correct = np.array([15745.78042, 0.015745781, np.nan, 1.574578055])
    assert np.allclose(result['dilution_factor'], correct, rtol=1e-4, atol=0, equal_nan=True)

    correct = np.array([[-1, -0.440821651, 0, -0.236455608, 0, -0.258723086],
                        [-0.999999992, 0.001162511, 0, -0.236455606, 0, -0.258723084],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [-0.999999991, 0.885130828, 0, -0.236455606, 0, -0.258723084]])
    assert np.allclose(result['eRi_values'], correct, rtol=0, atol=1E-4, equal_nan=True)

    # Test 6.3 - Multirow small df, even lower min_dilution_factor
    result = model.internal_normalisation('98mo/96mo', min_dilution_factor=0.001)

    correct = np.array([15745.78042, 0.015745781, 0.001574578, 1.574578055])
    assert np.allclose(result['dilution_factor'], correct, rtol=1e-4, atol=0, equal_nan=True)

    correct = np.array([[-1, -0.440821651, 0, -0.236455608, 0, -0.258723086],
                        [-0.999999992, 0.001162511, 0, -0.236455606, 0, -0.258723084],
                        [-0.999999992, 0.44314667, 0, -0.236455606, 0, -0.258723084],
                        [-0.999999991, 0.885130828, 0, -0.236455606, 0, -0.258723084]])
    assert np.allclose(result['eRi_values'], correct, rtol=0, atol=1E-4, equal_nan=True)

    # Test 7.1 - Multirow small df, lower largest offset
    result = model.internal_normalisation('98mo/96mo', largest_offset = 0.1)

    correct = np.array([157471.2203, 0.15747122, np.nan, 15.74712203])
    assert np.allclose(result['dilution_factor'], correct, rtol=1e-4, atol=0, equal_nan=True)

    correct = np.array([[-0.1, -0.044081715, 0, -0.023644975, -1.11022E-12, -0.025870272],
                        [-0.1, 0.000116448, 0, -0.023644975, 0, -0.025870272],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [-0.1, 0.088512772, 0, -0.023644975, 0, -0.025870272]])
    assert np.allclose(result['eRi_values'], correct, rtol=0, atol=1E-4, equal_nan=True)

    # Test 7.2 - Multirow small df, lower largest offset and min_dilution_factor
    result = model.internal_normalisation('98mo/96mo', largest_offset=0.1, min_dilution_factor=0.01)

    correct = np.array([157471.2203, 0.15747122, 0.015747122, 15.74712203])
    assert np.allclose(result['dilution_factor'], correct, rtol=1e-4, atol=0, equal_nan=True)

    correct = np.array([[-0.1, -0.044081715, 0, -0.023644975, -1.11022E-12, -0.025870272],
                        [-0.1, 0.000116448, 0, -0.023644975, 0, -0.025870272],
                        [-0.1, 0.04431461, 0, -0.023644975, -1.11022E-12, -0.025870272],
                        [-0.1, 0.088512772, 0, -0.023644975, 0, -0.025870272]])
    assert np.allclose(result['eRi_values'], correct, rtol=0, atol=1E-4, equal_nan=True)




