import pytest
from fixtures import *

import numpy as np

import simple
from simple import plotting

class TestGetData:
    def test_get_data1(self, collection, model1):
        correct_models = [model1]

        # x = masscoord
        modeldata, axis_labels = plotting.get_data(collection, 'x', xkey = '.masscoord')
        correct_keys = [0]

        assert type(modeldata) is dict
        assert type(axis_labels) is dict

        assert len(axis_labels) == 1
        assert axis_labels['x'] == model1.masscoord_label_latex

        assert len(modeldata) == len(correct_models)
        for mi, model in enumerate(correct_models):
            assert model.name in modeldata

            assert type(modeldata[model.name]) is tuple
            assert len(modeldata[model.name]) == len(correct_keys)
            for ki, key in enumerate(correct_keys):
                keydata = modeldata[model.name][ki]

                assert type(keydata) is dict
                assert len(keydata) == 2

                assert 'label' in keydata
                assert keydata['label'] == None # If only 1 model then no name by default

                assert 'x' in keydata
                np.testing.assert_array_equal(keydata['x'], model.masscoord)

        # x = masscoord, y = 96Mo, 98Mo
        modeldata, axis_labels = plotting.get_data(collection, {'x': '.masscoord', 'y':  '96Mo, 98Mo'})
        correct_keys = simple.asisotopes('96Mo 98Mo'.split())

        assert type(modeldata) is dict
        assert type(axis_labels) is dict

        assert len(axis_labels) == 2
        assert axis_labels['x'] == model1.masscoord_label_latex
        assert axis_labels['y'] == 'abundance | <y> [mol]'

        assert len(modeldata) == len(correct_models)
        for mi, model in enumerate(correct_models):
            assert model.name in modeldata

            assert type(modeldata[model.name]) is tuple
            assert len(modeldata[model.name]) == len(correct_keys)
            for ki, key in enumerate(correct_keys):
                keydata = modeldata[model.name][ki]

                assert type(keydata) is dict
                assert len(keydata) == 3

                assert 'label' in keydata
                assert keydata['label'] == f'<y: ${{}}^{{{key.mass}}}\\mathrm{{{key.symbol}}}$>'

                assert 'x' in keydata
                np.testing.assert_array_equal(keydata['x'], model.masscoord)

                assert 'y' in keydata
                np.testing.assert_array_equal(keydata['y'], model.abundance[key])

        # x = masscoord, y = Mo
        modeldata, axis_labels = plotting.get_data(collection, 'x, y', xkey='.masscoord', ykey='Mo')
        correct_keys = simple.asisotopes('94Mo 95Mo 96Mo 97Mo 98Mo 100Mo'.split())

        assert type(modeldata) is dict
        assert type(axis_labels) is dict

        assert len(axis_labels) == 2
        assert axis_labels['x'] == model1.masscoord_label_latex
        assert axis_labels['y'] == 'abundance | <y> [mol]'

        assert len(modeldata) == len(correct_models)
        for mi, model in enumerate(correct_models):
            assert model.name in modeldata

            assert type(modeldata[model.name]) is tuple
            assert len(modeldata[model.name]) == len(correct_keys)
            for ki, key in enumerate(correct_keys):
                keydata = modeldata[model.name][ki]

                assert type(keydata) is dict
                assert len(keydata) == 3

                assert 'label' in keydata
                assert keydata['label'] == f'<y: ${{}}^{{{key.mass}}}\\mathrm{{{key.symbol}}}$>'

                assert 'x' in keydata
                np.testing.assert_array_equal(keydata['x'], model.masscoord)

                assert 'y' in keydata
                np.testing.assert_array_equal(keydata['y'], model.abundance[key])

        # x = masscoord, y = 94Mo / 95Mo, 96Mo/95Mo
        modeldata, axis_labels = plotting.get_data(collection, 'x y', xkey='.masscoord', ykey='94Mo/95Mo, 96Mo/95Mo')
        correct_keys = simple.asratios('94Mo/95Mo 96Mo/95Mo'.split())

        assert type(modeldata) is dict
        assert type(axis_labels) is dict

        assert len(axis_labels) == 2
        assert axis_labels['x'] == model1.masscoord_label_latex
        assert axis_labels['y'] == 'abundance | <y> / ${}^{95}\\mathrm{Mo}$ [mol]'

        assert len(modeldata) == len(correct_models)
        for mi, model in enumerate(correct_models):
            assert model.name in modeldata

            assert type(modeldata[model.name]) is tuple
            assert len(modeldata[model.name]) == len(correct_keys)
            for ki, key in enumerate(correct_keys):
                keydata = modeldata[model.name][ki]

                assert type(keydata) is dict
                assert len(keydata) == 3

                assert 'label' in keydata
                assert keydata['label'] == f'<y: ${{}}^{{{key.numer.mass}}}\\mathrm{{{key.numer.symbol}}}$>'

                assert 'x' in keydata
                np.testing.assert_array_equal(keydata['x'], model.masscoord)

                assert 'y' in keydata
                np.testing.assert_array_equal(keydata['y'], model.abundance[key.numer] / model.abundance[key.denom])


    def test_get_data2(self, collection, model1, model2, model3a):
        correct_models = [model1, model2, model3a]

        # x = masscoord
        modeldata, axis_labels = plotting.get_data(collection, 'x', xkey = '.masscoord')
        correct_keys = [0]

        assert type(modeldata) is dict
        assert type(axis_labels) is dict

        assert len(axis_labels) == 1
        assert axis_labels['x'] == model1.masscoord_label_latex

        assert len(modeldata) == len(correct_models)
        for mi, model in enumerate(correct_models):
            assert model.name in modeldata

            assert type(modeldata[model.name]) is tuple
            assert len(modeldata[model.name]) == len(correct_keys)
            for ki, key in enumerate(correct_keys):
                keydata = modeldata[model.name][ki]

                assert type(keydata) is dict
                assert len(keydata) == 2

                assert 'label' in keydata
                assert keydata['label'] == f'{model.name}'

                assert 'x' in keydata
                np.testing.assert_array_equal(keydata['x'], model.masscoord)

        # x = masscoord, y = 96Mo, 98Mo
        modeldata, axis_labels = plotting.get_data(collection, {'x': '.masscoord', 'y':  '96Mo, 98Mo'})
        correct_keys = simple.asisotopes('96Mo 98Mo'.split())

        assert type(modeldata) is dict
        assert type(axis_labels) is dict

        assert len(axis_labels) == 2
        assert axis_labels['x'] == model1.masscoord_label_latex
        assert axis_labels['y'] == 'abundance | <y> [mol]'

        assert len(modeldata) == len(correct_models)
        for mi, model in enumerate(correct_models):
            assert model.name in modeldata

            assert type(modeldata[model.name]) is tuple
            assert len(modeldata[model.name]) == len(correct_keys)
            for ki, key in enumerate(correct_keys):
                keydata = modeldata[model.name][ki]

                assert type(keydata) is dict
                assert len(keydata) == 3

                assert 'label' in keydata
                assert keydata['label'] == f'<y: ${{}}^{{{key.mass}}}\\mathrm{{{key.symbol}}}$> ({model.name})'

                assert 'x' in keydata
                np.testing.assert_array_equal(keydata['x'], model.masscoord)

                assert 'y' in keydata
                np.testing.assert_array_equal(keydata['y'], model.abundance[key])

        # x = masscoord, y = Mo
        modeldata, axis_labels = plotting.get_data(collection, 'x, y', xkey='.masscoord', ykey='Mo')
        correct_keys = simple.asisotopes('92Mo 94Mo 95Mo 96Mo 97Mo 98Mo 100Mo'.split())

        assert type(modeldata) is dict
        assert type(axis_labels) is dict

        assert len(axis_labels) == 2
        assert axis_labels['x'] == model1.masscoord_label_latex
        assert axis_labels['y'] == 'abundance | <y> [mol]'

        assert len(modeldata) == len(correct_models)
        for mi, model in enumerate(correct_models):
            assert model.name in modeldata

            assert type(modeldata[model.name]) is tuple
            assert len(modeldata[model.name]) == len(correct_keys)
            for ki, key in enumerate(correct_keys):
                keydata = modeldata[model.name][ki]

                assert type(keydata) is dict
                assert len(keydata) == 3

                assert 'x' in keydata
                np.testing.assert_array_equal(keydata['x'], model.masscoord)

                assert 'label' in keydata
                assert 'y' in keydata

                if key in model.abundance.dtype.names:
                    assert keydata['label'] == f'<y: ${{}}^{{{key.mass}}}\\mathrm{{{key.symbol}}}$> ({model.name})'
                    np.testing.assert_array_equal(keydata['y'], model.abundance[key])
                else:
                    assert keydata['label'] == f'<y: !{key}> ({model.name})'
                    np.testing.assert_array_equal(keydata['y'], np.full(len(model.abundance), np.nan))

        # x = masscoord, y = 94Mo / 95Mo, 96Mo/95Mo
        modeldata, axis_labels = plotting.get_data(collection, 'x y', xkey='.masscoord', ykey='92Mo/95Mo, 94Mo/95Mo, 96Mo/95Mo')
        correct_keys = simple.asratios('92Mo/95Mo 94Mo/95Mo 96Mo/95Mo'.split())

        assert type(modeldata) is dict
        assert type(axis_labels) is dict

        assert len(axis_labels) == 2
        assert axis_labels['x'] == model1.masscoord_label_latex
        assert axis_labels['y'] == 'abundance | <y> / ${}^{95}\\mathrm{Mo}$ [mol]'

        assert len(modeldata) == len(correct_models)
        for mi, model in enumerate(correct_models):
            assert model.name in modeldata

            assert type(modeldata[model.name]) is tuple
            assert len(modeldata[model.name]) == len(correct_keys)
            for ki, key in enumerate(correct_keys):
                keydata = modeldata[model.name][ki]

                assert type(keydata) is dict
                assert len(keydata) == 3

                assert 'label' in keydata
                assert 'y' in keydata

                assert 'x' in keydata
                np.testing.assert_array_equal(keydata['x'], model.masscoord)

                if key.numer in model.abundance.dtype.names:
                    assert keydata['label'] == f'<y: ${{}}^{{{key.numer.mass}}}\\mathrm{{{key.numer.symbol}}}$> ({model.name})'
                    np.testing.assert_array_equal(keydata['y'], model.abundance[key.numer] / model.abundance[key.denom])
                else:
                    assert keydata['label'] == f'<y: !{key.numer}> ({model.name})'
                    np.testing.assert_array_equal(keydata['y'], np.full(len(model.abundance), np.nan))


