import pytest
from fixtures import *

import numpy as np

import simple
from simple import plotting


def test_get_mask(collection, model1, model2, model3a):
    mask = model1.get_mask('.masscoord <= 3')
    np.testing.assert_array_equal(mask, (model1.masscoord <= 3))

    mask = model2.get_mask('.masscoord <= 3')
    np.testing.assert_array_equal(mask, (model2.masscoord <= 3))

    mask = model3a.get_mask('.masscoord <= 3')
    np.testing.assert_array_equal(mask, (model3a.masscoord <= 3))

    mask = model1.get_mask('.masscoord <= 3 & .masscoord > 1')
    np.testing.assert_array_equal(mask, np.logical_and((model1.masscoord <= 3), (model1.masscoord > 1)))

    mask = model2.get_mask('.masscoord <= 3 & .masscoord > 1')
    np.testing.assert_array_equal(mask, np.logical_and((model2.masscoord <= 3), (model2.masscoord > 1)))

    mask = model3a.get_mask('.masscoord <= 3 & .masscoord > 1')
    np.testing.assert_array_equal(mask, np.logical_and((model3a.masscoord <= 3), (model3a.masscoord > 1)))

    mask = model1.get_mask('.masscoord > 3 | .masscoord <= 1')
    np.testing.assert_array_equal(mask, np.logical_or((model1.masscoord > 3), (model1.masscoord <= 1)))

    mask = model2.get_mask('.masscoord > 3 | .masscoord <= 1')
    np.testing.assert_array_equal(mask, np.logical_or((model2.masscoord > 3), (model2.masscoord <= 1)))

    mask = model3a.get_mask('.masscoord > 3 | .masscoord <= 1')
    np.testing.assert_array_equal(mask, np.logical_or((model3a.masscoord > 3), (model3a.masscoord <= 1)))

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
            assert model in modeldata

            assert type(modeldata[model]) is tuple
            assert len(modeldata[model]) == len(correct_keys)
            for ki, key in enumerate(correct_keys):
                keydata = modeldata[model][ki]

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
            assert model in modeldata

            assert type(modeldata[model]) is tuple
            assert len(modeldata[model]) == len(correct_keys)
            for ki, key in enumerate(correct_keys):
                keydata = modeldata[model][ki]

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
            assert model in modeldata

            assert type(modeldata[model]) is tuple
            assert len(modeldata[model]) == len(correct_keys)
            for ki, key in enumerate(correct_keys):
                keydata = modeldata[model][ki]

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
            assert model in modeldata

            assert type(modeldata[model]) is tuple
            assert len(modeldata[model]) == len(correct_keys)
            for ki, key in enumerate(correct_keys):
                keydata = modeldata[model][ki]

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
            assert model in modeldata

            assert type(modeldata[model]) is tuple
            assert len(modeldata[model]) == len(correct_keys)
            for ki, key in enumerate(correct_keys):
                keydata = modeldata[model][ki]

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
            assert model in modeldata

            assert type(modeldata[model]) is tuple
            assert len(modeldata[model]) == len(correct_keys)
            for ki, key in enumerate(correct_keys):
                keydata = modeldata[model][ki]

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
            assert model in modeldata

            assert type(modeldata[model]) is tuple
            assert len(modeldata[model]) == len(correct_keys)
            for ki, key in enumerate(correct_keys):
                keydata = modeldata[model][ki]

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
            assert model in modeldata

            assert type(modeldata[model]) is tuple
            assert len(modeldata[model]) == len(correct_keys)
            for ki, key in enumerate(correct_keys):
                keydata = modeldata[model][ki]

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

class TestWeights:
    @pytest.mark.parametrize('weights', [1, 3.14, '94Mo', '94Mo, 95Mo', 'Mo'])
    @pytest.mark.parametrize('sum_weights', [True, False])
    @pytest.mark.parametrize('norm_weights', [True, False])
    @pytest.mark.parametrize('ykey', ['96Mo', '96Mo, 98Mo'])
    @pytest.mark.parametrize('mask', [None, '.masscoord <= 3', '.masscoord >= 3 | .masscoord < 1', '.masscoord >= 3 & .masscoord < 1'])
    @pytest.mark.parametrize('mask_na', [True, False])
    @pytest.mark.parametrize('ccsne', [True, False])
    def test_add_weights(self, collection, model1, model3a, weights, sum_weights, norm_weights,
                             ykey, mask, mask_na, ccsne):
        modeldata, axis_labels = plotting.get_data(collection, {'xmass': '.masscoord_mass', 'y': ykey},
                                                   mask=mask, mask_na=mask_na)


        fail = False
        if type(weights) == str:
            modeldata_w, _ = plotting.get_data(collection, {'w': weights},
                                               mask=mask, mask_na=mask_na)
            for model in modeldata_w:
                if sum_weights:
                    sum = 0
                    for dp in modeldata_w[model]: sum += dp['w']
                    modeldata_w[model] = [{'w': sum for dp in modeldata[model]}]
                if len(modeldata_w[model]) != len(modeldata[model]):
                    if len(modeldata_w[model]) == 1:
                        modeldata_w[model] = [modeldata_w[model][0].copy() for _ in range(len(modeldata[model]))]
                    else:
                        fail = True
        else:
            modeldata_w = {}
            for model in modeldata:
                modeldata_w[model] = []
                for dp in modeldata[model]:
                    if mask is None:
                        n = dp['y'].size
                        if n == 0:
                            result = np.full(dp['y'].shape, np.nan, dtype=np.float64)
                        else:
                            result = np.full(dp['y'].shape, 1 / n if norm_weights else weights, dtype=np.float64)
                    else:
                        m = model.get_mask(mask)
                        n = np.count_nonzero(m)

                        if n == 0:
                            result = np.full(dp['y'].shape if mask_na else 0, np.nan, dtype=np.float64)
                        elif mask_na:
                            result = np.full(dp['y'].shape, weights, dtype=np.float64)
                            result[np.invert(m)] = np.nan
                        else:
                            result = np.full(n, weights, dtype=np.float64)
                    modeldata_w[model].append({'w': result})

        if ccsne:
            add_weights = simple.add_weights_ccsne
            for model in modeldata_w:
                for dp in modeldata_w[model]:
                    dp['w'] = dp['w'] * modeldata[model][0]['xmass']
        else:
            add_weights = simple.add_weights

        if norm_weights:
            for model in modeldata_w:
                for i, dp in enumerate(modeldata_w[model]):
                    sum = np.nansum(dp['w'])
                    if sum > 0: dp['w'] = dp['w'] / sum


        if fail:
            with pytest.raises(ValueError):
                add_weights(modeldata, 'y', weights,
                            sum_weights=sum_weights, norm_weights=norm_weights,
                            mask=mask, mask_na=mask_na)
            return
        else:
            add_weights(modeldata, 'y', weights,
                        sum_weights=sum_weights, norm_weights=norm_weights,
                        mask=mask, mask_na=mask_na)


        model = model1
        for i, datapoint in enumerate(modeldata[model]):
            assert isinstance(datapoint, dict)
            assert 'w' in datapoint

            w = datapoint['w']
            y = datapoint['y']
            assert type(w) is np.ndarray
            assert w.shape == y.shape

            np.testing.assert_allclose(np.nansum(w), np.nansum(modeldata_w[model][i]['w']))
            np.testing.assert_allclose(w, modeldata_w[model][i]['w'])

        model = model3a
        for i, datapoint in enumerate(modeldata[model]):
            assert isinstance(datapoint, dict)
            assert 'w' in datapoint

            w = datapoint['w']
            y = datapoint['y']
            assert type(w) is np.ndarray
            assert w.shape == y.shape

            np.testing.assert_allclose(np.nansum(w), np.nansum(modeldata_w[model][i]['w']))
            np.testing.assert_allclose(w, modeldata_w[model][i]['w'])

    @pytest.mark.parametrize('ccsne', [True, False])
    def test_add_weights2(self, collection, model1, model3a, ccsne):
        modeldata, axis_labels = plotting.get_data(collection, {'y': 'Mo96, Mo98'})
        if ccsne:
            add_weights = simple.add_weights_ccsne
        else:
            add_weights = simple.add_weights

        ### Number
        add_weights(modeldata, 'y', 1)
        for model in modeldata:
            weight = np.full(model.abundance['Mo-95'].shape, 1)
            if ccsne: weight = weight * model.masscoord_mass
            weight = weight / np.sum(weight)
            for dp in modeldata[model]:
                np.testing.assert_allclose(dp['w'], weight)

        add_weights(modeldata, 'y', 1, norm_weights=False)
        for model in modeldata:
            weight = np.full(model.abundance['Mo-95'].shape, 1)
            if ccsne: weight = weight * model.masscoord_mass
            for dp in modeldata[model]:
                np.testing.assert_allclose(dp['w'], weight)

        ### Abundance
        add_weights(modeldata, 'y', 'Mo95')
        for model in modeldata:
            weight = model.abundance['Mo-95']
            if ccsne: weight = weight * model.masscoord_mass
            weight = weight / np.sum(weight)
            for dp in modeldata[model]:
                np.testing.assert_allclose(dp['w'], weight)

        add_weights(modeldata, 'y', 'Mo95', norm_weights=False)
        for model in modeldata:
            weight = model.abundance['Mo-95']
            if ccsne: weight = weight * model.masscoord_mass
            for dp in modeldata[model]:
                np.testing.assert_allclose(dp['w'], weight)

        ### Abundance 2-1
        add_weights(modeldata, 'y', 'Mo95, Mo94')
        for model in modeldata:
            weight = model.abundance['Mo-95'] + model.abundance['Mo-94']
            if ccsne: weight = weight * model.masscoord_mass
            weight = weight / np.sum(weight)
            for dp in modeldata[model]:
                np.testing.assert_allclose(dp['w'], weight)

        add_weights(modeldata, 'y', 'Mo95, Mo94', norm_weights=False)
        for model in modeldata:
            weight = model.abundance['Mo-95'] + model.abundance['Mo-94']
            if ccsne: weight = weight * model.masscoord_mass
            for dp in modeldata[model]:
                np.testing.assert_allclose(dp['w'], weight)

        ### Abundance 2-2
        add_weights(modeldata, 'y', 'Mo95, Mo94', sum_weights=False)
        for model in modeldata:
            weight = model.abundance['Mo-95']
            if ccsne: weight = weight * model.masscoord_mass
            weight = weight / np.sum(weight)
            np.testing.assert_allclose(modeldata[model][0]['w'], weight)

            weight = model.abundance['Mo-94']
            if ccsne: weight = weight * model.masscoord_mass
            weight = weight / np.sum(weight)
            np.testing.assert_allclose(modeldata[model][1]['w'], weight)

        add_weights(modeldata, 'y', 'Mo95, Mo94', sum_weights=False, norm_weights=False)
        for model in modeldata:
            weight = model.abundance['Mo-95']
            if ccsne: weight = weight * model.masscoord_mass
            np.testing.assert_allclose(modeldata[model][0]['w'], weight)

            weight = model.abundance['Mo-94']
            if ccsne: weight = weight * model.masscoord_mass
            np.testing.assert_allclose(modeldata[model][1]['w'], weight)











