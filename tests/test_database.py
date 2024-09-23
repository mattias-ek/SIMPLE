import pytest
import os
import numpy as np
from numpy.testing import assert_equal
from simple import database, utils

class TestModel:
    def test_attrs(self):
        models = database.ModelCollection()

        model = models.new_model('Test', 'Model')
        assert model.name == 'Model'
        assert type(model) is database.Test

        with pytest.raises(AttributeError):
            model.mass = 1

        model.add_attr('mass', 1, save=True)
        assert 'mass' in model.saved_attrs
        assert isinstance(model.mass, np.ndarray)
        assert model.mass.shape == ()
        assert model.mass.dtype == np.int64
        assert model.mass == 1
        assert model['mass'] is model.mass

        with pytest.raises(AttributeError):
            model.mass = 1

        with pytest.raises(AttributeError):
            model.add_attr('mass', 1.5, save=True)

        model.add_attr('mass', 1.5, save=True, overwrite=True)
        assert isinstance(model.mass, np.ndarray)
        assert model.mass.shape == ()
        assert model.mass.dtype == np.float64
        assert model.mass == 1.5
        assert model['mass'] is model.mass

        with pytest.raises(AttributeError):
            model.add_attr('mass', 2)

        with pytest.raises(AttributeError):
            model.mass = 2

        model.add_attr('mass', 2, overwrite=True)
        assert type(model.mass) is int
        assert model.mass == 2
        assert model['mass'] is model.mass

        with pytest.raises(AttributeError):
            model.add_attr('mass', 2.5)

        model.add_attr('mass', 2.5, overwrite=True)
        assert type(model.mass) is float
        assert model.mass == 2.5
        assert model['mass'] is model.mass

        with pytest.raises(AttributeError):
            model.add_attr('mass', 3, save=True)

        model.add_attr('mass', 3, save=True, overwrite=True)
        assert isinstance(model.mass, np.ndarray)
        assert model.mass.shape == ()
        assert model.mass.dtype == np.int64
        assert model.mass == 3
        assert model['mass'] is model.mass

        ###########
        # Strings #
        ###########
        model.add_attr('citation', 'Me')
        assert isinstance(model.citation, str)
        assert model.citation == 'Me'
        assert model['citation'] is model.citation

        model.add_attr('citation', 'Irene', save=True, overwrite=True)
        assert isinstance(model.citation, str)
        assert model.citation == 'Irene'
        assert model['citation'] is model.citation

        ############
        # Keyarray #
        ############
        keys = utils.asisotopes('Pd-102, Pd-104, Pd-105*')
        values = np.array([[21, 41, 51],
                           [22, 42, 52],
                           [23, 43, 53],
                           [24, 44, 54]])
        a = utils.askeyarray(values, keys)
        model.add_attr('data', a, save=True)
        assert model.data is a
        assert model['data'] is model.data
        assert isinstance(model.data, np.ndarray)
        assert model.data.dtype.names == keys
        assert_equal(model.data, a)

    def test_names(self):
        models = database.ModelCollection()

        model = models.new_model('Test', 'Model')
        assert model.name == 'Model'
        assert type(model) is database.Test

        with pytest.raises(AttributeError):
            model.name = 'Another Model'

        model.change_name('Yet Another Model')
        assert model.name == 'Yet Another Model'

    def test_copy(self):
        pass

    def test_save_load1(self):
        filename = 'tests/savetest.hdf5'
        if os.path.exists(filename):
            os.remove(filename)

        saving = database.ModelCollection()

        saved_model = saving.new_model('Test', 'Model')
        assert saved_model.name == 'Model'
        assert type(saved_model) is database.Test

        mass = 1
        citation = 'Me'
        abc = ('a', 'b', 'c')
        note = 'note'

        keys = utils.asisotopes('Pd-102, Pd-104, Pd-105*')
        values = np.array([[21, 41, 51],
                           [22, 42, 52],
                           [23, 43, 53],
                           [24, 44, 54]])
        data = utils.askeyarray(values, keys)

        saved_model.add_attr('mass', mass, save=True)
        saved_model.add_attr('citation', citation, save=True)
        saved_model.add_attr('abc', abc, save=True)
        saved_model.add_attr('note', note, save=False)
        saved_model.add_attr('data', data, save=True)

        saving.save(filename)

        loaded = database.ModelCollection()
        loaded.load_file(filename=filename)

        loaded_model = loaded['Model']

        assert loaded_model.mass == mass
        assert isinstance(loaded_model.mass, np.ndarray)
        assert loaded_model.mass.dtype == np.int64
        assert saved_model.mass.shape == ()

        assert loaded_model.citation == citation
        assert type(loaded_model.citation) is str

        assert loaded_model.abc == abc
        assert type(loaded_model.abc) is tuple
        for item in loaded_model.abc:
            assert type(item) is str

        with pytest.raises(AttributeError):
            loaded_model.note

        assert_equal(loaded_model.data, data)

    def test_ccsne(self):
        filename = 'tests/testzn.hdf5'
        if os.path.exists(filename):
            os.remove(filename)



        mc = database.load_models(filename, 'data/SIMPLE_CCSNeV1.hdf5')


    def test_ref(self):
        keys = utils.asisotopes('101Ru,102Ru,104Ru,103Rh,102Pd,104Pd,105Pd,106Pd,108Pd,110Pd,107Ag,109Ag')
        stdabu = np.array([0.304, 0.562, 0.332, 0.37, 0.0139, 0.1513, 0.3032, 0.371, 0.359, 0.159, 0.254, 0.236])

        collection = database.ModelCollection()
        ref_abu = collection.new_model('IsoRef', 'abu', type='ABU', citation='', values=stdabu, keys=keys)
        model = collection.new_model('Test', 'testing',
                                     refid_isoabu='abu',
                                     )
        assert 'abu' in collection.refs
        assert collection.refs['abu'] is ref_abu
        assert collection.get_ref('abu') is ref_abu
        assert model.get_ref('abu') is ref_abu
        assert model.get_ref(model.refid_isoabu) is ref_abu
