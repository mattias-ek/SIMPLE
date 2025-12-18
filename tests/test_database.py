import pytest
import os
import numpy as np
from numpy.testing import assert_equal
from simple import models, utils

@pytest.fixture
def collection():
    return models.ModelCollection()

@pytest.fixture
def test_model_cls():
    class TestModel(models.ModelBase):
        pass

    return TestModel

class TestModel:
    def test_attrs(self, collection, test_model_cls):
        print(models.AllModelClasses)
        model = collection.new_model('TestModel', 'test')
        assert model.name == 'test'
        assert type(model) is test_model_cls

        with pytest.raises(AttributeError):
            model.mass = 1

        model.setattr('mass', 1, hdf5_compatible=True)
        assert 'mass' in model.hdf5_attrs
        assert 'mass' not in model.normal_attrs
        assert type(model.mass) == int
        assert model.mass == 1
        assert model['mass'] is model.mass

        with pytest.raises(AttributeError):
            model.mass = 1

        with pytest.raises(AttributeError):
            model.setattr('mass', 1.5, hdf5_compatible=True)

        model.setattr('mass', 1.5, hdf5_compatible=True, overwrite=True)
        assert 'mass' in model.hdf5_attrs
        assert 'mass' not in model.normal_attrs
        assert type(model.mass) == float
        assert model.mass == 1.5
        assert model['mass'] is model.mass

        with pytest.raises(AttributeError):
            model.setattr('mass', 2)

        with pytest.raises(AttributeError):
            model.mass = 2

        model.setattr('mass', 2, overwrite=True)
        assert 'mass' not in model.hdf5_attrs
        assert 'mass' in model.normal_attrs
        assert type(model.mass) is int
        assert model.mass == 2
        assert model['mass'] is model.mass

        with pytest.raises(AttributeError):
            model.setattr('mass', 2.5)

        model.setattr('mass', 2.5, overwrite=True)
        assert 'mass' not in model.hdf5_attrs
        assert 'mass' in model.normal_attrs
        assert type(model.mass) is float
        assert model.mass == 2.5
        assert model['mass'] is model.mass

        with pytest.raises(AttributeError):
            model.setattr('mass', 3, hdf5_compatible=True)

        model.setattr('mass', 3, hdf5_compatible=True, overwrite=True)
        assert 'mass' in model.hdf5_attrs
        assert 'mass' not in model.normal_attrs
        assert type(model.mass) is int
        assert model.mass == 3
        assert model['mass'] is model.mass

        ###########
        # Strings #
        ###########
        model.setattr('citation', 'Me')
        assert isinstance(model.citation, str)
        assert model.citation == 'Me'
        assert model['citation'] is model.citation

        model.setattr('citation', 'Irene', hdf5_compatible=True, overwrite=True)
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
        model.setattr('data', a, hdf5_compatible=True)
        assert model.data is a
        assert model['data'] is model.data
        assert isinstance(model.data, np.ndarray)
        assert model.data.dtype.names == keys
        assert_equal(model.data, a)

    def test_names(self, collection, test_model_cls):
        model = collection.new_model('TestModel', 'ModelBase')
        assert model.name == 'ModelBase'
        assert type(model) is test_model_cls

        with pytest.raises(AttributeError):
            model.name = 'Another ModelBase'

    def test_save_load1(self, test_model_cls):
        filename = 'tests/savetest.hdf5'
        if os.path.exists(filename):
            os.remove(filename)

        saving = models.ModelCollection()

        saved_model = saving.new_model('TestModel', 'ModelBase')
        assert saved_model.name == 'ModelBase'
        assert type(saved_model) is test_model_cls

        mass = 1
        citation = 'Me'
        abc = ('a', 'b', 'c')
        note = 'note'
        array0 = np.array(3.14)
        array1 = np.array([1, 2, 3])

        keys = utils.asisotopes('Pd-102, Pd-104, Pd-105*')
        values = np.array([[21, 41, 51],
                           [22, 42, 52],
                           [23, 43, 53],
                           [24, 44, 54]])
        data = utils.askeyarray(values, keys)

        saved_model.setattr('mass', mass, hdf5_compatible=True)
        saved_model.setattr('citation', citation, hdf5_compatible=True)
        saved_model.setattr('abc', abc, hdf5_compatible=True)
        saved_model.setattr('note', note, hdf5_compatible=False)
        saved_model.setattr('array0', array0, hdf5_compatible=True)
        saved_model.setattr('array1', array1, hdf5_compatible=True)
        saved_model.setattr('data', data, hdf5_compatible=True)

        saving.save(filename)

        loaded = models.ModelCollection()
        loaded.load_file(filename=filename)

        loaded_model = loaded['ModelBase']

        assert loaded_model.mass == mass
        assert type(loaded_model.mass) is int

        assert loaded_model.citation == citation
        assert type(loaded_model.citation) is str

        assert loaded_model.abc == abc
        assert type(loaded_model.abc) is tuple
        for item in loaded_model.abc:
            assert type(item) is str

        with pytest.raises(AttributeError):
            loaded_model.note

        assert isinstance(loaded_model.array0, np.ndarray)
        assert loaded_model.array0.size == 1
        assert loaded_model.array0.ndim == 0
        assert_equal(loaded_model.array0, array0)

        assert isinstance(loaded_model.array1, np.ndarray)
        assert loaded_model.array1.size == 3
        assert loaded_model.array1.ndim == 1
        assert_equal(loaded_model.array1, array1)

        assert_equal(loaded_model.data, data)


    def test_ref(self, collection, test_model_cls):
        keys = utils.asisotopes('101Ru,102Ru,104Ru,103Rh,102Pd,104Pd,105Pd,106Pd,108Pd,110Pd,107Ag,109Ag')
        stdabu = np.array([0.304, 0.562, 0.332, 0.37, 0.0139, 0.1513, 0.3032, 0.371, 0.359, 0.159, 0.254, 0.236])

        ref_abu = collection.new_ref('IsoRef', 'abu',
                                       type='ABU', citation='',
                                       data_values=stdabu, data_keys=keys, data_unit='mass')

        model = collection.new_model('TestModel', 'testing',
                                     refid_isoabu='abu',
                                     )

        assert ref_abu in collection.refs
        assert collection.get_ref('abu') is ref_abu
        assert collection.get_ref(model.refid_isoabu) is ref_abu
        assert model.ref_isoabu is ref_abu
