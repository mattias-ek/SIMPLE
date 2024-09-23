import pytest
from simple import utils
import numpy as np
from numpy.testing import assert_equal

def test_asisotope():
    for string in '102Pd, 102pd , Pd102,pd102 , 102-Pd, 102-pd, Pd-102, pd-102'.split(','):
        iso = utils.asisotope(string)
        assert type(iso) is utils.Isotope
        assert isinstance(iso, str)
        assert iso == 'Pd-102'
        assert iso.mass == '102'
        assert iso.element == 'Pd'
        assert iso.suffix == ''

    ###############
    # Test suffix #
    ###############
    for string in '102Pd*, 102pd** , Pd102 suffix '.split(','):
        iso = utils.asisotope(string)
        assert type(iso) is utils.Isotope
        assert isinstance(iso, str)
        assert iso != 'Pd-102'
        assert iso.mass == '102'
        assert iso.element == 'Pd'
        assert iso.suffix != ''

    iso = utils.asisotope('102Pd*')
    assert iso == 'Pd-102*'
    assert iso.suffix == '*'

    iso = utils.asisotope('102Pd*', without_suffix=True)
    assert iso == 'Pd-102'
    assert iso.suffix == ''

    iso = utils.asisotope('102pd** ', without_suffix=False)
    assert iso == 'Pd-102**'
    assert iso.suffix == '**'

    iso = utils.asisotope('102pd** ', without_suffix=True)
    assert iso == 'Pd-102'
    assert iso.suffix == ''

    iso = utils.asisotope(' Pd102 suffix ', without_suffix=False)
    assert iso == 'Pd-102 suffix'
    assert iso.suffix == ' suffix'

    iso = utils.asisotope(' Pd102 suffix ', without_suffix=True)
    assert iso == 'Pd-102'
    assert iso.suffix == ''

    iso = utils.asisotope('102Pd*').without_suffix()
    assert iso == 'Pd-102'
    assert iso.suffix == ''

    iso = utils.asisotope('102Pd').without_suffix()
    assert iso == 'Pd-102'
    assert iso.suffix == ''

    iso = utils.asisotope('102Pd*', without_suffix=True).without_suffix()
    assert iso == 'Pd-102'
    assert iso.suffix == ''

    ################
    # Test invalid #
    ################
    for string in 'invalid, mattias , Pd-1022, 1022pd, Pd102A, 102-Pdd, 102pd1, *102pd'.split(','):
        with pytest.raises(ValueError):
            utils.asisotope(string)

        iso = utils.asisotope(string, allow_invalid=True)
        assert type(iso) is str
        assert iso == string.strip()

    #########
    # Latex #
    #########
    iso = utils.asisotope(' Pd102*')
    assert iso.latex() == r'${}^{102}\mathrm{Pd*}$'
    assert iso.latex(dollar = False) == r'{}^{102}\mathrm{Pd*}'

    iso = utils.asisotope(' Pd102*', without_suffix=True)
    assert iso.latex() == r'${}^{102}\mathrm{Pd}$'
    assert iso.latex(dollar=False) == r'{}^{102}\mathrm{Pd}'

def test_asisotopes():
    for strings in ['102Pd, pd104, Pd-105*', ['102Pd', 'pd104', 'Pd-105*']]:
        isos = utils.asisotopes(strings)
        assert type(isos) is tuple
        assert len(isos) == 3

        assert isos[0] == 'Pd-102'
        assert isos[1] == 'Pd-104'
        assert isos[2] == 'Pd-105*'
        for iso in isos:
            assert type(iso) is utils.Isotope

    ##################
    # without suffix #
    ##################
    for strings in ['102Pd, pd104, Pd-105*', ['102Pd', 'pd104', 'Pd-105*']]:
        isos = utils.asisotopes(strings, without_suffix=True)
        assert type(isos) is tuple
        assert len(isos) == 3

        assert isos[0] == 'Pd-102'
        assert isos[1] == 'Pd-104'
        assert isos[2] == 'Pd-105'
        for iso in isos:
            assert type(iso) is utils.Isotope

    ###########
    # Invalid #
    ###########
    for strings in [' invalid, pd104, Pd-105*', [' invalid', 'pd104', 'Pd-105*']]:
        with pytest.raises(ValueError):
            utils.asisotopes(strings)

        isos = utils.asisotopes(strings, allow_invalid=True)
        assert type(isos) is tuple
        assert len(isos) == 3

        assert isos[0] == 'invalid'
        assert type(isos[0]) is str
        assert isos[1] == 'Pd-104'
        assert type(isos[1]) is utils.Isotope
        assert isos[2] == 'Pd-105*'
        assert type(isos[2]) is utils.Isotope

def test_asratio():
    for string in '108pd**/105pd*, Pd108** / Pd-105*'.split(','):
        rat = utils.asratio(string)
        assert type(rat) is utils.Ratio
        assert isinstance(rat, str)
        assert rat == 'Pd-108**/Pd-105*'

        assert type(rat.numer) is utils.Isotope
        assert rat.numer == 'Pd-108**'

        assert type(rat.denom) is utils.Isotope
        assert rat.denom == 'Pd-105*'

    ##################
    # Without suffix #
    ##################
    for string in '108pd**/105pd*, Pd108** / Pd-105*'.split(','):
        rat = utils.asratio(string, without_suffix=True)
        assert type(rat) is utils.Ratio
        assert isinstance(rat, str)
        assert rat == 'Pd-108/Pd-105'

        assert type(rat.numer) is utils.Isotope
        assert rat.numer == 'Pd-108'

        assert type(rat.denom) is utils.Isotope
        assert rat.denom == 'Pd-105'

        rat = utils.asratio(string).without_suffix()
        assert type(rat) is utils.Ratio
        assert isinstance(rat, str)
        assert rat == 'Pd-108/Pd-105'

        assert type(rat.numer) is utils.Isotope
        assert rat.numer == 'Pd-108'

        assert type(rat.denom) is utils.Isotope
        assert rat.denom == 'Pd-105'

    #################
    # Allow invalid #
    #################
    with pytest.raises(ValueError):
        utils.asratio('invalid/Pd-105*')

    rat = utils.asratio('invalid/Pd-105*', allow_invalid=True)
    assert type(rat) is utils.Ratio
    assert isinstance(rat, str)
    assert rat == 'invalid/Pd-105*'

    assert type(rat.numer) is str
    assert rat.numer == 'invalid'

    assert type(rat.denom) is utils.Isotope
    assert rat.denom == 'Pd-105*'

    ##########
    # Errors #
    ##########
    with pytest.raises(ValueError):
        utils.asratio('108pd** 105pd*')

    with pytest.raises(ValueError):
        utils.asratio('108pd**//105pd*')

    #########
    # Latex #
    #########
    rat = utils.asratio('108pd**/105pd*')
    assert rat.latex() == fr'{rat.numer.latex()}/{rat.denom.latex()}'

    rat = utils.asratio('108pd**/105pd*')
    assert rat.latex(dollar=False) == fr'{rat.numer.latex(dollar=False)}/{rat.denom.latex(dollar=False)}'

    rat = utils.asratio('invalid/105pd*', allow_invalid=True)
    assert rat.latex() == fr'invalid/{rat.denom.latex()}'

    rat = utils.asratio('invalid/thistoo', allow_invalid=True)
    assert rat.latex() == fr'invalid/thistoo'

def test_asisolist():
    string = '102pd*'
    isolist = utils.asisolist(string)

    assert type(isolist) is dict
    assert len(isolist) == 1
    assert 'Pd-102*' in isolist
    assert type(isolist['Pd-102*']) is tuple
    assert len(isolist['Pd-102*']) == 1
    assert isolist['Pd-102*'] == ('Pd-102*',)

    for k, v in isolist.items():
        assert type(k) is utils.Isotope
        for w in v:
            assert type(w) is utils.Isotope

    keys = 'Pd-102 Pd-104 Pd-105*'.split()
    for strings in ['102Pd, pd104, Pd-105*', ('102Pd', 'pd104', 'Pd-105*')]:
        isolist = utils.asisolist(strings)

        assert type(isolist) is dict
        assert len(isolist) == len(keys)
        for key in keys:
            assert key in isolist
            assert type(isolist[key]) is tuple
            assert len(isolist[key]) == 1
            assert isolist[key] == (key,)

        for k, v in isolist.items():
            assert type(k) is utils.Isotope
            for w in v:
                assert type(w) is utils.Isotope

    ##################
    # Without suffix #
    ##################
    string = '102pd*'
    isolist = utils.asisolist(string, without_suffix=True)

    assert type(isolist) is dict
    assert len(isolist) == 1
    assert 'Pd-102' in isolist
    assert type(isolist['Pd-102']) is tuple
    assert len(isolist['Pd-102']) == 1
    assert isolist['Pd-102'] == ('Pd-102', )

    for k, v in isolist.items():
        assert type(k) is utils.Isotope
        for w in v:
            assert type(w) is utils.Isotope

    keys = 'Pd-102 Pd-104 Pd-105'.split()
    for strings in ['102Pd, pd104, Pd-105*', ('102Pd', 'pd104', 'Pd-105*')]:
        isolist = utils.asisolist(strings, without_suffix=True)

        assert type(isolist) is dict
        assert len(isolist) == len(keys)
        for key in keys:
            assert key in isolist
            assert type(isolist[key]) is tuple
            assert len(isolist[key]) == 1
            assert isolist[key] == (key, )

        for k, v in isolist.items():
            assert type(k) is utils.Isotope
            for w in v:
                assert type(w) is utils.Isotope

    #################
    # allow invalid #
    #################
    string = 'invalid'
    with pytest.raises(ValueError):
        utils.asisolist(string)

    isolist = utils.asisolist(string, allow_invalid=True)

    assert type(isolist) is dict
    assert len(isolist) == 1
    assert 'invalid' in isolist
    assert type(isolist['invalid']) is tuple
    assert len(isolist['invalid']) == 1
    assert isolist['invalid'] == ('invalid',)

    keys = 'invalid Pd-104 Pd-105*'.split()
    for strings in ['invalid, pd104, Pd-105*', ['invalid', 'pd104', 'Pd-105*']]:
        with pytest.raises(ValueError):
            utils.asisolist(strings)

        isolist = utils.asisolist(strings, allow_invalid=True)

        assert type(isolist) is dict
        assert len(isolist) == len(keys)
        for key in keys:
            assert key in isolist
            assert type(isolist[key]) is tuple
            assert len(isolist[key]) == 1
            assert isolist[key] == (key,)

    ########
    # dict #
    ########
    keys = 'Pd-102 Pd-104 Pd-105*'.split()
    dictionary = {'102pd': '102pd, pd104, pd-105*',
                  'pd104': '102pd, pd104, Pd-105*'.split(','),
                  'Pd-105*': '102pd, pd104, pd-105*'}

    isolist = utils.asisolist(dictionary)
    assert type(isolist) is dict
    assert len(isolist) == len(keys)
    for key in keys:
        assert key in isolist
        assert type(isolist[key]) is tuple
        assert len(isolist[key]) == len(keys)
        assert isolist[key] == tuple(keys)

    for k, v in isolist.items():
        assert type(k) is utils.Isotope
        for w in v:
            assert type(w) is utils.Isotope

    #################
    # ignore suffix #
    #################
    keys = 'Pd-102 Pd-104 Pd-105'.split()
    dictionary = {'102pd': '102pd, pd104, pd-105*',
                  'pd104': '102pd, pd104, Pd-105*'.split(','),
                  'Pd-105*': '102pd, pd104, pd-105*'}

    isolist = utils.asisolist(dictionary, without_suffix=True)
    assert type(isolist) is dict
    assert len(isolist) == len(keys)
    for key in keys:
        assert key in isolist
        assert type(isolist[key]) is tuple
        assert len(isolist[key]) == len(keys)
        assert isolist[key] == tuple(keys)

    for k, v in isolist.items():
        assert type(k) is utils.Isotope
        for w in v:
            assert type(w) is utils.Isotope

    #################
    # allow invalid #
    #################
    keys = 'invalid Pd-104 Pd-105*'.split()
    dictionary = {'invalid': 'invalid, pd104, pd-105*',
                  'pd104': 'invalid, pd104, Pd-105*'.split(','),
                  'Pd-105*': 'invalid, pd104, pd-105*'}

    with pytest.raises(ValueError):
        utils.asisolist(dictionary)

    isolist = utils.asisolist(dictionary, allow_invalid=True)

    assert type(isolist) is dict
    assert len(isolist) == len(keys)
    for key in keys:
        assert key in isolist
        assert type(isolist[key]) is tuple
        assert len(isolist[key]) == len(keys)
        assert isolist[key] == tuple(keys)

def test_askeyarray():
    keys = utils.asisotopes('Pd-102, Pd-104, Pd-105*')
    values = [1,2,3]
    array = utils.askeyarray(values, keys)

    assert isinstance(array, np.ndarray)
    assert array.ndim == 1
    assert array.shape == (1,)
    for i, (name, dt) in enumerate(array.dtype.fields.items()):
        assert dt[0] == np.int64
        assert name == keys[i]
    assert keys == array.dtype.names

    keys = utils.asisotopes('Pd-102, Pd-104, Pd-105*')
    values = [[1, 2, 3]]
    array = utils.askeyarray(values, keys)

    assert isinstance(array, np.ndarray)
    assert array.ndim == 1
    assert array.shape == (1,)
    for i, (name, dt) in enumerate(array.dtype.fields.items()):
        assert dt[0] == np.int64
        assert name == keys[i]
    assert keys == tuple(array.dtype.names)

    keys = utils.asisotopes('Pd-102, Pd-104, Pd-105*')
    values = [[1, 2, 3], [11, 12, 13], [21, 22, 23], [31, 32, 33]]
    array = utils.askeyarray(values, keys)

    assert isinstance(array, np.ndarray)
    assert array.ndim == 1
    assert array.shape == (4,)
    for i, (name, dt) in enumerate(array.dtype.fields.items()):
        assert dt[0] == np.int64
        assert name == keys[i]
    assert keys == tuple(array.dtype.names)

    #########
    # dtype #
    #########
    keys = utils.asisotopes('Pd-102, Pd-104, Pd-105*')
    values = [[1, 2, 3], [11, 12, 13], [21, 22, 23], [31, 32, 33]]
    array = utils.askeyarray(values, keys, dtype=np.float64)

    assert isinstance(array, np.ndarray)
    assert array.ndim == 1
    assert array.shape == (4,)
    for i, (name, dt) in enumerate(array.dtype.fields.items()):
        assert dt[0] == np.float64
        assert name == keys[i]
    assert keys == tuple(array.dtype.names)

    ##########
    # errors #
    ##########
    keys = utils.asisotopes('Pd-102, Pd-104, Pd-105*')
    values = [1, 2, 3, 4]
    with pytest.raises(ValueError):
        utils.askeyarray(values, keys)

    keys = utils.asisotopes('Pd-102, Pd-104, Pd-105*')
    values = [[1, 2, 3, 4], [11, 12, 13, 14], [21, 22, 23, 24], [31, 32, 33, 34]]
    with pytest.raises(ValueError):
        utils.askeyarray(values, keys)

def test_asarray():
    value = 1
    array = utils.asarray(value)

    assert isinstance(array, np.ndarray)
    assert array.ndim == 0
    assert array.shape == ()
    assert array.dtype == np.int64
    assert_equal(array, np.array(value))

    value = [1, 2, 3, 4, 5]
    array = utils.asarray(value)

    assert isinstance(array, np.ndarray)
    assert array.ndim == 1
    assert array.shape == (5,)
    assert array.dtype == np.int64
    assert_equal(array, np.array(value))

    #########
    # dtype #
    #########

    value = 1
    array = utils.asarray(value, dtype=np.float64)

    assert isinstance(array, np.ndarray)
    assert array.ndim == 0
    assert array.shape == ()
    assert array.dtype == np.float64
    assert_equal(array, np.array(value))

    value = [1, 2, 3, 4, 5]
    array = utils.asarray(value, dtype=np.float64)

    assert isinstance(array, np.ndarray)
    assert array.ndim == 1
    assert array.shape == (5,)
    assert array.dtype == np.float64
    assert_equal(array, np.array(value))


    ###########
    # Strings #
    ###########
    value = 'one'
    string = utils.asarray(value)

    assert type(string) is str
    assert string == value

    for value in [['one', 'two', 'three'], ('one', 'two', 'three'), np.array(['one', 'two', 'three'])]:
        strings = utils.asarray(value)

        assert isinstance(strings, tuple)
        for i, item in enumerate(strings):
            assert type(item) is str
            assert item == str(value[i])


    ##########
    # saving #
    ##########
    value = 'one'
    array = utils.asarray(value, saving=True)

    assert isinstance(array, np.ndarray)
    assert array.ndim == 0
    assert array.shape == ()
    assert array.dtype.type == np.bytes_
    assert_equal(array, np.array(value, dtype=np.bytes_))

    value = ['one', 'two', 'three']
    array = utils.asarray(value, saving=True)

    assert isinstance(array, np.ndarray)
    assert array.ndim == 1
    assert array.shape == (3,)
    assert array.dtype.type == np.bytes_
    assert_equal(array, np.array(value, dtype=np.bytes_))

def test_model_eval():
    class Item:
        a = 'A'
        b = 3.6

    eval = utils.model_eval
    dattrs = {"a": 'A', 'b': 3.6}
    oattrs = Item()

    # == and !=
    for attrs in [oattrs, dattrs]:
        result = eval.eval_where(attrs, 'a == A')
        assert result is False

        result = eval.eval_where(attrs, 'a != A')
        assert result is True

        result = eval.eval_where(attrs, '.a == A')
        assert result is True

        result = eval.eval_where(attrs, '.a != A')
        assert result is False

        result = eval.eval_where(attrs, '.a == {A}', A='x')
        assert result is False

        result = eval.eval_where(attrs, '.a != {A}', A='x')
        assert result is True

        result = eval.eval_where(attrs, '.a == {A}', A='A')
        assert result is True

        result = eval.eval_where(attrs, '.a != {A}', A='A')
        assert result is False

        result = eval.eval_where(attrs, '.b == 3.6')
        assert result is True

        result = eval.eval_where(attrs, '.b != 3.6')
        assert result is False

        result = eval.eval_where(attrs, '.c != c')
        assert result is False

        result = eval.eval_where(attrs, '{c} != c')
        assert result is False

    # <, >, <=, >=
    for attrs in [oattrs, dattrs]:
        result = eval.eval_where(attrs, '.b > 3')
        assert result is True

        result = eval.eval_where(attrs, '.b > 3.6')
        assert result is False

        result = eval.eval_where(attrs, '.b >= 3.6')
        assert result is True

        result = eval.eval_where(attrs, '3 < .b')
        assert result is True

        result = eval.eval_where(attrs, '3.6 < .b')
        assert result is False

        result = eval.eval_where(attrs, '3.6 <= .b')
        assert result is True

    # IN and NOT IN
    for attrs in [oattrs, dattrs]:
        result = eval.eval_where(attrs, '.a == A & .b > 3 & x IN {arg}', arg=['x', 'y', 'z'])
        assert result is True

        result = eval.eval_where(attrs, '.a != A & .b > 3 & x IN {arg}', arg=['x', 'y', 'z'])
        assert result is False

        result = eval.eval_where(attrs, '.a == A & .b < 3 & x IN {arg}', arg=['x', 'y', 'z'])
        assert result is False

        result = eval.eval_where(attrs, '.a == A & .b > 3 & x NOT IN {arg}', arg=['x', 'y', 'z'])
        assert result is False

    ##########
    # Errors #
    ##########
    for attrs in [oattrs, dattrs]:
        with pytest.raises(ValueError):
            eval.eval_where(attrs, '.a = A')

def test_select_isolist():
    keys = utils.asisotopes('Pd-102, Pd-104, Pd-105*')
    values = [[21, 41, 51],
              [22, 42, 52],
              [23, 43, 53],
              [24, 44, 54]]
    a = utils.askeyarray(values, keys)

    isolist = ('Pd-104', 'Pd-105', 'Pd-102')
    b = utils.select_isolist('104pd, 105pd, 102pd', a)
    assert isinstance(b, np.ndarray)
    assert b.dtype.names == isolist
    assert b.shape == (4,)
    assert_equal(b['Pd-104'], a['Pd-104'])
    assert_equal(b['Pd-105'], np.zeros(4))
    assert_equal(b['Pd-102'], a['Pd-102'])

    b = utils.select_isolist(isolist, a)
    assert isinstance(b, np.ndarray)
    assert b.dtype.names == isolist
    assert b.shape == (4,)
    assert_equal(b['Pd-104'], a['Pd-104'])
    assert_equal(b['Pd-105'], np.zeros(4))
    assert_equal(b['Pd-102'], a['Pd-102'])

    isolist = {'Pd-104': '104pd',
               'Pd-105': ['105pd*', 'pd105'],
               'Pd-102': '102pd, 104pd',
               'Pd-106*': [],
               'Pd-108': '',
               'Pd-110*': '102pd, 104pd, 105pd, 105pd*, 106pd',}
    b = utils.select_isolist(isolist, a)
    assert isinstance(b, np.ndarray)
    assert b.dtype.names == tuple(isolist.keys())
    assert b.shape == (4,)
    assert_equal(b['Pd-104'], a['Pd-104'])
    assert_equal(b['Pd-105'], a['Pd-105*'])
    assert_equal(b['Pd-102'], a['Pd-102']+a['Pd-104'])
    assert_equal(b['Pd-106*'], np.zeros(4))
    assert_equal(b['Pd-108'], np.zeros(4))
    assert_equal(b['Pd-110*'], a['Pd-102'] + a['Pd-104']+ a['Pd-105*'])







