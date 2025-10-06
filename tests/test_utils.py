import pytest
from simple import utils
import numpy as np
from numpy.testing import assert_equal

from simple.utils import DefaultKwargs


class TestDefaultKwargs:

    def test_set_default_kwargs1(self):
        # Test the standard implementation
        from simple.utils import set_default_kwargs, add_shortcut

        # Test a simple function without a kwargs item or **kwargs
        # Extra kwargs should just be discarded in this scenario

        @add_shortcut('test1', two=-2, four=-4)
        @set_default_kwargs()
        def test_func1(one, two, three=3, four=4):
            return (one, two, three, four)

        assert test_func1(11, 22, 33) == (11, 22, 33, 4)
        assert test_func1(11, 22, four=44, five=55) == (11, 22, 3, 44)
        assert test_func1.test1(11, four=44, five=55) == (11, -2, 3, 44)

        # Test case with a **kwargs but no kwargs item
        # All undefined args should be in **kwargs

        @add_shortcut('test1', two=-2, four=-4)
        @set_default_kwargs()
        def test_func2(one, two, three=3, four=4, **kwargs):
            return (one, two, three, four, kwargs)

        assert test_func2(11, 22, 33) == (11, 22, 33, 4, {})
        assert test_func2(11, 22, four=44, five=55) == (11, 22, 3, 44, dict(five=55))
        assert test_func2.test1(11, four=44, five=55) == (11, -2, 3, 44, dict(five=55))

        # Test case with a kwargs item but noo **kwargs
        # All args should be in the kwargs item

        @add_shortcut('test1', two=-2, four=-4)
        @set_default_kwargs()
        def test_func3(one, two, three=3, four=4, kwargs=None):
            return (one, two, three, four, kwargs)

        assert test_func3(11, 22, 33) == (11, 22, 33, 4,
                                          dict())
        assert test_func3(11, 22, four=44, five=55) == (11, 22, 3, 44,
                                          dict(five=55))
        assert test_func3.test1(11, four=44, five=55) == (11, -2, 3, 44,
                                          dict(five=55))


        # Test case with a kwargs item and **kwargs
        # All args should be in the kwargs item and any undefined ones should also be present in **kwargs

        @add_shortcut('test1', two=-2, four=-4)
        @set_default_kwargs()
        def test_func3(one, two, three=3, four=4, kwargs=None, **kwargs_):
            return (one, two, three, four, kwargs, kwargs_)

        assert test_func3(11, 22, 33) == (11, 22, 33, 4,
                                          {},
                                          {})
        assert test_func3(11, 22, four=44, five=55) == (11, 22, 3, 44,
                                                        dict(five=55),
                                                        dict(five=55))
        assert test_func3.test1(11, four=44, five=55) == (11, -2, 3, 44,
                                                          dict(five=55),
                                                          dict(five=55))

    def test_set_default_kwargs2(self):
        # Test inheritance
        from simple.utils import set_default_kwargs, add_shortcut

        @add_shortcut('test1', two=-2, four=-4)
        @set_default_kwargs()
        def test_func1(one, two, three=3, four=4):
            return (one, two, three, four)

        @add_shortcut('test2', three=-30, five=-50)
        @set_default_kwargs(inherits_=test_func1)
        def test_func2(one, three, four=40, five=50, kwargs=None):
            return (one, three, four, five, kwargs)

        assert test_func2(11) == (11, 3, 40, 50,
                                  dict())
        assert test_func2(11, 33, four=44) == (11, 33, 44, 50,
                                          dict())

        # Should inherit the shortcut from func1
        assert test_func2.test1(11) == (11, 3, -4, 50,
                                  dict(two=-2))
        assert test_func2.test1(11, 33, four=44) == (11, 33, 44, 50,
                                               dict( two=-2))

        # Should take precedent over the inherited shortcut
        test_func2.add_shortcut('test1', three=-30, five=-50)

        assert test_func2.test1(11) == (11, -30, 40, -50,
                                        dict())
        assert test_func2.test1(11, 33, four=44) == (11, 33, 44, -50,
                                                     dict())

    def test_default_kwargs_Dict(self):
        # Test that the dict objects are inherited and that updates are propagated
        # Through to the different sub dictionaries

        from simple.utils import DefaultKwargs

        a = {'one': 1, 'two': 2}
        b = {'two': 22, 'three': 33}


        # Make sure dict objects are inherited by DefaultKwargs.Dict
        aa = a.copy()
        tt = DefaultKwargs.Dict(aa)
        assert tt == aa
        assert tt.get('two') == 2
        assert 'two' in aa
        assert tt.pop('two') == 2
        assert 'two' not in aa

        # Make sure the inherited dicts are updated
        tt['two'] = 222
        assert aa.get('two') == 222

        # Make sure the priority is correct
        aa, bb = a.copy(), b.copy()
        ab = a.copy(); ab.update(b)
        tt = DefaultKwargs.Dict(aa, bb)
        assert tt == ab
        assert tt.get('two') == 22
        assert 'two' in aa
        assert 'two' in bb
        assert tt.pop('two') == 22
        assert 'two' not in aa
        assert 'two' not in bb

        tt['two'] = 222
        assert aa.get('two') == 222
        assert bb.get('two') == 222

    def test_extract_fetch(self):
        from simple.utils import DefaultKwargs

        original = {'one': 1, 'two': 2, 'three': 3, 'one_two': 12, 'one_three': 13}

        def test_func(two, three, four):
            pass

        # Get & Pop
        d = DefaultKwargs.Dict(original.copy())
        assert d.get('two') == 2
        assert 'two' in d

        assert d.pop('three') == 3
        assert 'three' not in d

        # Fetch
        # Works like an advanced get. Items remain in the original dictionary
        expected = dict(two=2, three=33, five=55)
        assert d.get_many('two, three, four', three=33, five=55) == expected
        assert all(['two' in d, 'three' not in d, 'four' not in d, 'five' not in d])

        assert d.get_many('two three four', three=33, five=55) == expected
        assert all(['two' in d, 'three' not in d, 'four' not in d, 'five' not in d])

        assert d.get_many(['two', 'three', 'four'], three=33, five=55) == expected
        assert all(['two' in d, 'three' not in d, 'four' not in d, 'five' not in d])

        assert d.get_many(test_func, three=33, five=55) == expected
        assert all(['two' in d, 'three' not in d, 'four' not in d, 'five' not in d])

        assert d.get_many(prefix='one') == dict(two=12, three=13)
        assert d.get_many('one', prefix='one') == dict(one=1, two=12, three=13)
        assert d.get_many('one', prefix='one', remove_prefix=False) == dict(one=1, one_two=12, one_three=13)

        # Extract
        # Works like an advanced

    def test_kwargs_update_remove(self):
        # Test changing the default values for functions and shortcuts
        # [x] update
        # [x] remove
        # [x] clear

        from simple.utils import DefaultKwargs, set_default_kwargs, add_shortcut
        @add_shortcut('genvag', seven=777)
        @set_default_kwargs()
        def myfunc(one, two=2, three=3, *, four=4, six=6, kwargs=None):
            return one, two, three, four, six, kwargs

        assert myfunc.kwargs == dict(two=2, three=3, four=4, six=6)
        assert myfunc.genvag.kwargs == dict(two=2, three=3, four=4, six=6, seven=777)

        myfunc.update(two=22, four=44, seven=77)
        assert myfunc.kwargs == dict(two=22, three=3, four=44, six=6, seven=77)
        assert myfunc.genvag.kwargs == dict(two=22, three=3, four=44, six=6, seven=777)

        myfunc.genvag.update(two=222, five=555, six=666)
        assert myfunc.kwargs == dict(two=22, three=3, four=44, six=6, seven=77)
        assert myfunc.genvag.kwargs == dict(two=222, three=3, four=44, five=555, six=666, seven=777)

        myfunc.remove('three', 'four', 'seven')
        assert myfunc.kwargs == dict(two=22, three=3, four=4, six=6)
        assert myfunc.genvag.kwargs == dict(two=222, three=3, four=4, five=555, six=666, seven=777)

        myfunc.genvag.remove('three', 'seven')
        assert myfunc.kwargs == dict(two=22, three=3, four=4, six=6)
        assert myfunc.genvag.kwargs == dict(two=222, three=3, four=4, five=555, six=666)

        myfunc.clear()
        assert myfunc.kwargs == dict(two=2, three=3, four=4, six=6)
        assert myfunc.genvag.kwargs == dict(two=222, three=3, four=4, five=555, six=666)

        myfunc.genvag.clear()
        assert myfunc.kwargs == dict(two=2, three=3, four=4, six=6)
        assert myfunc.genvag.kwargs == dict(two=2, three=3, four=4, six=6)

    @pytest.mark.parametrize("args, kwargs", [
        ((), {'one': 111, 'three': 333, 'four': 444, 'five': 555, 'six': 666}),
        ((), {'one': 111, 'two': 222, 'four': 444, }),
        ((1111,), {}),
        ((1111,), {'three': 333, 'five': 555}),
        ((1111, 2222, 3333), {'three': 333, 'four': 444, 'five': 555, 'six': 666}),
        ((1111, 2222, 3333), {'two': 222, 'six': 666})
    ])
    @pytest.mark.parametrize('default_kwargs', [
        {},
        {'two': 22, 'three': 33, 'four': 44, 'five': 55, 'six': 66},
        {'two': 22, 'four': 44, 'six': 66, 'seven': 77},
        {'five': 55, 'six': 66, 'seven': 77}
    ])
    @pytest.mark.parametrize('shortcut_kwargs', [
        {},
        {'two': 0.2, 'three': 0.3, 'four': 0.4, 'five': 0.5, 'six': 0.6},
        {'two': 0.2, 'four': 0.4, 'six': 0.6, 'seven': 0.7},
        {'five': 0.5, 'six': 0.6, 'seven': 0.7}
    ])
    @pytest.mark.parametrize('inherited_kwargs', [
        {},
        {'two': 0.22, 'five': 0.55, 'six': 0.66},
        {'three': 0.33, 'four': 0.44, 'seven': 0.77},
        {'six': 0.66, 'seven': 0.77}
    ])
    def test_default_kwargs(self, args, kwargs, default_kwargs, shortcut_kwargs, inherited_kwargs):
        # This tests checks the following
        # [x] Standard usage of set_default_kwargs
        # [x] Standard usage of Shortcuts
        # [x] Standard usage of inherits with the same basic signature
        # [x] Inheritance of Shortcuts
        # [x] Overriding inherited shortcuts

        from simple.utils import DefaultKwargs, set_default_kwargs, add_shortcut

        @add_shortcut('genvag', **shortcut_kwargs)
        @set_default_kwargs(**default_kwargs)
        def myfunc(one, two=2, three=3, *, four=4, six=6, kwargs=None):
            return one, two, three, four, six, kwargs

        myfunc_pos_args = ['one', 'two', 'three']
        myfunc_kwargs = {'two': 2, 'three': 3, 'four': 4, 'six': 6}

        @set_default_kwargs(**inherited_kwargs, inherits_=myfunc)
        def otherfunc(one, two=2.2, three=3.3, *, five=5.5, seven=7.7, kwargs=None):
            return one, two, three, five, seven, kwargs

        otherfunc_pos_args = ['one', 'two', 'three']
        otherfunc_kwargs = {'two': 2.2, 'three': 3.3, 'five': 5.5, 'seven': 7.7}

        ##############
        ### myfunc ###
        ##############
        # Test main function
        if True:
            myfunc_default_kwargs = myfunc_kwargs.copy()
            myfunc_default_kwargs.update(default_kwargs)

            assert type(myfunc) is DefaultKwargs

            fkwargs = myfunc.kwargs
            assert type(fkwargs) is DefaultKwargs.Dict
            assert fkwargs is not myfunc.kwargs  # Creates a new one each time

            # Should contain all args with defaults either in the signature or set by the decorator
            assert len(fkwargs) == len(myfunc_default_kwargs)
            for k in myfunc_default_kwargs:
                assert fkwargs[k] == myfunc_default_kwargs[k]

            # Create expected result
            result_expected_kwargs = myfunc_default_kwargs.copy()
            result_expected_kwargs.update(kwargs)
            for i, arg in enumerate(myfunc_pos_args[:len(args)]): result_expected_kwargs[arg] = args[i]
            result_expected = (result_expected_kwargs.pop('one'), result_expected_kwargs.pop('two'),
                               result_expected_kwargs.pop('three'), result_expected_kwargs.pop('four'),
                               result_expected_kwargs.pop('six'), result_expected_kwargs)

            # Standard case
            result = myfunc(*args, **kwargs)
            assert type(result[-1]) is DefaultKwargs.Dict
            assert result == result_expected

            # passing kwargs item
            kwargs_instance = kwargs.copy()
            result = myfunc(*args, kwargs=kwargs_instance)
            assert type(result[-1]) is DefaultKwargs.Dict
            assert result == result_expected

            # These should be removed from the original dict
            for k in myfunc_kwargs:
                assert k not in kwargs_instance

        # Test shortcut
        if True:
            shortcut_default_kwargs = myfunc_default_kwargs.copy()
            shortcut_default_kwargs.update(shortcut_kwargs)

            assert type(myfunc.genvag) is DefaultKwargs.Shortcut

            fkwargs = myfunc.genvag.kwargs
            assert type(fkwargs) is DefaultKwargs.Dict
            assert fkwargs is not myfunc.genvag.kwargs  # Creates a new one each time

            # Should contain all args with defaults either in the signature or set by the decorator
            assert len(fkwargs) == len(shortcut_default_kwargs)
            for k in shortcut_default_kwargs:
                assert fkwargs[k] == shortcut_default_kwargs[k]

            # Create expected result
            result_expected_kwargs = shortcut_default_kwargs.copy()
            result_expected_kwargs.update(kwargs)
            for i, arg in enumerate(myfunc_pos_args[:len(args)]): result_expected_kwargs[arg] = args[i]
            result_expected = (result_expected_kwargs.pop('one'), result_expected_kwargs.pop('two'),
                               result_expected_kwargs.pop('three'), result_expected_kwargs.pop('four'),
                               result_expected_kwargs.pop('six'), result_expected_kwargs)

            # Standard case
            result = myfunc.genvag(*args, **kwargs)
            assert type(result[-1]) is DefaultKwargs.Dict
            assert result == result_expected

            # passing kwargs item
            kwargs_instance = kwargs.copy()
            result = myfunc.genvag(*args, kwargs=kwargs_instance)
            assert type(result[-1]) is DefaultKwargs.Dict
            assert result == result_expected

            # These should be removed from the original dict
            for k in myfunc_kwargs:
                assert k not in kwargs_instance

        #################
        ### otherfunc ###
        #################
        # Test main function
        if True:
            otherfunc_default_kwargs = myfunc_kwargs.copy()
            otherfunc_default_kwargs.update(default_kwargs)
            otherfunc_default_kwargs.update(otherfunc_kwargs)
            otherfunc_default_kwargs.update(inherited_kwargs)

            assert type(myfunc) is DefaultKwargs

            fkwargs = otherfunc.kwargs
            assert type(fkwargs) is DefaultKwargs.Dict
            assert fkwargs is not myfunc.kwargs  # Creates a new one each time

            # Should contain all args with defaults either in the signature or set by the decorator
            assert len(fkwargs) == len(otherfunc_default_kwargs)
            for k in otherfunc_default_kwargs:
                assert fkwargs[k] == otherfunc_default_kwargs[k]

            # Create expected result
            result_expected_kwargs = otherfunc_default_kwargs.copy()
            result_expected_kwargs.update(kwargs)
            for i, arg in enumerate(otherfunc_pos_args[:len(args)]): result_expected_kwargs[arg] = args[i]
            result_expected = (result_expected_kwargs.pop('one'), result_expected_kwargs.pop('two'),
                               result_expected_kwargs.pop('three'), result_expected_kwargs.pop('five'),
                               result_expected_kwargs.pop('seven'), result_expected_kwargs)

            # Standard case
            result = otherfunc(*args, **kwargs)
            assert type(result[-1]) is DefaultKwargs.Dict
            assert result == result_expected

            # passing kwargs item
            kwargs_instance = kwargs.copy()
            result = otherfunc(*args, kwargs=kwargs_instance)
            assert type(result[-1]) is DefaultKwargs.Dict
            assert result == result_expected

            # These should be removed from the original dict
            for k in otherfunc_kwargs:
                assert k not in kwargs_instance

        # Test shortcut
        if True:
            shortcut_default_kwargs = otherfunc_default_kwargs.copy()
            shortcut_default_kwargs.update(shortcut_kwargs)

            assert type(otherfunc.genvag) is DefaultKwargs.Shortcut

            fkwargs = otherfunc.genvag.kwargs
            assert type(fkwargs) is DefaultKwargs.Dict
            assert fkwargs is not otherfunc.genvag.kwargs  # Creates a new one each time

            # Should contain all args with defaults either in the signature or set by the decorator
            assert len(fkwargs) == len(shortcut_default_kwargs)
            for k in shortcut_default_kwargs:
                assert fkwargs[k] == shortcut_default_kwargs[k]

            # Create expected result
            result_expected_kwargs = shortcut_default_kwargs.copy()
            result_expected_kwargs.update(kwargs)
            for i, arg in enumerate(otherfunc_pos_args[:len(args)]): result_expected_kwargs[arg] = args[i]
            result_expected = (result_expected_kwargs.pop('one'), result_expected_kwargs.pop('two'),
                               result_expected_kwargs.pop('three'), result_expected_kwargs.pop('five'),
                               result_expected_kwargs.pop('seven'), result_expected_kwargs)

            # Standard case
            result = otherfunc.genvag(*args, **kwargs)
            assert type(result[-1]) is DefaultKwargs.Dict
            assert result == result_expected

            # passing kwargs item
            kwargs_instance = kwargs.copy()
            result = otherfunc.genvag(*args, kwargs=kwargs_instance)
            assert type(result[-1]) is DefaultKwargs.Dict
            assert result == result_expected

            # These should be removed from the original dict
            for k in otherfunc_kwargs:
                assert k not in kwargs_instance

        # Test shortcut override
        if True:
            otherfunc.add_shortcut('genvag')
            shortcut_default_kwargs = otherfunc_default_kwargs.copy()
            shortcut_default_kwargs.update({})

            assert type(otherfunc.genvag) is DefaultKwargs.Shortcut

            fkwargs = otherfunc.genvag.kwargs
            assert type(fkwargs) is DefaultKwargs.Dict
            assert fkwargs is not otherfunc.genvag.kwargs  # Creates a new one each time

            # Should contain all args with defaults either in the signature or set by the decorator
            assert len(fkwargs) == len(shortcut_default_kwargs)
            for k in shortcut_default_kwargs:
                assert fkwargs[k] == shortcut_default_kwargs[k]

            # Create expected result
            result_expected_kwargs = shortcut_default_kwargs.copy()
            result_expected_kwargs.update(kwargs)
            for i, arg in enumerate(otherfunc_pos_args[:len(args)]): result_expected_kwargs[arg] = args[i]
            result_expected = (result_expected_kwargs.pop('one'), result_expected_kwargs.pop('two'),
                               result_expected_kwargs.pop('three'), result_expected_kwargs.pop('five'),
                               result_expected_kwargs.pop('seven'), result_expected_kwargs)

            # Standard case
            result = otherfunc.genvag(*args, **kwargs)
            assert type(result[-1]) is DefaultKwargs.Dict
            assert result == result_expected

            # passing kwargs item
            kwargs_instance = kwargs.copy()
            result = otherfunc.genvag(*args, kwargs=kwargs_instance)
            assert type(result[-1]) is DefaultKwargs.Dict
            assert result == result_expected

            # These should be removed from the original dict
            for k in otherfunc_kwargs:
                assert k not in kwargs_instance



def test_get_last_attr():
    class Item:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                self.__setattr__(k, v)

    a = Item(name = 'A', b = {'name': 'B', 'c': Item(name='C')})

    result = utils.get_last_attr(a, '.name')
    assert result == 'A'

    result = utils.get_last_attr(a, 'name')
    assert result == 'A'

    result = utils.get_last_attr(a, '.b')
    assert result is a.b

    result = utils.get_last_attr(a, 'b')
    assert result is a.b

    result = utils.get_last_attr(a, '.b.name')
    assert result is 'B'

    result = utils.get_last_attr(a, 'b.name')
    assert result is 'B'

    result = utils.get_last_attr(a, '.b.c')
    assert result is a.b['c']

    result = utils.get_last_attr(a, 'b.c')
    assert result is a.b['c']

    result = utils.get_last_attr(a, '.b.c.name')
    assert result == 'C'

    result = utils.get_last_attr(a, 'b.c.name')
    assert result == 'C'

def test_asisotope():
    for string in '102Pd, 102pd , Pd102,pd102 , 102-Pd, 102-pd, Pd-102, pd-102'.split(','):
        iso = utils.asisotope(string)
        assert type(iso) is utils.Isotope
        assert isinstance(iso, str)
        assert iso == 'Pd-102'
        assert iso.mass == '102'
        assert iso.symbol == 'Pd'
        assert iso.suffix == ''
        assert type(iso.element) is utils.Element
        assert iso.element == 'Pd'

    ###############
    # Test suffix #
    ###############
    for suffix in ['*', '_', ':', '*s', '_s', ':s', ' s']:
        string = 'Pd102' + suffix
        iso = utils.asisotope(string)
        assert type(iso) is utils.Isotope
        assert isinstance(iso, str)
        assert iso != 'Pd-102'
        assert iso.mass == '102'
        assert iso.symbol == 'Pd'
        assert iso.suffix == suffix
        assert type(iso.element) is utils.Element
        assert iso.element != 'Pd'

    iso = utils.asisotope('102Pd*')
    assert iso == 'Pd-102*'
    assert iso.suffix == '*'
    assert iso.element == 'Pd*'

    iso = utils.asisotope('102Pd*', without_suffix=True)
    assert iso == 'Pd-102'
    assert iso.suffix == ''
    assert iso.element == 'Pd'

    iso = utils.asisotope('102pd** ', without_suffix=False)
    assert iso == 'Pd-102**'
    assert iso.suffix == '**'
    assert iso.element == 'Pd**'

    iso = utils.asisotope('102pd** ', without_suffix=True)
    assert iso == 'Pd-102'
    assert iso.suffix == ''
    assert iso.element == 'Pd'

    iso = utils.asisotope(' Pd102 suffix ', without_suffix=False)
    assert iso == 'Pd-102 suffix'
    assert iso.suffix == ' suffix'
    assert iso.element == 'Pd suffix'

    iso = utils.asisotope(' Pd102 suffix ', without_suffix=True)
    assert iso == 'Pd-102'
    assert iso.suffix == ''
    assert iso.element == 'Pd'

    iso = utils.asisotope('102Pd*').without_suffix()
    assert iso == 'Pd-102'
    assert iso.suffix == ''
    assert iso.element == 'Pd'

    iso = utils.asisotope('102Pd').without_suffix()
    assert iso == 'Pd-102'
    assert iso.suffix == ''
    assert iso.element == 'Pd'

    iso = utils.asisotope('102Pd*', without_suffix=True).without_suffix()
    assert iso == 'Pd-102'
    assert iso.suffix == ''
    assert iso.element == 'Pd'

    ################
    # Test invalid #
    ################
    for string in 'invalid , Pd-1022, 1022pd, Pd102A, 102-Pdd, 102pd1, *102pd, 102pda, 102pd/ 102pd*/'.split(','):
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
    assert type(rat) is str
    assert rat == 'invalid/Pd-105*'

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

def test_select_isolist():
    keys = utils.asisotopes('invalid, ar40, fe56*, zn70, pd105*, pt196', allow_invalid=True)
    values = np.array([[-100, 40, 56, 70, 105, 196],
                       [-100 * 2, 40 * 2, 56 * 2, 70 * 2, 105 * 2, 196 * 2]], dtype=np.float64)
    array = utils.askeyarray(values, keys)

    isolist = {'ar40': 'ar40, zn70, ar40',
               'fe56*': 'fe56, zn70, pt196',
               'pd105': 'fe56*, pd105*, pt196*'}

    # array
    if True:
        correct_keys = utils.asisotopes(['ar40', 'fe56*', 'pd105'])
        correct_values = np.array([[150, 266, 161], [150 * 2, 266 * 2, 161 * 2]], dtype=np.float64)
        correct_array = utils.askeyarray(correct_values, correct_keys)

        result = utils.select_isolist(isolist, array)
        assert isinstance(result, np.ndarray)
        assert result.dtype.names == correct_keys
        np.testing.assert_array_equal(result, correct_array)

    # array - without_suffix=True
    if True:
        correct_keys = utils.asisotopes(['ar40', 'fe56', 'pd105'])
        correct_values = np.array([[150, 266, 196], [150 * 2, 266 * 2, 196 * 2]], dtype=np.float64)
        correct_array = utils.askeyarray(correct_values, correct_keys)

        result = utils.select_isolist(isolist, array, without_suffix=True)
        assert isinstance(result, np.ndarray)
        assert result.dtype.names == correct_keys
        np.testing.assert_array_equal(result, correct_array)

def test_askeyarray():
    keys = utils.asisotopes('Pd-102, Pd-104, Pd-105*')
    values = [1,2,3]
    array = utils.askeyarray(values, keys)

    assert isinstance(array, np.ndarray)
    assert array.ndim == 1
    assert array.shape == (1,)
    for i, (name, dt) in enumerate(array.dtype.fields.items()):
        assert np.issubdtype(dt[0], np.integer)
        assert name == keys[i]
    assert keys == array.dtype.names

    keys = utils.asisotopes('Pd-102, Pd-104, Pd-105*')
    values = [[1, 2, 3]]
    array = utils.askeyarray(values, keys)

    assert isinstance(array, np.ndarray)
    assert array.ndim == 1
    assert array.shape == (1,)
    for i, (name, dt) in enumerate(array.dtype.fields.items()):
        assert np.issubdtype(dt[0], np.integer)
        assert name == keys[i]
    assert keys == tuple(array.dtype.names)

    keys = utils.asisotopes('Pd-102, Pd-104, Pd-105*')
    values = [[1, 2, 3], [11, 12, 13], [21, 22, 23], [31, 32, 33]]
    array = utils.askeyarray(values, keys)

    assert isinstance(array, np.ndarray)
    assert array.ndim == 1
    assert array.shape == (4,)
    for i, (name, dt) in enumerate(array.dtype.fields.items()):
        assert np.issubdtype(dt[0], np.integer)
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
    assert np.issubdtype(array.dtype, np.integer)
    assert_equal(array, np.array(value))

    value = [1, 2, 3, 4, 5]
    array = utils.asarray(value)

    assert isinstance(array, np.ndarray)
    assert array.ndim == 1
    assert array.shape == (5,)
    assert np.issubdtype(array.dtype, np.integer)
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
        true = True
        false = False

    eval = utils.simple_eval
    dattrs = {"a": 'A', 'b': 3.6, 'true': True, 'false': False}
    oattrs = Item()

    # == and !=
    for attrs in [oattrs, dattrs]:
        result = eval.eval(attrs, 'a == A')
        assert result is False

        result = eval.eval(attrs, 'a != A')
        assert result is True

        result = eval.eval(attrs, '.a == A')
        assert result is True

        result = eval.eval(attrs, '.a != A')
        assert result is False

        result = eval.eval(attrs, '.a == {A}', A='x')
        assert result is False

        result = eval.eval(attrs, '.a != {A}', A='x')
        assert result is True

        result = eval.eval(attrs, '.a == {A}', A='A')
        assert result is True

        result = eval.eval(attrs, '.a != {A}', A='A')
        assert result is False

        result = eval.eval(attrs, '.b == 3.6')
        assert result is True

        result = eval.eval(attrs, '.b != 3.6')
        assert result is False

        result = eval.eval(attrs, '.c != c')
        assert result is False

        result = eval.eval(attrs, '{c} != c')
        assert result is False

    # <, >, <=, >=
    for attrs in [oattrs, dattrs]:
        result = eval.eval(attrs, '.b > 3')
        assert result is True

        result = eval.eval(attrs, '.b > 3.6')
        assert result is False

        result = eval.eval(attrs, '.b >= 3.6')
        assert result is True

        result = eval.eval(attrs, '3 < .b')
        assert result is True

        result = eval.eval(attrs, '3.6 < .b')
        assert result is False

        result = eval.eval(attrs, '3.6 <= .b')
        assert result is True

    # IN and NOT IN
    for attrs in [oattrs, dattrs]:
        result = eval.eval(attrs, '.a == A & .b > 3 & x IN {arg}', arg=['x', 'y', 'z'])
        assert result is True

        result = eval.eval(attrs, '.a != A & .b > 3 & x IN {arg}', arg=['x', 'y', 'z'])
        assert result is False

        result = eval.eval(attrs, '.a == A & .b < 3 & x IN {arg}', arg=['x', 'y', 'z'])
        assert result is False

        result = eval.eval(attrs, '.a == A & .b > 3 & x NOT IN {arg}', arg=['x', 'y', 'z'])
        assert result is False

        ####
        result = eval.eval(attrs, '.a == A | .b > 3 | x IN {arg}', arg=['x', 'y', 'z'])
        assert result is True

        result = eval.eval(attrs, '.a != A | .b > 3 | x IN {arg}', arg=['x', 'y', 'z'])
        assert result is True

        result = eval.eval(attrs, '.a != A | .b < 3 | x IN {arg}', arg=['x', 'y', 'z'])
        assert result is True

        result = eval.eval(attrs, '.a != A | .b < 3 | x NOT IN {arg}', arg=['x', 'y', 'z'])
        assert result is False

    # Single attr
    for attrs in [oattrs, dattrs]:
        result = eval.eval(attrs, 'text')
        assert result is False

        result = eval.eval(attrs, 'true')
        assert result is False

        result = eval.eval(attrs, 'True')
        assert result is True

        result = eval.eval(attrs, '.true')
        assert result is True

        result = eval.eval(attrs, 0)
        assert result is False

        result = eval.eval(attrs, 1)
        assert result is True

        result = eval.eval(attrs, '.b')
        assert result is True

        result = eval.eval(attrs, True)
        assert result is True

        result = eval.eval(attrs, False)
        assert result is False


def test_mask_eval():
    eval = utils.mask_eval

    correct = np.full(11, False)
    correct[10] = True

    result = eval.eval({}, 10, 11)
    np.testing.assert_array_equal(result, correct)

    result = eval.eval({}, -1, 11)
    np.testing.assert_array_equal(result, correct)

    result = eval.eval({}, '10', 11)
    np.testing.assert_array_equal(result, correct)

    result = eval.eval({}, '-1', 11)
    np.testing.assert_array_equal(result, correct)

    #--------------------------------
    correct = np.full(11, False)
    correct[slice(1,2,3)] = True

    result = eval.eval({}, slice(1,2,3), 11)
    np.testing.assert_array_equal(result, correct)

    result = eval.eval({}, '1:2:3', 11)
    np.testing.assert_array_equal(result, correct)

    # --------------------------------
    correct = np.full(11, False)
    correct[:10] = True

    result = eval.eval({}, ':10', 11)
    np.testing.assert_array_equal(result, correct)

    # --------------------------------
    correct = np.full(11, False)
    correct[-5:] = True

    result = eval.eval({}, '-5:', 11)
    np.testing.assert_array_equal(result, correct)

    # --------------------------------
    correct = np.full(11, True)

    result = eval.eval({}, '', 11)
    np.testing.assert_array_equal(result, correct)

    ##################################
    array = np.array([1, 2, 3, 4, 5])

    result = eval.eval({'data': array}, '.data > 3', 5)
    np.testing.assert_array_equal(result, np.array([False, False, False, True, True]))

    result = eval.eval({'data': array}, '3 >= {data}', 5, **{'data': array})
    np.testing.assert_array_equal(result, np.array([True, True, True, False, False]))

    result = eval.eval({'data': array}, 'data > 3', 5)
    np.testing.assert_array_equal(result, [False, False, False, False, False])

    result = eval.eval({'data': array}, '.data > 3 & .data < 5', 5)
    np.testing.assert_array_equal(result, np.array([False, False, False, True, False]))

    result = eval.eval({'data': array}, '.data > 3 | .data <= 2', 5)
    np.testing.assert_array_equal(result, np.array([True, True, False, True, True]))

    result = eval.eval({'data': array}, '-1 | .data <= 2', 5)
    np.testing.assert_array_equal(result, np.array([True, True, False, False, True]))

    result = eval.eval({'data': array}, '.data > 1 & :3', 5)
    np.testing.assert_array_equal(result, np.array([False, True, True, False, False]))

    result = eval.eval({'data': array}, '.data > 1 & :3 & 2', 5)
    np.testing.assert_array_equal(result, np.array([False, False, True, False, False]))

    with pytest.raises(ValueError):
        result = eval.eval({'data': array}, '.data > 1 & :3 | 2', 5)

    ############################
    result = eval.eval({'data': array}, [False, False, True, False, False], 5)
    np.testing.assert_array_equal(result, np.array([False, False, True, False, False]))

    # Incorrect shape
    result = eval.eval({'data': array}, [False, False, True, False], 5)
    np.testing.assert_array_equal(result, np.array([False, False, False, False, False]))

    result = eval.eval({'data': array}, [True], 5)
    np.testing.assert_array_equal(result, np.array([True, True, True, True, True]))

    # Non-zero floats have a boolean value of True
    result = eval.eval({'data': array}, 2.0, 5)
    np.testing.assert_array_equal(result, np.array([True, True, True, True, True]))

    # Zero value floats have a boolean value of False
    result = eval.eval({'data': array}, 0.0, 5)
    np.testing.assert_array_equal(result, np.array([False, False, False, False, False]))

    # None has a boolean value of True
    result = eval.eval({'data': array}, None, 5)
    np.testing.assert_array_equal(result, np.array([False, False, False, False, False]))

    ###################
    result = eval.eval({'data': array, 'i': 3, 'slice':slice(None, 2)}, '.i', 5)
    np.testing.assert_array_equal(result, np.array([False, False, False, True, False]))

    result = eval.eval({'data': array, 'i': 3, 'slice': slice(None, 2)}, '.slice', 5)
    np.testing.assert_array_equal(result, np.array([True, True, False, False, False]))

    result = eval.eval({'data': array, 'i': 3, 'slice': slice(None, 2)}, '.slice | .i', 5)
    np.testing.assert_array_equal(result, np.array([True, True, False, True, False]))

    ###############
    result = eval.eval({'data': array}, 'one', 5)
    np.testing.assert_array_equal(result, np.array([False, False, False, False, False]))

    result = eval.eval({'data': array}, '7', 5)
    np.testing.assert_array_equal(result, np.array([False, False, False, False, False]))

    result = eval.eval({'data': array}, 7, 5)
    np.testing.assert_array_equal(result, np.array([False, False, False, False, False]))

    ################
    result = eval.eval({'data': array}, None, 5)
    np.testing.assert_array_equal(result, np.array([False, False, False, False, False]))

    result = eval.eval({'data': array}, 'None', 5)
    np.testing.assert_array_equal(result, np.array([False, False, False, False, False]))

    result = eval.eval({'data': array}, False, 5)
    np.testing.assert_array_equal(result, np.array([False, False, False, False, False]))

    result = eval.eval({'data': array}, 'False', 5)
    np.testing.assert_array_equal(result, np.array([False, False, False, False, False]))

    result = eval.eval({'data': array}, True, 5)
    np.testing.assert_array_equal(result, np.array([True, True, True, True, True]))

    result = eval.eval({'data': array}, 'True', 5)
    np.testing.assert_array_equal(result, np.array([True, True, True, True, True]))

def test_EndlessList():
    l = utils.EndlessList([1, 2, 3])
    assert type(l) is utils.EndlessList
    assert l[0] == 1
    assert l[2] == 3
    assert l[3] == 1
    assert l[5] == 3
    assert l[7] == 2

    l2 = l[1:]
    assert type(l2) is utils.EndlessList
    assert l2[0] == 2
    assert l2[1] == 3
    assert l2[2] == 2
    assert l2[3] == 3




