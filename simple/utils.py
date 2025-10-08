import re, operator
import numpy as np
import logging
import yaml
import functools, itertools, inspect
import collections

SimpleLogger = logging.getLogger('SIMPLE')
SimpleLogger.addHandler(logging.StreamHandler())

logger = logging.getLogger('SIMPLE.utils')

__all__ = ['load_defaults', 'set_logging_level',
           'asarray', 'askeyarray', 'asisolist', 'get_isotopes_of_element',
           'aselement', 'aselements', 'asisotope', 'asisotopes', 'asratio', 'asratios']

UNITS = dict(mass = ['mass', 'massfrac', 'wt', 'wt%'],
             mole = ['mole', 'moles', 'mol', 'molfrac'])
"""
A dictionary containing the names associated with different unit types

Current unit types are:

- ``mass`` that represents data being stored in a mass unit or as mass fractions.
-  ``mole`` that represents data being stored in moles or as mole fractions.
"""

OptionalArg = object()

def set_logging_level(level):
    """
    Set the level of messages to be displayed.

    Options are: DEBUG, INFO, WARNING, ERROR.
    """
    if level.upper() == 'DEBUG':
        level = logging.DEBUG
    elif level.upper() == 'INFO':
        level = logging.INFO
    elif level.upper() == 'WARNING':
        level = logging.WARNING
    elif level.upper() == 'ERROR':
        level = logging.ERROR

    SimpleLogger.setLevel(level)

def parse_attrname(attrname):
    if attrname is None:
        return None
    else:
        return '.'.join([aa for a in attrname.split('.') if (aa := a.strip()) != ''])

def get_last_attr(item, attrname, default=OptionalArg):
    # Return the final attribute in the chain ``attrname`` starting from ``item``
    attrnames = parse_attrname(attrname).split('.')
    attr = item
    for i, name in enumerate(attrnames):
        try:
            if isinstance(attr, dict):
                attr = attr[name]
            else:
                attr = getattr(attr, name)
        except (AttributeError, KeyError):
            if default is not OptionalArg:
                # Log message?
                return default
            else:
                raise AttributeError(f'Item {item} has no attribute {".".join(attrnames[:i+1])}')
    return attr


class EndlessList(list):
    """
    A subclass of ``list`` that where the index will never go out of bounds. If a requested
    index is out of bounds, it will cycle around to the start of the list.

    Examples:
        >>> ls = simple.plot.Endlesslist(["a", "b", "c"])
        >>> ls[3]
        "a"
    """
    # Index will never go out of bounds. It will just start from the beginning if larger than the initial list.
    def __getitem__(self, index):
        if type(index) == slice:
            return EndlessList(super().__getitem__(index))
        else:

            return super().__getitem__(index % len(self))

class NamedDict(dict):
    """
    A subclass of a normal ``dict`` where item in the dictionary can also be accessed as attributes.

    Examples:
        >>> nd = simple.utils.NamedDict({'a': 1, 'b': 2, 'c': 3})
        >>> nd.a
        1
    """
    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def __setattr__(self, name, value, item=True):
        if item:
            return self.__setitem__(name, value)
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name):
        return self.__getitem__(name)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def setdefault(self, key, default_value):
        if key not in self:
            self[key] = default_value

        return self[key]


class DefaultKwargs:
    """
    A callable wrapper that provides dynamic default keyword argument management for functions.

    Allows attaching shortcuts with alternative default kwargs, inheritance from other `DefaultKwargs` instances,
    and matching function signatures while still accepting extended `kwargs` or `**kwargs_`. Optionally exposes the
    original wrapped function as `._func`.

    Args:
        func (Callable): The function to be wrapped.
        parent (DefaultKwargs): If supplied, all kwargs from this instance will be inherited unless overridden.
        **kwargs: Default keyword arguments that will be applied unless explicitly overridden.

    """

    class Shortcut:
        """
        A callabe shortcut with its own additional set of default keyword arguments.
        """

        def __init__(self, name, parent, kwargs):
            self._name = name
            self._parent = parent
            self._kwargs = kwargs

        def __repr__(self):
            return f'<DefaultKwargs.Shortcut "{self._name}" for function: {self._parent._func.__name__}>'

        def __call__(self, *args, kwargs=None, **kwargs_):
            return self._parent._call_func(self.kwargs, args, kwargs, kwargs_)

        @property
        def kwargs(self):
            """
            The effective keyword arguments for the current instance, with inheritance and overrides applied.
            """
            new_kwargs = self._parent.kwargs
            new_kwargs.update(self._kwargs)
            return new_kwargs

        def update(self, **kwargs):
            """
            Update the default kwargs of this shortcut.
            """
            self._kwargs.update(kwargs)

        def remove(self, *args):
            """
            Remove items from the default kwargs of this shortcut.
            """
            for arg in args:
                self._kwargs.pop(arg, None)

        def clear(self):
            """
            Removes all items from the default kwargs of this shortcut.
            """
            self._kwargs.clear()

    class DontUpdateDict:
        def __init__(self, dict):
            # Just to avoid the overhead of updating dicts without references.
            self.dict = dict

    class Dict(collections.abc.MutableMapping):
        """
        A dictionary-like wrapper for managing and merging multiple keyword argument dictionaries.

        This class allows combining several dictionaries into a single mapping. It enables key access, mutation, and deletion
        across all source dictionaries. It also supports extracting keys or key groups by prefix, optionally removing them
        in the process. Keys are resolved in insertion order.

        Used to support decorated functions where `kwargs` are passed and consumed progressively.

        Args:
            *kwargs (dict): One or more dictionaries to be combined and managed.
        """

        def __init__(self, *kwarg_dicts):
            self.dicts = [d for d in kwarg_dicts if
                          (isinstance(d, (dict, self.__class__)) and not isinstance(d, DefaultKwargs.DontUpdateDict))]
            self.combined = {}
            for d in kwarg_dicts:
                if isinstance(d, DefaultKwargs.DontUpdateDict):
                    self.combined.update(d.dict)
                elif isinstance(d, (dict, self.__class__)):
                    self.combined.update(d)
                elif d is None:
                    pass
                else:
                    raise TypeError(f'Items must be dictionaries or None (got {type(d)})')

        def __repr__(self):
            return f'DefaultKwargs.Dict({repr(self.combined)})'

        def __getitem__(self, key):
            return self.combined.__getitem__(key)

        def __setitem__(self, key, value):
            self.combined.__setitem__(key, value)
            for d in self.dicts:
                d.__setitem__(key, value)

        def __delitem__(self, key):
            self.combined.__delitem__(key)
            for d in self.dicts:
                if key in d:
                    d.__delitem__(key)

        def __iter__(self):
            return self.combined.__iter__()

        def __len__(self):
            return self.combined.__len__()

        def copy(self):
            return self.__class__(self.combined.copy())

        def _parse_keys(self, name, keys, allow_callabe=True):
            if isinstance(keys, (list, tuple)):
                new_keys = []
                for k in keys:
                    new_keys.extend(self._parse_keys(name, k, allow_callabe=allow_callabe))

            elif keys is None:
                new_keys = []
            elif isinstance(keys, str):
                if ',' in keys:
                    new_keys = keys.split(',')
                else:
                    new_keys = keys.split()

                new_keys = [k.strip() for k in new_keys]
            elif callable(keys) and allow_callabe:
                new_keys = [name
                        for name, param in inspect.signature(keys).parameters.items()
                        if (param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD or
                            param.kind == inspect.Parameter.KEYWORD_ONLY)]
            else:
                raise TypeError(f'Item "{keys}" is not a string')

            return new_keys

        def pop_many(self, keys=None, prefix=None, remove_prefix=True, **default_kwargs):
            """
            Created a new dictionary containing the matching keys.

            This will remove the corresponding item from the original dictionary and its subdictionaries.

            Args:
                keys (): Keys to be extracted. This can be either a comma or space seperated string, a list of
                    strings, or a callable where the named arguments will be used as keys.
                    Missing keys are ignored. Default values can be defined using the *default_kwargs* argument.
                prefix (): Any keyword with this prefix will be extracted. A ``"_"`` will be added to the end of the
                    prefix if not already present. Can be a comma or space seperated string, or a list of strings.
                remove_prefix ():  If ``True`` the prefix part of the keyword is removed from the key in the returned
                    dictionary.
                **default_kwargs (): Forms the basis of the returned dictionary. Extracted values will overwrite these
                    values.

            Returns:
                dict: A dictionary containing the extracted items.
            """
            return self._get_many(keys, prefix, remove_prefix, default_kwargs, True)

        def get_many(self, keys=None, prefix=None, remove_prefix=True, **default_kwargs):
            """
            Created a new dictionary containing the matching keys.

            This will not alter the original dictionary and its associated subdictionaries.

            Args:
                keys (): Keys to be extracted. This can be either a comma or space seperated string, a list of
                    strings, or a callable where the named arguments will be used as keys.
                    Missing keys are ignored. Default values can be defined using the *default_kwargs* argument.
                prefix (): Any keyword with this prefix will be extracted. A ``"_"`` will be added to the end of the
                    prefix if not already present. Can be a comma or space seperated string, or a list of strings.
                remove_prefix ():  If ``True`` the prefix part of the keyword is removed from the key in the returned
                    dictionary.
                **default_kwargs (): Forms the basis of the returned dictionary. Extracted values will overwrite these
                    values.

            Returns:
                dict: A dictionary containing the extracted items.
            """
            return self._get_many(keys, prefix, remove_prefix, default_kwargs, False)

        def _get_many(self, keys, prefix, remove_prefix, default_kwargs, delete):
            new_dict = self.__class__(default_kwargs)

            keys = self._parse_keys('keys', keys, allow_callabe=True)
            prefix = self._parse_keys('prefix', prefix, allow_callabe=False)

            for k in keys:
                if k in self:
                    if delete:
                        new_dict[k] = self.pop(k)
                    else:
                        new_dict[k] = self.get(k)

            for prfx in prefix:
                if prfx[-1] != '_': prfx += '_'

                for k in list(self.keys()):
                    if k[:len(prfx)] == prfx:
                        if remove_prefix:
                            name = k[len(prfx):]
                        else:
                            name = k
                        if delete:
                            new_dict[name] = self.pop(k)
                        else:
                            new_dict[name] = self.get(k)

            return new_dict

    def __init__(self, func, default_kwargs, inherits):
        self._func = func
        self._kwargs = default_kwargs

        if inherits:
            if type(inherits) is not self.__class__:
                raise TypeError('Can only inherit from other DefaultKwargs instances')
            elif inherits is self:
                raise ValueError('Cannot inherit from itself')
            else:
                self._inherits_from = inherits
        else:
            self._inherits_from = None

        self._shortcuts = {}

        parameters = inspect.signature(func).parameters

        if any([param.kind == inspect.Parameter.POSITIONAL_ONLY
                for name, param in parameters.items()]):
            raise ValueError('The DefaultKwargs wrapper does not work on functions with positional only arguments')


        # All the arguments that can be positional
        self._func_pos_names = [name
                                for name, param in parameters.items()
                                if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD]

        # All the arguments that can be used as keywords
        self._func_kw_names = [name
                               for name, param in parameters.items()
                               if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
                               or param.kind == inspect.Parameter.KEYWORD_ONLY]

        # All the arguments with default values.
        self._func_kw_defaults = {name: param.default
                                  for name, param in parameters.items()
                                  if param.default is not inspect.Parameter.empty}

        if 'kwargs' in self._func_kw_names:
            self._has_kwargs_arg = True
            self._func_kw_names.remove('kwargs')
            self._func_kw_defaults.pop('kwargs')
        else:
            self._has_kwargs_arg = False

        if any([param.kind == inspect.Parameter.VAR_KEYWORD for name, param in parameters.items()]):
            self._has_star_star_arg = True
        else:
            self._has_star_star_arg = False

        functools.update_wrapper(self, func)

    def __repr__(self):
        return f'<DefaultKwargs for function: {self._func.__name__}>'

    def __getattr__(self, name, inherit_from=None):
        if name in self._shortcuts:
            return self.Shortcut(name, inherit_from or self, self._shortcuts[name])

        elif self._inherits_from:
            try:
                return self._inherits_from.__getattr__(name, inherit_from=inherit_from or self)
            except AttributeError:
                pass

        raise AttributeError(f'{self._func.__name__} does not have, or inherits, a shortcut called {name}')

    def __call__(self, *args, kwargs=None, **kwargs_):
        """
        Calls the wrapped function with arguments, resolving default kwargs and matching against the function signature.
        Values in `kwargs` take precedence over `**kwargs_`.
        """
        return self._call_func(self.kwargs, args, kwargs, kwargs_)

    def _call_func(self, default_kwargs, args, kwargs_item, kwargs_):
        # The order of priority is: *args, **kwargs, kwargs_item, defaults_inherited, (defaults_wrapper, default_signature)
        if len(args) > len(self._func_kw_names):
            raise ValueError(f'Too many positional arguments ({len(args)}). The function `{self._func.__name__}` '
                             f'only accepts {len(self._func_kw_names)} positional arguments')

        # Combine all the arguments
        all_kwargs = self.Dict(self.DontUpdateDict(default_kwargs),
                               kwargs_item,
                               self.DontUpdateDict(kwargs_),
                               self.DontUpdateDict(dict(zip(self._func_pos_names, args))))

        # Fetch all the names kwargs
        kw_kwargs = all_kwargs.pop_many(self._func_kw_names)

        # Sort out what which arguments to send.
        if self._has_star_star_arg:
            if self._has_kwargs_arg:
                return self._func(kwargs=all_kwargs, **kw_kwargs, **all_kwargs)
            else:
                return self._func(**kw_kwargs, **all_kwargs)
        else:
            if self._has_kwargs_arg:
                return self._func(kwargs=all_kwargs, **kw_kwargs)
            else:
                return self._func(**kw_kwargs)

    def add_shortcut(self, name,  **kwargs):
        """
        Defines a named variant of this function with modified defaults. Accessible via attribute syntax.
        """
        self._shortcuts[name] = kwargs

    @property
    def kwargs(self):
        """
        The effective keyword arguments for the current instance, with inheritance and overrides applied.
        """
        if self._inherits_from:
            new_kwargs = self._inherits_from.kwargs
        else:
            new_kwargs = self.Dict()

        new_kwargs.update(self._func_kw_defaults)
        new_kwargs.update(self._kwargs)

        return new_kwargs

    def update(self, d=None, **kwargs):
        """
        Update the default kwargs of this function.
        """
        if d is not None:
            self._kwargs.update(d)
        if kwargs:
            self._kwargs.update(kwargs)

    def remove(self, *args):
        """
        Remove items from the default kwargs of this function.

        Note: This will not delete default kwargs defined in the signature.
        """
        for arg in args:
            self._kwargs.pop(arg, None)

    def clear(self):
        """
        Removes all items from the default kwargs of this function.

        Note: This will not delete default kwargs defined in the signature.
        """
        self._kwargs.clear()


def set_default_kwargs(*, inherits_ = None, **default_kwargs):
    """
    Decorator sets the default keyword arguments for the function. It wraps the function so that the
    default kwargs are always passed to the function.

    The kwargs can be accessed from ``<func>.kwargs``. To update the dictionary use the function
    ``update__kwargs`` attached to the return function.
    """
    def decorator(func):
        return DefaultKwargs(func, default_kwargs, inherits_)
    return decorator

def add_shortcut(name, **shortcut_kwargs):
    """
    Decorator that creates a named shortcut for a given `DefaultKwargs` instance.
    """
    def inner(func):
        if not isinstance(func, DefaultKwargs):
            raise TypeError("First call `set_default_kwargs` on function before adding shortcuts")
        else:
            func.add_shortcut(name, **shortcut_kwargs)

        return func
    return inner

def deprecation_warning(message):
    """
    Used to signal that a function is deprecated. Will raise a warning with *message* when the function is used.
    """
    def inner(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.warning(f'DeprecationWarning: {message}')
            return func(*args, **kwargs)
        return wrapper
    return inner

def load_defaults(filename: str):
    """
    Loads default arguments for functions from a YAML formatted file.

    To use a set of default values, unpack the arguments in the function call (See example).

    You can still arguments and keyword arguments as normal as long as they are not included in the default dictionary.

    Returns:
        A named dictionary mapping the prefixes given in the yaml file to another dictionary mapping the arguments
        to the specified values.

    Examples:
        The file ``default.yaml`` is expected to look like this:
        ```
        somefunction:
            arg: value
            listarg:
                - first thing in list
                - second thing in list

        anotherfunction:
            arg: value
        ```

        It can be used like this
        >>> defaults = simple.load_defaults('defaults.yaml')
        >>> somefunction(**defaults['somefunction']) # Unpack arguments into function call


    """
    return dict(yaml.safe_load(open(filename, 'r').read()))

def askeyarray(values, keys, dtype=None):
    """
    Returns a numpy array where the columns can be accessed by the column key.

    Args:
        values (): An array consisting of 2 dimensions where first dimension is the row and the second
            dimension is the column.
        keys (): The keys for each column in ``values``. Must be the same length as the second dimension of ``values``.
            of ``array``.
        dtype (): The values type of the returned array. All columns will have the same dtype.

    **Notes**
    If ``values`` has less then 2 dimensions then it is assumed to represent a single row of values.

    It is not possible to save this type of array in hdf5 files if they have more than a few hundred columns.

    Examples:
        >>> a = simple.askeyarray([[1,2,3],[4,5,6]], ['Pd-104','Pd-105','Pd-106']); a
        array([(1, 2, 3), (4, 5, 6)],
              dtype=[('Pd-104', '<i8'), ('Pd-105', '<i8'), ('Pd-106', '<i8')])
        >>> a['Pd-104']
        array([1, 4])

    """
    if type(keys) is str:
        keys = [k.strip() for k in keys.split(',')]

    a = np.asarray(values, dtype=dtype)
    dtype = [(k, a.dtype) for k in keys]

    if a.ndim < 2:
        a = a.reshape((-1, a.size))
    elif a.ndim > 2:
        raise ValueError('``values`` cannot have more than 2 dimensions')

    if a.shape[1] != len(keys):
        raise ValueError(
            f'item (r:{a.shape[0]}, c:{a.shape[1]}) must have same number of columns as there are keys ({len(keys)})')

    return np.array([tuple(r) for r in a], dtype=dtype)

def asarray(values, dtype=None, saving=False):
    """
    Convert ``data`` to a numpy array.

    If ``data`` is a string or a sequence of strings and ``saving=False``, either a single string or a tuple
    of string will be returned. If ``saving`` is ``True`` the values will be converted to an array with a byte dtype.
    This ensures values are compatible with the hdf5 library.

    Arrays with a ``bytes`` dtype will automatically be converted to the ``str`` dtype. If ``saving`` is ``False`` then
    this values will be converted to either a string or a tuple of strings (see above).

    Args:
        values (): An values like object.
        dtype (): The data type of the returned values.
        saving (): Should be ``True`` is the data is to be saved in a hdf5 file.

    """
    values = np.asarray(values, dtype=dtype)

    if values.dtype.type is np.bytes_:
        values = values.astype(np.str_)

    if not saving and values.dtype.type is np.str_:
        values = values.tolist()
        if type(values) is list:
            values = tuple(values)

    if saving and values.dtype.type is np.str_:
        values = values.astype(np.bytes_)

    return values

def select_isolist(isolist, data, *, without_suffix=False):
    """
    Creates a subselection of ``data`` containing only the isotopes in ``isolist``.

    If multiple input isotopes are given for an output isotope in ``isolist`` the values of the input isotopes will
    be added together. Any isotopes missing from ``data`` will be given a value of 0.

    When using this function to account for radioactive decay you want the unit of ``data`` to be in moles.

    Args:
        isolist (): Either a list of isotopes to be selected or a dictionary consisting of the
             final isotope mapped to a list of isotopes to be added together for this isotope.
        data (): A key array from which the subselection will be made.
        without_suffix (): If ``True`` the suffix will be removed from all isotope strings in ``isolist``.


    Returns:
        A new key array containing only the isotopes in ``isolist``.

    """
    isolist = asisolist(isolist, without_suffix=without_suffix)
    data = np.asarray(data)

    new_data = []
    missing_isotopes = []

    if data.dtype.names is None:
        raise ValueError('``data`` must be a key arrray')

    for mainiso, inciso in isolist.items():
        value = np.zeros(data.shape)

        for iso in inciso:
            if iso in data.dtype.names:
                value += data[iso]
            else:
                missing_isotopes.append(iso)

        new_data.append(value)

    result = askeyarray(np.array(new_data).transpose(), isolist.keys())

    if len(missing_isotopes) != 0:
        logger.warning(f'Missing isotopes set to 0: {", ".join(missing_isotopes)}')

    return result

#######################
### Isotope strings ###
#######################
class Element(str):
    RE = r'([a-zA-Z]{1,2})([*_:][a-zA-Z0-9_]*)?'
    def __new__(cls, string, without_suffix=False):
        string = string.strip()
        m = re.fullmatch(cls.RE, string)
        if m:
            element, suffix = m.group(1).capitalize(), m.group(2) or ''
        else:
            raise ValueError(f"String '{string}' is not a valid element")

        return cls._new_(element, '' if without_suffix else suffix)

    @classmethod
    def _new_(cls, symbol, suffix):
        self = super().__new__(cls, f"{symbol}{suffix}")
        self.symbol = symbol
        self.suffix = suffix
        return self

    def latex(self, dollar=True):
        """
        Returns a latex representation of the string e.g. Pd -> $\mathrm{Pd}$

        Args:
            dollar (bool): Whether to include the bracketing ``$`` signs.

        """
        string = fr"\mathrm{{{self.symbol}{self.suffix}}}"
        if dollar:
            return f"${string}$"
        else:
            return string

    def without_suffix(self):
        """
        Return a new element string without the suffix.
        """
        return self._new_(self.symbol, '')

class Isotope(str):
    """
    A subclass of string representing an isotope using the format ``<element symbol>-<mass number><suffix>`` e.g.
    ``Pd-105``.

    The order of the element symbol and mass number in ``string`` is not important, but they must proceed the suffix.
    The element symbol and mass number can optionally be seperated by ``-``. The case of the element symbol is not
    considered.

    Args:
        string (str): A string element symbol and a mass number.
        without_suffix (bool): If ``True`` the suffix part of the ``string`` is ignored.

    Attributes:
        element (str): The element symbol of the isotope
        mass (str): The mass number of the isotope
        suffix (str): The suffix of the isotope

    Raises:
        ValueError: If ``string`` does not represent a valid isotope.
    """
    RE = r'((([a-zA-Z]{1,2})[-]?([0-9]{1,3}))|(([0-9]{1,3})[-]?([a-zA-Z]{1,2})))([*_: ].*)?'

    def __new__(cls, string, without_suffix=False):
        string = string.strip()
        m = re.fullmatch(cls.RE, string)
        if m:
            if m.group(2) is not None:
                element, mass, suffix = m.group(3).capitalize(), m.group(4), m.group(8) or ''
            if m.group(5) is not None:
                element, mass, suffix = m.group(7).capitalize(), m.group(6), m.group(8) or ''
        else:
            raise ValueError(f"String '{string}' is not a valid isotope")

        if '/' in suffix:
            raise ValueError(f"Suffix '{suffix}' is not allowed to contain '/'")

        element = Element._new_(element, '' if without_suffix else suffix)
        return cls._new_(mass, element)

    def __init__(self, string, without_suffix=False):
        # Never called. Just for the docstring.
        super(Isotope, self).__init__()

    @classmethod
    def _new_(cls, mass, element):
        self = super().__new__(cls, f"{element.symbol}-{mass}{element.suffix}")
        self.mass = mass
        self.element = element
        self.symbol = element.symbol
        self.suffix = element.suffix
        return self

    def latex(self, dollar=True):
        """
        Returns a latex representation of the string e.g. Pd-105 -> ${}^{105}\mathrm{Pd}$

        Args:
            dollar (bool): Whether to include the bracketing ``$`` signs.

        """
        string = fr"{{}}^{{{self.mass}}}\mathrm{{{self.element}}}"
        if dollar:
            return f"${string}$"
        else:
            return string

    def without_suffix(self):
        """
        Return a new isotope string without the suffix.
        """
        return self._new_(self.mass, self.element.without_suffix())

class Ratio(str):
    """
    A subclass of string representing a ratio of two isotopes using the format ``<numer>/<denom>`` e.g.
    ``Pd-108/Pd-105``.

    Args:
        string (str): A string consisting of two isotope seperated by a ``/``.
        without_suffix (bool): If ``True`` the suffix part of the numerator and denominator`isotopes is ignored.

    Attributes:
        numer (str): The numerator isotope
        mass (str): The denominator isotope

    Raises:
        ValueError: If ``string`` does not represent a isotope ratio.
    """
    def __new__(cls, string, without_suffix=False):
        strings = string.split('/')
        if len(strings) != 2:
            raise ValueError(f'"{string}" is not a valid ratio')
        numer = Isotope(strings[0], without_suffix=without_suffix)
        denom = Isotope(strings[1], without_suffix=without_suffix)
        return cls._new_(numer, denom)

    @classmethod
    def _new_(cls, numer, denom):
        self = super().__new__(cls, f"{numer}/{denom}")
        self.numer = numer
        self.denom = denom
        return self

    def latex(self, dollar=True):
        """
        Returns a latex representation of the string e.g. Pd-108/Pd-105 -> ${}^{108}\mathrm{Pd}/{}^{105}\mathrm{Pd}$

        Args:
            dollar (bool): Whether to include the bracketing ``$`` signs.
        """
        if type(self.numer) is str:
            numer = self.numer
        else:
            numer = self.numer.latex(dollar=dollar)
        if type(self.denom) is str:
            denom = self.denom
        else:
            denom = self.denom.latex(dollar=dollar)

        return f"{numer}/{denom}"

    def without_suffix(self):
        """
        Return a new isotope string without the suffix.
        """
        return self._new_(self.numer.without_suffix(), self.denom.without_suffix())

def aselement(string, without_suffix=False, allow_invalid=False):
    """
    Returns a [``Element``][simple.utils.Element] representing an element symbol.

    The returned element format is the capitalised element
    symbol followed by the suffix, if present. E.g. ``Pd-104*`` where
    ``*`` is the suffix.

    The case of the element symbol is not considered.

    Args:
        string (str): A string containing an element symbol.
        without_suffix (): If ``True`` the suffix part of the string is ignored.
        allow_invalid ():  If ``False``, and ``string`` cannot be parsed into an element string, an exception is
            raised. If ``True`` then ``string.strip()`` is returned instead.

    Examples:
        >>> ele = simple.asisotope("pd"); ele
        "Pd"


    """
    if type(string) is Element:
        if without_suffix:
            return string.without_suffix()
        else:
            return string
    elif isinstance(string, str):
        string = string.strip()
    else:
        raise TypeError(f'``string`` must a str not {type(string)}')

    try:
        return Element(string, without_suffix=without_suffix)
    except ValueError:
        if allow_invalid:
            return string
        else:
            raise

def aselements(strings, without_suffix=False, allow_invalid=False):
    """
    Returns a tuple of [``Element``][simple.utils.Element] strings where each string represents an element symbol.

    Args:
        strings (): Can either be a string with element symbol seperated by a ``,`` or a sequence of strings.
        without_suffix (): If ``True`` the suffix part of each isotope string is ignored.
        allow_invalid ():  If ``False``, and a string cannot be parsed into an isotope string, an exception is
            raised. If ``True`` then ``string.strip()`` is returned instead.

    Examples:
        >>> simple.asisotopes('ru, pd, cd')
        ('Ru', 'Pd', 'Cd')

        >>> simple.asisotopes(['ru', 'pd', 'cd'])
        ('Ru', 'Pd', 'Cd')
    """
    if type(strings) is str:
        strings = [s for s in strings.split(',')]

    return tuple(aselement(string, without_suffix=without_suffix, allow_invalid=allow_invalid) for string in strings)

def asisotope(string, without_suffix=False, allow_invalid=False):
    """
    Returns a [``Isotope``][simple.utils.Isotope] representing an isotope.

    The returned isotope format is the capitalised element
    symbol followed by a dash followed by the mass number followed by the suffix, if present. E.g. ``Pd-104*`` where
    ``*`` is the suffix.

    The order of the element symbol and mass number in ``string`` is not important, but they must proceed the suffix.
    The element symbol and mass number may be seperated by ``-``. The case of the element symbol is not
    considered.

    Args:
        string (str): A string element symbol and a mass number.
        without_suffix (): If ``True`` the suffix part of the isotope string is ignored.
        allow_invalid ():  If ``False``, and ``string`` cannot be parsed into an isotope string, an exception is
            raised. If ``True`` then ``string.strip()`` is returned instead.

    Examples:
        >>> iso = simple.asisotope("104pd"); iso # pd104, 104-Pd etc are also valid
        "Pd-104"
        >>> iso.symbol, iso.mass
        "Pd", "104"

    """
    if type(string) is Isotope:
        if without_suffix:
            return string.without_suffix()
        else:
            return string
    elif isinstance(string, str):
        string = string.strip()
    else:
        raise TypeError(f'``string`` must a str not {type(string)}')

    try:
        return Isotope(string, without_suffix=without_suffix)
    except ValueError:
        if allow_invalid:
            return string
        else:
            raise

def asisotopes(strings, without_suffix=False, allow_invalid=False):
    """
    Returns a tuple of [``Isotope``][simple.utils.Isotope] strings where each string represents an isotope.

    Args:
        strings (): Can either be a string with isotopes seperated by a ``,`` or a sequence of strings.
        without_suffix (): If ``True`` the suffix part of each isotope string is ignored.
        allow_invalid ():  If ``False``, and a string cannot be parsed into an isotope string, an exception is
            raised. If ``True`` then ``string.strip()`` is returned instead.

    Examples:
        >>> simple.asisotopes('104pd, pd105, 106-Pd')
        ('Pd-104', 'Pd-105, 106-Pd')

        >>> simple.asisotopes(['104pd', 'pd105', '106-Pd'])
        ('Pd-104', 'Pd-105, 106-Pd')
    """
    if type(strings) is str:
        strings = [s for s in strings.split(',')]

    return tuple(asisotope(string, without_suffix=without_suffix, allow_invalid=allow_invalid) for string in strings)

def asratio(string, without_suffix=False, allow_invalid=False):
    """
    Returns a [``Ratio``][simple.utils.Isotope] string representing the ratio of two isotopes.

    The format of the returned string is the numerator followed by
    a ``/`` followed by the normiso. The numerator and normiso string be parsed by ``asisotope`` together with
    the given ``without_suffix`` and ``allow_invalid`` arguments passed to this function.

    Args:
        string (str): A string contaning two strings seperated by a single ``/``.
        without_suffix (bool): If ``True`` the suffix part of the numerator and normiso string is ignored.
        allow_invalid (bool): Whether the numerator and normiso has to be a valid isotope string.

    If the returned string is an isotope string it will have the following attributes and methods.

    Attributes:
        numer (str): The numerator string
        denom (str): The normiso string

    Methods:
        latex(string): Returns a latex formatted version of the isotope.
        without_suffix(): Returns a ratio string omitting the numerator and normiso suffix.

    """
    if type(string) is Ratio:
        return string
    elif isinstance(string, str):
        string = string.strip()
    else:
        raise TypeError(f'``string`` must a str not {type(string)}')

    try:
        return Ratio(string, without_suffix=without_suffix)
    except ValueError:
        if allow_invalid:
            return string
        else:
            raise

def asratios(strings, without_suffix=False, allow_invalid=False):
    """
    Returns a tuple of [``Ratio``][simple.utils.Isotope] strings where each string represents the ratio of two isotopes.

    Args:
        strings (): Can either be a string with isotope ratios seperated by a ``,`` or a sequence of strings.
        without_suffix (): If ``True`` the suffix part of each isotope string is ignored.
        allow_invalid ():  If ``False``, and a string cannot be parsed into an isotope string, an exception is
            raised. If ``True`` then ``string.strip()`` is returned instead.
    """
    if type(strings) is str:
        strings = [s.strip() for s in strings.split(',')]

    return tuple(asratio(string, without_suffix=without_suffix, allow_invalid=allow_invalid) for string in strings)

def asisolist(isolist, without_suffix=False, allow_invalid=False):
    """
    Return a dictionary consisting of an isotope key mapped to a tuple of isotopes that should make up the
    key isotope.

    If ``isolist`` is list or tuple of keys then each key will be mapped only to itself.

    Args:
        isolist (): Either a dictionary mapping a single isotope to a list of isotopes or a sequence of isotopes that
            will be mapped to themselfs.
        without_suffix (): If ``True`` the suffix part of each isotope string is ignored.
        allow_invalid (): If ``True`` invalid isotopes string are allowed. If ``False`` they will instead raise
            an exception.
    """
    if type(isolist) is not dict:
        isolist = asisotopes(isolist, without_suffix=without_suffix, allow_invalid=allow_invalid)
        return {iso: (iso,) for iso in isolist}
    else:
        return {asisotope(k, without_suffix=without_suffix, allow_invalid=allow_invalid):
                asisotopes(v, without_suffix=without_suffix, allow_invalid=allow_invalid)
                for k,v in isolist.items()}

def get_isotopes_of_element(isotopes, element, isotopes_without_suffix=False):
    """
    Returns a tuple of all isotopes in a sequence that contain the given element symbol.

    **Note** The strings in ``isotopes`` will be passed through [asisotopes][simple.asisotopes] before
    the evaluation and therefore do not have to be correcly formatted. Invalid isotope string are allowed
    but will be ignored by the evaluation.

    Args:
        isotopes (): An iterable of strings representing isotopes.
        element (str): The element symbol.
        isotopes_without_suffix (bool): If ``True`` suffixes will be removed from the isotopes in ``isotopes``
            before the evaluation takes place.

    Examples:
        >>> simple.utils.get_isotopes_of_element(["Ru-101", "Pd-102", "Rh-103", "Pd-104"], "Pd")
        >>> ("Pd-102", "Pd-104")
    """
    isotopes = asisolist(isotopes, without_suffix=isotopes_without_suffix, allow_invalid=True)
    element = aselement(element)
    return tuple(iso for iso in isotopes if (type(iso) is Isotope and iso.element == element))


##############
### Select ###
##############
# More complicated regex that gets more information from the text but is slower to execute.
# REATTR = r'(?:([ ]*[+-]?[0-9]*([.]?)[0-9]*(?:[Ee]?[+-]?[0-9]+)?[ ]*)|(?:[ ]*([.]?)(?:(?:[ ]*[{](.*)[}][ ]*)|(.*))))' # number, is_float, is_attr, kwarg, string
REATTR = r'(?:[ ]*([+-]?[0-9]*[.]?[0-9]*(?:[Ee]?[+-]?[0-9]+)?)[ ]*|(.*))'  # number, string
REINDEX = r'([-]?[0-9]*)[:]([-]?[0-9]*)[:]?([-]?[0-9]*)|([-]?[0-9]+)'
class EvalArg:
    NoAttr = object()

    class NoAttributeError(AttributeError):
        pass

    def __repr__(self):
        return f"({self.value}, {self.is_attr}, {self.is_kwarg})"

    def __init__(self, number, string):
        if number is not None:
            if '.' in number:
                v = float(number.strip())
            else:
                v = int(number)
            self.value = v
            self.is_attr = False
            self.is_kwarg = False
        else:
            string = string.strip()
            if string[0] == '.':
                is_attr = True
                string = string[1:].strip()
            else:
                is_attr = False
            if string[0] == '{' and string[-1] == '}':
                is_kwarg = True
                string = string[1:-1].strip()
            else:
                is_kwarg = False

            self.value = string
            self.is_attr = is_attr
            self.is_kwarg = is_kwarg
            if self.is_attr is False and self.is_kwarg is False:
                if string == 'True':
                    self.value = True
                elif string == 'False':
                    self.value = False
                elif string == 'None':
                    self.value = None

    def __call__(self, item, kwargs):
        v = self.value
        if self.is_kwarg:
            w = kwargs.get(v, self.NoAttr)
            if w is self.NoAttr:
                raise self.NoAttributeError(f'No keyword argument called "{v}" found in {kwargs}')
            else:
                v = w

        if self.is_attr:
            try:
                v = get_last_attr(item, v)
            except AttributeError:
                raise self.NoAttributeError(f'No attribute "{v}" found in {item}')

        return v

class BoolEvaluator:
    def __init__(self):
        self._evaluators = []
        self._and = True

    def add(self, operator, *args):
        self._evaluators.append((operator, args))

    def __call__(self, item, kwargs=None):
        result = False
        for operator, args in self._evaluators:
            if operator is None and len(args) == 1:
                val = args[0]
                if type(val) is EvalArg:
                    r = val(item, kwargs)
                else:
                    r = val

                if type(r) is str:
                    r = False
            else:
                try:
                    r = operator(*(arg(item, kwargs) for arg in args))
                except EvalArg.NoAttributeError:
                    r = False

            if r:
                result = True
            elif self._and:
                return False

        return result

    def eval(self, item, kwargs=None):
        return self.__call__(item, kwargs)

class MaskEvaluator(BoolEvaluator):
    def __call__(self, item, shape, kwargs=None):
        result = []
        for operator, args in self._evaluators:
            if operator is None and len(args) == 1:
                val = args[0]
                if type(val) is EvalArg:
                    val = val(item, kwargs)

                if type(val) is int or type(val) is slice:
                    result.append(np.full(shape, False))
                    try:
                        result[-1][val] = True
                    except IndexError:
                        pass # Out of bounds error

                elif type(val) is str:
                    result.append(False)
                else:
                    result.append(val)
            else:
                try:
                    result.append(operator(*(arg(item, kwargs) for arg in args)))
                except:
                    result.append(np.full(shape, False))

        if len(result) == 0:
            return np.full(shape, True)
        else:
            if self._and:
                op = np.logical_and
            else:
                op = np.logical_or
            try:
                r = functools.reduce(op, result)
                return np.full(shape, r, dtype=bool)
            except:
                return np.full(shape, False)

class AttrEval:
    def __init__(self):
        self.ab_evalstrings = []

    def add_ab_evaluator(self, opstr, operator):
        self.ab_evalstrings.append((opstr, f'{REATTR}{opstr}{REATTR}', operator))

    def parse_where(self, where):
        eval = BoolEvaluator()

        if type(where) is not str:
            eval.add(None, where)
            return eval

        if "&" in where and "|" in where:
            raise ValueError('Where strings cannot contain both & and |')
        elif "|" in where:
            evalstrings = where.split('|')
            eval._and = False
        else:
            evalstrings = where.split('&')

        for evalstr in evalstrings:
            evalstr = evalstr.strip()
            if len(evalstr) == 0: continue

            for opstr, regex, opfunc in self.ab_evalstrings:
                # The first check significantly speeds up the evaluation
                if (opstr in evalstr) and (m := re.fullmatch(regex, evalstr.strip())):
                    a_number, a_string, b_number, b_string = m.groups()
                    eval.add(opfunc, EvalArg(a_number, a_string), EvalArg(b_number, b_string))
                    break
            else:
                if (m := re.fullmatch(REATTR, evalstr.strip())):
                    a_number, a_string = m.groups()
                    eval.add(None, EvalArg(a_number, a_string))
                else:
                    raise ValueError(f'Unable to parse condition "{evalstr}"')
        return eval

    def eval(self, item, where, **where_kwargs):
        evaluate = self.parse_where(where)
        return evaluate(item, where_kwargs)

    def __call__(self, item, where, **where_kwargs):
        return self.eval(item, where, **where_kwargs)

class MaskEval:
    def __init__(self):
        self.ab_evalstrings = []

    def add_ab_evaluator(self, opstr, operator):
        self.ab_evalstrings.append((opstr, f'{REATTR}{opstr}{REATTR}', operator))

    def parse_mask(self, mask):
        eval = MaskEvaluator()

        if type(mask) is not str:
            eval.add(None, mask)
            return eval

        if "&" in mask and "|" in mask:
            raise ValueError('Mask string cannot contain & and |')
        elif "|" in mask:
            evalstrings = mask.split('|')
            eval._and = False
        else:
            evalstrings = mask.split('&')

        for evalstr in evalstrings:
            evalstr = evalstr.strip()
            if len(evalstr) == 0:
                pass

            elif (m := re.fullmatch(REINDEX, evalstr)):
                start, stop, step, index = m.groups()
                if index is not None:
                    eval.add(None, int(index))
                else:
                    eval.add(None, slice(int(start) if start.strip() else None,
                                                        int(stop) if stop.strip() else None,
                                                        int(step) if step.strip() else None))
            else:
                for opstr, regex, opfunc in self.ab_evalstrings:
                    # The first check significantly speeds up the evaluation
                    if (opstr in evalstr) and (m := re.fullmatch(regex, evalstr.strip())):
                        a_number, a_string, b_number, b_string = m.groups()
                        eval.add(opfunc, EvalArg(a_number, a_string), EvalArg(b_number, b_string))
                        break
                else:
                    if (m := re.fullmatch(REATTR, evalstr.strip())):
                        a_number, a_string = m.groups()
                        eval.add(None, EvalArg(a_number, a_string))
                    else:
                        raise ValueError(f'Unable to parse condition "{evalstr}"')
        return eval

    def eval(self, item, mask, shape, **mask_kwargs):
        evaluate = self.parse_mask(mask)
        return evaluate(item, shape, mask_kwargs)

    def __call__(self, item, mask, shape, **mask_kwargs):
        return self.eval(item, mask, shape, **mask_kwargs)


simple_eval = AttrEval()
simple_eval.add_ab_evaluator('==', operator.eq)
simple_eval.add_ab_evaluator('!=', operator.ne)
simple_eval.add_ab_evaluator('>=', operator.ge)
simple_eval.add_ab_evaluator('<=', operator.le)
simple_eval.add_ab_evaluator('<', operator.lt)
simple_eval.add_ab_evaluator('>', operator.gt)
simple_eval.add_ab_evaluator(' NOT IN ', lambda a, b: not operator.contains(b, a))
simple_eval.add_ab_evaluator(' IN ', lambda a, b: operator.contains(b, a))

def models_where(models, where, **where_kwargs):
    eval = simple_eval.parse_where(where)
    out = []
    for model in models:
        if eval.eval(model, where_kwargs):
            out.append(model)

    return out

mask_eval = MaskEval()
mask_eval.add_ab_evaluator('==', operator.eq)
mask_eval.add_ab_evaluator('!=', operator.ne)
mask_eval.add_ab_evaluator('>=', operator.ge)
mask_eval.add_ab_evaluator('<=', operator.le)
mask_eval.add_ab_evaluator('<', operator.lt)
mask_eval.add_ab_evaluator('>', operator.gt)


