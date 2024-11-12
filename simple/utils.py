import re, operator
import numpy as np
import logging
import yaml
import collections

logger = logging.getLogger('SIMPLE.utils')

__all__ = ['load_defaults',
           'asarray', 'askeyarray',
           'asisotope', 'asisotopes', 'asratio', 'asratios']

MASS_UNITS = ['mass', 'massfrac', 'wt', 'wt%']
"""
A list of units that represent data being stored in a mass unit or as mass fractions.
"""

MOLE_UNITS = ['mol', 'molfrac']
"""
A list of units representing data being stored in moles or as mole fractions.
"""

def load_defaults(filename: str):
    """
    Loads default arguments for functions from a YAML formatted file.

    To use a set of default values, unpack the arguments in the function call (See example).

    You can still arguments and keyword arguments as normal as long as they are not included in the default dictionary.

    Returns:
        A named dictionary containing mapping the name given in the yaml file to another dictionary mapping the arguments
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
    return NamedDict(yaml.safe_load(open(filename, 'r').read()))

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

def select_isolist(isolist, data, *, without_suffix=False, massunit = False, convert_unit = True):
    """
    Creates a subselection of ``data`` containing only the isotopes in ``isolist``.

    Args:
        isolist (): Either a list of isotopes to be selected or a dictionary consisting of the
             final isotope mapped to a list of isotopes to be added together for this isotope.
        data (): A key array from which the subselection will be made.
        without_suffix (): If ``True`` the suffix will be removed from all isotope strings in ``isolist``.
        massunit (): Whether the data is stored in a mass unit.
        convert_unit: If ``True``  and ``mass_unit=True`` all values in ``data`` will be divided by the mass number of
            the isotope before summing values together. The final value is then multiplied by the mass number of the
            output isotope.

    Returns:
        A new key array containing only the isotopes in ``isolist``.

    Examples:
        >>>
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
                if massunit:
                    value += (data[iso] / float(iso.mass))
                else:
                    value += data[iso]
            else:
                missing_isotopes.append(iso)

        if massunit:
            new_data.append(value * float(mainiso.mass))
        else:
            new_data.append(value)

    result = askeyarray(np.array(new_data).transpose(), isolist.keys())

    if len(missing_isotopes) != 0:
        logger.warning(f'Missing isotopes set to 0: {", ".join(missing_isotopes)}')

    return result

#######################
### Isotope strings ###
#######################
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
    RE = r'((([a-zA-Z]{1,2})[-]?([0-9]{1,3}))|(([0-9]{1,3})[-]?([a-zA-Z]{1,2})))([^a-zA-Z0-9].*)?'

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

        return cls._new_(mass, element, '' if without_suffix else suffix)

    def __init__(self, string, without_suffix=False):
        # Never called. Just for the docstring.
        super(Isotope, self).__init__()

    @classmethod
    def _new_(cls, mass, element, suffix):
        self = super().__new__(cls, f"{element}-{mass}{suffix}")
        self.mass = mass
        self.element = element
        self.suffix = suffix
        return self

    def latex(self, dollar=True):
        """
        Returns a latex representation of the string e.g. Pd-105 -> ${}^{105}\mathrm{Pd}$

        Args:
            dollar (bool): Whether to include the bracketing ``$`` signs.

        """
        string = fr"{{}}^{{{self.mass}}}\mathrm{{{self.element}{self.suffix}}}"
        if dollar:
            return f"${string}$"
        else:
            return string

    def without_suffix(self):
        """
        Return a new isotope string without the suffix.
        """
        return self._new_(self.mass, self.element, '')

class Ratio(str):
    """
    A subclass of string representing a isotopes_or_ratios of two isotopes using the format ``<numer>/<denom>`` e.g.
    ``Pd-108/Pd-105``.

    Args:
        string (str): A string consisting of two isotope seperated by a ``/``.
        without_suffix (bool): If ``True`` the suffix part of the numerator and denominator`isotopes is ignored.

    Attributes:
        numer (str): The numerator isotope
        mass (str): The denominator isotope

    Raises:
        ValueError: If ``string`` does not represent a isotope isotopes_or_ratios.
    """
    def __new__(cls, string, without_suffix=False):
        strings = string.split('/')
        if len(strings) != 2:
            raise ValueError(f'"{string}" is not a valid isotopes_or_ratios')
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
        >>> iso.element, iso.mass
        "Pd", "104"

    """
    if type(string) is Isotope:
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
    Returns a [``Ratio``][simple.utils.Isotope] string representing the isotopes_or_ratios of two isotopes.

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
        without_suffix(): Returns a isotopes_or_ratios string omitting the numerator and normiso suffix.

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
    Returns a tuple of [``Ratio``][simple.utils.Isotope] strings where each string represents the isotopes_or_ratios of two isotopes.

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

def get_isotopes_of_element(isotopes, element, suffix=None, isotopes_without_suffix=False):
    """
    Returns a tuple of all isotopes in a sequence that contain the given element symbol.

    **Note** The strings in ``isotopes`` will be passed through [asisotopes][simple.asisotopes] before
    the evaluation and therefore do not have to be correcly formatted. Invalid isotope string are allowed
    but will be ignored by the evaluation.

    Args:
        isotopes (): An iterable of strings representing isotopes.
        element (str): The element symbol.
        suffix (str): If given the isotopes must also have this suffix.
        isotopes_without_suffix (bool): If ``True`` suffixes will be removed from the isotopes in ``isotopes``
            before the evaluation takes place.

    Examples:
        >>> simple.utils.get_isotopes_of_element(["Ru-101", "Pd-102", "Rh-103", "Pd-104"], "Pd")
        >>> ("Pd-102", "Pd-104")
    """
    isotopes = asisolist(isotopes, without_suffix=isotopes_without_suffix, allow_invalid=True)
    element = element.capitalize()
    return tuple(iso for iso in isotopes if
                 (type(iso) is Isotope and
                  iso.element == element and
                  (suffix is None or iso.suffix == suffix))
                 )

####################
### Attr objects ###
####################
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

    def __setattr__(self, name, value):
        return self.__setitem__(name, value)

    def __getattr__(self, name):
        return self.__getitem__(name)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def setdefault(self, key, default_value):
        if key not in self:
            self[key] = default_value

        return self[key]


# TODO maybe give an error if the array has object type as this probably cant be saved and loaded properly?
class HDF5Dict(NamedDict):
    """
    A subclass of [NamedDict][simple.utils.NamedDict] where all values are passed to
    [asarray][simple.asarray] before being added to the dictionary.

    All contents on this dictionary should be compatiable with HDF5 files.

    Examples:
        >>> nd = simple.utils.NamedDict({'a': 1, 'b': 2, 'c': 3})
        >>> nd.a
        array(1)
    """
    def __setitem__(self, name, value):
        value = asarray(value)
        super().__setitem__(name, value)

    def get(self, key, value, default=None):
        if key in self:
            return self[key]
        else:
            return asarray(default)

##############
### Select ###
##############
# More complicated regex that gets more information from the text but is slower to execute.
# REATTR = r'(?:([ ]*[+-]?[0-9]*([.]?)[0-9]*(?:[Ee]?[+-]?[0-9]+)?[ ]*)|(?:[ ]*([.]?)(?:(?:[ ]*[{](.*)[}][ ]*)|(.*))))' # number, is_float, is_attr, kwarg, string


class AttrEval:
    REATTR = r'(?:[ ]*([+-]?[0-9]*[.]?[0-9]*(?:[Ee]?[+-]?[0-9]+)?)[ ]*|(.*))'  # number, string

    class NoAttributeError(AttributeError):
        pass

    class Arg:
        NoAttr = object()

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

        def __call__(self, item, kwargs):
            v = self.value
            if self.is_kwarg:
                w = kwargs.get(v, self.NoAttr)
                if w is self.NoAttr:
                    raise AttrEval.NoAttributeError(f'No keyword argument called "{v}" found in {kwargs}')
                else:
                    v = w

            if self.is_attr:
                if isinstance(item, dict):
                    w = item.get(v, self.NoAttr)
                    if w is self.NoAttr:
                        raise AttrEval.NoAttributeError(f'No item "{v}" found in {item}')
                    else:
                        v = w
                else:
                    w = getattr(item, v, self.NoAttr)
                    if w is self.NoAttr:
                        raise AttrEval.NoAttributeError(f'No attribute called "{v}" found in {item}')
                    else:
                        v = w

            return v

    class Evaluator:
        def __init__(self):
            self._evaluators = []

        def add(self, operator, *args):
            self._evaluators.append((operator, args))

        def __call__(self, item, kwargs):
            for operator, args in self._evaluators:
                try:
                    if not operator(*(arg(item, kwargs) for arg in args)):
                        return False
                except AttrEval.NoAttributeError:
                    return False
            else:
                return True

    def __init__(self):
        self.ab_evalstrings = []

    def add_ab_evaluator(self, opstr, operator):
        self.ab_evalstrings.append((opstr, f'{self.REATTR}{opstr}{self.REATTR}', operator))

    def parse_where(self, evalstring):
        eval = self.Evaluator()

        evalstrings = evalstring.split('&')
        for evalstr in evalstrings:
            evalstr = evalstr.strip()
            if len(evalstr) == 0: continue

            for opstr, regex, opfunc in self.ab_evalstrings:
                # The first check significantly speeds up the evaluation
                if (opstr in evalstr) and (m := re.fullmatch(regex, evalstr.strip())):
                    a_number, a_string, b_number, b_string = m.groups()
                    eval.add(opfunc, self.Arg(a_number, a_string), self.Arg(b_number, b_string))
                    break
            else:
                raise ValueError(f'Unable to parse condition "{evalstr}"')
        return eval

    def eval_where(self, item, evalstring, **kwargs):
        evaluate = self.parse_where(evalstring)
        return evaluate(item, kwargs)

model_eval = AttrEval()
"""
Evaluator used to evaluate models. 

The following operators are currently supported: ``==``, ``!=``, ``>=``, ``<=``, ``>``, ``<``, `` IN ``, `` NOT IN ``.

Use as ``model_eval(model, where, **where_kwargs)``
"""


model_eval.add_ab_evaluator('==', operator.eq)
model_eval.add_ab_evaluator('!=', operator.ne)
model_eval.add_ab_evaluator('>=', operator.ge)
model_eval.add_ab_evaluator('<=', operator.le)
model_eval.add_ab_evaluator('<', operator.lt)
model_eval.add_ab_evaluator('>', operator.gt)
model_eval.add_ab_evaluator(' NOT IN ', lambda a, b: not operator.contains(b, a))
model_eval.add_ab_evaluator(' IN ', lambda a, b: operator.contains(b, a))




