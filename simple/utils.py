import re, operator
import numpy as np
import logging
import yaml

logger = logging.getLogger('SIMPLE.utils')

__all__ = ['load_defaults',
           'asarray', 'askeyarray',
           'asisotope', 'asisotopes', 'asratio', 'asratios',
           'asisolist', 'select_isolist', 'get_isotopes_of_element']

def load_defaults(filename: str):
    """
    Loads default arguments for functions from a YAML formatted file.

    Return a dictionary containing a dictionary with the default arguments mapped to the argument name. To use unpack
    the arguments in the function call (See example).

    You can still pass normal arguments and keyword arguments as long as they are not included in the default dictionary.

    Examples
    >>> defaults = simple.load_defaults('defaults.yaml')
    >>> somefunction(**defaults['somefunction']) # Unpack arguments
    """
    return yaml.safe_load(open(filename, 'r').read())


def askeyarray(array, keys, dtype=None):
    """
    Returns a numpy array where the columns can be accessed by the column key.

    Args:
        array (): An array like object containing at most 2 dimensions. The first dimension is the row and the second
        dimension is the column.
        keys (): The columns keys to be associated with the data. Must be the same length as the second dimension
        of ``array``.
        dtype (): The data type of the returned array. All columns will have the same dtype.

    **Notes**
    If ``array`` has less then 2 dimensions then it is assumed to represent a single row of data.

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

    a = np.asarray(array, dtype=dtype)
    dtype = [(k, a.dtype) for k in keys]

    if a.ndim < 2:
        a = a.reshape((-1, a.size))
    elif a.ndim > 2:
        raise ValueError('item must be 1D or 2D')

    if a.shape[1] != len(keys):
        raise ValueError(
            f'item (r:{a.shape[0]}, c:{a.shape[1]}) must have same number of columns as there are keys ({len(keys)})')

    return np.array([tuple(r) for r in a], dtype=dtype)

def asarray(data, dtype=None, saving=False):
    """
    Convert ``data`` to a numpy array.

    If ``data`` is a string or a collection of strings and ``saving`` is ``False``, either a single string or a tuple
    of string will be returned. If ``saving`` is ``True`` the array will be converted to a byte array. This is so the
    array is compatible with the hdf5 library.

    Arrays with a ``bytes`` dtype will automatically be converted to the ``str`` dtype. If ``saving`` is ``False`` then
    this array will be converted to either a string or a tuple of strings (see above).

    Args:
        data (): An array like object.
        dtype (): The data type of the returned array.
        saving (): Should be ``True`` is the data is to be saved in a hdf5 file.

    """
    data = np.asarray(data, dtype=dtype)

    if data.dtype.type is np.bytes_:
        data = data.astype(np.str_)

    if not saving and data.dtype.type is np.str_:
        data = data.tolist()
        if type(data) is list:
            data = tuple(data)

    if saving and data.dtype.type is np.str_:
        data = data.astype(np.bytes_)

    return data


def select_isolist(isolist, data, keys= None, *, without_suffix=False, massunit = False):
    isolist = asisolist(isolist, without_suffix=without_suffix)
    data = np.asarray(data)
    if keys is not None:
        keys = asisotopes(keys, allow_invalid=True)

    new_data = []
    missing_isotopes = []

    if data.dtype.names is None:
        if keys is None:
            raise ValueError('No keys given for the data')
        data = np.atleast_1d(data)

        for mainiso, inciso in isolist.items():
            value = np.zeros(data.shape[0])
            for iso in inciso:
                if iso in keys:
                    index = keys.index(iso)
                    if massunit:
                        value += (data[:,index] / float(iso.mass))
                    else:
                        value += data[:, index]
                else:
                    missing_isotopes.append(iso)

            if massunit:
                new_data.append(value * float(mainiso.mass))
            else:
                new_data.append(value)

        result = np.transpose(new_data), tuple(isolist.keys())
    else:
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

z_names = ['Neut', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al',
           'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co',
           'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb',
           'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs',
           'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm',
           'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
           'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U']

#######################
### Isotope strings ###
#######################
class Isotope(str):
    """
    Isotope(string, without_suffix=False, allow_invalid=False)

    Convert string to a whatever

    Args:
        string (str): A string element symbol and a mass number.
        without_suffix (bool): If ``True`` the suffix part of the ``string`` is ignored. Default is ``False``
        allow_invalid (bool): If ``string`` is not a valid isotope an exception is raised if ``False``.
                              If ``True`` ``string`` is returned. Default is ``False``.

    Attributes:
        element (str): The element symbol of the isotope
        mass (str): The mass of the isotope
        suffix (str): The suffix of the isotope

    """
    RE = r'((([a-zA-Z]{1,2})[-]?([0-9]{1,3}))|(([0-9]{1,3})[-]?([a-zA-Z]{1,2})))([^a-zA-Z0-9].*)?'

    def __new__(cls, string, without_suffix=False, allow_invalid=False):

        string = string.strip()
        m = re.fullmatch(cls.RE, string)
        if m:
            if m.group(2) is not None:
                element, mass, suffix = m.group(3).capitalize(), m.group(4), m.group(8) or ''
            if m.group(5) is not None:
                element, mass, suffix = m.group(7).capitalize(), m.group(6), m.group(8) or ''
        else:
            if allow_invalid:
                return string
            else:
                raise ValueError(f"String '{string}' is not a valid isotope")

        return cls._new_(mass, element, '' if without_suffix else suffix)

    @classmethod
    def _new_(cls, mass, element, suffix):
        self = super().__new__(cls, f"{element}-{mass}{suffix}")
        self.mass = mass
        self.element = element
        self.suffix = suffix
        return self

    def latex(self, dollar=True):
        """
        Convert string to latex.

        Args:
            dollar (bool): Whether to include the bracketing ``$`` signs.

        Returns: str

        """
        string = fr"{{}}^{{{self.mass}}}\mathrm{{{self.element}{self.suffix}}}"
        if dollar:
            return f"${string}$"
        else:
            return string

    def without_suffix(self):
        """
        Returns an isotope string without the suffix.

        Returns: Isotope

        """
        return self._new_(self.mass, self.element, '')

class Ratio(str):
    def __new__(cls, string, without_suffix=False, allow_invalid=False):
        strings = string.split('/')
        if len(strings) != 2:
            raise ValueError(f'"{string}" is not a valid ratio')
        numer = Isotope(strings[0], without_suffix=without_suffix, allow_invalid=allow_invalid)
        denom = Isotope(strings[1], without_suffix=without_suffix, allow_invalid=allow_invalid)
        return cls._new_(numer, denom)

    @classmethod
    def _new_(cls, numer, denom):
        self = super().__new__(cls, f"{numer}/{denom}")
        self.numer = numer
        self.denom = denom
        return self

    def latex(self, dollar=True):
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
        return self._new_(self.numer.without_suffix(), self.denom.without_suffix())

def asisotope(string, without_suffix=False, allow_invalid=False):
    """
    Parse ``string`` into correctly formatted isotope string. The returned isotope format is the capitalised element
    symbol followed by a dash followed by the mass number followed by the suffix, if present.

    Args:
        string (str): A string element symbol and a mass number.
        without_suffix (bool): If ``True`` the suffix part of the ``string`` is ignored.
        allow_invalid (bool): If ``string`` is not a valid isotope an exception is raised if ``False``.
                              If ``True`` ``string`` is returned.

    If the returned string is an isotope string it will have the following attributes and methods.

    Attributes:
        element (str): The element symbol of the isotope
        mass (str): The mass of the isotope
        suffix (str): The suffix of the isotope

    Methods:
        latex(string): Returns a latex formatted version of the isotope.
        without_suffix(): Returns a isotope string without the suffix.

    """
    return Isotope(string, without_suffix=without_suffix, allow_invalid=allow_invalid)

def asisotopes(strings, without_suffix=False, allow_invalid=False):
    """
    Returns a tuple of isotope strings. ``strings`` can either be iterable containing multiple stings or a single string
    using ``,`` to separate different isotopes.

    See [asisotope](#asisotope) for more details and a description of the other arguments.

    Examples:
        >>> simple.asisotopes('104pd, pd105, 106-Pd')
        ('Pd-104', 'Pd-105, 106-Pd')

        >>> simple.asisotopes(['104pd', 'pd105', '106-Pd'])
        ('Pd-104', 'Pd-105, 106-Pd')
    """
    if type(strings) is str:
        strings = [s.strip() for s in strings.split(',')]

    return tuple(Isotope(string, without_suffix=without_suffix, allow_invalid=allow_invalid)
            for string in strings if string != '')

def asratio(string, without_suffix=False, allow_invalid=False):
    """
    Return a ratio string of two isotopes.

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
    return Ratio(string, without_suffix=without_suffix, allow_invalid=allow_invalid)

def asratios(strings, without_suffix=False, allow_invalid=False):
    if type(strings) is str:
        strings = [s.strip() for s in strings.split(',')]

    return tuple(Ratio(string, without_suffix=without_suffix, allow_invalid=allow_invalid)
            for string in strings if string != '')

def asisolist(isolist, without_suffix=False, allow_invalid=False):
    """
    Return a dictionary consisting of a single isotope key mapped to a tuple of isotopes that should make up the
    key isotope.

    If ``default_isolist`` is list or tuple of keys then each key will be mapped only to itself.

    Args:
        isolist ():
        without_suffix ():
        allow_invalid ():
    """
    if type(isolist) is not dict:
        isolist = asisotopes(isolist, without_suffix, allow_invalid)
        return {iso: (iso,) for iso in isolist}
    else:
        return {asisotope(k, without_suffix, allow_invalid): asisotopes(v, without_suffix, allow_invalid)
                for k,v in isolist.items()}

def get_isotopes_of_element(isotopes, element, suffix=None):
    isotopes = asisolist(isotopes, without_suffix=False, allow_invalid=True)
    element = element.capitalize()
    return tuple(iso for iso in isotopes if
                 (type(iso) is Isotope and
                  iso.element == element and
                  (suffix is None or iso.suffix == suffix))
                 )
##############
### Select ###
##############
# More complicated regex that gets more information from the text but is slower to execute.
# REATTR = r'(?:([ ]*[+-]?[0-9]*([.]?)[0-9]*(?:[Ee]?[+-]?[0-9]+)?[ ]*)|(?:[ ]*([.]?)(?:(?:[ ]*[{](.*)[}][ ]*)|(.*))))' # number, is_float, is_attr, kwarg, string


class NoAttributeError(AttributeError):
    pass

class AttrEval:
    REATTR = r'(?:[ ]*([+-]?[0-9]*[.]?[0-9]*(?:[Ee]?[+-]?[0-9]+)?)[ ]*|(.*))'  # number, string

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
                    raise NoAttributeError(f'No keyword argument called "{v}" found in {kwargs}')
                else:
                    v = w

            if self.is_attr:
                if isinstance(item, dict):
                    w = item.get(v, self.NoAttr)
                    if w is self.NoAttr:
                        raise NoAttributeError(f'No item "{v}" found in {item}')
                    else:
                        v = w
                else:
                    w = getattr(item, v, self.NoAttr)
                    if w is self.NoAttr:
                        raise NoAttributeError(f'No attribute called "{v}" found in {item}')
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
                except NoAttributeError:
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
model_eval.add_ab_evaluator('==', operator.eq)
model_eval.add_ab_evaluator('!=', operator.ne)
model_eval.add_ab_evaluator('>=', operator.ge)
model_eval.add_ab_evaluator('<=', operator.le)
model_eval.add_ab_evaluator('<', operator.lt)
model_eval.add_ab_evaluator('>', operator.gt)
model_eval.add_ab_evaluator(' NOT IN ', lambda a, b: not operator.contains(b, a))
model_eval.add_ab_evaluator(' IN ', lambda a, b: operator.contains(b, a))




