import datetime, os, csv
import numpy as np
import h5py
import logging
import re

import simple.utils as utils
import simple.norm as norm
from simple.utils import NamedDict

__all__ = ['load_collection', 'load_models', 'new_collection']

logger = logging.getLogger('SIMPLE.models')

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
        value = utils.asarray(value)
        super().__setitem__(name, value)

    def get(self, key, value, default=None):
        if key in self:
            return self[key]
        else:
            return utils.asarray(default)

##############
### Models ###
##############
def load_collection(filename, dbfilename=None, *, default_isolist=None, convert_unit=True, overwrite=False,
                    where=None, **where_kwargs):
    """
    Loads a selection of models from a file.

    If that file does not exist it will create the file from the specified models file. Only when doing this
    is the ``default_isolist`` applied. If ``filename`` already exits the **assumption** is it has the correct isolist.

    ***Notes**

    The entire file will be read into memory. This might be an issue if reading very large files. The hdf5 are
    compressed so will be significantly larger when stored in memory.

    When reading the database file to create a subselection of the data using ``default_isolist``, the subselection is
     made when each model is loaded which reduces the amount of memory used.

    Args:
        filename (str): Name of the file to load or create.
        dbfilename (str): Name of the raw models file
        default_isolist (): Isolist applied to loaded models from ``dbfilename``.
        convert_units (bool): If ``True``  and data is stored in a mass unit all values will be divided by the
            mass number of the isotope before summing values together. The final value is then multiplied by the
            mass number of the output isotope.
        overwrite (bool): If ``True`` a new file will be created even if ``filename`` already exists.
        where (str): Used to select which models to load.
        **where_kwargs (): Keyword arguments used in combination with ``where``.

    Returns:
        A [ModelCollection][simple.models.ModelCollection] object containing all the loaded models.
    """
    mc = ModelCollection()
    if os.path.exists(filename) and not overwrite:
        logger.info(f'Loading existing file: {filename}')
        mc.load_file(filename, where=where, **where_kwargs)
    elif filename[-5:].lower() != '.hdf5' and os.path.exists(f'{filename}.hdf5') and not overwrite:
        logger.info(f'Loading existing file: {filename}.hdf5')
        mc.load_file(f'{filename}.hdf5', where=where, **where_kwargs)
    elif dbfilename is None:
        raise ValueError(f'File {filename} does not exist')
    elif os.path.exists(dbfilename):
        logger.info(f'Creating: "{filename}" from database: "{dbfilename}"')
        mc.load_file(dbfilename, isolist=default_isolist, convert_unit=convert_unit, where=where, **where_kwargs)
        mc.save(filename)
    else:
        raise ValueError(f'Neither "{filename}" or "{dbfilename}" exist')
    return mc

def load_models(*args, **kwargs):
    # Keeps for legacy reasons. Use load_collection instead
    return load_collection(*args, **kwargs)

def new_collection():
    """
    Return an empty [ModelCollection][simple.models.ModelCollection] object.
    """
    return ModelCollection()

class ModelCollection:
    """
    The main interface for working with a collection of models.
    """
    def __repr__(self):
        models = ", ".join([f'{k}: <{v.__class__.__name__}>' for k, v in self.models.items()])
        refs = ", ".join([f'{k}: <{v.__class__.__name__}>' for k, v in self.refs.items()])
        return f'{self.__class__.__name__}(models={{{models}}}, refs={{{refs}}})'

    def _repr_markdown_(self):
        models = "\n".join([f'- **[{i}]** ``{k}`` ({v.__class__.__name__})' for i, (k, v) in enumerate(self.models.items())])
        refs = "\n".join([f'- ``{k}`` ({v.__class__.__name__})' for k, v in self.refs.items()])
        return f"""
Models in collection:

{models}

References in collection:

{refs}

""".strip()

    def __init__(self):
        self.refs = {}
        self.models = {}

    def __iter__(self):
        return self.models.values().__iter__()

    def __len__(self):
        return len(self.models)

    def __getitem__(self, key):
        if type(key) is int:
            return self.models[tuple(self.models.keys())[key]]
        elif key in self.models:
            return self.models[key]
        elif key in self.refs:
            return self.refs[key]
        else:
            raise ValueError(f"No model or reference called '{key}' exists")

    ###################
    ### Load / Save ###
    ###################

    def save(self, filename):
        """
        Save the current selection of models.

        Args:
            filename (): Name of the file to be created.
        """
        if filename[-5:].lower() != '.hdf5':
            filename += '.hdf5'

        t0 = datetime.datetime.now()
        with h5py.File(filename, 'w') as file:
            ref_group = file.create_group('ref', track_order=True)
            for name, ref in self.refs.items():
                self._save_model(ref_group, ref)

            model_group = file.create_group('models', track_order=True)
            for name, model in self.models.items():
                self._save_model(model_group, model)

        t = datetime.datetime.now() - t0
        logger.info(f'Filesize: {os.path.getsize(filename) / 1024 / 1024:.2f} MB')
        logger.info(f'Time to save file: {t}')

    def _save_model(self, parent_group, model):
        group = parent_group.create_group(model.name, track_order=True)
        for name, value in model.hdf5_attrs.items():
            v = utils.asarray(value, saving=True)

            if v.ndim == 0:
                group.attrs.create(name, v)
            else:
                # Track order has to be set to true or saving will fail as there are too many
                # columns in CCSNe values
                group.create_dataset(name, data = v, compression = 'gzip', compression_opts = 9, track_order=True)

    def load_file(self, filename, isolist=None, convert_unit=True, where=None, **where_kwargs):
        """
        Add models from file to the current collection.

        **Note** existing models with the same name one of the loaded models will be overwritten.

        Args:
            filename (): Name of the file to load.
            isolist (): Isolist applied to loaded models. If ``None`` no subselection is made.
            where (): String evaluation used to select which models to load.
            **where_kwargs (): Additional keyword arguments used together with ``where``.
        """
        logger.info(f'Loading file: {filename}')
        t0 = datetime.datetime.now()
        with h5py.File(filename, 'r') as efile:
            for name, group in efile['models'].items():
                model = self._load_model(efile, group, name, isolist, convert_unit, where, where_kwargs)
                #self.models[name] = model

        t = datetime.datetime.now() - t0
        logger.info(f'Time to load file: {t}')

    def _load_model(self, file, group, model_name, isolist, convert_unit, where, where_kwargs):
        attrs = {}
        if where is not None:
            eval = utils.simple_eval.parse_where(where)

        # Load attributes
        for name, value in group.attrs.items():
            attrs[name] = utils.asarray(value)

        if 'clsname' not in attrs:
            raise ValueError(f"Model '{attrs[name]}' has no clsname")

        if where is None or eval.eval(attrs, where_kwargs):
            logger.info(f'Loading model: {model_name} ({attrs["clsname"]})')
            for name, value in group.items():
                if not isinstance(value, h5py.Dataset): continue
                attrs[name] = utils.asarray(value)

            model = self.new_model(name = model_name, **attrs)
            if isolist is not None:
                model.select_isolist(isolist, convert_unit=convert_unit)

            for attr in attrs:
                if attr[:6] == 'refid_':
                    self._load_ref(file, attrs[attr])

            return model
        else:
            logger.info(f'Ignored model: {model_name} ({attrs["clsname"]})')
            return None

    def _load_ref(self, file, refname):
        if refname in self.refs:
            return

        try:
            group = file['ref'][refname]
        except KeyError:
            raise ValueError(f"Reference '{refname}' does not exist")
        else:
            self.refs[refname] = self._load_model(file, group, refname, None, True, None, None)


    ##############
    # Get models #
    ##############
    def get_model(self, name, attr = None):
        """
        Returns the model with the given name.

        If ``attr`` is given then the value of that attribute from the named model is returned instead.
        """
        if name in self.models:
            model = self.models[name]
        else:
            raise ValueError(f"No model called '{name}' exists")

        if attr is None:
            return model
        else:
            return model[attr]

    def get_ref(self, name, attr=None):
        """
        Returns the reference model with the given name.

        If ``attr`` is given then the value of that attribute from the named model is returned instead.
        """
        if name in self.refs:
            ref = self.refs[name]
        else:
            raise ValueError(f"No reference called '{name}' exists")

        if attr is None:
            return ref
        else:
            return ref[attr]

    def new_model(self, clsname, name, **attrs):
        """
        Create a new model and add it to the current collection.

        **Note** if a model already exists called ``name`` it will be overwritten.

        Args:
            clsname (): The name of the model class to be created.
            name (): Name of the new model.
            **attrs (): Attributes to be added to the new model.

        Returns:
            The newly created model.
        """
        if clsname in AllModelClasses:
            return AllModelClasses[clsname](self, name, **attrs)
        else:
            raise ValueError(f"No model class called '{clsname}' exists")

    def select_isolist(self, isolist=None):
        """
        Used to create a subselection of data from each model.

        **Note** The original array may be overwritten with the new subselection.

        Args:
            isolist (): Either a list of isotopes to be selected or a dictionary consisting of the
            final isotope mapped to a list of isotopes to be added together for this isotope.

        Raises:
            NotImplementedError: Raised if this method has not been implemented for a model class.
        """
        for model in self.models.values():
            model.select_isolist(isolist)

    def where(self, where, **where_kwargs):
        """
        Returns a copy of the collection containing only the models which match the ``where`` argument.

        Use ``&`` or ``|`` to combine multiple evaluations. To evaluate an attribute of each model put a
        dot before the name e.g. ``.mass == 15``. To use one of the ``where_kwargs`` values put the name
        of the kwarg within pointy brackets e.g. ``.mass == {mass_kwarg}``.

        The available operators for evaluations are ``==``, ``!=``, ``>``, ``>=``, ``<``, ``<=``,
        `` IN ``, and `` NOT IN ``.

        **Note** that a shallow copy of the matching models is returned.

        Args:
            where (): A string with the evaluation to perform for each model.
            **where_kwargs (): Arguments used for the evaluation.

        Returns:
            bool

        """
        eval = utils.simple_eval.parse_where(where)
        new_collection = self.__class__()
        for model in self.models.values():
            if eval.eval(model, where_kwargs):
                model.copy_to(new_collection)

        return new_collection

    def copy(self):
        """
        Returns a new collection containing a shallow copy of all the models in the current collection.
        """
        new_collection = self.__class__()
        for model in self.models.values():
            model.copy_to(new_collection)

        return new_collection

    def internal_normalisation(self, normrat, *, isotopes = None,
                               enrichment_factor=1, relative_enrichment=True,
                               convert_unit=True, attrname = 'intnorm',
                               method='largest_offset', **method_kwargs):
        """
        Internally normalise the appropriate data of the model. See
        [internal_normalisation][simple.norm.internal_normalisation] for a description of the procedure and a
        description of the arguments.

        The result of the normalisation will be saved to each model under the ``normname`` attribute.

        Raises:
            NotImplementedError: Raised if the data to be normalised has not been specified for this model class.
        """
        for model in self.models.values():
            model.internal_normalisation(normrat, isotopes=isotopes,
                                         enrichment_factor=enrichment_factor, relative_enrichment=relative_enrichment,
                                         convert_unit=convert_unit, attrname=attrname,
                                         method=method, **method_kwargs)

    def simple_normalisation(self, normiso, enrichment_factor=1, relative_enrichment=True,
                             convert_unit=True, attrname = 'simplenorm'):
        """
        Normalise the appropriate data of the model. See
        [simple_normalisation][simple.norm.simple_normalisation] for a description of the procedure and a
        description of the arguments.

        The result of the normalisation will be saved to each model under the ``normname`` attribute.

        Raises:
            NotImplementedError: Raised if the data to be normalised has not been specified for this model class.
        """
        for model in self.models.values():
            model.simple_normalisation(normiso, enrichment_factor=enrichment_factor,
                                       relative_enrichment=relative_enrichment,
                                       convert_unit=convert_unit, attrname=attrname)

# A dict containing all model classes that exist.
# A model is automatically added to the dict when created.

AllModelClasses = {}
"""
A dictionary containing all the avaliable model classes. 

When a new model class is created subclassing [``ModelTemplate``][simple.models.ModelTemplate] it is automatically
added to this dictionary.
"""
class ModelTemplate:
    """
    This class can be subclassed to create new model classes.

    Once subclassed the new model will automatically be available through ``ModelCollection.new_model``.

    There are a number of class attributes that can be set to determine behaviour of the new class:

    - ``REQUIRED_ATTRS`` - A list of attributes that must be supplied when creating the class. An exception
        will be raised if any of these attributes are missing.
    - ``REPR_ATTRS`` - A list of the attributes that values will be shown in the repr.
    - ``ABUNDANCE_KEYARRAY`` - The name of a key array containing the abundances that should be normalised. Alternatively
        you can subclass the ``internal_normalisation`` and ``simple_normalisation`` methods for more customisation.
    - ``VALUES_KEYS_TO_ARRAY`` - If ``True`` a key array named ``<name>`` is automatically created upon model
        initialisation if attributes called.
        ``<name>_values`` and ``<name>_keys`` exits.
    - ``ISREF`` - Should be ``True`` for models specifically for storing reference values. These will be stored
        in the ``ModelCollection.refs`` dictionary rather than ``ModelCollection.models`` dictionary.
    """
    REQUIRED_ATTRS = []
    REPR_ATTRS = ['name']
    ABUNDANCE_KEYARRAY = None
    VALUES_KEYS_TO_ARRAY = True
    ISREF = False

    def __init_subclass__(cls, **kwargs):
        # Called each time a new subclass is created
        # Registers the class so that it can be found upon loading
        super().__init_subclass__(**kwargs)
        logger.debug(f'registering class: {cls.__name__}')
        if cls.__name__ != 'ModelTemplate':
            AllModelClasses[cls.__name__] = cls

    def __repr__(self):
        attrs = ", ".join([f"{attr}={getattr(self, attr)}" for attr in self.REPR_ATTRS])
        return f'{self.__class__.__name__}({attrs})'

    def _repr_markdown_(self):
        all_attrs = ['*name*'] + [f"*{name}*" for name in self.hdf5_attrs] + [f"*{name}*" for name in self.normal_attrs]
        attrs = '\n'.join([f'**{attr.capitalize()}**: {getattr(self, attr)}\\' for attr in self.REPR_ATTRS])
        return f'{attrs}\n**Attributes**: {", ".join(all_attrs)}'

    def __init__(self, collection, name, **hdf5_attrs):
        super().__setattr__('collection', collection)
        super().__setattr__('name', name)
        super().__setattr__('hdf5_attrs', HDF5Dict())
        super().__setattr__('normal_attrs', NamedDict())

        for attr in self.REQUIRED_ATTRS:
            if attr not in hdf5_attrs:
                raise ValueError(f"{{Required attribute '{attr}' does not exist in initialisation arguments}}")

        for k, v in hdf5_attrs.items():
            self.setattr(k, v, hdf5_compatible=True)

        # This is how we know which class to create. Need to be set after attrs incase you remapp the
        # class name
        self.setattr('clsname', self.__class__.__name__, hdf5_compatible=True, overwrite=True)

        # Automatically creates <name> key array if <name>_values and <name>_keys exists
        if self.VALUES_KEYS_TO_ARRAY:
            for key in self.hdf5_attrs:
                if key[-7:] == '_values':
                    aattr = key[:-7]
                    vattr = key
                    kattr = key[:-7] + '_keys'
                    if kattr in self.hdf5_attrs and aattr not in self.hdf5_attrs:
                        v = self.hdf5_attrs[vattr]
                        k = self.hdf5_attrs[kattr]
                        self.setattr(aattr, utils.askeyarray(v, k), hdf5_compatible=False)


        if self.ISREF:
            self.collection.refs[self.name] = self
        else:
            self.collection.models[self.name] = self

    def __getattr__(self, name):
        if name in self.hdf5_attrs:
            return self.hdf5_attrs[name]
        elif name in self.normal_attrs:
            return self.normal_attrs[name]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __getitem__(self, name):
        return self.__getattr__(name)

    def __setattr__(self, name, value):
        raise AttributeError('Use the `setattr` method to add attributes to this object')

    def __contains__(self, name):
        return name in self.hdf5_attrs or name in self.normal_attrs

    def setattr(self, name: str, value, hdf5_compatible: bool=False, overwrite:bool=False):
        """
        Set the value of a model attribute.

        Args:
            name (): Name of the attribute
            value (): The value of the attribute
            hdf5_compatible (): Should be ``True`` if the attribute should be included when the model is
                saved as a hdf5 file. ``value`` will be automatically converted to a hdf5 compatible value.
                An exception may be raised if this is not possible.
            overwrite (): Overwrite any existing attribute called ``name``. An exception is raised if ``name``
                exists and ``overwrite`` is ``False``.
        """
        if name in self.__dict__:
            raise AttributeError(f"'{self.__class__.__name__}' object has an attribute '{name}'")

        if (name in self.hdf5_attrs or name in self.normal_attrs) and overwrite is False:
            raise AttributeError(f"'{name}' already exists")
        elif name in self.hdf5_attrs:
            self.hdf5_attrs.pop(name)
        elif name in self.normal_attrs:
            if name in self.REQUIRED_ATTRS and hdf5_compatible is False:
                raise ValueError(f"{name} is a required attribute and therefore must be hdf5 compatible")
            else:
                self.normal_attrs.pop(name)

        if hdf5_compatible is True:
            self.hdf5_attrs.__setitem__(name, value)
        else:
            self.normal_attrs.__setitem__(name, value)

    def get_ref(self, name, attr=None):
        """
        Returns the reference model from the parent collection with the given name.

        If ``attr`` is given then the value of that attribute from the named model is returned instead.
        """
        return self.collection.get_ref(name, attr)

    def change_name(self, name):
        """
        Change the name of the current model  to ``name``.

        **Note** if another model already exists with this name in the collection it will be replaced with this model.
        """
        self.collection.models.pop(self.name)
        super().__setattr__('name', name)
        self.collection.models[self.name] = self

    def copy_to(self, collection):
        """
        Create a shallow copy of the current model in ``collection``.

        Returns:
            The new model
        """
        new_model = self.__class__(collection, self.name, **self.hdf5_attrs)

        for k, v in self.hdf5_attrs.items():
            # Make sure ref are in new models object
            if k[:6] == 'refid_':
                if type(v) is str and v not in collection.refs:
                    collection.refs[v] = self.collection.refs[v]

        for k, v in self.normal_attrs.items():
            new_model.setattr(k, v, hdf5_compatible=False, overwrite=True)

        return new_model

    def get_mask(self, mask, shape = None, **mask_attrs):
        """
        Returns a selection mask for an array with ``shape``.

        This function is used by plotting functions to plot only a sub selction of the data. The mask string
        can an integer representing an index, a slice or a condition that generates a mask. Use ``&`` or ``|``
        to combine multiple indexes and/or conditions.

        Supplied attributes can be accesed by putting a dot infront of the name, e.g. ``.data > 1``. The available
        operators for mask conditions are ``==``, ``!=``, ``>``, ``>=``, ``<``, ``<=``.

        The result of the mask evaluation must be broadcastable with ``shape``. If it is not an all ``False`` mask is
        returned.

        **Note**
        - It is not possible to mix ``&`` and ``|`` seperators. Doing so will raise an exception.
        - Any text not precceded by a dot will be evaluated as text. Text on its own will always be evaluated
        as ``False``.
        - An empty string will be evaluated as ``True``


        Args:
            mask (): String or object that will be evaluated to create a mask.
            shape (): Shape of the returned mask. If omitted the shape of the default abundance array is used.
            **mask_attrs (): Attributes to be used during the evaluation.

        Examples:
            >>> a = np.array([0,1,2,3,4])
            >>> model.get_mask('3', a.shape)
            array([False, False, False,  True,  False])

            >>> model.get_mask('1:3', a.shape)
            array([False, True, True,  False,  False])

            >>> model.get_mask('.data >= 1 & .data < 3', a.shape, data=a)
            array([False, True, True,  False,  False])

            >>> model.get_mask('.data >= 1 | .data > 3', a.shape, data=a)
            rray([True, True, False,  False,  True])

        Returns:
            A boolean numpy array with ``shape``.

        """
        if shape is None:
            if self.ABUNDANCE_KEYARRAY is None:
                raise ValueError("Shape is required as there is no default key array associated with this model")
            else:
                shape = getattr(self, self.ABUNDANCE_KEYARRAY).shape

        return utils.mask_eval.eval(mask_attrs, mask, shape, **mask_attrs)

    def convert_keyarray(self, array, unit, desired_unit):
        """
        Return a copy of the array converted to the desired unit.

        Supported units are the [mole and mass units][simple.utils.UNITS]. Converting between the *mass* and *mole** units is done by
        dividing/multiplying the values by the mass number.

        Always return a copy of ``array`` even if not conversion takes place.

        Args:
            array (): A key array.
            unit (): The current unit of the data in ``array``. If ``None`` the unit is assumed to be that of the
            ``desired_unit``.
            desired_unit (): The unit the array should be converted to. If ``None`` no conversion is made
            and the original array`is returned.

        Raises:
            ValueError: If the array cannot be converted from ``unit`` to ``desired_unit``.

        Returns:
            A copy of ``array`` with the desired unit.
        """
        array = array.copy()
        if unit is None:
            logger.warning(f"the key array does not have an assigned unit. Assuming the unit is the desired unit")
            return array
        elif desired_unit is None:
            logger.info(f"No desired unit specified. Assuming the desired unit is the current unit")
            return array

        if unit in utils.UNITS['mole']:
            if desired_unit.lower() in utils.UNITS['mole']:
                logger.info('Both array unit and desired unit are ``mole`` units.')
                return array
            elif desired_unit.lower() in utils.UNITS['mass']:
                logger.info('Converting array from ``mole`` to ``mass`` unit by multiplying the data by the mass number')
                for key in array.dtype.names:
                    try:
                        m = float(utils.asisotope(key).mass)
                    except ValueError:
                        pass # Leaves non-isotope values unchanged
                    else:
                        array[key] *= m
                return array

        elif unit in utils.UNITS['mass']:
            if desired_unit.lower() in utils.UNITS['mass']:
                logger.info('Both array unit and desired unit are ``mass`` units.')
                return array
            elif desired_unit.lower() in utils.UNITS['mole']:
                logger.info('Converting array from ``mass`` to ``mole`` unit by dividing the data by the mass number')
                for key in array.dtype.names:
                    try:
                        m = float(utils.asisotope(key).mass)
                    except ValueError:
                        pass # Leaves non-isotope values unchanged
                    else:
                        array[key] /= m
                return array

        raise ValueError(f"Unable to convert from '{unit}' to '{desired_unit}'")

    def get_keyarray(self, name=None, desired_unit=None):
        """
        Returns a copy of the named array with the desired unit.

        Args:
            name (): Name of the array to return. If ``None`` the default abundance array is returned.
            desired_unit (): The desired unit of the returned array.

        Returns:
            A copy of the array with the desired unit.
        """
        if name is None:
            if self.ABUNDANCE_KEYARRAY is None:
                raise ValueError("No default array associated with this model")
            else:
                name = self.ABUNDANCE_KEYARRAY
        try:
            a = self[name]
        except KeyError:
            raise AttributeError(f"This model has no attribute called '{name}'")
        else:
            if not isinstance(a, np.ndarray) or a.dtype.names is None:
                raise TypeError(f"Model attribute '{name}' is not a keyarray")

        if desired_unit is None:
            return a.copy()
        else:
            unit = getattr(self, f"{name}_unit", None)
            if unit is None:
                logger.warning(f"Keyarray '{name}' has no unit attribute. Assuming unit is {desired_unit}")
                return a.copy()
            else:
                return self.convert_keyarray(a, unit, desired_unit)

    def select_isolist(self, isolist, convert_unit=True):
        """
        Used to create a subselection of the data used for normalisation in the current model.

        **Notes**

        The original array will be overwritten with the new subselection.

        If ``<name>_values`` and ``<name>_keys`` exist they will also be overwritten with the
        result of the new selection.

        Args:
            isolist (): Either a list of isotopes to be selected or a dictionary consisting of the
             final isotope mapped to a list of isotopes to be added together for this isotope.
            convert_unit: If ``True``  and data is stored in a mass unit all values will be divided by the mass number of
                the isotope before summing values together. The final value is then multiplied by the mass number of the
                output isotope.

        Raises:
            NotImplementedError: Raised if the data to be normalised has not been specified for this model class.
        """
        # Updates the relevant arrays inplace
        if self.ABUNDANCE_KEYARRAY is None:
            raise NotImplementedError('The data to be normalised in has not been specified for this model')

        abu = self.get_keyarray(self.ABUNDANCE_KEYARRAY, 'mol' if convert_unit else None)

        abu = utils.select_isolist(isolist, abu)

        if convert_unit:
            original_unit = getattr(self, f"{self.ABUNDANCE_KEYARRAY}_unit", "mol")
            abu = self.convert_keyarray(abu, 'mol', original_unit)

        self.setattr(self.ABUNDANCE_KEYARRAY, abu, hdf5_compatible=False, overwrite=True)

        vname = f"{self.ABUNDANCE_KEYARRAY}_values"
        if vname in self.hdf5_attrs:
            self.setattr(vname, np.asarray(abu.tolist()), hdf5_compatible=True, overwrite=True)
        elif vname in self.normal_attrs:
            self.setattr(vname, np.asarray(abu.tolist()), hdf5_compatible=False, overwrite=True)

        kname = f"{self.ABUNDANCE_KEYARRAY}_keys"
        if kname in self.hdf5_attrs:
            self.setattr(kname, utils.asisotopes(abu.dtype.names), hdf5_compatible=True, overwrite=True)
        elif kname in self.normal_attrs:
            self.setattr(kname, utils.asisotopes(abu.dtype.names), hdf5_compatible=False, overwrite=True)

    def internal_normalisation(self, normrat, *, isotopes = None,
                               enrichment_factor=1, relative_enrichment=True,
                               convert_unit=True, attrname='intnorm',
                               method='largest_offset', **method_kwargs):
        """
        Internally normalise the appropriate data of the model. See
        [internal_normalisation][simple.norm.internal_normalisation] for a description of the procedure and a
        description of the arguments.

        The result of the normalisation will be saved under the ``normname`` attribute.

        Raises:
            NotImplementedError: Raised if the data to be normalised has not been specified for this model class.
        """
        if self.ABUNDANCE_KEYARRAY is None:
            raise NotImplementedError('The data to be normalised in has not been specified for this model')

        # The abundances to be normalised
        abu = self.get_keyarray(self.ABUNDANCE_KEYARRAY, 'mol' if convert_unit else None)

        # Isotope masses
        ref_stdmass = self.get_ref(self.refid_isomass)
        stdmass = ref_stdmass.get_keyarray('data')

        # The reference abundances. Typically, the initial values of the model
        ref_stdabu = self.get_ref(self.refid_isoabu)
        stdabu = ref_stdabu.get_keyarray('data', 'mol' if convert_unit else None)


        result = norm.internal_normalisation(abu, isotopes, normrat, stdmass, stdabu,
                                             enrichment_factor=enrichment_factor, relative_enrichment=relative_enrichment,
                                             method=method, **method_kwargs)

        self.setattr(attrname, result, hdf5_compatible=False, overwrite=True)
        return result

    def simple_normalisation(self, normiso, *, isotopes = None, enrichment_factor = 1, relative_enrichment=True,
                             convert_unit=True, attrname='simplenorm', dilution_factor = None):
        """
        Normalise the appropriate data of the model. See
        [simple_normalisation][simple.norm.simple_normalisation] for a description of the procedure and a
        description of the arguments.

        The result of the normalisation will be saved under the ``normname`` attribute.

        Raises:
            NotImplementedError: Raised if the data to be normalised has not been specified for this model class.
        """
        if self.ABUNDANCE_KEYARRAY is None:
            raise NotImplementedError('The data to be normalised has not been specified for this model')

        abu = self.get_keyarray(self.ABUNDANCE_KEYARRAY, 'mol' if convert_unit else None)

        ref_stdabu = self.get_ref(self.refid_isoabu)
        stdabu = ref_stdabu.get_keyarray('data', 'mol' if convert_unit else None)

        result = norm.simple_normalisation(abu, isotopes, normiso, stdabu,
                                           enrichment_factor=enrichment_factor, relative_enrichment=relative_enrichment,
                                           dilution_factor=dilution_factor)

        self.setattr(attrname, result, hdf5_compatible=False, overwrite=True)
        return result


##################
### Ref Values ###
##################
class IsoRef(ModelTemplate):
    """
    Model specifically for storing reference isotope values.

    Attributes:
        type (str): The type of data stored in the model. **Required at initialisation**
        citation (str): A citation for the data. **Required at initialisation**
        data (): A key array containing the data. Is created upon model initiation from the
            ``data_values`` and ``data_keys`` attributes.
        data_values (): A 2dim array containing the data. **Required at initialisation**
        data_keys (): Keys for the second dimension of ``data_values``. **Required at initialisation**
        data_unit (): Unit for the data. **Required at initialisation**
    """
    REQUIRED_ATTRS = ['type', 'citation',
                      'data_values', 'data_keys', 'data_unit']
    REPR_ATTRS = ['name', 'type']
    ISREF = True


def load_csv_h(filename):
    """
    Returns a key array from a csv file where the first from is the columns keys and the remaining
    rows contain the data.
    """
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, skipinitialspace=True)
        data = [row for row in reader if row[0][0] != '#']

    return np.transpose([float(i) for i in data[1]]), utils.asisotopes(data[0], allow_invalid=True)


def load_ppn(filename):
    """
    Load a key array from a ppn file.
    """
    isotopes = []
    values = []
    with open(filename, 'r') as f:
        for row in f.readlines():
            isotopes.append(row[3:9].replace(' ', ''))
            values.append(float(row[10:].strip()))

    return np.transpose(values), utils.asisotopes(isotopes, allow_invalid=True)
