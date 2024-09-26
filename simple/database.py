import re, datetime, os, csv, operator, inspect
import numpy as np
import h5py
from nugridpy import nugridse as mp
import logging

import simple.utils as utils
import simple.norm as norm

__all__ = ['ModelCollection', 'load_models']

logger = logging.getLogger('SIMPLE.database')

model_eval = utils.AttrEval()
model_eval.add_ab_evaluator('==', operator.eq)
model_eval.add_ab_evaluator('!=', operator.ne)
model_eval.add_ab_evaluator('<', operator.lt)
model_eval.add_ab_evaluator('<=', operator.le)
model_eval.add_ab_evaluator('>', operator.gt)
model_eval.add_ab_evaluator('>=', operator.ge)
model_eval.add_ab_evaluator(' IN ', lambda a, b: operator.contains(b, a))
model_eval.add_ab_evaluator(' NOT IN ', lambda a, b: not operator.contains(b, a))

class Attrs(dict):
    def __setattr__(self, name, value):
        return self.__setitem__(name, value)

    def __getattr__(self, name):
        return self.__getitem__(name)

    def update(self, *args, **kwargs):
        raise TypeError('Please set each item individually')
    
class ArrayAttrs(Attrs):
    def __setitem__(self, name, value):
        value = utils.asarray(value)
        super().__setitem__(name, value)

def load_models(filename, dbfilename, default_isolist=None, where=None, overwrite=False, **where_kwargs):
    """
    Loads a selection of models from a file.

    If that file does not exist it will create the file from the specified database file. Only when doing this
    is the ``default_isolist`` applied. If ``filename`` already exits the **assumption** is it has the correct isolist.

    Args:
        filename (): Name of the file to load or create.
        dbfilename (): Name of the raw database file
        default_isolist (): Isolist applied to loaded models from ``dbfilename``.
        where (): Used to select which models to load.
        overwrite (): If ``True`` a new file will be created even if ``filename`` already exists.
        **where_kwargs (): Values used together with ``where``.

    Returns:
    """
    mc = ModelCollection()
    if os.path.exists(filename) and not overwrite:
        mc.load_file(filename, where=where, **where_kwargs)
    elif filename[-5:].lower() != '.hdf5' and os.path.exists(f'{filename}.hdf5') and not overwrite:
        mc.load_file(f'{filename}.hdf5', where=where, **where_kwargs)
    elif os.path.exists(dbfilename):
        mc.load_file(dbfilename, isolist=default_isolist, where=where, **where_kwargs)
        mc.save(filename)
    else:
        raise ValueError(f'Neither "{filename}" or "{dbfilename}" exist')
    return mc


class ModelCollection:
    def __repr__(self):
        models = ", ".join([f'{k}: <{v.__class__.__name__}>' for k, v in self.models.items()])
        refs = ", ".join([f'{k}: <{v.__class__.__name__}>' for k, v in self.refs.items()])
        return f'{self.__class__.__name__}(models={{{models}}}, refs={{{refs}}})'

    def _repr_markdown_(self):
        models = "\n".join([f'- [{i}] {k} ({v.__class__.__name__})' for i, (k, v) in enumerate(self.models.items())])
        refs = "\n".join([f'- {k} ({v.__class__.__name__})' for k, v in self.refs.items()])
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

        Returns:

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
        logger.info(f'Filesize: {os.path.getsize(filename) / 1024 / 1024:.0f} MB')
        logger.info(f'Time to save file: {t}')

    def _save_model(self, parent_group, model):
        group = parent_group.create_group(model.name)
        for name, value in model.saved_attrs.items():
            v = utils.asarray(value, saving=True)

            if v.ndim == 0:
                group.attrs.create(name, v)
            else:
                # Track order has to be set to true or saving will fail as there are too many
                # columns in CCSNe data
                group.create_dataset(name, data = v, compression = 'gzip', compression_opts = 9)

    def load_file(self, filename, isolist=None, where=None, **where_kwargs):
        """
        Add models from file to the current collection.

        **Note** existing models with the same name one of the loaded models will be overwritten.

        Args:
            filename ():
            isolist ():
            where ():
            **where_kwargs ():

        Returns:

        """
        logger.info(f'Loading file: {filename}')
        t0 = datetime.datetime.now()
        with h5py.File(filename, 'r') as efile:
            for name, group in efile['models'].items():
                model = self._load_model(efile, group, name, isolist, where, where_kwargs)
                #self.models[name] = model

        t = datetime.datetime.now() - t0
        logger.info(f'Time to load file: {t}')

    def _load_model(self, file, group, model_name, isolist, where, where_kwargs):
        attrs = {}
        if where is not None:
            eval = utils.model_eval.parse_where(where)

        # Load attributes
        for name, value in group.attrs.items():
            attrs[name] = utils.asarray(value)

        if 'clsname' not in attrs:
            raise ValueError(f"Model '{attrs[name]}' has no clsname")

        if where is None or eval(attrs, where_kwargs):
            logger.info(f'Loading model: {model_name} ({attrs["clsname"]})')
            for name, value in group.items():
                if not isinstance(value, h5py.Dataset): continue
                attrs[name] = utils.asarray(value)

            model = Model(self, model_name, **attrs)
            if isolist is not None:
                model.select_isolist(isolist)

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
            self.refs[refname] = self._load_model(file, group, refname, None, None, None)


    ##############
    # Get models #
    ##############
    def get_model(self, name):
        if name in self.models:
            return self.models[name]
        else:
            raise ValueError(f"No model called '{name}' exists")

    def get_ref(self, name, attr=None):
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
            clsname ():
            name ():
            **attrs ():

        Returns:

        """
        return Model(self, name, clsname=clsname, **attrs)

    def select_isolist(self, isolist=None):
        for model in self.models.values():
            model.select_isolist(isolist)

    def where(self, where, **where_kwargs):
        eval = model_eval.parse_where(where)
        new_collection = self.__class__()
        for model in self.models.values():
            if eval(model, where_kwargs):
                model.copy(new_collection, include_unsaved_attrs=True)

        return new_collection

    def copy(self, include_unsaved_attrs = False):
        new_collection = self.__class__()
        for model in self.models.values():
            model.copy(new_collection, include_unsaved_attrs=include_unsaved_attrs)

        return new_collection

    def internal_normalisation(self, normrat, enrichment_factor=1, relative_enrichment=True,
                               method='largest_offset', **method_kwargs):

        for model in self.models.values():
            model.internal_normalisation(normrat, enrichment_factor=enrichment_factor,
                                         relative_enrichment=relative_enrichment,
                                             method=method, **method_kwargs)

    def simple_normalisation(self, normiso, enrichment_factor=1, relative_enrichment=True):
        for model in self.models.values():
            model.simple_normalisation(normiso, enrichment_factor=enrichment_factor,
                                         relative_enrichment=relative_enrichment)

class Model:
    SUBCLASSES = {}
    ISREF = False
    def __init_subclass__(cls, **kwargs):
        # Called each time a new subclass is created
        # Registers the class so that it can be found upon loading
        super().__init_subclass__(**kwargs)
        cls.SUBCLASSES[cls.__name__] = cls

    def __new__(cls, collection, name, **kwargs):
        # When Model(_attrs, item) is called then it returns the subclass according to _attrs[clsname]
        # If a Subclass of model is called then that subclass is return
        if cls is Model:
            clsname = kwargs.get('clsname', None)
            if clsname is None:
                raise ValueError(f'No clsname is given in attrs')
            elif clsname not in cls.SUBCLASSES:
                raise ValueError(f'No model subclass called "{clsname}" exists')
            else:
                return cls.SUBCLASSES[clsname](collection, name, **kwargs)
        else:
            obj = super().__new__(cls)
            obj.__init__(collection, name, **kwargs)
            return obj
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"
    def __init__(self, collection, name, **saved_attrs):
        super().__setattr__('collection', collection)
        super().__setattr__('name', name)
        super().__setattr__('saved_attrs', ArrayAttrs())
        super().__setattr__('unsaved_attrs', Attrs())

        for k, v in saved_attrs.items():
            self.add_attr(k, v, save=True)

        self.add_attr('clsname', self.__class__.__name__, save=True, overwrite=True)

        if self.ISREF:
            self.collection.refs[self.name] = self
        else:
            self.collection.models[self.name] = self

    def __getattr__(self, name):
        if name in self.saved_attrs:
            return self.saved_attrs[name]
        elif name in self.unsaved_attrs:
            return self.unsaved_attrs[name]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __getitem__(self, name):
        return self.__getattr__(name)

    def __setattr__(self, name, value):
        raise AttributeError('Use the `add_attr` method to add attributes to this object')

    def add_attr(self, name: str, value, save: bool=False, overwrite:bool =False):
        if name in self.__dict__:
            raise AttributeError(f"'{self.__class__.__name__}' object has an attribute '{name}'")

        if (name in self.saved_attrs or name in self.unsaved_attrs) and overwrite is False:
            raise AttributeError(f"'{name}' already exists")
        elif name in self.saved_attrs:
            self.saved_attrs.pop(name)
        elif name in self.unsaved_attrs:
            self.unsaved_attrs.pop(name)

        if save is True:
            self.saved_attrs.__setitem__(name, value)
        else:
            self.unsaved_attrs.__setitem__(name, value)

    def get_ref(self, name, attr=None):
        return self.collection.get_ref(name, attr)

    def change_name(self, name):
        self.collection.models.pop(self.name)
        super().__setattr__('name', name)
        self.collection.models[self.name] = self

    def copy(self, to_models, include_unsaved_attrs=True):
        new_model = Model(to_models, self.name, **self.saved_attrs)

        for k, v in self.saved_attrs.items():
            # Make sure ref are in new models object
            if k[:6] == 'refid_':
                if type(v) is str and v not in to_models.refs:
                    to_models.refs[v] = self.collection.refs[v]

        if include_unsaved_attrs:
            for k, v in self.unsaved_attrs.items():
                new_model.add_attr(k, v, save=False, overwrite=True)

        return new_model

    def select_isolist(self, isolist):
        # Updates the relevant arrays inplace
        raise NotImplementedError()

    def internal_normalisation(self, normrat, abu_factor=1, relative_abu_factor=True,
                               method='largest_offset', **method_kwargs):
        raise NotImplementedError()

    def simple_normalisation(self, normiso, enrichment_factor=1, relative_enrichment=True):
        raise NotImplementedError()

class Test(Model):
    def internal_normalisation(self, normrat, enrichment_factor=1, relative_enrichment=True,
                               attrname = 'intnorm',
                               method='largest_offset', **method_kwargs):
        normrat = utils.asratios(normrat)

        # Find the all the isotopes with the same element and suffix as the normiso of normrat
        numerators = []
        for nr in normrat:
            if nr.numer.element != nr.denom.element:
                raise ValueError('The ``normrat`` numerator and normiso isotopes must be of the same element')
            if nr.numer.suffix != nr.denom.suffix:
                raise ValueError('The ``normrat`` numerator and normiso isotopes must have the same suffix')

            isotopes = utils.get_isotopes_of_element(self.abundance_keys, nr.denom.element, nr.denom.suffix)
            numerators.append(isotopes)

        stdmass = self.get_ref(self.refid_isomass, 'data')
        stdabu = self.get_ref(self.refid_isoabu, 'data')

        result = norm.internal_normalisation(self.abundance, numerators, normrat, stdmass, stdabu,
                                             enrichment_factor=enrichment_factor, relative_enrichment=relative_enrichment,
                                             method=method, **method_kwargs)

        for k, v in result.items():
            if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == 1:
                result[k] = v.squeeze()
        self.add_attr(attrname, Attrs(result), save=False, overwrite=True)
        return result

class IsoRef(Model):
    ISREF = True
    def __init__(self, collection, name, *, type, citation, data_values, data_keys, data_unit, **attrs):
        super().__init__(collection, name, type=type, citation=citation,
                         data_values=data_values, data_keys=data_keys, data_unit=data_unit, **attrs)
        self.add_attr('data_keys', utils.asisotopes(self.data_keys, allow_invalid=True), save=True,
                      overwrite=True)
        self.add_attr('data', utils.askeyarray(self.data_values, self.data_keys), save=False)

    def __repr__(self):
        return f'{self.__class__.__name__}(name={self.name}, type={self.type})'

    def _repr_markdown_(self):
        attrs = ['*name*'] + [f"*{name}*" for name in self.saved_attrs] + [f"*{name}*" for name in self.unsaved_attrs]
        return f"""
**Name**: {self.name}\\
**Class Name**: {self.__class__.__name__}\\
**Type**: {self.type}\\
**Attributes**: {", ".join(sorted(attrs))}
        """.strip()

class CCSNe(Model):
    def __init__(self, collection, name, *,
                 type, dataset, citation,
                 mass, masscoord, abundance_values, abundance_keys, abundance_unit,
                 refid_isoabu, refid_isomass,
                 **attrs):
        super().__init__(collection, name, type=type, dataset=dataset, citation=citation,
                        mass=mass, masscoord=masscoord, abundance_values=abundance_values, abundance_keys=abundance_keys,
                        abundance_unit=abundance_unit, refid_isoabu=refid_isoabu, refid_isomass=refid_isomass,
                         **attrs)
        self.add_attr('abundance_keys', utils.asisotopes(self.abundance_keys, allow_invalid=True), save=True,
                      overwrite=True)
        self.add_attr('abundance', utils.askeyarray(self.abundance_values, self.abundance_keys), save=False)

    def __repr__(self):
        return f'{self.__class__.__name__}(name={self.name}, type={self.type}, dataset={self.dataset}, mass={self.mass})'

    def _repr_markdown_(self):
        attrs = ['*name*'] + [f"*{name}*" for name in self.saved_attrs] + [f"*{name}*" for name in self.unsaved_attrs]
        return f"""
**Name**: {self.name}\\
**Class Name**: {self.__class__.__name__}\\
**Type**: {self.type}\\
**Dataset**: {self.dataset}\\
**Mass**: {self.mass}\\
**Attributes**: {", ".join(sorted(attrs))}
        """.strip()
    def select_isolist(self, isolist):
        abu, keys = utils.select_isolist(isolist, self.abundance_values, self.abundance_keys,
                                         massunit=True if self.abundance_unit == 'mass' else False)

        self.add_attr('abundance_values', abu, save=True, overwrite=True)
        self.add_attr('abundance_keys', keys, save=True, overwrite=True)
        self.add_attr('abundance', utils.askeyarray(abu, keys), save=False, overwrite=True)

    def internal_normalisation(self, normrat, enrichment_factor=1, relative_enrichment=True,
                               attrname='intnorm',
                               method='largest_offset', **method_kwargs):
        normrat = utils.asratios(normrat)

        # Find the all the isotopes with the same element and suffix as the normiso of normrat
        numerators = []
        for nr in normrat:
            if nr.numer.element != nr.denom.element:
                raise ValueError('The ``normrat`` numerator and normiso isotopes must be of the same element')
            if nr.numer.suffix != nr.denom.suffix:
                raise ValueError('The ``normrat`` numerator and normiso isotopes must have the same suffix')

            isotopes = utils.get_isotopes_of_element(self.abundance_keys, nr.denom.element, nr.denom.suffix)
            numerators.append(isotopes)

        stdmass = self.get_ref(self.refid_isomass, 'data')
        stdabu = self.get_ref(self.refid_isoabu, 'data')

        result = norm.internal_normalisation(self.abundance, numerators, normrat, stdmass, stdabu,
                                             enrichment_factor=enrichment_factor, relative_enrichment=relative_enrichment,
                                             method=method, **method_kwargs)

        for k, v in result.items():
            if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == 1:
                result[k] = v.squeeze()
        self.add_attr(attrname, Attrs(result), save=False, overwrite=True)
        return result

    def simple_normalisation(self, normiso, enrichment_factor = 1, relative_enrichment=True, attrname='simplenorm',):
        normiso = utils.asisotopes(normiso)

        numerators = []
        for denom in normiso:
            isotopes = utils.get_isotopes_of_element(self.abundance_keys, denom.element, denom.suffix)
            numerators.append(isotopes)

        stdabu = self.get_ref(self.refid_isoabu, 'data')
        stdabu_massunit = True if self.get_ref(self.refid_isoabu, 'data_unit') == "mass" else False

        abu = self.abundance
        abu_massunit = True if self.abundance_unit == 'mass' else False

        result = norm.simple_normalisation(abu, numerators, normiso, stdabu,
                                           enrichment_factor=enrichment_factor, relative_enrichment=relative_enrichment,
                                           abu_massunit=abu_massunit, stdabu_massunit=stdabu_massunit)

        for k, v in result.items():
            if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == 1:
                result[k] = v.squeeze()

        self.add_attr(attrname, result, save=False, overwrite=True)
        return result


##################
### Ref Values ###
##################
def load_csv_h(filename):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, skipinitialspace=True)
        data = [row for row in reader if row[0][0] != '#']

    return np.transpose([float(i) for i in data[1]]), utils.asisotopes(data[0], allow_invalid=True)


def load_ppn(filename):
    isotopes = []
    values = []
    with open(filename, 'r') as f:
        for row in f.readlines():
            isotopes.append(row[3:9].replace(' ', ''))
            values.append(float(row[10:].strip()))

    return np.transpose(values), utils.asisotopes(isotopes, allow_invalid=True)
