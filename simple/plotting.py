import re

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import functools
from collections.abc import Sequence

import numpy as np

import simple.models, simple.roseaxes
from simple import utils

import logging

from simple.utils import set_default_kwargs, add_shortcut

logger = logging.getLogger('SIMPLE.plot')

__all__ = ['create_rose_plot', 'create_subplots',
           'get_data', 'add_weights',
           'plot', 'slope', 'hist',
           'create_legend', 'update_axes',]


# colours appropriate for colour blindness
# Taken from https://davidmathlogic.com/colorblind/#%23000000-%23E69F00-%2356B4E9-%23009E73-%23F0E442-%230072B2-%23D55E00-%23CC79A7
default_colors=utils.EndlessList(["#D55E00", "#56B4E9", "#009E73", "#E69F00", "#CC79A7", "#0072B2", "#F0E442"])
"""
An [``Endlesslist``][simple.plot.EndlessList] containing the default colors used by simple plotting functions.
"""

default_linestyles = utils.EndlessList(['-', (0, (4, 4)), (0, (2, 1)),
                                        (0, (4,2,1,2)), (0, (4,2,1,1,1,2)), (0, (4,2,1,1,1,1,1,2)),
                                        (0, (2,1,2,2,1,2)), (0, (2,1,2,2,1,1,1,2)), (0, (2,1,2,2,1,1,1,1,1,2)),
                                        (0, (2,1,2,1,2,2,1,2)), (0, (2,1,2,1,2,2,1,1,1,2)), (0, (2,1,2,1,2,2,1,1,1,1,1,2))])
"""
[``Endlesslist``][simple.plot.EndlessList] default line styles used by simple plotting functions.
"""

default_markers = utils.EndlessList(["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "p", "d", "H"])
"""
[``Endlesslist``][simple.plot.EndlessList] default marker styles used by simple plotting functions.
"""

def get_axes(axes, projection=None):
    """
    Return the axes that should be used for plotting.

    Args:
        axes (): Must either be ``None``, in which case ``plt.gca()`` will be returned, a matplotlib axes instance, or
            any object that has a ``.gca()`` method.
        projection (): If given, an exception is raised if ``axes`` does not have this projection.
    """
    if axes is None:
        axes = plt.gca()
    elif isinstance(axes, Axes):
        pass
    elif hasattr(axes, 'gca'):
        axes = axes.gca() # if ax = plt
    else:
        raise ValueError('ax must be an Axes, Axes instance or have a gca() method that return an Axes')

    if projection is not None and axes.name != projection:
        raise TypeError(f'The selected ax has the wrong projection. Expected {projection} got {axes.name}.')
        
    return axes

def get_models(models, where=None, **where_kwargs):
    """
    Return a selection of models.

    If *models* is a ModelCollection a new ModelCollection will be returned. Otherwise a list of Models will
    be returned.

    Args:
        models (): A collection of models or a list/tuple of models.
        where (str): Only models fitting this criteria will be selected. If not given all models are selected.
        **where_kwargs (): Keyword arguments to go with the ``where`` string.
    """
    if isinstance(models, simple.models.ModelCollection):
        pass
    elif isinstance(models, (list, tuple)) and False not in [isinstance(m, simple.models.ModelBase) for m in models]:
        pass
    else:
        raise TypeError(f'models must be a ModelCollection object or a list/tuple of Models objects - not {type(models)}.')

    # select models
    if where is not None:
        models = utils.models_where(models, where, **where_kwargs)

    return models

def parse_lscm(linestyle = False, color = False, marker=False):
    """
    Convert the ``linestyle``, ``color`` and ``marker`` arguments into [EndlessList][simple.plot.EndlessList]
    objects for plotting.

    Args:
        linestyle (): Either a single line style or a list of line styles. If ``True`` then the
            [default line styles][simple.plot.default_linestyles] is returned. If ``False`` or ``None`` a list
            containing only the no line sentinel is returned.
        color (): Single colour or a list of colours. If ``True`` then the
            [default colors][simple.plot.default_colors] is returned. If ``False`` or ``None`` a list
            containing only the color black is returned.
        marker (): Either a marker shape or a list of marker shapes. If ``True`` then the
            [default marker shapes][simple.plot.default_markers] shapes is returned. If ``False`` or ``None`` a
            list containing only the no marker sentinel is returned.

    Returns:
        (EndlessList, EndlessList, EndlessList): linestyles, colors, markers
    """
    if color is False or color is None:
        colors = utils.EndlessList(["#000000"])
    elif color is True:
        colors = default_colors
    else:
        colors = utils.EndlessList(color)

    if linestyle is False or linestyle is None:
        linestyles = utils.EndlessList([""])
    elif linestyle is True:
        linestyles = default_linestyles
    else:
        linestyles = utils.EndlessList(linestyle)

    if marker is False or marker is None:
        markers = utils.EndlessList([""])
    elif marker is True:
        markers = default_markers
    else:
        markers = utils.EndlessList(marker)

    return linestyles, colors, markers

@add_shortcut('AB', mosaic='AB', fig_size=(12, 5.5))
@add_shortcut('AB_CD', mosaic='AB;CD', fig_size=(12, 11))
@set_default_kwargs(layout='constrained')
def create_subplots(mosaic, update_fig=True, kwargs=None):
    """
    Create a series of subplots.

    See [matplotlib's documentation](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplot_mosaic.html#matplotlib.pyplot.subplot_mosaic)
    for a more thorough description of the *mosaic* argument and other possible arguments.

    Args:
        mosaic (): A visual layout of how you want your subplots to be arranged. This can either
            be a nested list of strings or a single string with each subplot represented by a single character, where
            ``;`` represent a new row.
        update_fig (): If ``True`` (default), the figure will be updated using any *kwargs* prefixed with ``fig_``.
        kwargs (): Keyword arguments to go with the ``mosaic`` argument.

    Returns:
        dict: A dictionary containing the subplots.
    """
    # pass extract fig and update

    fig_kwargs = kwargs.pop_many(prefix='fig', remove_prefix=False)

    fig, subplots = plt.subplot_mosaic(mosaic, **kwargs)

    update_figure(fig, fig_kwargs, update_fig=update_fig)

    for key, sp in subplots.items():
        sp._SIMPLE_subplot_dict = subplots
        sp._SIMPLE_subplot_dict_name = key

    return subplots

################
### get data ###
################
@utils.set_default_kwargs()
def get_data(models, axis_names, *, where=None, latex_labels = True,
             key=None, default_attrname=None, unit=None, default_value = np.nan,
             key_in_label=None, numer_in_label=None, denom_in_label=None,
             model_in_label=None, unit_in_label=None, attrname_in_label=None, axis_name_in_label=None,
             label = True, prefix_label = None, suffix_label = None,
             mask = None, mask_na = True,
             kwargs = None):
    """
    Get one or more datasets from a group of models together with suitable labels.

    Each data point is a dictionary that contains a value for each of the axis given in *axis_names* plus a label
    describing the data point. The value for each axis is determined by the *key* argument. This argument has two
    possible components; the name of the attribute and the index, or key, to be applied this, or the
    *default_attrname*, attribute.

    The name of the attribute must start with a ``.`` followed by the path to the attribute relative to the Model
    object using successive ``.`` for nested attributes, e.g. ``.intnorm.eRi``.

    The index, or key, part of the *key* can either be an integer, a slice or a sequence of keys seperated by ``,``.
    The keys will be parsed into either [Isotope][simple.utils.Isotope], [Ratio][simple.utils.Ratio], or
    [Element][simple.utils.Element] strings. If a key is given it is assumed that the attribute contains an isotope
    key array. Therefore, Element strings will be replaced with all the isotopes of that element
    present in the attribute (Across all models) and Ratio strings will return the numerator value divided by the
    denominator value.

    If the attribute name is given in *key* then the index, or key, part must be enclosed in square brackets, e.g.
    ``.intnorm.eRi[105Pd]``. If the *default_attrname* should be used then *key* should only contain the index, or key.

    By the default the label for each data point only contains the information is not shared with all other data points.
    Information that is shared between all data points is instead included in the axis labels.

    Args:
        models (): A collection of models to plot. A subselection of these models can be made using the *where*
            argument.
        axis_names ():
        where (str): If given will be used to create a subselection of *models*. Any *kwargs* prefixed
            with ``where_`` will be supplied as keyword arguments. See
             [``ModelCollection.where``][(]simple.models.ModelCollection.where] for more details.
        latex_labels (bool): Whether to use the latex formatting in the labels, when available.
        key (str, int, slice): This can either be a valid index to the *default_attrname* array or the path, with
            or without a valid index, of a different attribute. Accepts either single universal value
            or a list of values, one for each axis (See below for details).
        default_attrname (): The name of the default attribute to use if *xkey* and *ykey* are indexes. By default,
            the default key array is used. Accepts either single universal value or a list of values, one for each
            axis (See below for details).
        unit (): The desired unit for the *xkey* and *ykey*. Different units for *xkey* and *ykey* can be specified
            by supplying a ``(<xkey_unit>, <ykey_unit>)`` sequence. Accepts either single universal value
            or a list of values, one for each axis (See below for details).
        default_value (): The value given to invalid indexes of arrays. Must have a shape compatible with the size
            of the indexed array. Accepts either single universal value
            or a list of values, one for each axis (See below for details).
        key_in_label (bool): Whether to include the key index in the label. Accepts either single universal value
            or a list of values, one for each axis (See below for details).
        numer_in_label (bool): Whether to include the numerator of a key index in the label. Accepts either single
            universal value or a list of values, one for each axis (See below for details).
        denom_in_label (bool): Whether to include the denominator of a key index in the label. Accepts either single
            universal value or a list of values, one for each axis (See below for details).
        model_in_label (bool): Whether to include the model name in the label. Accepts either single universal value
            or a list of values, one for each axis (See below for details).
        unit_in_label (bool): Whether to include the unit in the label. Accepts either single universal value
            or a list of values, one for each axis (See below for details).
        attrname_in_label (bool): Whether to include the attribute name in the label. By default
            the name is only included if it is different from the *default_attrname*. Accepts a single universal
            value or a list of values, one for each axis (See below for details).
        axis_name_in_label (bool): Whether to include the axis name in the label. Accepts either single universal value
            or a value for each axis (See below for details).
        label (str, bool, None): The label for individual datapoints. Accepts either a single universal value or
            a list of values, one per data point (See below for details).
        prefix_label (): Text to be added to the beginning of each data point label. Accepts either a single universal
            value or a list of values, one per data point (See below for details).
        suffix_label (): Text to be added at the end of each data point label. Accepts either a single universal
            value or a list of values, one per data point (See below for details).
        mask (str, int, slice): Can be used to apply a mask to the data which is plotted. See the
            ``get_mask`` function of the Model object. Accepts either a single universal
            value or a list of values, one per model (See below for details).
        mask_na (bool): If ``True`` masked values will be replaced by ``np.nan`` values. Only works if all arrays
            in a dataset have a float based datatype. Accepts either a single universal
            value or a list of values, one per model (See below for details).
        **kwargs:

    One per axis arguments:
        These arguments allow you to set a different value for each axis in *axis_names*. This can be
        either a single value used for all the axis or a sequence of values, one per axis.

        It is also possible to define the value for a specific axis by  including a keyword argument consiting
        of the axis name followed directly by the argument name. The value specified this way will take presidence over
        the value given by the argument itself. For example ``xkey=102Pd`` will set the *key* argument for
        the *x* axis to ``102Pd``.


    One per data point arguments:
        These arguments allow you to set a different value for each data point. The number of data points is
        equal to the number of models multiplied by the number of datasets generated. This can be
        either a single value used for all the axis or a sequence of values, one per data point.


    One per model arguments:
        These arguments allow you to set a different value for each model in *models*. This can be
        either a single value used for all the axis or a sequence of values, one per model.


    Returns:
        Tuple[dict, dict]: Two dictionaries containing:

            - A dictionary with the data points for each model, mapped to the model name

            - A dictionary containing labels for each axis, mapped to the axis name.

    Examples:
        Here is an example of how the return data can be used.
        ```
        model_datapoints, axis_labels = simple.get_data(models, 'x, y', xkey=..., ykey=...)

        # Set the axis labels
        plt.set_xlabel(axis_labels['x'])
        plt.set_ylabel(axis_labels['y'])

        # Iterate though the data and plot it
        for model_name, datapoints in model_datapoints.items():
            for dp in datapoints:
                plt.plot(dp['x'], dp['y'], label=dp['label'])
        ```

    """

    where_kwargs = kwargs.pop_many(prefix='where')
    models = get_models(models, where=where, where_kwargs=where_kwargs)

    if type(axis_names) is dict:
        axis_name_args = list(axis_names.keys())
        axis_key_args = list(axis_names.values())
    elif type(axis_names) is str:
        if ',' in axis_names:
            axis_name_args = list(n.strip() for n in axis_names.split(','))
        else:
            axis_name_args = list(n.strip() for n in axis_names.split())
        axis_key_args = None
    else:
        raise TypeError('``axis_name`` must be a string or a dict')

    lenargs = len(axis_name_args)
    lenmodels = len(models)

    def one_per_n(n, n_name, arg_name, arg_value):
        if isinstance(arg_value, str) or not isinstance(arg_value, Sequence):
            return [arg_value for i in range(n)]
        elif len(arg_value) == n:
            if type(arg_value) is list:
                return arg_value
            else:
                return list(arg_value)
        else:
            raise ValueError(f'Length of ``{arg_name}`` ({len(arg_value)}) must be equal to number of {n_name} ({n})')

    def one_per_arg(name, value):
        args = one_per_n(lenargs, 'axis', name, value)

        for i, axis in enumerate(axis_name_args):
            if (k:=f'{axis}{name}') in kwargs:
                args[i] = kwargs.pop(k)
            if (k:=f'{axis}{name}') in kwargs:
                args[i] = kwargs.pop(k)

        return args


    def parse_key_string(key, data_arrays):
        try:
            return utils.asisotopes(key), 'iso'
        except:
            pass

        try:
            return utils.asratios(key), 'rat'
        except:
            pass

        try:
            elements =  utils.aselements(key)
        except:
            raise ValueError(f'Unable to parse "{key}" into a sequence of valid Element, Isotope or Ratio string.')
        else:
            # Because the key list should be the same for all models we need to go through them all here
            # incase some have more or less isotopes in the specified data array
            all_isotopes = ()
            for element in elements:
                element_isotopes = []
                for data in data_arrays:
                    if not isinstance(data, (np.ndarray)) or data.dtype.fields is None:
                        raise ValueError(f'Data array "{attrname}" of model {model.name} is not a key array. '
                                         f'Cannot extract isotope keys.')

                    for iso in utils.get_isotopes_of_element(data.dtype.fields, element):
                        if iso not in element_isotopes:
                            element_isotopes.append(iso)
                all_isotopes += tuple(sorted(element_isotopes, key=lambda iso: float(iso.mass)))
            return all_isotopes, 'iso'

    def get_data_label(keylabels, keys, key_in_label, numer_in_label, denom_in_label):
        labels = []
        for key in keys:
            if type(key) is utils.Isotope:
                if key_in_label:
                    label = keylabels.get(key, f"!{key}")
                else:
                    label = ''
            elif key_in_label:
                if numer_in_label and denom_in_label:
                    label = f'{keylabels.get(key.numer, f"!{key.numer}")}/{keylabels.get(key.denom, f"!{key.denom}")} '
                elif numer_in_label:
                    label = keylabels.get(key.numer, f"!{key.numer}")
                elif denom_in_label:
                    label = keylabels.get(key.denom, f"!{key.denom}")
                else:
                    label = ''
            else:
                label = ''
            labels.append(label)
        return labels

    def get_data_index(data_array, index, mi, ai):
        try:
            return data_array[index]
        except ValueError as error:
            if isinstance(index, str):
                logger.warning(f'{models[mi]}.{attrname_a[ai]}: Missing field "{index}" replaced by the default value ({default_value})')
                return np.full(len(data_array), default_value_args[ai])
            else:
                raise error

    if axis_key_args is None:
        axis_key_args = one_per_arg('key', key)

    default_attrname_args = one_per_arg('default_attrname', default_attrname)
    desired_unit_args = one_per_arg('unit', unit)
    default_value_args = one_per_arg('default_value', default_value)
    key_in_label_args = one_per_arg('key_in_label', key_in_label)
    numer_in_label_args = one_per_arg('numer_in_label', numer_in_label)
    denom_in_label_args = one_per_arg('denom_in_label', denom_in_label)
    model_in_label_args = one_per_arg('model_in_label', model_in_label)
    unit_in_label_args = one_per_arg('unit_in_label', unit_in_label)
    attrname_in_label_args = one_per_arg('attrname_in_label', attrname_in_label)
    axis_name_in_label_args = one_per_arg('axis_name_in_label', axis_name_in_label)

    mask_args = one_per_n(lenmodels, 'models', 'mask', mask)
    mask_na_args = one_per_n(lenmodels, 'models', 'mask_na', mask_na)

    # _a -   [arg1, arg2, ...]
    # _am -  [(arg1_model1, arg1_model2, ...), (arg2_model1, arg2_model2, ...)]
    # _amk - [({arg1_model1_key1, arg1_model1_key2, ...}, {arg1_model2_key1, arg1_model2_key2, ...}), ...]
    attrname_a, keys_ak, keytype_a = [], [], []
    data_arrays_am, data_units_am = [], []
    data_label_am, data_keylabels_am = [], []
    for ai, arg in enumerate(axis_key_args):
        if type(arg) is not str:
            attrname = utils.parse_attrname(default_attrname_args[ai])
            key = arg
        else:
            if arg.startswith('.'):
                m = re.match(r'^([A-Za-z0-9_.]+)(?:\[(.*?)\])?$', arg)
                if m:
                    attrname = utils.parse_attrname(m.group(1))
                    key = m.group(2)
                    if attrname_in_label_args[ai] is None:
                        attrname_in_label_args[ai] = True
                else:
                    raise ValueError(f'Invalid arg: {key}')
            else:
                attrname = utils.parse_attrname(default_attrname_args[ai])
                key = arg

        if attrname_in_label_args[ai] is None and attrname is None:
            attrname_in_label_args[ai] = True

        attrname_a.append(attrname)

        # Here we get all the data arrays
        # For this we only need the attrname
        data_arrays_am.append([])
        data_units_am.append([])
        data_label_am.append([])
        data_keylabels_am.append([])
        for model in models:
            data, data_unit = model.get_array(attrname, desired_unit_args[ai])
            data_arrays_am[-1].append(data)
            data_units_am[-1].append(data_unit)

            attr_label, key_labels = model.get_array_labels(attrname, latex=latex_labels)
            data_label_am[-1].append(attr_label)
            data_keylabels_am[-1].append(key_labels)


        # Parse the key
        # Is the key is an element symbol it will extract all isotopes of that element from the data
        # Hence why we need to get the data arrays before this step
        if type(key) is str:
            # Check if key is an integer index or slice
            m = re.match(r'^\s*(-?\d+)\s*$|^\s*(-?\d*)\s*:\s*(-?\d*)\s*(?::\s*(-?\d*))?\s*$', key)
            if m:
                if m.group(1):
                    key = int(m.group(1))
                else:  # Slice
                    key = slice(int(m.group(2)) if m.group(2) is not None else None,
                                int(m.group(3)) if m.group(3) is not None else None,
                                int(m.group(4)) if m.group(4) is not None else None)
                key = (key,)
                keytype = 'index'

            else:
                key, keytype = parse_key_string(key, data_arrays_am[-1])
        elif type(key) is int or type(key) is slice:
            key = (key,)
            keytype = 'index'
        elif key is None:
            key = (key, )
            keytype = 'none'
        else:
            key, keytype = parse_key_string(key, data_arrays_am[-1])

        keys_ak.append(key)
        keytype_a.append(keytype)

    # Make sure the size of the keys is the same for all args
    size = {len(k) for k in keys_ak}
    size.discard(1)
    if len(size) > 1:
        raise ValueError(f'Length of indexes for not compatible {[len(k) for k in attrname_a]}')
    elif len(size) == 1:
        lenkeys = size.pop()
    else:
        lenkeys = 1

    for ai in range(lenargs):
        if len(keys_ak[ai]) != lenkeys:
            # current size can only be 1. Repeat until it is the correct length
            keys_ak[ai] = [keys_ak[ai][0] for i in range(lenkeys)]

    axis_label_args = [kwargs.pop(f'{name}label', True) for name in axis_name_args]
    axis_prefix_label_args = [kwargs.pop(f'{name}prefix_label', '') for name in axis_name_args]
    axis_suffix_label_args = [kwargs.pop(f'{name}suffix_label', '') for name in axis_name_args]
    label_args = one_per_n(lenmodels * lenkeys, 'datapoints', 'label', label)
    prefix_label_args = one_per_n(lenmodels * lenkeys, 'datapoints', 'prefix_label', prefix_label)
    suffix_label_args = one_per_n(lenmodels * lenkeys, 'datapoints', 'suffix_label', suffix_label)

    result_data = {}
    result_axis_label = {}
    data_labels_amk = []

    # Get the arg label which can be used as an axis label
    # Get the data point label for each arg. These are all combined later
    # Model name is not added. It will be added once the individual arg labels have been joined
    for ai in range(lenargs):
        keys = keys_ak[ai]
        keytype = keytype_a[ai]

        # Find common keys that can go into the arg label
        if keytype == 'iso':
            unique_keylabels = set()
            for mi, keylabels in enumerate(data_keylabels_am[ai]):
                if keylabels is None:
                    raise ValueError(f"Data array '{attrname_a[ai]}' of model '{models[mi].name}' is not a key array.")
                unique_keylabels = {*unique_keylabels, *(keylabels.get(k, None) for k in keys)}

            unique_keylabels.discard(None)
            if len(unique_keylabels) == 1: # Same label for all data points
                arg_label = unique_keylabels.pop()
                if key_in_label_args[ai] is None:
                    key_in_label_args[ai] = False
                if axis_name_in_label_args[ai] is None:
                    axis_name_in_label_args[ai] = False
            else:
                arg_label = f"<{axis_name_args[ai]}>"
                if key_in_label_args[ai] is None:
                    key_in_label_args[ai] = True
                if axis_name_in_label_args[ai] is None:
                    axis_name_in_label_args[ai] = True

        elif keytype == 'rat':
            unique_n_keylabels = {}
            unique_d_keylabels = {}
            for keylabels in data_keylabels_am[ai]:
                if keylabels is None:
                    raise ValueError(f"Data array '{attrname[ai]}' of model '{models[ai].name}' is not a key array.")

                unique_n_keylabels = {*unique_n_keylabels, *(keylabels.get(k.numer, None) for k in keys)}
                unique_d_keylabels = {*unique_d_keylabels, *(keylabels.get(k.denom, None) for k in keys)}

            unique_n_keylabels.discard(None)
            unique_d_keylabels.discard(None)

            if key_in_label_args[ai] is not None:
                if numer_in_label_args[ai] is None:
                    numer_in_label_args[ai] = key_in_label_args[ai]
                if denom_in_label_args[ai] is None:
                    denom_in_label_args[ai] = key_in_label_args[ai]

            if len(unique_n_keylabels) == 1 and len(unique_d_keylabels) == 1:
                arg_label = f"{unique_n_keylabels.pop()} / {unique_d_keylabels.pop()}"
                if key_in_label_args[ai] is None:
                    key_in_label_args[ai] = False
                if numer_in_label_args[ai] is None:
                    numer_in_label_args[ai] = False
                if denom_in_label_args[ai] is None:
                    denom_in_label_args[ai] = False
                if axis_name_in_label_args[ai] is None:
                    axis_name_in_label_args[ai] = False
            elif len(unique_n_keylabels) == 1:
                arg_label = f'{unique_n_keylabels.pop()} / <{axis_name_args[ai]}>'
                if key_in_label_args[ai] is None:
                    key_in_label_args[ai] = True
                if numer_in_label_args[ai] is None:
                    numer_in_label_args[ai] = False
                if denom_in_label_args[ai] is None:
                    denom_in_label_args[ai] = True
                if axis_name_in_label_args[ai] is None:
                    axis_name_in_label_args[ai] = True
            elif len(unique_d_keylabels) == 1:
                arg_label = f"<{axis_name_args[ai]}> / {unique_d_keylabels.pop()}"
                if key_in_label_args[ai] is None:
                    key_in_label_args[ai] = True
                if numer_in_label_args[ai] is None:
                    numer_in_label_args[ai] = True
                if denom_in_label_args[ai] is None:
                    denom_in_label_args[ai] = False
                if axis_name_in_label_args[ai] is None:
                    axis_name_in_label_args[ai] = True
            else:
                arg_label = f"<{axis_name_args[ai]}>"
                if key_in_label_args[ai] is None:
                    key_in_label_args[ai] = True
                if numer_in_label_args[ai] is None:
                    numer_in_label_args[ai] = True
                if denom_in_label_args[ai] is None:
                    denom_in_label_args[ai] = True
                if axis_name_in_label_args[ai] is None:
                    axis_name_in_label_args[ai] = True

        else:
            arg_label = ''

            # Neither are possible so just set to False
            key_in_label_args[ai] = False
            axis_name_in_label_args[ai] = False

        # Create labels for each key
        data_labels_amk.append([])
        if keytype == 'iso' or keytype == 'rat':
            for keylabels in data_keylabels_am[ai]:
                data_labels_amk[-1].append(get_data_label(keylabels, keys,
                                                          key_in_label_args[ai], numer_in_label_args[ai], denom_in_label_args[ai]))
                if attrname_in_label_args[ai] is None:
                    attrname_in_label_args[ai] = False

        else:
            # No keys so just creates an empty label
            data_labels_amk[-1].extend([['' for i in range(lenkeys)] for j in range(lenmodels)])
            if attrname_in_label_args[ai] is None:
                attrname_in_label_args[ai] = True

        # Add the unit either to arg label if common across all models or to each key label
        # [unit] added to the end of the string
        if unit_in_label_args[ai] is not False:
            unique_data_units = {*data_units_am[ai]}
            unique_data_units.discard(None)
            if len(unique_data_units) == 1:
                arg_label = f'{arg_label} [{unique_data_units.pop()}]'
            elif len(unique_data_units) > 1:
                for mi, arg_data_labels_k in enumerate(data_labels_amk[-1]):
                    for ki, key in enumerate(arg_data_labels_k):
                        if data_units_am[ai][mi] is not None:
                            data_labels_amk[-1][mi][ki] = f'{key} [{data_units_am[ai][mi]}]'.strip()

        # Add the attrname either to arg label if common across all models or to each key label
        # arrname added to the start of the string
        if attrname_in_label_args[ai]:
            unique_data_label = {*data_label_am[ai]}
            unique_data_label.discard(None)
            if len(unique_data_label) == 1:
                if arg_label == '':
                    arg_label = unique_data_label.pop().strip()
                else:
                    arg_label = f'{unique_data_label.pop()} | {arg_label}'.strip()
            elif len(unique_data_label) > 1:
                for mi, arg_data_labels_k in enumerate(data_labels_amk[-1]):
                    for ki, key in enumerate(arg_data_labels_k):
                        if data_label_am[ai][mi] is not None:
                            if key == '':
                                data_labels_amk[-1][mi][ki] = data_label_am[ai][mi].strip()
                            else:
                                data_labels_amk[-1][mi][ki] = f'{data_label_am[ai][mi]} | {key}'.strip()

        axis_label_arg = axis_label_args[ai]
        if axis_label_arg is True:
            prefix = axis_prefix_label_args[ai]
            suffix = axis_suffix_label_args[ai]
            if prefix:
                arg_label = f"{prefix}{arg_label}"
            if suffix:
                arg_label = f"{arg_label}{suffix}"

            result_axis_label[axis_name_args[ai]] = arg_label or None
        else:
            result_axis_label[axis_name_args[ai]] = axis_label_arg or None

    has_labels = False
    for mi in range(lenmodels):
        results = []
        for ki in range(lenkeys):
            results.append({})

            label = ''
            for ai in range(lenargs):
                data_array = data_arrays_am[ai][mi]
                keytype = keytype_a[ai]

                if keytype == 'rat':
                    key = keys_ak[ai][ki]
                    n = get_data_index(data_array, key.numer, mi, ai)
                    d = get_data_index(data_array, key.denom, mi, ai)
                    data = n/d

                elif keytype == 'iso' or keytype == 'index':
                    key = keys_ak[ai][ki]
                    data = get_data_index(data_array, key, mi, ai)
                else: # keytype == 'none'
                    data = data_array

                results[-1][axis_name_args[ai]] = data

                data_label = data_labels_amk[ai][mi][ki].strip()
                if data_label != '':
                    if axis_name_in_label_args[ai]:
                        label += f"<{axis_name_args[ai]}: {data_label}>"
                    else:
                        label += data_label

            if mask_args[mi] is not None:
                imask = models[mi].get_mask(mask, **results[-1])
                if mask_na_args[mi] and False not in (np.issubdtype(v.dtype, np.floating) for v in results[-1].values()):
                    for k, v in results[-1].items():
                        v = v.copy()
                        v[np.logical_not(imask)] = np.nan
                        results[-1][k] = v
                else:
                    for k, v in results[-1].items():
                        results[-1][k] = v[imask]

            label_arg = label_args[(mi * lenkeys) + ki]
            if label_arg is True:
                prefix = prefix_label_args[mi * lenkeys + ki]
                suffix = suffix_label_args[mi * lenkeys + ki]

                if model_in_label_args[ai] or (model_in_label_args[ai] is None and lenmodels > 1):
                    if label == '':
                        label = models[mi].name
                    else:
                        label = f'{label} ({models[mi].name})'.strip()
                if type(prefix) is str:
                    label = f"{prefix}{label}"
                if type(suffix) is str:
                    label = f"{label}{suffix}"

                results[-1]['label'] = label.strip() or None
            else:
                results[-1]['label'] = label_arg or None
            if results[-1]['label']:
                has_labels = True

        result_data[models[mi]] = tuple(results)

    if has_labels and kwargs.get('legend', False) is None:
        kwargs['legend'] = True

    return result_data, result_axis_label

@utils.set_default_kwargs()
def add_weights(modeldata, axis, weights=1, *,
                sum_weights=True, norm_weights=True,
                default_attrname=None, unit=None, default_value=0,
                mask=None, mask_na=True, axisname='w'):
    """
    Add weights to the specified axis of each datapoint in the modeldata dictionary.

    This function appends a new array of weights (under `axisname`) to each datapoint
    in `modeldata`. The weights can be a constant or a string referring to data to be individually
    retrieved from each model. Optionally, the weights can be summed, normalized, and masked for missing data.

    The 'mask' and 'mask_na' arguments should be the same as those used to generate `modeldata` to ensure
    consistent results.

    Args:
        modeldata (dict): The data dictionary returned from `get_data`. It should be a
            dict of models, each containing a list of datapoint dictionaries.
        axis (str): The axis in the datapoints that the weights correspond to (e.g., 'x', 'y').
        weights (int, float, str): The weights to add. Can be:
            - A scalar to apply uniformly,
            - A string key that will be used to retrieve data from each model individually.
        sum_weights (bool): If True and `weights` is a string consisting of multiple keys, the values for the different
            keys are summed together and used for each datapoint for a given model. Default is True.
        norm_weights (bool): If True, normalise weights along the specified axis. Default is True.
        default_attrname (str or None): Attribute name to use when fetching weights if not included
            in labels. Optional.
        unit (str or None): Unit to assign to the fetched weight values. Optional.
        default_value (float): Default value to assign if weights are missing. Default is 0.
        mask (str or None): Optional mask to apply to the data when computing or assigning weights.
        mask_na (bool): If True, mask values will be replaced with NaNs. If False, masked values are omitted.
        axisname (str): The name under which the weight data will be stored in each datapoint. Default is 'w'.

    Returns:
        dict: The modified `modeldata`, with weight arrays added to each datapoint.

    Raises:
        ValueError: If the number of weight arrays does not match the number of datapoints
            and they cannot be broadcast.
    """

    # Remove any previous weights
    for model in modeldata:
        for dp in modeldata[model]:
            dp.pop(axisname, None)

    models = list(modeldata.keys())
    if type(weights) is str:
        modeldata_w, axis_labels_w = get_data(models, {'w': weights},
                                              mask=mask, mask_na=mask_na,
                                              default_attrname=default_attrname, unit=unit,
                                              default_value=default_value,
                                              attrname_in_label=True, model_in_label=False, axis_name_in_label=False,
                                              latex_labels=False)

        if sum_weights:
            for model, datapoints_w in modeldata_w.items():
                if len(datapoints_w) > 1:
                    labels = [dw.get('label', 'Missing label') for dw in datapoints_w]
                    logger.info(
                        f'{model}: Calculating weights by adding together: {axis_labels_w["w"]} <w: {", ".join(labels)}>')
                    modeldata_w[model] = [{axisname: functools.reduce(np.add, [dw['w'] for dw in datapoints_w])}]

    else:
        modeldata_w = {}
        for model in models:
            datapoints_xy = modeldata[model]
            if mask:
                m = model.get_mask(mask)
                if mask_na:
                    modeldata_w[model] = [{axisname: np.full(m.shape, weights, dtype=np.float64)}]
                    modeldata_w[model][0][axisname][np.logical_not(m)] = np.nan
                else:
                    modeldata_w[model] = [{axisname: np.full(np.count_nonzero(m), weights, dtype=np.float64)}]
            else:
                modeldata_w[model] = [{axisname: np.full_like(dp[axis], weights, dtype=np.float64)} for dp in datapoints_xy]

    for model, datapoints in modeldata.items():
        datapoints_w = modeldata_w[model]

        if len(datapoints_w) == 1 and len(datapoints) > 1:
            for datapoint in datapoints:
                datapoint[axisname] = datapoints_w[0][axisname].copy()
        elif len(datapoints_w) == len(datapoints):
            for i, datapoint in enumerate(datapoints):
                datapoint[axisname] = datapoints_w[i][axisname]
        else:
            raise ValueError(f'Size of weights data incompatible with size of {axis} data')

    if norm_weights:
        _norm_weights(modeldata, axisname)

    return modeldata

def _norm_weights(modeldata, axisname):
    logger.info(
        f'Normalising weights so that the sum of all weights is equal to 1.')
    for model, datapoints in modeldata.items():
        for datapoint in datapoints:
            weights = datapoint[axisname]
            sum = np.nansum(weights)
            if sum > 0: datapoint[axisname] = weights / sum

@utils.set_default_kwargs()
def _make_table(models, axis_names, kwargs=None):
    model_datapoints, axis_labels = simple.get_data(models, axis_names, **kwargs)
    pass

@set_default_kwargs(layout='constrained')
def create_rose_plot(ax=None, *, xscale=1, yscale=1,
                     segment=None, rres=None):
    """
    Create a plot with a [rose projection][simple.roseaxes.RoseAxes].

    The rose ax is a subclass of matplotlibs
    [polar axes](https://matplotlib.org/stable/api/projections/polar.html#matplotlib.projections.polar.PolarAxes).

    Args:
        ax (): A matplotlib axes object, or an object with a ``gca()`` method (e.g. ``plt``). If the
            axes does not have a rose projection, it will be destroyed and replaced by a new [RoseAxes][(]simple.roseaxes.RoseAxes].
        xscale (): The scale of the x axis.
        yscale (): The scale of the y axis.
        segment (): Which segment of the rose diagram to show. Options are ``N``, ``E``, ``S``, ``W``,
            ``NE``, ``SE``, ``SW``, ``NW`` and ``None``. If ``None`` the entire circle is shown.
        rres (): The resolution of lines drawn along the radius ``r``. The number of points in a line is calculated as
        ``r*rres+1`` (Min. 2).

    Returns:
        RoseAxes : The new rose ax.
    """
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': 'rose'}, layout='constrained')
    else:
        ax = get_axes(ax)
        if ax.name != 'rose':
            logger.warning(f'Wrong Axes projection for rose plot. Deleting axes and creating a new one.')
            subplot_dict = getattr(ax, '_SIMPLE_subplot_dict', None)
            subplot_dict_key = getattr(ax, '_SIMPLE_subplot_dict_key', None)
            fig = ax.get_figure()
            rows, cols, start, stop = ax.get_subplotspec().get_geometry()

            ax.remove()
            ax = fig.add_subplot(rows, cols, start + 1, projection='rose')
            if subplot_dict is not None:
                subplot_dict[subplot_dict_key] = ax
                ax._SIMPLE_subplot_dict = subplot_dict
                ax._SIMPLE_subplot_dict_key = subplot_dict_key

    ax.set_xyscale(xscale, yscale)

    if segment:
        ax.set_xysegment(segment)

    if rres:
        ax.set_rres(rres)

    return ax

@utils.set_default_kwargs()
def create_legend(ax, outside = False, outside_margin=0.01, kwargs=None):
    """
    Add a legend to a plot.

    Args:
        ax (): The working axes. Accepted values are any matplotlib Axes object or plt instance.
        outside (bool): If ``True`` the legend will be drawn just outside the upper left corner of the plot. This will
            overwrite any ``loc`` and ``bbox_to_anchor`` arguments in ``kwargs``.
        outside_margin (): Margin between the plot and the legend. Relative to the width of the plot.
        **kwargs (): Any valid argument for matplotlibs
            [``legend``](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html) function.
    """
    ax = get_axes(ax)

    if outside:
        kwargs['loc'] = 'upper left'
        kwargs['bbox_to_anchor'] = (1+outside_margin, 1)

    kwargs.pop_many(create_legend)
    ax.legend(**kwargs)

def update_axes(ax, kwargs, *, update_ax = True, update_fig = True):
    """
    Updates the axes and figure objects.

    Keywords beginning with ``ax_<name>``, ``xax_<name>``, ``yax_<name>`` and ``fig_<name>`` will be stripped
    from kwargs. These will then be used to call the ``set_<name>`` or ``<name>`` method of the axes, axis or
    figure object.

    If the value mapped to the above arguments is:
    - A boolean it is used to determine whether to call the method. The boolean itself will not be passed to
        the method. To pass a boolean to a method place it in a tuple, e.g. ``(True, )``.
    - A tuple then the contents of the tuple is unpacked as arguments for the method call.
    - A dictionary then the contents of the dictionary is unpacked as keyword arguments for the method call.
    - Any other value will be passed as the first argument to the method call.

    Additional keyword arguments can be passed to methods by mapping e.g. ``<ax|xax|yax|fig>_kw_<name>_<keyword>``
    kwargs to the value. These additional keyword arguments are only used if the
    ``<ax|xax|yax|fig>_<name>`` kwargs exists. Note however that they are always stripped from ``kwargs``.

    The figure will not be updated if ``ax`` is a subplot created by [simple.create_subplots][simple.plotting.create_subplots].
    """

    ax = get_axes(ax)
    axes_meth = kwargs.pop_many(prefix='ax')
    axes_kw = axes_meth.pop_many(prefix='kw')

    xaxes_meth = kwargs.pop_many(prefix='xax')
    xaxes_kw = xaxes_meth.pop_many(prefix='kw')

    yaxes_meth = kwargs.pop_many(prefix='yax')
    yaxes_kw = yaxes_meth.pop_many(prefix='kw')

    if update_ax:
        _update_fig_or_ax(ax, 'ax', axes_meth, axes_kw)
        if xaxes_meth:
            _update_fig_or_ax(ax.xaxis, 'xax', xaxes_meth, xaxes_kw)
        if yaxes_meth:
            _update_fig_or_ax(ax.yaxis, 'yax', yaxes_meth, yaxes_kw)

    # Dont update figure if subplot was created using create subplots.
    # Figure stuff should be done there instead
    update_figure(ax.get_figure(), kwargs, update_fig = False if hasattr(ax, '_SIMPLE_subplot_dict') else update_fig)

def update_figure(fig, kwargs, *, update_fig=True):
    """
    Updates the figure object.

    See [update_axes][simple.plotting.update_axes] for more information. **Note** that this function only removes
    keywords beginning with ``fig_`` from ``kwargs``.
    """
    figure_meth = kwargs.pop_many(prefix='fig')
    figure_kw = figure_meth.pop_many(prefix='kw')

    # Special cases
    if 'size' in figure_meth: figure_meth.setdefault('size_inches', figure_meth.pop('size'))

    if update_fig:
        _update_fig_or_ax(fig, 'fig', figure_meth, figure_kw)

def _update_fig_or_ax(obj, name, meth_kwargs, kw_kwargs):
    # Companion to update_axes, update_figure
    for var, arg in meth_kwargs.items():
        var_kwargs = kw_kwargs.pop_many(prefix=var)
        try:
            method = getattr(obj, f'set_{var}')
        except:
            try:
                method = getattr(obj, var)
            except:
                raise AttributeError(f'The {name} object has no method called ``set_{var}`` or ``{var}``')

        if arg is False:
            continue
        elif arg is True:
            arg = ()
        elif type(arg) is dict:
            var_kwargs.update(arg)
            arg = ()
        elif type(arg) is not tuple:
            arg = (arg,)

        method(*arg, **var_kwargs)


@utils.add_shortcut('abundance', default_attrname ='abundance', unit=None)
@utils.add_shortcut('stdnorm', default_attrname ='stdnorm.Ri', unit=None)
@utils.add_shortcut('intnorm', default_attrname='intnorm.eRi', unit=None)
@utils.set_default_kwargs(
    linestyle=False, color=True, marker=True,
    fixed_model_linestyle = None, fixed_model_color = None, fixed_model_marker = None,
    ax_kw_xlabel_fontsize=15,
    ax_kw_ylabel_fontsize=15,
    markersize=4,
    legend_outside=True,
    ax_tick_params=dict(axis='both', left=True, right=True, top=True),
    fig_size=(7,6.5),
    )
def plot(models, xkey, ykey, *,
         default_attrname=None, unit=None,
         where=None, mask = None, mask_na = True, ax = None,
         legend = None, update_ax = True, update_fig = True,
         hist=False, hist_size=0.3, hist_pad=0.05, kwargs=None):
    """
    Plot *xkey* against *ykey* for each model in *models*.

    This function retrieves data using [`get_data`][simple.get_data] and plots it using matplotlib. It supports
    optional filtering, masking, and per-model or per-dataset styling. Additional arguments can be passed using
    keyword prefixes to control axes, figure appearance, legends, and more.

    This function is split into two stages: [`plot_get_data`][simple.plotting.plot_get_data] and
    [`plot_draw`][simple.plotting.plot_draw], which can be used independently.

    Args:
        models (ModelCollection): A collection of models to plot. A subset can be selected using the *where* argument.
        xkey, ykey (str, int, or slice): Keys or indices used to retrieve the x and y data arrays. These may refer to
            array indices (relative to *default_attrname*) or full attribute paths.
            See [`get_data`][simple.get_data] for more.
        default_attrname (str): Name of the default attribute used when *xkey* or *ykey* is an index.
        unit (str or tuple): Desired unit(s) for the x and y axes. Use a tuple `(xunit, yunit)` for different units.
        where (str): Filter expression to select a subset of *models*. Any *kwargs* prefixed with `where_`
            are passed to the model filtering function.
            See [`ModelCollection.where`][simple.models.ModelCollection.where].
        mask (str, int, or slice): Optional mask to apply to the data. See the `get_mask` method on model instances.
        mask_na (bool): If True, masked values are replaced with `np.nan`. Only applies if *xkey* and *ykey* are
            float-based.
        ax (matplotlib.axes.Axes or None): The axes to plot on. If None, defaults to `plt.gca()`.
        legend (bool): Whether to add a legend. If None, a legend is shown if at least one datapoint has a label.
        update_ax, update_fig (bool): Whether to apply `ax_` and `fig_` keyword argument updates using
            [`update_axes`][simple.plotting.update_axes].
        hist (bool): Whether to show marginal histograms along the axes.
        hist_size (float): Relative size of the histogram axes.
        hist_pad (float): Padding between the main plot and histogram axes.
        kwargs (dict, optional): A dictionary of keyword arguments. This provides an explicit way to pass additional
            options and overrides any conflicting values passed via `**kwargs_`.
        **kwargs_ (): Additional keyword arguments passed directly to the function. These are merged into *kwargs* unless
            a key already exists in *kwargs*, in which case the value in *kwargs* takes precedence.
            kwargs (dict): Additional keyword arguments. These may include:
        kwargs (dict or KwargDict, optional): A dictionary of keyword arguments that may be modified internally.
            Any keys in *kwargs* that match parameters in the function's signature are automatically passed to those
            parameters. If a key exists in both *kwargs* and ***kwargs*, the value in *kwargs* takes precedence.
        **kwargs (): Additional keyword arguments passed directly to the function. These are merged into *kwargs*.
            Possible keyword arguments are:

            - Parameters passed to [`simple.get_data`][simple.get_data]
            - Matplotlib [`plot()`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html#matplotlib.axes.Axes.plot) arguments (e.g., `color`, `linestyle`, `marker`)
            - Prefix-based controls (e.g., `ax_xlabel`, `legend_outside`, `fig_size`)

    Iterable plot arguments:
        The following matplotlib arguments support iterable or preset behaviour:

        - `color`: Can be a list of colours, `True` for defaults, or `False` to use black.
        - `linestyle`: Can be a list of styles, `True` for defaults, or `False` to disable lines.
        - `marker`: Can be a list of markers, `True` for defaults, or `False` to disable markers.

        These are assigned per-model or per-dataset depending on the context:

        - If multiple models are plotted, all datasets for a model share the same *color*.
        - If one model has multiple datasets, each gets a different *color*.
        - *linestyle* and *marker* behave similarly, controlled by:
            - `fixed_model_color`
            - `fixed_model_linestyle`
            - `fixed_model_marker`

    Axis and data labels:
        Axis labels are automatically inferred based on shared and unique elements in the data. You can override them
        using `ax_xlabel` and `ax_ylabel` in *kwargs*. Datapoint labels can be overridden with a list of labels
        (one per line in the legend).

    Shortcuts and default values:
        This function includes shortcut variants with predefined argument:

        - `plot.intnorm`: sets `default_attrname="intnorm.eRi"` and `unit=None`
        - `plot.stdnorm`: sets `default_attrname="stdnorm.Ri"` and `unit=None`
        - `plot.abundance`: sets `default_attrname="abundance"` and `unit=None`

        Default argument values can also be updated through `plot.update_kwargs()`. Values defined in the function
        signature are used only if not overridden there. You can retrieve the default arguments using `plot.kwargs`

    Returns:
        matplotlib.axes.Axes: The axes object used for plotting.
    """

    modeldata, axis_labels = plot_get_data(models, xkey, ykey,
                                           default_attrname=default_attrname, unit=unit,
                                           where=where, mask=mask, mask_na=mask_na, hist=hist,
                                           kwargs=kwargs)
    return plot_draw(modeldata, axis_labels, ax=ax, legend=legend,
                     update_ax=update_ax, update_fig=update_fig,
                     hist=hist, hist_size=hist_size, hist_pad=hist_pad,
                     kwargs=kwargs)

@utils.set_default_kwargs(inherits_=plot)
def plot_get_data(models, xkey, ykey, *, default_attrname=None, unit=None,
                  where=None, mask = None, mask_na = True,
                  kwargs=None):
    """
    Retrieve model data and axis labels for plotting.

    This function performs the data preparation step of [`plot`][simple.plotting.plot]. See the documentation of this
    function for a description of the arguments.

    Returns:
        tuple:
            - modeldata (dict): Structured data for plotting.
            - axis_labels (dict): Suggested axis labels.
    """
    modeldata, axis_labels = get_data(models, {'x': xkey, 'y': ykey},
                                      where=where,
                                      default_attrname=default_attrname, unit=unit,
                                      mask=mask, mask_na=mask_na,
                                      kwargs=kwargs)

    func_weights = kwargs.pop('_SIMPLE_add_weights', add_weights)
    weights_kwargs = kwargs.pop_many('weights, sum_weights, norm_weights', prefix='weights')
    if kwargs.get('hist', False) or kwargs.get('xhist', False) or kwargs.get('yhist', False):
        func_weights(modeldata, 'x', mask=mask, mask_na=mask_na, **weights_kwargs)

    return modeldata, axis_labels


@utils.set_default_kwargs(inherits_=plot)
def plot_draw(modeldata, axis_labels, *, ax=None, legend=None,
              update_ax=True, update_fig=True,
              hist=False, hist_size=0.3, hist_pad=0.05, kwargs=None):
    """
    Render the plot using matplotlib.

    This function handles the styling, axes setup, and drawing logic of [`plot`][simple.plotting.plot]. See the
    documentation of this function for a description of the arguments.

    Returns:
        matplotlib Axes: The axes on which the data was plotted.
    """
    ax = get_axes(ax)  # We are working on the axes object proper

    # Removes any inherited kwargs for the get_data step
    kwargs.pop_many([plot_get_data, get_data, kwargs.get('_SIMPLE_add_weights', add_weights)])

    # Get the linestyle, color and marker for each thing to be plotted.
    linestyles, colors, markers = parse_lscm(**kwargs.pop_many(parse_lscm))
    fixed_model_linestyle = kwargs.pop('fixed_model_linestyle', None)
    fixed_model_color = kwargs.pop('fixed_model_color', None)
    fixed_model_marker = kwargs.pop('fixed_model_marker', None)

    # Extract the legend arguments
    legend_kwargs = kwargs.pop_many(prefix='legend')

    # Extract the hist arguments
    xhist = kwargs.pop('xhist', hist)
    yhist = kwargs.pop('yhist', hist)

    hist_kwargs = kwargs.pop_many(prefix='hist',
                                  size = hist_size, pad = hist_pad,
                                  legend=False, fig_size=False,
                                  linestyle=linestyles if linestyles != [''] else True, color=colors,
                                  fixed_model_linestyle=fixed_model_linestyle, fixed_model_color=fixed_model_color, )
    xhist_kwargs = kwargs.pop_many(prefix='xhist', ax_xlabel=None, **hist_kwargs)
    yhist_kwargs = kwargs.pop_many(prefix='yhist', ax_ylabel=None, **hist_kwargs)

    # Update the axes/figure
    kwargs.setdefault('ax_xlabel', axis_labels['x'])
    kwargs.setdefault('ax_ylabel', axis_labels['y'])
    update_axes(ax, kwargs, update_ax=update_ax, update_fig=update_fig)

    kwargs.pop_many(prefix='_')

    mfc = kwargs.pop('markerfacecolor', None)

    for mi, (model, datapoints) in enumerate(modeldata.items()):
        for di, datapoint in enumerate(datapoints):
            if fixed_model_linestyle or (fixed_model_linestyle is None and len(datapoints) == 1):
                ls = linestyles[mi]
            else:
                ls = linestyles[di]
            if fixed_model_color is True or (fixed_model_color is None and len(modeldata) > 1):
                c = colors[mi]
            else:
                c = colors[di]
            if fixed_model_marker is True or (fixed_model_marker is None and len(datapoints) == 1):
                m = markers[mi]
            else:
                m = markers[di]

            ax.plot(datapoint['x'], datapoint['y'],
                    label = datapoint.get('label', None),
                    color=c, ls=ls, marker=m,
                    markerfacecolor=mfc or c,
                    **kwargs)

    # Create the hist plots
    if xhist:
        pad, size = xhist_kwargs.pop('pad'), xhist_kwargs.pop('size')
        if xhist_kwargs.get('ax', None) is None:
            ax_histx = getattr(ax, '_SIMPLE_ax_histx', None)
            if ax_histx is None:
                ax_histx = ax.inset_axes([0, 1 + pad, 1, size], sharex=ax)
                setattr(ax, '_SIMPLE_ax_histx', ax_histx)
            xhist_kwargs['ax'] = ax_histx

        xhist_kwargs.setdefault('range', ax.get_xlim())
        xhist_kwargs.setdefault('xax_visible', (False,))
        hist_draw1d(modeldata, axis_labels, 'x', **xhist_kwargs)

    if yhist:
        pad, size = yhist_kwargs.pop('pad'), yhist_kwargs.pop('size')
        if yhist_kwargs.get('ax', None) is None:
            legend_kwargs['outside_margin'] = (legend_kwargs.get('outside_margin', 0.01)
                                               + pad + size)
            ax_histy = getattr(ax, '_SIMPLE_ax_histy', None)
            if ax_histy is None:
                ax_histy = ax.inset_axes([1 + pad, 0, size, 1], sharey=ax)
                setattr(ax, '_SIMPLE_ax_histy', ax_histy)
            yhist_kwargs['ax'] = ax_histy

        yhist_kwargs.setdefault('yax_visible', (False,))
        yhist_kwargs.setdefault('range', ax.get_ylim())
        hist_draw1d(modeldata, axis_labels, 'y', **yhist_kwargs)

    if legend:
        create_legend(ax, **legend_kwargs)

    return ax


@utils.add_shortcut('abundance', default_attrname ='abundance', unit=None)
@utils.add_shortcut('stdnorm', default_attrname ='stdnorm.Ri', unit=None)
@utils.add_shortcut('intnorm', default_attrname='intnorm.eRi', unit=None)
@utils.set_default_kwargs(
    linestyle=True, color=True,
    fixed_model_linestyle = None, fixed_model_color = None,
    ax_kw_xlabel_fontsize=15,
    ax_kw_ylabel_fontsize=15,
    legend_outside=True,
    ax_tick_params=dict(axis='both', left=True, right=True, top=True),
    fig_size=(7,6.5),
    )
def hist(models, xkey=None, ykey=None, weights=1, r=None, *,
         sum_weights=True, norm_weights=True,
         bins = True, fill=None, rescale=False,
         default_attrname=None, unit=None,
         weights_default_attrname = None, weights_unit=None, weights_default_value=0,
         where=None, mask=None, ax=None,
         legend=None, update_ax=True, update_fig=True,
         kwargs=None):
    """
    Make a traditional histogram of *xkey* or *ykey*, or a circular histogram for the slope of *ykey*/*xkey*
    on *axis*, for each model in *models*.

    This function retrieves data using [`get_data`][simple.get_data] and plots it using matplotlib. It supports
    optional filtering, masking, and per-model or per-dataset styling. Additional arguments can be passed using
    keyword prefixes to control axes, figure appearance, legends, and more.

    This function is split into two stages: [`hist_get_data`][simple.plotting.hist_get_data] and
    [`hist_draw`][simple.plotting.hist_draw], which can be used independently.

    Args:
        models (ModelCollection): A collection of models to plot. A subset can be selected using the *where* argument.
        xkey, ykey (str, int, or slice): Keys or indices used to retrieve the x and y data arrays. These may refer to
            array indices (relative to *default_attrname*) or full attribute paths. See [`get_data`][simple.get_data]
            for more information. If only *xkey* or *ykey* is specified then a traditional histogram is drawn. If
            both are specified then a circular histogram of the slopes is drawn.
        default_attrname (str): Name of the default attribute used when *xkey* or *ykey* is an index.
        unit (str or tuple): Desired unit(s) for the x and y axes. Use a tuple `(xunit, yunit)` for different units.
        where (str): Filter expression to select a subset of *models*. Any *kwargs* prefixed with `where_`
            are passed to the model filtering function.
            See [`ModelCollection.where`][simple.models.ModelCollection.where].
        mask (str, int, or slice): Optional mask to apply to the data. See the `get_mask` method on model instances.
        mask_na (bool): If True, masked values are replaced with `np.nan`. Only applies if *xkey* and *ykey* are
            float-based.
        ax (matplotlib.axes.Axes or None): The axes to plot on. If None, defaults to `plt.gca()`.
        legend (bool): Whether to add a legend. If None, a legend is shown if at least one datapoint has a label.
        update_ax, update_fig (bool): Whether to apply `ax_` and `fig_` keyword argument updates using
            [`update_axes`][simple.plotting.update_axes].
        hist (bool): Whether to show marginal histograms along the axes.
        hist_size (float): Relative size of the histogram axes.
        hist_pad (float): Padding between the main plot and histogram axes.
        kwargs (dict, optional): A dictionary of keyword arguments. This provides an explicit way to pass additional
            options and overrides any conflicting values passed via `**kwargs_`.
        **kwargs_ (): Additional keyword arguments passed directly to the function. These are merged into *kwargs* unless
            a key already exists in *kwargs*, in which case the value in *kwargs* takes precedence.
            kwargs (dict): Additional keyword arguments. These may include:

            - Parameters passed to [`simple.get_data`][simple.get_data]
            - Matplotlib [`stairs()`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.stairs.html) arguments (e.g., `color`, `linestyle`)
            - Prefix-based controls (e.g., `ax_xlabel`, `legend_outside`, `fig_size`)

    Iterable plot arguments:
        The following matplotlib arguments support iterable or preset behaviour:

        - `color`: Can be a list of colours, `True` for defaults, or `False` to use black.
        - `linestyle`: Can be a list of styles, `True` for defaults, or `False` to disable lines.

        These are assigned per-model or per-dataset depending on the context:

        - If multiple models are plotted, all datasets for a model share the same *color*.
        - If one model has multiple datasets, each gets a different *color*.
        - *linestyle* and *marker* behave similarly, controlled by:
            - `fixed_model_color`
            - `fixed_model_linestyle`

    Axis and data labels:
        Axis labels are automatically inferred based on shared and unique elements in the data. You can override them
        using `ax_xlabel` and `ax_ylabel` in *kwargs*. Datapoint labels can be overridden with a list of labels
        (one per line in the legend).

    Shortcuts and default values:
        This function includes shortcut variants with predefined default values:

        - `plot.intnorm`: sets `default_attrname="intnorm.eRi"` and `unit=None`
        - `plot.stdnorm`: sets `default_attrname="stdnorm.Ri"` and `unit=None`
        - `plot.abundance`: sets `default_attrname="abundance"` and `unit=None`

        Default argument values can also be updated through `plot.update_kwargs()`. Values defined in the function
        signature are used only if not overridden there. You can retrieve the default arguments using `plot.kwargs`

    Returns:
        matplotlib.axes.Axes: The axes object used for plotting.
    """

    modeldata, axis_labels, axis = hist_get_data(models, xkey, ykey, weights,
                                                 sum_weights=sum_weights, norm_weights=norm_weights,
                                                 default_attrname=default_attrname, unit=unit,
                                                 weights_default_attrname=weights_default_attrname,
                                                 weights_unit=weights_unit, weights_default_value=weights_default_value,
                                                 where=where, mask=mask, kwargs=kwargs)

    if axis == 'xy':
        # 2d - circular histogram
        return hist_draw2d(modeldata, axis_labels, r, bins=bins, fill=fill, rescale=rescale,
                             legend=legend, update_ax=update_ax, update_fig=update_fig,
                             ax=ax, kwargs=kwargs)
    else:
        # 1d - traditional histogram
        return hist_draw1d(modeldata, axis_labels, axis, bins=bins, fill=fill, rescale=rescale,
                             legend=legend, update_ax=update_ax, update_fig=update_fig,
                             ax=ax, kwargs=kwargs)


@utils.add_shortcut('abundance', default_attrname ='abundance', unit=None)
@utils.add_shortcut('stdnorm', default_attrname ='stdnorm.Ri', unit=None)
@utils.add_shortcut('intnorm', default_attrname='intnorm.eRi', unit=None)
@utils.set_default_kwargs(inherits_=hist)
def hist_get_data(models, xkey, ykey, weights=1, *,
                  sum_weights=True, norm_weights=True,
                  default_attrname=None, unit=None,
                  weights_default_attrname=None, weights_unit=None, weights_default_value=0,
                  where=None, mask=None,
                  kwargs=None):

    axis_names = {}
    axis =''
    if xkey is not None:
        axis_names['x'] = xkey
        axis+= 'x'
    if ykey is not None:
        axis_names['y'] = ykey
        axis+= 'y'
    if len(axis_names) == 0:
        raise ValueError('At least one axis must be specified.')

    modeldata, axis_labels = get_data(models, axis_names,
                                      where=where,
                                      default_attrname=default_attrname, unit=unit,
                                      mask=mask, mask_na=False,
                                      kwargs=None)

    func_weights = kwargs.pop('_SIMPLE_add_weights', add_weights)
    func_weights(modeldata, axis[0],
                 weights=weights, sum_weights=sum_weights, norm_weights=norm_weights,
                 default_attrname=weights_default_attrname, unit=weights_unit,
                 default_value=weights_default_value, mask=mask, mask_na=False)

    return modeldata, axis_labels, axis


@utils.set_default_kwargs(inherits_=hist)
def hist_draw1d(modeldata, axis_labels, axis, *, bins=20, fill=None, rescale=False,
              ax=None, legend=None, update_ax=True, update_fig=True,
              kwargs=None):
    """
    Render the standard histogram using matplotlib.

    This function handles the styling, axes setup, and drawing logic of [`hist`][simple.plotting.hist]. See the
    documentation of this function for a description of the arguments.

    Returns:
        matplotlib Axes: The axes on which the data was plotted.
    """
    ax = get_axes(ax)  # We are working on the axes object proper

    if bins is True: bins = hist_draw1d.kwargs.get('bins', 20)

    # Removes any inherited kwargs for the get_data step
    kwargs.pop_many(['r', hist_get_data, get_data, kwargs.get('_SIMPLE_add_weights', add_weights)])

    # Get the linestyle, color and marker for each thing to be plotted.
    linestyles, colors, markers = parse_lscm(**kwargs.pop_many(parse_lscm))
    fixed_model_linestyle = kwargs.pop('fixed_model_linestyle', None)
    fixed_model_color = kwargs.pop('fixed_model_color', None)
    fixed_model_marker = kwargs.pop('fixed_model_marker', None)

    legend_kwargs = kwargs.pop_many(prefix='legend')

    if fill is None:
        if len(modeldata) > 1 or (len(modeldata) > 0 and len(next(iter(modeldata.values()))) > 1):
            fill = False
        else:
            fill = True

    if axis == 'y':
        kwargs.setdefault('orientation', 'horizontal')
        kwargs.setdefault('ax_ylabel', axis_labels['y'])
    else:
        kwargs.setdefault('orientation', 'vertical')
        kwargs.setdefault('ax_xlabel', axis_labels['x'])

    update_axes(ax, kwargs, update_ax=update_ax, update_fig=update_fig)
    simple_kwargs = kwargs.pop_many(prefix='_')

    if rescale:
        logger.info('Normalising all bin values to the largest bin value.')

    histogram_kwargs = kwargs.pop_many('range, density')

    has_labels = False
    if bins:
        for mi, (model, datapoints) in enumerate(modeldata.items()):
            for di, datapoint in enumerate(datapoints):
                if fixed_model_linestyle or (fixed_model_linestyle is None and len(datapoints) == 1):
                    ls = linestyles[mi]
                else:
                    ls = linestyles[di]
                if fixed_model_color is True or (fixed_model_color is None and len(modeldata) > 1):
                    c = colors[mi]
                else:
                    c = colors[di]

                if not has_labels and datapoint.get('label', None):
                    has_labels = True

                values, edges = np.histogram(datapoint[axis], bins=bins, weights=datapoint['w'], **histogram_kwargs)
                if rescale:
                    values = values/np.max(values)

                ax.stairs(values, edges,
                        label=datapoint.get('label', None),
                        color=c, ls=ls, fill=fill,
                        **kwargs)

    if legend or (legend is None and has_labels):
        create_legend(ax, **legend_kwargs)

    return ax

@utils.set_default_kwargs(inherits_=hist, ax_kw_ylabel_labelpad=20)
def hist_draw2d(modeldata, axis_labels, r=None, *, bins=72, fill=None, rescale=False,
                   ax = None, legend=None, update_ax = True, update_fig = True,
                   kwargs=None):
    """
    Render the circular histogram using on a [Rose Axes][simple.roseaxes.RoseAxes].

    This function handles the styling, axes setup, and drawing logic of [`hist`][simple.plotting.hist]. See the
    documentation of this function for a description of the arguments.

    Returns:
        matplotlib Axes: The axes on which the data was plotted.
    """
    ax = get_axes(ax)

    if bins is True: bins = hist_draw2d.kwargs.get('bins', 20)

    # Removes any inherited kwargs for the get_data step
    kwargs.pop_many(['axis', hist_get_data, get_data, kwargs.get('_SIMPLE_add_weights', add_weights)])

    # Get the linestyle, color and marker for each thing to be plotted.
    linestyles, colors, markers = parse_lscm(**kwargs.pop_many(parse_lscm))
    fixed_model_linestyle = kwargs.pop('fixed_model_linestyle', None)
    fixed_model_color = kwargs.pop('fixed_model_color', None)
    fixed_model_marker = kwargs.pop('fixed_model_marker', None)

    rose_kwargs = kwargs.pop_many(prefix='rose')
    if ax.name != 'rose':
        ax = create_rose_plot(ax, **rose_kwargs)

    if fill is None:
        if len(modeldata) > 1 or (len(modeldata) > 0 and len(next(iter(modeldata.values()))) > 1):
            fill = False
        else:
            fill = True

    if r is None and fill is False:
        r = ax._last_hist_r + 1

    if not isinstance(r, Sequence):
        r = [r for i in range(len(modeldata))]
    elif len(r) != len(modeldata):
        raise ValueError(f'Size of r must match size of models ({len(r)}!={len(modeldata)})')

    kwargs.setdefault('ax_xlabel', axis_labels['x'])
    kwargs.setdefault('ax_ylabel', axis_labels['y'])

    legend_kwargs = kwargs.pop_many(prefix='legend')
    update_axes(ax, kwargs, update_ax=update_ax, update_fig=update_fig)
    simple_kwargs = kwargs.pop_many(prefix='_SIMPLE')

    if rescale:
        logger.info('Normalising all bin values to the largest bin value.')

    has_labels = False
    if bins:
        for mi, (model, datapoints) in enumerate(modeldata.items()):
            for di, datapoint in enumerate(datapoints):
                if fixed_model_linestyle or (fixed_model_linestyle is None and len(datapoints) == 1):
                    ls = linestyles[mi]
                else:
                    ls = linestyles[di]
                if fixed_model_color is True or (fixed_model_color is None and len(modeldata) > 1):
                    c = colors[mi]
                else:
                    c = colors[di]

                if not has_labels and datapoint.get('label', None):
                    has_labels = True

                ax.mhist((datapoint['x'], datapoint['y']), r=r[mi], weights=datapoint['w'],
                         label=datapoint.get('label', None), bins=bins, fill=fill,
                         color=c, linestyle=ls, rescale=rescale, **kwargs)

    if legend or (legend is None and has_labels):
        if ax._colorbar is not None:
            legend_kwargs.setdefault('outside_margin', 0.35)
        create_legend(ax, **legend_kwargs)

    return ax

@utils.add_shortcut('abundance', default_attrname ='abundance', unit=None)
@utils.add_shortcut('stdnorm', default_attrname ='stdnorm.Ri', unit=None)
@utils.add_shortcut('intnorm', default_attrname='intnorm.eRi', unit=None)
@utils.set_default_kwargs(
    linestyle=True, color=True,
    fixed_model_linestyle = None, fixed_model_color = None,
    ax_kw_xlabel_fontsize=15,
    ax_kw_ylabel_fontsize=15,
    markersize=4,
    legend_outside=True,
    ax_tick_params=dict(axis='both', left=True, right=True, top=True),
    fig_size=(7,6.5),
    arrow_linewidth=0, arrow_length_includes_head=True, arrow_head_width=0.05,
    arrow_zorder=3
    )
def slope(models, xkey, ykey, xycoord=(0, 0), *,
          arrow=True, arrow_position=0.9,
          default_attrname=None, unit=None,
          where=None, mask = None, mask_na = True, ax = None,
          legend = None, update_ax = True, update_fig = True,
          kwargs=None):

    """
    Plot the slope of *ykey*/*xkey* for each model in *models*.

    This function retrieves data using [`get_data`][simple.get_data] and plots it using matplotlib. It supports
    optional filtering, masking, and per-model or per-dataset styling. Additional arguments can be passed using
    keyword prefixes to control axes, figure appearance, legends, and more.

    This function is split into two stages: [`slope_get_data`][simple.plotting.slope_get_data] and
    [`pslope_draw`][simple.plotting.slope_draw], which can be used independently.

    Args:
        models (ModelCollection): A collection of models to plot. A subset can be selected using the *where* argument.
        xkey, ykey (str, int, or slice): Keys or indices used to retrieve the x and y data arrays. These may refer to
            array indices (relative to *default_attrname*) or full attribute paths. See [`get_data`][simple.get_data]
            for more information.
        xycoord (tuple): Coordinates to a point the slope passes through. Defaults to `(0, 0)`.
        arrow (bool): Whether to draw arrows indicating the direction of the endmember given by the x and y coordinates.
        arrow_position (float): Relative position of the arrow on the line. Defaults to 0.9.
        default_attrname (str): Name of the default attribute used when *xkey* or *ykey* is an index.
        unit (str or tuple): Desired unit(s) for the x and y axes. Use a tuple `(xunit, yunit)` for different units.
        where (str): Filter expression to select a subset of *models*. Any *kwargs* prefixed with `where_`
            are passed to the model filtering function.
            See [`ModelCollection.where`][simple.models.ModelCollection.where].
        mask (str, int, or slice): Optional mask to apply to the data. See the `get_mask` method on model instances.
        mask_na (bool): If True, masked values are replaced with `np.nan`. Only applies if *xkey* and *ykey* are
            float-based.
        ax (matplotlib.axes.Axes or None): The axes to plot on. If None, defaults to `plt.gca()`.
        legend (bool): Whether to add a legend. If None, a legend is shown if at least one datapoint has a label.
        update_ax, update_fig (bool): Whether to apply `ax_` and `fig_` keyword argument updates using
            [`update_axes`][simple.plotting.update_axes].
        kwargs (dict, optional): A dictionary of keyword arguments. This provides an explicit way to pass additional
            options and overrides any conflicting values passed via `**kwargs_`.
        **kwargs_ (): Additional keyword arguments passed directly to the function. These are merged into *kwargs* unless
            a key already exists in *kwargs*, in which case the value in *kwargs* takes precedence.
            kwargs (dict): Additional keyword arguments. These may include:

            - Parameters passed to [`simple.get_data`][simple.get_data]
            - Matplotlib [`plot()`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html#matplotlib.axes.Axes.plot) arguments (e.g., `color`, `linestyle`, `marker`)
            - Prefix-based controls (e.g., `ax_xlabel`, `legend_outside`, `fig_size`)

    Iterable plot arguments:
        The following matplotlib arguments support iterable or preset behaviour:

        - `color`: Can be a list of colours, `True` for defaults, or `False` to use black.
        - `linestyle`: Can be a list of styles, `True` for defaults, or `False` to disable lines.
        - `marker`: Can be a list of markers, `True` for defaults, or `False` to disable markers.

        These are assigned per-model or per-dataset depending on the context:

        - If multiple models are plotted, all datasets for a model share the same *color*.
        - If one model has multiple datasets, each gets a different *color*.
        - *linestyle* and *marker* behave similarly, controlled by:
            - `fixed_model_color`
            - `fixed_model_linestyle`
            - `fixed_model_marker`

    Axis and data labels:
        Axis labels are automatically inferred based on shared and unique elements in the data. You can override them
        using `ax_xlabel` and `ax_ylabel` in *kwargs*. Datapoint labels can be overridden with a list of labels
        (one per line in the legend).

    Shortcuts and default values:
        This function includes shortcut variants with predefined argument:

        - `slope.intnorm`: sets `default_attrname="intnorm.eRi"` and `unit=None`
        - `slope.stdnorm`: sets `default_attrname="stdnorm.Ri"` and `unit=None`
        - `slope.abundance`: sets `default_attrname="abundance"` and `unit=None`

        Default argument values can also be updated through `slope.update_kwargs()`. Values defined in the function
        signature are used only if not overridden there. You can retrieve the default arguments using `plot.kwargs`

    Returns:
        matplotlib.axes.Axes: The axes object used for plotting.
    """

    modeldata, axis_labels = slope_get_data(models, xkey, ykey,
                                                default_attrname=default_attrname, unit=unit,
                                                where=where, mask=mask, mask_na=mask_na,
                                                kwargs=kwargs)
    return slope_draw(modeldata, axis_labels, xycoord=xycoord,
                   arrow=arrow, arrow_position=arrow_position,
                   ax=ax,
                   legend=legend, update_ax=update_ax, update_fig=update_fig,
                   kwargs=kwargs)

@utils.set_default_kwargs(inherits_=slope)
def slope_get_data(models, xkey, ykey, *,
                   default_attrname=None, unit=None,
                   where=None, mask=None, mask_na=True,
                   kwargs=None):
    modeldata, axis_labels = get_data(models, {'x': xkey, 'y': ykey},
                                      where=where,
                                      default_attrname=default_attrname, unit=unit,
                                      mask=mask, mask_na=mask_na,
                                      kwargs=kwargs)
    return modeldata, axis_labels

@utils.set_default_kwargs(inherits_=slope)
def slope_draw(modeldata, axis_labels, xycoord=(0,0), *,
             arrow=True, arrow_position=0.9,
             ax = None,
             legend=None, update_ax=True, update_fig=True,
             kwargs=None):
    ax = get_axes(ax)  # We are working on the axes object proper

    kwargs.pop_many([plot_get_data, get_data])

    # Get the linestyle, color and marker for each thing to be plotted.
    linestyles, colors, markers = parse_lscm(**kwargs.pop_many(parse_lscm))
    fixed_model_linestyle = kwargs.pop('fixed_model_linestyle', None)
    fixed_model_color = kwargs.pop('fixed_model_color', None)
    fixed_model_marker = kwargs.pop('fixed_model_marker', None)

    legend_kwargs = kwargs.pop_many(prefix='legend')
    arrow_kwargs = kwargs.pop_many(prefix='arrow')

    kwargs.setdefault('ax_xlabel', axis_labels['x'])
    kwargs.setdefault('ax_ylabel', axis_labels['y'])
    update_axes(ax, kwargs, update_ax=update_ax, update_fig=update_fig)
    simple_kwargs = kwargs.pop_many(prefix='_SIMPLE')

    has_labels = False
    for mi, (model, datapoints) in enumerate(modeldata.items()):
        for di, datapoint in enumerate(datapoints):
            if fixed_model_linestyle or (fixed_model_linestyle is None and len(datapoints) == 1):
                ls = linestyles[mi]
            else:
                ls = linestyles[di]
            if fixed_model_color is True or (fixed_model_color is None and len(modeldata) > 1):
                c = colors[mi]
            else:
                c = colors[di]

            if not has_labels and datapoint.get('label', None):
                has_labels = True

            label = datapoint.get('label', None)
            for i in range(len(datapoint['x'])):
                x, y = datapoint['x'][i], datapoint['y'][i]
                slope = y / x
                ax.axline(xycoord, slope=slope, label=label,
                          color = c, ls=ls, **kwargs)
                label = None

                if arrow:
                    if np.abs(slope) > 1:
                        y_arrow = np.array([arrow_position, arrow_position + 0.01]) * (-1 if y < 0 else 1)
                        x_arrow = 1 / slope * y_arrow
                    else:
                        x_arrow = np.array([arrow_position, arrow_position + 0.01]) * (-1 if x < 0 else 1)
                        y_arrow = slope * x_arrow

                    ax.arrow(x_arrow[0], y_arrow[1], x_arrow[1] - x_arrow[0], y_arrow[1] - y_arrow[0],
                            facecolor=c, **arrow_kwargs)

    if legend or (legend is None and has_labels):
        create_legend(ax, **legend_kwargs)

    return ax

