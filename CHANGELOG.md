# Release Notes

## [Latest] 

Added the LGPL-3.0+ licence

Updated GitHub workflows.

Updated Example notebooks.

### Model and ModelCollection implementation
Redid how data is meant to be saved to the database file is handled. The type of the data
is now preserved. I.e. a list will always be a list etc. The types that can be saved are 
``int``, ``float``, ``string``, ``list``, ``tuple``, and ``ndarray``. 

Updated how models and references are stored in the model collection. They are now stored in lists instead
of dictionaries. This change is only internal and normal user experience remains the same.

If a model contains a ``refid_<name>`` hdf5 attribute when created it will automatically create
a normal attribute called ``ref_<name>`` that points towards that model object. An error raised
if a reference models with that name does not exist.

Models are not tied to a model collection. One model can belong to several different model collections.
Adding a model to a collection will also add any reference models that are linked through
normal attributes called ``ref_<name>``.

Updated how ``get_mask`` works. Any attribute associated with the Model can now be accessed using
the ``.`` method, the same as for ``where``.

The ``where`` method now return a model collection containing the original model objects rather than copies.

Renamed ``ModelTemplate`` to ``ModelBase``.

### CCSNe Models
Added the *masscoord_mass* attribute to the mandatory attributes of the CCSNe models. 

Added a ``zone`` attribute to CCSNe models with an onion shell structure. This is a ndarray with the name
of the shell for each mass coordinate. Removed the ability to specify individual shells 
in ``get_mask``. Instead use ``.zone == <name>`` instead. 

### Plotting
Added a new plot type - A traditional histogram. The new plot type can be accessed using 
the ``hist`` function. Additionally, it is now possible to automatically add a histogram to
of the data show in ``plot`` using the ``hist``, ``yhist`` and ``xhist`` keywords. This replaces
the old ``hist_host``/``mhist`` functions to plot circular histograms.

``get_data`` now return the datapoints mapped to the Model object rather than the model name. 

Added ``add_weights`` and ``add_weights_ccsne`` functions that will calculate and add the weight
for each datapoint in a *data* returned from ``get_data``.

Split the main plotting functions into two steps. <func>_get_data and <func>_draw to allow easier customisation 
of plots.

Added test based on Tutorial 3. Does not verify output only that the commands run without error.
Added check that the tutorial notebooks run without error.

### Utils
Updated how the default kwargs works. You can now pass the whole kwargs dictionary to another function so that 
removed/added items are synced between the functions. 

Shortcuts can now be "inherited".