# API documentation

These functions are specifically for dealing with CCSNe data and are 
avaliable through through ``simple.ccsne``

## Models 

::: simple.ccsne.CCSNe
    options:
        inherited_members: True

## Plotting

Function specifically for plotting CCSNe data.

:::simple.ccsne.plot_abundance

:::simple.ccsne.plot_intnorm

:::simple.ccsne.plot_simplenorm

## Import model data

The following functions are used to load and process the raw data from different CCSNe models. These all
return a ``dict`` containing mapping the name of the model to another dictionary containing 
the attributes of that model. 

:::simple.ccsne.load_Ri18

:::simple.ccsne.load_Pi16

:::simple.ccsne.load_La22

:::simple.ccsne.load_Si18

:::simple.ccsne.load_Ra02

:::simple.ccsne.load_LC18

:::simple.ccsne.calc_default_onion_structure