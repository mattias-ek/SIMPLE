import numpy as np
import scipy.optimize

import simple.utils as utils
from scipy.optimize import fsolve

import logging

logger = logging.getLogger('SIMPLE.norm')

def intnorm_linear(abu_up, abu_down, abu_norm,
                   mass_up, mass_down, mass_norm,
                   solar_up, solar_down, solar_norm,
                   mass_coef = 'better'):

    rho_ij = ((abu_up / abu_down) / (solar_up / solar_down)) - 1
    rho_kj = ((abu_norm/abu_down) / (solar_norm/solar_down)) - 1

    if mass_coef.lower() == 'better':
        mass_diff = (np.log(mass_up) - np.log(mass_down)) / (np.log(mass_norm) - np.log(mass_down))
    elif mass_coef.lower() == 'simplified':
        mass_diff = (mass_up - mass_down) / (mass_norm - mass_down)

    return rho_ij - rho_kj * mass_diff


def intnorm_largest_offset(abu_up, abu_down, abu_norm,
                           mass_up, mass_down, mass_norm,
                           solar_up, solar_down, solar_norm,
                           largest_offset = 1, min_dilution_factor=0.1, max_iterations=100,
                           largest_offset_rtol = 1E-4):
    """

    Args:
        abu_up ():
        abu_down ():
        abu_norm ():
        mass_up ():
        mass_down ():
        mass_norm ():
        solar_up ():
        solar_down ():
        solar_norm ():
        largest_offset ():
        min_dilution_factor (): If the largest offset at this dilution factor
        max_iterations ():
        largest_offset_rtol ():

    Returns:

    """

    # Make sure everything is at least 2d
    abu_up = np.atleast_2d(abu_up)
    abu_down = np.atleast_2d(abu_down)
    abu_norm = np.atleast_2d(abu_norm)
    mass_up = np.atleast_2d(mass_up)
    mass_down = np.atleast_2d(mass_down)
    mass_norm = np.atleast_2d(mass_norm)
    solar_up = np.atleast_2d(solar_up)
    solar_down = np.atleast_2d(solar_down)
    solar_norm = np.atleast_2d(solar_norm)

    negQ = ((np.log(mass_up/mass_down)/
                     np.log(mass_norm/mass_down))) * -1 # Needs to be negative later

    R_solar_ij = solar_up / solar_down
    R_solar_kj = solar_norm / solar_down


    # Has to begin at largest offset or it might accidentally ignore rows
    dilution_factor = np.full((abu_up.shape[0], 1), min_dilution_factor, dtype=np.float64)

    first_only = True
    logger.info(f'Internally normalising {abu_up.shape[0]} rows using the largest offset method.')
    for i in range (max_iterations):
        smp_up = solar_up + (abu_up / dilution_factor)
        smp_down = solar_down + (abu_down / dilution_factor)
        smp_norm = solar_norm + (abu_norm / dilution_factor)

        r_smp_ij = smp_up / smp_down
        r_smp_kj = smp_norm / smp_down

        # Equation 6 in Lugaro et al 2023
        eR_smp_ij = (((r_smp_ij / R_solar_ij) * (r_smp_kj / R_solar_kj) ** negQ) - 1) * 10_000

        offset = np.nanmax(np.abs(eR_smp_ij), axis=1, keepdims=True)

        # Set these values to nan
        if first_only:
            ignore = offset < largest_offset
            include = np.invert(ignore)
            if ignore.any():
                logger.warning(f'{np.count_nonzero(ignore)} rows have largest offsets smaller than'
                            f' {largest_offset} at the minimun dilution factor of {min_dilution_factor}. '
                            f'These rows are set to nan.')
            first_only = False

        isclose = np.isclose(offset, largest_offset, rtol=largest_offset_rtol, atol=0)
        if not np.all(isclose[include]):
            dilution_factor[include] = dilution_factor[include] * (offset[include]/largest_offset)
        else:
            break
    else:
        logger.warning(f'Not all {abu_up.shape[0]} rows converged after {max_iterations}. '
                       f'{np.count_nonzero(np.invert(isclose))} non-converged rows set to nan.')

    if ignore.any():
        eR_smp_ij[ignore.flatten(), :] = np.nan
        dilution_factor[ignore] = np.nan

    return dict(eRi_values = eR_smp_ij, dilution_factor = dilution_factor,
                largest_offset=largest_offset, min_dilution_factor=min_dilution_factor,
                )


def intnorm_precision(abu_up, abu_down, abu_norm,
                      mass_up, mass_down, mass_norm,
                      solar_up, solar_down, solar_norm,
                      dilution_step = 0.1, precision = 0.01):
    # Im not sure I understand this well enough to implement so I'll leave ir for now
    # The way the method works means it needs to calculate a ratio between two slopes to work
    # Also the give_ratio_gm method is very inefficent...
    pass


def internal_normalisation(abu, numerators, normrat, stdmass, stdabu, enrichment_factor=1, relative_enrichment=True,
                           method='largest_offset', **method_kwargs):
    # iso_slope is not done here. This should be done later.
    # That way you know the direction of the slope and you dont have to rerun for different slopes.

    if abu.dtype.names is None:
        raise ValueError('``abu`` must be a keyarray')
    if stdmass.dtype.names is None:
        raise ValueError('``stdmass`` must be a keyarray')
    if stdabu.dtype.names is None:
        raise ValueError('``stdabu`` must be a keyarray')

    if method.lower() == 'largest_offset':
        methodfunc = intnorm_largest_offset
    elif method.lower() == 'precision':
        methodfunc = intnorm_precision
    elif method.lower() == 'simplified_linear':
        methodfunc = intnorm_linear
        method_kwargs['mass_coef'] = 'simplified'
    elif method.lower() == 'better_linear' or  method.lower() == 'linear':
        methodfunc = intnorm_linear
        method_kwargs['mass_coef'] = 'better'
    else:
        raise ValueError('``method`` must be one of "largest_offset", "precision", "simplified_linear", "better_linear"')

    if isinstance(normrat, (list, tuple)):
        if not isinstance(numerators, (list, tuple)) or len(numerators) != len(normrat):
            raise ValueError('``numerators`` must be an iterable the same length as ``normrat``')

        if isinstance(enrichment_factor, (list, tuple)):
            if len(enrichment_factor) != len(normrat):
                raise ValueError('``enrichment_factor`` must be an iterable the same length as ``normrat``')
            else:
                pass # Fine
        else:
            enrichment_factor = [enrichment_factor] * len(normrat)

    else:
        numerators = (numerators,)
        normrat = (normrat,)
        enrichment_factor = (enrichment_factor,)

    all_abu_up = []
    all_abu_down = []
    all_abu_norm = []
    all_mass_up = []
    all_mass_down = []
    all_mass_norm = []
    all_solar_up = []
    all_solar_down = []
    all_solar_norm = []
    all_iso_up = ()
    all_iso_down = ()
    all_iso_norm = ()
    for numerators_, normrat_, abu_factor_ in zip(numerators, normrat, enrichment_factor):
        numerators_ = utils.asisotopes(numerators_)
        normrat_ = utils.asratio(normrat_)

        if normrat_.numer not in numerators_:
            numerators_ += (normrat_.numer,)
        if normrat_.denom not in numerators_:
            numerators_ += (normrat_.denom, )

        numeri = numerators_.index(normrat_.numer)
        denomi = numerators_.index(normrat_.denom)

        all_iso_up += numerators_
        all_iso_down += tuple(normrat_.denom for n in numerators_)
        all_iso_norm += tuple(normrat_.numer for n in numerators_)

        abu_up = np.array([abu[numerator] for numerator in numerators_])
        if relative_enrichment is False:
            # Renormalise so that the sum of all numerators = 1
            # This works for both ndim = 1 & 2 as long as it is done before transpose
            abu_up = abu_up / abu_up.sum(axis=0)

        abu_up = abu_up * abu_factor_

        all_abu_up.append(abu_up)

        # Same isotope for all numerators
        all_abu_down.append(np.ones(abu_up.shape) * abu_up[denomi])
        all_abu_norm.append(np.ones(abu_up.shape) * abu_up[numeri])

        mass_up = np.array([stdmass[numerator.without_suffix()] for numerator in numerators_])
        all_mass_up.append(mass_up)
        all_mass_down.append(np.ones(mass_up.shape) * mass_up[denomi])
        all_mass_norm.append(np.ones(mass_up.shape) * mass_up[numeri])

        solar_up = np.array([stdabu[numerator.without_suffix()] for numerator in numerators_])
        all_solar_up.append(solar_up)
        all_solar_down.append(np.ones(solar_up.shape) * solar_up[denomi])
        all_solar_norm.append(np.ones(solar_up.shape) * solar_up[numeri])

    # Joins all arrays and makes sure dimensions line up
    all_abu_up = np.concatenate(all_abu_up, axis=0).transpose()
    all_abu_down = np.concatenate(all_abu_down, axis=0).transpose()
    all_abu_norm = np.concatenate(all_abu_norm, axis=0).transpose()

    all_mass_up = np.concatenate(all_mass_up, axis=0).transpose()
    all_mass_down = np.concatenate(all_mass_down, axis=0).transpose()
    all_mass_norm = np.concatenate(all_mass_norm, axis=0).transpose()

    all_solar_up = np.concatenate(all_solar_up, axis=0).transpose()
    all_solar_down = np.concatenate(all_solar_down, axis=0).transpose()
    all_solar_norm = np.concatenate(all_solar_norm, axis=0).transpose()

    result = methodfunc(all_abu_up, all_abu_down, all_abu_norm,
                        all_mass_up, all_mass_down, all_mass_norm,
                        all_solar_up, all_solar_down, all_solar_norm,
                        **method_kwargs)

    result['eRi_keys'] = all_iso_up

    result['i_key'] = dict(zip(all_iso_up, all_iso_up))
    result['j_key'] = dict(zip(all_iso_up, all_iso_down))
    result['k_key'] = dict(zip(all_iso_up, all_iso_norm))
    result['ij_key'] = dict(zip(all_iso_up, utils.asratios([f'{n}/{d}' for n, d in zip(all_iso_up, all_iso_down)])))
    result['kj_key'] = dict(zip(all_iso_up, utils.asratios([f'{n}/{d}' for n, d in zip(all_iso_norm, all_iso_down)])))
    result['label'] = dict(zip(all_iso_up, [f'ε{i}({kj.numer.mass[-1]}{kj.denom.mass[-1]})'
                              for i, kj in result['kj_key'].items()]))
    result['label_latex'] = dict(zip(all_iso_up, [fr'$\epsilon{i.latex(dollar=False)}{{}}_{{({kj.numer.mass[-1]}{kj.denom.mass[-1]})}}$'
                                    for i, kj in result['kj_key'].items()]))
    result['eRi'] = utils.askeyarray(result['eRi_values'], all_iso_up)

    return result

def simple_normalisation(abu, numerators, normiso, stdabu, enrichment_factor=1, relative_enrichment=True):
    if abu.dtype.names is None:
        raise ValueError('``abu`` must be a keyarray')
    if stdabu.dtype.names is None:
        raise ValueError('``stdabu`` must be a keyarray')

    if isinstance(normiso, (list, tuple)):
        if not isinstance(numerators, (list, tuple)) or len(numerators) != len(normiso):
            raise ValueError('``numerators`` must be an iterable the same length as ``normiso``')

        if isinstance(enrichment_factor, (list, tuple)):
            if len(enrichment_factor) != len(normiso):
                raise ValueError('``enrichment_factor`` must be an iterable the same length as ``normiso``')
            else:
                pass # Fine
        else:
            enrichment_factor = [enrichment_factor] * len(normiso)

    else:
        numerators = (numerators,)
        normiso = (normiso,)
        enrichment_factor = (enrichment_factor,)

    all_abu_up = []
    all_abu_down = []
    all_solar_up = []
    all_solar_down = []
    all_iso_up = ()
    all_iso_down = ()
    for numerators_, normiso_, abu_factor_ in zip(numerators, normiso, enrichment_factor):
        numerators_ = utils.asisotopes(numerators_)
        normiso_ = utils.asisotope(normiso_)

        if normiso_ not in numerators_:
            numerators_ += (normiso_,)

        denomi = numerators_.index(normiso_)

        all_iso_up += numerators_
        all_iso_down += tuple(normiso_ for n in numerators_)

        abu_up = np.array([abu[numerator] for numerator in numerators_])
        if relative_enrichment is False:
            # Renormalise so that the sum of all numerators = 1
            # This works for both ndim = 1 & 2 as long as it is done before transpose
            abu_up = abu_up / abu_up.sum(axis=0)

        abu_up = abu_up * abu_factor_

        all_abu_up.append(abu_up)

        # Same isotope for all numerators
        all_abu_down.append(np.ones(abu_up.shape) * abu_up[denomi])

        solar_up = np.array([stdabu[numerator.without_suffix()] for numerator in numerators_])
        all_solar_up.append(solar_up)
        all_solar_down.append(np.ones(solar_up.shape) * solar_up[denomi])

    # Joins all arrays and makes sure dimensions line up
    all_abu_up = np.atleast_2d(np.concatenate(all_abu_up, axis=0).transpose())
    all_abu_down = np.atleast_2d(np.concatenate(all_abu_down, axis=0).transpose())

    all_solar_up = np.atleast_2d(np.concatenate(all_solar_up, axis=0).transpose())
    all_solar_down = np.atleast_2d(np.concatenate(all_solar_down, axis=0).transpose())

    # There is only one way to do this so no need for a separate function
    Rij = (all_abu_up/all_abu_down)/(all_solar_up/all_solar_down) - 1.0

    result = dict(Ri_values = Rij, Ri_keys=all_iso_up, Ri = utils.askeyarray(Rij, all_iso_up))

    result['i_key'] = dict(zip(all_iso_up, all_iso_up))
    result['j_key'] = dict(zip(all_iso_up, all_iso_down))
    result['ij_key'] = dict(zip(all_iso_up, utils.asratios([f'{n}/{d}' for n, d in zip(all_iso_up, all_iso_down)])))
    result['label'] = dict(zip(all_iso_up, [f'{i}/{j.mass}'
                                             for i, j in result['j_key'].items()]))
    result['label_latex'] = dict(
        zip(all_iso_up, [fr'${{}}^{{{i.mass}/{j.mass}}}\mathrm{{{i.element}}}$'
                         for i, j in result['j_key'].items()]))


    return result