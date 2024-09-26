import numpy as np
import scipy.optimize

import simple.utils as utils
from scipy.optimize import fsolve

import logging

logger = logging.getLogger('SIMPLE.norm')

def intnorm_linear(abu_i, abu_j, abu_k,
                   mass_i, mass_j, mass_k,
                   std_i, std_j, std_k,
                   mass_coef = 'better'):
    """
    Normalise the data using the linearised internal normalisation procedure.

    The internal normalisation procedure uses Equation 7 from
    [Lugaro et al. 2023](https://doi.org/10.1140/epja/s10050-023-00968-y),

    $$
    \\epsilon R^{\\mathrm{ABU}}_{ij} = {\\left[{\\left(\\frac{r^{\\mathrm{ABU}}_{ij}}{R^{\\mathrm{STD}}_{ij}}-1\\right)}
    -{Q}_{i} \\times {\\left(\\frac{r^{\\mathrm{ABU}}_{kj}}{R^{\\mathrm{STD}}_{kj}}-1\\right)}
    \\right]} \\times 10^4 $$

    Where, $Q$ is the difference of the masses calculated in one of two ways. If ``mass_coeff="better"`` the
    definition from Lugaro et al. (2023) is used,

    $$
    {Q}^{\\mathrm{better}} = \\frac{\\ln{(m_i)} - \\ln{(m_j)}}{\\ln{(m_k)} - \\ln{(m_j)}}
    $$

    if ``mass_coeff="simplified"`` the definition from e.g. Dauphas et al. (2004) is used,

    $$
    {Q}^{\\rm{simplified}} = \\frac{(m_i) - (m_j)}{(m_k) - (m_j)}
    $$

    Args:
        abu_i (): Abundance of the numerator isotopes.
        abu_j (): Abundance of the denominator isotopes
        abu_k (): Abundance of the normalising numerators.
        mass_i (): The mass of each isotope in ``abu_i``.
        mass_j (): The mass of each isotope in ``abu_j``.
        mass_k (): The mass of each isotope in ``abu_k``.
        std_i (): The reference abundance of each isotope in ``abu_i``.
        std_j (): The reference abundance of each isotope in ``abu_j``.
        std_k (): The reference abundance of each isotope in ``abu_k``.
        largest_offset (): The absolute value of the largest offset for each row finished calculation in epsilon units.
        min_dilution_factor (): The smallest dilution factor considered in the calcualtion. If the largest offset found
            at this dilution factor is smaller than ``largest_offset`` the result is set to ``np.nan``.
        max_iterations (): Any row for which the results have not converged after this number of iterations is set
            to ``np.nan``
        largest_offset_rtol (): The relative tolerance for convergence of largest offset calculation.

    **Notes**
    Enrichment factors will have no impact on the the results from this method.

    Returns: A dictionary containing the results of the normalisation.

    The dictionary contains the following items:
        - ``eRi_values``: An 2dim array containing the eRi values for each isotope.
        - ``dilution_factor``: The dilution factor for each row in ``eRi_values``.
        - ``largest_offset``: The ``largest_offset`` argument.
        - ``min_dilution_factor``: The ``min_dilution_factor`` argument.
    """

    abu_i, abu_j, abu_k = np.atleast_2d(abu_i), np.atleast_2d(abu_j), np.atleast_2d(abu_k)
    mass_i, mass_j, mass_k = np.atleast_2d(mass_i), np.atleast_2d(mass_j), np.atleast_2d(mass_k)
    std_i, std_j, std_k = np.atleast_2d(std_i), np.atleast_2d(std_j), np.atleast_2d(std_k)

    rho_ij = ((abu_i / abu_j) / (std_i / std_j)) - 1.0
    rho_kj = ((abu_k / abu_j) / (std_k / std_j)) - 1.0

    if mass_coef.lower() == 'better':
        Q = (np.log(mass_i) - np.log(mass_j)) / (np.log(mass_k) - np.log(mass_j))
    elif mass_coef.lower() == 'simplified':
        Q = (mass_i - mass_j) / (mass_k - mass_j)
    else:
        raise ValueError('``mass_coef`` must be either "better" or "simplified"')

    # Equation 7 in Lugaro et al., 2023
    eR_smp_ij = (rho_ij - Q * rho_kj) * 10_000
    return dict(eRi_values=eR_smp_ij,
                method='linear', mass_coeff=mass_coef)



def intnorm_largest_offset(abu_i, abu_j, abu_k,
                           mass_i, mass_j, mass_k,
                           std_i, std_j, std_k,
                           largest_offset = 1, min_dilution_factor=0.1, max_iterations=100,
                           largest_offset_rtol = 1E-4):
    """
    Internally normalises a synthetic sample such that the largest offset is equal to the specified value.


    The internal normalisation procedure uses Equation 6 from
    [Lugaro et al. 2023](https://doi.org/10.1140/epja/s10050-023-00968-y),

    $$
    \\epsilon R^{\\mathrm{SMP}}_{ij} = {\\left[{\\left(\\frac{r^{\\mathrm{SMP}}_{ij}}{R^{\\mathrm{STD}}_{ij}}\\right)}
    {\\left(\\frac{r^{\\mathrm{SMP}}_{kj}}{R^{\\mathrm{STD}}_{kj}}\\right)}^{-Q_i} - 1
    \\right]} \\times 10^4 $$

    Where, $Q$ is the difference in the natural logarithm of the masses,

    $$
    Q = \\frac{\\ln{(m_i)} - \\ln{(m_j)}}{\\ln{(m_k)} - \\ln{(m_j)}}
    $$

    The composition of the synthetic sample ($\\mathrm{SMP}$) is calculated by adding
    $\\mathrm{ABU}$,  divided by the dilution factor ($\\mathrm{df}$), to $\\mathrm{STD}$,

    $$
    C_{SMP} = C_{\\mathrm{STD}} +  \\left(\\frac{C_{ABU}}{\\mathrm{df}}\\right)
    $$

    Args:
        abu_i (): Abundance of the numerator isotopes.
        abu_j (): Abundance of the denominator isotopes
        abu_k (): Abundance of the normalising numerators.
        mass_i (): The mass of each isotope in ``abu_i``.
        mass_j (): The mass of each isotope in ``abu_j``.
        mass_k (): The mass of each isotope in ``abu_k``.
        std_i (): The reference abundance of each isotope in ``abu_i``.
        std_j (): The reference abundance of each isotope in ``abu_j``.
        std_k (): The reference abundance of each isotope in ``abu_k``.
        largest_offset (): The absolute value of the largest offset for each row finished calculation in epsilon units.
        min_dilution_factor (): The smallest dilution factor considered in the calcualtion. If the largest offset found
            at this dilution factor is smaller than ``largest_offset`` the result is set to ``np.nan``.
        max_iterations (): Any row for which the results have not converged after this number of iterations is set
            to ``np.nan``
        largest_offset_rtol (): The relative tolerance for convergence of largest offset calculation.

    Returns: A dictionary containing the results of the normalisation.

    The dictionary contains the following items:
        - ``eRi_values``: An 2dim array containing the eRi values for each isotope.
        - ``dilution_factor``: The dilution factor for each row in ``eRi_values``.
        - ``largest_offset``: The ``largest_offset`` argument.
        - ``min_dilution_factor``: The ``min_dilution_factor`` argument.
    """



    # Make sure everything is at least 2d
    abu_i,  abu_j, abu_k = np.atleast_2d(abu_i), np.atleast_2d(abu_j), np.atleast_2d(abu_k)
    mass_i, mass_j, mass_k = np.atleast_2d(mass_i), np.atleast_2d(mass_j), np.atleast_2d(mass_k)
    std_i, std_j, std_k  = np.atleast_2d(std_i), np.atleast_2d(std_j), np.atleast_2d(std_k)

    negQ = ((np.log(mass_i / mass_j) /
             np.log(mass_k / mass_j))) * -1 # Needs to be negative later

    R_solar_ij = std_i / std_j
    R_solar_kj = std_k / std_j

    # Has to begin at largest offset or it might accidentally ignore rows
    dilution_factor = np.full((abu_i.shape[0], 1), min_dilution_factor, dtype=np.float64)

    first_time = True
    logger.info(f'Internally normalising {abu_i.shape[0]} rows using the largest offset method.')
    for i in range (max_iterations):
        smp_up = std_i + (abu_i / dilution_factor)
        smp_down = std_j + (abu_j / dilution_factor)
        smp_norm = std_k + (abu_k / dilution_factor)

        r_smp_ij = smp_up / smp_down
        r_smp_kj = smp_norm / smp_down

        # Equation 6 in Lugaro et al 2023
        eR_smp_ij = (((r_smp_ij / R_solar_ij) * (r_smp_kj / R_solar_kj) ** negQ) - 1) * 10_000

        offset = np.nanmax(np.abs(eR_smp_ij), axis=1, keepdims=True)

        if first_time:
            ignore = offset < largest_offset
            include = np.invert(ignore)
            if ignore.any():
                logger.warning(f'{np.count_nonzero(ignore)} rows have largest offsets smaller than'
                            f' {largest_offset} at the minimun dilution factor of {min_dilution_factor}. '
                            f'These rows are set to nan.')
            first_time = False

        isclose = np.isclose(offset, largest_offset, rtol=largest_offset_rtol, atol=0)
        if not np.all(isclose[include]):
            dilution_factor[include] = dilution_factor[include] * (offset[include]/largest_offset)
        else:
            break
    else:
        logger.warning(f'Not all {abu_i.shape[0]} rows converged after {max_iterations}. '
                       f'{np.count_nonzero(np.invert(isclose))} non-converged rows set to nan.')

    if ignore.any():
        eR_smp_ij[ignore.flatten(), :] = np.nan
        dilution_factor[ignore] = np.nan

    return dict(eRi_values = eR_smp_ij, dilution_factor = dilution_factor,
                largest_offset=largest_offset, min_dilution_factor=min_dilution_factor,
                method='largest_offset')


def intnorm_precision(abu_up, abu_down, abu_norm,
                      mass_up, mass_down, mass_norm,
                      solar_up, solar_down, solar_norm,
                      dilution_step = 0.1, precision = 0.01):
    # Im not sure I understand this well enough to implement so I'll leave ir for now
    # The way the method works means it needs to calculate a ratio between two slopes to work
    # Also the give_ratio_gm method is very inefficent...
    pass


def internal_normalisation(abu, numerators, normrat, stdmass, stdabu, enrichment_factor=1, relative_enrichment=True,
                           abu_massunit=False, stdabu_massunit=False,
                           method='largest_offset', **method_kwargs):
    """


    Args:
        abu (): Must be a [keyarray][simple.askeyarray].
        numerators (): The numerator isotopes (i) in the calculation.
        normrat (): The ratio (kj) used for internal normalisation.
        stdmass (): A [keyarray][simple.askeyarray] containing the isotope masses.
        stdabu (): A [keyarray][simple.askeyarray] containing the reference abundances.
        enrichment_factor (): Enrichment factor applied to ``abu``. Useful when doing multiple elements at once.
        relative_enrichment (): If ''True'' the enrichment factor is applied to the ``abu`` abundances as is.
            If ``False`` the abundance of all isotopes in ``numerators`` is renormalised such that their sum = 1 before
            being multiplied by ``enrichment_factor``.
        method (): The method used. See options below.
        **method_kwargs (): Keyword arguments for the chosen method.

    **Notes**
    The ``normrat`` numerator and denominator isotopes will be appended to ``numerators`` if not initially included.
    This is done before the enrichment factor calculation.

    **Methods**
     - ``largest_offset`` This is the default method which internally normalises a synthetic sample such that
        the largest offset, in epsilon units, is equal to a specified value. For more details and a list of additional
        arguments see [here][simple.norm.intnorm_largest_offset].

    Returns: A dictionary containing the results of the normalisation.

    The dictionary at minimum contains the following items:
        - ``eRi_values``: An 2dim array containing the eRi values for each isotope.
        - ``eRi_keys``: The numerator isotopes for each column in ``eRi_values``.
        - ``eRi``: A keyarray containing the eRi values for each column in ``eRi_keys``.
        - ``ij_key``, ``kj_key``: Dictionaries mapping the ``eRi_keys`` to the numerator-denominator ratio and the
            normalising ratio for each column in ``eRi``.
        - ``label``, ``label_latex``: Dictionaries mapping the ``eRi_keys`` to plain text and latex labels suitable
            for plotting. Contains the ε symbol followed by the numerator isotope and the last digit of each mass in
            the normalising ratio, in brackets.

    Additional entries might be supplied by the different methods.

    """
    # iso_slope is not done here. This should be done later.
    # That way you know the direction of the slope and you don't have to rerun for different slopes.

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
    elif method.lower() == 'better_linear':
        methodfunc = intnorm_linear
        method_kwargs['mass_coef'] = 'better'
    elif method.lower() == 'linear':
        methodfunc = intnorm_linear
    else:
        raise ValueError('``method`` must be one of "largest_offset", "precision", "linear", "simplified_linear", "better_linear"')

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

    all_iso_up, all_iso_down, all_iso_norm = (), (), ()
    all_abu_up, all_abu_down, all_abu_norm = [], [], []
    all_mass_up, all_mass_down, all_mass_norm = [], [], []
    all_solar_up, all_solar_down, all_solar_norm = [], [], []

    for isotopes, rat, abu_factor in zip(numerators, normrat, enrichment_factor):
        isotopes = utils.asisotopes(isotopes)
        rat = utils.asratio(rat)

        if rat.numer not in isotopes:
            isotopes += (rat.numer,)
        if rat.denom not in isotopes:
            isotopes += (rat.denom, )

        numeri = isotopes.index(rat.numer)
        denomi = isotopes.index(rat.denom)

        all_iso_up += isotopes
        all_iso_down += tuple(rat.denom for n in isotopes)
        all_iso_norm += tuple(rat.numer for n in isotopes)

        abu_up = np.array([abu[numerator] for numerator in isotopes])
        if relative_enrichment is False:
            # Renormalise so that the sum of all numerators = 1
            # This works for both ndim = 1 & 2 as long as it is done before transpose
            abu_up = abu_up / abu_up.sum(axis=0)

        abu_up = abu_up * abu_factor
        if abu_massunit:
            abu_up = abu_up / [[float(numerator.mass)] for numerator in isotopes]

        all_abu_up.append(abu_up)

        # Same isotope for all numerators
        all_abu_down.append(np.ones(abu_up.shape) * abu_up[denomi])
        all_abu_norm.append(np.ones(abu_up.shape) * abu_up[numeri])

        # Ignore the suffix for the arrays containing standard data
        mass_up = np.array([stdmass[numerator.without_suffix()] for numerator in isotopes])
        all_mass_up.append(mass_up)
        all_mass_down.append(np.ones(mass_up.shape) * mass_up[denomi])
        all_mass_norm.append(np.ones(mass_up.shape) * mass_up[numeri])

        solar_up = np.array([stdabu[numerator.without_suffix()] for numerator in isotopes])
        if stdabu_massunit:
            solar_up = solar_up / [[float(numerator.mass)] for numerator in isotopes]
        all_solar_up.append(solar_up)
        all_solar_down.append(np.ones(solar_up.shape) * solar_up[denomi])
        all_solar_norm.append(np.ones(solar_up.shape) * solar_up[numeri])

    # Make one big array
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

    result['eRi'] = utils.askeyarray(result['eRi_values'], all_iso_up)
    result['eRi_keys'] = all_iso_up

    # Create mappings linking the array keys to the different isotopes used in the equations
    #result['i_key'] = dict(zip(all_iso_up, all_iso_up))
    #result['j_key'] = dict(zip(all_iso_up, all_iso_down))
    #result['k_key'] = dict(zip(all_iso_up, all_iso_norm))
    result['ij_key'] = dict(zip(all_iso_up, utils.asratios([f'{n}/{d}' for n, d in zip(all_iso_up, all_iso_down)])))
    result['kj_key'] = dict(zip(all_iso_up, utils.asratios([f'{n}/{d}' for n, d in zip(all_iso_norm, all_iso_down)])))

    # Labels suitable for plotting
    result['label'] = dict(zip(all_iso_up, [f'ε{i}({kj.numer.mass[-1]}{kj.denom.mass[-1]})'
                              for i, kj in result['kj_key'].items()]))
    result['label_latex'] = dict(zip(all_iso_up, [fr'$\epsilon{i.latex(dollar=False)}{{}}_{{({kj.numer.mass[-1]}{kj.denom.mass[-1]})}}$'
                                    for i, kj in result['kj_key'].items()]))


    return result

def simple_normalisation(abu, numerators, normiso, stdabu, enrichment_factor=1, relative_enrichment=True, abu_massunit=False):
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
    for isotopes, denominator, abu_factor in zip(numerators, normiso, enrichment_factor):
        isotopes = utils.asisotopes(isotopes)
        denominator = utils.asisotope(denominator)

        if denominator not in isotopes:
            isotopes += (denominator,)

        denomi = isotopes.index(denominator)

        all_iso_up += isotopes
        all_iso_down += tuple(denominator for n in isotopes)

        abu_up = np.array([abu[numerator] for numerator in isotopes])
        if relative_enrichment is False:
            # Renormalise so that the sum of all numerators = 1
            # This works for both ndim = 1 & 2 as long as it is done before transpose
            abu_up = abu_up / abu_up.sum(axis=0)

        abu_up = abu_up * abu_factor
        if abu_massunit:
            abu_up = abu_up / [[float(numerator.mass)] for numerator in isotopes]

        all_abu_up.append(abu_up)

        # Same isotope for all numerators
        all_abu_down.append(np.ones(abu_up.shape) * abu_up[denomi])

        solar_up = np.array([stdabu[numerator.without_suffix()] for numerator in isotopes])
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