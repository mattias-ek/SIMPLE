import numpy as np
import simple.utils as utils

import logging
logger = logging.getLogger('SIMPLE.norm')

__all__ = []

IntNormMethods = {}

@utils.add_shortcut('better', mass_coef ='better')
@utils.add_shortcut('simplified', mass_coef ='simplified')
@utils.set_default_kwargs()
def intnorm_linear(abu_i, abu_j, abu_k,
                   mass_i, mass_j, mass_k,
                   std_i, std_j, std_k,
                   mass_coef = 'better', dilution_factor=None, *, msg_prefix=''):
    """
    Internally normalise the abundances using the linearised internal normalisation procedure.

    The internal normalisation procedure uses Equation 7 from
    [Lugaro et al. 2023](https://doi.org/10.1140/epja/s10050-023-00968-y),

    $$
    \\epsilon R^{\\mathrm{ABU}}_{ij} = {\\left[{\\left(\\frac{r^{\\mathrm{ABU}}_{ij}}{R^{\\mathrm{STD}}_{ij}}-1\\right)}
    -{Q}_{i} \\times {\\left(\\frac{r^{\\mathrm{ABU}}_{kj}}{R^{\\mathrm{STD}}_{kj}}-1\\right)}
    \\right]} \\times 10^4 $$

    Where, $Q$ is the difference of the masses calculated in one of two ways. If ``mass_coeff="better"`` the
    definition from e.g. Lugaro et al. (2023) is used,

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
        abu_k (): Abundance of the normalising isotopes.
        mass_i (): The mass of each isotope in ``abu_i``.
        mass_j (): The mass of each isotope in ``abu_j``.
        mass_k (): The mass of each isotope in ``abu_k``.
        std_i (): The reference abundance of each isotope in ``abu_i``.
        std_j (): The reference abundance of each isotope in ``abu_j``.
        std_k (): The reference abundance of each isotope in ``abu_k``.
        mass_coef (string): Decides how the mass coefficient Q is calculated. Acceptable values are
        ``better`` and ``simplified``.

    **Notes**
    The epsilon values returned by this method will typically be very large as no dilution takes places.

    Enrichment factors will have no impact on the results from this method.


    Returns:
        dict: A dictionary containing the results of the normalisation. The dictionary contains the following attrs:

            - ``eRi_values``: An 2dim values containing the eRi values for each isotope.
            - ``mass_coef``: The value of the ``mass_coef`` parameter.
            - ``method``: The method used to normalise the abundances. Always ``"linear"`` for this method.
    """
    if msg_prefix:
        msg_prefix = f'{msg_prefix}-linear'
    else:
        msg_prefix = f'linear'
    abu_i, abu_j, abu_k = np.atleast_2d(abu_i), np.atleast_2d(abu_j), np.atleast_2d(abu_k)
    mass_i, mass_j, mass_k = np.atleast_2d(mass_i), np.atleast_2d(mass_j), np.atleast_2d(mass_k)
    std_i, std_j, std_k = np.atleast_2d(std_i), np.atleast_2d(std_j), np.atleast_2d(std_k)

    if dilution_factor is None or dilution_factor <= 0:
        df = 0
    else:
        abu_i = abu_i / dilution_factor + std_i
        abu_j = abu_j / dilution_factor + std_j
        df = dilution_factor

    rho_ij = ((abu_i / abu_j) / (std_i / std_j)) - 1.0
    rho_kj = ((abu_k / abu_j) / (std_k / std_j)) - 1.0

    if mass_coef.lower() == 'better':
        Q = (np.log(mass_i) - np.log(mass_j)) / (np.log(mass_k) - np.log(mass_j))
    elif mass_coef.lower() == 'simplified':
        Q = (mass_i - mass_j) / (mass_k - mass_j)
    else:
        raise ValueError(f'{msg_prefix}: ``mass_coef`` must be either "better" or "simplified"')

    # Equation 7 in Lugaro et al., 2023
    eR_smp_ij = (rho_ij - Q * rho_kj) * 10_000
    return dict(eRi_values=eR_smp_ij,
                method='linear', mass_coeff=mass_coef, dilution_factor=df)



@utils.set_default_kwargs()
def intnorm_largest_offset(abu_i, abu_j, abu_k,
                           mass_i, mass_j, mass_k,
                           std_i, std_j, std_k,
                           largest_offset = 1, min_dilution_factor=0.1, max_iterations=100,
                           largest_offset_rtol = 1E-4, *, msg_prefix=''):
    """
    Creates and internally normalises a synthetic sample such that the largest offset is equal to the specified value.

    The composition of the synthetic sample ($\\mathrm{SMP}$) is calculated by adding
    $\\mathrm{ABU}$,  divided by the dilution factor ($\\mathrm{df}$), to $\\mathrm{STD}$,

    $$
    C_{SMP} = C_{\\mathrm{STD}} +  \\left(\\frac{C_{ABU}}{\\mathrm{df}}\\right)
    $$

    The internal normalisation procedure uses Equation 6 from
    [Lugaro et al. 2023](https://doi.org/10.1140/epja/s10050-023-00968-y),

    $$
    \\epsilon R^{\\mathrm{SMP}}_{ij} = {\\left[{\\left(\\frac{r^{\\mathrm{SMP}}_{ij}}{R^{\\mathrm{STD}}_{ij}}\\right)}
    {\\left(\\frac{r^{\\mathrm{SMP}}_{kj}}{R^{\\mathrm{STD}}_{kj}}\\right)}^{-Q_i} - 1
    \\right]} \\times 10^4
    $$

    Where, $Q$ is the difference in the natural logarithm of the masses,

    $$
    Q = \\frac{\\ln{(m_i)} - \\ln{(m_j)}}{\\ln{(m_k)} - \\ln{(m_j)}}
    $$

    Args:
        abu_i (): Abundance of the numerator isotopes.
        abu_j (): Abundance of the denominator isotopes
        abu_k (): Abundance of the normalising isotopes.
        mass_i (): The mass of each isotope in ``abu_i``.
        mass_j (): The mass of each isotope in ``abu_j``.
        mass_k (): The mass of each isotope in ``abu_k``.
        std_i (): The reference abundance of each isotope in ``abu_i``.
        std_j (): The reference abundance of each isotope in ``abu_j``.
        std_k (): The reference abundance of each isotope in ``abu_k``.
        largest_offset (): The absolute value of the largest offset for each row finished calculation, in epsilon units.
        min_dilution_factor (): The smallest dilution factor considered in the calculation. If the offset found
            at this dilution factor is smaller than ``largest_offset`` the result is set to ``np.nan``.
        max_iterations (): Any row for which the results have not converged after this number of iterations is set
            to ``np.nan``
        largest_offset_rtol (): The relative tolerance used to test for convergence of the largest offset value.

    Returns:
        dict: A dictionary containing the results of the normalisation. The dictionary contains the following attrs:

            - ``eRi_values``: An 2dim values containing the eRi values for each isotope.
            - ``dilution_factor``: The dilution factor for each row in ``eRi_values``.
            - ``largest_offset``: The ``largest_offset`` parameter.
            - ``min_dilution_factor``: The ``min_dilution_factor`` parameter.
            - ``method``: The method used to normalise the abundances. Always ``"largest_offset"`` for this method.
    """
    if msg_prefix:
        msg_prefix = f'{msg_prefix}-largest_offset'
    else:
        msg_prefix = 'largest_offset'

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
    logger.info(f'{msg_prefix}: Internally normalising {abu_i.shape[0]} rows using the largest offset method.')
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
                logger.warning(f'{msg_prefix}: {np.count_nonzero(ignore)} rows out of {ignore.size} have largest offsets smaller than'
                            f' {largest_offset} ε-units at the minimum dilution factor of {min_dilution_factor}. '
                            f'These rows are set to nan.')
            first_time = False

        isclose = np.isclose(offset, largest_offset, rtol=largest_offset_rtol, atol=0)
        if not np.all(isclose[include]):
            dilution_factor[include] = dilution_factor[include] * (offset[include]/largest_offset)
        else:
            break
    else:
        logger.warning(f'{msg_prefix}: Not all {abu_i.shape[0]} rows converged after {max_iterations}. '
                       f'{np.count_nonzero(np.invert(isclose))} non-converged rows set to nan.')

    if ignore.any():
        eR_smp_ij[ignore.flatten(), :] = np.nan
        dilution_factor[ignore.flatten()] = np.nan

    return dict(eRi_values = eR_smp_ij, dilution_factor = dilution_factor,
                largest_offset=largest_offset, min_dilution_factor=min_dilution_factor,
                method='largest_offset')

IntNormMethods['linear'] = intnorm_linear
IntNormMethods['better_linear'] = intnorm_linear.better
IntNormMethods['simplified_linear'] = intnorm_linear.simplified
IntNormMethods['largest_offset'] = intnorm_largest_offset

@utils.set_default_kwargs()
def internal_normalisation(abu, isotopes, normrat, stdmass, stdabu,
                           enrichment_factor=1, relative_enrichment=True,
                           std_enrichment_factor=1, std_relative_enrichment=True,
                           method='largest_offset', *, msg_prefix='', **method_kwargs):
    """
    Normalise the abundances of ``abu`` relative to the keys ``normrat`` using the internal normalisation procedure
    commonly used for data measured by mass spectrometers.

    Multiple normalisations can be done at once by supplying a list of normalising isotopes. If doing multiple
    elements at once then ``isotopes``, if not ``None``, and optionally ``enrichment_factor`` and
    ``std_enrichement_factor``, must be lists with the same
     length as ``normiso``. If ``enrichment_factor``/``std_enrichment_factor`` is a single value it is applied to
     all elements.

    Args:
        abu (): A [keyarray][simple.askeyarray] containing the abundances to be normalised.
        isotopes (): The numerator isotopes (i) in the calculation. If ``None`` all the isotopes in ``abu`` with the
            same element symbol and suffix as ``normrat`` will be selected.
        normrat (): The keys (kj) used for internal normalisation.
        stdmass (): A [keyarray][simple.askeyarray] containing the isotope masses.
        stdabu (): A [keyarray][simple.askeyarray] containing the reference abundances.
        enrichment_factor (): Enrichment factor applied to ``abu``. Useful when doing multiple elements at once.
        relative_enrichment (): If ''True'' the abundances of all ``isotopes`` in ``abu`` are multiplied by
            ``enrichment_factor``. If ``False`` the abundances are first renormalised such that their sum = 1
            and **then** multiplied by ``enrichment_factor``.
        std_enrichment_factor (): Enrichment factor applied to ``stdabu``. Useful when doing multiple elements at once.
        std_relative_enrichment (): If ''True'' the abundances of all ``isotopes`` in ``stdabu`` are multiplied by
            ``std_enrichment_factor``. If ``False`` the abundances are first renormalised such that their sum = 1
            and **then** multiplied by ``std_enrichment_factor``.
        method (string): The method used. See options in section below.
        **method_kwargs (): Additional arguments for the chosen ``method``.

    **Notes**

    The ``normrat`` numerator and denominator isotopes will be appended to ``isotopes`` if not initially included.
    This is done before the enrichment factor calculation.

    The enrichment factor should only be used in conjunction with the ``largest_offset`` method. It might not work as
    expected for other methods.

    **Methods**

    - ``largest_offset`` This is the default method which internally normalises a synthetic sample such that
        the largest offset, in epsilon units, is equal to a specified value, by default 1. For more details and
        a list of additional arguments see [intnorm_largest_offset][simple.norm.intnorm_largest_offset].

    - ``linear`` Internally normalise the abundances using the linearised internal normalisation procedure. For more
        details anda list of additional arguments see [intnorm_linear][simple.norm.intnorm_linear].

    Returns:
        dict: A dictionary containing the results of the normalisation. The dictionary at minimum contains the following attrs:

            - ``eRi``: A key array containing the eRi values for each column in ``eRi_keys``.
            - ``eRi_keys``: The numerator isotopes for each column in ``eRi_values``.
            - ``ij_key``, ``kj_key``: Dictionaries mapping ``eRi_keys`` to the numerator-denominator keys (ij) and the
                normalising keys (kj) for each column in ``eRi``.
            - ``label_args``, ``label_latex``: Dictionaries mapping ``eRi_keys`` to plain text and latex labels suitable
                for plotting. Contains the ε symbol followed by the numerator isotope and the last digit of each mass in
                the normalising keys, in brackets. E.g. ε104Pd(85) and $\\epsilon{}^{105}\\mathrm{Pd}_{(85)}$,
                where i=Pd-104, k=Pd-108 and j=Pd-105.
            - Additional attrs might be supplied by the different methods.
    """
    if msg_prefix:
        msg_prefix = f'{msg_prefix}-intnorm'
    else:
        msg_prefix = 'intnorm'

    methodfunc = IntNormMethods.get(method.lower(), None)
    if methodfunc is None:
        raise ValueError(f'{msg_prefix}: ``method`` must be  "largest_offset" or "linear"')

    if abu.dtype.names is None:
        raise ValueError(f'{msg_prefix}: ``abu`` must be a keyarray')
    if stdmass is not None and stdmass.dtype.names is None:
        raise ValueError(f'{msg_prefix}: ``stdmass`` must be a keyarray')
    if stdabu is not None and stdabu.dtype.names is None:
        raise ValueError(f'{msg_prefix}: ``stdabu`` must be a keyarray')

    if isinstance(normrat, (list, tuple)):
        if isotopes is None:
            isotopes = [None] * len(normrat)
        elif not isinstance(isotopes, (list, tuple)) or len(isotopes) != len(normrat):
            raise ValueError(f'{msg_prefix}: ``isotopes`` must be an iterable the same length as ``normrat``')

        if isinstance(enrichment_factor, (list, tuple)):
            if len(enrichment_factor) != len(normrat):
                raise ValueError(f'{msg_prefix}: ``enrichment_factor`` must be an iterable the same length as ``normrat``')
        else:
            enrichment_factor = [enrichment_factor] * len(normrat)

        if isinstance(std_enrichment_factor, (list, tuple)):
            if len(std_enrichment_factor) != len(normrat):
                raise ValueError(f'{msg_prefix}: ``solar_enrichment_factor`` must be an iterable the same length as ``normiso``')
        else:
            std_enrichment_factor = [std_enrichment_factor] * len(normrat)

    else:
        isotopes = (isotopes,)
        normrat = (normrat,)
        enrichment_factor = (enrichment_factor,)
        std_enrichment_factor = (std_enrichment_factor,)

    all_iso_up, all_iso_down, all_iso_norm = (), (), ()
    all_abu_up, all_abu_down, all_abu_norm = [], [], []
    all_mass_up, all_mass_down, all_mass_norm = [], [], []
    all_solar_up, all_solar_down, all_solar_norm = [], [], []

    for numerators, rat, abu_factor, solar_factor in zip(isotopes, normrat, enrichment_factor, std_enrichment_factor):
        rat = utils.asratio(rat)

        if rat.numer.symbol != rat.denom.symbol:
            raise ValueError(f'{msg_prefix}: The ``normrat`` numerator and normiso isotopes must be of the same element')

        if numerators is None:
            if rat.numer.suffix != rat.denom.suffix:
                raise ValueError(f'{msg_prefix}: The ``normrat`` numerator and normiso isotopes must have the same suffix '
                                 'for auto discovery of numerator isotopes')

            numerators = utils.get_isotopes_of_element(abu.dtype.names, rat.denom.element)
        else:
            numerators = utils.asisotopes(numerators)

        if rat.numer not in numerators:
            numerators += (rat.numer,)
        if rat.denom not in numerators:
            numerators += (rat.denom,)

        logger.info(f'{msg_prefix}: Internally normalising {numerators} to {rat}.')

        if relative_enrichment is False:
            logger.info(f'{msg_prefix}: Applying absolute enrichment factor to model abundances. '
                        f'Setting the sum of all isotopes to {abu_factor}')
        elif abu_factor != 1:
            logger.info(f'{msg_prefix}: Applying relative enrichment factor to model abundances. '
                        f'Multiplying all isotopes by {abu_factor}.')

        if std_relative_enrichment is False:
            logger.info(f'{msg_prefix}: Applying absolute enrichment factor to standard abundances. '
                        f'Setting the sum of all isotopes to {solar_factor}')
        elif solar_factor != 1:
            logger.info(f'{msg_prefix}: Applying relative enrichment factor to standard abundances. '
                        f'Multiplying all isotopes by {solar_factor}.')

        numeri = numerators.index(rat.numer)
        denomi = numerators.index(rat.denom)

        all_iso_up += numerators
        all_iso_down += tuple(rat.denom for n in numerators)
        all_iso_norm += tuple(rat.numer for n in numerators)

        abu_up = np.array([abu[numerator] for numerator in numerators])

        if relative_enrichment is False:
            # Renormalise so that the sum of all isotopes = 1
            # This works for both ndim = 1 & 2 as long as it is done before transpose
            abu_up = abu_up / abu_up.sum(axis=0)

        abu_up = abu_up * abu_factor
        all_abu_up.append(abu_up)

        # Same isotope for all isotopes
        all_abu_down.append(np.ones(abu_up.shape) * abu_up[denomi])
        all_abu_norm.append(np.ones(abu_up.shape) * abu_up[numeri])

        # Ignore the suffix for the arrays containing standard values
        mass_up = np.array([stdmass[numerator.without_suffix()] for numerator in numerators])
        all_mass_up.append(mass_up)
        all_mass_down.append(np.ones(mass_up.shape) * mass_up[denomi])
        all_mass_norm.append(np.ones(mass_up.shape) * mass_up[numeri])

        solar_up = np.array([stdabu[numerator.without_suffix()] for numerator in numerators])
        if std_relative_enrichment is False:
            # Renormalise so that the sum of all isotopes = 1
            # This works for both ndim = 1 & 2 as long as it is done before transpose
            solar_up = solar_up / solar_up.sum(axis=0)

        solar_up = solar_up * solar_factor
        all_solar_up.append(solar_up)
        all_solar_down.append(np.ones(solar_up.shape) * solar_up[denomi])
        all_solar_norm.append(np.ones(solar_up.shape) * solar_up[numeri])

    # Make one big values
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
                        msg_prefix=msg_prefix, **method_kwargs)

    result['eRi'] = utils.askeyarray(result['eRi_values'], all_iso_up)
    result['eRi_keys'] = all_iso_up

    # Create mappings linking the values keys to the different isotopes used in the equations
    result['ij_keys'] = dict(zip(all_iso_up, utils.asratios([f'{n}/{d}' for n, d in zip(all_iso_up, all_iso_down)])))
    result['kj_keys'] = dict(zip(all_iso_up, utils.asratios([f'{n}/{d}' for n, d in zip(all_iso_norm, all_iso_down)])))

    # Labels suitable for plotting
    result['eRi_keylabels'] = dict(zip(all_iso_up, [f'ε{i}({kj.numer.mass[-1]}{kj.denom.mass[-1]})'
                              for i, kj in result['kj_keys'].items()]))
    result['eRi_keylabels_latex'] = dict(zip(all_iso_up, [fr'$\epsilon{i.latex(dollar=False)}{{}}_{{({kj.numer.mass[-1]}{kj.denom.mass[-1]})}}$'
                                    for i, kj in result['kj_keys'].items()]))

    # Flattens arrays that are meant for the second dimension, so they are compatible with keyarrays.
    for k, v in result.items():
        if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == 1:
            result[k] = v.squeeze()

    return utils.NamedDict(result)

@utils.set_default_kwargs()
def standard_normalisation(abu, isotopes, normiso, stdabu,
                           enrichment_factor=1, relative_enrichment=True,
                           std_enrichment_factor=1, std_relative_enrichment=True,
                           dilution_factor=0, *, msg_prefix=''):
    """
    Normalise the abundances of ``abu`` relative to a specified isotope ``normiso`` as commonly done for
    stardust data.

    The equation used to normalise the data is,

    $$
    R^{\\mathrm{ABU}}_{ij} = {\\left(\\frac{r^{\\mathrm{ABU}}_{ij}}{R^{\\mathrm{STD}}_{ij}}\\right)} - 1
    $$

    Multiple normalisations can be done at once by supplying a list of normalising isotopes. If doing multiple
    elements at once then ``isotopes``, if not ``None``, and optionally ``enrichment_factor`` and
    ``std_enrichement_factor``, must be lists with the same
     length as ``normiso``. If ``enrichment_factor``/``std_enrichment_factor`` is a single value it is applied to
     all elements.

    Args:
        abu (): A [keyarray][simple.askeyarray] containing the abundances to be normalised.
        isotopes (): The numerator isotopes (i) in the calculation. If ``None`` all the isotopes in ``abu`` with the
            same element symbol and suffix as ``normiso`` will be selected.
        normiso (): The denominator isotope for the normalisation.
        stdmass (): A [keyarray][simple.askeyarray] containing the isotope masses.
        stdabu (): A [keyarray][simple.askeyarray] containing the reference abundances.
        enrichment_factor (): Enrichment factor applied to ``abu``. Useful when doing multiple elements at once.
        relative_enrichment (): If ''True'' the abundances of all ``isotopes`` in ``abu`` are multiplied by
            ``enrichment_factor``. If ``False`` the abundances are first renormalised such that their sum = 1
            and **then** multiplied by ``enrichment_factor``.
        std_enrichment_factor (): Enrichment factor applied to ``stdabu``. Useful when doing multiple elements at once.
        std_relative_enrichment (): If ''True'' the abundances of all ``isotopes`` in ``stdabu`` are multiplied by
            ``std_enrichment_factor``. If ``False`` the abundances are first renormalised such that their sum = 1
            and **then** multiplied by ``std_enrichment_factor``.


    **Notes**
    The ``normiso`` will be appended to ``isotopes`` if not initially included. This is done before the enrichment
    factor calculation.

    Returns:
        dict: A dictionary containing the results of the normalisation. The dictionary at minimum contains the following attrs:

            - ``Ri``: A key array containing the eRi values for each column in ``Ri_keys``.
            - ``Ri_keys``: The numerator isotopes for each column in ``Ri_values``.
            - ``ij_keys``: Dictionaries mapping ``Ri_keys`` to the numerator-denominator keys (ij) for each column
                in ``Ri``.
            - ``Ri_keylabels``, ``Ri_keylabels_latex``: Dictionaries mapping ``Ri_keys`` to plain text and latex labels suitable
                for plotting. Consists of the ij mass keys followed by the element symbol of the numerator.
                E.g. 104/105Pd and ${}^{104/105}\\mathrm{Pd}$, where i=Pd-104 and j=Pd-105.
    """
    if msg_prefix:
        msg_prefix = f'{msg_prefix}-ratnorm'
    else:
        msg_prefix = 'ratnorm'

    if abu.dtype.names is None:
        raise ValueError(f'{msg_prefix}: ``abu`` must be a keyarray')
    if stdabu.dtype.names is None:
        raise ValueError(f'{msg_prefix}: ``stdabu`` must be a keyarray')

    if isinstance(normiso, (list, tuple)):
        if isotopes is None:
            isotopes = [None] * len(normiso)
        elif not isinstance(isotopes, (list, tuple)) or len(isotopes) != len(normiso):
            raise ValueError(f'{msg_prefix}: ``isotopes`` must be an iterable the same length as ``normiso``')

        if isinstance(enrichment_factor, (list, tuple)):
            if len(enrichment_factor) != len(normiso):
                raise ValueError(f'{msg_prefix}: ``enrichment_factor`` must be an iterable the same length as ``normiso``')
        else:
            enrichment_factor = [enrichment_factor] * len(normiso)

        if isinstance(std_enrichment_factor, (list, tuple)):
            if len(std_enrichment_factor) != len(normiso):
                raise ValueError(f'{msg_prefix}: ``solar_enrichment_factor`` must be an iterable the same length as ``normiso``')
        else:
            std_enrichment_factor = [std_enrichment_factor] * len(normiso)

    else:
        isotopes = (isotopes,)
        normiso = (normiso,)
        enrichment_factor = (enrichment_factor,)
        std_enrichment_factor = (std_enrichment_factor,)

    all_abu_up, all_abu_down = [], []
    all_solar_up, all_solar_down = [], []
    all_iso_up, all_iso_down = (), ()
    for numerators, denominator, abu_factor, solar_factor in zip(isotopes, normiso, enrichment_factor, std_enrichment_factor):
        denominator = utils.asisotope(denominator)

        if numerators is None:
            numerators = utils.get_isotopes_of_element(abu.dtype.names, denominator.element)
        else:
            numerators = utils.asisotopes(numerators)

        if denominator not in numerators:
            numerators += (denominator,)

        logger.info(f'{msg_prefix}: Normalising {numerators} to {denominator}.')

        if relative_enrichment is False:
            logger.info(f'{msg_prefix}: Applying absolute enrichment factor to model abundances. '
                        f'Setting the sum of all isotopes to {abu_factor}')
        elif abu_factor != 1:
            logger.info(f'{msg_prefix}: Applying relative enrichment factor to model abundances. '
                        f'Multiplying all isotopes by {abu_factor}.')

        if std_relative_enrichment is False:
            logger.info(f'{msg_prefix}: Applying absolute enrichment factor to standard abundances. '
                        f'Setting the sum of all isotopes to {solar_factor}')
        elif solar_factor != 1:
            logger.info(f'{msg_prefix}: Applying relative enrichment factor to standard abundances. '
                        f'Multiplying all isotopes by {solar_factor}.')

        denomi = numerators.index(denominator)

        all_iso_up += numerators
        all_iso_down += tuple(denominator for n in numerators)

        abu_up = np.array([abu[numerator] for numerator in numerators])

        if relative_enrichment is False:
            # Renormalise so that the sum of all isotopes = 1
            # This works for both ndim = 1 & 2 as long as it is done before transpose
            abu_up = abu_up / abu_up.sum(axis=0)

        abu_up = abu_up * abu_factor
        all_abu_up.append(abu_up)

        # Same isotope for all isotopes
        all_abu_down.append(np.ones(abu_up.shape) * abu_up[denomi])

        solar_up = np.array([stdabu[numerator.without_suffix()] for numerator in numerators])
        if std_relative_enrichment is False:
            # Renormalise so that the sum of all isotopes = 1
            # This works for both ndim = 1 & 2 as long as it is done before transpose
            solar_up = solar_up / solar_up.sum(axis=0)

        solar_up = solar_up * solar_factor
        all_solar_up.append(solar_up)
        all_solar_down.append(np.ones(solar_up.shape) * solar_up[denomi])

    # Joins all arrays and makes sure dimensions line up
    all_abu_up = np.atleast_2d(np.concatenate(all_abu_up, axis=0).transpose())
    all_abu_down = np.atleast_2d(np.concatenate(all_abu_down, axis=0).transpose())

    all_solar_up = np.atleast_2d(np.concatenate(all_solar_up, axis=0).transpose())
    all_solar_down = np.atleast_2d(np.concatenate(all_solar_down, axis=0).transpose())

    if dilution_factor is None or dilution_factor <= 0:
        all_smp_up = all_abu_up
        all_smp_down = all_abu_down
        df = 0
    else:
        all_smp_up = all_abu_up / dilution_factor + all_solar_up
        all_smp_down = all_abu_down / dilution_factor + all_solar_down
        df = dilution_factor

    # There is only one way to do this so no need for a separate function
    Rij = (all_smp_up/all_smp_down)/(all_solar_up/all_solar_down) - 1.0

    result = dict(Ri_values = Rij, Ri_keys=all_iso_up, Ri = utils.askeyarray(Rij, all_iso_up), dilution_factor = df)
    result['ij_keys'] = dict(zip(all_iso_up, utils.asratios([f'{n}/{d}' for n, d in zip(all_iso_up, all_iso_down)])))

    # The labels assume that both the numerator and denominator is the same element.
    result['Ri_keylabels'] = dict(zip(all_iso_up, [f'{ij.numer.mass}/{ij.denom.mass}{ij.numer.element}' for ij in result['ij_keys'].values()]))
    result['Ri_keylabels_latex'] = dict(
        zip(all_iso_up, [fr'${{}}^{{{ij.numer.mass}/{ij.denom.mass}}}\mathrm{{{ij.numer.element}}}$'
                         for ij in result['ij_keys'].values()]))

    for k, v in result.items():
        if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == 1:
            result[k] = v.squeeze()

    return utils.NamedDict(result)