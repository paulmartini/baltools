# baltools.fitbal
# ===============
#
# Routines to fit PCA components to QSOs and calculate BAL properties.
#
# Major revisions in 2025 to improve performance, clarity, and maintainability
# by vectorizing calculations and refactoring duplicated code.

import math
import warnings
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from numpy.typing import NDArray

from baltools import balconfig as bc

debug = False


# ==============================================================================
# 1. DICTIONARY AND DATA INITIALIZATION
# ==============================================================================


def initialize() -> Dict:
    """Initialize the balinfo dictionary."""
    balinfo = {
        'TROUGH_10K': 0,
        'SNR_CIV': -1.,

        'BI_CIV': 0., 'BI_CIV_ERR': 0., 'NCIV_2000': 0,
        'VMIN_CIV_2000': -1. * np.ones(bc.NBI, dtype=float),
        'VMAX_CIV_2000': -1. * np.ones(bc.NBI, dtype=float),
        'POSMIN_CIV_2000': -1. * np.ones(bc.NBI, dtype=float),
        'FMIN_CIV_2000': -1. * np.ones(bc.NBI, dtype=float),

        'AI_CIV': 0., 'AI_CIV_ERR': 0., 'NCIV_450': 0,
        'VMIN_CIV_450': -1. * np.ones(bc.NAI, dtype=float),
        'VMAX_CIV_450': -1. * np.ones(bc.NAI, dtype=float),
        'POSMIN_CIV_450': -1. * np.ones(bc.NAI, dtype=float),
        'FMIN_CIV_450': -1. * np.ones(bc.NAI, dtype=float),

        'BI_SIIV': 0., 'BI_SIIV_ERR': 0., 'NSIIV_2000': 0,
        'VMIN_SIIV_2000': -1. * np.ones(bc.NBI, dtype=float),
        'VMAX_SIIV_2000': -1. * np.ones(bc.NBI, dtype=float),
        'POSMIN_SIIV_2000': -1. * np.ones(bc.NBI, dtype=float),
        'FMIN_SIIV_2000': -1. * np.ones(bc.NBI, dtype=float),

        'AI_SIIV': 0., 'AI_SIIV_ERR': 0., 'NSIIV_450': 0,
        'VMIN_SIIV_450': -1. * np.ones(bc.NAI, dtype=float),
        'VMAX_SIIV_450': -1. * np.ones(bc.NAI, dtype=float),
        'POSMIN_SIIV_450': -1. * np.ones(bc.NAI, dtype=float),
        'FMIN_SIIV_450': -1. * np.ones(bc.NAI, dtype=float),
    }
    return balinfo


# ==============================================================================
# 2. CORE BALNICIY CALCULATION ROUTINES
# ==============================================================================


def determine_troughs(
    norm_flux: NDArray[np.float64],
    norm_sigma: NDArray[np.float64],
    speed: NDArray[np.float64],
    min_width: float,
    is_ai: bool = False
) -> Tuple[List[int], List[int]]:
    """
    Identifies absorption troughs in a normalized QSO spectrum.

    Args:
        norm_flux: Normalized flux array.
        norm_sigma: Normalized error array.
        speed: Velocity array corresponding to the flux.
        min_width: The minimum width (km/s) for a trough to be valid.
        is_ai: If True, applies an additional significance test for AI troughs.

    Returns:
        A tuple of (start_indices, end_indices) for each valid trough.
    """
    start_indices, end_indices = [], []
    expression = (1. - norm_flux / bc.CONTINUUM_THRESHOLD) - bc.ERROR_SCALING_FACTOR * norm_sigma

    if np.median(norm_flux) < np.median(norm_sigma):
        return [], []

    vel_range = 0.0
    in_trough = False
    for i in range(len(expression)):
        if expression[i] > 0:
            if not in_trough:
                start_idx = i
                in_trough = True
                vel_range = 0.0001
            else:
                if i > 0:
                    vel_range += speed[i] - speed[i - 1]
        else:
            if in_trough:
                # End of a potential trough
                if vel_range > min_width:
                    end_idx = i - 1
                    if is_ai:
                        # Apply significance test for AI troughs
                        trough_flux = norm_flux[start_idx:end_idx + 1]
                        trough_sigma = norm_sigma[start_idx:end_idx + 1]
                        mean_flux = np.mean(trough_flux)
                        # Error on the mean
                        mean_sigma = np.mean(trough_sigma) / np.sqrt(len(trough_flux))
                        # Test if the mean flux + 3-sigma is still below the continuum
                        if (1. - (mean_flux + 3.0 * mean_sigma) / bc.CONTINUUM_THRESHOLD) > 0:
                            start_indices.append(start_idx)
                            end_indices.append(end_idx)
                    else: # BI troughs do not require the extra test
                        start_indices.append(start_idx)
                        end_indices.append(end_idx)
            in_trough = False
            vel_range = 0.0

    # Handle trough extending to the end of the array
    if in_trough and vel_range > min_width:
        end_indices.append(len(expression) - 1)
        start_indices.append(start_idx)

    return start_indices, end_indices


def calculate_index(
    speed: NDArray[np.float64],
    pca_model: NDArray[np.float64],
    norm_flux: NDArray[np.float64],
    sigma: NDArray[np.float64],
    diff: float
) -> Tuple[float, float]:
    """
    Calculates a BALnicity or Absorption Index and its associated uncertainty.
    This is a vectorized implementation for performance.

    Returns:
        A tuple of (index_value, index_variance). The sqrt of variance should
        be taken after summing contributions from all troughs.
    """
    if len(speed) < 2:
        return 0., 0.

    dv = np.diff(speed)
    integrand = 1. - (norm_flux[1:] / bc.CONTINUUM_THRESHOLD)

    # Calculate the index value (the integral)
    value = np.sum(integrand * dv)

    # Correctly propagate the error (sum of weighted variances)
    # sigma_tt_sq is the variance of the normalized flux at each point
    sigma_tt_sq = sigma[1:]**2 + (diff / pca_model[1:])**2
    variance = np.sum((sigma_tt_sq / bc.CONTINUUM_THRESHOLD**2) * (dv**2))

    return (value, variance) if value > 0. else (0., 0.)


def _process_ion_line(
    idata: NDArray[np.float64],
    model: NDArray[np.float64],
    ion_lambda: float,
    ion_name: str,
    balinfo: Dict,
    verbose: bool = False
) -> None:
    """
    Helper function to find troughs and calculate BAL properties for a single ion.
    This function populates the `balinfo` dictionary directly.
    """
    balwave, balspec, balerror = idata
    speed = bc.c * (balwave - ion_lambda) / ion_lambda

    min_wave_req = ion_lambda * (1. + bc.VMIN_BAL / bc.c)
    if balwave[0] > min_wave_req:
        if verbose:
            print(f"Skipping {ion_name}: Spectrum starts at {balwave[0]:.1f}Å, "
                  f"but {min_wave_req:.1f}Å is needed.")
        return

    # Define velocity indices
    idx_min_bal = np.searchsorted(speed, bc.VMIN_BAL)
    idx_max_bal = np.searchsorted(speed, bc.VMAX_BAL)
    idx_max_bi = np.searchsorted(speed, bc.BI_VMAX)
    idx_max_ai = np.searchsorted(speed, bc.AI_VMAX)

    # --- AI Calculation ---
    speed_ai = speed[idx_min_bal:idx_max_ai]
    flux_ai = balspec[idx_min_bal:idx_max_ai]
    model_ai = model[idx_min_bal:idx_max_ai]
    error_ai = balerror[idx_min_bal:idx_max_ai]

    # Suppress known RuntimeWarnings for this block
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        norm_flux_ai = np.divide(flux_ai, model_ai, out=np.ones_like(model_ai), where=model_ai!=0)
        sigma_ai = np.divide(error_ai, model_ai, out=np.full_like(model_ai, np.inf), where=model_ai!=0)

    start_idx, end_idx = determine_troughs(norm_flux_ai, sigma_ai, speed_ai, bc.AI_MIN_WIDTH, is_ai=True)

    # Create mask to find non-trough regions for `difference` calculation
    mask_ai = np.ones_like(speed_ai, dtype=bool)
    for i in range(len(start_idx)):
        mask_ai[start_idx[i]:end_idx[i] + 1] = False
    difference = np.mean(np.abs(model_ai[mask_ai] - flux_ai[mask_ai]))

    for i in range(min(len(start_idx), bc.NAI)):
        s, e = start_idx[i], end_idx[i]
        ai, ai_var = calculate_index(speed_ai[s:e+1], model_ai[s:e+1], norm_flux_ai[s:e+1], sigma_ai[s:e+1], difference)

        balinfo[f'AI_{ion_name}'] += ai
        balinfo[f'AI_{ion_name}_ERR'] += ai_var
        balinfo[f'N{ion_name}_{int(bc.AI_MIN_WIDTH)}'] += 1

        trough_flux = norm_flux_ai[s:e+1]
        min_flux_idx = np.argmin(trough_flux)
        balinfo[f'VMAX_{ion_name}_{int(bc.AI_MIN_WIDTH)}'][i] = -speed_ai[s]
        balinfo[f'VMIN_{ion_name}_{int(bc.AI_MIN_WIDTH)}'][i] = -speed_ai[e]
        balinfo[f'FMIN_{ion_name}_{int(bc.AI_MIN_WIDTH)}'][i] = trough_flux[min_flux_idx]
        balinfo[f'POSMIN_{ion_name}_{int(bc.AI_MIN_WIDTH)}'][i] = -speed_ai[s:e+1][min_flux_idx]


    # --- BI Calculation ---
    speed_bi = speed[idx_min_bal:idx_max_bi]
    flux_bi = balspec[idx_min_bal:idx_max_bi]
    model_bi = model[idx_min_bal:idx_max_bi]
    error_bi = balerror[idx_min_bal:idx_max_bi]

    # Suppress known RuntimeWarnings for this block
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        norm_flux_bi = np.divide(flux_bi, model_bi, out=np.ones_like(model_bi), where=model_bi!=0)
        sigma_bi = np.divide(error_bi, model_bi, out=np.full_like(model_bi, np.inf), where=model_bi!=0)

    start_idx, end_idx = determine_troughs(norm_flux_bi, sigma_bi, speed_bi, bc.BI_MIN_WIDTH)

    for i in range(min(len(start_idx), bc.NBI)):
        s, e = start_idx[i], end_idx[i]
        bi, bi_var = calculate_index(speed_bi[s:e+1], model_bi[s:e+1], norm_flux_bi[s:e+1], sigma_bi[s:e+1], difference)

        balinfo[f'BI_{ion_name}'] += bi
        balinfo[f'BI_{ion_name}_ERR'] += bi_var
        balinfo[f'N{ion_name}_{int(bc.BI_MIN_WIDTH)}'] += 1

        trough_flux = norm_flux_bi[s:e+1]
        min_flux_idx = np.argmin(trough_flux)
        balinfo[f'VMAX_{ion_name}_{int(bc.BI_MIN_WIDTH)}'][i] = -speed_bi[s]
        balinfo[f'VMIN_{ion_name}_{int(bc.BI_MIN_WIDTH)}'][i] = -speed_bi[e]
        balinfo[f'FMIN_{ion_name}_{int(bc.BI_MIN_WIDTH)}'][i] = trough_flux[min_flux_idx]
        balinfo[f'POSMIN_{ion_name}_{int(bc.BI_MIN_WIDTH)}'][i] = -speed_bi[s:e+1][min_flux_idx]


def calculatebalinfo(idata: NDArray[np.float64], model: NDArray[np.float64], verbose: bool = False) -> Dict:
    """
    Calculate BAL quantities by processing CIV and SiIV lines.

    This function now delegates the core logic to the `_process_ion_line`
    helper function to avoid code duplication.
    """
    balinfo = initialize()

    # Calculate median SNR over the CIV AI range
    balwave, balspec, balerror = idata
    speed_civ = bc.c * (balwave - bc.lambdaCIV) / bc.lambdaCIV
    idx_min_bal = np.searchsorted(speed_civ, bc.VMIN_BAL)
    idx_max_ai = np.searchsorted(speed_civ, bc.AI_VMAX)
    if idx_max_ai > idx_min_bal:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            balinfo['SNR_CIV'] = np.median(balspec[idx_min_bal:idx_max_ai] / balerror[idx_min_bal:idx_max_ai])

    # Process each ion
    _process_ion_line(idata, model, bc.lambdaCIV, 'CIV', balinfo, verbose)
    _process_ion_line(idata, model, bc.lambdaSiIV, 'SIIV', balinfo, verbose)

    # Check for high-velocity troughs
    if balinfo['VMAX_CIV_450'][0] > 10000.:
        balinfo['TROUGH_10K'] = 1

    # Finalize errors by taking the square root of the summed variances
    for key in ['BI_CIV_ERR', 'AI_CIV_ERR', 'BI_SIIV_ERR', 'AI_SIIV_ERR']:
        balinfo[key] = np.sqrt(balinfo[key])

    return balinfo


# ==============================================================================
# 3. PCA FITTING AND ORCHESTRATION
# ==============================================================================


def createpcatemplate(pcaeigen: np.void, pcacoeffs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Create a PCA template from eigenvectors and coefficients."""
    outspec = np.zeros_like(pcaeigen['WAVE'])
    for i, name in enumerate(pcaeigen.dtype.names):
        if name == 'WAVE':
            continue
        outspec += pcacoeffs[i-1] * pcaeigen[name]
    return outspec


def fitpca(
    idata: NDArray[np.float64],
    ipca: NDArray[np.float64],
    mmask: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """Fit PCA components to a masked QSO spectrum."""
    balspec = idata[1]
    balerror = idata[2]
    x0 = np.ones(len(ipca))

    def chisqfunc(alpha: NDArray[np.float64]) -> float:
        m_lambda = np.dot(alpha, ipca)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            chisq = np.sum(((m_lambda[mmask] - balspec[mmask]) / balerror[mmask])**2)
        return chisq

    result = opt.minimize(chisqfunc, x0, method='Nelder-Mead')
    chisq = result.fun
    rchisq = chisq / (np.sum(mmask) - len(x0))
    return np.append(result.x, [rchisq])



def baltomask(balinfo: Dict, wave: NDArray[np.float64], baltype: str = 'AI') -> NDArray[np.bool_]:
    """Create a boolean mask of BAL features for a given wavelength array."""
    balmask = np.ones_like(wave, dtype=bool)
    if baltype == 'AI':
        vmin_civ, vmax_civ = balinfo['VMIN_CIV_450'], balinfo['VMAX_CIV_450']
        vmin_siiv, vmax_siiv = balinfo['VMIN_SIIV_450'], balinfo['VMAX_SIIV_450']
    else: # BI
        vmin_civ, vmax_civ = balinfo['VMIN_CIV_2000'], balinfo['VMAX_CIV_2000']
        vmin_siiv, vmax_siiv = balinfo['VMIN_SIIV_2000'], balinfo['VMAX_SIIV_2000']

    for i in range(len(vmax_civ)):
        if vmin_civ[i] > 0.:
            w1 = bc.lambdaCIV * (1. - vmax_civ[i] / bc.c)
            w2 = bc.lambdaCIV * (1. - vmin_civ[i] / bc.c)
            idx1, idx2 = find_nearest(wave, w1), find_nearest(wave, w2)
            balmask[idx1:idx2 + 1] = False

    for i in range(len(vmax_siiv)):
        if vmin_siiv[i] > 0.:
            w1 = bc.lambdaSiIV * (1. - vmax_siiv[i] / bc.c)
            w2 = bc.lambdaSiIV * (1. - vmin_siiv[i] / bc.c)
            idx1, idx2 = find_nearest(wave, w1), find_nearest(wave, w2)
            balmask[idx1:idx2 + 1] = False

    return balmask


def calcbalparams(
    qsospec: np.void,
    pcaeigen: np.void,
    zspec: float,
    maxiter: int = 10,
    verbose: bool = False
) -> Tuple[Dict, NDArray[np.float64], NDArray[np.bool_]]:
    """
    Iteratively fit BAL features and return BAL properties, PCA fit, and mask.
    """
    try:
        wave_rest = qsospec['wave'] / (1 + zspec)
        is_desi_spec = True
    except (ValueError, KeyError):
        wave_rest = 10**qsospec['loglam'] / (1 + zspec)
        is_desi_spec = False

    if wave_rest[-1] < bc.lambdaCIV or wave_rest[0] > bc.lambdaCIV:
        raise RuntimeError("Spectrum does not cover the CIV line region.")

    pca_eigen_interp = [np.interp(wave_rest, pcaeigen['WAVE'], pcaeigen[name])
                        for name in pcaeigen.dtype.names if name != 'WAVE']

    idx1 = find_nearest(wave_rest, bc.BAL_LAMBDA_MIN) + 1 if bc.BAL_LAMBDA_MIN > wave_rest[0] else 0
    idx2 = find_nearest(wave_rest, bc.BAL_LAMBDA_MAX) - 1 if bc.BAL_LAMBDA_MAX < wave_rest[-1] else len(wave_rest) - 1

    ipca = np.array([comp[idx1:idx2] for comp in pca_eigen_interp])
    balwave = wave_rest[idx1:idx2]
    balspec = qsospec['flux'][idx1:idx2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        balerror = np.nan_to_num(np.sqrt(1. / qsospec['ivar'][idx1:idx2]))
    idata = np.array([balwave, balspec, balerror])

    calcmask = np.ones(len(balwave), dtype=bool)
    nmasked_prev = -1
    itr = 0
    while itr < maxiter and np.sum(calcmask) != nmasked_prev:
        nmasked_prev = np.sum(calcmask)
        calcpcaout = fitpca(idata, ipca, calcmask)
        calcpcacoeffs = calcpcaout[:-1]
        calcmodel = np.dot(calcpcacoeffs, ipca)
        calcinfo = calculatebalinfo(idata, calcmodel, verbose=verbose)
        calcmask = baltomask(calcinfo, balwave)
        itr += 1

    sdsschi2 = 9e9
    if not is_desi_spec:
        sdsschi2 = sdsschisq(qsospec, zspec)
        if calcpcaout[-1] > 1.5 * sdsschi2:
            if verbose:
                print("Warning: SDSS model is a better fit. Using it for BAL params.")
            calcinfo = calculatebalinfo(idata, qsospec['model'][idx1:idx2], verbose)
            calcmask = baltomask(calcinfo, balwave)
            calcpcaout[:len(ipca)] = -1. # Invalidate PCA coeffs

    calcpcaout = np.append(calcpcaout, sdsschi2)
    return calcinfo, calcpcaout, calcmask


# ==============================================================================
# 4. UTILITY FUNCTIONS
# ==============================================================================


def qsocatsearch(
    catalog: np.void,
    thing_id: int = -9999,
    sdssname: str = "",
    pmf: Optional[List[int]] = None
) -> int:
    """Retrieve index of a QSO from a catalog using various identifiers."""
    if thing_id >= 0:
        indices = np.where(catalog['THING_ID'] == thing_id)[0]
    elif len(sdssname) > 0:
        indices = np.where(catalog['SDSS_NAME'] == sdssname)[0]
    elif pmf and len(pmf) == 3:
        mask = (catalog['PLATE'] == pmf[0]) & (catalog['MJD'] == pmf[1]) & (catalog['FIBERID'] == pmf[2])
        indices = np.where(mask)[0]
    else:
        raise ValueError("Specify THING_ID, SDSS_NAME, or PLATE/MJD/FIBERID")

    if len(indices) == 0:
        raise ValueError("QSO not found in catalog.")
    return indices[0]


def find_nearest(array: NDArray[np.float64], value: float) -> int:
    """Find the index of the nearest value in a sorted array."""
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx - 1
    else:
        return idx

def sdsschisq(qsospec: np.void, zspec: float) -> float:
    """Evaluate chi-squared of the default SDSS model over the PCA wavelength range."""
    wave_rest = 10**qsospec['loglam'] / (1 + zspec)
    idx1 = find_nearest(wave_rest, bc.BAL_LAMBDA_MIN) + 1 if bc.BAL_LAMBDA_MIN > wave_rest[0] else 0
    idx2 = find_nearest(wave_rest, bc.BAL_LAMBDA_MAX) - 1 if bc.BAL_LAMBDA_MAX < wave_rest[-1] else len(wave_rest) - 1

    balspec = qsospec['flux'][idx1:idx2]
    balmodel = qsospec['model'][idx1:idx2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        balerror = np.nan_to_num(np.sqrt(1. / qsospec['ivar'][idx1:idx2]))

    def chisqfunc(alpha: float) -> float:
        m_lambda = alpha * balmodel
        residuals = (m_lambda - balspec) / balerror
        return np.sum(residuals**2)

    result = opt.minimize(chisqfunc, np.array([1]), method='Nelder-Mead')
    chisq = result.fun
    rchisq = chisq / (len(balspec) - 1)
    return rchisq

