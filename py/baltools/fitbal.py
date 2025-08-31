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

import numpy as np
import scipy.optimize as opt
from numpy.typing import NDArray

from baltools import balconfig as bc

debug = False
if debug:
    import matplotlib.pyplot as plt

# ==============================================================================
# 1. DICTIONARY AND DATA INITIALIZATION
# ==============================================================================


def initialize() -> Dict:
    """Initialize the balinfo dictionary."""
    # This function is the same as the original
    balinfo = {
        'TROUGH_10K': 0, 'SNR_CIV': -1.,
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
    Identifies troughs, replicating the original script's logic for
    single-pixel gap bridging and significance testing.
    """
    start_indices, end_indices = [], []
    expression = (1. - norm_flux / bc.CONTINUUM_THRESHOLD) - bc.ERROR_SCALING_FACTOR * norm_sigma

    if np.median(norm_flux) < np.median(norm_sigma):
        return [], []

    vel_range = 0.0
    start_idx = 0
    in_trough = False
    i = 0
    while i < len(expression):
        # In an absorption feature
        if expression[i] > 0:
            if not in_trough:
                start_idx = i
                in_trough = True
                vel_range = 0.0001
            elif i > 0:
                vel_range += speed[i] - speed[i - 1]
            i += 1
        # Not in absorption
        else:
            # Check for a single-pixel bridge
            can_bridge = (in_trough and i + 1 < len(expression) and expression[i+1] > 0)
            if is_ai and can_bridge:
                # AI logic is stricter, requires >2 pixels before the gap
                if not (i > 1 and expression[i-1] > 0 and expression[i-2] > 0):
                    can_bridge = False
            
            if can_bridge:
                vel_range += speed[i] - speed[i - 1] # Bridge over the non-absorbed pixel
                i += 1
            else:
                # Proper end of a potential trough
                if in_trough and vel_range > min_width:
                    end_idx = i - 1
                    if is_ai:
                        # Original significance test for AI troughs
                        trough_flux = norm_flux[start_idx:end_idx + 1]
                        trough_sigma = norm_sigma[start_idx:end_idx + 1]
                        if trough_flux.size > 0:
                            mean_flux = np.mean(trough_flux)
                            mean_sigma = np.mean(trough_sigma) / np.sqrt(trough_flux.size)
                            if (1. - (mean_flux + 3.0 * mean_sigma) / bc.CONTINUUM_THRESHOLD) > 0:
                                start_indices.append(start_idx)
                                end_indices.append(end_idx)
                    else: # BI troughs have no extra test
                        start_indices.append(start_idx)
                        end_indices.append(end_idx)
                in_trough = False
                vel_range = 0.0
                i += 1
    
    if in_trough and vel_range > min_width:
        start_indices.append(start_idx)
        end_indices.append(len(expression) - 1)
        
    return start_indices, end_indices


def calculate_index(
    speed: NDArray[np.float64],
    pca_model: NDArray[np.float64],
    norm_flux: NDArray[np.float64],
    sigma: NDArray[np.float64],
    diff: float,
    min_width_for_sum: float
) -> Tuple[float, float]:
    """
    Calculates BALnicity/Absorption Index, replicating the original logic
    of only summing after a minimum velocity width is reached and using the
    original error propagation formula.
    """
    if len(speed) < 2:
        return 0., 0.

    integrand = 1. - (norm_flux / bc.CONTINUUM_THRESHOLD)
    dv = np.diff(speed)
    
    # Replicate original logic: find where the cumulative velocity range
    # first exceeds the minimum width required for summation.
    v_range = np.cumsum(np.insert(dv, 0, 0))
    integration_mask = v_range[:-1] >= min_width_for_sum

    value = np.sum(integrand[1:][integration_mask] * dv[integration_mask])

    # Suppress expected overflow warnings from squaring infinite sigma values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        
        # Replicate original error propagation (sum of variance * dv)
        sigma_tt_sq = sigma[:-1]**2 + (diff / pca_model[:-1])**2
        variance_term = (sigma_tt_sq / bc.CONTINUUM_THRESHOLD**2) * dv
        val_error = np.sum(variance_term[integration_mask])
    
    return (value, val_error) if value > 0. else (0., 0.)


def calculatebalinfo(idata: NDArray[np.float64], model: NDArray[np.float64], verbose: bool = False) -> Dict:
    """
    Calculates BAL quantities. Updated to use modernized helper functions and safe 
    division.
    """
    balwave, balspec, balerror = idata
    balinfo = initialize()

    # --- CIV Calculations ---
    speed_civ = bc.c * (balwave - bc.lambdaCIV) / bc.lambdaCIV
    idx_min_bal = np.searchsorted(speed_civ, bc.VMIN_BAL)
    idx_max_bi = np.searchsorted(speed_civ, -3000.)
    idx_max_ai = np.searchsorted(speed_civ, 0.)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if idx_min_bal < idx_max_ai:
            snr_slice = balspec[idx_min_bal:idx_max_ai] / balerror[idx_min_bal:idx_max_ai]
            balinfo['SNR_CIV'] = np.median(snr_slice[np.isfinite(snr_slice)])

        # Normalize flux/error for CIV windows using safe division
        model_ai_civ = model[idx_min_bal:idx_max_ai]
        norm_flux_ai = np.divide(balspec[idx_min_bal:idx_max_ai], model_ai_civ, out=np.ones_like(model_ai_civ), where=model_ai_civ!=0)
        sigma_ai = np.divide(balerror[idx_min_bal:idx_max_ai], model_ai_civ, out=np.full_like(model_ai_civ, np.inf), where=model_ai_civ!=0)
        
        model_bi_civ = model[idx_min_bal:idx_max_bi]
        norm_flux_bi = np.divide(balspec[idx_min_bal:idx_max_bi], model_bi_civ, out=np.ones_like(model_bi_civ), where=model_bi_civ!=0)
        sigma_bi = np.divide(balerror[idx_min_bal:idx_max_bi], model_bi_civ, out=np.full_like(model_bi_civ, np.inf), where=model_bi_civ!=0)

    # Step 1: Find CIV AI troughs
    start_ai_civ, end_ai_civ = determine_troughs(norm_flux_ai, sigma_ai, speed_civ[idx_min_bal:idx_max_ai], bc.AI_MIN_WIDTH, is_ai=True)
    
    # Step 2: Calculate the single 'difference' metric
    continuum_mask = np.ones_like(norm_flux_ai, dtype=bool)
    for s, e in zip(start_ai_civ, end_ai_civ):
        continuum_mask[s:e+1] = False
    
    difference = 0.0
    if np.any(continuum_mask):
        difference = np.mean(np.abs(model[idx_min_bal:idx_max_ai][continuum_mask] - balspec[idx_min_bal:idx_max_ai][continuum_mask]))
    
    # Step 3: Measure CIV AI troughs using the calculated difference
    for i in range(min(len(start_ai_civ), bc.NAI)):
        s, e = start_ai_civ[i], end_ai_civ[i]
        ai, ai_err = calculate_index(speed_civ[idx_min_bal:idx_max_ai][s:e+1], model[idx_min_bal:idx_max_ai][s:e+1], norm_flux_ai[s:e+1], sigma_ai[s:e+1], difference, bc.AI_MIN_WIDTH)
        balinfo['AI_CIV'] += ai
        balinfo['AI_CIV_ERR'] += ai_err
        balinfo['NCIV_450'] += 1
        balinfo['VMAX_CIV_450'][i] = -speed_civ[idx_min_bal:idx_max_ai][s]
        balinfo['VMIN_CIV_450'][i] = -speed_civ[idx_min_bal:idx_max_ai][e]
        min_flux = norm_flux_ai[s:e+1].min()
        balinfo['FMIN_CIV_450'][i] = min_flux
        min_flux_idx = np.where(norm_flux_ai[s:e+1] == min_flux)[0][0]
        balinfo['POSMIN_CIV_450'][i] = -speed_civ[idx_min_bal:idx_max_ai][s:e+1][min_flux_idx]
    
    if balinfo['VMAX_CIV_450'][0] > 10000.: balinfo['TROUGH_10K'] = 1

    # Step 4: Find and Measure CIV BI troughs, using the same difference
    start_bi_civ, end_bi_civ = determine_troughs(norm_flux_bi, sigma_bi, speed_civ[idx_min_bal:idx_max_bi], bc.BI_MIN_WIDTH, is_ai=False)
    for i in range(min(len(start_bi_civ), bc.NBI)):
        s, e = start_bi_civ[i], end_bi_civ[i]
        bi, bi_err = calculate_index(speed_civ[idx_min_bal:idx_max_bi][s:e+1], model[idx_min_bal:idx_max_bi][s:e+1], norm_flux_bi[s:e+1], sigma_bi[s:e+1], difference, bc.BI_MIN_WIDTH)
        balinfo['BI_CIV'] += bi
        balinfo['BI_CIV_ERR'] += bi_err
        balinfo['NCIV_2000'] += 1
        balinfo['VMAX_CIV_2000'][i] = -speed_civ[idx_min_bal:idx_max_bi][s]
        balinfo['VMIN_CIV_2000'][i] = -speed_civ[idx_min_bal:idx_max_bi][e]
        min_flux = norm_flux_bi[s:e+1].min()
        balinfo['FMIN_CIV_2000'][i] = min_flux
        min_flux_idx = np.where(norm_flux_bi[s:e+1] == min_flux)[0][0]
        balinfo['POSMIN_CIV_2000'][i] = -speed_civ[idx_min_bal:idx_max_bi][s:e+1][min_flux_idx]

    # --- SiIV Calculations (if in bandpass) ---
    if balwave[0] <= bc.lambdaSiIV * (1. + bc.VMIN_BAL / bc.c):
        speed_siiv = bc.c * (balwave - bc.lambdaSiIV) / bc.lambdaSiIV
        idx_min_bal_siiv = np.searchsorted(speed_siiv, bc.VMIN_BAL)
        idx_max_bi_siiv = np.searchsorted(speed_siiv, -3000.)
        idx_max_ai_siiv = np.searchsorted(speed_siiv, 0.)

        # Suppress warnings for SiIV normalization as well
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            model_ai_siiv = model[idx_min_bal_siiv:idx_max_ai_siiv]
            norm_flux_ai_siiv = np.divide(balspec[idx_min_bal_siiv:idx_max_ai_siiv], model_ai_siiv, out=np.ones_like(model_ai_siiv), where=model_ai_siiv!=0)
            sigma_ai_siiv = np.divide(balerror[idx_min_bal_siiv:idx_max_ai_siiv], model_ai_siiv, out=np.full_like(model_ai_siiv, np.inf), where=model_ai_siiv!=0)
            
            model_bi_siiv = model[idx_min_bal_siiv:idx_max_bi_siiv]
            norm_flux_bi_siiv = np.divide(balspec[idx_min_bal_siiv:idx_max_bi_siiv], model_bi_siiv, out=np.ones_like(model_bi_siiv), where=model_bi_siiv!=0)
            sigma_bi_siiv = np.divide(balerror[idx_min_bal_siiv:idx_max_bi_siiv], model_bi_siiv, out=np.full_like(model_bi_siiv, np.inf), where=model_bi_siiv!=0)

        # Find and Measure SiIV AI troughs, using CIV difference
        start_ai_siiv, end_ai_siiv = determine_troughs(norm_flux_ai_siiv, sigma_ai_siiv, speed_siiv[idx_min_bal_siiv:idx_max_ai_siiv], bc.AI_MIN_WIDTH, is_ai=True)
        for i in range(min(len(start_ai_siiv), bc.NAI)):
            s, e = start_ai_siiv[i], end_ai_siiv[i]
            ai, ai_err = calculate_index(speed_siiv[idx_min_bal_siiv:idx_max_ai_siiv][s:e+1], model[idx_min_bal_siiv:idx_max_ai_siiv][s:e+1], norm_flux_ai_siiv[s:e+1], sigma_ai_siiv[s:e+1], difference, bc.AI_MIN_WIDTH)
            balinfo['AI_SIIV'] += ai
            balinfo['AI_SIIV_ERR'] += ai_err
            balinfo['NSIIV_450'] += 1
            balinfo['VMAX_SIIV_450'][i] = -speed_siiv[idx_min_bal_siiv:idx_max_ai_siiv][s]
            balinfo['VMIN_SIIV_450'][i] = -speed_siiv[idx_min_bal_siiv:idx_max_ai_siiv][e]
            min_flux = norm_flux_ai_siiv[s:e+1].min()
            balinfo['FMIN_SIIV_450'][i] = min_flux
            min_flux_idx = np.where(norm_flux_ai_siiv[s:e+1] == min_flux)[0][0]
            balinfo['POSMIN_SIIV_450'][i] = -speed_siiv[idx_min_bal_siiv:idx_max_ai_siiv][s:e+1][min_flux_idx]
            
        # Find and Measure SiIV BI troughs, using CIV difference
        start_bi_siiv, end_bi_siiv = determine_troughs(norm_flux_bi_siiv, sigma_bi_siiv, speed_siiv[idx_min_bal_siiv:idx_max_bi_siiv], bc.BI_MIN_WIDTH, is_ai=False)
        for i in range(min(len(start_bi_siiv), bc.NBI)):
            s, e = start_bi_siiv[i], end_bi_siiv[i]
            bi, bi_err = calculate_index(speed_siiv[idx_min_bal_siiv:idx_max_bi_siiv][s:e+1], model[idx_min_bal_siiv:idx_max_bi_siiv][s:e+1], norm_flux_bi_siiv[s:e+1], sigma_bi_siiv[s:e+1], difference, bc.BI_MIN_WIDTH)
            balinfo['BI_SIIV'] += bi
            balinfo['BI_SIIV_ERR'] += bi_err
            balinfo['NSIIV_2000'] += 1
            balinfo['VMAX_SIIV_2000'][i] = -speed_siiv[idx_min_bal_siiv:idx_max_bi_siiv][s]
            balinfo['VMIN_SIIV_2000'][i] = -speed_siiv[idx_min_bal_siiv:idx_max_bi_siiv][e]
            min_flux = norm_flux_bi_siiv[s:e+1].min()
            balinfo['FMIN_SIIV_2000'][i] = min_flux
            min_flux_idx = np.where(norm_flux_bi_siiv[s:e+1] == min_flux)[0][0]
            balinfo['POSMIN_SIIV_2000'][i] = -speed_siiv[idx_min_bal_siiv:idx_max_bi_siiv][s:e+1][min_flux_idx]

    # Final sqrt on error terms
    for key in balinfo:
        if '_ERR' in key:
            balinfo[key] = np.sqrt(balinfo[key]) if balinfo[key] > 0 else 0.0

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

