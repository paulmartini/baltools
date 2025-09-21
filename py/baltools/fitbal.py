"""

baltools.fitbal
===============

Routines to fit PCA components to QSOs and calculate BAL properties


"""

import math
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import scipy.optimize as opt
from numpy.typing import NDArray

from baltools import balconfig as bc

debug = False


# ==============================================================================
# 1. DICTIONARY AND DATA INITIALIZATION
# ==============================================================================


def initialize() -> Dict:
    """
    Initialize the balinfo dictionary.
    """

    balinfo = {
        'TROUGH_10K': 0, 'SNR_CIV': -1.,
        'SNR_REDSIDE': -1., 'SNR_FOREST': -1.,
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


def determine_trough_BI(norm_flux, norm_sigma, speed_c):
    '''
    Identify BI troughs in a QSO spectrum

    Parameters
    ----------
    norm_flux : 1-d float array
        QSO  / PCA fit
    norm_sigma : 1-d float array
        uncertainty / PCA fit
    speed_c : 1-d float array
        velocity range for BI that corresponds to norm_flux, norm_sigma

    Returns
    -------
    Start_indx : 1-d int array
        starting indices for troughs
    End_indx : 1-d int array
        final indices for troughs
    '''

    Start_indx = []
    End_indx = []
    vel_range = 0
#    expression = 1 - (norm_flux)/0.9
    expression = (1. - (norm_flux)/0.9 ) - 0.5*norm_sigma  # Test adding error

    # Don't do calculation if SNR is too low
    if np.median(norm_flux) < np.median(norm_sigma): 
        return Start_indx, End_indx

    # Step through the normalized flux 
    for i in range(len(norm_flux)):
        if expression[i] > 0:
            if i == len(expression) - 1:
                vel_range = vel_range + speed_c[i] - speed_c[i-1]
                if vel_range > 2000.:
                    Start_indx.append(start_indx)
                    End_indx.append(i)
            if vel_range == 0:
                start_indx = i
                vel_range = 0.0001
            else:
                vel_range = vel_range + speed_c[i] - speed_c[i-1]
        else:
            # allow for a single noise spike
            if i < (len(expression)-2) and (expression[i+1] > 0):
                if vel_range != 0:
                    vel_range = vel_range + speed_c[i] - speed_c[i-1]
            else:
                if vel_range > 2000.:
                    Start_indx.append(start_indx)
                    End_indx.append(i-1)
                vel_range = 0

    # Hack to make sure there are not too many troughs
    if len(Start_indx) > bc.NBI: 
        print("Warning: len(Start_indx) > {0}, so truncated".format(bc.NBI))
        Start_indx = Start_indx[:bc.NBI]
        End_indx = End_indx[:bc.NBI]

    return Start_indx, End_indx


def determine_trough_AI(norm_flux, norm_sigma, speed_c):
    '''
    Identify AI troughs in a QSO spectrum

    Parameters
    ----------
    norm_flux : 1-d float array
        QSO  / PCA fit
    norm_sigma : 1-d float array
        uncertainty / PCA fit
    speed_c : 1-d float array
        velocity range for AI that corresponds to norm_flux, norm_sigma

    Returns
    -------
    Start_indx : 1-d int array
        starting indices for troughs
    End_indx : 1-d int array
        final indices for troughs
    '''

    Start_indx = []
    End_indx = []
    vel_range = 0
#    expression = (1. - (norm_flux)/0.9 )
    expression = (1. - (norm_flux)/0.9 ) - 0.5*norm_sigma # Test adding error
    
    # Don't do calculation if SNR is too low
    if np.median(norm_flux) < np.median(norm_sigma): 
        return Start_indx, End_indx

    # Step through the normalized flux
    for i in range(len(norm_flux)):
        if expression[i] > 0:  # enough below continuum to be a trough
            if i == len(expression) - 1:  # true when at the end of the range
                vel_range = vel_range + speed_c[i] - speed_c[i-1]
                if vel_range > 450.:  # record if broad enough
                    Start_indx.append(start_indx)
                    End_indx.append(i)
            if vel_range == 0: # start counting
                start_indx = i
                vel_range = 0.0001
            else:  # in the midst of a trough, increment vel_range
                vel_range = vel_range + speed_c[i] - speed_c[i-1]
        else:
            # allow for a single noise spike (more demanding than BI)
            if i < (len(expression)-2) and (expression[i+1] > 0)\
                    and expression[i-1] > 0 and expression[i-2] > 0:
                if vel_range != 0:
                    vel_range = vel_range + speed_c[i] - speed_c[i-1]
            else: # end of potential trough
                if vel_range > 450.: # see if trough is wide enough
                    # if so, determine if it is significant
                    meanflux = np.mean(norm_flux[start_indx:i-1])
                    meansigma = np.mean(norm_sigma[start_indx:i-1])/np.sqrt(i-1-start_indx)
                    meantest = (1. - (meanflux + 3.0*meansigma)/0.9)
                    if meantest > 0.: # store if it was significant
                        Start_indx.append(start_indx)
                        End_indx.append(i-1)

                vel_range = 0

    # Hack to make sure there are not too many
    if len(Start_indx) > bc.NAI: 
        print("Warning: len(Start_indx) > {0}, so truncated".format(bc.NAI))
        Start_indx = Start_indx[:bc.NAI]
        End_indx = End_indx[:bc.NAI]

    if debug: 
        print("determine_trough_AI(): Start, End = ", Start_indx, End_indx)
    return Start_indx, End_indx


def calculate_Index(speed, pca, norm_flux, sigma, ai_or_bi, diff):
    '''
    Calculate AI or BI and the associated uncertainty
    
    Parameters
    ----------
    speed: 1-d float array
        wavelength converted to velocity relative to line center
    pca : 1-d float array
        pca fit to QSO spectrum
    norm_flux : 1-d float array
        normalized QSO flux
    sigma : 1-d float array
        uncertainty in normalized flux
    ai_or_bi : float
        either 450 or 2000 (km/s) 
    diff : 1-d float array
        model - QSO spectrum
    
    Returns
    -------
    value : float
        value of AI or BI
    valerror : float
        error in value
    '''
    
    value = 0.
    vrange = 0.
    valerror = 0.
    para = 1 - (norm_flux)/0.9
    for i in range(len(speed)-1):
        vrange = vrange + speed[i+1] - speed[i]
        if vrange >= ai_or_bi:
            sigma_tt = sigma[i]**2+(diff/pca[i])**2
            value = value + (speed[i+1]-speed[i])*para[i+1]
            valerror = valerror + (sigma_tt/(0.9**2))*(speed[i+1] - speed[i])
    if value < 0.:
        value = 0. 
        valerror = 0.
    return(value, valerror)


def calculatebalinfo(idata, model, verbose=False): 
    '''
    Calculate BAL quantities

    Parameters
    ----------
    idata : 2-d float array
        balwave, flux, error of SDSS spectrum region to fit
    model : 1-D float array
        model fit to QSO continuum (dispersion should match idata)
    verbose : bool (optional)
        display debugging information

    Returns
    -------
    balinfo : dict
        dictionary of BAL parameters
    '''

    # PM Note: Formerly called calculatedr14

    balwave = idata[0]
    balspec = idata[1]
    balerror = idata[2]
    sigma_origin = balerror

    if debug: 
        plt.figure(figsize=(8,6))
        plt.plot(balwave, balspec) 
        plt.plot(balwave, model) 
        plt.plot(balwave, balerror)

    # PCA template fit to QSO
    model_1_origin = model

    # Initialize the dictionary 
    balinfo = initialize() 

    # Wavelength range for CIV BAL search
    # convert balwave to velocity relative to CIV
    speed = bc.c * ((balwave - bc.lambdaCIV)/bc.lambdaCIV)
    speed_plot = speed
    speed_min = np.where(speed >= bc.VMIN_BAL)[0][0]
    plotmin = speed_min # Index of about -25000. 
    # plotmax = plotmin + 500
    plotmax = np.where(speed >= bc.VMAX_BAL)[0][0]   # Changed 29 Mar 2020 by PM
    speed_c = speed[plotmin:plotmax]
    speed_max_bi = np.where(speed >= -3000)[0][0]
    speed_max_ai = np.where(speed >= 0)[0][0]
    speed_c_bi = speed[speed_min:speed_max_bi]
    speed_c_ai = speed[speed_min:speed_max_ai]

    # Produce spectra for AI, BI determination that are normalized by the model
    #normal_flux_bi = balspec[plotmin:speed_max_bi] / model_1_origin[plotmin:speed_max_bi]
    #normal_flux_ai = balspec[plotmin:speed_max_ai] / model_1_origin[plotmin:speed_max_ai]
    model_1_origin_safe = np.clip(model_1_origin, 1e-10, None)
    normal_flux_bi = np.divide(balspec[plotmin:speed_max_bi], model_1_origin_safe[plotmin:speed_max_bi], out=np.full_like(model_1_origin_safe[plotmin:speed_max_bi], np.inf), where=model_1_origin_safe[plotmin:speed_max_bi] !=0)
    normal_flux_ai = np.divide(balspec[plotmin:speed_max_ai], model_1_origin_safe[plotmin:speed_max_ai], out=np.full_like(model_1_origin_safe[plotmin:speed_max_ai], np.inf), where=model_1_origin_safe[plotmin:speed_max_ai] !=0)
    #sigma_bi = sigma_origin[plotmin:speed_max_bi] / model_1_origin[plotmin:speed_max_bi]
    #sigma_ai = sigma_origin[plotmin:speed_max_ai] / model_1_origin[plotmin:speed_max_ai]
    sigma_bi = np.divide(sigma_origin[plotmin:speed_max_bi], model_1_origin_safe[plotmin:speed_max_bi], out=np.full_like(model_1_origin_safe[plotmin:speed_max_bi], np.inf), where=model_1_origin_safe[plotmin:speed_max_bi] !=0)
    sigma_ai = np.divide(sigma_origin[plotmin:speed_max_ai], model_1_origin_safe[plotmin:speed_max_ai], out=np.full_like(model_1_origin_safe[plotmin:speed_max_ai], np.inf), where=model_1_origin_safe[plotmin:speed_max_ai] !=0)

    # Compute the median SNR over the range from -25000 to 0 km/s
    # balinfo['SNR_CIV'] = np.median( balspec[plotmin:speed_max_ai] / np.sqrt( sigma_origin[plotmin:speed_max_ai] ) ) 
    balinfo['SNR_CIV'] = np.median( balspec[plotmin:speed_max_ai] / ( sigma_origin[plotmin:speed_max_ai] ) ) 

    if verbose:
        print("calculatebalinfo: CIV -- ") 
        print("calculatebalinfo: ", balwave[plotmin], speed_c_bi[0], normal_flux_bi[0], sigma_bi[0])
        print("calculatebalinfo: ", balwave[speed_max_bi], speed_c_bi[-1], normal_flux_bi[-1], sigma_bi[-1])
        print("calculatebalinfo: medians are", np.median(normal_flux_ai), np.median(sigma_ai))

    if debug: 
        plt.figure(figsize=(8,6))
        plt.plot(speed_c_bi, normal_flux_bi)
        plt.plot(speed_c_bi, sigma_bi)

    # Determine CIV AI troughs if median continuum level is greater than the uncertainty
    if np.median(balspec[plotmin:speed_max_ai]) > np.median(sigma_origin[plotmin:speed_max_ai]): 
        if verbose: 
            print("len(normal_flux_ai) = ", len(normal_flux_ai))
        start_indx, end_indx = determine_trough_AI(normal_flux_ai, sigma_ai, speed_c_ai)
    else: 
        start_indx = []
        end_indx = []
        if verbose: 
            print("Too noisy to calculate AI")

    # Create a mask that identifies pixels not in troughs 
    difference_mask = np.ones(len(speed_plot[plotmin:plotmax]), dtype=bool)
    if debug: 
        print("len(difference_mask) = ", len(difference_mask))
        print("len(speed_plot) = ", len(speed_plot))
        print("len(speed_plot[plotmin:plotmax]) = ", len(speed_plot[plotmin:plotmax]))
        print("plotmin, plotmax, start_indx, end_indx = ", plotmin, plotmax, start_indx, end_indx)

    # Loop through starting indices for each trough
    # Set difference_mask to False for all pixels in troughs
    for i in range(len(start_indx)):
        for j in range(len(difference_mask)):
            if (speed_plot[plotmin:plotmax][j] >= speed_plot[plotmin:plotmax][start_indx[i]] and 
                speed_plot[plotmin:plotmax][j] <= speed_plot[plotmin:plotmax][end_indx[i]]):
                difference_mask[j] = False

    # Compute the average difference of the non-trough pixels
    difference = np.mean(abs(model_1_origin[plotmin:plotmax][difference_mask]
                             - balspec[plotmin:plotmax][difference_mask]))

    PCA_Model = model_1_origin_safe[plotmin:speed_max_ai]

    # Loop over troughs to sum up AI, AI_ERR
    for i in range(len(start_indx)):
        ai, ai_err = calculate_Index(speed_c_ai[start_indx[i]:end_indx[i]],
                                     PCA_Model[start_indx[i]:end_indx[i]],
                                     normal_flux_ai[start_indx[i]:end_indx[i]],
                                     sigma_ai[start_indx[i]:end_indx[i]],
                                     450, difference)
        balinfo['AI_CIV'] = balinfo['AI_CIV'] + ai
        balinfo['VMAX_CIV_450'][i] = -speed_c_ai[start_indx[i]]
        balinfo['VMIN_CIV_450'][i] = -speed_c_ai[end_indx[i]]
        min_flux = normal_flux_ai[start_indx[i]:end_indx[i]].min()
        balinfo['FMIN_CIV_450'][i] = min_flux
        balinfo['POSMIN_CIV_450'][i] = -1.*speed_c_ai[start_indx[i]:end_indx[i]]\
            [np.where(normal_flux_ai[start_indx[i]:end_indx[i]]
            == min_flux)[0][0]]
        balinfo['NCIV_450'] += 1
        balinfo['AI_CIV_ERR'] = balinfo['AI_CIV_ERR'] + ai_err

    # make sure there was a detection
    if balinfo['AI_CIV'] == 0:
        balinfo['AI_CIV'] = 0.
        balinfo['AI_CIV_ERR'] = 0.
        balinfo['NCIV_450'] = 0.
        balinfo['VMIN_CIV_450'] = -1.*np.ones(bc.NAI, dtype=float)
        balinfo['VMAX_CIV_450'] = -1.*np.ones(bc.NAI, dtype=float)
        balinfo['POSMIN_CIV_450'] = -1.*np.ones(bc.NAI, dtype=float)
        balinfo['FMIN_CIV_450'] = -1.*np.ones(bc.NAI, dtype=float)

    if balinfo['VMAX_CIV_450'][0] > 10000.:
        balinfo['TROUGH_10K'] = 1

    # Determine CIV BI troughs if median continuum level is greater than the uncertainty
    if np.median(balspec[plotmin:speed_max_bi]) > np.median(sigma_origin[plotmin:speed_max_bi]): 
        start_indx, end_indx = determine_trough_BI(normal_flux_bi, sigma_bi, speed_c_bi)
    else: 
        start_indx = []
        end_indx = []
        if verbose: 
            print("Too noisy to calculate BI")

#                                            speed_plot[plotmin:plotmax])
    PCA_Model = model_1_origin[plotmin:speed_max_bi]
    for i in range(len(start_indx)):
        bi, bi_err = calculate_Index(speed_c_bi[start_indx[i]:end_indx[i]],
                                     PCA_Model[start_indx[i]:end_indx[i]],
                                     normal_flux_bi[start_indx[i]:end_indx[i]],
                                     sigma_bi[start_indx[i]:end_indx[i]],
                                     2000, difference)
        balinfo['BI_CIV'] = balinfo['BI_CIV'] + bi
        balinfo['VMAX_CIV_2000'][i] = -speed_c_bi[start_indx[i]]
        balinfo['VMIN_CIV_2000'][i] = -speed_c_bi[end_indx[i]]
        min_flux = normal_flux_bi[start_indx[i]:end_indx[i]].min()
        balinfo['FMIN_CIV_2000'][i] = min_flux
        balinfo['POSMIN_CIV_2000'][i] = -speed_c_bi[start_indx[i]:end_indx[i]]\
                [np.where(normal_flux_bi[start_indx[i]:end_indx[i]] == min_flux)[0][0]]
        balinfo['NCIV_2000'] += 1
        balinfo['BI_CIV_ERR'] = balinfo['BI_CIV_ERR'] + bi_err

    # make sure there was a detection
    if balinfo['BI_CIV'] == 0:
        balinfo['BI_CIV'] = 0.
        balinfo['BI_CIV_ERR'] = 0.
        balinfo['NCIV_2000'] = 0.
        balinfo['VMIN_CIV_2000'] = -1.*np.ones(bc.NBI, dtype=float)
        balinfo['VMAX_CIV_2000'] = -1.*np.ones(bc.NBI, dtype=float)
        balinfo['POSMIN_CIV_2000'] = -1.*np.ones(bc.NBI, dtype=float)
        balinfo['FMIN_CIV_2000'] = -1.*np.ones(bc.NBI, dtype=float)

    #############

    # Repeat measurement for SiIV region if that entire region is in the 
    # bandpass. This requires balwave[0] > 1281.5 or z > 1.8 

    if verbose: 
        print("calculatebalinfo: SiIV -- ") 
        print("calculatebalinfo: ", balwave[0], bc.lambdaSiIV*(1.-25000./bc.c)) 
    if balwave[0] <= bc.lambdaSiIV*(1.-25000./bc.c):  # < 1281.4
        # print("Should include SIIV region") 
        # Wavelength range for SIIV BAL search
        speed = bc.c * ((balwave - bc.lambdaSiIV)/bc.lambdaSiIV)
        speed_plot = speed
        speed_min = np.where(speed >= bc.VMIN_BAL)[0][0]
        speed_max = np.where(speed >= bc.VMAX_BAL)[0][0]
        plotmin = speed_min
        speed_max_bi = np.where(speed >= -3000 )[0][0]
        speed_max_ai = np.where(speed >= 0)[0][0]
        speed_c_bi = speed[speed_min:speed_max_bi]
        speed_c_ai = speed[speed_min:speed_max_ai]
        # normal_flux = balspec[speed_min:speed_max] / model_1_origin[speed_min:speed_max]
        speed_c = speed[speed_min:speed_max]
        #normal_flux_bi = balspec[plotmin:speed_max_bi] / model_1_origin[plotmin:speed_max_bi]
        #normal_flux_ai = balspec[plotmin:speed_max_ai] / model_1_origin[plotmin:speed_max_ai]
        normal_flux_bi = np.divide(balspec[plotmin:speed_max_bi], model_1_origin_safe[plotmin:speed_max_bi], out=np.full_like(model_1_origin_safe[plotmin:speed_max_bi], np.inf), where=model_1_origin_safe[plotmin:speed_max_bi] !=0)
        normal_flux_ai = np.divide(balspec[plotmin:speed_max_ai], model_1_origin_safe[plotmin:speed_max_ai], out=np.full_like(model_1_origin_safe[plotmin:speed_max_ai], np.inf), where=model_1_origin_safe[plotmin:speed_max_ai] !=0)

        # plotmax = plotmin + 500
        plotmax = np.where(speed >= bc.VMAX_BAL)[0][0]   # Changed 29 Mar 2020 by PM
        #sigma_bi = sigma_origin[plotmin:speed_max_bi] / model_1_origin[plotmin:speed_max_bi]
        #sigma_ai = sigma_origin[plotmin:speed_max_ai] / model_1_origin[plotmin:speed_max_ai]
        sigma_bi = np.divide(sigma_origin[plotmin:speed_max_bi], model_1_origin_safe[plotmin:speed_max_bi], out=np.full_like(model_1_origin_safe[plotmin:speed_max_bi], np.inf), where=model_1_origin_safe[plotmin:speed_max_bi] !=0)
        sigma_ai = np.divide(sigma_origin[plotmin:speed_max_ai], model_1_origin_safe[plotmin:speed_max_ai], out=np.full_like(model_1_origin_safe[plotmin:speed_max_ai], np.inf), where=model_1_origin_safe[plotmin:speed_max_ai] !=0)


        # Determine SiIV AI troughs
        if np.median(balspec[plotmin:speed_max_ai]) > np.median(sigma_origin[plotmin:speed_max_ai]): 
            start_indx, end_indx = determine_trough_AI(normal_flux_ai, sigma_ai, speed_c_ai)
        else: 
            start_indx = []
            end_indx = []
            if verbose: 
                print("Too noisy to calculate AI")

        difference_mask = np.ones(len(speed_plot[plotmin:plotmax]), dtype=bool)
        for i in range(len(start_indx)):
            for j in range(len(difference_mask)):
                if (speed_plot[plotmin:plotmax][j] >= speed_plot[plotmin:plotmax][start_indx[i]] and 
                    speed_plot[plotmin:plotmax][j] <= speed_plot[plotmin:plotmax][end_indx[i]]):
                    difference_mask[j] = False
        difference = np.mean(abs(model_1_origin[plotmin:plotmax][difference_mask] -
                             balspec[plotmin:plotmax][difference_mask]))

        PCA_Model = model_1_origin[plotmin:speed_max_ai]
        for i in range(len(start_indx)):
            ai_SIV,ai_err_SIV = calculate_Index(speed_c_ai[start_indx[i]:end_indx[i]],PCA_Model[start_indx[i]:end_indx[i]],
                                    normal_flux_ai[start_indx[i]:end_indx[i]],sigma_ai[start_indx[i]:end_indx[i]],450,difference)
        
            balinfo['VMAX_SIIV_450'][i] = -speed_c_ai[start_indx[i]]
            balinfo['VMIN_SIIV_450'][i] = -speed_c_ai[end_indx[i]]
            min_flux = normal_flux_ai[start_indx[i]:end_indx[i]].min()
            balinfo['FMIN_SIIV_450'][i] = min_flux
            balinfo['POSMIN_SIIV_450'][i] = -speed_c_ai[start_indx[i]:end_indx[i]]\
                [np.where(normal_flux_ai[start_indx[i]:end_indx[i]] == min_flux)[0][0]]
            balinfo['NSIIV_450'] += 1

            balinfo['AI_SIIV'] = balinfo['AI_SIIV'] + ai_SIV
            balinfo['AI_SIIV_ERR'] = balinfo['AI_SIIV_ERR'] + ai_err_SIV

        # make sure there was a detection
        if balinfo['AI_SIIV'] == 0:
            balinfo['AI_SIIV'] = 0.
            balinfo['AI_SIIV_ERR'] = 0.
            balinfo['NSIIV_450'] = 0.
            balinfo['VMIN_SIIV_450'] = -1.*np.ones(bc.NAI, dtype=float)
            balinfo['VMAX_SIIV_450'] = -1.*np.ones(bc.NAI, dtype=float)
            balinfo['POSMIN_SIIV_450'] = -1.*np.ones(bc.NAI, dtype=float)
            balinfo['FMIN_SIIV_450'] = -1.*np.ones(bc.NAI, dtype=float)

        # Determine SIIV BI troughs
        if np.median(balspec[plotmin:speed_max_bi]) > np.median(sigma_origin[plotmin:speed_max_bi]): 
            start_indx, end_indx = determine_trough_BI(normal_flux_bi, sigma_bi, speed_c_bi)
        else: 
            start_indx = []
            end_indx = []
            if verbose: 
                print("Too noisy to calculate BI")

        PCA_Model = model_1_origin[plotmin:speed_max_bi]

        for i in range(len(start_indx)):
            bi_SIV,bi_err_SIV = calculate_Index(speed_c_bi[start_indx[i]:end_indx[i]],PCA_Model[start_indx[i]:end_indx[i]],
                                            normal_flux_bi[start_indx[i]:end_indx[i]],sigma_bi[start_indx[i]:end_indx[i]],2000,difference)
            balinfo['VMAX_SIIV_2000'][i] = -speed_c_bi[start_indx[i]]
            balinfo['VMIN_SIIV_2000'][i] = -speed_c_bi[end_indx[i]]
            min_flux = normal_flux_bi[start_indx[i]:end_indx[i]].min()
            balinfo['FMIN_SIIV_2000'][i] = min_flux
            balinfo['POSMIN_SIIV_2000'][i] = -speed_c_bi[start_indx[i]:end_indx[i]]\
                    [np.where(normal_flux_bi[start_indx[i]:end_indx[i]] == min_flux)[0][0]]
            balinfo['NSIIV_2000'] += 1
    
            balinfo['BI_SIIV'] = balinfo['BI_SIIV'] + bi_SIV
            balinfo['BI_SIIV_ERR'] =  balinfo['BI_SIIV_ERR'] + bi_err_SIV
        # make sure there was a detection
        if balinfo['BI_SIIV'] == 0:
            balinfo['BI_SIIV'] = 0.
            balinfo['BI_SIIV_ERR'] = 0.
            balinfo['NSIIV_2000'] = 0.
            balinfo['VMIN_SIIV_2000'] = -1.*np.ones(bc.NBI, dtype=float)
            balinfo['VMAX_SIIV_2000'] = -1.*np.ones(bc.NBI, dtype=float)
            balinfo['POSMIN_SIIV_2000'] = -1.*np.ones(bc.NBI, dtype=float)
            balinfo['FMIN_SIIV_2000'] = -1.*np.ones(bc.NBI, dtype=float)

    balinfo['BI_CIV_ERR'] = np.sqrt(balinfo['BI_CIV_ERR'])
    balinfo['AI_CIV_ERR'] = np.sqrt(balinfo['AI_CIV_ERR'])
    balinfo['BI_SIIV_ERR'] = np.sqrt(balinfo['BI_SIIV_ERR'])
    balinfo['AI_SIIV_ERR'] = np.sqrt(balinfo['AI_SIIV_ERR'])

    return balinfo
            
def createpcatemplate(pcaeigen, pcacoeffs):
    '''
    Create a PCA template for a QSO from eigenvectors in pcaeigen
    and coefficients in pcacoeffs

    Parameters
    ----------
    pcaeigen : np.array
        PCA wavelength and components
    pcacoeffs : 1-d float array
        coefficients of PCA fit

    Returns
    -------
    outspec : 1-d float array
        PCA model
    '''
    ### Modified pcacoeffs that allows alternate templates
    ### Replaces hardcoded array
    outspec = 0
    for index,item in enumerate(pcaeigen.dtype.names):
        if item == 'WAVE' : continue
        outspec = outspec + pcacoeffs[index-1]*pcaeigen[item]
    return outspec


def qsocatsearch(catalog, thing_id=-9999, sdssname="", pmf=[]):
    '''
    Retrieve index of QSO with unique THING_ID or SDSS_NAME from QSO catalog.

    Parameters
    ----------
    catalog : data arraay
        hdu[1].data of SDSS QSO catalog
    thing_id : int
        unique SDSS THING_ID integer
    sdssname : string
        unique SDSS_NAME
    pmf : 1-d int array
        PLATE/MJD/FIBERID for QSO

    Returns
    -------
    int
        index of thing_id in catalog
    '''

    # Try THING_ID first:
    if thing_id >= 0:
        qsoindx = np.where(catalog['THING_ID'] == thing_id)[0][0]
        return qsoindx
    elif len(sdssname) > 0:
        qsoindx = np.where(catalog['SDSS_NAME'] == sdssname)[0][0]
        return qsoindx
    elif len(pmf) == 3: 
        mask0 = catalog['PLATE'] == pmf[0]
        mask1 = catalog['MJD'] == pmf[1]
        mask2 = catalog['FIBERID'] == pmf[2]
        mask = mask0*mask1*mask2
        try: 
            qsoindx = np.where(mask == True)[0][0]
            return qsoindx
        except IndexError: 
            raise RuntimeError("[PLATE/MJD/FIBERID] not found in catalog") 
    else:
        raise RuntimeError("Specify THING_ID, SDSS_NAME, or PLATE/MJD/FIBERID")

def find_nearest(array, value):
    '''
    Find the (nearest) index with value in array

    Parameters
    ----------
    array : 1-D float arraay
        sorted array of floats
    value: float
        value to search for in array

    Returns
    -------
    int
        index closest element to value in array
    '''

    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1])
                    < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx

def sdsschisq(qsospec, zspec):
    '''
    Evaluate chisq of SDSS model over the PCA wavelength range 

    Parameters
    ----------
    qsospec : np.array
        QSO spectrum
    zspec : float
        redshift of QSO

    Returns
    -------
    redchisq : float
        reduced chi2 of SDSS model over PCA wavelength range
    '''

    wave_rest = np.power(10, qsospec['loglam'])/(1+zspec)

    # index of shortest wavelength will be the wavelength of either 
    # bc.BAL_LAMBDA_MIN or wave_rest[0]
    indx1 = 0
    if bc.BAL_LAMBDA_MIN > wave_rest[0]:
        indx1 = find_nearest(wave_rest, bc.BAL_LAMBDA_MIN) + 1

    # index of longest wavelength will be the wavelength of either 
    # bc.BAL_LAMBDA_MAX or wave_rest[-1]
    indx2 = len(wave_rest) - 1
    if bc.BAL_LAMBDA_MAX < wave_rest[-1]:
        indx2 = find_nearest(wave_rest, bc.BAL_LAMBDA_MAX) - 1

    # data over the same range 
    balwave = wave_rest[indx1:indx2]
    balspec = qsospec['flux'][indx1:indx2]
    balivar = qsospec['ivar'][indx1:indx2]
    balmodel = qsospec['model'][indx1:indx2]
    warnings.filterwarnings("ignore",category=RuntimeWarning) # eliminate divide by zero warning
    balerror = np.nan_to_num(np.sqrt(1./balivar))
  
    # mask doesn't do anything, left in as may want to use it in upgrade
    mmask = np.ones(indx2-indx1, dtype=bool) 

    x0 = np.array([1])

    def chisqfunc(alpha):
        m_lamda = alpha*balmodel
        chisq = np.sum(((m_lamda[mmask] - balspec[mmask])/balerror[mmask])**2)
        return chisq

    solve1 = opt.minimize(chisqfunc, x0, method='Nelder-Mead')

    chisq = chisqfunc(solve1.x)
    rchisq = chisq/(sum(mmask)-len(x0))
    return rchisq


def fitpca(idata, zspec, ipca, mmask):
    '''
    Fit PCA components to QSO spectrum in rest frame. PCA components
    are interpolated to match QSO spectrum. Wavelength region trimmed to 
    PCA fit limits from balconfig.py 

    Parameters
    ----------
    idata : 2-d float array
        wave_rest, flux, error of SDSS spectrum region to fit
    zspec : float
        redshift of QSO
    ipca : 2-d float array
        PCA components interpolated to match wave_rest
    mmask : boolean array
        mask of values to exclude in PCA fit, created from BAL info

    Returns
    -------
    pcadata : 1-d float array
        array of NPCA PCA coefficients + 2 chisq values
    '''
    
    ### Replaces hardcoded NPCA
    x0 = np.ones(len(ipca)) 

    balspec = idata[1]
    balerror = idata[2]
    def chisqfunc(alpha):
        m_lambda = np.zeros(len(balspec), dtype=float)
        for i in range(len(ipca)): 
            m_lambda += alpha[i]*ipca[i] 
        chisq = np.sum(((m_lambda[mmask] - balspec[mmask])/balerror[mmask])**2)
        return chisq

    solve1 = opt.minimize(chisqfunc, x0, method='Nelder-Mead')

    chisq = chisqfunc(solve1.x)
    rchisq = chisq/(sum(mmask)-len(x0))
    coeffs = solve1.x
    pcadata = np.append(coeffs, [rchisq])
    return pcadata


def baltomask(balinfo, wave, baltype='AI'):
    '''
    Create a mask of BAL features for a QSO based on balinfo
    based on wavelength array wave

    Parameters
    ----------
    balinfo : array
        information about BAL properties
    wave : 1-D float array
        wavelength array for mask

    Returns
    -------
    mask
        boolean array of mask values
    '''
    balmask = np.ones(len(wave), dtype=bool)

    if baltype == 'AI':
        for i in range(len(balinfo['VMIN_CIV_450'])):
            if balinfo['VMIN_CIV_450'][i] > 0.:
                w1 = bc.lambdaCIV*(1.-balinfo['VMAX_CIV_450'][i]/bc.c)
                w2 = bc.lambdaCIV*(1.-balinfo['VMIN_CIV_450'][i]/bc.c)
                indx1 = find_nearest(wave, w1)
                indx2 = find_nearest(wave, w2)
                balmask[indx1:indx2] = 0
        for i in range(len(balinfo['VMIN_SIIV_450'])):
            if balinfo['VMIN_SIIV_450'][i] > 0.:
                w1 = bc.lambdaSiIV*(1.-balinfo['VMAX_SIIV_450'][i]/bc.c)
                w2 = bc.lambdaSiIV*(1.-balinfo['VMIN_SIIV_450'][i]/bc.c)
                indx1 = find_nearest(wave, w1)
                indx2 = find_nearest(wave, w2)
                balmask[indx1:indx2] = 0
    else:  # assume BI
        for i in range(len(balinfo['VMIN_CIV_2000'])):
            if balinfo['VMIN_CIV_2000'][i] > 0.:
                w1 = bc.lambdaCIV*(1.-balinfo['VMAX_CIV_2000'][i]/bc.c)
                w2 = bc.lambdaCIV*(1.-balinfo['VMIN_CIV_2000'][i]/bc.c)
                indx1 = find_nearest(wave, w1)
                indx2 = find_nearest(wave, w2)
                balmask[indx1:indx2] = 0
        for i in range(len(balinfo['VMIN_SIV_2000'])):
            if balinfo['VMIN_SIIV_2000'][i] > 0.:
                w1 = bc.lambdaSiIV*(1.-balinfo['VMAX_SIIV_2000'][i]/bc.c)
                w2 = bc.lambdaSiIV*(1.-balinfo['VMIN_SIIV_2000'][i]/bc.c)
                indx1 = find_nearest(wave, w1)
                indx2 = find_nearest(wave, w2)
                balmask[indx1:indx2] = 0
    return balmask


def calcbalparams(qsospec, pcaeigen, zspec, maxiter=10, verbose=False):
    '''
    Iteratively fit BAL features in QSO spectrum qsospec at redshift zspec
    with eigenvectors in pcaeigen and return BAL data

    Parameters
    ----------
    qsospec : np.array
        QSO spectrum with 'wave' or 'loglam', 'flux', 'ivar', ('model')
    pcaeigen : np.array
        PCA wavelength and components
    zspec : float
        redshift of QSO
    maxiter : int (optional)
        information about BAL properties
    verbose : bool (optional)
        display debugging information

    Returns
    -------
    calcinfo : array
        BAL properties
    calcpcaout : 1-D float array
        coefficients from PCA fit to QSO spectrum + PCA chisq + SDSS chisq
    calcmask : boolean array
        mask of BAL troughs
    '''

    try:
        wave_rest = qsospec['wave'] / (1 + zspec)
        is_desi_spec = True
    except (ValueError, KeyError):
        wave_rest = 10**qsospec['loglam'] / (1 + zspec)
        is_desi_spec = False

    if wave_rest[-1] < bc.lambdaCIV or wave_rest[0] > bc.lambdaCIV:
        raise RuntimeError("Spectrum does not cover the CIV line region.")

    # interpolate the PCA eigenspectra onto this wavelength grid
    pca_eigen_interp = [np.interp(wave_rest, pcaeigen['WAVE'], pcaeigen[name])
                        for name in pcaeigen.dtype.names if name != 'WAVE']

    # compute the SNR values before trimming
    flux_full = qsospec['flux']
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ivar_full = qsospec['ivar']
        sigma_full = np.nan_to_num(np.sqrt(1.0 / ivar_full))
    snr_full = np.divide(flux_full, sigma_full, out=np.zeros_like(flux_full), where=sigma_full > 0)

    # Calculate SNR_FOREST (1040, 1205, require at least 50A)
    forest_min, forest_max = 1040., 1205.
    coverage_start = max(wave_rest[0], forest_min)
    coverage_end = min(wave_rest[-1], forest_max)
    if coverage_end - coverage_start >= 50.:
        i1 = np.searchsorted(wave_rest, max(forest_min, wave_rest[0]))
        i2 = np.searchsorted(wave_rest, min(forest_max, wave_rest[-1]))
        if i2 - i1 >= 1:
            snr_forest = np.median(snr_full[i1:i2][np.isfinite(snr_full[i1:i2])])
    else:
        snr_forest = -1.

    # Calculate SNR_REDSIDE (1420-1480 Angstroms)
    redside_min, redside_max = 1420., 1480.
    # Condition: Only calculate if the full range is available
    if wave_rest[0] <= redside_min and wave_rest[-1] >= redside_max:
        i1 = np.searchsorted(wave_rest, redside_min)
        i2 = np.searchsorted(wave_rest, redside_max)
        if i2 - i1 >= 1:
            snr_redside = np.median(snr_full[i1:i2][np.isfinite(snr_full[i1:i2])])
    else:
        snr_redside = -1.

    # Trim down to the BAL region for remaining calculations
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

    calcmask = np.ones(len(balwave), dtype=bool)
    nmasked_prev = -1
    itr = 0
    while itr < maxiter and np.sum(calcmask) != nmasked_prev:
        nmasked_prev = np.sum(calcmask)
        calcpcaout = fitpca(idata, zspec, ipca, calcmask)
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

    calcinfo['SNR_REDSIDE'] = snr_redside
    calcinfo['SNR_FOREST'] = snr_forest

    calcpcaout = np.append(calcpcaout, [sdsschi2])
    
    ### THE CODE BELOW STILL NEEDS TO BE CHANGED FOR SDSS DATA
    if calcpcaout[-2] > 1.5*sdsschi2: 
        calcinfo = calculatebalinfo(idata, qsospec['model'][indx1:indx2])
        calcmask = baltomask(calcinfo, balwave)
        if verbose: 
            print("Warning: Used SDSS model, PCA coefficients set to -1")
            # set PCA coefficients to -1. if SDSS model was better
            calcpcaout[0] = -1.
            calcpcaout[1] = -1.
            calcpcaout[2] = -1.
            calcpcaout[3] = -1.
            calcpcaout[4] = -1.

    return calcinfo, calcpcaout, calcmask

