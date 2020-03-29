"""

bal.baltable
============

Routines to build the output BAL catalog

2018 original code by Zhiyuan Guo
2019 updated and expanded by Paul Martini
2020 expanded for DESI

initbaltab_sdss()    Create empty BAL table 
findbaltab_sdss()    Return hdu for BAL table, initialize if necessary
updatebaltab_sdss()  Update BAL values in a row of the BAL table

initbaltab_desi()    Create empty BAL table 
findbaltab_desi()    Return hdu for BAL table, initialize if necessary
updatebaltab_desi()  Update BAL values in a row of the BAL table

"""

from __future__ import print_function, absolute_import, division

from astropy.io import fits
import numpy as np
import os
import fitbal
import balconfig as bc 


# TODO: remove astropy dependency


def cattobalinfo(array):
    '''
    Populate the balinfo dictionary from a row in a BAL catalog

    Parameters
    ----------
    array : data
        row of a BAL catalog

    Returns
    -------
    info : dict
        dictionary of BAL parameters
    '''

    info = {}

    info['BI_CIV'] = array['BI_CIV'] 
    info['BI_CIV_ERR'] = array['ERR_BI_CIV']
    info['NCIV_2000'] = array['NCIV_2000']
    info['VMIN_CIV_2000'] = array['VMIN_CIV_2000']
    info['VMAX_CIV_2000'] = array['VMAX_CIV_2000']
    info['POSMIN_CIV_2000'] = array['POSMIN_CIV_2000']
    info['FMIN_CIV_2000'] = array['FMIN_CIV_2000']
    info['AI_CIV'] = array['AI_CIV']
    info['AI_CIV_ERR'] = array['ERR_AI_CIV']
    info['NCIV_450'] = array['NCIV_450']
    info['VMIN_CIV_450'] = array['VMIN_CIV_450']
    info['VMAX_CIV_450'] = array['VMAX_CIV_450']
    info['POSMIN_CIV_450'] = array['POSMIN_CIV_450']
    info['FMIN_CIV_450'] = array['FMIN_CIV_450']
    info['BI_SIIV'] = array['BI_SIIV']
    info['BI_SIIV_ERR'] = array['ERR_BI_SIIV']
    info['NSIIV_2000'] = array['NSIIV_2000']
    info['VMIN_SIIV_2000'] = array['VMIN_SIIV_2000']
    info['VMAX_SIIV_2000'] = array['VMAX_SIIV_2000']
    info['POSMIN_SIIV_2000'] = array['POSMIN_SIIV_2000']
    info['FMIN_SIIV_2000'] = array['FMIN_SIIV_2000']
    info['AI_SIIV'] = array['AI_SIIV']
    info['AI_SIIV_ERR'] = array['ERR_AI_SIIV']
    info['NSIIV_450'] = array['NSIIV_450']
    info['VMIN_SIIV_450'] = array['VMIN_SIIV_450']
    info['VMAX_SIIV_450'] = array['VMAX_SIIV_450']
    info['POSMIN_SIIV_450'] = array['POSMIN_SIIV_450']
    info['FMIN_SIIV_450'] = array['FMIN_SIIV_450']

    return info

#
# ------------ DESI routines ---------------
#

def initbaltab_desi(specdata, zdata, outputfile, overwrite=False):
    '''
    Create an empty BAL table from a QSO catalog
    This should just include QSOs in the BAL redshift range

    Parameters
    ----------
    specdata : specobj.fibermap
        fiber map data for qsos to search
    zdata : redshift data
        redshift data for qsos to search
    outputfile : fitsfile
        name to use for output BAL catalog file

    Returns
    -------
    none
    '''

    if os.path.isfile(outputfile) and not overwrite: 
        print("initbaltab_desi(): BAL catalog {0} already exists and overwrite=False".format(outputfile))
        return

    NROWS = len(specdata) 

    # Columns to copy from the fibermap data
    col0 = fits.Column(name='TARGETID', format='K', array=specdata['TARGETID'])
    col1 = fits.Column(name='TARGET_RA', format='E', array=specdata['TARGET_RA'])
    col2 = fits.Column(name='TARGET_DEC', format='E', array=specdata['TARGET_DEC'])

    # Columns to copy from the redshift data
    col3 = fits.Column(name='Z', format='E', array=zdata['Z'])
    col4 = fits.Column(name='ZERR', format='E', array=zdata['ZERR'])
    col5 = fits.Column(name='ZWARN', format='E', array=zdata['ZWARN'])

    # PCA Fit Coefficients and chisq result
    pca_array = np.zeros([NROWS, bc.NPCA], dtype=float)  # PCA Fit Coefficients
    col6 = fits.Column(name='PCA_COEFFS', format='5E', array=pca_array)
    pca_chi2 = np.zeros([NROWS], dtype=float)  # PCA reduced chi2
    col7 = fits.Column(name='PCA_CHI2', format='E', array=pca_chi2)

    # Collection of empty arrays to set up rest of the BAL catalog
    # PM Note: Might change to -1. to indicate empty field
    zfloat_col = np.zeros([NROWS], dtype=float)
    zint_col = np.zeros([NROWS], dtype=float)
    zfloat_bicol = np.zeros([NROWS, bc.NBI], dtype=float)
    zfloat_aicol = np.zeros([NROWS, bc.NAI], dtype=float)

    # This will come from CNN Classifier
    col25 = fits.Column(name='BAL_PROB', format='E', array=zfloat_col)

    # These quantities are from trough fit
    col26 = fits.Column(name='BI_CIV', format='E', array=zfloat_col)
    col27 = fits.Column(name='ERR_BI_CIV', format='E', array=zfloat_col)
    col28 = fits.Column(name='NCIV_2000', format='J', array=zint_col)
    col29 = fits.Column(name='VMIN_CIV_2000', format='5E', array=zfloat_bicol)
    col30 = fits.Column(name='VMAX_CIV_2000', format='5E', array=zfloat_bicol)
    col31 = fits.Column(name='POSMIN_CIV_2000', format='5E', array=zfloat_bicol)
    col32 = fits.Column(name='FMIN_CIV_2000', format='5E', array=zfloat_bicol)

    col33 = fits.Column(name='AI_CIV', format='E', array=zfloat_col)
    col34 = fits.Column(name='ERR_AI_CIV', format='E', array=zfloat_col)
    col35 = fits.Column(name='NCIV_450', format='J', array=zint_col)
    col36 = fits.Column(name='VMIN_CIV_450', format='17E', array=zfloat_aicol)
    col37 = fits.Column(name='VMAX_CIV_450', format='17E', array=zfloat_aicol)
    col38 = fits.Column(name='POSMIN_CIV_450', format='17E', array=zfloat_aicol)
    col39 = fits.Column(name='FMIN_CIV_450', format='17E', array=zfloat_aicol)

    col40 = fits.Column(name='BI_SIIV', format='E', array=zfloat_col)
    col41 = fits.Column(name='ERR_BI_SIIV', format='E', array=zfloat_col)
    col42 = fits.Column(name='NSIIV_2000', format='J', array=zint_col)
    col43 = fits.Column(name='VMIN_SIIV_2000', format='5E', array=zfloat_bicol)
    col44 = fits.Column(name='VMAX_SIIV_2000', format='5E', array=zfloat_bicol)
    col45 = fits.Column(name='POSMIN_SIIV_2000', format='5E', array=zfloat_bicol)
    col46 = fits.Column(name='FMIN_SIIV_2000', format='5E', array=zfloat_bicol)

    col47 = fits.Column(name='AI_SIIV', format='E', array=zfloat_col)
    col48 = fits.Column(name='ERR_AI_SIIV', format='E', array=zfloat_col)
    col49 = fits.Column(name='NSIIV_450', format='J', array=zint_col)
    col50 = fits.Column(name='VMIN_SIIV_450', format='17E', array=zfloat_aicol)
    col51 = fits.Column(name='VMAX_SIIV_450', format='17E', array=zfloat_aicol)
    col52 = fits.Column(name='POSMIN_SIIV_450', format='17E', array=zfloat_aicol)
    col53 = fits.Column(name='FMIN_SIIV_450', format='17E', array=zfloat_aicol)

    balhead = fits.Header({'SIMPLE': True})
    balhead['EXTNAME'] = "BALCAT"
    tabhdu = fits.BinTableHDU.from_columns([col0, col1, col2, col3, col4, col5,
                                            col6, col7, 
                                            col25, col26,
                                            col27, col28, col29, col30, col31,
                                            col32, col33, col34, col35, col36,
                                            col37, col38, col39, col40, col41,
                                            col42, col43, col44, col45, col46,
                                            col47, col48, col49, col50, col51,
                                            col52, col53], 
                                            header=balhead)

    tabhdu.writeto(outputfile, overwrite=overwrite)


def updatebaltab_desi(targetid, balhdu, info, pcaout):
    '''
    Add information about QSO with targetid to baltab

    Parameters
    ----------
    targetid : int
        DESI TARGETID
    balcatfile : fitsfile
        name of BAL catalog file
    info : record array
        output from fitbal.calcbalparams
    pcaout : 1-D float array
        output from fitbal.fitpca

    Returns
    -------
    balhdu : FITS HDU
        fits hdu for updated bal catalog
    '''

    # Identify the index of TARGNAME in 
    # qindx = fitbal.qsocatsearch(balhdu[1].data, dssname)
    qindx = np.where(balhdu[1].data['TARGETID'] == targetid)[0][0]

    balhdu[1].data[qindx]['PCA_COEFFS'] = pcaout[:bc.NPCA]
    balhdu[1].data[qindx]['PCA_CHI2'] = pcaout[-2]
    # balhdu[1].data[qindx]['SDSS_CHI2'] = pcaout[-1]

    balhdu[1].data[qindx]['BI_CIV'] = info['BI_CIV']
    balhdu[1].data[qindx]['ERR_BI_CIV'] = info['BI_CIV_ERR']
    balhdu[1].data[qindx]['NCIV_2000'] = info['NCIV_2000']
    balhdu[1].data[qindx]['VMIN_CIV_2000'] = info['VMIN_CIV_2000']
    balhdu[1].data[qindx]['VMAX_CIV_2000'] = info['VMAX_CIV_2000']
    balhdu[1].data[qindx]['POSMIN_CIV_2000'] = info['POSMIN_CIV_2000']
    balhdu[1].data[qindx]['FMIN_CIV_2000'] = info['FMIN_CIV_2000']

    balhdu[1].data[qindx]['AI_CIV'] = info['AI_CIV']
    balhdu[1].data[qindx]['ERR_AI_CIV'] = info['AI_CIV_ERR']
    balhdu[1].data[qindx]['NCIV_450'] = info['NCIV_450']
    balhdu[1].data[qindx]['VMIN_CIV_450'] = info['VMIN_CIV_450']
    balhdu[1].data[qindx]['VMAX_CIV_450'] = info['VMAX_CIV_450']
    balhdu[1].data[qindx]['POSMIN_CIV_450'] = info['POSMIN_CIV_450']
    balhdu[1].data[qindx]['FMIN_CIV_450'] = info['FMIN_CIV_450']

    balhdu[1].data[qindx]['BI_SIIV'] = info['BI_SIIV']
    balhdu[1].data[qindx]['ERR_BI_SIIV'] = info['BI_SIIV_ERR']
    balhdu[1].data[qindx]['NSIIV_2000'] = info['NSIIV_2000']
    balhdu[1].data[qindx]['VMIN_SIIV_2000'] = info['VMIN_SIIV_2000']
    balhdu[1].data[qindx]['VMAX_SIIV_2000'] = info['VMAX_SIIV_2000']
    balhdu[1].data[qindx]['POSMIN_SIIV_2000'] = info['POSMIN_SIIV_2000']
    balhdu[1].data[qindx]['FMIN_SIIV_2000'] = info['FMIN_SIIV_2000']

    balhdu[1].data[qindx]['AI_SIIV'] = info['AI_SIIV']
    balhdu[1].data[qindx]['ERR_AI_SIIV'] = info['AI_SIIV_ERR']
    balhdu[1].data[qindx]['NSIIV_450'] = info['NSIIV_450']
    balhdu[1].data[qindx]['VMIN_SIIV_450'] = info['VMIN_SIIV_450']
    balhdu[1].data[qindx]['VMAX_SIIV_450'] = info['VMAX_SIIV_450']
    balhdu[1].data[qindx]['POSMIN_SIIV_450'] = info['POSMIN_SIIV_450']
    balhdu[1].data[qindx]['FMIN_SIIV_450'] = info['FMIN_SIIV_450']

    return balhdu

#
# ------------ SDSS routines ---------------
#

def initbaltab_sdss(qsocatfile, outputfile):
    '''
    Create an empty BAL table from a QSO catalog

    Parameters
    ----------
    qsocatfile : fitsfile
        name of QSO catalog
    outputfile : fitsfile
        name to use for output BAL catalog file

    Returns
    -------
    none
    '''

    flag_spectro = True
    flag_z_err = True

    qsocat = fits.open(bc.catdir + qsocatfile)
    NROWS = len(qsocat[1].data)
    # Columns to copy from the QSO Catalog
    col0 = qsocat[1].columns['SDSS_NAME']
    col1 = qsocat[1].columns['RA']
    col2 = qsocat[1].columns['DEC']
    col3 = qsocat[1].columns['THING_ID']
    col4 = qsocat[1].columns['PLATE']
    col5 = qsocat[1].columns['MJD']
    col6 = qsocat[1].columns['FIBERID']
    try: 
        col7 = qsocat[1].columns['SPECTRO']
    except ValueError: 
        flag_spectro = False
    except KeyError: 
        flag_spectro = False
    col8 = qsocat[1].columns['Z']
    try: 
        col9 = qsocat[1].columns['Z_ERR']
    except KeyError: 
        flag_z_err = False
    col10 = qsocat[1].columns['SOURCE_Z']
    col11 = qsocat[1].columns['Z_VI']
    col12 = qsocat[1].columns['Z_PIPE']
    try: 
        col13 = qsocat[1].columns['Z_PIPE_ERR']
    except KeyError: 
        flag_z_pipe_err = False
    col14 = qsocat[1].columns['ZWARNING']
    col15 = qsocat[1].columns['Z_PCA']
    try: 
        col16 = qsocat[1].columns['Z_PCA_ER']
    except KeyError: 
        flag_z_pca_er = False
    col17 = qsocat[1].columns['Z_MGII']

    try: 
        col18 = qsocat[1].columns['BI_CIV']
        col19 = qsocat[1].columns['ERR_BI_CIV']
        col18.name = 'BI_CIV_DR14'
        col19.name = 'ERR_BI_CIV_DR14'
    except KeyError: 
        flag_bi = False
    col20 = qsocat[1].columns['PSFMAG']
    try: 
        col21 = qsocat[1].columns['ERR_PSFMAG']
    except KeyError: 
        flag_err_psfmag = False

    # PCA Fit Coefficients and chisq result
    pca_array = np.zeros([NROWS, bc.NPCA], dtype=float)  # PCA Fit Coefficients
    col22 = fits.Column(name='PCA_COEFFS', format='5E', array=pca_array)
    pca_chi2 = np.zeros([NROWS], dtype=float)  # PCA reduced chi2
    col23 = fits.Column(name='PCA_CHI2', format='E', array=pca_chi2)
    col24 = fits.Column(name='SDSS_CHI2', format='E', array=pca_chi2)

    # Collection of empty arrays to set up rest of the BAL catalog
    # PM Note: Might change to -1. to indicate empty field
    zfloat_col = np.zeros([NROWS], dtype=float)
    zint_col = np.zeros([NROWS], dtype=float)
    zfloat_bicol = np.zeros([NROWS, bc.NBI], dtype=float)
    zfloat_aicol = np.zeros([NROWS, bc.NAI], dtype=float)

    # This will come from CNN Classifier
    col25 = fits.Column(name='BAL_PROB', format='E', array=zfloat_col)

    # These quantities are from trough fit
    col26 = fits.Column(name='BI_CIV', format='E', array=zfloat_col)
    col27 = fits.Column(name='ERR_BI_CIV', format='E', array=zfloat_col)
    col28 = fits.Column(name='NCIV_2000', format='J', array=zint_col)
    col29 = fits.Column(name='VMIN_CIV_2000', format='5E', array=zfloat_bicol)
    col30 = fits.Column(name='VMAX_CIV_2000', format='5E', array=zfloat_bicol)
    col31 = fits.Column(name='POSMIN_CIV_2000', format='5E', array=zfloat_bicol)
    col32 = fits.Column(name='FMIN_CIV_2000', format='5E', array=zfloat_bicol)

    col33 = fits.Column(name='AI_CIV', format='E', array=zfloat_col)
    col34 = fits.Column(name='ERR_AI_CIV', format='E', array=zfloat_col)
    col35 = fits.Column(name='NCIV_450', format='J', array=zint_col)
    col36 = fits.Column(name='VMIN_CIV_450', format='17E', array=zfloat_aicol)
    col37 = fits.Column(name='VMAX_CIV_450', format='17E', array=zfloat_aicol)
    col38 = fits.Column(name='POSMIN_CIV_450', format='17E', array=zfloat_aicol)
    col39 = fits.Column(name='FMIN_CIV_450', format='17E', array=zfloat_aicol)

    col40 = fits.Column(name='BI_SIIV', format='E', array=zfloat_col)
    col41 = fits.Column(name='ERR_BI_SIIV', format='E', array=zfloat_col)
    col42 = fits.Column(name='NSIIV_2000', format='J', array=zint_col)
    col43 = fits.Column(name='VMIN_SIIV_2000', format='5E', array=zfloat_bicol)
    col44 = fits.Column(name='VMAX_SIIV_2000', format='5E', array=zfloat_bicol)
    col45 = fits.Column(name='POSMIN_SIIV_2000', format='5E', array=zfloat_bicol)
    col46 = fits.Column(name='FMIN_SIIV_2000', format='5E', array=zfloat_bicol)

    col47 = fits.Column(name='AI_SIIV', format='E', array=zfloat_col)
    col48 = fits.Column(name='ERR_AI_SIIV', format='E', array=zfloat_col)
    col49 = fits.Column(name='NSIIV_450', format='J', array=zint_col)
    col50 = fits.Column(name='VMIN_SIIV_450', format='17E', array=zfloat_aicol)
    col51 = fits.Column(name='VMAX_SIIV_450', format='17E', array=zfloat_aicol)
    col52 = fits.Column(name='POSMIN_SIIV_450', format='17E', array=zfloat_aicol)
    col53 = fits.Column(name='FMIN_SIIV_450', format='17E', array=zfloat_aicol)


    if flag_spectro: 
        tabhdu = fits.BinTableHDU.from_columns([col0, col1, col2, col3, col4, col5,
                                            col6, col7, col8, col9, col10, col11,
                                            col12, col13, col14, col15, col16,
                                            col17, col18, col19, col20, col21,
                                            col22, col23, col24, col25, col26,
                                            col27, col28, col29, col30, col31,
                                            col32, col33, col34, col35, col36,
                                            col37, col38, col39, col40, col41,
                                            col42, col43, col44, col45, col46,
                                            col47, col48, col49, col50, col51,
                                            col52, col53])
    else: 
        tabhdu = fits.BinTableHDU.from_columns([col0, col1, col2, col3, col4, col5,
                                            col6, col8, col10, col11,
                                            col12, col14, col15, 
                                            col17, col20, 
                                            col22, col23, col24, col25, col26,
                                            col27, col28, col29, col30, col31,
                                            col32, col33, col34, col35, col36,
                                            col37, col38, col39, col40, col41,
                                            col42, col43, col44, col45, col46,
                                            col47, col48, col49, col50, col51,
                                            col52, col53])

    '''
    Note that this makes a table the full length of the DR14 QSO catalog, or
    526,356 long. Zhiyuan's table is just the QSOs with balprob > 0.5, or
    about 53,000. Shorten this table by only including QSOs with z > 1.57 and
    z < 5.6. This selects 229,508 quasars.
    '''

    '''
    mask = tabhdu.data['Z_VI'] > 1.57
    mask = mask*(tabhdu.data['Z_VI'] < 5.6)
    sum(mask)
    # This returns 229,500

    mask = tabhdu.data['Z_PCA'] > 1.57
    mask = mask*(tabhdu.data['Z_PCA'] < 5.6)
    sum(mask)
    # This returns 296,090
    '''

    # Switched to use Z_PCA for DR16 
    #mask = tabhdu.data['Z'] >= bc.BAL_ZMIN
    #mask = mask*(tabhdu.data['Z'] <= bc.BAL_ZMAX)
    mask = tabhdu.data['Z_PCA'] >= bc.BAL_ZMIN
    mask = mask*(tabhdu.data['Z_PCA'] <= bc.BAL_ZMAX)
    print("BAL table has {0} rows".format(sum(mask)))
    # This returns 320,760 with '>' and 320,860 with '>='

    # Decreases length to 320,860
    tabhdu.data = tabhdu.data[mask]

    tabhdu.writeto(bc.catdir + outputfile)  # file size is about 332 Mb


# PM Note: replace file change with try/except, runtime errors
def findbaltab_sdss(qsocatfile, balcatfile):
    '''
    Determine if balcatfile already exists, and create it if it does not

    Parameters
    ----------
    qsocatfile : fitsfile
        name of QSO catalog file
    balcatfile : fitsfile
        name of BAL catalog file

    Returns
    -------
    balhdu : FITS HDU
        fits hdu for bal catalog
    '''
    if not os.path.isfile(bc.catdir + balcatfile):
        print("Warning: BAL catalog %s doesn't exist, initializing..." %
              bc.catdir + balcatfile)
        if not os.path.isfile(bc.catdir + qsocatfile):
            print("Error: QSO catalog %s does not exist. Could not initialize"
                  % bc.catdir + qsocatfile)
        initbaltab_sdss(qsocatfile, balcatfile)

    balhdu = fits.open(bc.catdir + balcatfile)

    return balhdu


def updatebaltab_sdss(sdssname, balhdu, info, pcaout):
    '''
    Add information about QSO with SDSS_NAME to balhdu

    Parameters
    ----------
    sdssname : string
        SDSS name
    balcatfile : fitsfile
        name of BAL catalog file
    info : record array
        output from fitbal.calcbalparams
    pcaout : 1-D float array
        output from fitbal.fitpca

    Returns
    -------
    balhdu : FITS HDU
        fits hdu for updated bal catalog
    '''

    qindx = fitbal.qsocatsearch(balhdu[1].data, sdssname=sdssname)

    balhdu[1].data[qindx]['PCA_COEFFS'] = pcaout[:bc.NPCA]
    balhdu[1].data[qindx]['PCA_CHI2'] = pcaout[-2]
    balhdu[1].data[qindx]['SDSS_CHI2'] = pcaout[-1]

    balhdu[1].data[qindx]['BI_CIV'] = info['BI_CIV']
    balhdu[1].data[qindx]['ERR_BI_CIV'] = info['BI_CIV_ERR']
    balhdu[1].data[qindx]['NCIV_2000'] = info['NCIV_2000']
    balhdu[1].data[qindx]['VMIN_CIV_2000'] = info['VMIN_CIV_2000']
    balhdu[1].data[qindx]['VMAX_CIV_2000'] = info['VMAX_CIV_2000']
    balhdu[1].data[qindx]['POSMIN_CIV_2000'] = info['POSMIN_CIV_2000']
    balhdu[1].data[qindx]['FMIN_CIV_2000'] = info['FMIN_CIV_2000']

    balhdu[1].data[qindx]['AI_CIV'] = info['AI_CIV']
    balhdu[1].data[qindx]['ERR_AI_CIV'] = info['AI_CIV_ERR']
    balhdu[1].data[qindx]['NCIV_450'] = info['NCIV_450']
    balhdu[1].data[qindx]['VMIN_CIV_450'] = info['VMIN_CIV_450']
    balhdu[1].data[qindx]['VMAX_CIV_450'] = info['VMAX_CIV_450']
    balhdu[1].data[qindx]['POSMIN_CIV_450'] = info['POSMIN_CIV_450']
    balhdu[1].data[qindx]['FMIN_CIV_450'] = info['FMIN_CIV_450']

    balhdu[1].data[qindx]['BI_SIIV'] = info['BI_SIIV']
    balhdu[1].data[qindx]['ERR_BI_SIIV'] = info['BI_SIIV_ERR']
    balhdu[1].data[qindx]['NSIIV_2000'] = info['NSIIV_2000']
    balhdu[1].data[qindx]['VMIN_SIIV_2000'] = info['VMIN_SIIV_2000']
    balhdu[1].data[qindx]['VMAX_SIIV_2000'] = info['VMAX_SIIV_2000']
    balhdu[1].data[qindx]['POSMIN_SIIV_2000'] = info['POSMIN_SIIV_2000']
    balhdu[1].data[qindx]['FMIN_SIIV_2000'] = info['FMIN_SIIV_2000']

    balhdu[1].data[qindx]['AI_SIIV'] = info['AI_SIIV']
    balhdu[1].data[qindx]['ERR_AI_SIIV'] = info['AI_SIIV_ERR']
    balhdu[1].data[qindx]['NSIIV_450'] = info['NSIIV_450']
    balhdu[1].data[qindx]['VMIN_SIIV_450'] = info['VMIN_SIIV_450']
    balhdu[1].data[qindx]['VMAX_SIIV_450'] = info['VMAX_SIIV_450']
    balhdu[1].data[qindx]['POSMIN_SIIV_450'] = info['POSMIN_SIIV_450']
    balhdu[1].data[qindx]['FMIN_SIIV_450'] = info['FMIN_SIIV_450']

    return balhdu


