"""

baltools.initcreate
===================

Contains functions that add BAL columns to an existing QSO catalogue and
populate empty columns for a given object with information attained from
runbalfinder.py tables

2021 Original code by Simon Filbert

concatbaltabs()  : join  bal tables from runbalfinder.py into one fits file
inittab()        : add empty columns for BAL info to existing QSO catalogue
popqsocat()      : populate empty columns with BAL information 

"""


import os
import sys
import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
import fitsio

sys.path.append("/global/homes/s/simonmf/baltools/py")
from baltools import balconfig as bc
from baltools import fitbal


def concatbaltabs(rootbaldir, outtab):
    '''
    Stack rows from seperate bal tables to create one
    fits file with all objects in contained in rootbaldir.

    Parameters
    ----------
    rootbaldir : str
        path to root directory containing bal tables
       
    outtab  : fits file
        where to write resulting catalogue to.

    Returns
    -------
    None
    
    '''
    
    tiles = os.listdir(rootbaldir)
    
    baltabpaths = []

    for tile in tiles:
        tilepath = os.path.join(rootbaldir, tile)
        nights = os.listdir(tilepath)
        for night in nights:
            nightpath = os.path.join(tilepath, night)

            fitsfiles = os.listdir(nightpath)
            for fitsfile in fitsfiles:
                specificfilepath = os.path.join(nightpath, fitsfile)
                baltabpaths.append(specificfilepath)
                
    baltables = []
    for baltabpath in baltabpaths:
        baltab = Table.read(baltabpath)
        if len(baltab) > 0:
            tables.append(baltab)

    concatbaltab = vstack(baltables)
    
    try:
        concatbaltab.write(outtab , format = 'fits')
    except OSError:
        print(f"File {outtab} already exists")
        
    print("BAL tables located under {} stacked into new table. BAL information table was written to {}.".format(rootbaldir, outtab))
        


def inittab(qsocatpath, outtab, alttemp=False, truthlist=False): 
    '''
    Populate table with information from QSO catalogue,
    make empty tables for BAL information to be added later.

    Parameters
    ----------
    qsocatpath : fits file
        QSO catalogue to be read.
    outtab  : fits file
        where to write resulting catalogue to.
    alttemp : bool
        use alternate (Brodzeller) templates? (default False) 
    truthlist : bool
        use shorter list when combining mock qso and truth catalogs

    Returns
    -------
    colnames : list
        card names relating to BALs
    
    '''
    
    #Open input catalogue fits file to get num objects in catalogue.
    cathdu = fits.open(qsocatpath, lazy_load_hdus=False)
    NROWS  = len(cathdu[1].data)

    # Check if strings have been turned into character arrays, and if so read the catalog with fitsio instead
    # This is necessary for fuji and guadalupe
    try: 
        if cathdu[1].data['SURVEY'].dtype == 'O':
            cathdu[1].data = fitsio.read(qsocatpath) 
    except KeyError:
        print(f"Warning: no SURVEY data column in {qsocatpath}")
    
    # PCA Fit Coefficients and chisq result
    if alttemp: 
        pca_array = np.zeros([NROWS, 4], dtype=float)  # PCA Fit Coefficients
        col0 = fits.Column(name='PCA_COEFFS', format='4E', array=pca_array)
    else: 
        pca_array = np.zeros([NROWS, bc.NPCA], dtype=float)  # PCA Fit Coefficients
        col0 = fits.Column(name='PCA_COEFFS', format='5E', array=pca_array)

    pca_chi2 = np.zeros([NROWS], dtype=float)  # PCA reduced chi2
    col1 = fits.Column(name='PCA_CHI2', format='E', array=pca_chi2)

    # Collection of empty arrays to set up rest of the BAL catalog
    # PM Note: Might change to -1. to indicate empty field
    zfloat_col = np.zeros([NROWS], dtype=float)
    zint_col = np.zeros([NROWS], dtype=float)
    zneg_col = np.array([-99]*NROWS, dtype=float) # For BAL_PROB
    zbyte_col = np.ones(NROWS, dtype=np.ubyte) # For bit masking in BALMASK
    zfloat_bicol = np.zeros([NROWS, bc.NBI], dtype=float) # Second dimension is max number of BI troughs
    zfloat_aicol = np.zeros([NROWS, bc.NAI], dtype=float) # Second dimension is max number of AI troughs

    # This will come from CNN Classifier
    # All set to -99 to indicate that data from BALcats not been read yet
    # Will be set to -1 once updated
    col2 = fits.Column(name='BAL_PROB', format='E', array=zneg_col)

    # These quantities are from trough fit
    col3 = fits.Column(name='BI_CIV', format='E', array=zfloat_col)
    col4 = fits.Column(name='ERR_BI_CIV', format='E', array=zfloat_col)
    col5 = fits.Column(name='NCIV_2000', format='J', array=zint_col)
    col6 = fits.Column(name='VMIN_CIV_2000', format='5E', array=zfloat_bicol)
    col7 = fits.Column(name='VMAX_CIV_2000', format='5E', array=zfloat_bicol)
    col8 = fits.Column(name='POSMIN_CIV_2000', format='5E', array=zfloat_bicol)
    col9 = fits.Column(name='FMIN_CIV_2000', format='5E', array=zfloat_bicol)

    col10 = fits.Column(name='AI_CIV', format='E', array=zfloat_col)
    col11 = fits.Column(name='ERR_AI_CIV', format='E', array=zfloat_col)
    col12 = fits.Column(name='NCIV_450', format='J', array=zint_col)
    col13 = fits.Column(name='VMIN_CIV_450', format='17E', array=zfloat_aicol)
    col14 = fits.Column(name='VMAX_CIV_450', format='17E', array=zfloat_aicol)
    col15 = fits.Column(name='POSMIN_CIV_450', format='17E', array=zfloat_aicol)
    col16 = fits.Column(name='FMIN_CIV_450', format='17E', array=zfloat_aicol)

    col17 = fits.Column(name='BI_SIIV', format='E', array=zfloat_col)
    col18 = fits.Column(name='ERR_BI_SIIV', format='E', array=zfloat_col)
    col19 = fits.Column(name='NSIIV_2000', format='J', array=zint_col)
    col20 = fits.Column(name='VMIN_SIIV_2000', format='5E', array=zfloat_bicol)
    col21 = fits.Column(name='VMAX_SIIV_2000', format='5E', array=zfloat_bicol)
    col22 = fits.Column(name='POSMIN_SIIV_2000', format='5E', array=zfloat_bicol)
    col23 = fits.Column(name='FMIN_SIIV_2000', format='5E', array=zfloat_bicol)

    col24 = fits.Column(name='AI_SIIV', format='E', array=zfloat_col)
    col25 = fits.Column(name='ERR_AI_SIIV', format='E', array=zfloat_col)
    col26 = fits.Column(name='NSIIV_450', format='J', array=zint_col)
    col27 = fits.Column(name='VMIN_SIIV_450', format='17E', array=zfloat_aicol)
    col28 = fits.Column(name='VMAX_SIIV_450', format='17E', array=zfloat_aicol)
    col29 = fits.Column(name='POSMIN_SIIV_450', format='17E', array=zfloat_aicol)
    col30 = fits.Column(name='FMIN_SIIV_450', format='17E', array=zfloat_aicol)
    
    # Seperate column not populated in runbalfinder which serves as a bitmask
    col31 = fits.Column(name='BALMASK', format='B', array=zbyte_col) # default is '1' (not found in baltable)
    col32 = fits.Column(name='SNR_CIV', format='E', array=zfloat_col)

    
    # Columns relating to BAL information from runbalfinder
    PCAcols = fits.ColDefs([col0, col1]) 
    CIVcols = fits.ColDefs([col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15, col16])
    SiIVcols = fits.ColDefs([col17, col18, col19, col20, col21, col22, col23, col24, col25, col26, col27, col28, col29, col30])
    
    #Columns already contained in QSO catalogue
    catcols = cathdu[1].columns
    
    if truthlist: 
        BALcols = col2 + CIVcols 
    else: 
        BALcols  = PCAcols + col2 + CIVcols + SiIVcols + col31 + col32

    totcols = catcols + BALcols 
    
    #List of card names in BinHDU relating to BALs
    #Will beiterated through in initcreate.popqsotab()
    balcolnames = BALcols.info('name', False)['name']
    
    
    #Create a new BinHDU with all columns
    newbhdu = fits.BinTableHDU.from_columns(totcols)
    #Update BinHDU of cathdu with newbhdu
    cathdu[1] = newbhdu 
    cathdu[1].header['EXTNAME'] = 'ZCATALOG'

    try:
        cathdu.writeto(outtab)
    except OSError:
        print(f"File {outtab} already exists")
        return balcolnames
        
    print(f"Empty BAL columns added to input QSO catalogue at {qsocatpath} and written to {outtab}")
    
    return balcolnames 

  
def addbalinfo(cattab, rootbaldir, cindx, colnames, overwrite=False, verbose=False):
    '''
    Write BAL information to QSO catalogue table with BAL information.
    
    Parameters
    ----------
    
    cattab : fits file
        catalogue that BAL information is to be added to.
    rootbaldir : str
        path to root directory containing bal tables
    cindx : int
        index of targetid in qso catalogue        
    colnames : list, strings
        list of columns in fits file which are to contain BAL info.
    overwrite : bool
        overwrite current table (if exists)? (default is False)
    verbose : bool
        provide verbose output? (default is False)
    
    Returns
    -------
    none
    
    
    Bitmask definitions
    -------------------

    0       Successfully ran balfinder 
    1       Not found in a baltable
    2       Out of the redshift range to identify BALs
    4       Redshift difference between QSO catalog and baltable

    '''
    
    cathdu = fits.open(cattab, mode='update')
    
    # Indices of objects in BAL info catalogue and QSO catalogue 
    # are not necessarily the same.
    targetid = cathdu[1].data['TARGETID'][cindx]
   
    tile  = str(cathdu[1].data['TILE'][cindx])
    # runbalfinder.py uses last_night card to specify observation night
    # as of (last check) 06/24/2021
    night = str(cathdu[1].data['LAST_NIGHT'][cindx])
    spec  = str(cathdu[1].data['PETAL'][cindx])
    
    if cathdu[1].data['BAL_PROB'][cindx] == -1 and not overwrite:
        print("BAL information already exists for targetid ", str(targetid), " and overwrite == False. Moving to next target.")
        print(cindx)
        return
    
    # Find bal table for specific object, open corresponding fits file
    baltabpath = os.path.join(rootbaldir, tile, night, "baltable-{}-{}-thru{}.fits".format(spec, tile, night))
    balhdu = fits.open(baltabpath)

    if verbose:
            print("BAL information for target ", str(targetid), " being read from ", str(baltabpath))
            
    bindx  = 0
    intabs = False
    
    # IndexError may imply that the target is in the QSO cat but not in the BAL cat. 
    # Below populated empty bal columns and adds bitmasks for classifications.
    if targetid in balhdu[1].data['TARGETID']: 
        intabs = True
        bindx = np.where(balhdu[1].data['TARGETID'] == targetid)[0][0]
            # Read in BAL information from BAL table, populate QSO catalogue with this info.
        for colname in colnames:
            cathdu[1].data[colname][cindx] = balhdu[1].data[colname][bindx]

            # Indicates that BAL data was added to the catalogue, but BAL_PROB is still an unpopulated field.
            cathdu[1].data['BAL_PROB'][cindx] = -1 
            cathdu[1].data['BALMASK'][cindx]  = 0 # Indicates object is in baltable and in redshift range 

                
    # NOTE: Add reference to zbest file for object to check whether object is QSO and that is why
    # it is not in redshift range, or if it is simply not in the catalogue.
    
    else: # if not found in bal table for any reason
        if verbose:
            print("Target ",  str(targetid), " not found in bal tables. Information not added")
        cathdu[1].data['BAL_PROB'][cindx] = -1     
        cathdu[1].data['BALMASK'][cindx]  = 1
    
    # Checks if z of object in catdata is outside of z range for runbalfinder for bit mask
    if cathdu[1].data['Z'][cindx] > bc.BAL_ZMAX or cathdu[1].data['Z'][cindx] < bc.BAL_ZMIN:
        if verbose:
            print("Target ",  str(targetid), " not in redshift range.") 
        cathdu[1].data['BALMASK'][cindx] += 2   
    # If objects are in z range and still in catalogue, this suggests that the object is not identified
    # as a QSO by RR.
    
    ztol = 0.001 # Tolarance for differences in z between catalogues
    zc   = cathdu[1].data['Z'][cindx] # z from qso catalogue for object
    if intabs: # If the object is not in bal tables, the following lines don't even make sense to do.
        zb   = balhdu[1].data['Z'][bindx] # z from bal table for object
        dz   = abs(zc - zb) 
        if intabs and dz > ztol:
            if verbose:
                print("Target ", str(targetid), " has significant difference in redshifts between BAL table and qso cat.") 
                print("Redshift in BAL table is ", zb, "while redshift in catalogue is ", zc)
            cathdu[1].data['BALMASK'][cindx] += 4 # Indicates large z error between catalogues
        
    # Keep track of number of objects ran
    print(cindx)
    #balhdu.close()
    #cathdu.close()
