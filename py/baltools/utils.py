"""

baltools.utils
==============

Module with tools for file IO, download, etc. for analysis of SDSS spectra

2018 by PM

"""
from __future__ import print_function, absolute_import, division

import os
import numpy as np
import urllib
import json
from astropy.io import fits
import fitsio

from baltools import balconfig as bc

seguelist = np.asarray([1960, 1961, 1962, 1963, 2078, 2079, 2174,
                        2185, 2247, 2255, 2256, 2333, 2338, 2377,
                        2475, 2476, 2667, 2671, 2675, 2800, 2821,
                        2887, 2912], dtype=int)


def zeropad(input, N=4):
    '''
    Extend the size of an input string of integers to length N
    '''

    if type(input) == str: 
        if N == 3: 
            output = "{:03d}".format(int(input))
        elif N == 4: 
            output = "{:04d}".format(int(input))
        elif N == 5: 
            output = "{:05d}".format(int(input))
        else: 
            print("Error: Only works for N = 3 to 5")
            output = str(input)
    else: 
        print("Error: Argument must be a string")
        output = str(input)
    return output


def gethpdir(healpix):
    '''
    Correctly parse a healpix to parent directory in the format
        .../hpdir/healpix/...
    where healpix is a string
    '''
    if len(healpix) < 3:
        hpdir = '0'
    elif len(healpix) == 3:
        hpdir = healpix[0]
    elif len(healpix) == 4:
        hpdir = healpix[0:2]
    else:
        hpdir = healpix[:len(healpix)-2]
    return hpdir


def pmmkdir(direct):
    '''
    Create a new directory, capture permissions issues
    '''
    if not os.path.isdir(direct):
        try:
            print(direct, "not found. Making new directory.")
            os.makedirs(direct)
        except PermissionError:
            print("Error: no permission to make directory ", direct)
            exit(1)


def getdr12spec(array, verbose=False):
    '''
    Return the HDU associated with QSO described with array
    array must include 'PLATE', 'FIBERID', 'MJD', and 'Z_VI'
    Open file in directory SPECDIR
    If not present, retrieve it using urllib.request.urlretrieve()
    '''

    detector = 'boss'
    run2d = 'v5_7_0'
    plate4 = str(array['PLATE'])
    fiberid4 = str(array['FIBERID'])
    mjd = array['MJD']
    plate4 = zeropad(plate4)
    fiberid4 = zeropad(fiberid4)
    specfits = "spec-%s-%s-%s.fits" % (plate4, mjd, fiberid4)
    specurl = "http://dr12.sdss3.org/sas/dr12/%s/spectro/redux/%s/spectra/%s/%s" % (detector, run2d, plate4, specfits)
    try:
        specfile = os.environ.get('SPECDIR')+specfits
        if verbose:
            print ("Found %s on disk" % specfits)
    except:
        urllib.request.urlretrieve(specurl, specfits)
        if verbose:
            print ("Downloaded %s" % specfits)
    spec = fitsio.read(specfile)
    return spec


def scidrivePublicURL(path):
    ''' Get the path to a file on SciDrive '''
    req = urllib.request.Request(url=SciServer.Config.SciDriveHost+'/vospace-2.0/1/media/sandbox/'+path,method='GET') 
    req.add_header('X-Auth-Token', token)
    req.add_header('Content-Type','application/xml')
    res=urllib.request.urlopen(req)
    jsonResponse = json.loads(res.read().decode())
    return jsonResponse['url']


def getdr14cat(): 
    '''
    Check to see if the DR14 QSO catalog already exists locally.
    Download it if it does not
    '''
    qso14file = "DR14Q_v4_4.fits"
    qso14url = "https://data.sdss.org/sas/dr14/eboss/qso/DR14Q/DR14Q_v4_4.fits"
    if os.path.isfile(qso14file):
        print("%s already downloaded" % qso14file)
    else:
        print("Downloading %s ..." % qso14file)
        urllib.request.urlretrieve(qso14url, qso14file)

def getdr14spectra(array, verbose=True):
    '''
    Download the spectra in catalog file HDU if they are not downloaded
    already
    Use urllib.request.urlretrieve()

    verbose : bool
        turn on or off some progress reporting

    '''

    seguelist = np.asarray([1960, 1961, 1962, 1963, 2078, 2079, 2174,
                            2185, 2247, 2255, 2256, 2333, 2338, 2377,
                            2475, 2476, 2667, 2671, 2675, 2800, 2821,
                            2887, 2912], dtype=int)
    detector = 'eboss'
    run2d = 'v5_10_0'
    try: 
        spectro = str(array['SPECTRO'])
    except KeyError: 
        spectro = 'BOSS'
    except ValueError: 
        spectro = 'BOSS'
    plate4 = str(array['PLATE'])
    fiberid4 = str(array['FIBERID'])
    mjd = array['MJD']
    z = array['Z']
    if spectro == 'SDSS':
        run2d = '26'
        detector = 'sdss'
        if int(plate4) in seguelist:
            run2d = '103'
    if plate4 == '2865':
        mjd = '54503'
    if plate4 == '2516':
        mjd = '54241'
    if plate4 == '2812':
        mjd = '54639'
    plate4 = zeropad(plate4)
    fiberid4 = zeropad(fiberid4)

    specfits = "spec-%s-%s-%s.fits" % (plate4, mjd, fiberid4)
    specurl = "https://dr14.sdss.org/sas/dr14/%s/spectro/redux/%s/spectra/lite/%s/%s" \
    % (detector, run2d, plate4, specfits)
    if verbose: 
        print(specurl)

    # if at NERSC, all data should be on disk
    if '/global/homes' in bc.homedir:
        if os.path.isdir(bc.specdir1 + plate4):  # First search in DR14 directories
            try: 
                specfile = bc.specdir1 + plate4 + '/' + specfits
                # spec = fitsio.read(specfile)
                spec = fits.open(specfile)
                if verbose:
                    print ("Found %s on disk" % specfits)
            except: 
                raise FileNotFoundError("Couldn't find file {0}".format(specfits)) 
        else: # Second search in DR9 directories
            if os.path.isdir(bc.specdir2 + '26/spectra/lite/' + plate4):  
                try: 
                    specfile = bc.specdir2 + '26/spectra/lite/' + plate4 + '/' + specfits
                    # spec = fitsio.read(specfile)
                    spec = fits.open(specfile)
                    if verbose:
                        print ("Found %s on disk" % specfits)
                except: 
                    raise FileNotFoundError("Couldn't find file {0}".format(specfits)) 
            elif os.path.isdir(bc.specdir2 + '103/spectra/lite/' + plate4):  
                try: 
                    specfile = bc.specdir2 + '103/spectra/lite/' + plate4 + '/' + specfits
                    # spec = fitsio.read(specfile)
                    spec = fits.open(specfile)
                    if verbose:
                        print ("Found %s on disk" % specfits)
                except: 
                    raise FileNotFoundError("Couldn't find file {0}".format(specfits)) 
            elif os.path.isdir(bc.specdir2 + '104/spectra/lite/' + plate4):  
                try: 
                    specfile = bc.specdir2 + '104/spectra/lite/' + plate4 + '/' + specfits
                    #spec = fitsio.read(specfile)
                    spec = fits.open(specfile)
                    if verbose:
                        print ("Found %s on disk" % specfits)
                except: 
                    raise FileNotFoundError("Couldn't find file {0}".format(specfits)) 
            else: 
                raise FileNotFoundError("Couldn't find file {0}".format(specfits)) 
    else: # Download as needed 
        # Create directory for raw data if needed
        try:
            os.mkdir(bc.specdir + plate4 + '/')
            if verbose:
                print ("Created {0}".format(bc.specdir + plate4 + '/'))
        except FileExistsError:
            if verbose:
                print ("Directory {0} exists".format(bc.specdir + plate4 + '/'))
        except PermissionError: # for Utah
            if verbose:
                print ("Directory {0} exists".format(bc.specdir + plate4 + '/'))
    
    
        # Download spectra if needed
        try:
            specfile = bc.specdir + plate4 + '/' + specfits
            # spec = fitsio.read(specfile)
            spec = fits.open(specfile)
            if verbose:
                print ("Found %s on disk" % specfits)
        except FileNotFoundError:
            urllib.request.urlretrieve(specurl, bc.specdir+ plate4 + '/' + specfits)
            # spec = fitsio.read(specfile)
            spec = fits.open(specfile)
            if verbose:
                print("Downloaded %s" % specfits)

    return spec

def getdr16spectra(array, verbose=True):
    '''
    Find the DR16 spectra at Utah

    verbose : bool
        turn on or off some progress reporting

    '''

    seguelist = np.asarray([1960, 1961, 1962, 1963, 2078, 2079, 2174,
                            2185, 2247, 2255, 2256, 2333, 2338, 2377,
                            2475, 2476, 2667, 2671, 2675, 2800, 2821,
                            2887, 2912], dtype=int)
    detector = 'eboss'
    run2d = 'v5_10_0'
    try: 
        spectro = str(array['SPECTRO'])
    except KeyError: 
        spectro = 'BOSS'
    except ValueError: 
        spectro = 'BOSS'
    plate4 = str(array['PLATE'])
    fiberid4 = str(array['FIBERID'])
    mjd = array['MJD']
    z = array['Z']
    if spectro == 'SDSS':
        run2d = '26'
        detector = 'sdss'
        if int(plate4) in seguelist:
            run2d = '103'
    if plate4 == '2865':
        mjd = '54503'
    if plate4 == '2516':
        mjd = '54241'
    if plate4 == '2812':
        mjd = '54639'
    plate4 = zeropad(plate4)
    fiberid4 = zeropad(fiberid4)

    try: 
        ebossdir = bc.specdir
        sdssdir = bc.specdir
    except AttributeError: 
        try: 
            ebossdir = bc.specdir1
            sdssdir = bc.specdir2
            bc.specdir = bc.specdir1
        except:
            print("Error with ebossdir") 

    specfits = "spec-%s-%s-%s.fits" % (plate4, mjd, fiberid4)
    if verbose: 
        print(specfits)

    if os.path.isdir(bc.specdir + plate4): 
        try: 
            specfile = bc.specdir + plate4 + '/' + specfits
            # spec = fitsio.read(specfile)
            spec = fits.open(specfile)
            if verbose:
                print ("Found %s on disk" % specfits)
        except: 
            raise FileNotFoundError("Couldn't find file {0}".format(specfits)) 
    else: 
        if os.path.isdir(sdssdir + '26/spectra/lite/' + plate4):  
            try: 
                specfile = sdssdir + '26/spectra/lite/' + plate4 + '/' + specfits
                # spec = fitsio.read(specfile)
                spec = fits.open(specfile)
                if verbose:
                    print ("Found %s on disk" % specfits)
            except: 
                raise FileNotFoundError("Couldn't find file {0}".format(specfits)) 
        elif os.path.isdir(sdssdir + '103/spectra/lite/' + plate4):  
            try: 
                specfile = sdssdir + '103/spectra/lite/' + plate4 + '/' + specfits
                # spec = fitsio.read(specfile)
                spec = fits.open(specfile)
                if verbose:
                    print ("Found %s on disk" % specfits)
            except: 
                raise FileNotFoundError("Couldn't find file {0}".format(specfits)) 
        elif os.path.isdir(sdssdir + '104/spectra/lite/' + plate4):  
            try: 
                specfile = sdssdir + '104/spectra/lite/' + plate4 + '/' + specfits
                # spec = fitsio.read(specfile)
                spec = fits.open(specfile)
                if verbose:
                    print ("Found %s on disk" % specfits)
            except: 
                raise FileNotFoundError("Couldn't find file {0}".format(specfits)) 
        else: 
            raise FileNotFoundError("Couldn't find file {0}".format(specfits)) 
        
    return spec

