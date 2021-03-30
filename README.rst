========
baltools
========

Introduction
------------

Tools to identify, characterize, and simulate BAL QSOs. 


Installation
------------

These tools include dependencies to several DESI packages, and it should be most straightforward to use them at NERSC. 

To install, clone the repository, add the "py" directory to your $PYTHONPATH, and the "bin" directory to your $PATH. 

Note the code expects that you have a directory called "Catalogs/" in your home directory that contains the file PCA_Eigenvectors.fits that is in the data/ directory of this repository.

Getting Started
---------------

See the notebook "DESI balfinder tutorial" in the "doc" directory for examples of how to run the balfinder on SV data and on mocks. 

Another option is to run "bin/runbalfinder.py" on the command line. This will run on all spectra for a given tile. There are various options, including to restrict the date range and spectrographs.


Tour of Code
------------

Here is a brief description of the files in the "py" directory:

**balconfig.py** : common modules and global variables used by balfinder

**baltable.py** : routines to construct BAL catalogs

**desibal.py** : main balfinder for DESI

**fitbal.py** : core code to identify and characterize BALs

**plotter.py** : various convenience plotting routines

**utils.py** : utilities for manipulating SDSS spectra

