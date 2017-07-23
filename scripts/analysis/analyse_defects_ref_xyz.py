#!/usr/bin/env python

"""
This is an example of how to run the defect analysis from a script instead of
the GUI. The steps we take are:

    1. Read in the file formats
    2. Load the reference file (with the appropriate format)
    3. Load the input file (with the appropriate format)
    4. Create and configure the filter list
    5. Run the filter list

@author: Chris Scott

"""
from __future__ import print_function
import os
import sys

# this line forces the use of this atoman (not required if atoman is installed, e.g. with PYTHONPATH)
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir, os.pardir))
print(os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir, os.pardir))

# setup the logger
import logging
logging.basicConfig(level=logging.DEBUG)  # this can be changed to `logging.INFO` to reduce output

from atoman.system import latticeReaderGeneric
from atoman.filtering import filterer
from atoman.filtering.filters import pointDefectsFilter


def main():
    #
    # read in the file formats
    #
    fileFormats = latticeReaderGeneric.FileFormats()
    fileFormats.read()

    #
    # create a generic file reader object that will be used to read in all files
    #
    reader = latticeReaderGeneric.LatticeReaderGeneric()

    #
    # read in the reference file
    #
    refFile = os.path.join(os.pardir, os.pardir, "testing", "animation-reference.xyz.gz")  # path to the animation-reference file
    refFormat = fileFormats.getFormat("LBOMD REF")  # "LBOMD REF" is the format for loading LBOMD "animation-reference.xyz" format files
    status, refLattice = reader.readFile(refFile, refFormat, rouletteIndex=None, linkedLattice=None)
    if status:
        raise RuntimeError("Failed to read in the reference lattice")

    #
    # read in the xyz input file
    #
    xyzFile = os.path.join(os.pardir, os.pardir, "testing", "PuGaH0080.xyz")  # path to the XYZ file
    xyzFormat = fileFormats.getFormat("LBOMD XYZ")  # "LBOMD XYZ" is the format for loading LBOMD "xyz" format files
    status, inputLattice = reader.readFile(xyzFile, xyzFormat, linkedLattice=refLattice)  # XYZ files must be linked to another lattice that provides the species information, typically an animation-reference
    if status:
        raise RuntimeError("Failed to read in the input lattice")

    #
    # create the filter pipeline
    #
    defectsSettings = pointDefectsFilter.PointDefectsFilterSettings()  # settings for the point defects filter
    defectsSettings.updateSetting("showAntisites", False)
    defectsSettings.updateSetting("useAcna", True)  # use adaptive common neighbour analysis to refine defect detection
    defectsSettings.updateSetting("acnaStructureType", 1)  # FCC - see structure types in `atoman.filtering.atomStructure`

    filterNames = ["Point defects"]  # list of filters that will be applied (in order)
    filterSettings = [defectsSettings]  # corresponding list of settings

    pipeline = filterer.Filterer(None)  # the object that does the filtering

    pipeline.toggleDriftCompensation(True)  # enable drift compensation (atoms in reference and input must correspond to one another)

    pipeline.runFilters(filterNames, filterSettings, inputLattice, refLattice)  # filter - usually this is called multiple times with a different `inputLattice`

    # results
    # indices of interstitials in `inputLattice` are stored in `pipeline.interstitials`
    # indices of vacancies in `refLattice` are stored in `pipeline.vacancies`
    # split interstitials are stored in `pipeline.splitInterstitials`, for each split we store the vacancy index first and then the two interstitial indices
    nint = len(pipeline.interstitials) + len(pipeline.splitInterstitials) / 3
    nvac = len(pipeline.vacancies)
    print("Number of interstitials is: {0}".format(nint))
    print("Number of vacancies is: {0}".format(nvac))

    # more detailed information is available to, for example counters for defects, species, etc...


if __name__ == "__main__":
    main()
