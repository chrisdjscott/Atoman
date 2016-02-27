
"""
Module for computing the radial distribution function.

"""
from __future__ import absolute_import
from __future__ import unicode_literals
import logging

import numpy as np

from . import _rdf


class RDFCalculator(object):
    """
    Object for calculating the radial distribution function.
    
    """
    def calculateRDF(self, visibleAtoms, inputLattice, binMin, binMax, binWidth,
                     speciesIndex1, speciesIndex2):
        """
        Calculate the radial distribution function for the given selection of atoms.
        
        Parameters
        ----------
        visibleAtoms : numpy.ndarray
            Array containing the indices of atoms that are currently visible.
        inputLattice : Lattice
            Lattice object containing positions of atoms, etc.
        binMin : float
            The minimum separation to consider when constructing the histogram.
        binMin : float
            The maximum separation to consider when constructing the histogram.
        binWidth : float
            The separation between bins.
        speciesIndex1 : int
            The index of species of atom in the first selection.
        speciesIndex2 : int
            The index of species of atom in the second selection.
        numThreads : int, optional
            The number of threads to run on (default is to use all available).
        
        """
        # logger
        logger = logging.getLogger(__name__)
        logger.debug("Calculating RDF")
        
        # check system size / parameters
        warnDims = []
        if inputLattice.PBC[0] and binMax > inputLattice.cellDims[0] / 2.0:
            warnDims.append("x")
        if inputLattice.PBC[1] and binMax > inputLattice.cellDims[1] / 2.0:
            warnDims.append("y")
        if inputLattice.PBC[2] and binMax > inputLattice.cellDims[2] / 2.0:
            warnDims.append("z")
        if len(warnDims):
            msg = "The maximum radius you have requested is greater than half the box length "
            msg += "in the %s periodic direction(s)!" % ",".join(warnDims)
            raise RuntimeError("RDFCalculator: %s" % msg)

        # the number of bins in the histogram
        numBins = int((binMax - binMin) / binWidth)
        logger.debug("Bin width is %f; number of bins is %d", binWidth, numBins)
        
        # create array for storing the result
        rdfArray = np.zeros(numBins, np.float64)
        
        # call the C extension to calculate the RDF
        _rdf.calculateRDF(visibleAtoms, inputLattice.specie, inputLattice.pos, speciesIndex1,
                          speciesIndex2, inputLattice.cellDims, inputLattice.PBC, binMin,
                          binMax, binWidth, numBins, rdfArray)
        
        # x values for plotting the RDF
        xvals = np.arange(binMin + binWidth / 2.0, binMax, binWidth, dtype=np.float64)
        
        return xvals, rdfArray
