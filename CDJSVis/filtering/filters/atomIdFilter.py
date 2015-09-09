
"""
Atom ID
=======

This filter is used to show only the atoms with the specified IDs. Mulitple
atom IDs can be separated by commas and ranges of atom IDs can be specified
using hyphens. For example:

.. code ::
    
    1-4,56,105-107

will show atoms: 1, 2, 3, 4, 56, 105, 106 and 107.

"""
from . import base
from . import _filtering

import numpy as np


class AtomIdFilterSettings(base.BaseSettings):
    """
    Settings for the atom ID filter
    
    """
    def __init__(self):
        super(AtomIdFilterSettings, self).__init__()
        
        self.registerSetting("filterString", default="")


class AtomIdFilter(base.BaseFilter):
    """
    The Atom ID filter.
    
    """
    def apply(self, filterInput, settings):
        """Apply the atom ID filter."""
        # unpack inputs
        inputState = filterInput.inputState
        NScalars = filterInput.NScalars
        fullScalars = filterInput.fullScalars
        NVectors = filterInput.NVectors
        fullVectors = filterInput.fullVectors
        visibleAtoms = filterInput.visibleAtoms
        
        # settings
        text = settings.getSetting("filterString")
        self.logger.debug("Atom ID raw text: '%s'", text)
        
        if not text:
            # return no visible atoms if input string was empty
            self.logger.warning("No visible atoms specified in AtomID filter")
            NVisible = 0
        
        else:
            # parse text
            array = [val for val in text.split(",") if val]
            num = len(array)
            rangeArray = np.empty((num, 2), np.int32)
            for i, item in enumerate(array):
                if "-" in item:
                    values = [val for val in item.split("-") if val]
                    minval = int(values[0])
                    if len(values) == 1:
                        maxval = minval
                    else:
                        maxval = int(values[1])
                else:
                    minval = maxval = int(item)
            
                self.logger.debug("  %d: %d -> %d", i, minval, maxval)
                rangeArray[i][0] = minval
                rangeArray[i][1] = maxval
        
            # run displacement filter
            NVisible = _filtering.atomIndexFilter(visibleAtoms, inputState.atomID, rangeArray, 
                                                   NScalars, fullScalars, NVectors, fullVectors)
        
        # resize visible atoms
        visibleAtoms.resize(NVisible, refcheck=False)
        
        # construct result
        result = base.FilterResult()
        
        return result
