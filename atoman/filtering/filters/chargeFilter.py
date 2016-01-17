
"""
Charge
======

The charge filter allows you to filter atoms by their charge.

The parameters are:

.. glossary::
    
    Minimum charge
        The minimum charge for an atom to be visible.
    
    Maximum charge
        The maximum charge for an atom to be visible.

"""
from __future__ import absolute_import
from __future__ import unicode_literals
from . import base
from . import _filtering


class ChargeFilterSettings(base.BaseSettings):
    """
    Settings for the charge filter
    
    """
    def __init__(self):
        super(ChargeFilterSettings, self).__init__()
        
        self.registerSetting("minCharge", default=-100.0)
        self.registerSetting("maxCharge", default=100.0)


class ChargeFilter(base.BaseFilter):
    """
    The charge filter.
    
    """
    def apply(self, filterInput, settings):
        """Apply the charge filter."""
        # unpack inputs
        lattice = filterInput.inputState
        NScalars = filterInput.NScalars
        fullScalars = filterInput.fullScalars
        NVectors = filterInput.NVectors
        fullVectors = filterInput.fullVectors
        visibleAtoms = filterInput.visibleAtoms
        
        # settings
        minCharge = settings.getSetting("minCharge")
        maxCharge = settings.getSetting("maxCharge")
        self.logger.debug("Visible charge range: %f -> %f", minCharge, maxCharge)
        
        # call C lib
        self.logger.debug("Calling charge filter C library")
        NVisible = _filtering.chargeFilter(visibleAtoms, lattice.charge, minCharge, maxCharge, 
                                           NScalars, fullScalars, NVectors, fullVectors)

        # resize visible atoms
        visibleAtoms.resize(NVisible, refcheck=False)
        
        # result
        result = base.FilterResult()
        
        return result
