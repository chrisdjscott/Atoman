
"""
Unit tests for the bond renderer

"""
import unittest

import numpy as np
import vtk
from vtk.util import numpy_support

from .. import bondRenderer
from ... import utils


################################################################################

# required unless ColouringOptions is rewritten to have a non GUI dependent settings object
class DummyColouringOpts(object):
    def __init__(self):
        self.colourBy = "Species"
        self.heightAxis = 1
        self.minVal = 0.0
        self.maxVal = 1.0
        self.solidColourRGB = (1.0, 0.0, 0.0)
        self.scalarBarText = "Height in Y (A)"

################################################################################

class TestBondCalculator(unittest.TestCase):
    """
    Test the bond calculator

    """
    def setUp(self):
        """
        Called before each test

        """
        

    def tearDown(self):
        """
        Called after each test

        """
        
    
    def test_bondCalculator(self):
        """
        Bond calculator
        
        """
        self.fail("Bond calculator not ready yet")

################################################################################

class TestBondRenderer(unittest.TestCase):
    """
    Test the bond renderer

    """
    def setUp(self):
        """
        Called before each test

        """
        

    def tearDown(self):
        """
        Called after each test

        """
        
    
    def test_bondRenderer(self):
        """
        Bond renderer
        
        """
        self.fail("Bond renderer not ready yet")
