
"""
Rendering utils

@author: Chris Scott

"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
import ctypes as C
import logging

import numpy as np
import vtk
from vtk.util import numpy_support
from six.moves import range


################################################################################

class NumpyVTKData(object):
    """
    Hold paired numpy and VTK data.
    
    """
    def __init__(self, data_numpy, name=None):
        self._numpy = data_numpy
        self._vtk = numpy_support.numpy_to_vtk(data_numpy)
        if name is not None:
            self._vtk.SetName(name)
    
    def getVTK(self):
        """Return VTK array."""
        return self._vtk
    
    def getNumpy(self):
        """Return Numpy array."""
        return self._numpy

################################################################################

class ActorObject(object):
    """
    Holds a VTK actor and a boolean to say whether it is loaded
    
    """
    def __init__(self, actor):
        self.actor = actor
        self.visible = False

################################################################################

def getScalarsType(colouringOptions):
    """
    Return scalars type based on colouring options
    
    """
    # scalar type
    if colouringOptions.colourBy == "Species" or colouringOptions.colourBy == "Solid colour":
        scalarType = 0
    
    elif colouringOptions.colourBy == "Height":
        scalarType = 1
    
    elif colouringOptions.colourBy == "Charge":
        scalarType = 4
    
    else:
        scalarType = 5
    
    return scalarType

################################################################################

# call back class
class RGBCallBackClass(object):
    CFUNCTYPE = C.CFUNCTYPE(C.c_long, C.c_int)
    
    def __init__(self, lut):
        self._lut = lut
        self._rgb = np.empty(3, np.float64)
    
    def __call__(self, scalar):
        """
        Calculate RGB
    
        """
        # rgb array
        self._rgb = np.empty(3, np.float64)
        
        # colour
        self._lut.GetColor(scalar, self._rgb)
        
        print("RGB CALLBACK; scalar %f; rgb %r" % (scalar, self._rgb))
        
        return self._rgb.ctypes.data_as(C.c_void_p).value

    def getcfunc(self):
        return self.CFUNCTYPE(self)

    cfunc = property(getcfunc)

################################################################################

def makeScalarBar(lut, colouringOptions, text_colour, renderingPrefs):
    """
    Make a scalar bar
    
    """
    scalarBar = vtk.vtkScalarBarActor()
    scalarBar.SetLookupTable(lut)
    
    if colouringOptions.colourBy == "Height":
        title = colouringOptions.scalarBarText
    elif colouringOptions.colourBy == "Charge":
        title = str(colouringOptions.scalarBarTextEdit3.text())
    else:
        title = str(colouringOptions.scalarBarTexts[colouringOptions.colourBy].text())
    
    scalarBar.SetTitle(title)
    scalarBar.SetOrientationToHorizontal()
    scalarBar.SetNumberOfLabels(renderingPrefs.numScalarBarLabels)
    if renderingPrefs.enableFmtScalarBarLabels:
        scalarBar.SetLabelFormat(renderingPrefs.fmtScalarBarLabels)
    
    lprop = scalarBar.GetTitleTextProperty()
    lprop.SetColor(text_colour)
    lprop.ItalicOff()
    lprop.BoldOn()
    lprop.SetFontSize(20)
    lprop.SetFontFamilyToArial()
    
    lprop = scalarBar.GetLabelTextProperty()
    lprop.SetColor(text_colour)
    lprop.ItalicOff()
    lprop.BoldOn()
    lprop.SetFontSize(10)
    lprop.SetFontFamilyToArial()
    
    scalarBar.SetWidth(0.85)
    scalarBar.GetPositionCoordinate().SetValue(0.1, 0.01)
    scalarBar.SetHeight(0.12)
    
    return scalarBar

################################################################################

def setMapperScalarRange(mapper, colouringOptions, NSpecies):
    """
    Set scalar range on mapper
    
    """
    if colouringOptions.colourBy == "Species" or colouringOptions.colourBy == "Solid colour":
        mapper.SetScalarRange(0, NSpecies - 1)
    
    elif colouringOptions.colourBy == "Height":
        mapper.SetScalarRange(colouringOptions.minVal, colouringOptions.maxVal)
    
    elif colouringOptions.colourBy == "Charge":
        mapper.SetScalarRange(colouringOptions.chargeMinSpin.value(), colouringOptions.chargeMaxSpin.value())
    
    else:
        mapper.SetScalarRange(colouringOptions.scalarMinSpins[colouringOptions.colourBy].value(), 
                              colouringOptions.scalarMaxSpins[colouringOptions.colourBy].value())

################################################################################

def getScalar(colouringOptions, lattice, atomIndex, scalarVal=None):
    """
    Return the correct scalar value for using with LUT
    
    """
    if colouringOptions.colourBy == "Species" or colouringOptions.colourBy == "Solid colour":
        scalar = lattice.specie[atomIndex]
    
    elif colouringOptions.colourBy == "Height":
        scalar = lattice.pos[3*atomIndex+colouringOptions.heightAxis]
    
    elif colouringOptions.colourBy == "Charge":
        scalar = lattice.charge[atomIndex]
    
    elif scalarVal is not None:
        scalar = scalarVal
    
    else:
        scalar = lattice.specie[atomIndex]
    
    return scalar

################################################################################

def setupLUT(specieList, specieRGB, colouringOptions):
    """
    Setup the colour look up table
    
    """
    logger = logging.getLogger(__name__)
    logger.debug("Setting up LUT")
    
    lut = vtk.vtkLookupTable()
    
    if colouringOptions.colourBy == "Species" or colouringOptions.colourBy == "Solid colour":
        NSpecies = len(specieList)
        
        lut.SetNumberOfColors(NSpecies)
        lut.SetNumberOfTableValues(NSpecies)
        lut.SetTableRange(0, NSpecies - 1)
        lut.SetRange(0, NSpecies - 1)
        
        for i in range(NSpecies):
            if colouringOptions.colourBy == "Species":
                lut.SetTableValue(i, specieRGB[i][0], specieRGB[i][1], specieRGB[i][2], 1.0)
            
            elif colouringOptions.colourBy == "Solid colour":
                lut.SetTableValue(i, colouringOptions.solidColourRGB[0], colouringOptions.solidColourRGB[1], colouringOptions.solidColourRGB[2])
    
    elif colouringOptions.colourBy == "Height":
        lut.SetNumberOfColors(1024)
        lut.SetHueRange(0.667,0.0)
        lut.SetRange(colouringOptions.minVal, colouringOptions.maxVal)    
        lut.SetRampToLinear()
        lut.Build()
    
    elif colouringOptions.colourBy == "Charge":
        lut.SetNumberOfColors(1024)
        lut.SetHueRange(0.667,0.0)
        lut.SetRange(colouringOptions.chargeMinSpin.value(), colouringOptions.chargeMaxSpin.value())    
        lut.SetRampToLinear()
        lut.Build()
    
    else:
        lut.SetNumberOfColors(1024)
        lut.SetHueRange(0.667,0.0)
        lut.SetRange(colouringOptions.scalarMinSpins[colouringOptions.colourBy].value(), 
                     colouringOptions.scalarMaxSpins[colouringOptions.colourBy].value())    
        lut.SetRampToLinear()
        lut.Build()
    
    return lut

################################################################################

def setRes(num, displayOptions):
    #res = 15.84 * (0.99999**natoms)
    #if(LowResVar.get()=="LowResOff"):
    if(num==0):
        res = 100
    else:
#         #if(ResVar.get()=="LowResOn"):
#         #    
#         #    res = -1.0361*math.log(num,e) + 14.051
#         #    #res = round(res,0)
#         #    #res = 176*(num**-0.36)
#         #    res = int(res)
#         #    
#         #elif(ResVar.get()=="HighResOn"):
#         #    
#         #    res = -2.91*math.log(num,e) + 35
#         #    res = round(res,0)
#         #    res = 370*(num**-0.36)
#         #    res = int(res)
#         #    
#         #else:
#         
#         res = -2.91*math.log(num,2.7) + 35
#         res = round(res,0)
#         res = 170*(num**-0.36)
#         res = int(res)
    
        res = int(displayOptions.resA * num ** (-displayOptions.resB))
        
        logger = logging.getLogger(__name__)
        logger.debug("Setting sphere resolution (N = %d): %d", num, res)
    
    return res
