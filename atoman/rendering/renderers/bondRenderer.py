
"""
Module for rendering bonds

"""
import time
import logging
import functools

import vtk
import numpy as np

from . import baseRenderer
from .. import utils
from .. import _rendering
from ...filtering import bonds


def _bondGlyphMethod(bondGlyph, bondGlyphSource, *args, **kwargs):
    """Bond glyph method for programmable glyph filter."""
    # get vector and position
    pointID = bondGlyph.GetPointId()
    vector = bondGlyph.GetPointData().GetVectors().GetTuple3(pointID)
    pos = bondGlyph.GetPoint()
    
    # set ends for line
    bondGlyphSource.SetPoint1(pos)
    bondGlyphSource.SetPoint2(pos[0] + vector[0], pos[1] + vector[1], pos[2] + vector[2])


class BondRenderer(baseRenderer.BaseRenderer):
    """
    Render a set of bonds.
    
    """
    def __init__(self):
        super(BondRenderer, self).__init__()
        self._logger = logging.getLogger(__name__ + ".BondRenderer")
    
    def render(self, bondCoords, bondVectors, bondScalars, numSpecies, colouringOptions, bondsOptions, lut):
        """
        Render the given bonds.
        
        """
        self._logger.debug("Rendering bonds")
        
        renderBondsTime = time.time()
        
        # SETTINGS
        bondThicknessVTK = bondsOptions.bondThicknessVTK
        bondNumSides = bondsOptions.bondNumSides
        # END SETTINGS
        
        # points
        bondPoints = vtk.vtkPoints()
        bondPoints.SetData(bondCoords.getVTK())
        
        # poly data
        bondPolyData = vtk.vtkPolyData()
        bondPolyData.SetPoints(bondPoints)
        bondPolyData.GetPointData().SetScalars(bondScalars.getVTK())
        bondPolyData.GetPointData().SetVectors(bondVectors.getVTK())
        
        # line source
        lineSource = vtk.vtkLineSource()
        
        # tubes
        tubes = vtk.vtkTubeFilter()
        tubes.SetInputConnection(lineSource.GetOutputPort())
        tubes.SetRadius(bondThicknessVTK)
        tubes.SetNumberOfSides(bondNumSides)
        tubes.SetCapping(1)
        
        # glyph filter
        bondGlyphFilter = vtk.vtkProgrammableGlyphFilter()
        bondGlyphFilter.SetGlyphMethod(functools.partial(_bondGlyphMethod, bondGlyphFilter, lineSource))
        if vtk.vtkVersion.GetVTKMajorVersion() <= 5:
            bondGlyphFilter.SetSource(tubes.GetOutput())
            bondGlyphFilter.SetInput(bondPolyData)
        else:
            bondGlyphFilter.SetSourceConnection(tubes.GetOutputPort())
            bondGlyphFilter.SetInputData(bondPolyData)
        
        # mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(bondGlyphFilter.GetOutputPort())
        mapper.SetLookupTable(lut)
        utils.setMapperScalarRange(mapper, colouringOptions, numSpecies)
        
        # actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(1)
        actor.GetProperty().SetLineWidth(bondThicknessVTK)
        
        # time taken
        renderBondsTime = time.time() - renderBondsTime
        self._logger.debug("Render bonds time: %f s", renderBondsTime)
        
        # store attributes
        self._actor = utils.ActorObject(actor)
        self._data["Points"] = bondCoords
        self._data["Scalars"] = bondScalars
        self._data["Vectors"] = bondVectors


class BondCalculator(object):
    """
    Object that calculates bonds between visible atoms.
    
    """
    def __init__(self):
        self._logger = logging.getLogger(__name__ + ".BondCalculator")
    
    def calculateBonds(self, inputState, visibleAtoms, scalarsArray, bondMinArray, bondMaxArray, drawList,
                       maxBondsPerAtom=50):
        """Find bonds."""
        self._logger.info("Calculating bonds")
        
        # arrays for results
        # TODO: create array within C lib so not so big!
        nvis = len(visibleAtoms)
        nspecs = len(inputState.specieList)
        size = int(nvis * maxBondsPerAtom / 2)
        bondArray = np.empty(size, np.int32)
        NBondsArray = np.zeros(nvis, np.int32)
        bondVectorArray = np.empty(3 * size, np.float64)
        bondSpecieCounter = np.zeros((nspecs, nspecs), dtype=np.int32)
        maxBond = bondMaxArray.max()
        
        # call C library
        status = bonds.calculateBonds(visibleAtoms, inputState.pos, inputState.specie, len(inputState.specieList),
                                      bondMinArray, bondMaxArray, maxBond, maxBondsPerAtom, inputState.cellDims,
                                      inputState.PBC, bondArray, NBondsArray, bondVectorArray, bondSpecieCounter)
        
        if status:
            if status == 1:
                msg = "Max bonds per atom exceeded! This would suggest you bond range is too big!"
            else:
                msg = "Error in bonds clib (%d)" % status
            raise RuntimeError(msg)
        
        # total number of bonds
        NBondsTotal = np.sum(NBondsArray)
        self._logger.info("Total number of bonds: %d (x2 for actors)", NBondsTotal)
        
        # resize bond arrays
        bondArray.resize(NBondsTotal)
        bondVectorArray.resize(NBondsTotal * 3)
        
        # construct bonds arrays for rendering
        res = _rendering.makeBondsArrays(visibleAtoms, scalarsArray.getNumpy(), inputState.pos, NBondsArray, bondArray,
                                         bondVectorArray)
        bondCoords, bondVectors, bondScalars = res
        bondCoords = utils.NumpyVTKData(bondCoords)
        bondVectors = utils.NumpyVTKData(bondVectors, name="vectors")
        bondScalars = utils.NumpyVTKData(bondScalars, name="colours")
        
        # specie counters
        specieList = inputState.specieList
        for i in xrange(nspecs):
            syma = specieList[i]
            
            for j in xrange(i, nspecs):
                symb = specieList[j]
                
                # check if selected
                pairStr = "%s-%s" % (syma, symb)
                pairStr2 = "%s-%s" % (symb, syma)
                
                if pairStr in drawList or pairStr2 in drawList:
                    NBondsPair = bondSpecieCounter[i][j]
                    if i != j:
                        NBondsPair += bondSpecieCounter[j][i]
                        bondSpecieCounter[i][j] = NBondsPair
                        bondSpecieCounter[j][i] = NBondsPair
                    
                    self._logger.info("%d %s - %s bonds", NBondsPair, syma, symb)
        
        return bondCoords, bondVectors, bondScalars, bondSpecieCounter


class DisplacmentVectorCalculator(object):
    """
    Calculate displacement vectors.
    
    """
    def __init__(self):
        self._logger = logging.getLogger(__name__ + ".DisplacmentVectorCalculator")
    
    def calculateDisplacementVectors(self, pos, refPos, pbc, cellDims, atomList, scalarsArray):
        """Calculate displacement vectors for the set of atoms."""
        self._logger.debug("Calculating displacement vectors")
        
        # calculate vectors
        numAtoms = len(atomList)
        bondVectors = np.empty(3 * numAtoms, np.float64)
        drawBondVector = np.empty(numAtoms, np.int32)
        numBonds = bonds.calculateDisplacementVectors(atomList, pos, refPos, cellDims, pbc, bondVectors, drawBondVector)
        
        self._logger.debug("Number of displacement vectors to draw = %d (/ %d)", numBonds, numAtoms)
        
        # calculate arrays for rendering
        res = _rendering.makeDisplacementVectorBondsArrays(numBonds, atomList, scalarsArray, pos, drawBondVector,
                                                           bondVectors)
        bondCoords, bondVectors, bondScalars = res
        bondCoords = utils.NumpyVTKData(bondCoords)
        bondVectors = utils.NumpyVTKData(bondVectors, name="vectors")
        bondScalars = utils.NumpyVTKData(bondScalars, name="colours")
        
        return bondCoords, bondVectors, bondScalars
