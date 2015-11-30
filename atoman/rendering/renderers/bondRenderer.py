
"""
Module for rendering bonds

"""
import time
import logging
import functools

import vtk
import numpy as np

from .. import utils
from ...filtering import bonds

################################################################################

def _bondGlyphMethod(bondGlyph, bondGlyphSource, *args, **kwargs):
    """Bond glyph method for programmable glyph filter."""
    pointID = bondGlyph.GetPointId()
    
    vector = bondGlyph.GetPointData().GetVectors().GetTuple3(pointID)
    pos = bondGlyph.GetPoint()
    
    bondGlyphSource.SetPoint1(pos)
    bondGlyphSource.SetPoint2(pos[0] + vector[0], pos[1] + vector[1], pos[2] + vector[2])

################################################################################

class BondRenderer(object):
    """
    Render a set of bonds.
    
    """
    def __init__(self):
        self._logger = logging.getLogger(__name__)
    
    def render(self, lattice, visibleAtoms, NBondsArray, bondArray, bondVectorArray, scalarsArray, colouringOptions,
               bondsOptions, lut):
        """
        Render the given atoms.
        
        Explain...
        
        """
        self._logger.debug("Rendering bonds")
        
        renderBondsTime = time.time()
        
        # SETTINGS
        bondThicknessVTK = bondsOptions.bondThicknessVTK
        bondNumSides = bondsOptions.bondNumSides
        # END SETTINGS
        
        # values
        NVisible = len(visibleAtoms)
        NBondsHalf = np.sum(NBondsArray)
        NBonds = NBondsHalf * 2
        
        # number of species
        NSpecies = len(lattice.specieList)
        
        # vtk array storing coords of bonds
        bondCoords = vtk.vtkFloatArray()
        bondCoords.SetNumberOfComponents(3)
        bondCoords.SetNumberOfTuples(NBonds)
        
        # vtk array storing scalars (for lut)
        bondScalars = vtk.vtkFloatArray()
        bondScalars.SetNumberOfComponents(1)
        bondScalars.SetNumberOfTuples(NBonds)
        
        # vtk array storing vectors (for tube)
        bondVectors = vtk.vtkFloatArray()
        bondVectors.SetNumberOfComponents(3)
        bondVectors.SetNumberOfTuples(NBonds)
        
        # construct vtk bond arrays
        count = 0
        bcount = 0
        for i in xrange(NVisible):
            index = visibleAtoms[i]
            
            # scalar
            scalar = scalarsArray[i]
            
            # pos
            xpos = lattice.pos[3 * index]
            ypos = lattice.pos[3 * index + 1]
            zpos = lattice.pos[3 * index + 2]
            
            for _ in xrange(NBondsArray[i]):
                bondCoords.SetTuple3(bcount, xpos, ypos, zpos)
                bondVectors.SetTuple3(bcount, bondVectorArray[3 * count], bondVectorArray[3 * count + 1],
                                      bondVectorArray[3 * count + 2])
                bondScalars.SetTuple1(bcount, scalar)
                bcount += 1
                
                # second half
                visIndex = bondArray[count]
                index2 = visibleAtoms[visIndex]
                scalar2 = scalarsArray[visIndex]
                
                bondCoords.SetTuple3(bcount, lattice.pos[3 * index2], lattice.pos[3 * index2 + 1],
                                     lattice.pos[3 * index2 + 2])
                bondVectors.SetTuple3(bcount, -1 * bondVectorArray[3 * count], -1 * bondVectorArray[3 * count + 1],
                                      -1 * bondVectorArray[3 * count + 2])
                bondScalars.SetTuple1(bcount, scalar2)
                
                bcount += 1
                count += 1
        
        # points
        bondPoints = vtk.vtkPoints()
        bondPoints.SetData(bondCoords)
        
        # poly data
        bondPolyData = vtk.vtkPolyData()
        bondPolyData.SetPoints(bondPoints)
        bondPolyData.GetPointData().SetScalars(bondScalars)
        bondPolyData.GetPointData().SetVectors(bondVectors)
        
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
        utils.setMapperScalarRange(mapper, colouringOptions, NSpecies)
        
        # actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(1)
        actor.GetProperty().SetLineWidth(bondThicknessVTK)
        
        # time taken
        renderBondsTime = time.time() - renderBondsTime
        self._logger.debug("Render bonds time: %f s", renderBondsTime)
        
        return actor

################################################################################

class BondCalculator(object):
    """
    Object that calculates bonds between visible atoms.
    
    """
    def __init__(self):
        self._logger = logging.getLogger(__name__)
    
    def calculateBonds(self, inputState, visibleAtoms, bondMinArray, bondMaxArray, drawList, maxBondsPerAtom=50):
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
        
        return NBondsTotal, bondArray, NBondsArray, bondVectorArray, bondSpecieCounter
