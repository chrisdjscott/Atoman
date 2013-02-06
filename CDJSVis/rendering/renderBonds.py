
"""
Render bonds

@author: Chris Scott

"""
import os
import functools

import numpy as np
import vtk

from .utils import setRes, setupLUT


################################################################################

def getScalarValue(lattice, index, scalars, scalarsIndex, colouringOptions):
    """
    Set the scalar value.
    
    """
    if colouringOptions.colourBy == "Specie" or colouringOptions.colourBy == "Solid colour":
        scalar = lattice.specie[index]
    
    elif colouringOptions.colourBy == "Height":
        scalar = lattice.pos[3*index+colouringOptions.heightAxis]
    
    elif colouringOptions.colourBy == "Atom property":
        if colouringOptions.atomPropertyType == "Kinetic energy":
            scalar = lattice.KE[index]
        elif colouringOptions.atomPropertyType == "Potential energy":
            scalar = lattice.PE[index]
        else:
            scalar = lattice.charge[index]
    
    else:
        scalar = scalars[scalarsIndex]
    
    return scalar

################################################################################

def bondGlyphMethod(bondGlyph, bondGlyphSource, *args, **kwargs):
    """
    Bond glyph method for programmable glyph filter.
    
    """    
    pointID = bondGlyph.GetPointId()
    
    vector = bondGlyph.GetPointData().GetVectors().GetTuple3(pointID)
    pos = bondGlyph.GetPoint()
    
    bondGlyphSource.SetPoint1(pos)
    bondGlyphSource.SetPoint2(pos[0] + vector[0], pos[1] + vector[1], pos[2] + vector[2])

################################################################################

def renderBonds(visibleAtoms, mainWindow, actorsCollection, colouringOptions, povfile, scalars, bondArray, NBondsArray, bondVectorArray):
    """
    Render bonds.
    
    """
    # SETTINGS
    bondThicknessVTK = 0.2
    bondThicknessPOV = 0.08
    bondNumSides = 5
    # END SETTINGS
    
    NVisible = len(visibleAtoms)
    
    NBondsHalf = np.sum(NBondsArray)
    NBonds = NBondsHalf * 2
    
#    if NVisibleForRes is None:
#        NVisibleForRes = NVisible
    
    # povray file
    povFilePath = os.path.join(mainWindow.tmpDirectory, povfile)
    fpov = open(povFilePath, "w")
    
    # resolution
#    res = setRes(NBonds)
    
    lattice = mainWindow.inputState
    
    # make LUT
    lut = setupLUT(lattice.specieList, lattice.specieRGB, colouringOptions)
    
    # number of species
    NSpecies = len(lattice.specieList)
#    specieCount = np.zeros(NSpecies, np.int32)
    
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
        scalar = getScalarValue(lattice, index, scalars, i, colouringOptions)
        
        # pos
        xpos = lattice.pos[3*index]
        ypos = lattice.pos[3*index+1]
        zpos = lattice.pos[3*index+2]
        
        for _ in xrange(NBondsArray[i]):
            bondCoords.SetTuple3(bcount, xpos, ypos, zpos)
            bondVectors.SetTuple3(bcount, bondVectorArray[3*count], bondVectorArray[3*count+1], bondVectorArray[3*count+2])
            bondScalars.SetTuple1(bcount, scalar)
            
            # povray
            
            
            bcount += 1
            
            # second half
            visIndex = bondArray[count]
            index2 = visibleAtoms[visIndex]
            
            scalar2 = getScalarValue(lattice, index2, scalars, visIndex, colouringOptions)
            
            bondCoords.SetTuple3(bcount, lattice.pos[3*index2], lattice.pos[3*index2+1], lattice.pos[3*index2+2])
            bondVectors.SetTuple3(bcount, -1 * bondVectorArray[3*count], -1 * bondVectorArray[3*count+1], -1 * bondVectorArray[3*count+2])
            bondScalars.SetTuple1(bcount, scalar2)
            
            # povray
            
            
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
    tubes.UseDefaultNormalOn()
    tubes.SetCapping(1)
    tubes.SetDefaultNormal(0.577, 0.577, 0.577)
    
    # glyph filter
    bondGlyphFilter = vtk.vtkProgrammableGlyphFilter()
    bondGlyphFilter.SetGlyphMethod(functools.partial(bondGlyphMethod, bondGlyphFilter, lineSource))
    bondGlyphFilter.SetSource(tubes.GetOutput())
    bondGlyphFilter.SetInput(bondPolyData)
    
    # mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(bondGlyphFilter.GetOutputPort())
    mapper.SetLookupTable(lut)
    if colouringOptions.colourBy == "Specie":
        mapper.SetScalarRange(0, NSpecies - 1)
    
    elif colouringOptions.colourBy == "Height":
        mapper.SetScalarRange(colouringOptions.minVal, colouringOptions.maxVal)
    
    elif colouringOptions.colourBy == "Atom property":
        mapper.SetScalarRange(colouringOptions.propertyMinSpin.value(), colouringOptions.propertyMaxSpin.value())
    
    else:
        mapper.SetScalarRange(colouringOptions.scalarMinSpin.value(), colouringOptions.scalarMaxSpin.value())
    
    # actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(1)
    actor.GetProperty().SetLineWidth(bondThicknessVTK)
    
    # add to actors collection
    actorsCollection.AddItem(actor)
    
    # close pov file
    fpov.close()
    
    
    




