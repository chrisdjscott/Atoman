
"""
Render bonds

@author: Chris Scott

"""
import os
import functools

import numpy as np
import vtk

from .utils import setupLUT, getScalarsType
from .povutils import povrayBond


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

def renderBonds(visibleAtoms, mainWindow, pipelinePage, actorsCollection, colouringOptions, povfile, scalarsDict, bondArray, NBondsArray, bondVectorArray, bondsOptions):
    """
    Render bonds.
    
    """
    # SETTINGS
    bondThicknessVTK = bondsOptions.bondThicknessVTK
    bondThicknessPOV = bondsOptions.bondThicknessPOV
    bondNumSides = bondsOptions.bondNumSides
    # END SETTINGS
    
    # scalar type
    scalarType = getScalarsType(colouringOptions)
    
    # scalars array
    if scalarType == 5:
        scalars = scalarsDict[colouringOptions.colourBy]
    else:
        scalars = np.array([], dtype=np.float64)
    
    NVisible = len(visibleAtoms)
    
    NBondsHalf = np.sum(NBondsArray)
    NBonds = NBondsHalf * 2
    
    # povray file
    povFilePath = os.path.join(mainWindow.tmpDirectory, povfile)
    fpov = open(povFilePath, "w")
    
    lattice = pipelinePage.inputState
    
    # make LUT
    lut = setupLUT(lattice.specieList, lattice.specieRGB, colouringOptions)
    
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
    pov_rgb = np.empty(3, np.float64)
    pov_rgb2 = np.empty(3, np.float64)
    count = 0
    bcount = 0
    for i in xrange(NVisible):
        index = visibleAtoms[i]
        
        # scalar
        scalar = getScalarValue(lattice, index, scalars, i, colouringOptions)
        
        # colour for povray
        lut.GetColor(scalar, pov_rgb)
        
        # pos
        xpos = lattice.pos[3*index]
        ypos = lattice.pos[3*index+1]
        zpos = lattice.pos[3*index+2]
        
        for _ in xrange(NBondsArray[i]):
            bondCoords.SetTuple3(bcount, xpos, ypos, zpos)
            bondVectors.SetTuple3(bcount, bondVectorArray[3*count], bondVectorArray[3*count+1], bondVectorArray[3*count+2])
            bondScalars.SetTuple1(bcount, scalar)
            
            # povray bond
            fpov.write(povrayBond(lattice.pos[3*index:3*index+3], 
                                  lattice.pos[3*index:3*index+3] + bondVectorArray[3*count:3*count+3], 
                                  bondThicknessPOV, pov_rgb, 0.0))
            
            bcount += 1
            
            # second half
            visIndex = bondArray[count]
            index2 = visibleAtoms[visIndex]
            
            scalar2 = getScalarValue(lattice, index2, scalars, visIndex, colouringOptions)
            
            bondCoords.SetTuple3(bcount, lattice.pos[3*index2], lattice.pos[3*index2+1], lattice.pos[3*index2+2])
            bondVectors.SetTuple3(bcount, -1 * bondVectorArray[3*count], -1 * bondVectorArray[3*count+1], -1 * bondVectorArray[3*count+2])
            bondScalars.SetTuple1(bcount, scalar2)
            
            # colour for povray
            lut.GetColor(scalar2, pov_rgb2)
            
            # povray bond
            fpov.write(povrayBond(lattice.pos[3*index2:3*index2+3], 
                                  lattice.pos[3*index2:3*index2+3] - bondVectorArray[3*count:3*count+3], 
                                  bondThicknessPOV, pov_rgb2, 0.0))
            
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

