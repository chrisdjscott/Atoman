
"""
Render bonds

@author: Chris Scott

"""
import os
import functools
import logging
import time
import copy

import numpy as np
import vtk
from vtk.util import numpy_support

from .utils import setupLUT, getScalarsType, setMapperScalarRange
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

def renderBonds(visibleAtoms, mainWindow, pipelinePage, actorsDict, colouringOptions, povfile, 
                scalarsDict, bondArray, NBondsArray, bondVectorArray, bondsOptions):
    """
    Render bonds.
    
    """
    renderBondsTime = time.time()
    logger = logging.getLogger(__name__)
    
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
    setMapperScalarRange(mapper, colouringOptions, NSpecies)
    
    # actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(1)
    actor.GetProperty().SetLineWidth(bondThicknessVTK)
    
    # add to actors collection
    actorsDict["Bonds"] = actor
    
    # close pov file
    fpov.close()
    
    # time taken
    renderBondsTime = time.time() - renderBondsTime
    logger.debug("Render bonds time: %f s", renderBondsTime)

################################################################################

def renderDisplacementVectors(visibleAtoms, mainWindow, pipelinePage, actorsDict, colouringOptions, povfile, 
                              scalarsDict, numBonds, bondVectorArray, drawBondVector, bondsOptions):
    """
    Render displacement vectors
    
    """
    renderBondsTime = time.time()
    logger = logging.getLogger(__name__)
    
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
    
    # povray file
    povFilePath = os.path.join(mainWindow.tmpDirectory, povfile)
    fpov = open(povFilePath, "w")
    
    inputState = pipelinePage.inputState
    
    # make LUT
    lut = setupLUT(inputState.specieList, inputState.specieRGB, colouringOptions)
    
    # number of species
    NSpecies = len(inputState.specieList)
    
    # vtk array storing coords of bonds
    bondCoords = vtk.vtkFloatArray()
    bondCoords.SetNumberOfComponents(3)
    bondCoords.SetNumberOfTuples(numBonds)
    
    # vtk array storing scalars (for lut)
    bondScalars = vtk.vtkFloatArray()
    bondScalars.SetNumberOfComponents(1)
    bondScalars.SetNumberOfTuples(numBonds)
    
    # vtk array storing vectors (for tube)
    bondVectors = vtk.vtkFloatArray()
    bondVectors.SetNumberOfComponents(3)
    bondVectors.SetNumberOfTuples(numBonds)
    
    # construct vtk bond arrays
    pov_rgb = np.empty(3, np.float64)
    count = 0
    for i in xrange(NVisible):
        if not drawBondVector[i]:
            continue
        
        index = visibleAtoms[i]
        
        # scalar
        scalar = getScalarValue(inputState, index, scalars, i, colouringOptions)
        
        # colour for povray
        lut.GetColor(scalar, pov_rgb)
        
        # pos
        xpos = inputState.pos[3*index]
        ypos = inputState.pos[3*index+1]
        zpos = inputState.pos[3*index+2]
        
        bondCoords.SetTuple3(count, xpos, ypos, zpos)
        bondVectors.SetTuple3(count, bondVectorArray[3*i], bondVectorArray[3*i+1], bondVectorArray[3*i+2])
        bondScalars.SetTuple1(count, scalar)
        
        # povray bond
        fpov.write(povrayBond(inputState.pos[3*index:3*index+3], 
                              inputState.pos[3*index:3*index+3] + bondVectorArray[3*i:3*i+3], 
                              bondThicknessPOV, pov_rgb, 0.0))
        
        count += 1
    
    assert count == numBonds
    
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
    setMapperScalarRange(mapper, colouringOptions, NSpecies)
    
    # actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(1)
    actor.GetProperty().SetLineWidth(bondThicknessVTK)
    
    # add to actors collection
    actorsDict["Displacement vectors"] = actor
    
    # close pov file
    fpov.close()
    
    # time taken
    renderBondsTime = time.time() - renderBondsTime
    logger.debug("Render displacement vectors time: %f s", renderBondsTime)

################################################################################

def renderTraceVectors2(visibleAtoms, mainWindow, pipelinePage, actorsDict, colouringOptions, povfile, 
                        scalarsDict, numBonds, previousPos, drawTrace, bondsOptions, traceDict):
    """
    Render trace vectors
    
    """
    renderBondsTime = time.time()
    logger = logging.getLogger(__name__)
    logger.debug("Rendering trace vectors...")
    
    # SETTINGS
    bondThicknessVTK = bondsOptions.bondThicknessVTK
    bondThicknessPOV = bondsOptions.bondThicknessPOV
    bondNumSides = bondsOptions.bondNumSides
    # END SETTINGS
    
    inputState = pipelinePage.inputState
    NVisible = len(visibleAtoms)
    
    # scalar type
    scalarType = getScalarsType(colouringOptions)
    
    # scalars array
    if scalarType == 5:
        scalars = scalarsDict[colouringOptions.colourBy]
    else:
        scalars = np.array([], dtype=np.float64)
    
    # reconstruct trace dict
    newd = {}
    for i in xrange(NVisible):
        # if atom is already in, we just append new pos (previous is already there)
        # otherwise we append previous and new
        index = visibleAtoms[i]
        if index in traceDict:
            newd[index] = traceDict[index]
            if drawTrace[i]:
                curpos = copy.deepcopy(inputState.atomPos(index))
                curscl = getScalarValue(inputState, index, scalars, i, colouringOptions)
                newd[index].append((curpos, curscl))
        
        else:
            if drawTrace[i]:
                newd[index] = []
                curscl = getScalarValue(inputState, index, scalars, i, colouringOptions)
                curpos = copy.deepcopy(previousPos[3*index:3*index+3])
                newd[index].append((curpos, curscl))
                curpos = copy.deepcopy(inputState.atomPos(index))
                newd[index].append((curpos, curscl))
    
    # get size
    size = 0
    for key in newd.keys():
        pointsList = newd[key]
        size += len(pointsList)
    logger.debug("Trace points size = %d", size)
    
    if size < 2:
        logger.debug("Returning as not enough points to trace")
        return newd
    
    # arrays
    traceCoords = np.empty((size, 3), np.float64)
    traceScalars = np.empty(size, np.float64)
    
    # povray file
    povFilePath = os.path.join(mainWindow.tmpDirectory, povfile)
    fpov = open(povFilePath, "w")
    
    # populate
    count = 0
    lines = vtk.vtkCellArray()
    for key in newd.keys():
        atomList = newd[key]
        lines.InsertNextCell(len(atomList))
        for mypos, myscalar in atomList:
            traceCoords[count][:] = mypos[:]
            traceScalars[count] = myscalar
            
            lines.InsertCellPoint(count)
            
            count += 1
            
            # write POV-Ray vector
#             lut.GetColor(myscal, pov_rgb)
#             fpov.write(povrayBond(mypos, 
#                                   mypos + mybvect, 
#                                   bondThicknessPOV, pov_rgb, 0.0))
    
    # close pov file
    fpov.close()
    
    # points
    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(traceCoords, deep=1))
    
    # polydata
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)
    polydata.GetPointData().SetScalars(numpy_support.numpy_to_vtk(traceScalars, deep=1))
    
    # tubes
    tubes = vtk.vtkTubeFilter()
    if vtk.vtkVersion.GetVTKMajorVersion() <= 5:
        tubes.SetInput(polydata)
    else:
        tubes.SetInputData(polydata)
    tubes.SetRadius(bondThicknessVTK)
    tubes.SetNumberOfSides(bondNumSides)
    tubes.SetCapping(1)
    
    # make LUT
    lut = setupLUT(inputState.specieList, inputState.specieRGB, colouringOptions)
    
    # number of species
    NSpecies = len(inputState.specieList)
    
    # mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(tubes.GetOutputPort())
    mapper.SetLookupTable(lut)
    setMapperScalarRange(mapper, colouringOptions, NSpecies)
    
    # actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(1)
    actor.GetProperty().SetLineWidth(bondThicknessVTK)
    
    # add to actors collection
    actorsDict["Trace vectors 2"] = actor
    
    # time taken
    renderBondsTime = time.time() - renderBondsTime
    logger.debug("Render trace vectors time: %f s", renderBondsTime)
    
    return newd

################################################################################

def renderTraceVectors(visibleAtoms, mainWindow, pipelinePage, actorsDict, colouringOptions, povfile, 
                       scalarsDict, numBonds, bondVectorArray, drawBondVector, bondsOptions, traceDict):
    """
    Render trace vectors
    
    """
    renderBondsTime = time.time()
    logger = logging.getLogger(__name__)
    logger.debug("Rendering trace vectors...")
    
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
    
    # povray file
    povFilePath = os.path.join(mainWindow.tmpDirectory, povfile)
    fpov = open(povFilePath, "w")
    
    inputState = pipelinePage.inputState
    
    # make LUT
    lut = setupLUT(inputState.specieList, inputState.specieRGB, colouringOptions)
    
    # number of species
    NSpecies = len(inputState.specieList)
    
    # update trace dict with latest bond vectors
    newd = {}
    for i in xrange(NVisible):
        index = visibleAtoms[i]
        
        if index in traceDict:
            newd[index] = traceDict[index]
        
        # do we add a new vector for this atom
        if drawBondVector[i]:
            mypos = copy.deepcopy(inputState.pos[3*index:3*index+3])
            mybvect = copy.deepcopy(bondVectorArray[3*i:3*i+3])
            myscal = getScalarValue(inputState, index, scalars, i, colouringOptions)
            
            if index in newd:
                newd[index].append((mypos, mybvect, myscal))
            
            else:
                newd[index] = [(mypos, mybvect, myscal)]
    
    # first pass to get size of arrays
    size = 0
    for key in newd.keys():
        size += len(newd[key])
    
    logger.debug("Size of trace arrays = %d", size)
    
    # return if nothing to do
    if size == 0:
        return newd
    
    # make trace arrays
    traceCoords = np.empty((size, 3), np.float64)
    traceVectors = np.empty((size, 3), np.float64)
    traceScalars = np.empty(size, np.float64)
    pov_rgb = np.empty(3, np.float64)
    count = 0
    for key in newd.keys():
        for mypos, mybvect, myscal in newd[key]:
            traceCoords[count][:] = mypos[:]
            traceVectors[count][:] = mybvect[:]
            traceScalars[count] = myscal
            
            count += 1
            
            # write POV-Ray vector
            lut.GetColor(myscal, pov_rgb)
            fpov.write(povrayBond(mypos, 
                                  mypos + mybvect, 
                                  bondThicknessPOV, pov_rgb, 0.0))
    
    # points
    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(traceCoords, deep=1))
    
    # poly data
    polyData = vtk.vtkPolyData()
    polyData.SetPoints(points)
    polyData.GetPointData().SetScalars(numpy_support.numpy_to_vtk(traceScalars, deep=1))
    polyData.GetPointData().SetVectors(numpy_support.numpy_to_vtk(traceVectors, deep=1))
    
    # line source
    lineSource = vtk.vtkLineSource()
    
    # tubes
    tubes = vtk.vtkTubeFilter()
    tubes.SetInputConnection(lineSource.GetOutputPort())
    tubes.SetRadius(bondThicknessVTK)
    tubes.SetNumberOfSides(bondNumSides)
#     tubes.UseDefaultNormalOn()
    tubes.SetCapping(1)
#     tubes.SetDefaultNormal(0.577, 0.577, 0.577)
    
    # glyph filter
    glyphFilter = vtk.vtkProgrammableGlyphFilter()
    glyphFilter.SetGlyphMethod(functools.partial(bondGlyphMethod, glyphFilter, lineSource))
    if vtk.vtkVersion.GetVTKMajorVersion() <= 5:
        glyphFilter.SetSource(tubes.GetOutput())
        glyphFilter.SetInput(polyData)
    else:
        glyphFilter.SetSourceConnection(tubes.GetOutputPort())
        glyphFilter.SetInputData(polyData)
    
    # mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(glyphFilter.GetOutputPort())
    mapper.SetLookupTable(lut)
    setMapperScalarRange(mapper, colouringOptions, NSpecies)
    
    # actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(1)
    actor.GetProperty().SetLineWidth(bondThicknessVTK)
    
    # add to actors collection
    actorsDict["Trace vectors"] = actor
    
    # close pov file
    fpov.close()
    
    # time taken
    renderBondsTime = time.time() - renderBondsTime
    logger.debug("Render trace vectors time: %f s", renderBondsTime)
    
    return newd
