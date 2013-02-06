
"""
Rendering utils

@author: Chris Scott

"""
import math

import vtk



################################################################################

def setupLUT(specieList, specieRGB, colouringOptions):
    """
    Setup the colour look up table
    
    """
    lut = vtk.vtkLookupTable()
    
    if colouringOptions.colourBy == "Specie" or colouringOptions.colourBy == "Solid colour":
        NSpecies = len(specieList)
        
        lut.SetNumberOfColors(NSpecies)
        lut.SetNumberOfTableValues(NSpecies)
        lut.SetTableRange(0, NSpecies - 1)
        lut.SetRange(0, NSpecies - 1)
        
        for i in xrange(NSpecies):
            if colouringOptions.colourBy == "Specie":
                lut.SetTableValue(i, specieRGB[i][0], specieRGB[i][1], specieRGB[i][2], 1.0)
            
            elif colouringOptions.colourBy == "Solid colour":
                lut.SetTableValue(i, colouringOptions.solidColourRGB[0], colouringOptions.solidColourRGB[1], colouringOptions.solidColourRGB[2])
    
    elif colouringOptions.colourBy == "Height":
        lut.SetNumberOfColors(1024)
        lut.SetHueRange(0.667,0.0)
        lut.SetRange(colouringOptions.minVal, colouringOptions.maxVal)    
        lut.SetRampToLinear()
        lut.Build()
    
    elif colouringOptions.colourBy == "Atom property":
        lut.SetNumberOfColors(1024)
        lut.SetHueRange(0.667,0.0)
        lut.SetRange(colouringOptions.propertyMinSpin.value(), colouringOptions.propertyMaxSpin.value())    
        lut.SetRampToLinear()
        lut.Build()
    
    else:
        lut.SetNumberOfColors(1024)
        lut.SetHueRange(0.667,0.0)
        lut.SetRange(colouringOptions.scalarMinSpin.value(), colouringOptions.scalarMaxSpin.value())    
        lut.SetRampToLinear()
        lut.Build()
    
    return lut

################################################################################

def setRes(num):
    #res = 15.84 * (0.99999**natoms)
    #if(LowResVar.get()=="LowResOff"):
    if(num==0):
        res = 100
    else:
        #if(ResVar.get()=="LowResOn"):
        #    
        #    res = -1.0361*math.log(num,e) + 14.051
        #    #res = round(res,0)
        #    #res = 176*(num**-0.36)
        #    res = int(res)
        #    
        #elif(ResVar.get()=="HighResOn"):
        #    
        #    res = -2.91*math.log(num,e) + 35
        #    res = round(res,0)
        #    res = 370*(num**-0.36)
        #    res = int(res)
        #    
        #else:
        
        res = -2.91*math.log(num,2.7) + 35
        res = round(res,0)
        res = 170*(num**-0.36)
        res = int(res)    
    
#    print "RES = ",res,num    
    return res
