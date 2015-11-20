
"""
On screen text

@author: Chris Scott

"""
import vtk



################################################################################

class vtkRenderWindowText(vtk.vtkTextActor):
    """
    On screen information text.
    
    @author: Marc Robinson
    
    """
    def __init__(self, inputtext, Size, x, y, r, g, b):
        self.Input =  inputtext
        self.Size =  Size
        self.x =  x
        self.y =  y    
        self.SetDisplayPosition(self.x, self.y)
        self.SetInput(self.Input)
        #textActor.UseBorderAlignOn
        self.GetPosition2Coordinate().SetCoordinateSystemToNormalizedViewport()
        #textActor.GetPosition2Coordinate().SetValue(0.6, 0.4)
        tprop = self.GetTextProperty()
        tprop.SetFontSize(self.Size)
        tprop.SetFontFamilyToArial()
        tprop.SetJustificationToLeft()
        tprop.SetVerticalJustificationToTop()
        tprop.BoldOn()
        #tprop.ItalicOn()
        #tprop.ShadowOn()
        #tprop.SetShadowOffset(2,2)
        tprop.SetColor(r,g,b)
    
    def change_input(self,inputtext1):
        self.Input = inputtext1
        self.SetInput(self.Input)
    
    def change_pos(self,x,y):
        self.x =  x
        self.y =  y    
        self.SetDisplayPosition(self.x, self.y)

