
"""
Axes class.

@author: Chris Scott

"""
import vtk



################################################################################

class AxesBasic(object):
    """
    @author: Marc Robinson
    
    Modified slightly
    
    """
    def __init__(self, ren, reinit):
        self.Edgesx = vtk.vtkActor()
        self.Edgesy = vtk.vtkActor()
        self.Edgesz = vtk.vtkActor()
        self.conexActor = vtk.vtkActor()
        self.coneyActor = vtk.vtkActor()
        self.coneyActor.RotateZ(90)
        self.conezActor = vtk.vtkActor()
        self.conezActor.RotateY(270)
        self.xlabelActor = vtk.vtkFollower()
        self.xlabelActor.SetScale(3, 3, 3)
        self.ylabelActor = vtk.vtkFollower()
        self.ylabelActor.SetScale(3, 3, 3)
        self.zlabelActor = vtk.vtkFollower()
        self.zlabelActor.SetScale(3, 3, 3)
        
        self.actorsList = []
        self.actorsList.append(self.Edgesx)
        self.actorsList.append(self.Edgesy)
        self.actorsList.append(self.Edgesz)
        self.actorsList.append(self.conexActor)
        self.actorsList.append(self.coneyActor)
        self.actorsList.append(self.conezActor)
        self.actorsList.append(self.xlabelActor)
        self.actorsList.append(self.ylabelActor)
        self.actorsList.append(self.zlabelActor)
        
        self.ren = ren
        self.reinit = reinit
        
        self.visible = 1
    
    def remove(self):
        
        for actor in self.actorsList:
            self.ren.RemoveActor(actor)
        
        self.reinit()
        
        self.visible = 0
    
    def add(self):
        
        for actor in self.actorsList:
            self.ren.AddActor(actor)
            
        self.reinit()
        
        self.visible = 1
    
    def refresh(self,x0,y0,z0,xl,yl,zl,xtext,ytext,ztext):
        
        self.remove()
        
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.xl = xl
        self.yl = yl
        self.zl = zl
        self.xtext = xtext
        self.ytext = ytext
        self.ztext = ztext
        
        linex = vtk.vtkLineSource()
        linex.SetPoint1(x0,y0,z0)
        linex.SetPoint2(x0+xl,y0,z0)    
        Tubesx = vtk.vtkTubeFilter()
        Tubesx.SetInputConnection(linex.GetOutputPort())
        Tubesx.SetRadius(0.5)
        Tubesx.SetNumberOfSides(5)
        TubeMapperx = vtk.vtkPolyDataMapper()
        TubeMapperx.SetInputConnection(Tubesx.GetOutputPort())
        self.Edgesx.SetMapper(TubeMapperx)
        self.Edgesx.GetProperty().SetDiffuseColor(1,0,0)
        self.Edgesx.GetProperty().SetSpecular(.4)
        self.Edgesx.GetProperty().SetSpecularPower(10)
        self.Edgesx.GetProperty().SetLineWidth(2.0)
        
        liney = vtk.vtkLineSource()
        liney.SetPoint1(x0,y0,z0)
        liney.SetPoint2(x0,y0+yl,z0)    
        Tubesy = vtk.vtkTubeFilter()
        Tubesy.SetInputConnection(liney.GetOutputPort())
        Tubesy.SetRadius(0.5)
        Tubesy.SetNumberOfSides(5)
        TubeMappery = vtk.vtkPolyDataMapper()
        TubeMappery.SetInputConnection(Tubesy.GetOutputPort())
        
        self.Edgesy.SetMapper(TubeMappery)
        self.Edgesy.GetProperty().SetDiffuseColor(0,1,0)
        self.Edgesy.GetProperty().SetSpecular(.4)
        self.Edgesy.GetProperty().SetSpecularPower(10)
        self.Edgesy.GetProperty().SetLineWidth(2.0)
        #self.AddItem(self.Edgesy)  
        linez = vtk.vtkLineSource()
        linez.SetPoint1(x0,y0,z0)
        linez.SetPoint2(x0,y0,z0+zl)    
        Tubesz = vtk.vtkTubeFilter()
        Tubesz.SetInputConnection(linez.GetOutputPort())
        Tubesz.SetRadius(0.5)
        Tubesz.SetNumberOfSides(5)
        TubeMapperz = vtk.vtkPolyDataMapper()
        TubeMapperz.SetInputConnection(Tubesz.GetOutputPort())
        
        self.Edgesz.SetMapper(TubeMapperz)
        self.Edgesz.GetProperty().SetDiffuseColor(0,0,1)
        self.Edgesz.GetProperty().SetSpecular(.4)
        self.Edgesz.GetProperty().SetSpecularPower(10)
        self.Edgesz.GetProperty().SetLineWidth(2.0)
        #self.AddItem(self.Edgesz)
        
        conex = vtk.vtkConeSource()
        conex.SetRadius(1.2)
        conex.SetHeight(2.0)
        conex.SetResolution(50)
        #conex.SetThetaResolution(10)
        conex.SetCenter(x0+xl+1.0,y0,z0)
    
    
        conexMapper = vtk.vtkPolyDataMapper()
        conexMapper.SetInputConnection(conex.GetOutputPort())
        
        self.conexActor.SetMapper(conexMapper)
        self.conexActor.GetProperty().SetDiffuseColor(1,0,0)
        #self.AddItem(self.conexActor)
        
        coney = vtk.vtkConeSource()
        coney.SetRadius(1.2)
        coney.SetHeight(2.0)
        coney.SetResolution(50)
        coney.SetCenter(x0,y0+yl+1.0,z0)
    
        
        coneyMapper = vtk.vtkPolyDataMapper()
        coneyMapper.SetInputConnection(coney.GetOutputPort())
        self.coneyActor.SetMapper(coneyMapper)
        self.coneyActor.GetProperty().SetDiffuseColor(0,1,0)
        self.coneyActor.SetOrigin(x0,y0+yl+1.0,z0)
        
        #self.AddItem(self.coneyActor)
        
        conez = vtk.vtkConeSource()
        conez.SetRadius(1.2)
        conez.SetHeight(2.0)
        conez.SetResolution(50)
        conez.SetCenter(x0,y0,z0+zl+1.0)
        conezMapper = vtk.vtkPolyDataMapper()
        conezMapper.SetInputConnection(conez.GetOutputPort())
        self.conezActor.SetMapper(conezMapper)
        self.conezActor.GetProperty().SetDiffuseColor(0,0,1)
        self.conezActor.SetOrigin(x0,y0,z0+zl+1.0)
        
        
        #self.AddItem(self.conezActor)
            
        caseLabel = vtk.vtkVectorText()
        caseLabel.SetText(self.xtext)
                
        labelMapper = vtk.vtkPolyDataMapper()
        labelMapper.SetInputConnection(caseLabel.GetOutputPort())
        
        self.xlabelActor.SetMapper(labelMapper)
        
        self.xlabelActor.SetPosition(x0+xl+5.0,y0-0.5,z0)
        self.xlabelActor.GetProperty().SetDiffuseColor(1,0,0)
        self.xlabelActor.SetCamera(self.ren.GetActiveCamera())
        
        
        #self.AddItem(self.xlabelActor)
        
        caseLabel = vtk.vtkVectorText()
        caseLabel.SetText(self.ytext)
        labelMapper = vtk.vtkPolyDataMapper()
        labelMapper.SetInputConnection(caseLabel.GetOutputPort())
        
        self.ylabelActor.SetMapper(labelMapper)
        #self.ylabelActor.SetScale(3, 3, 3)
    
        self.ylabelActor.SetPosition(x0,y0+yl+4.0,z0)
        self.ylabelActor.GetProperty().SetDiffuseColor(0,1,0)
        self.ylabelActor.SetCamera(self.ren.GetActiveCamera())
        #self.AddItem(self.ylabelActor)


        caseLabel = vtk.vtkVectorText()
        caseLabel.SetText(self.ztext)
        labelMapper = vtk.vtkPolyDataMapper()
        labelMapper.SetInputConnection(caseLabel.GetOutputPort())
        
        self.zlabelActor.SetMapper(labelMapper)
        #self.zlabelActor.SetScale(3, 3, 3)
        self.zlabelActor.SetPosition(x0,y0,z0+zl+3.0)
        self.zlabelActor.GetProperty().SetDiffuseColor(0,0,1)
        self.zlabelActor.GetProperty().SetEdgeColor(0,0,0)
        self.zlabelActor.GetProperty().EdgeVisibilityOn()
        self.zlabelActor.GetProperty().SetLineWidth(2)
        self.zlabelActor.SetCamera(self.ren.GetActiveCamera())
        #self.AddItem(self.zlabelActor)
        
        self.add()

################################################################################
class Axes(object):
    def __init__(self, ren, renWinInteract):
        
        self.ren = ren
        self.renWinInteract = renWinInteract
        
        self.actor = vtk.vtkAxesActor()
        self.actor.SetTipTypeToCone()
        self.actor.SetShaftTypeToCylinder()
#        self.actor.GetXAxisCaptionActor2D().SetWidth(0.1)
#        self.actor.GetXAxisCaptionActor2D().SetHeight(0.1)
#        self.actor.GetYAxisCaptionActor2D().SetWidth(0.1)
#        self.actor.GetYAxisCaptionActor2D().SetHeight(0.1)
#        self.actor.GetZAxisCaptionActor2D().SetWidth(0.1)
#        self.actor.GetZAxisCaptionActor2D().SetHeight(0.1)
        self.actor.GetXAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        self.actor.GetYAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        self.actor.GetZAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        
        transform = vtk.vtkTransform()
        transform.Translate(-10.0, -10.0, -10.0)
        self.actor.SetUserTransform(transform)
        
        self.actor.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetColor(1,0,0)
        self.actor.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(20)
        self.actor.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetColor(0,1,0)
        self.actor.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(20)
        self.actor.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetColor(0,0,1)
        self.actor.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(20)
        
        self.orientationWidget = vtk.vtkOrientationMarkerWidget()
        self.orientationWidget.SetOrientationMarker(self.actor)
#        self.orientationWidget.SetOutlineColor(0.93, 0.57, 0.13)
        self.orientationWidget.SetInteractor(self.renWinInteract)
#        self.orientationWidget.SetViewport(0.0, 0.0, 0.25, 0.25)
        self.orientationWidget.SetEnabled(1)
        self.orientationWidget.InteractiveOff()
        
        self.visible = 0
    
    def add(self, cellDims):
        """
        Add the axes.
        
        """
        self.actor.SetTotalLength(0.2 * cellDims[0], 0.2 * cellDims[1], 0.2 * cellDims[2])
        
        self.ren.AddActor(self.actor)
        self.visible = 1
        
    def remove(self):
        """
        Remove the axes actor.
        
        """
        self.ren.RemoveActor(self.actor)
        
        self.visible = 0

