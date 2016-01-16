
"""
Module for rendering axes.

"""
import vtk


class Axes(object):
    """
    Axes object.
    
    """
    def __init__(self, renWinInteract):
        # create axes
        self._axes = vtk.vtkAxesActor()
        self._axes.SetShaftTypeToCylinder()
        self._axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetColor(1, 0, 0)
        self._axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetFontFamilyToArial()
        self._axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().ShadowOff()
        self._axes.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetColor(0, 1, 0)
        self._axes.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetFontFamilyToArial()
        self._axes.GetYAxisCaptionActor2D().GetCaptionTextProperty().ShadowOff()
        self._axes.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetColor(0, 0, 1)
        self._axes.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetFontFamilyToArial()
        self._axes.GetZAxisCaptionActor2D().GetCaptionTextProperty().ShadowOff()
        
        # create axes marker
        self._marker = vtk.vtkOrientationMarkerWidget()
        self._marker.SetInteractor(renWinInteract)
        self._marker.SetOrientationMarker(self._axes)
        self._marker.SetViewport(0, 0, 0.25, 0.25)
        self._marker.SetEnabled(0)
        self._enabled = False
    
    def isEnabled(self):
        """Returns True if the axes is enabled."""
        return self._enabled
    
    def toggle(self):
        """Toggle axes visibilty."""
        if self.isEnabled():
            self.remove()
        else:
            self.add()
    
    def add(self):
        """Add the axis label."""
        if not self.isEnabled():
            self._marker.SetEnabled(1)
            self._enabled = True
    
    def remove(self):
        """Remove the axis label."""
        if self.isEnabled():
            self._marker.SetEnabled(0)
            self._enabled = False
