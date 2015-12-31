
"""
GUI utilities

@author: Chris Scott

"""
from __future__ import absolute_import
from __future__ import unicode_literals
import logging

from PySide import QtGui, QtCore


################################################################################

def positionWindow(window, windowSize, desktop, parentWidget, offset=30, border=10):
    """
    Positions window near cursor
    
    """
    logger = logging.getLogger(__name__)
    logger.debug("Positioning window")
    
    # need cursor position on screen to decide where to open window
    cursor = QtGui.QCursor()
    cursor_pos = cursor.pos()
    logger.debug("Cursor pos: (%d, %d)", cursor_pos.x(), cursor_pos.y())
    
    # first determine screen size, which screen, etc
    screenNumber = desktop.screenNumber(widget=parentWidget)
    logger.debug("Screen number: %d", screenNumber)
    screenGeometry = desktop.availableGeometry(parentWidget)
    logger.debug("Screen geometry: (%d, %d, %d, %d)", screenGeometry.left(), screenGeometry.top(), 
                 screenGeometry.width(), screenGeometry.height())
    
    # now window size
    windowWidth = windowSize.width()
    windowHeight = windowSize.height()
    logger.debug("Window size: %d x %d", windowWidth, windowHeight)
    logger.debug("Cursor offset: %d", offset)
    logger.debug("Screen border: %d", border)
    
    # first determine x position: right, left or centre
    logger.debug("Checking right")
    
    # fits right if: point_x + offset + window_width < screen_max_x - border
    window_x_max = cursor_pos.x() + offset + windowWidth
    screen_max_x = screenGeometry.left() + screenGeometry.width() - border
    logger.debug("  Window/screen max x: %d < %d?", window_x_max, screen_max_x)
    
    if window_x_max < screen_max_x:
        logger.debug("Window fits right")
        
        new_x = cursor_pos.x() + offset
        
    else:
        logger.debug("Checking left")
        
        # fits left if: point_x - offset - window_width > screen_min_x + border
        window_x_min = cursor_pos.x() - offset - windowWidth
        screen_min_x = screenGeometry.left() + border
        logger.debug("  Window/screen min x: %d > %d?", window_x_min, screen_min_x)
        
        if window_x_min > screen_min_x:
            logger.debug("Window fits left")
            
            new_x = cursor_pos.x() - offset - windowWidth
        
        else:
            logger.debug("Centering window left to right")
            
            new_x = screenGeometry.left() + (screenGeometry.width() - windowWidth) / 2.0
    
    # now determine y position: below, above or centre
    logger.debug("Checking fits below")
    
    # fits below if: point_y - offset - window_height > screen_min_y + border
    window_y_max = cursor_pos.y() + offset + windowHeight
    screen_max_y = screenGeometry.top() + screenGeometry.height() - border
    logger.debug("  Window/screen max y: %d < %d?", window_y_max, screen_max_y)
    
    if window_y_max < screen_max_y:
        logger.debug("Window fits below")
        
        new_y = cursor_pos.y() + offset
    
    else:
        logger.debug("Checking fits above")
        
        # fits above if: point_y + offset + window_height < screen_max_x - border
        window_y_min = cursor_pos.y() - offset - windowHeight
        screen_min_y = screenGeometry.top() + border
        logger.debug("  Window/screen min y: %d > %d?", window_y_min, screen_min_y)
        
        if window_y_min > screen_min_y:
            logger.debug("Window fits above")
            
            new_y = cursor_pos.y() - offset - windowHeight
        
        else:
            logger.debug("Centering window above to below")
            
            new_y = screenGeometry.top() + (screenGeometry.height() - windowHeight) / 2.0
    
    # set position of window
    windowPoint = QtCore.QPoint(new_x, new_y)
    
    logger.debug("Setting window position: (%d, %d)", new_x, new_y)
    
    window.setGeometry(QtCore.QRect(windowPoint, window.size()))

################################################################################

def showProgressDialog(title, label, parent, overrideCursor=True):
    """
    Show (and return) a progress dialog
    
    """
    progress = QtGui.QProgressDialog(parent=parent)
    progress.setWindowModality(QtCore.Qt.WindowModal)
    progress.setWindowTitle(title)
    progress.setLabelText(label)
    progress.setRange(0, 0)
    progress.setMinimumDuration(0)
    progress.setModal(True)
    QtGui.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
    progress.show()
    
    return progress

################################################################################

def cancelProgressDialog(progress, overrideCursor=True):
    """
    Cancel progress dialog
    
    """
    progress.cancel()
    
    if overrideCursor:
        QtGui.QApplication.restoreOverrideCursor()
