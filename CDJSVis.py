#!/usr/bin/env python

"""
Initialise application.

@author: Chris Scott

"""
import sys
import multiprocessing

from PySide import QtGui, QtCore

from CDJSVis import mainWindow
from CDJSVis.visutils.utilities import iconPath, imagePath

################################################################################

def main():
    # application
    app = QtGui.QApplication(sys.argv)
    
    # set application info used by QSettings
    app.setOrganizationName("chrisdjscott")
    app.setApplicationName("CDJSVis")
    
    # display splash screen
#     splash_pix = QtGui.QPixmap(imagePath("splash_loading.png"))
#     splash = QtGui.QSplashScreen(splash_pix, QtCore.Qt.WindowStaysOnTopHint)
#     splash.setMask(splash_pix.mask())
#     splash.show()
#     app.processEvents()
    
    # pass QDesktopWidget to app so it can access screen info
    desktop = app.desktop()
    
    # create main window
    mw = mainWindow.MainWindow(desktop)
    mw.setWindowIcon(QtGui.QIcon(iconPath("CDJSVis.ico")))
    
    # show main window and give it focus
    mw.show()
    mw.raise_()
    
    sys.exit(app.exec_())

################################################################################

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
