#!/usr/bin/env python

"""
Initialise application.

@author: Chris Scott

"""
import sys

from PyQt4 import QtGui, QtCore

from CDJSVis import mainWindow
from CDJSVis.visutils.utilities import iconPath, imagePath

################################################################################

def main():
    # application
    app = QtGui.QApplication(sys.argv)
    app.setOrganizationName("chrisdjscott")
    app.setApplicationName("CDJSVis")
    
    # display splash screen
    splash_pix = QtGui.QPixmap(imagePath("splash_loading.png"))
    splash = QtGui.QSplashScreen(splash_pix, QtCore.Qt.WindowStaysOnTopHint)
    splash.setMask(splash_pix.mask())
    splash.show()
    app.processEvents()
    
    # main window
    mw = mainWindow.MainWindow()
    mw.setWindowIcon(QtGui.QIcon(iconPath("CDJSVis.ico")))
    
    sys.exit(app.exec_())

################################################################################

if __name__ == '__main__':
    main()
