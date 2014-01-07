#!/usr/bin/env python

"""
Initialise application.

@author: Chris Scott

"""
# configure logging (we have to set logging.NOTSET here as global for root logger)
import logging
logging.basicConfig(format="%(levelname)s: %(name)s: %(message)s", level=logging.NOTSET)

# set default for stream handler (we don't want it to be NOTSET by default)
logging.getLogger().handlers[0].setLevel(logging.WARNING)

import sys
import multiprocessing

from PySide import QtGui, QtCore

from CDJSVis import mainWindow
from CDJSVis.visutils.utilities import iconPath, imagePath, setupLogging

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
    
    # set default logging
    setupLogging(sys.argv)
    
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
