
"""
Initialise the application.

@author: Chris Scott

"""
import sys
import platform
import logging
try:
    from rainbow_logging_handler import RainbowLoggingHandler
    root = logging.getLogger()
    root.setLevel(logging.NOTSET)
    formatter = logging.Formatter("%(name)s:%(funcName)s():%(lineno)d: %(message)s")
    handler = RainbowLoggingHandler(sys.stderr)
    handler.setFormatter(formatter)
    root.addHandler(handler)
except ImportError:
    # configure logging (we have to set logging.NOTSET here as global for root logger)
    logging.basicConfig(format="%(levelname)s: %(name)s: %(message)s", level=logging.NOTSET)
# set default for stream handler (we don't want it to be NOTSET by default)
logging.getLogger().handlers[0].setLevel(logging.WARNING)
import multiprocessing

from PySide import QtGui

from . import mainWindow
from .visutils.utilities import iconPath, setupLogging


def main():
    # fix font bug on mavericks
    if platform.version().startswith("Darwin Kernel Version 13.1.0"):
        QtGui.QFont.insertSubstitution(".Lucida Grande UI", "Lucida Grande")
    
    # application
    app = QtGui.QApplication(sys.argv)
    
    # set application info used by QSettings
    app.setOrganizationName("chrisdjscott")
    app.setApplicationName("atoman")
    
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
    mw.setWindowIcon(QtGui.QIcon(iconPath("atoman.ico")))
    
    # show main window and give it focus
    mw.show()
    mw.raise_()
    
    # run application
    sys.exit(app.exec_())

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
