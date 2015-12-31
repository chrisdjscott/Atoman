
"""
Initialise the application.

@author: Chris Scott

"""
from __future__ import absolute_import
from __future__ import unicode_literals
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

from PySide import QtGui, QtCore

# set default for stream handler (we don't want it to be NOTSET by default)
_argLevel = None
if len(sys.argv) > 1:
    if "DEBUG" in sys.argv:
        _argLevel = logging.DEBUG
    elif "INFO" in sys.argv:
        _argLevel = logging.INFO
    elif "WARNING" in sys.argv:
        _argLevel = logging.WARNING
    elif "ERROR" in sys.argv:
        _argLevel = logging.ERROR
    elif "CRITICAL" in sys.argv:
        _argLevel = logging.CRITICAL
if _argLevel is None:
    logging.getLogger().handlers[0].setLevel(logging.WARNING)
else:
    logging.getLogger().handlers[0].setLevel(_argLevel)

from .gui import mainWindow
from .visutils.utilities import iconPath


def main():
    # fix font bug on mavericks
    if platform.version().startswith("Darwin Kernel Version 13.1.0"):
        QtGui.QFont.insertSubstitution(".Lucida Grande UI", "Lucida Grande")
    
    # application
    app = QtGui.QApplication(sys.argv)
    
    # set application info used by QSettings
    app.setOrganizationName("chrisdjscott")
    app.setApplicationName("Atoman")
    
    # set default logging
    if _argLevel is None:
        settings = QtCore.QSettings()
        level = settings.value("logging/standard", logging.WARNING)
        logging.getLogger().handlers[0].setLevel(level)
    
    # pass QDesktopWidget to app so it can access screen info
    desktop = app.desktop()
    
    # create main window
    mw = mainWindow.MainWindow(desktop)
    mw.setWindowIcon(QtGui.QIcon(iconPath("atoman.png")))
    
    # show main window and give it focus
    mw.show()
    mw.raise_()
    
    # run application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
