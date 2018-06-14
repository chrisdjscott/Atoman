# -*- coding: utf-8 -*-

"""
The view ports widget

@author: Chris Scott

"""
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
import logging

from PyQt5 import QtWidgets

from . import rendererSubWindow


class ViewPortsWidget(QtWidgets.QWidget):
    """
    Class for holding view ports (renderer windows)

    """
    def __init__(self, parent=None):
        super(ViewPortsWidget, self).__init__(parent)

        self._logger = logging.getLogger(__name__)
        self._viewPorts = []
        self._layout = QtWidgets.QGridLayout(self)
        self._mainWindow = parent

    def numViewPortsChanged(self, num):
        """Add/remove view ports."""
        currentNum = len(self._viewPorts)
        if num == currentNum:
            self._logger.debug("No change in number of view ports ({0})".format(num))

        else:
            if num > currentNum:
                self._logger.debug("Adding more view ports ({0} was {1})".format(num, currentNum))

                for i in range(currentNum, num):
                    row = i // 2
                    col = i % 2
                    self._logger.debug("Adding view port with index {0} ({1}, {2})".format(i, row, col))

                    rw = rendererSubWindow.RendererWindow(self._mainWindow, i, parent=self)
                    self._viewPorts.append(rw)
                    self._layout.addWidget(rw, row, col)

            else:
                self._logger.debug("Removing view ports ({0} was {1})".format(num, currentNum))

                while len(self._viewPorts) > num:
                    rw = self._viewPorts.pop()
                    self._layout.removeWidget(rw)
                    rw.deleteLater()

#            for rw in self._viewPorts:
#                rw.outputDialog.imageTab.imageSequenceTab.refreshLinkedRenderers()

    def getViewPorts(self):
        """Return the list of current view ports."""
        return self._viewPorts
