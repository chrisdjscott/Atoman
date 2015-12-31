
"""
Modify path to include macports and usr/local dirs.

"""
from __future__ import absolute_import
from __future__ import unicode_literals

import os

oldpath = os.environ["PATH"]

prepath = ""
if os.uname()[0] == "Darwin":
    prepath += "/opt/local/bin:/opt/local/sbin" + os.pathsep

prepath += "/usr/local/bin" + os.pathsep

os.environ["PATH"] = prepath + oldpath
