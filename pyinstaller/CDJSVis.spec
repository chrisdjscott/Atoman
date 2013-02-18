# -*- mode: python -*-

import os
import platform
import shutil
import subprocess

__version__ = subprocess.Popen(["git", "describe"], stdout=subprocess.PIPE).communicate()[0].strip()

a = Analysis(['../CDJSVis.py'],
             pathex=[],
             hiddenimports=[],
             hookspath=None)

a.datas += [('data/atoms.IN', '../CDJSVis/data/atoms.IN', 'DATA'),
            ('data/bonds.IN', '../CDJSVis/data/bonds.IN', 'DATA')]

pyz = PYZ(a.pure)

exe = EXE(pyz,
          a.scripts,
          exclude_binaries=1,
          name=os.path.join('build/pyi.darwin/CDJSVis', 'CDJSVis'),
          debug=False,
          strip=None,
          upx=True,
          console=False)

coll = COLLECT( exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=None,
               upx=True,
               name=os.path.join('dist', 'CDJSVis'))

app = BUNDLE(coll,
             name=os.path.join('dist', 'CDJSVis.app'),
             version=__version__)

osname = platform.system()
if osname == "Darwin":
    # check qtmenu.nib got copied
    if not os.path.isdir("dist/CDJSVis.app/Contents/Resources/qt_menu.nib"):
        print "qt_menu.nib not found"
        
        shutil.copytree("/opt/local/Library/Frameworks/QtGui.framework/Versions/Current/Resources/qt_menu.nib", "dist/CDJSVis.app/Contents/Resources/qt_menu.nib")
        
