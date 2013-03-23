# -*- mode: python -*-

import os
import glob
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

so_files = glob.glob("../CDJSVis/visclibs/*.so")
so_files = [os.path.basename(fn) for fn in so_files]
dylib_files = glob.glob("../CDJSVis/visclibs/*.dylib")
dylib_files = [os.path.basename(fn) for fn in dylib_files]

os.chdir(os.path.join("dist", "CDJSVis.app", "Contents", "MacOS"))

for so_file in so_files:
    if os.path.exists(so_file):
        fn = so_file[:-3] + ".dylib"
        if os.path.exists(fn):
            print "Removing %s (%s exists)" % (fn, so_file)
            os.unlink(fn)

for dy_file in dylib_files:
    if os.path.exists(dy_file):
        fn = dy_file[:-6] + ".so"
        if os.path.exists(fn):
            print "Removing %s (%s exists)" % (fn, dy_file)
            os.unlink(fn)

os.chdir("../../../../")

osname = platform.system()
if osname == "Darwin":
    # check qtmenu.nib got copied
    if not os.path.isdir("dist/CDJSVis.app/Contents/Resources/qt_menu.nib"):
        print "qt_menu.nib not found"
        
        shutil.copytree("/opt/local/Library/Frameworks/QtGui.framework/Versions/Current/Resources/qt_menu.nib", "dist/CDJSVis.app/Contents/Resources/qt_menu.nib")
        
