# -*- mode: python -*-

import os
import glob
import platform
import shutil
import subprocess

__version__ = subprocess.Popen(["git", "describe"], stdout=subprocess.PIPE).communicate()[0].strip()

a = Analysis(['../cdjsvis.py'],
             pathex=[],
             hiddenimports=[],
             hookspath=None)

a.datas += [('data/atoms.IN', '../CDJSVis/data/atoms.IN', 'DATA'),
            ('data/bonds.IN', '../CDJSVis/data/bonds.IN', 'DATA'),
            ('data/file_formats.IN', '../CDJSVis/data/file_formats.IN', 'DATA')]

pyz = PYZ(a.pure)

exe = EXE(pyz,
          a.scripts,
          exclude_binaries=1,
          name=os.path.join('build/pyi.darwin/CDJSVis', 'CDJSVis'),
          debug=False,
          strip=None,
          upx=True,
          console=False)

coll = COLLECT(exe,
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
        
# copy icns file
new_icns = os.path.join("dist", "CDJSVis.app", "Contents", "Resources", "CDJSVis.icns")
cmd = "cp -f CDJSVis.icns %s" % os.path.join("dist", "CDJSVis.app", "Contents", "Resources", "CDJSVis.icns")
print cmd
os.system(cmd)

# edit plist
plist_file = os.path.join("dist", "CDJSVis.app", "Contents", "Info.plist")
f = open(plist_file)
lines = f.readlines()
f.close()
f = open(plist_file, "w")
for line in lines:
    if line.startswith("<string>icon-windowed.icns"):
        line = "<string>CDJSVis.icns</string>\n"
    f.write(line)
f.close()
