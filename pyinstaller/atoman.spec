# -*- mode: python -*-

from __future__ import print_function
import os
import sys
sys.path.insert(0, os.path.abspath(os.pardir))
import glob
import platform
import shutil
import subprocess
import tempfile

import pkg_resources


# get the version
_owd = os.getcwd()
os.chdir("..")
try:
    import atoman
    version = atoman.__version__
finally:
    os.chdir(_owd)

# write temporary script for use with PyInstaller
with tempfile.NamedTemporaryFile(mode="w", dir=os.pardir, delete=False) as fh:
    fh.write("import atoman.__main__\n")
    fh.write("atoman.__main__.main()\n")

try:
    version_file = os.path.join(os.pardir, "atoman", "version_freeze.py")
    with open(version_file, "w") as fver:
        fver.write("__version__ = '{0}'\n".format(version))
    
    try:
        # name of temporary file
        entryPointScript = fh.name
        
        # analysis object
        a = Analysis([entryPointScript],
                     pathex=[],
                     hiddenimports=['scipy.linalg.cython_blas', 'scipy.linalg.cython_lapack'],
                     hookspath=None)

        # add icons as data
        OWD = os.getcwd()
        os.chdir(os.path.join(os.pardir, "atoman"))
        try:
            extra_datas = []
            for root, dirs, files in os.walk("icons"):
                for fn in files:
                    if fn.startswith("."):
                        continue
                    data_path = os.path.join(root, fn)
                    rel_path = os.path.join(os.pardir, "atoman", data_path)
                    extra_datas.append((data_path, rel_path, "DATA"))
        finally:
            os.chdir(OWD)
        a.datas += extra_datas

        # add documentation as data
        OWD = os.getcwd()
        doc_root = os.path.join(os.pardir, "doc", "build", "html")
        os.chdir(doc_root)
        try:
            extra_datas = []
            for root, dirs, files in os.walk(os.getcwd()):
                for fn in files:
                    if fn.startswith("."):
                        continue
                    data_path = os.path.join("doc", os.path.relpath(root), fn)
                    rel_path = os.path.join(doc_root, os.path.relpath(root), fn)
                    extra_datas.append((data_path, rel_path, "DATA"))
        finally:
            os.chdir(OWD)
        a.datas += extra_datas
        
        # continue with build
        pyz = PYZ(a.pure)

        exe = EXE(pyz,
                  a.scripts,
                  exclude_binaries=True,
                  name=os.path.join('build/pyi.darwin/atoman', 'Atoman'),
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
                       name=os.path.join('dist', 'Atoman'))

        app = BUNDLE(coll,
                     name=os.path.join('dist', 'Atoman.app'),
                     version=version)

        # copy icns file
        new_icns = os.path.join("dist", "Atoman.app", "Contents", "Resources", "atoman.icns")
        cmd = "cp -f atoman.icns %s" % os.path.join("dist", "Atoman.app", "Contents", "Resources", "atoman.icns")
        print(cmd)
        os.system(cmd)

        # edit plist
        plist_file = os.path.join("dist", "Atoman.app", "Contents", "Info.plist")
        with open(plist_file) as f:
            lines = f.readlines()
        with open(plist_file, "w") as f:
            for line in lines:
                if line.startswith("<string>icon-windowed.icns"):
                    line = "<string>atoman.icns</string>\n"
                f.write(line)

    finally:
        # delete version file and pyc
        os.unlink(version_file)
        if os.path.exists(version_file + "c"):
            os.unlink(version_file + "c")

finally:
    # delete script
    os.unlink(entryPointScript)
