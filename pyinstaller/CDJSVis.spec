# -*- mode: python -*-

import os
import subprocess

__version__ = subprocess.Popen(["git", "describe"], stdout=subprocess.PIPE).communicate()[0].strip()

a = Analysis(['../CDJSVis.py'],
             pathex=[],
             hiddenimports=[],
             hookspath=None)

a.datas += [('data/atoms.IN', '../CDJSVis/data/atoms.IN', 'DATA')]

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
