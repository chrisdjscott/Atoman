# -*- mode: python -*-

import os
rootpath = os.getenv("HOME")

a = Analysis([os.path.join(rootpath, 'git/CDJSVis/src/CDJSVis.py')],
             pathex=[],
             hiddenimports=[],
             hookspath=None)

a.datas += [('data/atoms.IN', os.path.join(rootpath, 'git/CDJSVis/src/data/atoms.IN'), 'DATA')]

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
             version="0.3.2")
