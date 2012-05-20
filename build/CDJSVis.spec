# -*- mode: python -*-
a = Analysis(['/Users/macdjs/git/CDJSVis/src/CDJSVis.py'],
             pathex=['/Users/macdjs/git/pyinstaller'],
             hookspath=None)
a.datas += [('data/atoms.IN','/Users/macdjs/git/CDJSVis/src/data/atoms.IN', 'DATA')]
pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=1,
          name=os.path.join('build/pyi.darwin/CDJSVis', 'CDJSVis'),
          debug=False,
          strip=None,
          upx=True,
          console=False,
          icon='CDJSVis.icns' )
coll = COLLECT( exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=None,
               upx=True,
               name=os.path.join('dist', 'CDJSVis'))
app = BUNDLE(coll,
             name=os.path.join('dist', 'CDJSVis.app'),
             version="0.0.1")
