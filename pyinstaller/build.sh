#!/bin/bash -e

VERSION="0.1.1"

rm -rf build/ dist/ *.log

cd ../src

python setup.py

cd ../pyinstaller

python ${HOME}/git/pyinstaller/pyinstaller.py CDJSVis.spec

cd dist

echo zip -r CDJSVis-${VERSION}.zip CDJSVis.app
zip -r CDJSVis-${VERSION}.zip CDJSVis.app

exit 0
