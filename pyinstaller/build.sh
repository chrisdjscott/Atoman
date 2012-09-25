#!/bin/bash

rm -rf build/ dist/ *.log

python ${HOME}/git/pyinstaller/pyinstaller.py CDJSVis.spec

cd dist

echo zip -r CDJSVis.zip CDJSVis.app
zip -r CDJSVis.zip CDJSVis.app

exit 0
