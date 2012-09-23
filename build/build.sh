#!/bin/bash

rm -rf build/ dist/ *.log

python ${HOME}/git/pyinstaller/pyinstaller.py CDJSVis.spec

exit 0
