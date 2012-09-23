#!/bin/bash

rm -rf build/ dist/ *.log

python ${HOME}/git/pyinstaller/pyinstaller.py --log-level=DEBUG CDJSVis.spec

exit 0
