#!/bin/bash -e

## This script builds CDJSVis.app using pyinstaller
## First we get the current version and save it so 
##   it can be used when app is frozen

# CONFIG
PYINSTALLER_PATH="${HOME}/git/pyinstaller/pyinstaller.py"
# END CONFIG

VERSION=$(git describe)
echo "BUILDING: CDJSVis $VERSION"
echo "__version__ = \"$VERSION\"" > ../CDJSVis/visutils/version_freeze.py

rm -rf build/ dist/ *.log

cd ..

# start with a clean
python setup.py clean

# build_sphinx also runs 'build_ext --inplace'
python setup.py build_sphinx

# run tests
python setup.py test

cd pyinstaller

python ${PYINSTALLER_PATH} CDJSVis.spec

cd dist

echo zip -r CDJSVis-${VERSION}.zip CDJSVis.app
zip -r CDJSVis-${VERSION}.zip CDJSVis.app

exit 0
