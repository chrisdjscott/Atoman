#!/usr/bin/env python

from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
import os
import sys
import glob

SIZE = 512
FORCE = False

svg_fns = glob.glob("*.svgz")
for svg_fn in svg_fns:
    png_fn = os.path.splitext(svg_fn)[0] + ".png"
    if os.path.isfile(png_fn) and not FORCE:
        continue
    command = 'inkscape --without-gui --export-png="{0}" --export-dpi=72 --export-background-opacity=0 --export-width={1} --export-height={1} "{2}" > /dev/null'.format(png_fn, SIZE, svg_fn)
    print(command)
    stat = os.system(command)
    if stat:
        sys.exit('Failed to convert file: "%s"' % svg_fn)
