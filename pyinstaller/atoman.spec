# -*- mode: python -*-

import os
import glob
import platform
import shutil
import subprocess
import tempfile

import pkg_resources


# read the version
__version__ = subprocess.Popen(["git", "describe"], stdout=subprocess.PIPE).communicate()[0].strip()

# write temporary script for use with PyInstaller
with tempfile.NamedTemporaryFile(mode="w", dir="..", delete=False) as fh:
    fh.write("import atoman.__main__\n")
    fh.write("atoman.__main__.main()\n")
entryPointScript = fh.name

try:
	a = Analysis([entryPointScript],
				 pathex=[],
				 hiddenimports=['scipy.linalg.cython_blas', 'scipy.linalg.cython_lapack'],
				 hookspath=None)

	# data files
	a.datas += [('data/atoms.IN', '../atoman/data/atoms.IN', 'DATA'),
				('data/bonds.IN', '../atoman/data/bonds.IN', 'DATA'),
				('data/file_formats.IN', '../atoman/data/file_formats.IN', 'DATA')]

	# add icons as data
	OWD = os.getcwd()
	os.chdir("../atoman")
	try:
		extra_datas = []
		for root, dirs, files in os.walk("icons"):
			for fn in files:
				if fn.startswith("."):
					continue
				data_path = os.path.join(root, fn)
				rel_path = os.path.join("..", "atoman", data_path)
				extra_datas.append((data_path, rel_path, "DATA"))
	finally:
		os.chdir(OWD)
	a.datas += extra_datas

	# add documentation as data
	OWD = os.getcwd()
	os.chdir("../atoman")
	try:
		extra_datas = []
		for root, dirs, files in os.walk("doc"):
			for fn in files:
				if fn.startswith("."):
					continue
				data_path = os.path.join(root, fn)
				rel_path = os.path.join("..", "atoman", data_path)
				extra_datas.append((data_path, rel_path, "DATA"))
	finally:
		os.chdir(OWD)
	a.datas += extra_datas

	# continue with build
	pyz = PYZ(a.pure)

	exe = EXE(pyz,
			  a.scripts,
			  exclude_binaries=1,
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
				 version=__version__)

	osname = platform.system()
	if osname == "Darwin":
		# check qtmenu.nib got copied
		if not os.path.isdir("dist/Atoman.app/Contents/Resources/qt_menu.nib"):
			print "qt_menu.nib not found -> attempting to fix..."
			shutil.copytree("/opt/local/libexec/qt4/Library/Frameworks/QtGui.framework/Versions/Current/Resources/qt_menu.nib", "dist/Atoman.app/Contents/Resources/qt_menu.nib")

	# copy icns file
	new_icns = os.path.join("dist", "Atoman.app", "Contents", "Resources", "atoman.icns")
	cmd = "cp -f atoman.icns %s" % os.path.join("dist", "Atoman.app", "Contents", "Resources", "atoman.icns")
	print cmd
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
	# delete script
	os.unlink(entryPointScript)
