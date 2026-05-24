import glob
import py_compile
import sys

files = glob.glob('logic/**/*.py', recursive=True)
failed = False
for f in files:
    try:
        py_compile.compile(f, doraise=True)
    except Exception as e:
        print('COMPILE_ERROR:', f, e)
        failed = True
if not failed:
    print('ALL_COMPILED')
else:
    sys.exit(2)
