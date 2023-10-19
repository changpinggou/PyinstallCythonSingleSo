import os
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import shutil
import platform

CUR_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"CUR_FILE_DIR = {CUR_FILE_DIR}")
os.chdir(CUR_FILE_DIR)

build_dir=os.path.join(CUR_FILE_DIR, 'build')
c_file_build_dir=os.path.join(build_dir, 'cc-file')

try:
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
except Exception as e:
    print(f"shutil.rmtree={build_dir} exception={str(e)}")


need_compile_modules = ["foo"]
need_compile_modules_map ={}

def findAllFile(base,need_compile_list):
    for root, ds, fs in os.walk(base):
        for f in fs:
            ext = os.path.splitext(f)[-1]
            filename = os.path.splitext(f)[0]

            if ext == '.py' and filename !="__init__":
                full_path = os.path.join(root, f)
                need_compile_list.append(full_path)


for item in need_compile_modules:
    compile_list = []
    real_dir = os.path.join(CUR_FILE_DIR, item)
    findAllFile(real_dir, compile_list)
    need_compile_modules_map[item] = compile_list


need_compile_extensions = []
for key in need_compile_modules_map:
    extension=Extension(
            name=f"{key}.bootstrap",
            sources = need_compile_modules_map[key],
    )

    need_compile_extensions.append(extension)

all_compile_extensions_tuple= tuple(i for i in need_compile_extensions)

sourcefiles2 = ['foo/foo_sub_src/__init__.py']
extension2=Extension(
            name="foo.bootstrap",
            sources = sourcefiles2,
    )

extensions = cythonize((extension2), build_dir=c_file_build_dir, language_level=3)
 
kwargs = {
      'name':'applechangtest',
      'packages':find_packages(),
      'ext_modules':  extensions,
}
 
setup(**kwargs)



