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
exclude_compile_files = ["foo\\foo_sub_src\\sub_3\\test_sub_3.py"]
exclude_compile_dir = ["foo\\foo_sub_src\\sub_4"]

need_compile_modules_map ={}
need_mkdir_list = []

def dirInExcludeList(full_path):
    for item in exclude_compile_dir:
        if item in full_path:
            return True
    return False

def fileInExcludeList(full_path):
    for item in exclude_compile_files:
        if item in full_path:
            return True
    return False

def findAllFile(base,need_compile_list, module_name):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if dirInExcludeList(root):
                print(f"{root} in Exclude dir list, exit file tranverse")
                break

            ext = os.path.splitext(f)[-1]
            filename = os.path.splitext(f)[0]
            full_path = os.path.join(root, f)
            exclude_file = fileInExcludeList(full_path)
            if exclude_file:
                print(f"{full_path} in Exclude file list")

            if ext == '.py' and filename !="__init__" and not exclude_file:
                need_compile_list.append(full_path)
        for d in ds:
            full_path = os.path.join(root, d)
            exclude_dir = dirInExcludeList(full_path)

            if exclude_dir:
                print(f"{full_path} in Exclude dir list")

            if d !='__pycache__' and not exclude_dir:
                need_mkdir_list.append(full_path)


for item in need_compile_modules:
    compile_list = []
    real_dir = os.path.join(CUR_FILE_DIR, item)
    findAllFile(real_dir, compile_list, item)
    need_compile_modules_map[item] = compile_list

need_compile_extensions = []
for key in need_compile_modules_map:
    extension=Extension(
            name=f"{key}.bootstrap",
            sources = need_compile_modules_map[key],
    )

    need_compile_extensions.append(extension)

# sourcefiles2 = ['foo/foo_sub_src/__init__.py']
# extension2=Extension(
#             name="foo.bootstrap",
#             sources = sourcefiles2,
#     )
    
# need_compile_extensions.append(extension2)

all_compile_extensions_tuple= tuple(i for i in need_compile_extensions)

# need_compile_list = ['foo/bootstrap.py', 'foo/test_a.py', 'foo/test_b.py', 'foo/foo_sub_src/test_c.py', 'foo/utils/path_helper.py']
# sourcefiles2 = ['foo_src/foo_sub_src/bootstrap.py', 'foo_src/foo_sub_src/test_c.py']
# extension1=Extension(
#             name="foo.bootstrap",
#             sources = need_compile_list
#     )
# extension2=Extension(
#             name="foo.foo_sub_src.bootstrap",
#             sources = sourcefiles2,
#     )

extensions = cythonize(all_compile_extensions_tuple, build_dir=c_file_build_dir, language_level=3)
 
kwargs = {
      'name':'applechangtest',
      'packages':find_packages(),
      'ext_modules':  extensions,
}
 
setup(**kwargs)

build_out_dir = None
build_new_out_dir = None
if platform.system()=='Windows':
    build_out_dir = os.path.join(build_dir, 'lib.win-amd64-3.8')
else:
    build_out_dir = os.path.join(build_dir, 'lib.linux-x86_64-cpython-38')

build_new_out_dir = os.path.join(build_dir, 'temp')


for item in need_compile_modules:
    real_dir = os.path.join(CUR_FILE_DIR, item)
    init_py_path = os.path.join(real_dir, "__init__.py")
    dst_dir = os.path.join(build_out_dir, item, "__init__.py")
    shutil.copyfile(init_py_path, dst_dir)

if os.path.exists(build_out_dir):
    os.rename(build_out_dir, build_new_out_dir)

for item in need_mkdir_list:
    module_init_src_py = os.path.join(item, "__init__.py")
    x = item.replace(CUR_FILE_DIR, build_new_out_dir)
    if not os.path.exists(x):
        os.makedirs(x)
    module_init_dst_py = os.path.join(x, "__init__.py")
    shutil.copyfile(module_init_src_py, module_init_dst_py)

for key in need_compile_modules_map:
    for item in need_compile_modules_map[key]:
        if os.path.exists(item): 
            print(f'remove source file:{item}')
            os.remove(item) 
            portion = os.path.splitext(item)
            if portion[1] == '.py':
                c_temp_file = portion[0] + ".c"
                if os.path.exists(c_temp_file):
                    print(f'remove c file:{c_temp_file}')
                    os.remove(c_temp_file)
        else:
            print(f'no such file:{item}')


