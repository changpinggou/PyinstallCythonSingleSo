import os
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import shutil

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

try:
    linux_build_temp_dir = os.path.join(build_dir, 'lib.linux-x86_64-cpython-38')
    linux_build_new_dir = os.path.join(build_dir, 'temp')
    win_build_temp_dir = os.path.join(build_dir, 'lib.win-amd64-3.8')
    win_build_new_dir = os.path.join(build_dir, 'temp')
    if os.path.exists(linux_build_temp_dir):
        os.rename(linux_build_temp_dir, linux_build_new_dir)
    if os.path.exists(win_build_temp_dir):
        os.rename(win_build_temp_dir, win_build_new_dir)
except Exception as e:
    print(f"os.rename={linux_build_temp_dir}->{linux_build_new_dir} exception={str(e)}")
    print(f"os.rename={win_build_temp_dir}->{win_build_new_dir} exception={str(e)}")

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


