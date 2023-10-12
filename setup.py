import os
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

CUR_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"CUR_FILE_DIR = {CUR_FILE_DIR}")
os.chdir(CUR_FILE_DIR)

build_dir=os.path.join(CUR_FILE_DIR, 'build')
c_file_build_dir=os.path.join(build_dir, 'cc-file')

#  
need_compile_list = ['foo/bootstrap.py', 'foo/test_a.py', 'foo/test_b.py', 'foo/foo_sub_src/test_c.py', 'foo/utils/path_helper.py']
# sourcefiles2 = ['foo_src/foo_sub_src/bootstrap.py', 'foo_src/foo_sub_src/test_c.py']
extension1=Extension(
            name="foo.bootstrap",
            sources = need_compile_list,
    )

# extension2=Extension(
#             name="foo.foo_sub_src.bootstrap",
#             sources = sourcefiles2,
#     )

extensions = cythonize((extension1), build_dir=c_file_build_dir, language_level=3)
 
kwargs = {
      'name':'applechangtest',
      'packages':find_packages(),
      'ext_modules':  extensions,
}
 
setup(**kwargs)

for item in need_compile_list:
    if os.path.exists(item): 
        print(f'remove file:{item}')
        os.remove(item) 
    else:
        print(f'no such file:{item}')