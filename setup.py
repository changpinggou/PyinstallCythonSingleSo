from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

#  
sourcefiles1 = ['foo_src/bootstrap.py', 'foo_src/test_a.py', 'foo_src/test_b.py', 'foo_src/foo_sub_src/test_c.py']
# sourcefiles2 = ['foo_src/foo_sub_src/bootstrap.py', 'foo_src/foo_sub_src/test_c.py']
extension1=Extension(
            name="foo.bootstrap",
            sources = sourcefiles1,
    )
# extension2=Extension(
#             name="foo.foo_sub_src.bootstrap",
#             sources = sourcefiles2,
#     )

extensions = cythonize((extension1))
 
 
kwargs = {
      'name':'applechangtest',
      'packages':find_packages(),
      'ext_modules':  extensions,
}
 
 
setup(**kwargs)

# python setup.py build_ext --inplace