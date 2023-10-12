 
import sys
import importlib
import importlib.abc
 
# 选择正确的初始函数   
class CythonPackageMetaPathFinder(importlib.abc.MetaPathFinder):
    def __init__(self, name_filter):
        super(CythonPackageMetaPathFinder, self).__init__()
        self.name_filter = name_filter
 
    def find_spec(self, fullname, path, target=None):
        print(f"======= find: fullname = {fullname}, path = {path}")
        if fullname.startswith(self.name_filter):
            # 使用这个扩展文件,而不是其他模块的pyinit函数:
            loader = importlib.machinery.ExtensionFileLoader(fullname, __file__)
            print(f"======__file__ = {__file__}")
            print(f"======loader = {loader}")
            spec_from_loader = importlib.util.spec_from_loader(fullname, loader)
            print(f"======spec_from_loader = {spec_from_loader}")
            return spec_from_loader
    
# 将自定义查找器/加载 => sys.meta_path:
def bootstrap_cython_submodules():
    sys.meta_path.append(CythonPackageMetaPathFinder('foo')) 