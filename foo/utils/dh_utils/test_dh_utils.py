import os
import sys
import foo.foo_sub_src.sub_3.test_sub_3 as sub3

CURRENT_WORK_DIR = os.getcwd()
# WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(CURRENT_WORK_DIR)))
WORKSPACE_FOO = os.path.join(CURRENT_WORK_DIR, 'foo')
print(f"test_c.py 当前工作目录={os.getcwd()}")
# sys.path.append(WORKSPACE)
sys.path.append(WORKSPACE_FOO)

import foo.utils.path_helper as pathhelper

def print_me():
    sub3.print_me()
    
    cur_file_name = pathhelper.get_filename(__file__)
    print(f"我是test_dh_utils, cur_file_name={cur_file_name}")
    print(f"我是test_dh_utils.py 当前路径：={__file__}")