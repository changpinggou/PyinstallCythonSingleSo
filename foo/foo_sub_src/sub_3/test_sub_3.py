import os
import sys

from foo.foo_sub_src.sub_3.train_config_eye32  import config as cfg

CURRENT_WORK_DIR = os.getcwd()
# WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(CURRENT_WORK_DIR)))
WORKSPACE_FOO = os.path.join(CURRENT_WORK_DIR, 'foo')
print(f"test_c.py 当前工作目录={os.getcwd()}")
# sys.path.append(WORKSPACE)
sys.path.append(WORKSPACE_FOO)

import foo.utils.path_helper as pathhelper

def print_me():

    cur_file_name = pathhelper.get_filename(__file__)
    print(f"我是test_sub_3, cur_file_name={cur_file_name}")
    print(f"test_sub_3.py 当前路径：={__file__}")
    print(f"cfg.trace={cfg.TRACE.ema_or_one_euro}")
    
