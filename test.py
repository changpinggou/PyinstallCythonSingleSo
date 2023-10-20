import os
import importlib
import importlib.abc
import importlib.util

print(f"test.py 当前工作目录={os.getcwd()}")
# import foo.test_b as b
# import foo.test_a as a
# import foo.test_c as c
import foo.foo_sub_src.test_c as c
import foo.utils.dh_utils.test_dh_utils as dh
import foo.foo_sub_src.sub_3.test_sub_3 as sub3

# b.print_me()
# a.print_me()

c.print_me()
sub3.print_me()

# dh.print_me()
