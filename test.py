import os
import importlib
import importlib.abc

print(f"test.py 当前工作目录={os.getcwd()}")
# import foo.test_b as b
import foo.test_a as a
# import foo.test_c as c
import foo.foo_sub_src.test_c as c
# b.print_me()
# a.print_me()

c.print_me()