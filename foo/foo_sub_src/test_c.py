import foo.utils.path_helper as pathhelper

def print_me():
    cur_file_name = pathhelper.get_filename()
    print(f"我是test_c, cur_file_name={cur_file_name}")