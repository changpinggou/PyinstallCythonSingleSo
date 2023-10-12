import os

def get_filename(path):
    filename = os.path.splitext(os.path.basename(path))[0]
    return filename

def get_file_ext(path):
    ext = os.path.splitext(os.path.basename(path))[1]
    return ext

def get_file_by_ext(search_dir, file_suffix):
# 遍历指定目录及其子目录下的所有文件，匹配后缀名为 .txt 的那个文件并返回其路径
    for root, dirs, files in os.walk(search_dir):
        for file_name in files:
            if file_name.endswith(file_suffix):
                file_path = os.path.join(root, file_name)
                return file_path