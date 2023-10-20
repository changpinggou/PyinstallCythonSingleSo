#!/bin/bash

# 提取工作区目录
# workspace=$(cd "$(dirname "$0")"  && pwd)
# echo "workspace = $workspace"

# 进行二进制化打包
python setup.py build_ext

# 编译 python
pyinstaller -D  --noconfirm test.py   --hidden-import importlib --hidden-import easydict --add-data "./build/temp:."