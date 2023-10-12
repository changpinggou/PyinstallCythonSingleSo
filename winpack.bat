pip install cython
pip install setuptools

pip install pyinstaller==5.13.2 --index-url https://pypi.tuna.tsinghua.edu.cn/simple

cd /d %~dp0
python setup.py build_ext

pyinstaller -D --exclude-module tensorflow  --noconfirm test.py   --hidden-import importlib --add-data "./build/temp;."


git reset --hard HEAD

pause