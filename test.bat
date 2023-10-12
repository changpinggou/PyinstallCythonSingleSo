pip install cython
pip install setuptools
pip install pyinstaller==5.13.2 --index-url https://pypi.tuna.tsinghua.edu.cn/simple

cd /d %~dp0
set current_dir=%cd%
del %current_dir%\build /S /Q
python setup.py build_ext
xcopy %current_dir%\build\lib.win-amd64-3.8\foo %current_dir%\foo /S /E /Q /Y /I

pause