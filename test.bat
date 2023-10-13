pip install cython
pip install setuptools
pip install pyinstaller==5.13.2 --index-url https://pypi.tuna.tsinghua.edu.cn/simple

cd /d %~dp0
set current_dir=%cd%
del %current_dir%\build /S /Q
python setup.py build_ext
xcopy %current_dir%\build\temp\foo %current_dir%\foo_test /S /E /Q /Y /I

@REM git reset --hard HEAD
pause