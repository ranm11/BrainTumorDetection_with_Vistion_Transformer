rem install Visual Studio 9: https://www.microsoft.com/en-us/download/confirmation.aspx?id=44266

pushd %~dp0
set VENV=..\torch_venv
if not exist %VENV% (
    call pip install venv
    call python -m venv %VENV%
    call %VENV%\Scripts\activate.bat
    call pip install torch  matplotlib torchvision IPython seaborn pytorch_lightning opencv-python
) ELSE (
    call %VENV%\Scripts\activate.bat
)
popd
