@echo off
echo Setting up RDN Virtual Environment...
echo =====================================

REM Create virtual environment
echo Creating virtual environment 'rdn_env'...
python -m venv rdn_env

REM Check if virtual environment was created successfully
if not exist "rdn_env\Scripts\activate.bat" (
    echo ERROR: Failed to create virtual environment
    echo Please ensure Python is installed and accessible
    pause
    exit /b 1
)

echo Virtual environment created successfully!
echo.

REM Activate virtual environment
echo Activating virtual environment...
call rdn_env\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing compatible packages...
pip install -r requirements_fixed.txt

echo.
echo =====================================
echo Setup completed successfully!
echo.
echo To activate the environment in the future, run:
echo   rdn_env\Scripts\activate.bat
echo.
echo To test the installation, run:
echo   python test_basic.py
echo.
echo Press any key to test the installation now...
pause

REM Test the installation
echo Testing installation...
python test_basic.py

echo.
echo Setup and testing completed!
pause
