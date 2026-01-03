@echo off
REM Batch file to run tests in venv
REM Usage: run_tests_venv.bat [test_type]
REM Example: run_tests_venv.bat smart

echo Activating virtual environment...
call venv\Scripts\activate.bat

if "%1"=="" (
    echo Running all fast tests...
    python run_comprehensive_tests.py fast
) else (
    echo Running: %*
    python run_comprehensive_tests.py %*
)

deactivate
