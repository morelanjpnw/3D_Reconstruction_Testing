@echo off
cd /d "%~dp0"
if exist "tools\env.local.cmd" (
    call tools\env.local.cmd streamlit run scripts\app.py
) else (
    call tools\env.cmd streamlit run scripts\app.py
)
