@echo off
set "ROOT=%~dp0.."
cd /d "%ROOT%"
if not exist "logs" mkdir "logs"
python "scripts\train_local_ai_tools_models.py" > "logs\local_ai_tools_autotrain.log" 2>&1
