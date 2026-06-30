@echo off
set "ROOT=%~dp0.."
cd /d "%ROOT%"
if not exist "logs" mkdir "logs"
python "scripts\train_local_ai_tools_models.py" --persist-artifact --confirm-phase3-rebuild > "logs\local_ai_tools_autotrain.log" 2>&1
