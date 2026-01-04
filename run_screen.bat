@echo off
setlocal

cd /d C:\Users\kenic\Desktop\stock_screen

REM venvを使っている場合（例）
call .venv\Scripts\activate

REM ログ保存（日時入り）
for /f "tokens=1-3 delims=/ " %%a in ("%date%") do set YYYY=%%a& set MM=%%b& set DD=%%c
for /f "tokens=1-3 delims=:., " %%d in ("%time%") do set HH=%%d& set MI=%%e& set SS=%%f
set LOG=logs\run_%YYYY%-%MM%-%DD%_%HH%-%MI%-%SS%.log
if not exist logs mkdir logs

python screen_yearend.py >> "%LOG%" 2>&1
