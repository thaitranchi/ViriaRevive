@echo off
set "MODE=%~1"
if "%MODE%"=="" set "MODE=release"

echo === ViriaRevive Build [%MODE%] ===
echo.

REM Activate venv
call venv\Scripts\activate.bat

REM Generate tray icon first
python -c "from tray import _create_icon_image; _create_icon_image(); print('[+] Tray icon generated')" || (echo [!] Failed to generate tray icon. && pause && exit /b 1)

REM Build with PyInstaller
echo [*] Building with PyInstaller...
if "%MODE%"=="debug" (
    set "VIRIA_DEBUG=1"
    pyinstaller viria.spec --noconfirm --clean --debug=all
) else (
    pyinstaller viria.spec --noconfirm --clean
)

echo.
if exist "dist\ViriaRevive\ViriaRevive.exe" (
    echo [+] Build successful!
    echo     Output: dist\ViriaRevive\ViriaRevive.exe
    echo.
    echo NOTE: Copy these to the dist\ViriaRevive\ folder before running:
    echo   - client_secrets.json (for YouTube)
    echo   - music/ folder (for background music)
    echo   - ffmpeg.exe must be in PATH
) else (
    echo [!] Build failed. Check the output above for errors.
)
pause
