@echo off
setlocal

REM === SET YOUR FOLDER PATH HERE ===
set "FOLDER=Z:\Student Folders\Nina_Gill\data\nwb_out"

REM === LOOP THROUGH EACH FILE IN FOLDER ===
for %%F in ("%FOLDER%\*") do (
    echo Processing file: "%%~fF"
    REM === PUT YOUR COMMAND HERE ===
    REM Example: type "%%~fF"
    pynwb-validate "%%~fF"
)

endlocal
pause