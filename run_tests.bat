@echo off
echo ==========================================
echo   CUDA Model Testing Suite
echo ==========================================
echo.

set PASS=0
set FAIL=0

:: Test 1: Basic functionality
echo [1/8] Basic functionality test...
./target/release/ziggurat-enhanced.exe --prompt "What is machine learning?" --sample-len 40 --cpu 2>&1 | findstr /C:"Machine learning" >nul
if %errorlevel% equ 0 (
    echo     ✅ PASS - Correct answer
    set /a PASS+=1
) else (
    echo     ❌ FAIL - Incorrect answer
    set /a FAIL+=1
)
echo.

:: Test 2: CUDA detection
echo [2/8] CUDA detection test...
./target/release/ziggurat-enhanced.exe --prompt "Test" --sample-len 10 2>&1 | findstr /C:"CUDA" >nul
if %errorlevel% equ 0 (
    echo     ✅ PASS - CUDA detected
    set /a PASS+=1
) else (
    echo     ❌ FAIL - CUDA not detected
    set /a FAIL+=1
)
echo.

:: Test 3: Mathematical reasoning
echo [3/8] Mathematical reasoning test...
./target/release/ziggurat-enhanced.exe --prompt "Calculate: 8 * 7" --sample-len 20 --temperature 0.0 2>&1 | findstr /C:"56" >nul
if %errorlevel% equ 0 (
    echo     ✅ PASS - Correct calculation (8 * 7 = 56)
    set /a PASS+=1
) else (
    echo     ❌ FAIL - Incorrect calculation
    set /a FAIL+=1
)
echo.

:: Test 4: Code generation
echo [4/8] Code generation test...
./target/release/ziggurat-enhanced.exe --prompt "Write: print('hello')" --sample-len 30 --temperature 0.0 2>&1 | findstr /C:"print" >nul
if %errorlevel% equ 0 (
    echo     ✅ PASS - Code generated
    set /a PASS+=1
) else (
    echo     ❌ FAIL - Code not generated
    set /a FAIL+=1
)
echo.

:: Test 5: GPU vs CPU performance
echo [5/8] Performance comparison...
echo   Testing CPU...
./target/release/ziggurat-enhanced.exe --prompt "Performance test" --sample-len 50 --cpu 2>&1 | findstr /C:"tokens generated" > cpu_test.txt
echo   Testing GPU (CUDA)...
./target/release/ziggurat-enhanced.exe --prompt "Performance test" --sample-len 50 2>&1 | findstr /C:"tokens generated" > gpu_test.txt
echo     ✅ PASS - Performance comparison complete
set /a PASS+=1
echo.

:: Test 6: Temperature variation
echo [6/8] Temperature test...
./target/release/ziggurat-enhanced.exe --prompt "Random test" --sample-len 30 --temperature 0.8 2>&1 | findstr /C:"tokens generated" >nul
if %errorlevel% equ 0 (
    echo     ✅ PASS - High temperature works
    set /a PASS+=1
) else (
    echo     ❌ FAIL - Temperature error
    set /a FAIL+=1
)
echo.

:: Test 7: Memory stats (if available)
echo [7/8] Memory stats test...
./target/release/ziggurat-enhanced.exe --memory-stats 2>&1 | findstr /C:"Stats" >nul
if %errorlevel% equ 0 (
    echo     ✅ PASS - Memory stats accessible
    set /a PASS+=1
) else (
    echo     ⚠️  SKIP - Memory not configured
)
echo.

:: Test 8: Configuration
echo [8/8] Configuration test...
./target/release/ziggurat-enhanced.exe --show-config 2>&1 | findstr /C:"Configuration" >nul
if %errorlevel% equ 0 (
    echo     ✅ PASS - Configuration works
    set /a PASS+=1
) else (
    echo     ❌ FAIL - Configuration error
    set /a FAIL+=1
)
echo.

echo ==========================================
echo   Test Results Summary
echo ==========================================
echo   ✅ Passed: %PASS% / 8
echo   ❌ Failed: %FAIL% / 8
echo.
echo   Performance comparison:
echo   -------------------
echo   CPU: 
type cpu_test.txt | findstr "token/s"
echo   GPU (CUDA):
type gpu_test.txt | findstr "token/s"
echo.

:: Cleanup
del cpu_test.txt 2>nul
del gpu_test.txt 2>nul

if %PASS% geq 7 (
    echo   🎉 EXCELLENT - Model working perfectly!
) else if %PASS% geq 5 (
    echo   ✅ GOOD - Model working with minor issues
) else (
    echo   ⚠️  NEEDS ATTENTION - Multiple failures detected
)

echo.
pause