@echo off
echo === Testing CUDA Model Implementation ===
echo.

echo [Test 1] Basic functionality test
echo Query: "What is machine learning?"
echo Expected: Coherent explanation of machine learning
echo Command: ziggurat-enhanced.exe --prompt "What is machine learning?" --sample-len 60 --cpu
echo.
call :run_test "What is machine learning?" 60

echo.
echo [Test 2] Mathematical reasoning
echo Query: "Solve: 2 + 2 * 3"
echo Expected: Correct mathematical answer with explanation
echo Command: ziggurat-enhanced.exe --prompt "Solve: 2 + 2 * 3" --sample-len 30 --cpu
echo.
call :run_test "Solve: 2 + 2 * 3" 30

echo.
echo [Test 3] Creative response
echo Query: "Write a short poem about technology"
echo Expected: Creative poem about technology
echo Command: ziggurat-enhanced.exe --prompt "Write a short poem about technology" --sample-len 80 --cpu
echo.
call :run_test "Write a short poem about technology" 80

echo.
echo [Test 4] Code generation
echo Query: "Write a Python function to calculate factorial"
echo Expected: Correct Python code for factorial
echo Command: ziggurat-enhanced.exe --prompt "Write a Python function to calculate factorial" --sample-len 100 --cpu
echo.
call :run_test "Write a Python function to calculate factorial" 100

echo.
echo [Test 5] Temperature testing (more creative)
echo Query: "What is the meaning of life?" with temperature 0.8
echo Expected: More diverse, creative response
echo Command: ziggurat-enhanced.exe --prompt "What is the meaning of life?" --temperature 0.8 --sample-len 80 --cpu
echo.
call :run_test_temp "What is the meaning of life?" 0.8 80

echo.
echo === Testing Complete ===
echo All tests executed. Check the responses above for:
echo - Coherence and relevance
echo - Correctness of factual information
echo - Code syntax and logic
echo - Creativity and diversity
echo.
pause
goto :eof

:run_test
echo Running: ziggurat-enhanced.exe --prompt "%~1" --sample-len %2 --cpu
echo Response:
timeout /t 120 /nobreak >nul
.\target\release\ziggurat-enhanced.exe --prompt "%~1" --sample-len %2 --cpu 2>&1 | findstr /v "avx\|neon\|simd128\|f16c\|temp\|repeat\|retrieved\|loaded"
echo.
goto :eof

:run_test_temp
echo Running: ziggurat-enhanced.exe --prompt "%~1" --temperature %2 --sample-len %3 --cpu
echo Response:
timeout /t 120 /nobreak >nul
.\target\release\ziggurat-enhanced.exe --prompt "%~1" --temperature %2 --sample-len %3 --cpu 2>&1 | findstr /v "avx\|neon\|simd128\|f16c\|temp\|repeat\|retrieved\|loaded"
echo.
goto :eof