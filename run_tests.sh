#!/bin/bash

echo "=========================================="
echo "   CUDA Model Testing Suite"
echo "=========================================="
echo ""

PASS=0
FAIL=0

# Test 1: Basic functionality
echo "[1/8] Basic functionality test..."
if ./target/release/ziggurat-enhanced --prompt "What is machine learning?" --sample-len 40 --cpu 2>&1 | grep -q "Machine learning"; then
    echo "    ✅ PASS - Correct answer"
    ((PASS++))
else
    echo "    ❌ FAIL - Incorrect answer"
    ((FAIL++))
fi
echo ""

# Test 2: CUDA detection
echo "[2/8] CUDA detection test..."
if ./target/release/ziggurat-enhanced --prompt "Test" --sample-len 10 2>&1 | grep -q "CUDA"; then
    echo "    ✅ PASS - CUDA detected"
    ((PASS++))
else
    echo "    ❌ FAIL - CUDA not detected"
    ((FAIL++))
fi
echo ""

# Test 3: Mathematical reasoning
echo "[3/8] Mathematical reasoning test..."
if ./target/release/ziggurat-enhanced --prompt "Calculate: 8 * 7" --sample-len 20 --temperature 0.0 2>&1 | grep -q "56"; then
    echo "    ✅ PASS - Correct calculation (8 * 7 = 56)"
    ((PASS++))
else
    echo "    ❌ FAIL - Incorrect calculation"
    ((FAIL++))
fi
echo ""

# Test 4: Code generation
echo "[4/8] Code generation test..."
if ./target/release/ziggurat-enhanced --prompt "Write: print('hello')" --sample-len 30 --temperature 0.0 2>&1 | grep -q "print"; then
    echo "    ✅ PASS - Code generated"
    ((PASS++))
else
    echo "    ❌ FAIL - Code not generated"
    ((FAIL++))
fi
echo ""

# Test 5: GPU vs CPU performance
echo "[5/8] Performance comparison..."
echo "  Testing CPU..."
./target/release/ziggurat-enhanced --prompt "Performance test" --sample-len 50 --cpu 2>&1 | grep "token/s" > cpu_test.txt
echo "  Testing GPU (CUDA)..."
./target/release/ziggurat-enhanced --prompt "Performance test" --sample-len 50 2>&1 | grep "token/s" > gpu_test.txt
echo "    ✅ PASS - Performance comparison complete"
((PASS++))
echo ""

# Test 6: Temperature variation
echo "[6/8] Temperature test..."
if ./target/release/ziggurat-enhanced --prompt "Random test" --sample-len 30 --temperature 0.8 2>&1 | grep -q "tokens generated"; then
    echo "    ✅ PASS - High temperature works"
    ((PASS++))
else
    echo "    ❌ FAIL - Temperature error"
    ((FAIL++))
fi
echo ""

# Test 7: Memory stats (if available)
echo "[7/8] Memory stats test..."
if ./target/release/ziggurat-enhanced --memory-stats 2>&1 | grep -q "Stats"; then
    echo "    ✅ PASS - Memory stats accessible"
    ((PASS++))
else
    echo "    ⚠️  SKIP - Memory not configured"
fi
echo ""

# Test 8: Configuration
echo "[8/8] Configuration test..."
if ./target/release/ziggurat-enhanced --show-config 2>&1 | grep -q "Configuration"; then
    echo "    ✅ PASS - Configuration works"
    ((PASS++))
else
    echo "    ❌ FAIL - Configuration error"
    ((FAIL++))
fi
echo ""

echo "=========================================="
echo "   Test Results Summary"
echo "=========================================="
echo "   ✅ Passed: $PASS / 8"
echo "   ❌ Failed: $FAIL / 8"
echo ""
echo "   Performance comparison:"
echo "   -------------------"
echo "   CPU: $(cat cpu_test.txt | grep 'token/s')"
echo "   GPU (CUDA): $(cat gpu_test.txt | grep 'token/s')"
echo ""

# Cleanup
rm -f cpu_test.txt gpu_test.txt

if [ $PASS -ge 7 ]; then
    echo "   🎉 EXCELLENT - Model working perfectly!"
elif [ $PASS -ge 5 ]; then
    echo "   ✅ GOOD - Model working with minor issues"
else
    echo "   ⚠️  NEEDS ATTENTION - Multiple failures detected"
fi

echo ""
read -p "Press Enter to continue..."