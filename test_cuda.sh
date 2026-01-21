#!/bin/bash
# Automated test script for CUDA model testing

cd "C:\Users\zikkuratti\Documents\agent\zikkurat-mind"

echo "=== ZIGGURAT MIND CUDA Test Suite ==="
echo ""

echo "1. GPU Info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

echo "2. Building with CUDA..."
cargo build --features cuda 2>&1 | grep -E "(Finished|error)"
echo ""

echo "3. Testing simple math (2+2)..."
start_time=$(date +%s)
output=$(timeout 120 target/debug/ziggurat-unified.exe --prompt "What is 2+2?" 2>&1)
echo "$output" | grep -A2 "Assistant:"
echo ""

echo "4. Testing code generation..."
timeout 120 target/debug/ziggurat-unified.exe --prompt "Write a Python function to sort a list" 2>&1 | grep -A15 "```"
echo ""

echo "5. Testing Russian language..."
timeout 120 target/debug/ziggurat-unified.exe --prompt "Что такое искусственный интеллект?" 2>&1 | grep -A10 "Assistant:"
echo ""

echo "=== Test Complete ==="
