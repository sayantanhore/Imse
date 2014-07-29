#!/bin/bash

for i in {1..3}; do
    for file in $(find "gpu_test_$i" -iname "*variance.npy" | sort -n -t "/" -k 2); do
       ./compare.py "$file" "${file/gpu_test/cpu_test}";
       file="${file/variance/mean}"
       ./compare.py "$file" "${file/gpu_test/cpu_test}";
    done;
done;
