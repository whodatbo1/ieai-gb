#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python "src/main.py" \
  --log-level "debug" \
  --experiments "all" \
  --datasets "names" "professions" \
  --seed 1001 \
  --num-of-entries 150
