#!/bin/bash
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

pip3 install ray
python3 ./generate_dataset.py