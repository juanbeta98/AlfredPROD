#!/bin/bash

# First set of instances
for inst in AD3 AD4; do
#  for dist in precalced haversine; do
  for dist in haversine precalced; do
    echo "▶ Running $inst with $dist..."
    python run_dynamic_instance.py --inst "$inst" --dist_method "$dist" --save True
  done
done


