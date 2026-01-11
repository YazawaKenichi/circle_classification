#!/bin/bash

./circle_classification.py  0.0   0.0  --epochs 100 --alpha 7.5 -n 1000 -r 1.0
./circle_classification.py  0.0   1.0  --epochs 100 --alpha 7.5 -n 1000 -r 1.0
./circle_classification.py -0.5   0.3  --epochs 100 --alpha 7.5 -n 1000 -r 1.0
./circle_classification.py  0.75 -0.75 --epochs 100 --alpha 7.5 -n 1000 -r 1.0
./circle_classification.py -1.75  1.25 --epochs 100 --alpha 7.5 -n 1000 -r 1.0

