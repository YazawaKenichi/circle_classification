#!/bin/bash

./circle_classification.py  0.0  0.0 --epochs 100 --alpha 0.0005 -n 1000 -r 1.0
./circle_classification.py -1.0  0.0 --epochs 100 --alpha 0.0005 -n 1000 -r 1.0
./circle_classification.py  0.6 -0.6 --epochs 100 --alpha 0.0005 -n 1000 -r 1.0
./circle_classification.py -1.2 -1.2 --epochs 100 --alpha 0.0005 -n 1000 -r 1.0

