#!/bin/bash

./circle_classification.py  0.0  0.00 --epochs 100 --alpha 0.0010 -n 1000 -r 1.0 --seed 1768110697
# ./circle_classification.py  0.0  0.00 --epochs 500 --alpha 0.0010 -n 1000 -r 1.0 --seed 1768110697
./circle_classification.py  0.6  0.75 --epochs 500 --alpha 0.0010 -n 1000 -r 1.0 --seed 1768110697
# ./circle_classification.py  0.6  0.75 --epochs 500 --alpha 0.0100 -n 1000 -r 1.0 --seed 1768110697
./circle_classification.py  0.6  0.75 --epochs 500 --alpha 0.0100 -n 10000 -r 1.0 --seed 1768110697

# ./circle_classification.py  0.6  0.75 --epochs 500 --alpha 0.0005 -n 1000 -r 1.0 --seed 1768110697
# ./circle_classification.py  0.0  0.0 --epochs 100 --alpha 0.0005 -n 1000 -r 1.0 --seed 1768110697
# ./circle_classification.py  0.0  0.0 --epochs 100 --alpha 0.0005 -n 1000 -r 1.0
# ./circle_classification.py -1.0  0.0 --epochs 100 --alpha 0.0005 -n 1000 -r 1.0
# ./circle_classification.py  0.6 -0.6 --epochs 100 --alpha 0.0005 -n 1000 -r 1.0
# ./circle_classification.py -1.2 -1.2 --epochs 100 --alpha 0.0005 -n 1000 -r 1.0

