#!/bin/bash

wget -O fcn8s_073_model.pth https://www.dropbox.com/s/jn4v5s7gwtmk1v6/fcn8s_073_model.pth?dl=0
python3 p2_Test.py --test_directory=$1 --output_file=$2
