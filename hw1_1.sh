#!/bin/bash
wget -O acc8904.pth https://www.dropbox.com/s/ddz63c6s7n3zem3/acc8904.pth?dl=0
python3 p1_test.py --test_directory=$1 --output_file=$2
