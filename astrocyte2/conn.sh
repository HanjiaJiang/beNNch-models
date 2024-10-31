#!bin/bash

scale=$1

python conn.py bernoulli10_$scale
python conn.py bernoulli100_$scale
python conn.py bernoulli1000_$scale
python conn.py bernoulli10000_$scale

