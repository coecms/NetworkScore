#!/bin/bash

#PBS -q normal
#PBS -l ncpus=20
#PBS -l mem=100gb
#PBS -l walltime=01:00:00
#PBS -l jobfs=100gb
#PBS -l wd
#PBS -l storage=gdata/hh5+gdata/v45+gdata/w40
#PBS -W umask=0022
#PBS -j oe
#PBS -P v45

module load parallel
module use /g/data/hh5/public/modules
module load conda/analysis3-unstable

source activate /g/data/v45/sg7549/NetworkScore/netscore

seq 0 100 | parallel -j 20 python /g/data/v45/sg7549/NetworkScore/NetworkScore.py 1000 1000 100 100 --web 8 9 --score-nodes "54,28,51" --output-no-subdirectory --output {}