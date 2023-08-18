#!/bin/bash

#PBS -q normal
#PBS -l ncpus=40
#PBS -l mem=100gb
#PBS -l walltime=12:00:00
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

directory_path="/g/data/v45/sg7549/NetworkScore/output/"

score_nodes_options=("54,28,51" "72,67,69" "70,23,55" "5,60,36" "69,16,40" "29,41,44" "6,10,66" "66,69,44" "43,28,40" "52,19,49")
dir_names=('a' 'b' 'c' 'd' 'e' 'f' 'g' 'h' 'i' 'j')

for ((i=0; i<${#dir_names[@]}; ++i)); do
    mkdir -p "${directory_path}/${dir_names[i]}"
    seq 0 100 | parallel -j 40 python /g/data/v45/sg7549/NetworkScore/NetworkScore.py 1000 1000 100 100 --web 8 9 --score-nodes "${score_nodes_options[i]}" --output-no-subdirectory --output {}.csv

    mv "${directory_path}"/*.csv "${directory_path}/${dir_names[i]}"

done