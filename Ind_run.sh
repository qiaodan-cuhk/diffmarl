#!/bin/sh

seeds=( 1 27 42 100 1000 )
datatypes=( "expert" "medium" "medium-replay" "random" )
datanums=( 2 3 )


for i in "${seeds[@]}"
do
    for types in "${datatypes[@]}"
    do
        for num in "${datanums[@]}"
        do
            echo "seed: $i, datatypes: $types, data number: $num"
            python main.py --data_type $types --dataset_num $num --seed $i --device 0
        done
    done
done