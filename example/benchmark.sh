#!/bin/bash

n=($( echo "1 10 100 1000 10000" ))
for nj in ${n[@]}
do

    ../bin/cuda-checker -r 4K -C 3 -N ${nj}
    ../bin/cuda-checker -r 4K -C 3 -N ${nj} -c on       -B 2560 -G 10384
    ../bin/cuda-checker -r 4K -C 3 -N ${nj} -c on -p on -B 2560 -G 10384
    ../bin/cuda-checker -r 4K -C 3 -N ${nj} -c on -u on -B 2560 -G 10384

done
