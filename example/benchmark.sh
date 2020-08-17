#!/bin/bash

n=($( echo "1 10 100 1000 10000" ))
for nj in ${n[@]}
do

    ../bin/cuda-checker -r 4K -N ${nj}
    ../bin/cuda-checker -r 4K -N ${nj} -c    -B 2560
    ../bin/cuda-checker -r 4K -N ${nj} -c    -B  640 4
    ../bin/cuda-checker -r 4K -N ${nj} -c -p -B 2560
    ../bin/cuda-checker -r 4K -N ${nj} -c -p -B  640 4
    ../bin/cuda-checker -r 4K -N ${nj} -c -u -B 2560

done
