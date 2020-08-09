# _Outline_
This tool is for testing of image processing, assuming a situation of system that an image frame will be transfered from sensor to server.
The inner processes works, in order, allocation of host and device memory ( only a shared memory when `-u` option inputted ), frame acquisition, data transfer from host to device ( except `-u` option has inputted ), image process on the device memory ( shared memory when `-u` option inputted ), data transfer from device to host ( except `-u` option has inputted ) and free host and device memory ( only the shared memory when `-u` option inputted ). The loop processing works from the frame acquisition to the data transfer ( from device to host ).

# _Build_
```
git clone git@github.com:cetusk/cuda-checker.git
cd cuda-checker
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=Release ..
make
```

# _Usage_
First you have to change `bin` directory, and can try it.
```
    # CPU
    ./cuda-checker -v -r 4K -C 3 -N 1000
    # GPU
    ./cuda-checker -v -r 4K -C 3 -N 1000 -c on       -B 1024
    # GPU ( page-lock )
    ./cuda-checker -v -r 4K -C 3 -N 1000 -c on -p on -B 1024
    # GPU ( unified memory )
    ./cuda-checker -v -r 4K -C 3 -N 1000 -c on -u on -B 1024
```

# _Options_
```
    -h | --help           : Help options.
    -v | --verbose        : Debug logs.
    -c | --cuda           : (on/off) Use CUDA computing.
    -p | --page-lock      : (on/off) Use page-locked memory.
    -u | --unified-memory : (on/off) Use unified memory system.
    -r | --resolution     : (QQVGA/QVGA/VGA/XGA/HD/FHD/2K/3K/4K/5K/6K/8K/10K/16K)
    -N | --num-loops      : (int value) Number of image processing loops.
    -W | --width          : (int value) Image width.
    -H | --height         : (int value) Image height.
    -C | --channels       : (int value) Number of image color channels.
    -B | --blocks         : (int value) Number of GPU blocks.
    -G | --grids          : (int value) Number of GPU grids.
```

# _Note_
- You can input multiple dimensions of GPU block/grid as `-B x y z` or `-G x y`, but only x dimension is supported ( only x dimension is used in this programs, e.g. `blockDim.x`, `blockIdx.x` and `threadIdx.x`. ).
- You can input explicit value of GPU block and grid. In usual, you should better for specifying only either one because the block/grid value is automatically derived based on a width, height, channels and grid/block.
- The priority of `-u/--unified-memory` is higher than `-p/--page-lock`, so if you input `-u` then `-p` will be disabled.

# _Sample benchmark_
![Sample benchmark](example/time.png "sample")