# _Usage_
```
    # CPU
    ./imgtest -v -r 4K
    # GPU
    ./imgtest -v -c on -r 4K -B 1024
    # GPU ( unified memory )
    ./imgtest -v -c on -u on -r 4K -B 1024
```

# _Options_
```
    -h | --help           : Help options.
    -v | --verbose        : Debug logs.
    -c | --cuda           : (on/off) Use CUDA computing.
    -u | --unified-memory : (on/off) Use unified memory system.
    -r | --resolution     : (QQVGA/QVGA/VGA/XGA/HD/FHD/2K/3K/4K/5K/6K/8K/10K/16K)
    -W | --width          : (int value) Image width.
    -H | --height         : (int value) Image height.
    -C | --channels       : (int value) Number of image color channels.
    -B | --blocks         : (int value) Number of GPU blocks.
```