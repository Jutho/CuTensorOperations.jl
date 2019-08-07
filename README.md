# CuTensorOperations.jl

## Installation instructions

Get a working copy of Julia (v1.1 or later). Make sure CUDA is available at standard location.

Download [cuTENSOR](https://developer.nvidia.com/cuTensor) from the NVidia Developer Program; make sure `libcutensor.so` is available at a location that is found by the system, e.g. `/usr/local/lib` on a MacOS X or Linux system.

Launch the Julia REPL, enter package mode by typing `]` and install necessary packages
```
pgk> add CuArrays#ksh/tensor
pgk> add TensorOperations
pkg> add https://github.com/Jutho/CuTensorOperations.jl.git
```

<!-- Optional: By default, Julia's linear algebra is running on OpenBLAS. You can easily install MKL, but this slows down the boot time of julia, i.e. it will start up a bit slower and feel a bit less snappy during the first few instructions. This can then be fixed, but I will not explain this here. This will likely be resolved in an upcoming Julia version, but that's why currently the package to install MKL is also not official yet. To install, go back to package mode and do
```
pkg> add https://github.com/JuliaComputing/MKL.jl.git
``` -->
