# CuTensorOperations.jl

## Installation instructions

Get a working copy of Julia (v1.1). Make sure CUDA is available at standard location.

Launch the Julia REPL, enter package mode by typing `]` and install necessary packages
```
pgk> add CuArrays # this can take a while on a clean Julia installation
pgk> add TensorOperations
pkg> add BenchmarkTools
pkg> dev https://github.com/Jutho/CuTensor.jl.git
pkg> dev https://github.com/Jutho/CuTensorOperations.jl.git
```
The install/build process of CuTensor.jl will probably fail, unless you have "libcutensor.so" in your home directory, or in "/usr/local/cutensor/lib". If it fails, add "libcutensor.so" in one of these directories, or in "~/.julia/dev/CuTensor/bin". The `bin` directory will need to be created first. Then do
```
pkg> build CuTensor
```
Type backspace to return to the normal `julia>` REPL mode.

Optional: By default, Julia's linear algebra is running on OpenBLAS. You can easily install MKL, but this slows down the boot time of julia, i.e. it will start up a bit slower and feel a bit less snappy during the first few instructions. This can then be fixed, but I will not explain this here. This will likely be resolved in an upcoming Julia version, but that's why currently the package to install MKL is also not official yet. To install, go back to package mode and do
```
pkg> add https://github.com/JuliaComputing/MKL.jl.git
```

In Julia REPL mode, do
```
julia> using CuTensorOperations
# the first time, this will take a long time, because it also precompiles all the packages
# that CuTensorOperations depends on, mainly the compilation time of CuTensorOperations
# takes quite a bit of time
julia> include(joinpath(dirname(pathof(CuTensorOperations)),"../benchmark/bench.jl"))
```

This will bring a number of functions into the namespace, namely
* `mpscheck`, `peps1check`, `peps2check`
* `mpsbench`, `peps1bench`, `peps2bench`

These functions have one default positional argument, the element type, with default value `Float64`, and a number of keyword arguments to specify to different dimensions in the tensors. You can get a list of keyword arguments and suggestions for possible values by using the help system. Enter help mode by typing `?` at an empty `julia>` REPL

The `...check` functions evaluate a tensor contraction on the CPU, and also on the GPU, and compares that the result is the same (up to rounding). The tensor contrations are representative for typical MPS and PEPS algorithms, and evaluate to a scalar (no free modes). In tensor network algorithms, we also want to evaluate the gradient of that scalar with respect to any of the tensors, which amounts to omitting that tensor from the network and leaving the modes with which it was contracted open/free. They are often called environments in the tensor network community. However, if you can evaluate the closed tensor network (no free modes), you should be able to compute any of the environments.

As the check functions evaluate the tensor contraction only once, they are also useful for debugging purposes. For example, you can change the environment variables as follows
```
julia> ENV["CUTENSOR_DEBUG"] = "1"
```
to print out all the debug information of cuTensor during the different pairwise contractions.

The `...bench` functions use `@benchmark` from the BenchmarkTools.jl` package to run the contraction a large number of times and gather statistics. Here, three different benchmarks are ran. The first one evaluates the tensor network contraction on the CPU, the second one does the same with the tensors already living on the GPU. The third benchmark also includes the time to transfer the tensors to the GPU.

## Issues so far

* `peps2check` and `peps2bench` fail with CuTensor, not sure why. Error is:
   `ERROR: CUTENSORError(code CUTENSOR_STATUS_INTERNAL_ERROR, an internal operation failed)`
* All routines fail with `T <: Complex`, because the contractions involve complex conjugation. For `T <: Real`, this complex conjugation is ignored.
