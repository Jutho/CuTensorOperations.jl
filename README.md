# CuTensorOperations.jl

CuTensorOperations.jl is a Julia interface to NVidia's [cuTENSOR](https://developer.nvidia.com/cuTensor) library that enables TensorOperations.jl to work on `CuArray` objects. The low-level methods from cuTENSOR are wrapped by CuArrays.jl on the experimental branch `ksh/tensor`. CuTensorOperations.jl provides a high level interface, but does not (yet) provide any fallback definitions for those cases where the cuTENSOR library does not work (e.g. certain combinations of eltypes, certain trace operations, ...).

By `using CuTensorOperations`, definitions are provided for a set of methods from `TensorOperations.jl` for `CuArray` objects to make `@tensor` and friends work with `CuArray` objects. This is type piracy: defining methods from another package for objects from another package. In time, CuTensorOperations.jl should become integrated into TensorOperations.jl and loaded when it is loaded together with CuArrays.jl.

Furthermore, `CuTensorOperations` provides a `@cutensor` macro, that acts like `@tensor` but will first transfer objects to the GPU by calling `CuArray` on them (the precise transformation call might change in the future). This is a no-op on existing `CuArray` objects that already live on the GPU, but otherwise (for arrays in the main memory) transfers them prior to doing the computation, so that NVidia's cuTENSOR library can be used for the computation. In a complicated expression with many basic operations, the transfer to GPU is only performed just before that object is required, so that some of the computation and transfer time can hopefully coincide. Newly created objects will reside on the GPU; if instead the left hand side is an existing array in the main memory, the final output will be transferred back into it.

## Installation instructions

Get a working copy of Julia (v1.1 or later). Make sure CUDA is available at standard location.

Download [cuTENSOR](https://developer.nvidia.com/cuTensor) from the NVidia Developer Program and test your installation. Make sure `libcutensor.so` is available at a location that is found by the system, e.g. `/usr/local/lib` on a MacOS X or Linux system.

Launch the Julia REPL, enter package mode by typing `]` and install necessary packages
```
pgk> add CuArrays#ksh/tensor
pgk> add TensorOperations
pkg> add https://github.com/Jutho/CuTensorOperations.jl.git
```
