module CuTensorOperations

export @cutensor

using CuArrays
using CuArrays: CuArray
using CuArrays.CUTENSOR: handle, CuDefaultStream, CuTensorDescriptor, cudaDataType,
        CUTENSOR_OP_IDENTITY, CUTENSOR_OP_CONJ, CUTENSOR_OP_ADD,
        CUTENSOR_ALGO_DEFAULT,  CUTENSOR_WORKSPACE_RECOMMENDED,
        cutensorElementwiseBinary, cutensorReduction, cutensorReductionGetWorkspace,
        cutensorContraction, cutensorContractionGetWorkspace

using CuArrays.CUBLAS: CublasFloat, CublasReal
using TensorOperations
using TensorOperations: IndexTuple, isperm # for implementation
using TensorOperations: _flatten, tensorify, expandconj, processcontractorder,
                        extracttensors! # for macro

using TensorOperations: add!, trace!, contract!,
        similar_from_indices, cached_similar_from_indices, IndexError

include("cuarrays.jl")
include("cutensormacro.jl")

TensorOperations.memsize(a::CuArray) = sizeof(a)

end # module
