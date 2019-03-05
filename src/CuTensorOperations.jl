module CuTensorOperations

import CuArrays
import CuArrays: CuArray
import CuTensor
import CuTensor: CUTENSOR_OP_IDENTITY, CUTENSOR_OP_CONJ, CUTENSOR_OP_ADD, CuTensorDescriptor, cutensorPointwiseBinary, cutensorContraction, handle, cublastype, CUTENSOR_ALGO_DEFAULT

import TensorOperations
import TensorOperations: add!, contract!, IndexTuple, @tensor, isperm

export @tensor, CuArray


function add!(α, A::CuArray{<:Any, N}, CA::Symbol,
        β, C::CuArray{<:Any, N}, indCinA) where {N}

    N == length(indCinA) || throw(IndexError("Invalid permutation of length $N: $indCinA"))
    CA == :N || CA == :C || throw(ArgumentError("Value of conjA should be :N or :C instead of $CA"))
    opA = CA == :N ? CUTENSOR_OP_IDENTITY : CUTENSOR_OP_CONJ
    opC = CUTENSOR_OP_IDENTITY
    opAC = CUTENSOR_OP_ADD

    descA = CuTensorDescriptor(A)
    descC = CuTensorDescriptor(C)
    T = eltype(C)

    modeA = collect(Cint, 1:N)
    modeC = collect(Cint, indCinA)

    cutensorPointwiseBinary(handle(), convert(T, α), A, descA, modeA,
        convert(T, β), C, descC, modeC, opA, opC, opAC, C, cublastype(T))

    return C
end

# function trace!(α, A::AbstractArray{<:Any, NA}, CA::Symbol, β, C::AbstractArray{<:Any, NC},
#         indCinA, cindA1, cindA2) where {NA,NC}
#
#     NC == length(indCinA) ||
#         throw(IndexError("Invalid selection of $NC out of $NA: $indCinA"))
#     NA-NC == 2*length(cindA1) == 2*length(cindA2) ||
#         throw(IndexError("invalid number of trace dimension"))
#     if CA == :N
#         if isbitstype(eltype(A)) && isbitstype(eltype(C))
#             @unsafe_strided A C _trace!(α, A, β, C,
#                 (indCinA...,), (cindA1...,), (cindA2...,))
#         else
#             _trace!(α, StridedView(A), β, StridedView(C),
#                 (indCinA...,), (cindA1...,), (cindA2...,))
#         end
#     elseif CA == :C
#         if isbitstype(eltype(A)) && isbitstype(eltype(C))
#             @unsafe_strided A C _trace!(α, conj(A), β, C,
#                 (indCinA...,), (cindA1...,), (cindA2...,))
#         else
#             _trace!(α, conj(StridedView(A)), β, StridedView(C),
#                 (indCinA...,), (cindA1...,), (cindA2...,))
#         end
#     elseif CA == :A
#         if isbitstype(eltype(A)) && isbitstype(eltype(C))
#             @unsafe_strided A C _trace!(α, map(adjoint, A), β, C,
#                 (indCinA...,), (cindA1...,), (cindA2...,))
#         else
#             _trace!(α, map(adjoint, StridedView(A)), β, StridedView(C),
#                 (indCinA...,), (cindA1...,), (cindA2...,))
#         end
#     else
#         throw(ArgumentError("Unknown conjugation flag: $CA"))
#     end
#     return C
# end

function contract!(α, A::CuArray, CA::Symbol, B::CuArray, CB::Symbol,
        β, C::CuArray,
        oindA::IndexTuple, cindA::IndexTuple, oindB::IndexTuple, cindB::IndexTuple,
        indCinoAB::IndexTuple, syms::Union{Nothing, NTuple{3,Symbol}} = nothing)

    pA = (oindA...,cindA...)
    (length(pA) == ndims(A) && isperm(pA)) ||
        throw(IndexError("invalid permutation of length $(ndims(A)): $pA"))
    pB = (oindB...,cindB...)
    (length(pB) == ndims(B) && isperm(pB)) ||
        throw(IndexError("invalid permutation of length $(ndims(B)): $pB"))
    (length(oindA) + length(oindB) == ndims(C)) ||
        throw(IndexError("non-matching output indices in contraction"))
    (ndims(C) == length(indCinoAB) && isperm(indCinoAB)) ||
        throw(IndexError("invalid permutation of length $(ndims(C)): $indCinoAB"))

    sizeA = i->size(A, i)
    sizeB = i->size(B, i)
    sizeC = i->size(C, i)

    csizeA = sizeA.(cindA)
    csizeB = sizeB.(cindB)
    osizeA = sizeA.(oindA)
    osizeB = sizeB.(oindB)

    csizeA == csizeB ||
        throw(DimensionMismatch("non-matching sizes in contracted dimensions"))
    sizeAB = let osize = (osizeA..., osizeB...)
        i->osize[i]
    end
    sizeAB.(indCinoAB) == size(C) ||
        throw(DimensionMismatch("non-matching sizes in uncontracted dimensions"))

    TC = eltype(C)

    CA == :N || CA == :C || throw(ArgumentError("Value of conjA should be :N or :C instead of $CA"))
    CB == :N || CB == :C || throw(ArgumentError("Value of conjB should be :N or :C instead of $CB"))

    opA = (TC <: Real || CA == :N) ? CUTENSOR_OP_IDENTITY : CUTENSOR_OP_CONJ
    opB = (TC <: Real || CB == :N) ? CUTENSOR_OP_IDENTITY : CUTENSOR_OP_CONJ
    opC = CUTENSOR_OP_IDENTITY

    strideA = i->stride(A, i)
    strideB = i->stride(B, i)

    cstrideA = strideA.(cindA)
    cstrideB = strideB.(cindB)
    ostrideA = strideA.(oindA)
    ostrideB = strideB.(oindB)

    descA = CuTensorDescriptor(A; size = (osizeA..., csizeA...),
                            strides = (ostrideA..., cstrideA...))
    descB = CuTensorDescriptor(B; size = (osizeB..., csizeB...),
                            strides = (ostrideB..., cstrideB...))
    descC = CuTensorDescriptor(C)
    T = eltype(C)

    NoA = length(osizeA)
    NoB = length(osizeB)
    Nc = length(csizeA)
    modeoA = ntuple(n->n, NoA)
    modeoB = ntuple(n->NoA+n, NoB)
    modec = ntuple(n->NoA+NoB+n, Nc)

    modeA = collect(Cint, (modeoA...,modec...))
    modeB = collect(Cint, (modeoB...,modec...))
    modeC = collect(Cint, indCinoAB)

    cutensorContraction(handle(), convert(T, α), A, opA, descA, modeA, B, opB, descB, modeB,
        convert(T, β), C, opC, descC, modeC, cublastype(T), CUTENSOR_ALGO_DEFAULT)

    return C
end


end # module
