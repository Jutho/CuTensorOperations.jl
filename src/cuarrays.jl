# Default implmenentation for CuArrays
function TensorOperations.add!(α, A::CuArray{<:Any, N}, CA::Symbol,
        β, C::CuArray{<:Any, N}, indCinA) where {N}

    T = eltype(C)
    N == length(indCinA) || throw(IndexError("Invalid permutation of length $N: $indCinA"))
    CA == :N || CA == :C ||
        throw(ArgumentError("Value of conjA should be :N or :C instead of $CA"))
    opA = (T <: Real || CA == :N) ? CUTENSOR_OP_IDENTITY : CUTENSOR_OP_CONJ
    opAC = CUTENSOR_OP_ADD

    descA = CuTensorDescriptor(A; op = opA)
    descC = CuTensorDescriptor(C; op = CUTENSOR_OP_IDENTITY)
    descD = descC
    typeCompute = cudaDataType(T)
    modeA = collect(Cint, 1:N)
    modeC = collect(Cint, indCinA)
    stream = CuDefaultStream()
    cutensorElementwiseBinary(handle(), T[α], A, descA, modeA, T[β], C, descC,
                              modeC, C, descC, modeC, opAC, typeCompute, stream)
    return C
end
function TensorOperations.add!(α, A::CuArray{T, N}, CA::Symbol,
        β, C::CuArray{Complex{T}, N}, indCinA) where {T<:CublasReal,N}

    N == length(indCinA) || throw(IndexError("Invalid permutation of length $N: $indCinA"))
    CA == :N || CA == :C ||
        throw(ArgumentError("Value of conjA should be :N or :C instead of $CA"))
    opid = CUTENSOR_OP_IDENTITY
    opAC = CUTENSOR_OP_ADD

    if imag(β) != 0
        rmul!(C, β)
        γ = one(real(β))
    else
        γ = real(β)
    end
    sizeC = size(C)
    stridesC = 2 .* strides(C)
    Cr = reinterpret(T, C, (2*length(C),))
    @show sizeC, stridesC

    descA = CuTensorDescriptor(A; op = opid)
    descCr = CuTensorDescriptor(Cr; size = sizeC, strides = stridesC, op = opid)
    typeCompute = cudaDataType(T)
    modeA = collect(Cint, 1:N)
    modeC = collect(Cint, indCinA)
    stream = CuDefaultStream()
    cutensorElementwiseBinary(handle(), T[real(α)], A, descA, modeA, T[γ], Cr, descCr,
                              modeC, Cr, descCr, modeC, opAC, typeCompute, stream)
    if imag(α) != 0
        Ci = view(Cr, 2:length(Cr))
        descCi = CuTensorDescriptor(Ci; size = sizeC, strides = stridesC, op = opid)
        cutensorElementwiseBinary(handle(), T[imag(α)], A, descA, modeA, T[γ], Ci, descCi,
                                  modeC, Ci, descCi, modeC, opAC, typeCompute, stream)
    end
    return C
end

function TensorOperations.trace!(α, A::CuArray, CA::Symbol, β, C::CuArray,
                indCinA, cindA1, cindA2)

    T = eltype(C)
    NA, NC = ndims(A), ndims(C)
    NC == length(indCinA) ||
        throw(IndexError("Invalid selection of $NC out of $NA: $indCinA"))
    NA-NC == 2*length(cindA1) == 2*length(cindA2) ||
        throw(IndexError("invalid number of trace dimension"))

    opA = (T <: Real || CA == :N) ? CUTENSOR_OP_IDENTITY : CUTENSOR_OP_CONJ
    opReduce = CUTENSOR_OP_ADD

    sizeA = i->size(A, i)
    strideA = i->stride(A, i)
    tracesize = sizeA.(cindA1)
    tracesize == sizeA.(cindA2) || throw(DimensionMismatch("non-matching trace sizes"))
    size(C) == sizeA.(indCinA) || throw(DimensionMismatch("non-matching sizes"))

    newstrides = (strideA.(indCinA)..., (strideA.(cindA1) .+ strideA.(cindA2))...)
    newsize = (size(C)..., tracesize...)
    descA = CuTensorDescriptor(A; op = opA, size = newsize, strides = newstrides)
    descC = CuTensorDescriptor(C; op = CUTENSOR_OP_IDENTITY)
    descD = descC
    typeCompute = cudaDataType(T)
    modeA = collect(Cint, 1:NA)
    modeC = collect(Cint, indCinA)
    stream = CuDefaultStream()
    workspaceSize = Ref{UInt64}(C_NULL)
    cutensorReductionGetWorkspace(handle(), A, descA, modeA,
                                  C, descC, modeC, C, descC, modeC,
                                  opReduce, typeCompute, workspaceSize)
    workspace = CuArray{UInt8}(undef, 0)
    try
      workspace = CuArray{UInt8}(undef, workspaceSize[])
    catch
      workspace = CuArray{UInt8}(undef, 1<<13)
    end
    workspaceSize[] = length(workspace)

    cutensorReduction(handle(), T[α], A, descA, modeA, T[β], C, descC, modeC,
                      C, descC, modeC, opReduce, typeCompute, workspace,
                      workspaceSize[], stream)
    return C
end

function TensorOperations.contract!(
            α, A::CuArray, CA::Symbol,
            B::CuArray, CB::Symbol,
            β, C::CuArray,
            oindA::IndexTuple, cindA::IndexTuple,
            oindB::IndexTuple, cindB::IndexTuple,
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

    CA == :N || CA == :C ||
        throw(ArgumentError("Value of conjA should be :N or :C instead of $CA"))
    CB == :N || CB == :C ||
        throw(ArgumentError("Value of conjB should be :N or :C instead of $CB"))

    opA = (TC <: Real || CA == :N) ? CUTENSOR_OP_IDENTITY : CUTENSOR_OP_CONJ
    opB = (TC <: Real || CB == :N) ? CUTENSOR_OP_IDENTITY : CUTENSOR_OP_CONJ
    opC = CUTENSOR_OP_IDENTITY

    strideA = i->stride(A, i)
    strideB = i->stride(B, i)

    cstrideA = strideA.(cindA)
    cstrideB = strideB.(cindB)
    ostrideA = strideA.(oindA)
    ostrideB = strideB.(oindB)

    descA = CuTensorDescriptor(A; op = opA, size = (osizeA..., csizeA...),
                                    strides = (ostrideA..., cstrideA...))
    descB = CuTensorDescriptor(B; op = opB, size = (osizeB..., csizeB...),
                                    strides = (ostrideB..., cstrideB...))
    descC = CuTensorDescriptor(C)
    T = eltype(C)
    typeCompute = cudaDataType(T)
    opOut = CUTENSOR_OP_IDENTITY

    NoA = length(osizeA)
    NoB = length(osizeB)
    Nc = length(csizeA)
    modeoA = ntuple(n->n, NoA)
    modeoB = ntuple(n->NoA+n, NoB)
    modec = ntuple(n->NoA+NoB+n, Nc)

    modeA = collect(Cint, (modeoA..., modec...))
    modeB = collect(Cint, (modeoB..., modec...))
    modeC = collect(Cint, indCinoAB)

    algo = CUTENSOR_ALGO_DEFAULT
    stream = CuDefaultStream()
    pref = CUTENSOR_WORKSPACE_RECOMMENDED

    workspaceSize = Ref{UInt64}(C_NULL)
    cutensorContractionGetWorkspace(handle(), A, descA, modeA, B, descB, modeB, C, descC,
                                    modeC, C, descC, modeC, opOut, typeCompute, algo, pref,
                                    workspaceSize)
    workspace = CuArray{UInt8}(undef, 0)
    try
        workspace = CuArray{UInt8}(undef, workspaceSize[])
    catch
        workspace = CuArray{UInt8}(undef, 2^27)
    end
    workspaceSize[] = length(workspace)
    cutensorContraction(handle(), T[α], A, descA, modeA, B, descB, modeB,
                        T[β], C, descC, modeC, C, descC, modeC, opOut, typeCompute, algo,
                        workspace, workspaceSize[], stream)
    return C
end
