using CuTensorOperations
using CuArrays
using CuTensor
CuTensor.workspace(2^30)
using BenchmarkTools

const CA = CuArray

"""
    mpscheck(T = Float64; D = 256, d = 2, m = 8)

Check whether MPS contraction produces same result on CPU and GPU/CUDA; specify
*   D: MPS bond dimension (range 10 - 1000)
*   d: MPS physical dimension (range 2 - 5)
*   m: MPO bond dimension (range 2 - 20, occasionally larger)
"""
function mpscheck(T = Float64; D = 256, d = 2, m = 8)
    A = randn(T, (D, d, D))
    FL = randn(T, (D, m, D))
    FR = randn(T, (D, m, D))
    O = randn(T, (m, d, m, d))

    cA = CuArray(A)
    cFL = CuArray(FL)
    cFR = CuArray(FR)
    cO = CuArray(O)

    @tensor x1 =
        ((((FL[α',l,α]*A[α,u,β])*O[l,d,r,u])*conj(A[α',d,β']))*FR[β,r,β'])

    @tensor x2 =
        ((((cFL[α',l,α]*cA[α,u,β])*cO[l,d,r,u])*conj(cA[α',d,β']))*cFR[β,r,β'])

    return x1 ≈ x2
end


"""
    mpsbench(T = Float64; D = 256, d = 2, m = 8)

Benchmark MPS contraction, on CPU, GPU and GPU+transfer time; specify:
*   D: MPS bond dimension (range 10 - 1000)
*   d: MPS physical dimension (range 2 - 5)
*   m: MPO bond dimension (range 2 - 20, occasionally larger)
"""
function mpsbench(T = Float64; D = 256, d = 2, m = 8)
    A = randn(T, (D, d, D))
    FL = randn(T, (D, m, D))
    FR = randn(T, (D, m, D))
    O = randn(T, (m, d, m, d))

    cA = CuArray(A)
    cFL = CuArray(FL)
    cFR = CuArray(FR)
    cO = CuArray(O)

    b1 = @benchmark @tensor x1 =
        (((($FL[α',l,α]*$A[α,u,β])*$O[l,d,r,u])*conj($A[α',d,β']))*$FR[β,r,β'])

    b2 = @benchmark CuArrays.@sync @tensor x2 =
        (((($cFL[α',l,α]*$cA[α,u,β])*$cO[l,d,r,u])*conj($cA[α',d,β']))*$cFR[β,r,β'])

    b3 = @benchmark @tensor x3 =
        ((((CA($FL)[α',l,α]*CA($A)[α,u,β])*CA($O)[l,d,r,u])*
            conj(CA($A)[α',d,β']))*CA($FR)[β,r,β'])

    return b1, b2, b3
end

"""
    peps1check(T = Float64; D = 4, d = 2, χ = 256)

Check whether PEPS contraction type 1 produces same result on CPU and GPU/CUDA; specify
*   D: PEPS bond dimension (range 2 - 6, preferably larger)
*   d: PEPS physical dimension (range 2 - 5)
*   χ: boundary MPS bond dimension (range 10 - 1000, the larger the better)
"""
function peps1check(T = Float64; D = 4, d = 2, χ = 256)
    A = randn(T, (D, D, D, D, d))
    FL = randn(T, (χ, D, D, χ))
    FR = randn(T, (χ, D, D, χ))
    FU = randn(T, (χ, D, D, χ))
    FD = randn(T, (χ, D, D, χ))

    cA = CuArray(A)
    cFL = CuArray(FL)
    cFR = CuArray(FR)
    cFU = CuArray(FU)
    cFD = CuArray(FD)

    @tensor x1 =
        (FL[ld,l,l',lu]*FU[lu,u,u',ru]*A[l,d,r,u,s]*conj(A[l',d',r',u',s])*FR[rd,r,r',ru]*FD[ld,d,d',rd])

    @tensor x2 =
        (cFL[ld,l,l',lu]*cFU[lu,u,u',ru]*cA[l,d,r,u,s]*conj(cA[l',d',r',u',s])*cFR[rd,r,r',ru]*cFD[ld,d,d',rd])

    return x1 ≈ x2
end

"""
    peps1bench(T = Float64; D = 4, d = 2, χ = 256)

Benchmark PEPS contraction type 1, on CPU, GPU and GPU+transfer time; specify:
*   D: PEPS bond dimension (range 2 - 6, preferably larger)
*   d: PEPS physical dimension (range 2 - 5)
*   χ: boundary MPS bond dimension (range 10 - 1000, the larger the better)
"""
function peps1bench(T = Float64; D = 4, d = 2, χ = 256)
    A = randn(T, (D, D, D, D, d))
    FL = randn(T, (χ, D, D, χ))
    FR = randn(T, (χ, D, D, χ))
    FU = randn(T, (χ, D, D, χ))
    FD = randn(T, (χ, D, D, χ))

    cA = CuArray(A)
    cFL = CuArray(FL)
    cFR = CuArray(FR)
    cFU = CuArray(FU)
    cFD = CuArray(FD)

    b1 = @benchmark @tensor x1 =
        ($FL[ld,l,l',lu]*$FU[lu,u,u',ru]*$A[l,d,r,u,s]*conj($A[l',d',r',u',s])*$FR[rd,r,r',ru]*$FD[ld,d,d',rd])

    b2 = @benchmark CuArrays.@sync @tensor x2 =
        ($cFL[ld,l,l',lu]*$cFU[lu,u,u',ru]*$cA[l,d,r,u,s]*conj($cA[l',d',r',u',s])*$cFR[rd,r,r',ru]*$cFD[ld,d,d',rd])

    b3 = @benchmark CuArrays.@sync @tensor x3 =
        (CA($FL)[ld,l,l',lu]*CA($FU)[lu,u,u',ru]*CA($A)[l,d,r,u,s]*conj(CA($A)[l',d',r',u',s])*CA($FR)[rd,r,r',ru]*CA($FD)[ld,d,d',rd])

    return b1, b2, b3
end

"""
    peps2check(T = Float64; D = 4, d = 2, m = 2, χ = 256)

Check whether PEPS contraction type 2 produces same result on CPU and GPU/CUDA; specify
*   D: PEPS bond dimension (range 2 - 6, preferably larger)
*   d: PEPS physical dimension (range 2 - 5)
*   m: PEPO bond dimension (range 2 - 8??)
*   χ: boundary MPS bond dimension (range 10 - 1000, the larger the better)
"""
function peps2check(T = Float64; D = 4, d = 2, m = 2, χ = 128)
    A = randn(T, (D, D, D, D, d))
    FL = randn(T, (χ, D, m, D, χ))
    FR = randn(T, (χ, D, m, D, χ))
    FU = randn(T, (χ, D, m, D, χ))
    FD = randn(T, (χ, D, m, D, χ))
    O = randn(T, (m, m, m, m, d, d))

    cA = CuArray(A)
    cFL = CuArray(FL)
    cFR = CuArray(FR)
    cFU = CuArray(FU)
    cFD = CuArray(FD)
    cO = CuArray(O)

    @tensor x1 =
        (FL[ld,l,l',l'',lu]*FU[lu,u,u',u'',ru]*A[l,d,r,u,s]*O[l',d',r',u',s',s]*
        conj(A[l'',d'',r'',u'',s'])*FR[rd,r,r',r'',ru]*FD[ld,d,d',d'',rd])

    @tensor x2 =
        (cFL[ld,l,l',l'',lu]*cFU[lu,u,u',u'',ru]*cA[l,d,r,u,s]*cO[l',d',r',u',s',s]*
        conj(A[l'',d'',r'',u'',s'])*cFR[rd,r,r',r'',ru]*cFD[ld,d,d',d'',rd])

    return b1, b2, b3
end


"""
    peps2bench(T = Float64; D = 4, d = 2, m = 2, χ = 256)

Benchmark PEPS contraction type 2, on CPU, GPU and GPU+transfer time; specify:
*   D: PEPS bond dimension (range 2 - 6, preferably larger)
*   d: PEPS physical dimension (range 2 - 5)
*   m: PEPO bond dimension
*   χ: boundary MPS bond dimension (range 10 - 1000, the larger the better)
"""
function peps2bench(T = Float64; D = 4, d = 2, m = 2, χ = 128)
    A = randn(T, (D, D, D, D, d))
    FL = randn(T, (χ, D, m, D, χ))
    FR = randn(T, (χ, D, m, D, χ))
    FU = randn(T, (χ, D, m, D, χ))
    FD = randn(T, (χ, D, m, D, χ))
    O = randn(T, (m, m, m, m, d, d))

    cA = CuArray(A)
    cFL = CuArray(FL)
    cFR = CuArray(FR)
    cFU = CuArray(FU)
    cFD = CuArray(FD)
    cO = CuArray(O)

    b1 = @benchmark @tensor x1 =
        ($FL[ld,l,l',l'',lu]*$FU[lu,u,u',u'',ru]*$A[l,d,r,u,s]*$O[l',d',r',u',s',s]*
        conj($A[l'',d'',r'',u'',s'])*$FR[rd,r,r',r'',ru]*$FD[ld,d,d',d'',rd])

    b2 = @benchmark CuArrays.@sync @tensor x2 =
        ($cFL[ld,l,l',l'',lu]*$cFU[lu,u,u',u'',ru]*$cA[l,d,r,u,s]*$cO[l',d',r',u',s',s]*
        conj($A[l'',d'',r'',u'',s'])*$cFR[rd,r,r',r'',ru]*$cFD[ld,d,d',d'',rd])

    b3 = @benchmark CuArrays.@sync @tensor x3 =
        (CA($FL)[ld,l,l',l'',lu]*CA($FU)[lu,u,u',u'',ru]*CA($A)[l,d,r,u,s]*
        CA($O)[l',d',r',u',s',s]*conj(CA($A)[l'',d'',r'',u'',s'])*
        CA($FR)[rd,r,r',r'',ru]*CA($FD)[ld,d,d',d'',rd])

    return b1, b2, b3
end
