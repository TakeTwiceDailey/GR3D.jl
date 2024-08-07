module GR3D

using ParallelStencil
using CellArrays, StaticArrays, CUDA

using HDF5
using FileIO

using Plots, Printf, Statistics, BenchmarkTools, ForwardDiff

using Tensorial, InteractiveUtils

using RootSolvers # (Roots.jl does not work on GPU)

using Profile

ParallelStencil.@reset_parallel_stencil()

const USE_GPU = false  # Use GPU? If this is set false, then no GPU needs to be available

@static if USE_GPU
    @define_CuCellArray() # without this CuCellArrays don't work
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
end

# Alias for 4 vectors
const FourVector = Vec{4,Data.Number}

# Alias for non-symmetric 3 tensor
const ThreeTensor = SymmetricSecondOrderTensor{3,Data.Number,6}

# Alias for non-symmetric 4 tensor
const FourTensor = SecondOrderTensor{4,Data.Number,16}

# Alias for symmetric 4 tensor
const StateTensor = SymmetricSecondOrderTensor{4,Data.Number,10}

# Alias for symmetric 4th rank tensor
const Projection = SymmetricFourthOrderTensor{4,Data.Number}

# Alias for non-symmetric 4th rank tensor
const NSProjection = FourthOrderTensor{4,Data.Number}

# Alias for tensor to hold metric derivatives and Christoffel Symbols
# Defined to be symmetric in the last two indices
const Symmetric3rdOrderTensor = Tensor{Tuple{4,@Symmetry{4,4}},Data.Number,3,40}

# Struct for main memory and Runge-Kutta algorithms that holds all state vector variables
struct StateVector <: FieldVector{5,StateTensor}
    g::StateTensor  # metric tensor
    dx::StateTensor # x-derivative
    dy::StateTensor # y-derivative
    dz::StateTensor # z-derivative
    P::StateTensor  # normal projection derivative
end

math_operators = [:+,:-,:*]
# Define math operators for StateVector (Used in Runge-Kutta and Dissipation)
for op in math_operators
    @eval import Base.$op
    @eval @inline function $op(A::StateVector,B::StateVector)
        g  = @. $op(A.g,B.g)
        dx = @. $op(A.dx,B.dx)
        dy = @. $op(A.dy,B.dy)
        dz = @. $op(A.dz,B.dz)
        P  = @. $op(A.P,B.P)
        return StateVector(g,dx,dy,dz,P)
    end

    @eval @inline function $op(a::Number,B::StateVector)
        g  = $op(a,B.g)
        dx = $op(a,B.dx)
        dy = $op(a,B.dy)
        dz = $op(a,B.dz)
        P  = $op(a,B.P)
        return StateVector(g,dx,dy,dz,P)
    end

    @eval @inline $op(B::StateVector,a::Number) = $op(a,B)
end

@eval @inline function Base.:/(B::StateVector,a::Number)
    g  = B.g/a
    dx = B.dx/a
    dy = B.dy/a
    dz = B.dz/a
    P  = B.P/a
    return StateVector(g,dx,dy,dz,P)
end

@inline Base.:-(A::StateVector) = -1*A

@inline function Base.zero(::Type{StateVector})
    g  = zero(StateTensor)
    return StateVector(g,g,g,g,g)
end

# Some convenience values for these types
const Zero4   = zero(FourVector)
const ZeroST  = zero(StateTensor)
const ZeroSV  = zero(StateVector)
const MinST   = StateVector(StateTensor((-1,0,0,0,1,0,0,1,0,1)),ZeroST,ZeroST,ZeroST,ZeroST)

# Defintions of coefficents for embedded boundary finite differencing operators
# All of this can be found here: https://doi.org/10.1016/j.jcp.2016.12.034

##################################################################
# Coefficent functions for fourth order diagonal norm 
# embedded boundary SBP operator

# Boundary interpolation coefficents
# @inline el1(a) = (a+2)*(a+1)/2
# @inline el2(a) = -a*(a+2)
# @inline el3(a) = a*(a+1)/2

# # Norm coefficents
# @inline h11(a) = 17/48 + a + 11/12*a^2 + 1/3*a^3 + 1/24*a^4
# @inline h22(a) = 59/48 - 3/2*a^2 - 5/6*a^3 - 1/8*a^4
# @inline h33(a) = 43/48 + 3/4*a^2 + 2/3*a^3 + 1/8*a^4
# @inline h44(a) = 49/48 - 1/6*a^2 - 1/6*a^3 - 1/24*a^4

# # Q + Q^T = 0 coefficents
# @inline Q12(a) = 7/12*a^2 + a + 1/48*a^4 + 1/6*a^3 + 59/96 
# @inline Q13(a) = -1/12*a^4 - 5/12*a^3 - 7/12*a^2 - 1/4*a - 1/12 
# @inline Q14(a) = 1/16*a^4 + 1/4*a^3 + 1/4*a^2 - 1/32

# @inline Q23(a) = 3/16*a^4 + 5/6*a^3 + 3/4*a^2 + 59/96
# @inline Q24(a) = -1/6*a^2*(a + 2)^2

# @inline Q34(a) = 5/48*a^4 + 5/12*a^3 + 5/12*a^2 + 59/96

# # Finite difference coefficents 
# # (I have kept the norm) part out for speed
# @inline q11(a) = -el1(a)^2/2
# @inline q12(a) = Q12(a) - el1(a)*el2(a)/2
# @inline q13(a) = Q13(a) - el1(a)*el3(a)/2
# @inline q14(a) = Q14(a)

# @inline q21(a) = -Q12(a) - el1(a)*el2(a)/2
# @inline q22(a) = -el2(a)^2/2
# @inline q23(a) = Q23(a) - el2(a)*el3(a)/2
# @inline q24(a) = Q24(a)

# @inline q31(a) = -Q13(a) - el1(a)*el3(a)/2
# @inline q32(a) = -Q23(a) - el2(a)*el3(a)/2
# @inline q33(a) = -el3(a)^2/2
# @inline q34(a) = Q34(a)

# @inline q41(a) = -Q14(a)
# @inline q42(a) = -Q24(a)
# @inline q43(a) = -Q34(a)

##################################################################
# Coefficent functions for second order diagonal norm 
# embedded boundary SBP operator

# @inline el2_1(a) = (a+1)
# @inline el2_2(a) = -a

# @inline h2_11(a) = (a + 1)^2/2
# @inline h2_22(a) = 1 - a^2/2

# @inline Q2_12(a) = (a+1)/2

# @inline q2_21(a) = -Q2_12(a) - el2_1(a)*el2_2(a)/2
# @inline q2_22(a) = -el2_2(a)^2/2

@inline el1(a) = (a+1)
@inline el2(a) = -a

@inline h00(a) = (a+1)/4
@inline h11(a) = (a+1)^2/4
@inline h22(a) = (4+a-a^2)/4

@inline Q00(a) = -1/2
@inline Q01(a) = (a+1)/4
@inline Q02(a) = (1-a)/4
@inline Q12(a) = (a+1)/4

##################################################################
# Coefficent functions for 3-point embedded boundary SBP operator

# @inline el1(a,b) = 1+a-b/2
# @inline el2(a,b) = -a+b
# @inline el3(a,b) = -b/2

# @inline er1(a,b) = -a/2
# @inline er2(a,b) = a-b
# @inline er3(a,b) = 1+b-a/2

# @inline h11(a,b) = a^2/4 - b^2/4 + a + 3/4
# @inline h22(a,b) = 1/2
# @inline h33(a,b) = b^2/4 - a^2/4 + b + 3/4

# @inline Q12(a,b) = a^2/4 + b^2/4 - a*b/2 + a/2 - b/2 + 1/4
# @inline Q13(a,b) = -a^2/4 - b^2/4 + a*b/2 + a/4 + b/4 + 1/4
# @inline Q23(a,b) = Q12(a,b)

# @inline q11(a,b) = er1(a,b)^2/2 - el1(a,b)^2/2
# @inline q12(a,b) = Q12(a,b) + er1(a,b)*er2(a,b)/2 - el1(a,b)*el2(a,b)/2
# @inline q13(a,b) = Q13(a,b) + er1(a,b)*er3(a,b)/2 - el1(a,b)*el3(a,b)/2

# @inline q31(a,b) = -Q13(a,b) + er1(a,b)*er3(a,b)/2 - el1(a,b)*el3(a,b)/2
# @inline q32(a,b) = -Q23(a,b) + er3(a,b)*er2(a,b)/2 - el3(a,b)*el2(a,b)/2
# @inline q33(a,b) = er3(a,b)^2/2 - el3(a,b)^2/2

@inline fg(U::StateVector)  = U.g

@inline function u(U::StateVector) # time derivative of the metric
    
    g  = U.g
    dx = U.dx
    dy = U.dy
    dz = U.dz
    P  = U.P

    gi = inv(g)

    # if -gi[1,1] < 0
    #     println(g.data)
    # end

    α = 1/sqrt(-gi[1,1])

    βx = -gi[1,2]/gi[1,1]
    βy = -gi[1,3]/gi[1,1]
    βz = -gi[1,4]/gi[1,1]

    return βx*dx + βy*dy + βz*dz - α*P 

end

@inline function vx(U::StateVector) # x-component of the gradient-flux
    
    g  = U.g
    dx = U.dx
    dy = U.dy
    dz = U.dz
    P  = U.P

    gi = inv(g)

    α = 1/sqrt(-gi[1,1])

    βx = -gi[1,2]/gi[1,1]
    βy = -gi[1,3]/gi[1,1]
    βz = -gi[1,4]/gi[1,1]

    nt = 1.0/α; nx = -βx/α; ny = -βy/α; nz = -βz/α;

    n = @Vec [nt,nx,ny,nz]

    γi = gi + symmetric(@einsum n[μ]*n[ν])

    return rootγ(U)*(βx*P - α*(γi[2,2]*dx + γi[2,3]*dy + γi[2,4]*dz))

end

@inline function vy(U::StateVector) # y-component of the gradient-flux
    
    g  = U.g
    dx = U.dx
    dy = U.dy
    dz = U.dz
    P  = U.P

    gi = inv(g)

    α = 1/sqrt(-gi[1,1])

    βx = -gi[1,2]/gi[1,1]
    βy = -gi[1,3]/gi[1,1]
    βz = -gi[1,4]/gi[1,1]

    nt = 1.0/α; nx = -βx/α; ny = -βy/α; nz = -βz/α;

    n = @Vec [nt,nx,ny,nz]

    γi = gi + symmetric(@einsum n[μ]*n[ν])

    return rootγ(U)*(βy*P - α*(γi[3,2]*dx + γi[3,3]*dy + γi[3,4]*dz))

end

@inline function vz(U::StateVector) # z-component of the gradient-flux
    
    g  = U.g
    dx = U.dx
    dy = U.dy
    dz = U.dz
    P  = U.P

    gi = inv(g)

    α = 1/sqrt(-gi[1,1])

    βx = -gi[1,2]/gi[1,1]
    βy = -gi[1,3]/gi[1,1]
    βz = -gi[1,4]/gi[1,1]

    nt = 1.0/α; nx = -βx/α; ny = -βy/α; nz = -βz/α;

    n = @Vec [nt,nx,ny,nz]

    γi = gi + symmetric(@einsum n[μ]*n[ν])

    return rootγ(U)*(βz*P - α*(γi[4,2]*dx + γi[4,3]*dy + γi[4,4]*dz))

end

@inline function rootγ(U::StateVector) # square root of determinant of the 3-metric

    g = U.g

    _,_,_,_,γs... = g.data

    return sqrt(det(ThreeTensor(γs)))

end

Base.@propagate_inbounds @inline function Dx(f,Um,ns,nls,nrs,αls,αrs,i,j,k) # x-derivative
    nl,nr = (nls[1], nrs[1])
    nlr = (nl, nr)
    αs = (αls[1], αrs[1])
    @inbounds @inline U(x) = f(getindex(Um,x,j,k))
    if nr-nl>=3 # not enough points for 4th order stencil, go to 2nd order
        D_2_1(U,nlr,αs,i)
    elseif nr-nl==2 # not enough points for 2nd order stencil, go to 1st order 3 point
        D_3point(U,nlr,αs,i)
    elseif nr-nl==1 # not enough points for 2nd order stencil, go to 1st order 2 point
        -U(nl) + U(nr) # note the two point operator happens to be the same for both points
    else # only one grid point, extrapolate derivative
        throw(ZeroST) # not implemented
    end
end

Base.@propagate_inbounds @inline function Dy(f,Um,ns,nls,nrs,αls,αrs,i,j,k) # y-derivative
    nl,nr = (nls[2], nrs[2])
    nlr = (nl, nr)
    αs = (αls[2], αrs[2])
    @inbounds @inline U(x) = f(getindex(Um,i,x,k))
    if nr-nl>=3 # not enough points for 4th order stencil, go to 2nd order
        D_2_1(U,nlr,αs,j)
    elseif nr-nl==2 # not enough points for 2nd order stencil, go to 1st order 3 point
        D_3point(U,nlr,αs,j)
    elseif nr-nl==1 # not enough points for 2nd order stencil, go to 1st order 2 point
        -U(nl) + U(nr) # note the two point operator happens to be the same for both points
    else # only one grid point, extrapolate derivative
        throw(ZeroST) # not implemented
    end
end

Base.@propagate_inbounds @inline function Dz(f,Um,ns,nls,nrs,αls,αrs,i,j,k) # z-derivative
    nl,nr = (nls[3], nrs[3])
    nlr = (nl, nr)
    αs = (αls[3], αrs[3])
    @inbounds @inline U(x) = f(getindex(Um,i,j,x))
    if nr-nl>=3 # not enough points for 4th order stencil, go to 2nd order
        D_2_1(U,nlr,αs,k)
    elseif nr-nl==2 # not enough points for 2nd order stencil, go to 1st order 3 point
        D_3point(U,nlr,αs,k)
    elseif nr-nl==1 # not enough points for 2nd order stencil, go to 1st order 2 point
        -U(nl) + U(nr) # note the two point operator happens to be the same for both points
    else # only one grid point, extrapolate derivative
        throw(ZeroST) # not implemented
    end
end

# Base.@propagate_inbounds @inline function D_4_2(U,ns,αs,k) # Fourth order accurate stencil
#     nl,nr = ns
#     αl,αr = αs
#     if k in nl+4:nr-4
#         (U(k-2) - 8*U(k-1) + 8*U(k+1) - U(k+2))/12
#     elseif k==nl
#         (q11(αl)*U(nl) + q12(αl)*U(nl+1) + q13(αl)*U(nl+2) + q14(αl)*U(nl+3))/h11(αl)
#     elseif k==nl+1
#         (q21(αl)*U(nl) + q22(αl)*U(nl+1) + q23(αl)*U(nl+2) + q24(αl)*U(nl+3))/h22(αl)
#     elseif k==nl+2
#         (q31(αl)*U(nl) + q32(αl)*U(nl+1) + q33(αl)*U(nl+2) + q34(αl)*U(nl+3) - U(nl+4)/12)/h33(αl)
#     elseif k==nl+3
#         (q41(αl)*U(nl) + q42(αl)*U(nl+1) + q43(αl)*U(nl+2) + (2/3)*U(nl+4) - U(nl+5)/12)/h44(αl)
#     elseif k==nr
#         -(q11(αr)*U(nr) + q12(αr)*U(nr-1) + q13(αr)*U(nr-2) + q14(αr)*U(nr-3))/h11(αr)
#     elseif k==nr-1
#         -(q21(αr)*U(nr) + q22(αr)*U(nr-1) + q23(αr)*U(nr-2) + q24(αr)*U(nr-3))/h22(αr)
#     elseif k==nr-2
#         -(q31(αr)*U(nr) + q32(αr)*U(nr-1) + q33(αr)*U(nr-2) + q34(αr)*U(nr-3) - U(nr-4)/12)/h33(αr)
#     elseif k==nr-3
#         -(q41(αr)*U(nr) + q42(αr)*U(nr-1) + q43(αr)*U(nr-2) + (2/3)*U(nr-4) - U(nr-5)/12)/h44(αr)
#     else
#         throw(0.)
#     end
# end

# Base.@propagate_inbounds @inline function D_2_1(U,ns,αs,k) # Second order accurate stencil
#     nl,nr = ns
#     αl,αr = αs
#     if k in nl+2:nr-2
#         (-U(k-1) + U(k+1))/2
#     elseif k==nl
#         -U(nl) + U(nl+1)
#     elseif k==nl+1
#         (q2_21(αl)*U(nl) + q2_22(αl)*U(nl+1) + U(nl+2)/2)/h2_22(αl)
#     elseif k==nr
#         U(nr) - U(nr-1)
#     elseif k==nr-1
#         -(q2_21(αr)*U(nr) + q2_22(αr)*U(nr-1) + U(nr-2)/2)/h2_22(αr)
#     else
#         #return 0.
#         #println(nl," ",nr," ",αl," ",αr," ",k)
#         throw(ZeroST)
#     end
# end

Base.@propagate_inbounds @inline function D_2_1(U,ns,αs,k) # Second order accurate stencil
    nl,nr = ns
    αl,αr = αs
    if k in nl+2:nr-2
        (-U(k-1) + U(k+1))/2
    elseif k==nl
        Ub = el1(αl)*U(nl) + el2(αl)*U(nl+1)
        (-Q01(αl)*Ub + Q12(αl)*U(nl+1))/h11(αl)
    elseif k==nl+1
        Ub = el1(αl)*U(nl) + el2(αl)*U(nl+1)
        (-Q02(αl)*Ub - Q12(αl)*U(nl) + U(nl+2)/2)/h22(αl)
    elseif k==nr
        Ub = el1(αr)*U(nr) + el2(αr)*U(nr-1)
        -(-Q01(αr)*Ub + Q12(αr)*U(nr-1))/h11(αr)
    elseif k==nr-1
        Ub = el1(αr)*U(nr) + el2(αr)*U(nr-1)
        -(-Q02(αr)*Ub - Q12(αr)*U(nr) + U(nr-2)/2)/h22(αr)
    else
        throw(ZeroST)
    end

end

Base.@propagate_inbounds @inline function D_3point(U,ns,αs,k) # First order accurate stencil for 3 points
    nl,nr = ns
    αl,αr = αs
    throw(0.)
    if k == nl
        (q11(αl,αr)*U(nl) + q12(αl,αr)*U(nl+1) + q13(αl,αr)*U(nr))/h11(αl,αr)
    elseif k==nr
        (q31(αl,αr)*U(nl) + q32(αl,αr)*U(nl+1) + q33(αl,αr)*U(nr))/h33(αl,αr)
    else
        (-U(nl) + U(nr))/2
    end
end

Base.@propagate_inbounds @inline function D4(U,ns,αs,k) # Fourth derivative stencil for dissipation
    nl,nr = ns
    αl,αr = αs
    if k in nl+3:nr-3
        U(k-2) - 4*U(k-1) + 6*U(k) - 4*U(k+1) + U(k+2)
    elseif k==nl
        (2*U(nl) - 4*U(nl+1) + 2*U(nl+2))/h2_11(αl)
    elseif k==nl+1
        (-4*U(nl) + 9*U(nl+1) - 6*U(nl+2) + U(nl+3))/h2_22(αl)
    elseif k==nl+2
        (2*U(nl) - 6*U(nl+1) + 7*U(nl+2) - 4*U(nl+3) + U(nl+4))
    elseif k==nr
        (2*U(nr) - 4*U(nr-1) + 2*U(nr-2))/h2_11(αr)
    elseif k==nr-1
        (-4*U(nr) + 9*U(nr-1) - 6*U(nr-2) + U(nr-3))/h2_22(αr)
    elseif k==nr-2
        (2*U(nr) - 6*U(nr-1) + 7*U(nr-2) - 4*U(nr-3) + U(nr-4))
    else
        throw(ZeroSV)
    end
end

Base.@propagate_inbounds @inline function Dissipation(Um,ns,nls,nrs,αls,αrs,i,j,k) # Dissipation in all directions
    @inbounds @inline Ux(x) = getindex(Um,x,j,k)
    @inbounds @inline Uy(x) = getindex(Um,i,x,k)
    @inbounds @inline Uz(x) = getindex(Um,i,j,x)

    D4(Ux,(nls[1],nrs[1]),(αls[1],αrs[1]),i) + D4(Uy,(nls[2],nrs[2]),(αls[2],αrs[2]),j) + D4(Uz,(nls[3],nrs[3]),(αls[3],αrs[3]),k)
end

Base.@propagate_inbounds @inline function Div(vx,vy,vz,U,ns,nls,nrs,αls,αrs,ds,i,j,k) # Calculate the divergence of the flux
    dx,dy,dz,_ = ds
    (Dx(vx,U,ns,nls,nrs,αls,αrs,i,j,k)/dx + Dy(vy,U,ns,nls,nrs,αls,αrs,i,j,k)/dy + Dz(vz,U,ns,nls,nrs,αls,αrs,i,j,k)/dz)/rootγ(U[i,j,k])
end

@inline function vectors(U,outer,rb,ri,i,l)
    # Returns the null basis to the boundary
    # The embedded boundary method effectively treats the boundary as "Lego" objects
    # with boundary normals only in one of the 3 cardinal directions.
    # Retruns upper index vectors and a mixed index boundary 2-metric

    g = U.g

    sxn,syn,szn = (0.,0.,0.)
    sx,sy,sz = (0.,0.,0.)

    if     i==1
        (sx = l)
    elseif i==2 
        (sy = l)
    elseif i==3
        (sz = l)
    end

    # x,y,z = rb
    # sx = -x
    # sy = -y
    # sz = -z

    # if outer
    #     if     i==1
    #         (sx = l)
    #     elseif i==2 
    #         (sy = l)
    #     elseif i==3
    #         (sz = l)
    #     end
    # else
    #     x,y,z = rb
    #     sx = -x
    #     sy = -y
    #     sz = -z
    # end

    gi = inv(g)

    #if (-gi[1,1]<0) println()

    α = 1/sqrt(-gi[1,1])

    βx = -gi[1,2]/gi[1,1]
    βy = -gi[1,3]/gi[1,1]
    βz = -gi[1,4]/gi[1,1]

    nt = 1.0/α; nx = -βx/α; ny = -βy/α; nz = -βz/α;

    n = FourVector((nt,nx,ny,nz))

    sn = FourVector((0.,sxn,syn,szn))

    snormn = @einsum g[μ,ν]*sn[μ]*sn[ν]
    
    sn = sn/sqrt(snormn) 

    ℓn = (n + sn)/sqrt(2)
    kn = (n - sn)/sqrt(2)

    σn = StateTensor((μ,ν) -> gi[μ,ν] + kn[μ]*ℓn[ν] + ℓn[μ]*kn[ν]) # all indices up

    s = FourVector((0.,sx,sy,sz))

    snorm = @einsum g[μ,ν]*s[μ]*s[ν]
    
    try
    s = s/sqrt(snorm) 
    catch error
        println(ri)
        throw(error)
    end

    ℓ = (n + s)/sqrt(2)
    k = (n - s)/sqrt(2)

    σ = StateTensor((μ,ν) -> gi[μ,ν] + k[μ]*ℓ[ν] + ℓ[μ]*k[ν]) # all indices up

    return (k,ℓ,σ)#,k,ℓ,σ)
end

@inline function find_boundary(ns,ls,ds,r,ri)
    # Returns a tuple of various objects needed to know where the boundary is.
    # The nl and nr are the left and right farthest grid cell indices in all 3 directions
    # The αl and αr are the left and right distances from that last grid cell to the boundary 
    # in units of the grid spacing in all 3 directions
    #
    # To prevent unnecessary computations, we only perform the root solver if the boundary is within
    # five grid cells of the current point, as if this is not the case, then the boundary position 
    # does not matter to the stencils or boundary conditions.

    xi,yi,zi = ri

    lx,ly,lz=ls
    dx,dy,dz,_=ds

    nx,ny,nz = ns

    x,y,z = r

    R = lx/4.

    p = 2

    f(x,y,z) = x^p + y^p + z^p - R^p # Definitions of the boundary position

    #f(x,y,z) = x + 200

    xibl =  1; yibl =  1; zibl =  1;
    xibr = nx; yibr = ny; zibr = nz;

    αlx = 0.; αly = 0.; αlz = 0.;
    αrx = 0.; αry = 0.; αrz = 0.;

    # αlx = 0.5; αly = 0.5; αlz = 0.5;
    # αrx = 0.5; αry = 0.5; αrz = 0.5;

    s = sign(f(x,y,z))

    a = 1/1000

    # x-line
    i = xi+5
    x0 = x+5*dx
    if (i <= nx && sign(f(x0,y,z))≠s)
        #rb = find_zero(x->f(x,y,z), (x,x0), atol=a*dx, rtol=a*dx, Bisection())+lx/2
        rb = find_zero(x->f(x,y,z), SecantMethod{Float64}(x,x0), CompactSolution()).root+lx/2
        temp,αrx = divrem(rb,dx,RoundDown)
        αrx /= dx; xibr=Int(temp)+1;
    end

    i = xi-5
    x0 = x-5*dx
    if (i >= 1 && sign(f(x0,y,z))≠s)
        #rb = find_zero(x->f(x,y,z), (x0,x), atol=a*dx, rtol=a*dx, Bisection())+lx/2
        rb = find_zero(x->f(x,y,z), SecantMethod{Float64}(x0,x), CompactSolution()).root+lx/2
        temp,αlx = divrem(rb,dx,RoundUp)
        αlx /= -dx; xibl=Int(temp)+1;
    end

    # y-line
    i = yi+5
    y0 = y+5*dy
    if (i <= ny && sign(f(x,y0,z))≠s)
        #rb = find_zero(y->f(x,y,z), (y,y0), atol=a*dy, rtol=a*dy, Bisection())+ly/2
        rb = find_zero(y->f(x,y,z), SecantMethod{Float64}(y,y0), CompactSolution()).root+ly/2
        temp,αry = divrem(rb,dy,RoundDown)
        αry /= dy; yibr=Int(temp)+1;
    end

    i = yi-5
    y0 = y-5*dy
    if (i >= 1 && sign(f(x,y0,z))≠s)
        #rb = find_zero(y->f(x,y,z), (y0,y), atol=a*dy, rtol=a*dy, Bisection())+ly/2
        rb = find_zero(y->f(x,y,z), SecantMethod{Float64}(y0,y), CompactSolution()).root+ly/2
        temp,αly = divrem(rb,dy,RoundUp)
        αly /= -dy; yibl=Int(temp)+1;
    end

    # z-line
    i = zi+5
    z0 = z+5*dz
    if (i <= nz && sign(f(x,y,z0))≠s)
        #rb = find_zero(z->f(x,y,z), (z,z0), atol=a*dz, rtol=a*dz, Bisection())+lz/2
        rb = find_zero(z->f(x,y,z), SecantMethod{Float64}(z,z0), CompactSolution()).root+lz/2
        temp,αrz = divrem(rb,dz,RoundDown)
        αrz /= dz; zibr=Int(temp)+1;
    end

    i = zi-5
    z0 = z-5*dz
    if (i >= 1 && sign(f(x,y,z0))≠s)
        #rb = find_zero(z->f(x,y,z), (z0,z), atol=a*dz, rtol=a*dz, Bisection())+lz/2
        rb = find_zero(z->f(x,y,z), SecantMethod{Float64}(z0,z), CompactSolution()).root+lz/2
        temp,αlz = divrem(rb,dz,RoundUp)
        αlz /= -dz
        zibl=Int(temp)+1
    end

    nl = (xibl,yibl,zibl)
    nr = (xibr,yibr,zibr)
    αl = (αlx,αly,αlz)      # this needs to be negative of the right ones.
    αr = (αrx,αry,αrz)

    in_domain = !(s==-1)

    return (in_domain,nl,nr,αl,αr)

end

Base.@propagate_inbounds @inline function SAT(U,ns,ls,ds,nls,nrs,αls,αrs,r,t,ri,γ2)
    # Performs the boundary conditions in each direction
    # Since the StateVector on the boundary is not in the domain, its value must be extrapolated.
    # Since the boundary position is not in the domain, the application of boundary conditions
    # must also be extrapolated.

    lx,ly,lz = ls
    hx,hy,hz,_=ds

    xi,yi,zi=ri
    x,y,z = r

    SATg = ZeroST; SATdx = ZeroST; SATdy = ZeroST; SATdz = ZeroST; SATP = ZeroST; 

    for i in 1:3

        nl = nls[i]; nr = nrs[i];

        if nr-nl<3 
            #(println(nls,nrs); @assert false) 
            throw(ZeroSV)
        end

        if (ri[i]==nl==1) || (nl ≠ 1 && ri[i] in nl:nl+1) # in the boundary region on the left side of the line

            let 
            # I want to use the same variable names here in the left branch 
            # as in the right branch, so there is a let block to allow this
            # and prevent type instability
                
                αl = αls[i]

                # Interpolate the solution vector on the boundary 
                # and determine the boundary position on the coordinate line
                if i == 1 # On an x-line
                    rb = (-lx/2+(nl-1)*hx-αl*hx,y,z)
                    Ub = el1(αl)*U[nl,yi,zi] + el2(αl)*U[nl+1,yi,zi]
                elseif i == 2 # On a y-line
                    rb = (x,-ly/2+(nl-1)*hy-αl*hy,z)
                    Ub = el1(αl)*U[xi,nl,zi] + el2(αl)*U[xi,nl+1,zi]
                elseif i == 3 # On a z-line
                    rb = (x,y,-lz/2+(nl-1)*hz-αl*hz)
                    Ub = el1(αl)*U[xi,yi,nl] + el2(αl)*U[xi,yi,nl+1]
                end

                #return StateVector(SATg,SATdx,SATdy,SATdz,SATP)

                outer = (nl == 1)

                k,ℓ,σ = vectors(Ub,outer,rb,ri,i,-1.) # Form boundary basis

                g  = Ub.g
                dx = Ub.dx
                dy = Ub.dy
                dz = Ub.dz
                P  = Ub.P

                ℓ_ = @einsum g[μ,α]*ℓ[α]
                k_ = @einsum g[μ,α]*k[α]

                σm = @einsum g[μ,α]*σ[α,ν]  # mixed indices down up
                σ_ = @einsum g[μ,α]*σm[ν,α] # all indices down

                gi = inv(g)

                α = 1/sqrt(-gi[1,1])
            
                βx = -gi[1,2]/gi[1,1]
                βy = -gi[1,3]/gi[1,1]
                βz = -gi[1,4]/gi[1,1]

                ∂tg = βx*dx + βy*dy + βz*dz - α*P

                ∂gb = Symmetric3rdOrderTensor((σ,μ,ν) -> (σ==1 ? ∂tg[μ,ν] : σ==2 ? dx[μ,ν] : σ==3 ? dy[μ,ν] : σ==4 ? dz[μ,ν] : throw(ZeroST)))

                Upb = (@einsum  k[α]*∂gb[α,μ,ν])# + γ2*g/sqrt(2)
                U0b = @einsum  σm[α,β]*∂gb[β,μ,ν]

                s_ = (ℓ_ - k_)/sqrt(2)

                β = FourVector((0.,βx,βy,βz))

                βs = @einsum β[μ]*s_[μ]

                βU0 = @einsum β[σ]*U0b[σ,μ,ν]

                cp    =  α - βs
                cm    = -α - βs
                cperp = -βs

                UmBC1 = ZeroST #(cp/cm)*Upb + (sqrt(2)/cm)*βU0# + γ2*(1-cp/cm)*g/sqrt(2)

                # Amp = 1.
                # #Amp = 0.0
                # σ0 = 10
                # μ0 = 110

                # f(t,x,y,z) = (μ0-t-σ0)<x<(μ0-t+σ0) ? Amp : 0.

                # if 
                #     CBC = f(t,x,y,z)*(ones(FourVector))
                # else
                    
                # end
                CBC = zero(FourVector)

                # Three index constraint projector (indices up down down)
                Q3 = Symmetric3rdOrderTensor((α,μ,ν) -> 0.5*(ℓ_[μ]*σm[ν,α] + ℓ_[ν]*σm[μ,α] - σ_[μ,ν]*ℓ[α] - ℓ_[μ]*ℓ_[ν]*k[α])) 

                # Gravitational wave projector (indices down down up up)
                O = Projection((μ,ν,α,β) -> σm[μ,α]*σm[ν,β] - 0.5*σ_[μ,ν]*σ[α,β]) 

                # index down
                A_ = @einsum (2*ℓ[μ]*Upb[μ,α] - gi[μ,ν]*Upb[μ,ν]*ℓ_[α] - 2*gi[μ,ν]*U0b[μ,ν,α] + gi[μ,ν]*U0b[α,μ,ν] + 2*CBC[α]) # + 2*Hxy[α]

                G = minorsymmetric(NSProjection((μ,ν,α,β) -> (2k_[μ]*ℓ_[ν]*k[α]*ℓ[β] - 2k_[μ]*σm[ν,α]*ℓ[β] + k_[μ]*k_[ν]*ℓ[α]*ℓ[β])))

                UmBC = (@einsum Q3[α,μ,ν]*A_[α] + O[μ,ν,α,β]*UmBC1[α,β] + G[μ,ν,α,β]*UmBC1[α,β])# + γ2*g/sqrt(2)

                PBC  = -(Upb + UmBC)/sqrt(2)
                ∂gBC =  Symmetric3rdOrderTensor((α,μ,ν) -> U0b[α,μ,ν] - k_[α]*UmBC[μ,ν] - ℓ_[α]*Upb[μ,ν]) #Symmetric3rdOrderTensor

                if ri[i] == nl 
                    ε = abs(cm)*el1(αl)/h00(αl)/ds[i]
                elseif ri[i] == nl+1
                    ε = abs(cm)*el2(αl)/h00(αl)/ds[i]
                end

                # s = (k-ℓ)/sqrt(2)

                # sbx = -x; sby = -y; sbz = -z; 

                # norm = sqrt(sbx^2 + sby^2 + sbz^2)

                # sb = @Vec [0.,sbx/norm,sby/norm,sbz/norm]

                # ε *= abs(@einsum s[μ]*sb[μ])

                SATP  += ε*(PBC - P)
                SATdx += ε*(∂gBC[2,:,:] - dx)
                SATdy += ε*(∂gBC[3,:,:] - dy)
                SATdz += ε*(∂gBC[4,:,:] - dz)

            end

        end

        if (ri[i]==nr==ns[i]) || (nr ≠ ns[i] && ri[i] in (nr-1):nr) # in the boundary region on the right side of the line

            let
            # I want to use the same variable names here in the right branch 
            # as in the left branch before, so there is a let block to allow this
            # and prevent type instability

                αr = αrs[i]

                # Interpolate the solution vector on the boundary 
                # and determine the boundary position on the coordinate line
                if i == 1 # On an x-line
                    rb = (-lx/2+(nr-1)*hx+αr*hx,y,z)
                    Ub = el1(αr)*U[nr,yi,zi] + el2(αr)*U[nr-1,yi,zi]
                elseif i == 2 # On a y-line
                    rb = (x,-ly/2+(nr-1)*hy+αr*hy,z)
                    Ub = el1(αr)*U[xi,nr,zi] + el2(αr)*U[xi,nr-1,zi]
                elseif i == 3 # On a z-line
                    rb = (x,y,-lz/2+(nr-1)*hz+αr*hz)
                    Ub = el1(αr)*U[xi,yi,nr] + el2(αr)*U[xi,yi,nr-1]
                end
        
                outer =  (nr == ns[i])
        
                k,ℓ,σ = vectors(Ub,outer,rb,ri,i,1.) # Form boundary basis
        
                g  = Ub.g
                dx = Ub.dx
                dy = Ub.dy
                dz = Ub.dz
                P  = Ub.P
        
                ℓ_ = @einsum g[μ,α]*ℓ[α]
                k_ = @einsum g[μ,α]*k[α]

                σm = @einsum g[μ,α]*σ[α,ν]  # mixed indices down up
                σ_ = @einsum g[μ,α]*σm[ν,α] # all indices down
        
                gi = inv(g)
        
                α = 1/sqrt(-gi[1,1])
        
                βx = -gi[1,2]/gi[1,1]
                βy = -gi[1,3]/gi[1,1]
                βz = -gi[1,4]/gi[1,1]
        
                ∂tg = βx*dx + βy*dy + βz*dz - α*P
        
                ∂gb = Symmetric3rdOrderTensor((σ,μ,ν) -> (σ==1 ? ∂tg[μ,ν] : σ==2 ? dx[μ,ν] : σ==3 ? dy[μ,ν] : σ==4 ? dz[μ,ν] : throw(ZeroST)))
        
                Upb = (@einsum k[α]*∂gb[α,μ,ν])# + γ2*g/sqrt(2)
                U0b = @einsum σm[α,β]*∂gb[β,μ,ν] # Symmetric3rdOrderTensor
        
                s_ = (ℓ_ - k_)/sqrt(2)

                β = FourVector((0.,βx,βy,βz))

                βs = @einsum β[μ]*s_[μ]

                βU0 = @einsum β[σ]*U0b[σ,μ,ν]

                cp    =  α - βs
                cm    = -α - βs
                cperp = -βs

                UmBC1 = ZeroST#(cp/cm)*Upb + (sqrt(2)/cm)*βU0# + γ2*(1-cp/cm)*g/sqrt(2)

                Amp = 0.001
                #Amp = 0.0
                σ0 = 2.5
                μ0 = 61
                #x0 = μ0 - t

                # r0 = sqrt((x-x0)^2+y^2+z^2)

                # model = ((r0 < σ0) ? (Amp/σ0^8)*(r0+σ0)^4*(r0-σ0)^4 : 0.)

                model =  ((ri[i]==nr==ns[i])&&(μ0-t-σ0)<x<(μ0-t+σ0)&&-20<y<20&&-20<z<20) ? Amp : 0.

                # if i==1 && r[i]==nr
                    
                # else
                #     CBC = zero(FourVector)
                # end
                CBC = model*(FourVector((1,0,0,0)))#(ones(FourVector))

                # Three index constraint projector (indices up down down)
                Q3 = Symmetric3rdOrderTensor((α,μ,ν) -> 0.5*(ℓ_[μ]*σm[ν,α] + ℓ_[ν]*σm[μ,α] - σ_[μ,ν]*ℓ[α] - ℓ_[μ]*ℓ_[ν]*k[α])) 

                # Gravitational wave projector (indices down down up up)
                O = Projection((μ,ν,α,β) -> σm[μ,α]*σm[ν,β] - 0.5*σ_[μ,ν]*σ[α,β]) 

                # index down
                A_ = @einsum (2*ℓ[μ]*Upb[μ,α] - gi[μ,ν]*Upb[μ,ν]*ℓ_[α] - 2gi[μ,ν]*U0b[μ,ν,α] + gi[μ,ν]*U0b[α,μ,ν] + 2*CBC[α]) # + 2*Hxy[α]

                G = minorsymmetric(NSProjection((μ,ν,α,β) -> (2k_[μ]*ℓ_[ν]*k[α]*ℓ[β] - 2k_[μ]*σm[ν,α]*ℓ[β] + k_[μ]*k_[ν]*ℓ[α]*ℓ[β])))

                UmBC = (@einsum Q3[α,μ,ν]*A_[α] + O[μ,ν,α,β]*UmBC1[α,β] + G[μ,ν,α,β]*UmBC1[α,β])# + γ2*g/sqrt(2)
        
                PBC  = -(Upb + UmBC)/sqrt(2)
                ∂gBC =  Symmetric3rdOrderTensor((α,μ,ν) -> U0b[α,μ,ν] - k_[α]*UmBC[μ,ν] - ℓ_[α]*Upb[μ,ν]) #  Symmetric3rdOrderTensor
        
                if ri[i] == nr
                    ε = abs(cm)*el1(αr)/h00(αr)/ds[i]
                elseif ri[i] == nr-1
                    ε = abs(cm)*el2(αr)/h00(αr)/ds[i]
                end
        
                # s = (k-ℓ)/sqrt(2)
        
                # sbx = -x; sby = -y; sbz = -z; 
        
                # norm = sqrt(sbx^2 + sby^2 + sbz^2)
        
                # sb = @Vec [0.,sbx/norm,sby/norm,sbz/norm]
        
                # ε *= abs(@einsum s[μ]*sb[μ])
        
                SATP  += ε*(PBC - P)
                SATdx += ε*(∂gBC[2,:,:] - dx)
                SATdy += ε*(∂gBC[3,:,:] - dy)
                SATdz += ε*(∂gBC[4,:,:] - dz)

            end

        end
        
    end

    return StateVector(SATg,SATdx,SATdy,SATdz,SATP)

end

function energy_cell(U,ns,ls,ds,ri)
    # Calculates the energy estimate (not yet implemented)

    # ψx = U.ψx
    # ψy = U.ψy
    # ψz = U.ψz
    # Ψ  = U.Ψ

    # xi,yi,zi = ri

    # dx,dy,dz,_ = ds
    # lx,ly,lz=ls
    # x = -lx/2 + (xi-1)*dx; y = -ly/2 + (yi-1)*dy; z = -lz/2 + (zi-1)*dz;
    # r = (x,y,z)

    # outside,nls,nrs,αls,αrs = find_boundary(ns,ls,ds,r,ri)

    # h = 1.

    # for i in 1:3
    #     if ri[i] == nrs[i]
    #         h *= h2_11(αrs[i])
    #     elseif ri[i] == nrs[i]-1
    #         h *= h2_22(αrs[i])
    #     elseif ri[i] == nls[i]
    #         h *= h2_11(αls[i])
    #     elseif ri[i] == nls[i]+1
    #         h *= h2_22(αls[i])
    #     end
    # end

    # return (Ψ^2 + ψx^2 + ψy^2 + ψz^2)*h*dx*dy*dz
    return

end

function constraints(U::StateVector)

    g  = U.g
    dx = U.dx
    dy = U.dy
    dz = U.dz
    P  = U.P

    gi = inv(g)

    α = 1/sqrt(-gi[1,1])

    βx = -gi[1,2]/gi[1,1]
    βy = -gi[1,3]/gi[1,1]
    βz = -gi[1,4]/gi[1,1]

    # Calculate time derivative of the metric
    ∂tg = βx*dx + βy*dy + βz*dz - α*P

    ∂g = Symmetric3rdOrderTensor((σ,μ,ν) -> (σ==1 ? ∂tg[μ,ν] : σ==2 ? dx[μ,ν] : σ==3 ? dy[μ,ν] : σ==4 ? dz[μ,ν] : throw(ZeroST)))

    Γ  = Symmetric3rdOrderTensor((σ,μ,ν) -> 0.5*(∂g[ν,μ,σ] + ∂g[μ,ν,σ] - ∂g[σ,μ,ν]))    

    C_ = @einsum gi[ϵ,σ]*Γ[μ,ϵ,σ]

    return C_

end


@parallel_indices (xi,yi,zi) function rhs!(U1,U2,U3,ns::Tuple,ls::Tuple,ds::Tuple,t::Data.Number,iter::Int)
    # Performs the right hand side of the system of equations.
    # The if blocks at the beginning and end perform the 3rd order Runge-Kutta algorithm
    # which is done by calling rhs! three times, each with a different value in iter ranging from 1:3

    if iter == 1 # U is the current read memory, and Uw is the current write memory
        U = U1
        Uw = U2
    elseif iter == 2
        U = U2
        Uw = U3
    else
        U = U3
        Uw = U1
    end

    hx,hy,hz,dt = ds
    lx,ly,lz=ls
    x = -lx/2 + (xi-1)*hx; y = -ly/2 + (yi-1)*hy; z = -lz/2 + (zi-1)*hz;

    r = (x,y,z)
    ri = (xi,yi,zi)

    in_domain,nls,nrs,αls,αrs = find_boundary(ns,ls,ds,r,ri)

    if in_domain

        Uxyz = U[xi,yi,zi]

        g  = Uxyz.g
        dx = Uxyz.dx
        dy = Uxyz.dy
        dz = Uxyz.dz
        P  = Uxyz.P

        gi = inv(g)

        α = 1/sqrt(-gi[1,1]) # Calculate lapse
    
        βx = -gi[1,2]/gi[1,1] # Calculate shift vector
        βy = -gi[1,3]/gi[1,1]
        βz = -gi[1,4]/gi[1,1]

        n  = FourVector((1.0/α,-βx/α,-βy/α,-βz/α))
        n_ = FourVector((-α,0.,0.,0.))

        γi = gi + symmetric(@einsum n[μ]*n[ν])

        ∂tg = βx*dx + βy*dy + βz*dz - α*P

        # collect metric derivatives
        ∂g = Symmetric3rdOrderTensor((σ,μ,ν) -> (σ==1 ? ∂tg[μ,ν] : σ==2 ? dx[μ,ν] : σ==3 ? dy[μ,ν] : σ==4 ? dz[μ,ν] : throw(ZeroST)))
        
        # Calculate Christoffel symbols
        Γ  = Symmetric3rdOrderTensor((σ,μ,ν) -> 0.5*(∂g[ν,μ,σ] + ∂g[μ,ν,σ] - ∂g[σ,μ,ν]))

        ∂trootγ = @einsum 0.5*γi[μ,ν]*∂tg[μ,ν]

        #########################################################
        # Principle (linear) part of the evolution equations

        ∂tdx = Dx(u,U,ns,nls,nrs,αls,αrs,xi,yi,zi)/hx

        ∂tdy = Dy(u,U,ns,nls,nrs,αls,αrs,xi,yi,zi)/hy

        ∂tdz = Dz(u,U,ns,nls,nrs,αls,αrs,xi,yi,zi)/hz

        ∂tP  = Div(vx,vy,vz,U,ns,nls,nrs,αls,αrs,ds,xi,yi,zi)

        #########################################################
        # Non-linear terms in the evolution equations

        # ∂tP -=   α*symmetric(@einsum (μ,ν) ->  gi[λ,γ]*gi[ϵ,σ]*Γ[λ,ϵ,σ]*∂g[γ,μ,ν])

        # ∂tP += 2*α*symmetric(@einsum (μ,ν) -> gi[ϵ,σ]*gi[λ,ρ]*∂g[λ,ϵ,μ]*∂g[ρ,σ,ν])
    
        # ∂tP -= 2*α*symmetric(@einsum (μ,ν) ->   gi[ϵ,σ]*gi[λ,ρ]*Γ[μ,ϵ,λ]*Γ[ν,σ,ρ])

        # ∂tP -= ∂trootγ*P

        #########################################################
        # Constraints and constraint damping terms

        γ0 = 0. # Harmonic constraint damping (>0)
        #γ1 = -1. # Linear Degeneracy parameter (=-1)
        γ2 = 0. # Derivative constraint damping (>0)
        
        Cx = Dx(fg,U,ns,nls,nrs,αls,αrs,xi,yi,zi)/hx - dx
        Cy = Dy(fg,U,ns,nls,nrs,αls,αrs,xi,yi,zi)/hy - dy
        Cz = Dz(fg,U,ns,nls,nrs,αls,αrs,xi,yi,zi)/hz - dz

        ∂tdx += γ2*α*Cx
        ∂tdy += γ2*α*Cy
        ∂tdz += γ2*α*Cz

        # Generalized Harmonic constraints and damping

        C_ = @einsum gi[ϵ,σ]*Γ[μ,ϵ,σ] # - Hxy[μ]

        ∂tP += γ0*α*symmetric(@einsum (μ,ν) -> 2C_[μ]*n_[ν] - g[μ,ν]*n[ϵ]*C_[ϵ]) # + C_[ν]*n_[μ]

        ∂tP += -γ2*(βx*Cx + βy*Cy + βz*Cz)

        ########################################################

        ∂tU  = StateVector(∂tg,∂tdx,∂tdy,∂tdz,∂tP)

        # Perform boundary conditions
        ∂tU += SAT(U,ns,ls,ds,nls,nrs,αls,αrs,r,t,ri,γ2)

        # Add numerical dissipation
        #∂tU += -1*Dissipation(U,ns,nls,nrs,αls,αrs,xi,yi,zi)

        if iter == 1
            U1t = Uxyz
            Uwxyz = U1t + dt*∂tU
        elseif iter == 2
            U1t = U1[xi,yi,zi]
            U2t = Uxyz
            Uwxyz = (3/4)*U1t + (1/4)*U2t + (1/4)*dt*∂tU
        elseif iter == 3
            U1t = U1[xi,yi,zi]
            U2t = Uxyz
            Uwxyz = (1/3)*U1t + (2/3)*U2t + (2/3)*dt*∂tU
        end

        Uw[xi,yi,zi] = Uwxyz

    else # don't do anything if not in the computational domain


        Uw[xi,yi,zi] = MinST
        

    end

    return

end

# struct NothingException <: Exception end

# mutable struct ErrorContainer
#     error::Exception
#     stacktrace::AbstractArray
# end

# # Create variable to store error, initially a NothingException
# global_error = ErrorContainer(NothingException(),[1.])

# # Returns true when global_error is changed to some thown error
# function error_handler(state,t,integrator)
#     return !(typeof(global_error.error) == NothingException)
# end

##################################################
@views function main()

    # Physics
    lx, ly, lz = 100.0, 100.0, 100.0  # domain extends
    ls = (lx,ly,lz)
    t  = 0.0                          # physical start time

    scale = 1

    # Numerics
    n = Int(100*scale)
    ns = (n,n,n)             # numerical grid resolution; should be a mulitple of 32-1 for optimal GPU perf
    nx, ny, nz = ns
    nt         = 1000#Int(1000*scale)                 # number of timesteps

    SAVE = false
    VISUAL = true
    nout       = 2#Int(10*scale)                       # plotting frequency
    noutsave   = 10000#Int(50*scale) 


    save_size = 50
    step = Int(n/save_size)

    # Derived numerics
    hx, hy, hz = lx/(nx-1), ly/(ny-1), lz/(nz-1) # cell sizes
    CFL = 1/5
    dt = min(hx,hy,hz)*CFL
    ds = (hx,hy,hz,dt)

    coords = (-lx/2:hx:lx/2, -ly/2:hy:ly/2, -lz/2:hz:lz/2)
    X,Y,Z = coords

    # ri = (50,47,26)
    # println(X[ri[1]]," ",Y[ri[2]]," ",Z[ri[3]])
    # return find_boundary(ls,ds,ri)

    # test = WaveCell(1,2,3,4,5)
    # return test[2]

    # Initial Conditions
    @inline g_init(x,y,z,μ,ν) =    (( -1. ,     0.    ,    0.  ,   0.   ),
                                    (  0. ,     1.    ,    0.  ,   0.   ),
                                    (  0. ,     0.    ,    1.  ,   0.   ),
                                    (  0. ,     0.    ,    0.  ,   1.   ))[μ][ν]

    @inline ∂tg_init(x,y,z,μ,ν) =  ((  0. ,     0.    ,    0.  ,   0.   ),
                                    (  0. ,     0.    ,    0.  ,   0.   ),
                                    (  0. ,     0.    ,    0.  ,   0.   ),
                                    (  0. ,     0.    ,    0.  ,   0.   ))[μ][ν]

    # Derivatives of the initial conditions                            
    @inline ∂xg(x,y,z,μ,ν) = ForwardDiff.derivative(x -> g_init(x,y,z,μ,ν), x)
    @inline ∂yg(x,y,z,μ,ν) = ForwardDiff.derivative(y -> g_init(x,y,z,μ,ν), y)
    @inline ∂zg(x,y,z,μ,ν) = ForwardDiff.derivative(z -> g_init(x,y,z,μ,ν), z)

    # Gauge functions and derivatives (freely specifyable)
    @inline fH_(x,y,z,μ) = (0.,0.,0.,0.)[μ] # lower index
    
    @inline f∂H_(x,y,z,μ,ν) = ((0.,0.,0.,0.)[ν],
                               (0.,0.,0.,0.)[ν],
                               (0.,0.,0.,0.)[ν],
                               (0.,0.,0.,0.)[ν])[μ]

    # Array allocations
    # If we use the GPU, we need an intermediate array on the CPU to save
    if USE_GPU
        U0 = CPUCellArray{StateVector}(undef, nx, ny, nz)
        U1 =  CuCellArray{StateVector}(undef, nx, ny, nz)
        U2 =  CuCellArray{StateVector}(undef, nx, ny, nz)
        U3 =  CuCellArray{StateVector}(undef, nx, ny, nz)
    else
        U1 =  CPUCellArray{StateVector}(undef, nx, ny, nz)
        U2 =  CPUCellArray{StateVector}(undef, nx, ny, nz)
        U3 =  CPUCellArray{StateVector}(undef, nx, ny, nz)
    end

    # Write initial conditions to arrays
    for xi in 1:ns[1], yi in 1:ns[2], zi in 1:ns[3]

        x = X[xi]
        y = Y[yi]
        z = Z[zi]

        g   = StateTensor((μ,ν) -> g_init(x,y,z,μ,ν))
        dx  = StateTensor((μ,ν) -> ∂xg(x,y,z,μ,ν))
        dy  = StateTensor((μ,ν) -> ∂yg(x,y,z,μ,ν))
        dz  = StateTensor((μ,ν) -> ∂yg(x,y,z,μ,ν))
        ∂tg = StateTensor((μ,ν) -> ∂tg_init(x,y,z,μ,ν))

        gi = inv(g)

        α = 1/sqrt(-gi[1,1])
    
        βx = -gi[1,2]/gi[1,1]
        βy = -gi[1,3]/gi[1,1]
        βz = -gi[1,4]/gi[1,1]

        P = -(∂tg - βx*dx - βy*dy - βz*dz)/α

        if USE_GPU
            U0[xi,yi,zi] = StateVector(g,dx,dy,dz,P)
        else
            U1[xi,yi,zi] = StateVector(g,dx,dy,dz,P)
        end

    end

    if USE_GPU; copy!(U1.data, U0.data) end

    copy!(U2.data, U1.data)
    copy!(U3.data, U1.data)

    # return @code_warntype vectors(U1[1,1,1],1,1.)

    #ri=(1,1,1)
    # r=(1.,1.,1.)
    # return @code_warntype SAT(U1,ns,ls,ds,(1,1,1),(nx,ny,nz),(0.,0.,0.),(0.,0.,0.),r,0.,ri,0.)
    #return @code_warntype Dx(u,U1,ns,(1,1,1),(nx,ny,nz),(0.,0.,0.),(0.,0.,0.),ri...)

    #return @benchmark rhs!($U1,$U2,$U3,$ns,$ls,$ds,1)
    #return @benchmark @parallel (1:$nx,1:$ny,1:$nz) rhs!($U1,$U2,$U3,$ns,$ls,$ds,0.,1)

    # Preparation to save

    # path = string("3D_data")
    # old_files = readdir(path; join=true)
    # for i in 1:length(old_files) 
    #     rm(old_files[i]) 
    # end
    # datafile = h5open(path*"/data.h5","cw")
    # nsave = 1#Int64(nt/noutsave) + 1
    # gdata = create_dataset(datafile, "phi", datatype(Data.Number), 
    #         dataspace(nsave,6,save_size,save_size,save_size), 
    #         chunk=(1,6,save_size,save_size,save_size))

    # Preparation of visualisation
    ENV["GKSwstype"]="nul"; if isdir("viz3D_out")==false mkdir("viz3D_out") end; loadpath = "./viz3D_out/"; anim = Animation(loadpath,String[])
    old_files = readdir(loadpath; join=true)
    for i in 1:length(old_files) rm(old_files[i]) end
    println("Animation directory: $(anim.dir)")

    slice       = (:,:,Int(ceil(nz/2)))

    # bulk = (1:nx,1:ny,1:nz) #@parallel $bulk 
    # @profile @parallel bulk rhs!(U1,U2,U3,ns,ls,ds,3)

    #return Profile.print()
    # return @benchmark @parallel $bulk rhs!($U1,$U2,$U3,$ns,$ls,$ds,3)

    #evec = zeros(0)

    iter = 1

    try # try block to catch errors, close datafile if so

        # Time loop
        for it = 1:nt

            if (it==11)  global wtime0 = Base.time()  end # performance timing 

            # Perform RK3 algorithm
            for i in 1:3
                @parallel (1:nx,1:ny,1:nz) rhs!(U1,U2,U3,ns,ls,ds,t,i) 
            end

            t = t + dt

            # Visualisation
            if VISUAL && (mod(it,nout)==0)

                #A = Array(getindex.(CellArrays.field(U1,1),1,1))[slice...] .+ 1

                zi = Int(ceil(nz/2))

                #A = [(Dy(fg,U1,ns,find_boundary(ns,ls,ds,(X[xi],Y[yi],Z[zi]),(xi,yi,zi))[2:5]...,xi,yi,zi)/hy - U1[xi,yi,zi].dy)[1,1] for xi in 1:nx, yi in 1:ny]
                A = [constraints(U1[xi,yi,Int(ceil(nz/3))])[1] for xi in 1:nx, yi in 1:ny]

                heatmap(X, Y, A, aspect_ratio=1, xlims=(-lx/2,lx/2)
                ,ylims=(-ly/2,ly/2),clim=(-0.001,0.001), title = "Time = "*string(round(t; digits=2)), c=:viridis,frame=:box)

                ang = range(0, 2π, length = 60)
                circle(x,y,r) = Shape(r*sin.(ang).+x, r*cos.(ang).+y)  

                plot!(circle(0,0,25),fc=:transparent, legend = false, colorbar = true)

                frame(anim)

                # if USE_GPU
                #     copy!(U0.data, U1.data)
                #     result = 0.
                #     for xi in 1:nx, yi in 1:ny, zi in 1:nz
                #         result += energy_cell(U0[xi,yi,zi],ns,ls,ds,(xi,yi,zi))
                #     end
                # else
                #     result = 0.
                #     for xi in 1:nx, yi in 1:ny, zi in 1:nz
                #         result += energy_cell(U1[xi,yi,zi],ns,ls,ds,(xi,yi,zi))
                #     end
                # end

                # result = mapreduce(energycell,+,U1)

                # append!(evec,result)

            end

            if SAVE && mod(it,noutsave)==0

                if USE_GPU
                    copy!(U0.data, U1.data)
                    gdata[iter,:,:,:,:] = [U0[xi,yi,zi].g.data[i] for i in 1:6, zi in 1:step:nz, yi in 1:step:ny, xi in 1:step:nx]
                else
                    gdata[iter,:,:,:,:] = [U1[xi,yi,zi].g.data[i] for i in 1:6, zi in 1:step:nz, yi in 1:step:ny, xi in 1:step:nx]
                end

                iter += 1

            end

        end

    catch error
        #close(datafile)
        GC.gc(true)
        #throw(error)
        # Print error if one is encountered, with line of occurance

        st = stacktrace(catch_backtrace())

        j = 1
        for i in 1:length(st)
            if st[i].file == Symbol(@__FILE__)
                j = i
                break
            end
        end

        println(" ")
        println("Exited with error on line ", st[j].line)
        println(" ")
        printstyled(stderr,"ERROR: ", bold=true, color=:red)
        printstyled(stderr,sprint(showerror,error))
        println(stderr)
    end

    #close(datafile)

    # Performance metrics
    wtime    = Base.time() - wtime0
    bytes_accessed = sizeof(StateVector)*(2*3)   # Bytes accessed per iteration per gridpoint (1 read and write iter=1 and 2 reads 1 write for iter=2,3)
    A_eff    = bytes_accessed*nx*ny*nz/1e9       # Effective main memory access per iteration [GB] 
    wtime_it = wtime/(nt-10)                     # Execution time per iteration [s]
    T_eff    = A_eff/wtime_it                    # Effective memory throughput [GB/s]

    @printf("Total steps=%d, time=%1.3f min (@ T_eff = %1.2f GB/s) \n", nt, wtime/60, round(T_eff, sigdigits=3))
   
    GC.gc(true)

    if VISUAL 
        gif(anim, "acoustic3D.gif", fps = 15) 
        return #plot((evec)./evec[1], ylim = (0, 2))
    else
        return
    end

end

end