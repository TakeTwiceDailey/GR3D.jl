module GR3D

using ParallelStencil
using CellArrays, StaticArrays, CUDA

using HDF5
using FileIO

using Plots, Printf, Statistics, BenchmarkTools, ForwardDiff

using Tensorial, InteractiveUtils

#using RootSolvers # (Roots.jl does not work on GPU)

using Roots

using Profile

using Polyester

using Distributions

using DoubleFloats

ParallelStencil.@reset_parallel_stencil()

const USE_GPU = false  # Use GPU? If this is set false, then no GPU needs to be available

@static if USE_GPU
    @define_CuCellArray() # without this CuCellArrays don't work
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
    #@init_parallel_stencil(Polyester, Float64, 3)
end

# Alias for 4 vectors
const FourVector = Vec{4,Data.Number}

# Alias for 3 vector
const ThreeVector = Vec{3,Data.Number}

# Alias for symmetric 2 tensor
const TwoTensor = SymmetricSecondOrderTensor{2,Data.Number,3}

# Alias for symmetric 3 tensor
const ThreeTensor = SymmetricSecondOrderTensor{3,Data.Number,6}

# Alias for non-symmetric 4 tensor
const FourTensor = SecondOrderTensor{4,Data.Number,16}

# Alias for symmetric 4 tensor
const StateTensor = SymmetricSecondOrderTensor{4,Data.Number,10}

# Alias for symmetric 4th rank tensor
const Projection = SymmetricFourthOrderTensor{4,Data.Number}

# Alias for non-symmetric 4th rank tensor
const NSProjection = FourthOrderTensor{4,Data.Number}

const SurfaceProjector = Mat{3,2,Data.Number}

# Alias for tensor to hold metric derivatives and Christoffel Symbols
# Defined to be symmetric in the last two indices
const Symmetric3rdOrderTensor = Tensor{Tuple{4,@Symmetry{4,4}},Data.Number,3,40}

# Alias for 3 dimensional metric derivatives and Christoffel Symbols
const Symmetric3rdOrderTensor3 = Tensor{Tuple{3,@Symmetry{3,3}},Data.Number,3,18}

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
const NaNST   = StateTensor((NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN))
const ZeroSV  = zero(StateVector)
const NaNSV   = StateVector(NaNST,NaNST,NaNST,NaNST,NaNST)

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

@inline el2_1(a) = (a+1)
@inline el2_2(a) = -a

@inline h2_11(a) = (a + 1)^2/2
@inline h2_22(a) = 1 - (a^2)/2

@inline Q2_12(a) = (a+1)/2

# @inline q2_21(a) = -Q2_12(a) - el2_1(a)*el2_2(a)/2
# @inline q2_22(a) = -el2_2(a)^2/2

@inline q2_21(a) = (a^2-1)/2
@inline q2_22(a) = -a^2/2

# @inline q2_21(a) = -1/2
# @inline q2_22(a) = 0

##################################################################
# Coefficent functions for 3-point embedded boundary SBP operator

@inline el1(a,b) = 1+a-b/2
@inline el2(a,b) = -a+b
@inline el3(a,b) = -b/2

@inline er1(a,b) = -a/2
@inline er2(a,b) = a-b
@inline er3(a,b) = 1+b-a/2

@inline h11(a,b) = a^2/4 - b^2/4 + a + 3/4
@inline h22(a,b) = 1/2
@inline h33(a,b) = b^2/4 - a^2/4 + b + 3/4

@inline Q12(a,b) = a^2/4 + b^2/4 - a*b/2 + a/2 - b/2 + 1/4
@inline Q13(a,b) = -a^2/4 - b^2/4 + a*b/2 + a/4 + b/4 + 1/4
@inline Q23(a,b) = Q12(a,b)

@inline q11(a,b) = er1(a,b)^2/2 - el1(a,b)^2/2
@inline q12(a,b) = Q12(a,b) + er1(a,b)*er2(a,b)/2 - el1(a,b)*el2(a,b)/2
@inline q13(a,b) = Q13(a,b) + er1(a,b)*er3(a,b)/2 - el1(a,b)*el3(a,b)/2

@inline q31(a,b) = -Q13(a,b) + er1(a,b)*er3(a,b)/2 - el1(a,b)*el3(a,b)/2
@inline q32(a,b) = -Q23(a,b) + er3(a,b)*er2(a,b)/2 - el3(a,b)*el2(a,b)/2
@inline q33(a,b) = er3(a,b)^2/2 - el3(a,b)^2/2

@inline fg(U::StateVector)  = U.g
@inline fdx(U::StateVector) = U.dx
@inline fdy(U::StateVector) = U.dy
@inline fdz(U::StateVector) = U.dz
@inline fP(U::StateVector)  = U.P

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

    n = FourVector((1,-βx,-βy,-βz))/α

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

    n = FourVector((1,-βx,-βy,-βz))/α

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

    n = FourVector((1,-βx,-βy,-βz))/α

    γi = gi + symmetric(@einsum n[μ]*n[ν])

    return rootγ(U)*(βz*P - α*(γi[4,2]*dx + γi[4,3]*dy + γi[4,4]*dz))

end

@inline function rootγ(U::StateVector) # square root of determinant of the 3-metric

    g = U.g

    _,_,_,_,γs... = g.data

    return sqrt(det(ThreeTensor(γs)))

end

@inline function H(nl,nr,αl,αr,i) # 1D norm coefficients
    if nr-nl>=3 # 2nd order norm
        if i in nl+2:nr-2
            1.
        elseif i == nr
            h2_11(αr)
        elseif i == nr-1
            h2_22(αr)
        elseif i == nl
            h2_11(αl)
        elseif i == nl+1
            h2_22(αl)
        else
            println(nl," ",i," ",nr)
            println(αl,αr)
            throw(0.)
        end
    elseif nr-nl==2 # not enough points for 2nd order norm, go to 3 point
        if     i == nl
            h11(αl,αr)
        elseif i == nr
            h33(αl,αr)
        elseif nl<i<nr
            0.5
        else
            throw(0.)
        end
    elseif nr-nl==1 # not enough points for 2nd order norm, go to 2 point
        if     i == nl
            αl*(αl+2)/2 - (αr^2-1)/2
        elseif i == nr
            αr*(αr+2)/2 - (αl^2-1)/2
        else
            throw(0.)
        end
    else # only one grid point, norm is just 1
        1.
    end
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
        throw(ZeroST) # not implemented
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
        throw(ZeroST) # not implemented
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
        throw(ZeroST) # not implemented
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

Base.@propagate_inbounds @inline function D_2_1(U,ns,αs,k) # Second order accurate stencil
    nl,nr = ns
    αl,αr = αs
    if k in nl+2:nr-2
        (-U(k-1) + U(k+1))/2
    elseif k==nl
        -U(nl) + U(nl+1)
    elseif k==nl+1
        (q2_21(αl)*U(nl) + q2_22(αl)*U(nl+1) + U(nl+2)/2)/h2_22(αl)
    elseif k==nr
        U(nr) - U(nr-1)
    elseif k==nr-1
        -(q2_21(αr)*U(nr) + q2_22(αr)*U(nr-1) + U(nr-2)/2)/h2_22(αr)
    else
        #return 0.
        #println(nl," ",nr," ",αl," ",αr," ",k)
        throw(ZeroST)
    end
end

Base.@propagate_inbounds @inline function D_3point(U,ns,αs,k) # First order accurate stencil for 3 points
    nl,nr = ns
    αl,αr = αs
    throw(ZeroST)
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

Base.@propagate_inbounds @inline function Dissipation(Um,ns,ds,nls,nrs,αls,αrs,xi,yi,zi) # Dissipation in all directions
    @inbounds @inline Ux(x) = getindex(Um,x,yi,zi)
    @inbounds @inline Uy(x) = getindex(Um,xi,x,zi)
    @inbounds @inline Uz(x) = getindex(Um,xi,yi,x)
    dx,dy,dz,_ = ds

    D4(Ux,(nls[1],nrs[1]),(αls[1],αrs[1]),xi)/dx^2 + D4(Uy,(nls[2],nrs[2]),(αls[2],αrs[2]),yi)/dy^2 + D4(Uz,(nls[3],nrs[3]),(αls[3],αrs[3]),zi)/dz^2
end

Base.@propagate_inbounds @inline function Div(vx,vy,vz,U,ns,nls,nrs,αls,αrs,ds,i,j,k) # Calculate the divergence of the flux
    dx,dy,dz,_ = ds
    (Dx(vx,U,ns,nls,nrs,αls,αrs,i,j,k)/dx + Dy(vy,U,ns,nls,nrs,αls,αrs,i,j,k)/dy + Dz(vz,U,ns,nls,nrs,αls,αrs,i,j,k)/dz)/rootγ(U[i,j,k])
end

@inline function vectors(U,outer,ns,ri,rb,i,l)
    # Returns the null basis to the boundary
    # The embedded boundary method effectively treats the boundary as "Lego" objects
    # with boundary normals only in one of the 3 cardinal directions.
    # Retruns upper index vectors and a mixed index boundary 2-metric

    xi,yi,zi= ri
    #nx,ny,nz = ns

    g = U.g

    sxn,syn,szn = (0.,0.,0.)
    sx,sy,sz = (0.,0.,0.)


    if     i==1
        (sxn = l)
    elseif i==2 
        (syn = l)
    elseif i==3
        (szn = l)
    end

    # x,y,z = rb
    # sx = -x
    # sy = -y
    # sz = -z

    if outer
        # if     i==1
        #     (sx = l)
        # elseif i==2 
        #     (sy = l)
        # elseif i==3
        #     (sz = l)
        # end

        if xi==1; (sx = -1.) elseif xi==ns[1]; (sx = 1.) end
        if yi==1; (sy = -1.) elseif yi==ns[2]; (sy = 1.) end
        if zi==1; (sz = -1.) elseif zi==ns[3]; (sz = 1.) end

    else
        x,y,z = rb
        sx = -x
        sy = -y
        sz = -z
    end

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

    # s = FourVector((0.,sx,sy,sz))

    # snorm = @einsum g[μ,ν]*s[μ]*s[ν]
    
    # s = s/sqrt(snorm) 

    st_ = FourVector((βx*sx+βy*sy+βz*sz,sx,sy,sz))

    snorm = @einsum gi[μ,ν]*st_[μ]*st_[ν]
    
    s_ = st_/sqrt(snorm) 

    s = @einsum gi[μ,ν]*s_[ν]

    ℓ = (n + s)/sqrt(2)
    k = (n - s)/sqrt(2)

    σ = StateTensor((μ,ν) -> gi[μ,ν] + k[μ]*ℓ[ν] + ℓ[μ]*k[ν]) # all indices up

    #return (k,ℓ,σ,kn,ℓn,σn) # characteristic LEGO ball
    #return (kn,ℓn,σn,kn,ℓn,σn) # Complete LEGO ball
    return (k,ℓ,σ) # spherical


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

    #R = 15.

    p = 2

    f(x,y,z) = x^p + y^p + z^p - R^p # Definitions of the boundary position

    #f(x,y,z) = x + 200

    xibl =  1; yibl =  1; zibl =  1;
    xibr = nx; yibr = ny; zibr = nz;

    # αlx = 0.5; αly = 0.5; αlz = 0.5;
    # αrx = 0.5; αry = 0.5; αrz = 0.5;

    αlx = 0.; αly = 0.; αlz = 0.;
    αrx = 0.; αry = 0.; αrz = 0.;

    s = sign(f(x,y,z))

    a = 1/1000000

    # x-line
    i = xi+3
    x0 = x+3*dx
    if (i <= nx && sign(f(x0,y,z))≠s)
        rb = find_zero(x->f(x,y,z), (x,x0), atol=a*dx, rtol=a*dx, Bisection())+lx/2
        #rb = find_zero(x->f(x,y,z), SecantMethod{Float64}(x,x0), CompactSolution()).root+lx/2
        temp,αrx = divrem(rb,dx,RoundDown)
        αrx /= dx; xibr=Int(temp)+1;
    end

    i = xi-3
    x0 = x-3*dx
    if (i >= 1 && sign(f(x0,y,z))≠s)
        rb = find_zero(x->f(x,y,z), (x0,x), atol=a*dx, rtol=a*dx, Bisection())+lx/2
        #rb = find_zero(x->f(x,y,z), SecantMethod{Float64}(x0,x), CompactSolution()).root+lx/2
        temp,αlx = divrem(rb,dx,RoundUp)
        αlx /= -dx; xibl=Int(temp)+1;
    end

    # y-line
    i = yi+3
    y0 = y+3*dy
    if (i <= ny && sign(f(x,y0,z))≠s)
        rb = find_zero(y->f(x,y,z), (y,y0), atol=a*dy, rtol=a*dy, Bisection())+ly/2
        #rb = find_zero(y->f(x,y,z), SecantMethod{Float64}(y,y0), CompactSolution()).root+ly/2
        temp,αry = divrem(rb,dy,RoundDown)
        αry /= dy; yibr=Int(temp)+1;
    end

    i = yi-3
    y0 = y-3*dy
    if (i >= 1 && sign(f(x,y0,z))≠s)
        rb = find_zero(y->f(x,y,z), (y0,y), atol=a*dy, rtol=a*dy, Bisection())+ly/2
        #rb = find_zero(y->f(x,y,z), SecantMethod{Float64}(y0,y), CompactSolution()).root+ly/2
        temp,αly = divrem(rb,dy,RoundUp)
        αly /= -dy; yibl=Int(temp)+1;
    end

    # z-line
    i = zi+3
    z0 = z+3*dz
    if (i <= nz && sign(f(x,y,z0))≠s)
        rb = find_zero(z->f(x,y,z), (z,z0), atol=a*dz, rtol=a*dz, Bisection())+lz/2
        #rb = find_zero(z->f(x,y,z), SecantMethod{Float64}(z,z0), CompactSolution()).root+lz/2
        temp,αrz = divrem(rb,dz,RoundDown)
        αrz /= dz; zibr=Int(temp)+1;
    end

    i = zi-3
    z0 = z-3*dz
    if (i >= 1 && sign(f(x,y,z0))≠s)
        rb = find_zero(z->f(x,y,z), (z0,z), atol=a*dz, rtol=a*dz, Bisection())+lz/2
        #rb = find_zero(z->f(x,y,z), SecantMethod{Float64}(z0,z), CompactSolution()).root+lz/2
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

@inline function BoundaryConditions(Ub,outer,rb,C_BC,fGW,k,ℓ,σ)

    xb,yb,zb = rb

    g  = Ub.g
    dx = Ub.dx
    dy = Ub.dy
    dz = Ub.dz
    P  = Ub.P

    #H_ = FourVector((0.,0.,0.,0.))
    H_ = FourVector(μ -> fH_(xb,yb,zb,μ))

    ℓ_ = @einsum g[μ,α]*ℓ[α]
    k_ = @einsum g[μ,α]*k[α]

    # ℓn_ = @einsum g[μ,α]*ℓn[α]
    # kn_ = @einsum g[μ,α]*kn[α]

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
    U0b = @einsum  σm[γ,β]*∂gb[β,μ,ν]
    Um = (@einsum ℓ[α]*∂gb[α,μ,ν])# + γ2*g/sqrt(2)

    s_ = (ℓ_ - k_)/sqrt(2)

    β = FourVector((0.,βx,βy,βz))

    βs = @einsum β[μ]*s_[μ]

    βU0 = @einsum β[σ]*U0b[σ,μ,ν]

    cp    =  α - βs
    cm    = -α - βs
    cperp = -βs

    #Umfreeze(x) = (cp/cm)*Upb + (sqrt(2)/cm)*βU0 - sqrt(2)*ones(StateTensor)*x/cm
    #Umfreeze(x) = -Upb - sqrt(2)*ones(StateTensor)*x 
    Dirichlet(x) = -Upb - sqrt(2)*ones(StateTensor)*x
    Neumann(x)   =  Upb - sqrt(2)*ones(StateTensor)*x
    #UmBC1 = ZeroST
    if outer
        UmBC1  = StateTensor((μ,ν)->fUmBC(ℓ,xb,yb,zb,μ,ν)) #Dirichlet(0.)
        UmBCGW = fGW*StateTensor((μ,ν)->fUmBC(ℓ,xb,yb,zb,μ,ν))

        # UmBC1  = Neumann(0.)
        # UmBCGW = Neumann(0.)
    else
        # UmBC1  = Dirichlet(0.)
        # UmBCGW = Dirichlet(0.)

        UmBC1  = Neumann(0.)
        UmBCGW = Neumann(0.)

        # UmBC1  = StateTensor((μ,ν)->fUmBC(ℓ,xb,yb,zb,μ,ν)) #Dirichlet(0.)
        # UmBCGW = StateTensor((μ,ν)->fUmBC(ℓ,xb,yb,zb,μ,ν))
    end

    # UmBC1  = Umfreeze(0.)
    # UmBCGW = Umfreeze(0.)
    #UmBC1 = Umfreeze(0.)

    #UmBCGW = Umfreeze(0.) #Umfreeze(fGW)

    # Three index constraint projector (indices up down down)
    Q3 = Symmetric3rdOrderTensor((α,μ,ν) -> 0.5*(ℓ_[μ]*σm[ν,α] + ℓ_[ν]*σm[μ,α] - σ_[μ,ν]*ℓ[α] - ℓ_[μ]*ℓ_[ν]*k[α])) 

    # Gravitational wave projector (indices down down up up)
    O = Projection((μ,ν,α,β) -> σm[μ,α]*σm[ν,β] - 0.5*σ_[μ,ν]*σ[α,β]) 

    # index down
    A_ = @einsum (2*ℓ[μ]*Upb[μ,α] - gi[μ,ν]*Upb[μ,ν]*ℓ_[α] - 2*gi[μ,ν]*U0b[μ,ν,α] + gi[μ,ν]*U0b[α,μ,ν] + 2*C_BC[α] + 2*H_[α])# 

    G = minorsymmetric(NSProjection((μ,ν,α,β) -> (2k_[μ]*ℓ_[ν]*k[α]*ℓ[β] - 2k_[μ]*σm[ν,α]*ℓ[β] + k_[μ]*k_[ν]*ℓ[α]*ℓ[β])))

    UmBC = (@einsum Q3[α,μ,ν]*A_[α] + O[μ,ν,α,β]*UmBCGW[α,β] + G[μ,ν,α,β]*UmBC1[α,β])# + γ2*g/sqrt(2)

    # UmBC = UmBC1

    PBC  = -(Upb + UmBC)/sqrt(2)
    ∂gBC =  Symmetric3rdOrderTensor((α,μ,ν) -> U0b[α,μ,ν] - k_[α]*UmBC[μ,ν] - ℓ_[α]*Upb[μ,ν]) #Symmetric3rdOrderTensor

    return (StateVector(ZeroST,∂gBC[2,:,:],∂gBC[3,:,:],∂gBC[4,:,:],PBC),cm)

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

    # x = -lx/2 + (xi-1)*hx; y = -ly/2 + (yi-1)*hy; z = -lz/2 + (zi-1)*hz;

    SATg = ZeroST; SATdx = ZeroST; SATdy = ZeroST; SATdz = ZeroST; SATP = ZeroST; 

    for i in 1:3

        nl = nls[i]; nr = nrs[i];

        # if nr-nl<3 
        #     #(println(nls,nrs); @assert false) 
        #     throw(ZeroSV)
        # end

        if (ri[i]==nl==1) || (nl ≠ 1 && ri[i] in nl:nl+1) #ri[i] in nl:nl+1 # in the boundary region on the left side of the line

            let 
            # I want to use the same variable names here in the left branch 
            # as in the right branch, so there is a let block to allow this
            # and prevent type instability

                αl = αls[i]

                #xb,yb,zb = x,y,z

                # Interpolate the solution vector on the boundary 
                # and determine the boundary position on the coordinate line
                if i == 1 # On an x-line
                    xb = -lx/2+(nl-1)*hx-αl*hx
                    yb = y
                    zb = z
                    Ub = el2_1(αl)*U[nl,yi,zi] + el2_2(αl)*U[nl+1,yi,zi]
                elseif i == 2 # On a y-line
                    yb = -ly/2+(nl-1)*hy-αl*hy
                    xb = x
                    zb = z
                    Ub = el2_1(αl)*U[xi,nl,zi] + el2_2(αl)*U[xi,nl+1,zi]
                elseif i == 3 # On a z-line
                    zb = -lz/2+(nl-1)*hz-αl*hz
                    xb = x
                    yb = y
                    Ub = el2_1(αl)*U[xi,yi,nl] + el2_2(αl)*U[xi,yi,nl+1]
                end

                dx = Ub.dx
                dy = Ub.dy
                dz = Ub.dz
                P  = Ub.P

                rb = (xb,yb,zb)

                outer =  (nl == 1)

                Amp = 0.2

                σ0 = 2.5
                μ0 = -55

                σ1 = 20

                σx = 10.
                σy = 40.
                σz = 40.
                r0 = -60.

                rp = sqrt(((xb-(r0+t))/σx)^2+(yb/σy)^2+(zb/σz)^2)

                model =  (outer&&(rp<1)) ? Amp*(rp-1)^4*(rp+1)^4 : 0.

                # model = (outer&&((μ0+t-σ0)<xb<(μ0+t+σ0))&&-σ1<yb<σ1&&-σ1<zb<σ1) ? (Amp/(σ1^16*σ0^8))*(yb-σ1)^4*(yb+σ1)^4*(zb-σ1)^4*(zb+σ1)^4*(xb-(μ0+t-σ0))^4*(xb-(μ0+t+σ0))^4 : 0.

                #C_BC = model*FourVector((0.,0.,0.,0.))
                #C_BC = model*FourVector((1.,0.,0.,0.))

                #UmBCGW = zero(StateTensor)
                fGW =  500*model + 1


                k,ℓ,σ = vectors(Ub,outer,ns,ri,rb,i,-1) # Form boundary basis

                C_BC = FourVector((0.,0.,0.,0.))#model*FourVector((1.,0.,0.,0.))

                UBC,cm = BoundaryConditions(Ub,outer,rb,C_BC,fGW,k,ℓ,σ)

                dxBC = UBC.dx
                dyBC = UBC.dy
                dzBC = UBC.dz
                PBC  = UBC.P

                #cm = 1.

                if ri[i] == nl 
                    ε = abs(cm)*el2_1(αl)/h2_11(αl)/ds[i]
                elseif ri[i] == nl+1
                    ε = abs(cm)*el2_2(αl)/h2_22(αl)/ds[i]
                end

                SATP  += ε*(PBC - P)
                SATdx += ε*(dxBC - dx)
                SATdy += ε*(dyBC - dy)
                SATdz += ε*(dzBC - dz)

            end

        end

        if (ri[i]==nr==ns[i]) || (nr ≠ ns[i] && ri[i] in (nr-1):nr) #ri[i] in (nr-1):nr in the boundary region on the right side of the line

            let
            # I want to use the same variable names here in the right branch 
            # as in the left branch before, so there is a let block to allow this
            # and prevent type instability

                αr = αrs[i]


                # Interpolate the solution vector on the boundary 
                # and determine the boundary position on the coordinate line
                if i == 1 # On an x-line
                    rb = (-lx/2+(nr-1)*hx+αr*hx,y,z)
                    # rb = (x,y,z)
                    xb,yb,zb = rb
                    Ub = el2_1(αr)*U[nr,yi,zi] + el2_2(αr)*U[nr-1,yi,zi]
                elseif i == 2 # On a y-line
                    rb = (x,-ly/2+(nr-1)*hy+αr*hy,z)
                    # xb,yb,zb = rb
                    # rb = (x,y,z)
                    xb,yb,zb = rb
                    Ub = el2_1(αr)*U[xi,nr,zi] + el2_2(αr)*U[xi,nr-1,zi]
                else#if i == 3 # On a z-line
                    rb = (x,y,-lz/2+(nr-1)*hz+αr*hz)
                    xb,yb,zb = rb
                    # rb = (x,y,z)
                    Ub = el2_1(αr)*U[xi,yi,nr] + el2_2(αr)*U[xi,yi,nr-1]
                end

                dx = Ub.dx
                dy = Ub.dy
                dz = Ub.dz
                P  = Ub.P
        
                outer =  (nr == ns[i])
        
                #s_ = (ℓ_ - k_)/sqrt(2)

                #β = FourVector((0.,βx,βy,βz))

                #βs = @einsum β[μ]*s_[μ]

                # βU0 = @einsum β[σ]*U0b[σ,μ,ν]

                # cp    =  α - βs
                # cm    = -α - βs
                # cperp = -βs

                #UmBC1 = (cp/cm)*Upb + (sqrt(2)/cm)*βU0# + γ2*(1-cp/cm)*g/sqrt(2)
                #UmBC1 = ZeroST
                #UmBC1 = StateTensor((μ,ν)->UBC(ℓn,xb,yb,zb,μ,ν))

                Amp = 0.2

                σ0 = 2.5
                μ0 = 55

                σ1 = 20.

                σx = 10.
                σy = 40.
                σz = 40.
                r0 = 60.

                rp = sqrt(((xb-(r0-t))/σx)^2+(yb/σy)^2+(zb/σz)^2)

                model =  (outer&&(rp<1)) ? Amp*(rp-1)^4*(rp+1)^4 : 0.

                fGW =  500*model + 1

                k,ℓ,σ = vectors(Ub,outer,ns,ri,rb,i,1.) # Form boundary basis

                C_BC = FourVector((0.,0.,0.,0.))#model*FourVector((1.,0.,0.,0.))

                UBC,cm = BoundaryConditions(Ub,outer,rb,C_BC,fGW,k,ℓ,σ)

                dxBC = UBC.dx
                dyBC = UBC.dy
                dzBC = UBC.dz
                PBC  = UBC.P

                #cm = 1.

                if ri[i] == nr
                    ε = abs(cm)*el2_1(αr)/h2_11(αr)/ds[i]
                elseif ri[i] == nr-1
                    ε = abs(cm)*el2_2(αr)/h2_22(αr)/ds[i]
                end
        
                SATP  += ε*(PBC - P)
                SATdx += ε*(dxBC - dx)
                SATdy += ε*(dyBC - dy)
                SATdz += ε*(dzBC - dz)

            end

        end
        
    end

    return StateVector(SATg,SATdx,SATdy,SATdz,SATP)

end

function constraints(U::StateVector,ns,ds,ls,ri)

    g  = U.g
    dx = U.dx
    dy = U.dy
    dz = U.dz
    P  = U.P

    xi,yi,zi = ri

    hx,hy,hz,_ = ds
    lx,ly,lz=ls
    x = -lx/2 + (xi-1)*hx; y = -ly/2 + (yi-1)*hy; z = -lz/2 + (zi-1)*hz;

    H_ = FourVector(μ -> fH_(x,y,z,μ))

    r = (x,y,z)

    in_domain,_,_,_,_ = find_boundary(ns,ls,ds,r,ri)

    if in_domain

        gi = inv(g)

        α = 1/sqrt(-gi[1,1])

        βx = -gi[1,2]/gi[1,1]
        βy = -gi[1,3]/gi[1,1]
        βz = -gi[1,4]/gi[1,1]

        # Calculate time derivative of the metric
        ∂tg = βx*dx + βy*dy + βz*dz - α*P

        ∂g = Symmetric3rdOrderTensor((σ,μ,ν) -> (σ==1 ? ∂tg[μ,ν] : σ==2 ? dx[μ,ν] : σ==3 ? dy[μ,ν] : σ==4 ? dz[μ,ν] : throw(ZeroST)))

        Γ  = Symmetric3rdOrderTensor((σ,μ,ν) -> 0.5*(∂g[ν,μ,σ] + ∂g[μ,ν,σ] - ∂g[σ,μ,ν]))    

        C_ = @einsum gi[ϵ,σ]*Γ[μ,ϵ,σ] - H_[μ]

    else

        C_ = FourVector((0.,0.,0.,0.))

    end

    return C_

end

function constraint_energy_cell(U,ns,ls,ds,ri)
    # Calculates the energy estimate (not yet implemented)

    g  = U.g
    # dx = U.dx
    # dy = U.dy
    # dz = U.dz
    # P  = U.P

    xi,yi,zi = ri

    dx,dy,dz,_ = ds
    lx,ly,lz=ls
    x = -lx/2 + (xi-1)*dx; y = -ly/2 + (yi-1)*dy; z = -lz/2 + (zi-1)*dz;
    r = (x,y,z)

    in_domain,nls,nrs,αls,αrs = find_boundary(ns,ls,ds,r,ri)

    if in_domain

        gi = inv(g)

        α = 1/sqrt(-gi[1,1])

        βx = -gi[1,2]/gi[1,1]
        βy = -gi[1,3]/gi[1,1]
        βz = -gi[1,4]/gi[1,1]

        n  = FourVector((1,-βx,-βy,-βz))/α

        m = gi + 2*symmetric(@einsum n[μ]*n[ν])

        C_ = constraints(U,ns,ds,ls,ri)

        h = 1.

        for i in 1:3
            if ri[i] == nrs[i]
                h *= h2_11(αrs[i])
            elseif ri[i] == nrs[i]-1
                h *= h2_22(αrs[i])
            elseif ri[i] == nls[i]
                h *= h2_11(αls[i])
            elseif ri[i] == nls[i]+1
                h *= h2_22(αls[i])
            end
        end

        return abs(@einsum m[μ,ν]*C_[μ]*C_[ν])*h*rootγ(U)*dx*dy*dz

    else

        return 0.

    end

end

function invariant(U,∂tU,ns,ls,ds,ri)
    # Calculates the Kretschmann Scalar

    hx,hy,hz,_ = ds
    lx,ly,lz=ls
    x = -lx/2 + (xi-1)*dx; y = -ly/2 + (yi-1)*dy; z = -lz/2 + (zi-1)*dz;
    r = (x,y,z)

    in_domain,nls,nrs,αls,αrs = find_boundary(ns,ls,ds,r,ri)

    xi,yi,zi = ri

    if in_domain

        g  = U.g
        dx = U.dx
        dy = U.dy
        dz = U.dz
        P  = U.P
    
        # Second order spatial derivatives
        ∂xdx = Dx(fdx,U,ns,nls,nrs,αls,αrs,xi,yi,zi)/hx
        ∂xdy = Dx(fdy,U,ns,nls,nrs,αls,αrs,xi,yi,zi)/hx
        ∂xdz = Dx(fdz,U,ns,nls,nrs,αls,αrs,xi,yi,zi)/hx
    
        #∂ydx = Dy(fdx,U,ns,nls,nrs,αls,αrs,xi,yi,zi)/hy
        ∂ydy = Dy(fdy,U,ns,nls,nrs,αls,αrs,xi,yi,zi)/hy
        ∂ydz = Dy(fdz,U,ns,nls,nrs,αls,αrs,xi,yi,zi)/hy
    
        #∂zdx = Dz(fdx,U,ns,nls,nrs,αls,αrs,xi,yi,zi)/hz
        #∂zdy = Dz(fdy,U,ns,nls,nrs,αls,αrs,xi,yi,zi)/hz
        ∂zdz = Dz(fdz,U,ns,nls,nrs,αls,αrs,xi,yi,zi)/hz
    
        # Second order temporal derivatives
        ∂tg  = ∂tU.g
        ∂tdx = ∂tU.dx
        ∂tdy = ∂tU.dy
        ∂tdz = ∂tU.dz
        ∂tP  = ∂tU.P

        gi = inv(g)
        
        α = 1/sqrt(-gi[1,1]) # Calculate lapse
    
        βx = -gi[1,2]/gi[1,1] # Calculate shift vector
        βy = -gi[1,3]/gi[1,1]
        βz = -gi[1,4]/gi[1,1]

        n = FourVector((1,-βx,-βy,-βz))/α

        γi = gi + symetric(@einsum n[μ]*n[ν])

        ∂tα = -0.5*α*(@einsum n[μ]*n[ν]*∂tg[μ,ν])

        ∂tβ = α*(@einsum γi[σ,μ]*n[ν]*∂tg[μ,ν])

        ∂t∂tg = βx*∂tdx + βy*∂tdy + βz*∂tdz - α*∂tP + ∂tβ[2]*dx + ∂tβ[3]*dy + ∂tβ[4]*dz - ∂tα*P

        # collect metric derivatives
        ∂g = Symmetric3rdOrderTensor((σ,μ,ν) -> (σ==1 ? ∂tg[μ,ν] : σ==2 ? dx[μ,ν] : σ==3 ? dy[μ,ν] : σ==4 ? dz[μ,ν] : throw(ZeroST)))
        
        # Assumes derivatives commute, so we can use the symmetric 4th order type
        ∂∂g = Projection((α,β,μ,ν)-> ((∂t∂tg[μ,ν] , ∂tdx[μ,ν], ∂tdy[μ,ν], ∂tdz[μ,ν] )[β],     #t
                                      ( ∂tdx[μ,ν] , ∂xdx[μ,ν], ∂xdy[μ,ν], ∂xdz[μ,ν] )[β],     #x
                                      ( ∂tdy[μ,ν] , ∂xdy[μ,ν], ∂ydy[μ,ν], ∂ydz[μ,ν] )[β],     #y
                                      ( ∂tdz[μ,ν] , ∂xdz[μ,ν], ∂ydz[μ,ν], ∂zdz[μ,ν] )[β])[α]) #z

        Γ  = Symmetric3rdOrderTensor((σ,μ,ν) -> 0.5*(∂g[ν,μ,σ] + ∂g[μ,ν,σ] - ∂g[σ,μ,ν]))

        ∂Γ = Projection((α,β,μ,ν) -> 0.5*(∂∂g[α,ν,μ,β] + ∂∂g[α,μ,ν,β] - ∂∂g[α,β,μ,ν]))

        Rt = @einsum (α,β,μ,ν) -> ∂Γ[μ,β,ν,α] + gi[σ,ϵ]*Γ[α,ν,σ]*Γ[ϵ,β,μ]

        R  = @einsum (α,β,μ,ν) -> Rt[α,β,μ,ν] - Rt[α,β,ν,μ]  # Reimann Tensor (all lower indices)

        I = @einsum gi[α,β]*gi[μ,ν]*gi[σ,ϵ]*gi[λ,δ]*R[α,μ,σ,λ]*R[β,ν,ϵ,δ]

        return I

    else

        return 0.

    end

end

function Hawking_Mass_cell(Ub,rb,R,face)
    # Calculates the Hawking Mass at one point on a surface

    x,y,z = rb

    g  = Ub.g
    dx = Ub.dx
    dy = Ub.dy
    dz = Ub.dz
    P  = Ub.P

    gi = inv(g)

    α = 1/sqrt(-gi[1,1]) # Calculate lapse

    βx = -gi[1,2]/gi[1,1] # Calculate shift vector
    βy = -gi[1,3]/gi[1,1]
    βz = -gi[1,4]/gi[1,1]

    ∂tg = βx*dx + βy*dy + βz*dz - α*P

    # collect metric derivatives
    ∂g = Symmetric3rdOrderTensor((σ,μ,ν) -> (∂tg[μ,ν],dx[μ,ν],dy[μ,ν],dz[μ,ν])[σ])

    β = ThreeVector((βx,βy,βz))

    γ   = ThreeTensor( g.data[5:end])
    dxγ = ThreeTensor(dx.data[5:end])
    dyγ = ThreeTensor(dy.data[5:end])
    dzγ = ThreeTensor(dz.data[5:end])

    γi = inv(γ)

    # collect 3-metric derivatives
    ∂γ = Symmetric3rdOrderTensor3((k,i,j) -> (dxγ[i,j],dyγ[i,j],dzγ[i,j])[k])

    ∂tγ = ThreeTensor(∂tg.data[5:end])

    # Calculate Christoffel symbols
    Γ  = Symmetric3rdOrderTensor3((k,i,j) -> 0.5*(∂γ[j,i,k] + ∂γ[i,j,k] - ∂γ[k,i,j]))

    ∂∂φ = -2*one(ThreeTensor)

    ∂φ = -2*ThreeVector((x,y,z))

    ϕ = sqrt(@einsum γi[i,j]*∂φ[i]*∂φ[j])

    s_ = ∂φ/ϕ

    s = @einsum γi[i,j]*s_[j] 

    σ = @einsum γi[i,j] - s[i]*s[j]

    σ_ = @einsum γ[i,j]*γ[k,l]*σ[j,l]

    ϵ = 0.

    if     face == 1
        rp = -sqrt(R^2-y^2-z^2)
        e = SurfaceProjector((-ϵ*y/rp,1,0,-ϵ*z/rp,0,1))
    elseif face == 2
        rp = sqrt(R^2-y^2-z^2)
        e = SurfaceProjector((-ϵ*y/rp,1,0,-ϵ*z/rp,0,1))
    elseif face == 3
        rp = -sqrt(R^2-x^2-z^2)
        e = SurfaceProjector((1,-ϵ*x/rp,0,0,-ϵ*z/rp,1))
    elseif face == 4
        rp = sqrt(R^2-x^2-z^2)
        e = SurfaceProjector((1,-ϵ*x/rp,0,0,-ϵ*z/rp,1))
    elseif face == 5
        rp = -sqrt(R^2-x^2-y^2)
        e = SurfaceProjector((1,0,-ϵ*x/rp,0,1,-ϵ*y/rp))
    elseif face == 6
        rp = sqrt(R^2-x^2-y^2)
        e = SurfaceProjector((1,0,-ϵ*x/rp,0,1,-ϵ*y/rp))
    end

    σ_AB = @einsum σ_[i,j]*e[i,A]*e[j,B]

    # Calculate the divergence of s 

    Ds =  (@einsum σ[i,j]*∂∂φ[i,j])/ϕ 
    Ds -= (@einsum  γi[k,l]*∂φ[k]*σ[i,j]*Γ[l,i,j])/ϕ

    # Calculate the extrinsic curvature

    ∂βsym = ThreeTensor((i,j)->∂g[i+1,j+1,1]+∂g[j+1,i+1,1])

    Dβsym = @einsum ∂βsym[i,j] - 2*β[k]*Γ[k,i,j]

    Kij = -(∂tγ - Dβsym)/(2*α)

    σK = @einsum σ[i,j]*Kij[i,j]

    return (σK^2 - Ds^2)*sqrt(det(σ_AB))
    
end

function Area(Ub,rb,R,face)
    # Calculates the Surface Area element at one point on a surface

    x,y,z = rb

    g  = Ub.g

    γ  = ThreeTensor(g.data[5:end])

    γi = inv(γ)

    ∂φ = -2*ThreeVector((x,y,z))

    ϕ = sqrt(@einsum γi[i,j]*∂φ[i]*∂φ[j])

    s_ = ∂φ/ϕ

    σ_ = @einsum γ[i,j] - s_[i]*s_[j]

    ϵ = 0.

    if     face == 1
        rp = -sqrt(R^2-y^2-z^2)
        e = SurfaceProjector((-ϵ*y/rp,1,0,-ϵ*z/rp,0,1))
    elseif face == 2
        rp = sqrt(R^2-y^2-z^2)
        e = SurfaceProjector((-ϵ*y/rp,1,0,-ϵ*z/rp,0,1))
    elseif face == 3
        rp = -sqrt(R^2-x^2-z^2)
        e = SurfaceProjector((1,-ϵ*x/rp,0,0,-ϵ*z/rp,1))
    elseif face == 4
        rp = sqrt(R^2-x^2-z^2)
        e = SurfaceProjector((1,-ϵ*x/rp,0,0,-ϵ*z/rp,1))
    elseif face == 5
        rp = -sqrt(R^2-x^2-y^2)
        e = SurfaceProjector((1,0,-ϵ*x/rp,0,1,-ϵ*y/rp))
    elseif face == 6
        rp = sqrt(R^2-x^2-y^2)
        e = SurfaceProjector((1,0,-ϵ*x/rp,0,1,-ϵ*y/rp))
    end

    σ_AB = @einsum σ_[i,j]*e[i,A]*e[j,B]

    return sqrt(det(σ_AB))
    
end

@inline function SurfaceIntegral(f,U,ns,ds,ls,R)

    nx,ny,nz = ns
    hx,hy,hz,_ = ds
    lx,ly,lz = ls

    result = 0.

    # face 1 is -x, face 2 is +x etc...
    face_to_index(face,i1,i2) = ((1,i1,i2),(nx,i1,i2),(i1,1,i2),(i1,ny,i2),(i1,i2,1),(i1,i2,nz))[face] 

    for face in 1:6

        #R = lx/4.

        f3D(x1,x2,x3) = x1^2 + x2^2 + x3^2 - R^2

        f2D(x1,x2) = x1^2 + x2^2 - R^2

        f1D(x1) = x1^2 - R^2

        bounds = ((ny,nz),(ny,nz),(nx,nz),(nx,nz),(nx,ny),(nx,ny))[face]

        for i1 in 1:bounds[1], i2 in 1:bounds[2]

            ri = face_to_index(face,i1,i2)
            xi,yi,zi = ri

            x = -lx/2 + (xi-1)*hx; y = -ly/2 + (yi-1)*hy; z = -lz/2 + (zi-1)*hz;
            r = (x,y,z)

            pos    = ((y,z),(y,z),(x,z),(x,z),(x,y),(x,y))[face]
            index  = ((2,3),(2,3),(1,3),(1,3),(1,2),(1,2))[face]

            if (sign(f2D(pos...))!=-1) continue end

            #a = 1/1000000

            rb1 = -sqrt(R^2-pos[1]^2-pos[2]^2) + lx/2
            temp,αr0 = divrem(rb1,hx,RoundDown)
            αr0 /= hx; nr0=Int(temp)+1;
            
            rb1 = sqrt(R^2-pos[1]^2-pos[2]^2) + lx/2
            temp,αl0 = divrem(rb1,hx,RoundUp)
            αl0 /= -hx; nl0=Int(temp)+1;

            if     face == 1
                rb = (-lx/2+(nr0-1)*hx+αr0*hx,y,z)
                Ub = el2_1(αr0)*U[nr0,yi,zi] + el2_2(αr0)*U[nr0-1,yi,zi]
            elseif face == 2
                rb = (-lx/2+(nl0-1)*hx-αl0*hx,y,z)
                Ub = el2_1(αl0)*U[nl0,yi,zi] + el2_2(αl0)*U[nl0+1,yi,zi]
            elseif face == 3
                rb = (x,-ly/2+(nr0-1)*hy+αr0*hy,z)
                Ub = el2_1(αr0)*U[xi,nr0,zi] + el2_2(αr0)*U[xi,nr0-1,zi]
            elseif face == 4
                rb = (x,-ly/2+(nl0-1)*hy-αl0*hy,z)
                Ub = el2_1(αl0)*U[xi,nl0,zi] + el2_2(αl0)*U[xi,nl0+1,zi]
            elseif face == 5
                rb = (x,y,-lz/2+(nr0-1)*hz+αr0*hz)
                Ub = el2_1(αr0)*U[xi,yi,nr0] + el2_2(αr0)*U[xi,yi,nr0-1]
            elseif face == 6
                rb = (x,y,-lz/2+(nl0-1)*hz-αl0*hz)
                Ub = el2_1(αl0)*U[xi,yi,nl0] + el2_2(αl0)*U[xi,yi,nl0+1] 
            end

            rb1 = -sqrt(R^2-pos[1]^2) + lx/2
            temp,αr1 = divrem(rb1,hx,RoundDown)
            αr1 /= hx; nr1=Int(temp)+1;
            
            rb1 = sqrt(R^2-pos[1]^2) + lx/2
            temp,αl1 = divrem(rb1,hx,RoundUp)
            αl1 /= -hx; nl1=Int(temp)+1;

            #if !(nl1<=i1<=nr1); continue end

            h = 1.

            h *= H(nl1,nr1,αl1,αr1,i1)

            rb1 = -R + lx/2
            temp,αr2 = divrem(rb1,hx,RoundDown)
            αr2 /= hx; nr2=Int(temp)+1;
            
            rb1 = R + lx/2
            temp,αl2 = divrem(rb1,hx,RoundUp)
            αl2 /= -hx; nl2=Int(temp)+1;

            #if !(nl2<=i2<=nr2); continue end

            h *= H(nl2,nr2,αl2,αr2,i2)
        
            result += f(Ub,rb,R,face)*h*ds[index[1]]*ds[index[2]]

            #result += 1*h*ds[index[1]]*ds[index[2]]
        end
    end

    return result

end

@parallel_indices (xi,yi,zi) function rhs!(U1,U2,U3,ns::Tuple,ls::Tuple,ds::Tuple,t::Data.Number,iter::Int)
    # Performs the right hand side of the system of equations.
    # The if blocks at the beginning and end perform the 3rd order Runge-Kutta algorithm
    # which is done by calling rhs! three times, each with a different value in iter ranging from 1:3

    #xi,yi,zi = 1,1,1

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

        H_ = FourVector(μ -> fH_(x,y,z,μ))

        #∂H_ = FourTensor((μ,ν) -> f∂H_(x,y,z,μ,ν))
        ∂H_ = StateTensor((μ,ν) -> 0.5*(f∂H_(x,y,z,μ,ν)+f∂H_(x,y,z,ν,μ)))

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

        n  = FourVector((1,-βx,-βy,-βz))/α
        n_ = FourVector((-α,0.,0.,0.))

        γi = gi + symmetric(@einsum n[μ]*n[ν])

        ∂tg = βx*dx + βy*dy + βz*dz - α*P

        # collect metric derivatives
        ∂g = Symmetric3rdOrderTensor((σ,μ,ν) -> (∂tg[μ,ν],dx[μ,ν],dy[μ,ν],dz[μ,ν])[σ])

        # Calculate Christoffel symbols
        Γ  = Symmetric3rdOrderTensor((σ,μ,ν) -> 0.5*(∂g[ν,μ,σ] + ∂g[μ,ν,σ] - ∂g[σ,μ,ν]))

        ∂tlnrootγ = @einsum 0.5*γi[μ,ν]*∂tg[μ,ν]

        Γ_ = @einsum gi[ϵ,σ]*Γ[μ,ϵ,σ]

        #########################################################
        # Principle (linear) part of the evolution equations

        #try

        ∂tdx = Dx(u,U,ns,nls,nrs,αls,αrs,xi,yi,zi)/hx

        ∂tdy = Dy(u,U,ns,nls,nrs,αls,αrs,xi,yi,zi)/hy

        ∂tdz = Dz(u,U,ns,nls,nrs,αls,αrs,xi,yi,zi)/hz

        ∂tP  = Div(vx,vy,vz,U,ns,nls,nrs,αls,αrs,ds,xi,yi,zi)

        # catch e
        #     println(ri," ",nls," ",nrs)
        #     throw(e)
        # end

        #########################################################
        # Non-linear terms in the evolution equations (34%)

        ∂tP -= ∂tlnrootγ*P

        ∂tP += 2*α*∂H_
    
        ∂tP -= 2*α*symmetric(@einsum (μ,ν) -> gi[ϵ,σ]*H_[ϵ]*Γ[σ,μ,ν])

        ∂tP -=   α*symmetric(@einsum (μ,ν) -> gi[λ,γ]*Γ_[λ]*∂g[γ,μ,ν])

        ∂tP += 2*α*symmetric(@einsum (μ,ν) -> gi[λ,ρ]*gi[ϵ,σ]*∂g[λ,ϵ,μ]*∂g[ρ,σ,ν])
    
        ∂tP -= 2*α*symmetric(@einsum (μ,ν) -> gi[ϵ,σ]*gi[λ,ρ]*Γ[μ,ϵ,λ]*Γ[ν,σ,ρ])

        #########################################################
        # Constraints and constraint damping terms (16%)

        γ0 = 0.1 # Harmonic constraint damping (>0)
        #γ1 = -1. # Linear Degeneracy parameter (=-1)
        γ2 = 0.1 # Derivative constraint damping (>0)

        Cx = Dx(fg,U,ns,nls,nrs,αls,αrs,xi,yi,zi)/hx - dx
        Cy = Dy(fg,U,ns,nls,nrs,αls,αrs,xi,yi,zi)/hy - dy
        Cz = Dz(fg,U,ns,nls,nrs,αls,αrs,xi,yi,zi)/hz - dz

        ∂tdx += γ2*α*Cx
        ∂tdy += γ2*α*Cy
        ∂tdz += γ2*α*Cz

        # Generalized Harmonic constraints and damping

        C_ = Γ_ - H_

        ∂tP += γ0*α*symmetric(@einsum (μ,ν) -> 2C_[μ]*n_[ν] - g[μ,ν]*n[ϵ]*C_[ϵ]) # + C_[ν]*n_[μ]

        ∂tP += -γ2*(βx*Cx + βy*Cy + βz*Cz) # γ1 = -1 just assumed here

        ########################################################

        ∂tU  = StateVector(∂tg,∂tdx,∂tdy,∂tdz,∂tP)

        # Perform boundary conditions (12%)
        ∂tU += SAT(U,ns,ls,ds,nls,nrs,αls,αrs,r,t,ri,γ2)

        # if !in_bulk 
        # end

        # Add numerical dissipation (20%)
        ∂tU += -0.05*Dissipation(U,ns,ds,nls,nrs,αls,αrs,xi,yi,zi) #-0.05

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


        Uw[xi,yi,zi] = NaNSV
        

    end

    # 250 ms benchmark total
    return

end

@parallel_indices (xi,yi,zi) function test!(U1,U2,U3,ns,ds,ls,iter::Int)
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

        g = U[xi,yi,zi].g

        _,_,_,_,γs... = g.data

        gi = inv(g)

        if det(ThreeTensor(γs))<0
            println("γ ",(xi,yi,zi))
            throw(0.)
        elseif -gi[1,1]<0
            println("α ",(xi,yi,zi))
            throw(0.)
        end
    end

    return

end

@parallel_indices (xi,yi,zi) function initialize!(Uw,ns,ds,ls,constraint_init)
    # Initializes the initial conditions to arrays
    # the metric must be written to the array first if one desires constraint initialization
    
    hx,hy,hz,dt = ds
    lx,ly,lz=ls
    x = -lx/2 + (xi-1)*hx; y = -ly/2 + (yi-1)*hy; z = -lz/2 + (zi-1)*hz;

    r = (x,y,z)
    ri = (xi,yi,zi)

    in_domain,nls,nrs,αls,αrs = find_boundary(ns,ls,ds,r,ri)

    if in_domain

        gKS = Uw[xi,yi,zi].g

        # Do coordinate transformation

        Λ = StateTensor((μ,ν) -> Λ_init(x,y,z,μ,ν))  #indices up down

        g = @einsum gKS[α,β]*Λ[α,μ]*Λ[β,ν]

        ∂tg = StateTensor((μ,ν) -> ∂tg_init(x,y,z,μ,ν)) 

        if constraint_init
            dx = Dx(fg,Uw,ns,nls,nrs,αls,αrs,xi,yi,zi)/hx
            dy = Dy(fg,Uw,ns,nls,nrs,αls,αrs,xi,yi,zi)/hy
            dz = Dz(fg,Uw,ns,nls,nrs,αls,αrs,xi,yi,zi)/hz
        else
            dx  = StateTensor((μ,ν) -> ∂xg(x,y,z,μ,ν))      
            dy  = StateTensor((μ,ν) -> ∂yg(x,y,z,μ,ν))      
            dz  = StateTensor((μ,ν) -> ∂zg(x,y,z,μ,ν)) 
        end

        gi = inv(g)

        #if -gi[1,1] < 0; println((xi,yi,zi)) end

        α = 1/sqrt(-gi[1,1])
    
        βx = -gi[1,2]/gi[1,1]
        βy = -gi[1,3]/gi[1,1]
        βz = -gi[1,4]/gi[1,1]

        n_ = FourVector((-α,0,0,0))

        n = @einsum gi[μ,ν]*n_[ν]

        δ = one(FourTensor)

        γi = gi + symmetric(@einsum n[μ]*n[ν])

        γm = δ + (@einsum n_[μ]*n[ν])

        γ = g + symmetric(@einsum n_[μ]*n_[ν])

        P2 = -(∂tg - βx*dx - βy*dy - βz*dz)/α

        if constraint_init
            # collect metric derivatives
            ∂g = Symmetric3rdOrderTensor((σ,μ,ν) -> (σ==1 ? ∂tg[μ,ν] : σ==2 ? dx[μ,ν] : σ==3 ? dy[μ,ν] : σ==4 ? dz[μ,ν] : throw(ZeroST)))

            G = minorsymmetric(NSProjection((μ,ν,α,β)-> γm[μ,α]*γm[ν,β] - n_[μ]*n_[ν]*γi[α,β]))

            C = Symmetric3rdOrderTensor((α,μ,ν) -> n_[μ]*γm[ν,α]/2 + n_[ν]*γm[μ,α]/2 - n_[μ]*n_[ν]*n[α])

            #Cf_ = FourVector((0,0,0,0))

            μ0 = 35.
            σ0 = 4
            Amp = 0.1

            r=sqrt(x^2+y^2+z^2)

            model = (μ0-σ0)<r<(μ0+σ0) ? (Amp/σ0^8)*(r-(μ0-σ0))^4*(r-(μ0+σ0))^4 : 0.

            Cf_ = 0*model*n_

            H_ = FourVector(μ -> fH_(x,y,z,μ))

            A_ = @einsum ( 2*γi[i,μ]*∂g[i,μ,α] - gi[μ,ν]*γm[α,i]*∂g[i,μ,ν] - 2*H_[α] - 2*Cf_[α])
            # index down

            P = symmetric(@einsum C[α,μ,ν]*A_[α] + G[μ,ν,α,β]*P2[α,β])
        else
            P = P2
        end

        Uw[xi,yi,zi] = StateVector(g,dx,dy,dz,P)

    else

        Uw[xi,yi,zi] = NaNSV

    end

    
    return

end

# Initial Conditions for Minkowski Space
# @inline g_init(x,y,z,μ,ν) =    (( -1. ,     0.    ,    0.  ,   0.   ),
#                                 (  0. ,     1.    ,    0.  ,   0.   ),
#                                 (  0. ,     0.    ,    1.  ,   0.   ),
#                                 (  0. ,     0.    ,    0.  ,   1.   ))[μ][ν]

# @inline ∂tg_init(x,y,z,μ,ν) =  ((  0. ,     0.    ,    0.  ,   0.   ),
#                                 (  0. ,     0.    ,    0.  ,   0.   ),
#                                 (  0. ,     0.    ,    0.  ,   0.   ),
#                                 (  0. ,     0.    ,    0.  ,   0.   ))[μ][ν]

# Initial Conditions for accelerating Minkowski Space
# a = 10
# @inline g_init(x,y,z,μ,ν) =    ((   a*x+5  ,   0  ,   0  ,   0   ),
#                                 (       0,          1  ,   0  ,   0   ),
#                                 (       0,          0  ,   1  ,   0   ),
#                                 (       0,          0  ,   0  ,   1   ))[μ][ν]

# @inline ∂tg_init(x,y,z,μ,ν) =  ((  0. ,     0.    ,    0.  ,   0.   ),
#                                 (  0. ,     0.    ,    0.  ,   0.   ),
#                                 (  0. ,     0.    ,    0.  ,   0.   ),
#                                 (  0. ,     0.    ,    0.  ,   0.   ))[μ][ν]

# Initial Conditions for Kerr-Schild spinning black hole

# Used in coordinate transformation
@inline rc(x,y,z) = sqrt(x^2+y^2+z^2)
@inline rs(x,y,z) = sqrt(x^2+y^2+z^2)

# radius in Kerr-Schild is given by an implicit formula:
#@inline rs(x,y,z,a) = sqrt( (x^2+y^2+z^2-a^2)/2 + sqrt((x^2+y^2+z^2-a^2)^2/4 + a^2*z^2) )

@inline function g_init(x,y,z,μ,ν) # Spinning black hole in Kerr-Schild form
    M = 10. # Mass of the hole
    a = 0*M # Spin of the hole (0<a<M)
    s = 1.  # Makes the hole black (+1) or white (-1)

    ((   -1+2M/rs(x,y,z)  ,   2*s*M*x/rs(x,y,z)^2 ,   2*s*M*y/rs(x,y,z)^2 ,   2*s*M*z/rs(x,y,z)^2  ),
     ( 2*s*M*x/rs(x,y,z)^2, 1 + 2M*x^2/rs(x,y,z)^3,   2M*x*y/rs(x,y,z)^3  ,   2M*x*z/rs(x,y,z)^3   ),
     ( 2*s*M*y/rs(x,y,z)^2,   2M*x*y/rs(x,y,z)^3  , 1 + 2M*y^2/rs(x,y,z)^3,   2M*y*z/rs(x,y,z)^3   ),
     ( 2*s*M*z/rs(x,y,z)^2,   2M*x*z/rs(x,y,z)^3  ,   2M*y*z/rs(x,y,z)^3  , 1 + 2M*z^2/rs(x,y,z)^3 ))[μ][ν]

    # r = rs(x,y,z,a)

    # η_ = ((-1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1))[μ][ν]

    # l_ = (1,(r*x+a*y)/(r^2+a^2),(r*y-a*x)/(r^2+a^2),z/r)

    # H = M*r^3/(r^4+a^2*z^2)

    # return η_ + 2*H*l_[μ]*l_[ν]

end

@inline function Λ_init(xp,yp,zp,μ,ν) #Allows for a coordinate transformation from Kerr-Schild form
    M = 10.
    a = 0*M # Spin of the hole (0<a<M)

    r = rs(xp,yp,zp)

    x = xp*r/sqrt(r^2+a^2)
    y = yp*r/sqrt(r^2+a^2)
    z = zp

    ((   1  ,   0 ,   0 ,   0  ),
     ( 0, (r^4+a^2*(r^2-x^2))/(r^3*sqrt(r^2+a^2)),   -a^2*x*y/(r^3*sqrt(r^2+a^2))  ,   -a^2*x*z/(r^3*sqrt(r^2+a^2))   ),
     ( 0,   -a^2*x*y/(r^3*sqrt(r^2+a^2))  , (r^4+a^2*(r^2-y^2))/(r^3*sqrt(r^2+a^2)),   -a^2*y*z/(r^3*sqrt(r^2+a^2))   ),
     ( 0,   0  ,   0  , 1 ))[μ][ν]
end

@inline function fH_(x,y,z,μ) 
    M = 10. # Mass of the hole
    s = 1. # Makes the hole black (+1) or white (-1)

    (2*s*M/rc(x,y,z)^2,2M*x/rc(x,y,z)^3,2M*y/rc(x,y,z)^3,2M*z/rc(x,y,z)^3)[μ] # lower index

end

@inline function ∂tg_init(x,y,z,μ,ν) 
    ((  0. ,     0.    ,    0.  ,   0.   ),
     (  0. ,     0.    ,    0.  ,   0.   ),
     (  0. ,     0.    ,    0.  ,   0.   ),
     (  0. ,     0.    ,    0.  ,   0.   ))[μ][ν]
end

# Derivatives of the initial conditions                            
@inline ∂xg(x,y,z,μ,ν) = ForwardDiff.derivative(x -> g_init(x,y,z,μ,ν), x)
@inline ∂yg(x,y,z,μ,ν) = ForwardDiff.derivative(y -> g_init(x,y,z,μ,ν), y)
@inline ∂zg(x,y,z,μ,ν) = ForwardDiff.derivative(z -> g_init(x,y,z,μ,ν), z)

@inline fUmBC(ℓ::FourVector,x,y,z,μ,ν) = ℓ[1]*∂tg_init(x,y,z,μ,ν) + ℓ[2]*∂xg(x,y,z,μ,ν) + ℓ[3]*∂yg(x,y,z,μ,ν) + ℓ[4]*∂zg(x,y,z,μ,ν) 

# Gauge functions and derivatives (freely specifyable)
#@inline fH_(x,y,z,μ) = (0.,0.,0.,0.)[μ] # lower index

@inline ∂tH(x,y,z,μ) = 0. 
@inline ∂xH(x,y,z,μ) = ForwardDiff.derivative(x -> fH_(x,y,z,μ), x)
@inline ∂yH(x,y,z,μ) = ForwardDiff.derivative(y -> fH_(x,y,z,μ), y)
@inline ∂zH(x,y,z,μ) = ForwardDiff.derivative(z -> fH_(x,y,z,μ), z)

@inline f∂H_(x,y,z,μ,ν) = (∂tH(x,y,z,ν),∂xH(x,y,z,ν),∂yH(x,y,z,ν),∂zH(x,y,z,ν))[μ]

##################################################
@views function main()

    # Physics
    lx, ly, lz = 100.0, 100.0, 100.0  # domain extends
    ls = (lx,ly,lz)
    t  = 0.0                          # physical start time

    scale = 1.

    # Numerics
    num = Int(100*scale)
    ns = (num,num,num)             # numerical grid resolution; should be a mulitple of 32-1 for optimal GPU perf
    nx, ny, nz = ns
    nt         = 500 #Int(1000*scale)                 # number of timesteps

    SAVE = false
    VISUAL = true
    constraint_init = true
    nout       = 10 #Int(10*scale)                       # plotting frequency
    noutsave   = 500#Int(50*scale) 


    save_size = 50
    step = Int(num/save_size)

    # Derived numerics
    hx, hy, hz = lx/(nx-1), ly/(ny-1), lz/(nz-1) # cell sizes
    CFL = 1/5
    dt = min(hx,hy,hz)*CFL
    ds = (hx,hy,hz,dt)

    coords = (-lx/2:hx:lx/2, -ly/2:hy:ly/2, -lz/2:hz:lz/2)
    X,Y,Z = coords

    # Array allocations
    # If we use the GPU, we need an intermediate array on the CPU to save
    if USE_GPU
        U0 = CPUCellArray{StateVector}(undef, nx, ny, nz)
        U1 =  CuCellArray{StateVector}(undef, nx, ny, nz)
        U2 =  CuCellArray{StateVector}(undef, nx, ny, nz)
        U3 =  CuCellArray{StateVector}(undef, nx, ny, nz)
    else
        U0 =  CPUCellArray{StateVector}(undef, nx, ny, nz)
        U1 =  CPUCellArray{StateVector}(undef, nx, ny, nz)
        U2 =  CPUCellArray{StateVector}(undef, nx, ny, nz)
        U3 =  CPUCellArray{StateVector}(undef, nx, ny, nz)
    end

    if USE_GPU; Uw = U0 else Uw = U1 end

    # Sample the Kerr-Schild form of the metric only
    for xi in 1:ns[1], yi in 1:ns[2], zi in 1:ns[3]

        hx,hy,hz,dt = ds
        lx,ly,lz=ls
        x = -lx/2 + (xi-1)*hx; y = -ly/2 + (yi-1)*hy; z = -lz/2 + (zi-1)*hz;

        gKS = StateTensor((μ,ν) -> g_init(x,y,z,μ,ν))    

        Uw[xi,yi,zi] = StateVector(gKS,ZeroST,ZeroST,ZeroST,ZeroST)

    end

    # write the rest of the state vector
    @parallel (1:nx,1:ny,1:nz) initialize!(Uw,ns,ds,ls,constraint_init)

    # copy to other arrays
    if USE_GPU; copy!(U1.data, U0.data) end

    copy!(U2.data, U1.data)
    copy!(U3.data, U1.data)

    copy!(U0.data, U1.data)

    # ri=(75,53,53)
    # xi,yi,zi = ri
    # x = -lx/2 + (xi-1)*hx; y = -ly/2 + (yi-1)*hy; z = -lz/2 + (zi-1)*hz;
    # r=(x,y,z)
    # return @code_warntype find_boundary(ns,ls,ds,r,ri)

    # k,ℓ,σ,kn,ℓn,σn = vectors(MinST,false,(1.,1.,1.),1,-1) # Form boundary basis

    # return     (ℓ - k)/sqrt(2)

    # return @code_warntype vectors(U1[1,1,1],1,1.)

    # return @code_warntype fH_(1.,1.,1.,1)

    # ri=(1,1,1)
    # r=(1.,1.,1.)
    # return @code_warntype SAT(U1,ns,ls,ds,(1,1,1),ns,(0.,0.,0.),(0.,0.,0.),r,t,ri,1.)

    #return @code_warntype BoundaryConditions(U1[1,1,1],false,(1.,1.,1.),(@Vec [1.,1.,1.,1.]),1.,(@Vec [1.,1.,1.,1.]),(@Vec [1.,1.,1.,1.]),ZeroST)

    #return @code_warntype fUmBC((@Vec [1.,1.,1.,1.]),1.,1.,1.,1,1)

    #return @benchmark rhs!($U1,$U2,$U3,$ns,$ls,$ds,1)
    #return @benchmark @parallel (1:$nx,1:$ny,1:$nz) rhs!($U1,$U2,$U3,$ns,$ls,$ds,1)

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

    # slice       = (:,:,Int(ceil(nz/2)))

    # A = [U1[xi,yi,Int(ceil(nz/2))].dx[1,1]-a for xi in 1:nx, yi in 1:ny]

    # heatmap(X, Y, A, aspect_ratio=1, xlims=(-lx/2,lx/2)
    # ,ylims=(-ly/2,ly/2),clim=(-10^-13,10^-13), title = "Time = "*string(round(t; digits=2)), c=:viridis,frame=:box)

    # ang = range(0, 2π, length = 60)
    # circle(x,y,r) = Shape(r*sin.(ang).+x, r*cos.(ang).+y)  

    # plot!(circle(0,0,25),fc=:transparent, legend = false, colorbar = true)

    # frame(anim)

    # return

    # bulk = (1:nx,1:ny,1:nz) #@parallel $bulk 
    # @profile @parallel bulk rhs!(U1,U2,U3,ns,ls,ds,3)

    #return Profile.print()
    #return @benchmark @parallel (1:$nx,1:$ny,1:$nz) rhs!($U1,$U2,$U3,$ns,$ls,$ds,$t,3)

    C_vec = fill(10^(-10),Int(nt/nout+1))

    H_vec1 = fill(-10.,Int(nt/nout+1))

    H_vec2 = fill(-10.,Int(nt/nout+1))

    A_vec = fill(-10.,Int(nt/nout+1))

    iter = 1
    it = 1

    try # try block to catch errors, close datafile if so

        # Time loop
        while it <= nt

            if (it==11)  global wtime0 = Base.time()  end # performance timing 

            # Visualisation
            if VISUAL && (it==1 || mod(it,nout)==0)

                if USE_GPU
                    copy!(U0.data, U1.data)
                    Cnorm = 0.
                    for xi in 1:nx, yi in 1:ny, zi in 1:nz
                        Cnorm += energy_cell(U0[xi,yi,zi],ns,ls,ds,(xi,yi,zi))
                    end
                else
                    Cnorm = 0.
                    for xi in 1:nx, yi in 1:ny, zi in 1:nz
                        Cnorm += sqrt(constraint_energy_cell(U1[xi,yi,zi],ns,ls,ds,(xi,yi,zi)))
                    end

                    A1 = SurfaceIntegral(Area,U1,ns,ds,ls,25.)
                    int1 = SurfaceIntegral(Hawking_Mass_cell,U1,ns,ds,ls,25.)

                    Hmass1 = sqrt(A1/(16*pi))*(1+int1/(16*pi))

                    A2 = SurfaceIntegral(Area,U1,ns,ds,ls,48.)
                    int2 = SurfaceIntegral(Hawking_Mass_cell,U1,ns,ds,ls,48.)

                    Hmass2 = sqrt(A2/(16*pi))*(1+int2/(16*pi))

                    if it == 1; 
                        global A_init = A1; 
                        global Hmass1_init = Hmass1; 
                        global Hmass2_init = Hmass2;
                        println(round(Hmass1, sigdigits=3)," ",round(Hmass2, sigdigits=3)," ",
                        round(A1/(4π*(25)^2)-1, sigdigits=3)," ",round(A2/(4π*(48)^2)-1, sigdigits=3)) 
                    end

                end

                #append!(C_vec,result)
                C_vec[Int(round(it/nout+1))] = Cnorm

                H_vec1[Int(round(it/nout+1))] = (Hmass1 - Hmass1_init)/Hmass1_init

                H_vec2[Int(round(it/nout+1))] = (Hmass2 - Hmass2_init)/Hmass2_init

                A_vec[Int(round(it/nout+1))] = (A1 - A_init)/A_init

                #A = Array(getindex.(CellArrays.field(U1,1),1,1))[slice...] .+ 1

                zi = Int(ceil(nz/2))

                #A = [(Dx(fg,U1,ns,find_boundary(ns,ls,ds,(X[xi],Y[yi],Z[zi]),(xi,yi,zi))[2:5]...,xi,yi,zi)/hx - U1[xi,yi,zi].dx)[1,1] for xi in 1:nx, yi in 1:ny]
                #A = [constraints(U1[xi,yi,zi],ns,ds,ls,(xi,yi,zi))[1] for xi in 1:nx, yi in 1:ny]
                #A = [(rootγ(U1[xi,yi,zi])[1])^2-1 for xi in 1:nx, yi in 1:ny]
                B = [U1[xi,yi,zi].g[2,2]-U0[xi,yi,zi].g[2,2] for xi in 1:nx, yi in 1:ny]

                heatmap(X, Y, B, aspect_ratio=1, xlims=(-lx/2,lx/2)
                ,ylims=(-ly/2,ly/2),clim=(-0.05,0.05), title = "Time = "*string(round(t; digits=2)), c=:viridis,frame=:box)

                ang = range(0, 2π, length = 60)
                circle(x,y,r) = Shape(r*sin.(ang).+x, r*cos.(ang).+y)  

                p1 = plot!(circle(0,0,25),fc=:transparent, legend = false, colorbar = true)

                p2 = plot(C_vec, yaxis=:log, ylim = (10^-1, 10^4), minorgrid=true) #[C_vec H_vec]

                p3 = plot([H_vec1 H_vec2 A_vec], ylim = (-0.1, 0.1))

                plot(p1,p2,p3, layout=grid(3,1,heights=(3/5,1/5,1/5)),size=(700,800),legend=false)

                #plot(p1)

                frame(anim)

            end

            # Perform RK3 algorithm
            for i in 1:3
                @parallel (1:nx,1:ny,1:nz) test!(U1,U2,U3,ns,ds,ls,i) 
                @parallel (1:nx,1:ny,1:nz) rhs!(U1,U2,U3,ns,ls,ds,t,i) 
            end

            t = t + dt

            if SAVE && mod(it,noutsave)==0

                if USE_GPU
                    copy!(U0.data, U1.data)
                    gdata[iter,:,:,:,:] = [U0[xi,yi,zi].g.data[i] for i in 1:6, zi in 1:step:nz, yi in 1:step:ny, xi in 1:step:nx]
                else
                    gdata[iter,:,:,:,:] = [U1[xi,yi,zi].g.data[i] for i in 1:6, zi in 1:step:nz, yi in 1:step:ny, xi in 1:step:nx]
                end

                iter += 1

            end

            it += 1

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
    wtime_it = wtime/(it-10)                     # Execution time per iteration [s]
    T_eff    = A_eff/wtime_it                    # Effective memory throughput [GB/s]

    @printf("Total steps=%d, time=%1.3f min (@ T_eff = %1.2f GB/s) \n", it, wtime/60, round(T_eff, sigdigits=3))
   
    GC.gc(true)

    if VISUAL 
        gif(anim, "acoustic3D.gif", fps = 15) 
        return #plot((C_vec), yaxis=:log, ylim = (10^-5, 10^5)) # 
    else
        return
    end

end

end