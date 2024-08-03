module GR3D

const USE_GPU = false  # Use GPU? If this is set false, then no GPU needs to be available
using ParallelStencil
using CellArrays, StaticArrays

using HDF5
using FileIO

ParallelStencil.@reset_parallel_stencil()

@static if USE_GPU
    @define_CuCellArray() # without this CuCellArrays don't work
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
end

using Plots, Printf, Statistics, BenchmarkTools, ForwardDiff

using Tensorial, InteractiveUtils

#using RootSolvers # (Roots.jl does not work on GPU)

using Roots

# Alias for symmetric 4 tensor
const StateTensor = SymmetricSecondOrderTensor{4,Data.Number,10}

# Alias for non-symmetric 4 tensor
const FourTensor = SecondOrderTensor{4,Data.Number,16}

#@CellType WaveCell fieldnames=(ψ,ψx,ψy,ψz,Ψ)

# Struct for main memory and Runge-Kutta algorithms that holds all state vector variables
struct StateVector <: FieldVector{5,Data.Number}
    ψ::Data.Number  # scalar field
    ψx::Data.Number # x-derivative
    ψy::Data.Number # y-derivative
    ψz::Data.Number # z-derivative
    Ψ::Data.Number  # normal projection derivative
    # nαlx::Data.Number  # Store the boundary positions
    # nαly::Data.Number  # This applies to the unique point
    # nαlz::Data.Number  # in the bulk domain that connects
    # nαrx::Data.Number  # to this boundary point.
    # nαry::Data.Number
    # nαrz::Data.Number
end

# Struct for points that belong on the boundary
struct BoundaryPoint <: FieldVector{5,Data.Number}
    ψ::Data.Number  # scalar field
    ψx::Data.Number # x-derivative
    ψy::Data.Number # y-derivative
    ψz::Data.Number # z-derivative
    Ψ::Data.Number  # normal projection derivative
    # nαlx::Data.Number  # Store the boundary positions
    # nαly::Data.Number  # This applies to the unique point
    # nαlz::Data.Number  # in the bulk domain that connects
    # nαrx::Data.Number  # to this boundary point.
    # nαry::Data.Number
    # nαrz::Data.Number
end

# Some convenience values for these types
const NotANumber = StateVector(NaN,NaN,NaN,NaN,NaN)
const Zero = StateVector(0.,0.,0.,0.,0.)

# Defintions of coefficents for embedded boundary finite differencing operators
# All of this can be found here: https://doi.org/10.1016/j.jcp.2022.111341

##################################################################
# Coefficent functions for second order diagonal norm 
# embedded boundary SBP operator

@inline h00(a) = (a+1)/4
@inline h11(a) = (a+1)^2/4
@inline h22(a) = (4+a-a^2)/4

@inline Q00(a) = -1/2
@inline Q01(a) = (a+1)/4
@inline Q02(a) = (1-a)/4
@inline Q12(a) = (a+1)/4

@inline fψ(U::StateVector) = U.ψ

@inline ψu(U::StateVector) = -U.Ψ # Scalar gradient-flux

@inline vx(U::StateVector) = U.ψx # vector gradient-flux

@inline vy(U::StateVector) = U.ψy # vector gradient-flux

@inline vz(U::StateVector) = U.ψz # vector gradient-flux

# Base.@propagate_inbounds @inline function linear_interpolation(f,Um,Bm,Pm,ns,face,xi,yi,zi,n1,α1)

#     # nl,nr = (nls[1], nrs[1])
#     # nlr = (nl, nr)
#     # αl,αr = (αls[1], αrs[1]); 

#     @inbounds @inline U(xi,yi,zi) = f(getindex(Um,xi,yi,zi))
#     @inbounds @inline B(face) = f(getindex(Bm,face,yi,zi))
#     @inbounds @inline P(face) = f(getindex(Pm,face,yi,zi))

#     ri = (xi,yi,zi)
#     nα = P(face)

#     @inline index(x) = if face%3 == 1 (x,yi,zi) elseif face%3 == 2 (xi,x,zi) else (xi,yi,x) end

#     if face in 1:3
#         temp,α2 = divrem(nα,ds[face],RoundDown)
#         n = Int(temp)
#         if n==ri[face%3+1]
#             Ui = (1-α1/α2)*U(xi,yi,zi) + (α1/α2)*B(face)
#         else
#             Ui = (1-α1)*U(xi,yi,zi) + α1*U(index(n1+1)...)
#         end
#     else
#         temp,α2 = divrem(nα,ds[face-3],RoundUp)
#         α2 *= -1; n = Int(temp)
#         if n==ri[face%3+1]
#             Ui = (1-α1/α2)*U(xi,yi,zi) + (α1/α2)*B(face)
#         else
#             Ui = (1-α1)*U(xi,yi,zi) + α1*U(index(n1-1)...)
#         end
#     end

#     return Ui

# end

# BDx(ψu,U,B,ns,ls,ds,face,i1,i2)

Base.@propagate_inbounds @inline function BDx(f,Um,Bm,B2U,ns,ls,ds,face,i1,i2)  # x-derivative
    # nl,nr = (nls[1], nrs[1])
    # nlr = (nl, nr)
    # αl,αr = (αls[1], αrs[1]); 
    dx,dy,dz,_ = ds
    lx,ly,lz = ls
    @inbounds @inline U(xi,yi,zi) = f(getindex(Um,xi,yi,zi))
    @inbounds @inline B() = f(getindex(Bm,face,i1,i2))

    @inline function D_x(i,j,k)
        r = (-lx/2 + (i-1)*dx,-ly/2 + (j-1)*dy, -lz/2 + (k-1)*dz)
        Dx(f,Um,Bm,ns,find_boundary(ns,ls,ds,r,(i,j,k))[2:5]...,i,j,k)
    end

    xi,yi,zi = B2U[face,i1,i2]
    r = (-lx/2 + (xi-1)*dx,-ly/2 + (yi-1)*dy, -lz/2 + (zi-1)*dz)
    _,_,_,αls,αrs = find_boundary(ns,ls,ds,r,(xi,yi,zi))

    if     face==1
        α = αrs[1]
        return -(Q00(α)*B() + Q01(α)*U(xi,yi,zi) + Q02(α)*U(xi-1,yi,zi))/h00(α)
    elseif face==2
        α = αrs[2]
        return ((1+α)*D_x(xi,yi,zi) - α*D_x(xi,yi-1,zi))
    elseif face==3
        α = αrs[3]
        return ((1+α)*D_x(xi,yi,zi) - α*D_x(xi,yi,zi-1))
    elseif face==4
        α = αls[1]
        return (Q00(α)*B() + Q01(α)*U(xi,yi,zi) + Q02(α)*U(xi+1,yi,zi))/h00(α)
    elseif face==5
        α = αls[2]
        return ((1+α)*D_x(xi,yi,zi) - α*D_x(xi,yi+1,zi))
    elseif face==6
        α = αls[3]
        return ((1+α)*D_x(xi,yi,zi) - α*D_x(xi,yi,zi+1))
    else
        println(ns," ",(αl,αr)," ",k)
        throw(0.)
    end
    # if nr-nl>=3 # not enough points for 4th order stencil, go to 2nd order
    #     D_2_1(U,nlr,αs,i)
    # elseif nr-nl==2 # not enough points for 2nd order stencil, go to 1st order 3 point
    #     D_3point(U,nlr,αs,i)
    # elseif nr-nl==1 # not enough points for 2nd order stencil, go to 1st order 2 point
    #     -U(nl) + U(nr) # note the two point operator happens to be the same for both points
    # else # only one grid point, extrapolate derivative
    #     throw(0.) # not implemented
    # end
end

Base.@propagate_inbounds @inline function BDy(f,Um,Bm,B2U,ns,ls,ds,face,i1,i2)  # y-derivative
    dx,dy,dz,_ = ds
    lx,ly,lz = ls
    @inbounds @inline U(xi,yi,zi) = f(getindex(Um,xi,yi,zi))
    @inbounds @inline B() = f(getindex(Bm,face,i1,i2))

    @inline function D_y(i,j,k)
        r = (-lx/2 + (i-1)*dx,-ly/2 + (j-1)*dy, -lz/2 + (k-1)*dz)
        Dy(f,Um,Bm,ns,find_boundary(ns,ls,ds,r,(i,j,k))[2:5]...,i,j,k)
    end

    xi,yi,zi = B2U[face,i1,i2]
    r = (-lx/2 + (xi-1)*dx,-ly/2 + (yi-1)*dy, -lz/2 + (zi-1)*dz)
    _,_,_,αls,αrs = find_boundary(ns,ls,ds,r,(xi,yi,zi))

    if     face==1
        α = αrs[1]
        return ((1+α)*D_y(xi,yi,zi) - α*D_y(xi-1,yi,zi))
    elseif face==2
        α = αrs[2]
        return -(Q00(α)*B() + Q01(α)*U(xi,yi,zi) + Q02(α)*U(xi,yi-1,zi))/h00(α)
    elseif face==3
        α = αrs[3]
        return ((1+α)*D_y(xi,yi,zi) - α*D_y(xi,yi,zi-1))
    elseif face==4
        α = αls[1]
        return ((1+α)*D_y(xi,yi,zi) - α*D_y(xi+1,yi,zi))
    elseif face==5
        α = αls[2]
        return (Q00(α)*B() + Q01(α)*U(xi,yi,zi) + Q02(α)*U(xi,yi+1,zi))/h00(α)
    elseif face==6
        α = αls[3]
        return ((1+α)*D_y(xi,yi,zi) - α*D_y(xi,yi,zi+1))
    else
        println(ns," ",(αl,αr)," ",k)
        throw(0.)
    end
    # if nr-nl>=3 # not enough points for 4th order stencil, go to 2nd order
    #     D_2_1(U,nlr,αs,j)
    # elseif nr-nl==2 # not enough points for 2nd order stencil, go to 1st order 3 point
    #     D_3point(U,nlr,αs,j)
    # elseif nr-nl==1 # not enough points for 2nd order stencil, go to 1st order 2 point
    #     -U(nl) + U(nr) # note the two point operator happens to be the same for both points
    # else # only one grid point, extrapolate derivative
    #     throw(0.) # not implemented
    # end
end

Base.@propagate_inbounds @inline function BDz(f,Um,Bm,B2U,ns,ls,ds,face,i1,i2)  # z-derivative
    dx,dy,dz,_ = ds
    lx,ly,lz = ls
    @inbounds @inline U(xi,yi,zi) = f(getindex(Um,xi,yi,zi))
    @inbounds @inline B() = f(getindex(Bm,face,i1,i2))

    @inline function D_z(i,j,k)
        r = (-lx/2 + (i-1)*dx,-ly/2 + (j-1)*dy, -lz/2 + (k-1)*dz)
        Dz(f,Um,Bm,ns,find_boundary(ns,ls,ds,r,(i,j,k))[2:5]...,i,j,k)
    end

    xi,yi,zi = B2U[face,i1,i2]
    r = (-lx/2 + (xi-1)*dx,-ly/2 + (yi-1)*dy, -lz/2 + (zi-1)*dz)
    _,_,_,αls,αrs = find_boundary(ns,ls,ds,r,(xi,yi,zi))

    if     face==1
        α = αrs[1]
        return ((1+α)*D_z(xi,yi,zi) - α*D_z(xi-1,yi,zi))
    elseif face==2
        α = αrs[2]
        return ((1+α)*D_z(xi,yi,zi) - α*D_z(xi,yi-1,zi))
    elseif face==3
        α = αrs[3]
        return -(Q00(α)*B() + Q01(α)*U(xi,yi,zi) + Q02(α)*U(xi,yi,zi-1))/h00(α)
    elseif face==4
        α = αls[1]
        return ((1+α)*D_z(xi,yi,zi) - α*D_z(xi+1,yi,zi))
    elseif face==5
        α = αls[2]
        return ((1+α)*D_z(xi,yi,zi) - α*D_z(xi,yi+1,zi))
    elseif face==6
        α = αls[3]
        return (Q00(α)*B() + Q01(α)*U(xi,yi,zi) + Q02(α)*U(xi,yi,zi+1))/h00(α)
    else
        println(ns," ",(αl,αr)," ",k)
        throw(0.)
    end
    # if nr-nl>=3 # not enough points for 4th order stencil, go to 2nd order
    #     D_2_1(U,nlr,αs,k)
    # elseif nr-nl==2 # not enough points for 2nd order stencil, go to 1st order 3 point
    #     D_3point(U,nlr,αs,k)
    # elseif nr-nl==1 # not enough points for 2nd order stencil, go to 1st order 2 point
    #     -U(nl) + U(nr) # note the two point operator happens to be the same for both points
    # else # only one grid point, extrapolate derivative
    #     throw(0.) # not implemented
    # end
end

Base.@propagate_inbounds @inline function BDiv(vx,vy,vz,U,B,B2U,ns,ls,ds,face,i1,i2) # Calculate the divergence of the flux
    dx,dy,dz,_ = ds
    BDx(vx,U,B,B2U,ns,ls,ds,face,i1,i2)/dx + BDy(vy,U,B,B2U,ns,ls,ds,face,i1,i2)/dy + BDz(vz,U,B,B2U,ns,ls,ds,face,i1,i2)/dz
end

Base.@propagate_inbounds @inline function Dx(f,Um,Bm,ns,nls,nrs,αls,αrs,i,j,k)  # x-derivative
    nl,nr = (nls[1], nrs[1])
    nlr = (nl, nr)
    αs = (αls[1], αrs[1])
    @inbounds @inline U(x) = f(getindex(Um,x,j,k))
    @inbounds @inline Bl() = f(getindex(Bm,1,j,k))
    @inbounds @inline Br() = f(getindex(Bm,4,j,k))
    D_2_1(U,Bl,Br,ns[1],nlr,αs,i)
    # if nr-nl>=3 # not enough points for 4th order stencil, go to 2nd order
    #     D_2_1(U,nlr,αs,i)
    # elseif nr-nl==2 # not enough points for 2nd order stencil, go to 1st order 3 point
    #     D_3point(U,nlr,αs,i)
    # elseif nr-nl==1 # not enough points for 2nd order stencil, go to 1st order 2 point
    #     -U(nl) + U(nr) # note the two point operator happens to be the same for both points
    # else # only one grid point, extrapolate derivative
    #     throw(0.) # not implemented
    # end
end

Base.@propagate_inbounds @inline function Dy(f,Um,Bm,ns,nls,nrs,αls,αrs,i,j,k)  # y-derivative
    nl,nr = (nls[2], nrs[2])
    nlr = (nl, nr)
    αs = (αls[2], αrs[2])
    @inbounds @inline U(x) = f(getindex(Um,i,x,k))
    @inbounds @inline Bl() = f(getindex(Bm,2,i,k))
    @inbounds @inline Br() = f(getindex(Bm,5,i,k))
    D_2_1(U,Bl,Br,ns[2],nlr,αs,j)
    # if nr-nl>=3 # not enough points for 4th order stencil, go to 2nd order
    #     D_2_1(U,nlr,αs,j)
    # elseif nr-nl==2 # not enough points for 2nd order stencil, go to 1st order 3 point
    #     D_3point(U,nlr,αs,j)
    # elseif nr-nl==1 # not enough points for 2nd order stencil, go to 1st order 2 point
    #     -U(nl) + U(nr) # note the two point operator happens to be the same for both points
    # else # only one grid point, extrapolate derivative
    #     throw(0.) # not implemented
    # end
end

Base.@propagate_inbounds @inline function Dz(f,Um,Bm,ns,nls,nrs,αls,αrs,i,j,k)  # z-derivative
    nl,nr = (nls[3], nrs[3])
    nlr = (nl, nr)
    αs = (αls[3], αrs[3])
    @inbounds @inline U(x) = f(getindex(Um,i,j,x))
    @inbounds @inline Bl() = f(getindex(Bm,3,i,j))
    @inbounds @inline Br() = f(getindex(Bm,6,i,j))
    D_2_1(U,Bl,Br,ns[3],nlr,αs,k)
    # if nr-nl>=3 # not enough points for 4th order stencil, go to 2nd order
    #     D_2_1(U,nlr,αs,k)
    # elseif nr-nl==2 # not enough points for 2nd order stencil, go to 1st order 3 point
    #     D_3point(U,nlr,αs,k)
    # elseif nr-nl==1 # not enough points for 2nd order stencil, go to 1st order 2 point
    #     -U(nl) + U(nr) # note the two point operator happens to be the same for both points
    # else # only one grid point, extrapolate derivative
    #     throw(0.) # not implemented
    # end
end

# Base.@propagate_inbounds @inline function BD_2_1(U,Bl,Br,ns,αs,k) # Second order accurate stencil
#     nl,nr = ns
#     αl,αr = αs
#     if k==nl-1
#         (Q00(αl)*Bl() + Q01(αl)*U(nl) + Q02(αl)*U(nl+1))/h00(αl)
#     elseif k==nr+1
#         -(Q00(αr)*Br() + Q01(αr)*U(nr) + Q02(αr)*U(nr-1))/h00(αr)
#     else
#         println(ns," ",αs," ",k)
#         throw(0.)
#     end
# end

Base.@propagate_inbounds @inline function D_2_1(U,Bl,Br,n,nlr,αs,k) # Second order accurate stencil
    nl,nr = nlr
    αl,αr = αs
    if k in nl+2:nr-2
        (-U(k-1) + U(k+1))/2
    elseif k==1
        (Q00(αl)*U(1) + Q01(αl)*U(2) + Q02(αl)*U(3))/h00(αl)
    elseif k==2
        (-Q01(αl)*U(1) + Q12(αl)*U(3))/h11(αl)
    elseif k==3
        (-Q02(αl)*U(1) - Q12(αl)*U(2) + U(4)/2)/h22(αl)
    elseif k==n
        -(Q00(αr)*U(n) + Q01(αr)*U(n-1) + Q02(αr)*U(n-2))/h00(αr)
    elseif k==n-1
        -(-Q01(αr)*U(n) + Q12(αr)*U(n-2))/h11(αr)
    elseif k==n-2
        -(-Q02(αr)*U(n) - Q12(αr)*U(n-1) + U(n-3)/2)/h22(αr)
    elseif k==nl
        (-Q01(αl)*Br() + Q12(αl)*U(nl+1))/h11(αl)
    elseif k==nl+1
        (-Q02(αl)*Br() - Q12(αl)*U(nl) + U(nl+2)/2)/h22(αl)
    elseif k==nr-1
        -(Q00(αr)*Bl() + Q01(αr)*U(nr) + Q02(αr)*U(nr-1))/h00(αr)
    elseif k==nr
        -(-Q01(αr)*Bl() + Q12(αr)*U(nr-1))/h11(αr)
    else
        println(nlr," ",αs," ",k)
        throw(0.)
    end
end

Base.@propagate_inbounds @inline function D4(U,ns,αs,k) # Fourth derivative stencil for dissipation
    nl,nr = ns
    αl,αr = αs
    if k in nl+3:nr-3
        U(k-2) - 4*U(k-1) + 6*U(k) - 4*U(k+1) + U(k+2)
    elseif k==nl
        (2*U(nl) - 4*U(nl+1) + 2*U(nl+2))/h00(αl)
    elseif k==nl+1
        (-4*U(nl) + 9*U(nl+1) - 6*U(nl+2) + U(nl+3))/h11(αl)
    elseif k==nl+2
        (2*U(nl) - 6*U(nl+1) + 7*U(nl+2) - 4*U(nl+3) + U(nl+4))/h22(αl)
    elseif k==nr
        (2*U(nr) - 4*U(nr-1) + 2*U(nr-2))/h00(αr)
    elseif k==nr-1
        (-4*U(nr) + 9*U(nr-1) - 6*U(nr-2) + U(nr-3))/h11(αr)
    elseif k==nr-2
        (2*U(nr) - 6*U(nr-1) + 7*U(nr-2) - 4*U(nr-3) + U(nr-4))/h22(αr)
    else
        throw(0.)
    end
end

Base.@propagate_inbounds @inline function Dissipation(Um,ns,nls,nrs,αls,αrs,i,j,k)  # Dissipation in all directions
    @inbounds @inline Ux(x) = getindex(Um,x,j,k)
    @inbounds @inline Uy(x) = getindex(Um,i,x,k)
    @inbounds @inline Uz(x) = getindex(Um,i,j,x)
    D4(Ux,(nls[1],nrs[1]),(αls[1],αrs[1]),i) + D4(Uy,(nls[2],nrs[2]),(αls[2],αrs[2]),j) + D4(Uz,(nls[3],nrs[3]),(αls[3],αrs[3]),k)
end

Base.@propagate_inbounds @inline function Div(vx,vy,vz,U,B,ns,nls,nrs,αls,αrs,ds,i,j,k) # Calculate the divergence of the flux
    dx,dy,dz,_ = ds
    Dx(vx,U,B,ns,nls,nrs,αls,αrs,i,j,k)/dx + Dy(vy,U,B,ns,nls,nrs,αls,αrs,i,j,k)/dy + Dz(vz,U,B,ns,nls,nrs,αls,αrs,i,j,k)/dz
end

Base.@propagate_inbounds @inline function vectors(outer,rb)
    # Returns the null basis to the boundary
    # The embedded boundary method effectively treats the boundary as "Lego" objects
    # with boundary normals only in one of the 3 cardinal directions.
    # Retruns upper index vectors and a mixed index boundary 2-metric

    #sx,sy,sz = (0.,0.,0)

    # if     i==1
    #     (sx = l)
    # elseif i==2 
    #     (sy = l)
    # else#if i==3
    #      (sz = l)
    # end

    # if outer 
    #     if     i==1
    #         (sx = l)
    #     elseif i==2 
    #         (sy = l)
    #     else#if i==3
    #         (sz = l)
    #     end
    # else

    # end

    x,y,z = rb
    sx = -x
    sy = -y
    sz = -z

    norm = sqrt(sx^2 + sy^2 + sz^2)

    #if norm == 0.; println("what") end

    s = @Vec [0.,sx/norm,sy/norm,sz/norm]

    n = @Vec [1.,0.,0.,0.]

    ℓ = (n + s)/sqrt(2)
    k = (n - s)/sqrt(2)

    δ = one(FourTensor)

    σ = FourTensor((μ,ν) -> δ[μ,ν] + k[μ]*ℓ[ν] + ℓ[μ]*k[ν])

    return (k,ℓ,σ)
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

    f(x,y,z) = x^p + y^p + z^p - R^p

    #f(x,y,z) = x + 200

    xibl =  1; yibl =  1; zibl =  1;
    xibr = nx; yibr = ny; zibr = nz;

    αlx = 0.; αly = 0.; αlz = 0.;
    αrx = 0.; αry = 0.; αrz = 0.;

    s = sign(f(x,y,z))

    a = 1/1000

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

    outside = !(s==-1)

    return (outside,nl,nr,αl,αr)

end

Base.@propagate_inbounds @inline function SAT(B,B2U,ns,ls,ds,face,i1,i2)
    # Performs the boundary conditions in each direction
    # Since the StateVector on the boundary is not in the domain, its value must be extrapolated.
    # Since the boundary position is not in the domain, the application of boundary conditions
    # must also be extrapolated.

    lx,ly,lz = ls
    dx,dy,dz,_=ds

    #xi,yi,zi=ri

    #outer =  (nr == ns[i])

    xi,yi,zi = B2U[face,i1,i2]
    r = (-lx/2 + (xi-1)*dx,-ly/2 + (yi-1)*dy, -lz/2 + (zi-1)*dz)
    _,_,_,αls,αrs = find_boundary(ns,ls,ds,r,(xi,yi,zi))

    #face in 1:3 ? α = αrs[face] : α = αls[face-3]

    if     face==1
        α = αrs[1]
        rb = (r[1]+α*dx,r[2],r[3])
    elseif face==2
        α = αrs[2]
        rb = (r[1],r[2]+α*dy,r[3])
    elseif face==3
        α = αrs[3]
        rb = (r[1],r[2],r[3]+α*dz)
    elseif face==4
        α = αls[1]
        rb = (r[1]-α*dx,r[2],r[3])
    elseif face==5
        α = αls[2]
        rb = (r[1],r[2]-α*dy,r[3])
    elseif face==6
        α = αls[3]
        rb = (r[1],r[2],r[3]-α*dx)
    else
        throw(0.)
    end
  
    k,ℓ,σ = vectors(true,rb) # Form boundary basis

    ψx = B.ψx
    ψy = B.ψy
    ψz = B.ψz
    Ψ  = B.Ψ

    ∂ψ = @Vec [-Ψ,ψx,ψy,ψz]

    Up = (@einsum k[α]*∂ψ[α])
    U0 = @einsum σ[α,β]*∂ψ[β]

    UmBC = Up

    ΨBC  = -(Up + UmBC)/sqrt(2)
    ∂ψBC =  U0 - k*UmBC - ℓ*Up

    ε = 1/h00(α)/ds[face%3+1]

    SATΨ  = ε*(ΨBC - Ψ)
    SATψx = ε*(∂ψBC[2] - ψx)
    SATψy = ε*(∂ψBC[3] - ψy)
    SATψz = ε*(∂ψBC[4] - ψz)
    SATψ  = 0.

    return StateVector(SATψ,SATψx,SATψy,SATψz,SATΨ)

end

# function energy_cell(U,ns,ls,ds,ri) # calculates wave energy in a cell

#     ψx = U.ψx
#     ψy = U.ψy
#     ψz = U.ψz
#     Ψ  = U.Ψ

#     xi,yi,zi = ri

#     dx,dy,dz,_ = ds
#     lx,ly,lz=ls
#     x = -lx/2 + (xi-1)*dx; y = -ly/2 + (yi-1)*dy; z = -lz/2 + (zi-1)*dz;
#     r = (x,y,z)

#     outside,nls,nrs,αls,αrs = find_boundary(ns,ls,ds,r,ri)

#     h = 1.

#     for i in 1:3
#         if ri[i] == nrs[i]
#             h *= h2_11(αrs[i])
#         elseif ri[i] == nrs[i]-1
#             h *= h2_22(αrs[i])
#         elseif ri[i] == nls[i]
#             h *= h2_11(αls[i])
#         elseif ri[i] == nls[i]+1
#             h *= h2_22(αls[i])
#         end
#     end

#     return (Ψ^2 + ψx^2 + ψy^2 + ψz^2)*h*dx*dy*dz

# end

@parallel_indices (face,i1,i2) function boundary_evolution!(U1,U2,U3,B1,B2,B3,B2U,ns::Tuple,ls::Tuple,ds::Tuple,iter::Int)
    # Performs the right hand side of the system of equations.
    # The if blocks at the beginning and end perform the 3rd order Runge-Kutta algorithm
    # which is done by calling rhs! three times, each with a different value in iter ranging from 1:3

    if iter == 1
        U = U1
        B = B1
        Bw = B2
    elseif iter == 2
        U = U2
        B = B2
        Bw = B3
    else
        U = U3
        B = B3
        Bw = B1
    end

    if haskey(B2U,(face,i1,i2))

    # Blx = @view B[1,:,:]; Bly = @view B[2,:,:]; Blz = @view B[3,:,:];
    # Brx = @view B[4,:,:]; Bry = @view B[5,:,:]; Brz = @view B[6,:,:];

    dx,dy,dz,dt = ds
    #lx,ly,lz=ls
    #x = -lx/2 + (xi-1)*dx; y = -ly/2 + (yi-1)*dy; z = -lz/2 + (zi-1)*dz;
    #nx,ny,nz = ns

    #r = (x,y,z)
    #ri = (xi,yi,zi)

    #_,nls,nrs,αls,αrs = find_boundary(ns,ls,ds,r,ri)

    Bxyz = B[face,i1,i2]

    ψ = Bxyz.ψ; ψx = Bxyz.ψx; ψy = Bxyz.ψy; ψz = Bxyz.ψz; Ψ = Bxyz.Ψ

    γ2 = 0.

    Cx = BDx(fψ,U,B,B2U,ns,ls,ds,face,i1,i2)/dx - ψx
    Cy = BDy(fψ,U,B,B2U,ns,ls,ds,face,i1,i2)/dy - ψy
    Cz = BDz(fψ,U,B,B2U,ns,ls,ds,face,i1,i2)/dz - ψz

    ∂tψ = -Ψ

    ∂tψx = BDx(ψu,U,B,B2U,ns,ls,ds,face,i1,i2)/dx + γ2*Cx

    ∂tψy = BDy(ψu,U,B,B2U,ns,ls,ds,face,i1,i2)/dy + γ2*Cy

    ∂tψz = BDz(ψu,U,B,B2U,ns,ls,ds,face,i1,i2)/dz + γ2*Cz

    ∂tΨ  = -BDiv(vx,vy,vz,U,B,B2U,ns,ls,ds,face,i1,i2)

    ∂tB  = Zero #StateVector(∂tψ,∂tψx,∂tψy,∂tψz,∂tΨ)

    ∂tB += SAT(Bxyz,B2U,ns,ls,ds,face,i1,i2)

    #∂tU += -0.005*Dissipation(U,ns,nls,nrs,αls,αrs,xi,yi,zi)

    if iter == 1
        B1t = Bxyz
        Bwxyz = B1t + dt*∂tB
    elseif iter == 2
        B1t = B1[face,i1,i2]
        B2t = Bxyz
        Bwxyz = (3/4)*B1t + (1/4)*B2t + (1/4)*dt*∂tB
    elseif iter == 3
        B1t = B1[face,i1,i2]
        B2t = Bxyz
        Bwxyz = (1/3)*B1t + (2/3)*B2t + (2/3)*dt*∂tB
    end

    Bw[face,i1,i2] = Bwxyz

    else # do nothing if the boundary array is not in the B2U dictionary

    end

    return

end

@parallel_indices (xi,yi,zi) function bulk_evolution!(U1,U2,U3,B1,B2,B3,B2U,ns::Tuple,ls::Tuple,ds::Tuple,iter::Int)
    # Performs the right hand side of the system of equations.
    # The if blocks at the beginning and end perform the 3rd order Runge-Kutta algorithm
    # which is done by calling rhs! three times, each with a different value in iter ranging from 1:3

    if iter == 1
        U = U1
        B = B1
        Uw = U2
    elseif iter == 2
        U = U2
        B = B2
        Uw = U3
    else
        U = U3
        B = B3
        Uw = U1
    end

    # Blx = @view B[1,:,:]; Bly = @view B[2,:,:]; Blz = @view B[3,:,:];
    # Brx = @view B[4,:,:]; Bry = @view B[5,:,:]; Brz = @view B[6,:,:];

    dx,dy,dz,dt = ds
    lx,ly,lz=ls
    x = -lx/2 + (xi-1)*dx; y = -ly/2 + (yi-1)*dy; z = -lz/2 + (zi-1)*dz;
    #nx,ny,nz = ns

    r = (x,y,z)
    ri = (xi,yi,zi)

    outside,nls,nrs,αls,αrs = find_boundary(ns,ls,ds,r,ri)

    # #outside,nls,nrs,αls,αrs = true,(1,1,1),ns,(0.,0.,0.),(0.,0.,0.)

    if outside

        Uxyz = U[xi,yi,zi]

        ψ = Uxyz.ψ; ψx = Uxyz.ψx; ψy = Uxyz.ψy; ψz = Uxyz.ψz; Ψ = Uxyz.Ψ

        γ2 = 0.
        
        Cx = Dx(fψ,U,B,ns,nls,nrs,αls,αrs,xi,yi,zi)/dx - ψx
        Cy = Dy(fψ,U,B,ns,nls,nrs,αls,αrs,xi,yi,zi)/dy - ψy
        Cz = Dz(fψ,U,B,ns,nls,nrs,αls,αrs,xi,yi,zi)/dz - ψz

        ∂tψ = -Ψ

        ∂tψx = Dx(ψu,U,B,ns,nls,nrs,αls,αrs,xi,yi,zi)/dx + γ2*Cx

        ∂tψy = Dy(ψu,U,B,ns,nls,nrs,αls,αrs,xi,yi,zi)/dy + γ2*Cy

        ∂tψz = Dz(ψu,U,B,ns,nls,nrs,αls,αrs,xi,yi,zi)/dz + γ2*Cz

        ∂tΨ  = -Div(vx,vy,vz,U,B,ns,nls,nrs,αls,αrs,ds,xi,yi,zi)

        ∂tU  = StateVector(∂tψ,∂tψx,∂tψy,∂tψz,∂tΨ)

        #∂tU += -0.005*Dissipation(U,ns,nls,nrs,αls,αrs,xi,yi,zi)

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

    else # don't do anything if outside of the boundary

        # Uxyz = U[xi,yi,zi]

        # ∂tU  = Zero

        # if iter == 1
        #     U1t = Uxyz
        #     Uwxyz = U1t + dt*∂tU
        # elseif iter == 2
        #     U1t = U1[xi,yi,zi]
        #     U2t = Uxyz
        #     Uwxyz = (3/4)*U1t + (1/4)*U2t + (1/4)*dt*∂tU
        # elseif iter == 3
        #     U1t = U1[xi,yi,zi]
        #     U2t = Uxyz
        #     Uwxyz = (1/3)*U1t + (2/3)*U2t + (2/3)*dt*∂tU
        # end

        # Uw[xi,yi,zi] = Uwxyz


        Uw[xi,yi,zi] = NotANumber
        

    end

    return

end

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
    nt         = 1000 #Int(1000*scale)                 # number of timesteps

    SAVE = false
    VISUAL = true
    nout       = 10#Int(10*scale)                       # plotting frequency
    noutsave   = Int(50*scale) 


    save_size = 50
    step = Int(n/save_size)

    # Derived numerics
    dx, dy, dz = lx/(nx-1), ly/(ny-1), lz/(nz-1) # cell sizes
    CFL = 1/5
    dt = min(dx,dy,dz)*CFL
    ds = (dx,dy,dz,dt)

    coords = (-lx/2:dx:lx/2, -ly/2:dy:ly/2, -lz/2:dz:lz/2)
    #X, Y, Z = 0:dx:lx, 0:dy:ly, 0:dz:lz
    X,Y,Z = coords

    # ri = (50,47,26)
    # println(X[ri[1]]," ",Y[ri[2]]," ",Z[ri[3]])
    # return find_boundary(ls,ds,ri)

    # test = StateVector(1,2,3,4,5)
    # return test[2]

    # Initial Conditions
    #σ = 2.; x0 = lx/2; y0 = ly/2; z0 = lz/2;
    σ = 3.; x0 = 0.; y0 = 35.; z0 = 0.;
    @inline ψ_init(x,y,z) = exp(-((x-x0)^2+(y-y0)^2+(z-z0)^2)/σ^2)

    @inline ∂xψ(x,y,z) = ForwardDiff.derivative(x -> ψ_init(x,y,z), x)
    @inline ∂yψ(x,y,z) = ForwardDiff.derivative(y -> ψ_init(x,y,z), y)
    @inline ∂zψ(x,y,z) = ForwardDiff.derivative(z -> ψ_init(x,y,z), z)

    @inline ∂tψ(x,y,z) = 0.

    # Array allocations

    # U1 = @zeros(ns..., celltype=WaveCell)
    # U2 = @zeros(ns..., celltype=WaveCell)
    # U3 = @zeros(ns..., celltype=WaveCell)

    # Array allocations
    if USE_GPU # If we use the GPU, we need an intermediate array on the CPU to save
        # Holds main memory in the bulk of the domain
        U0 = CPUCellArray{StateVector}(undef, nx, ny, nz)
        U1 =  CuCellArray{StateVector}(undef, nx, ny, nz)
        U2 =  CuCellArray{StateVector}(undef, nx, ny, nz)
        U3 =  CuCellArray{StateVector}(undef, nx, ny, nz)

        # Holds the points on the boundary
        # Six faces of the rectangle (using the maximum length)
        # Can only accomidate one internal boundary, you need
        # more for more boundaries, or boundaries that overlap
        # in any coordinate direction. How can you do this
        # with only the memory you need?
        nmax = maximum(ns)
        #B0 = CPUCellArray{BoundaryPoint}(undef, 6, nmax, nmax) Do we really need this to save?
        B1 =  CuCellArray{StateVector}(undef, 6, nmax, nmax)
        B2 =  CuCellArray{StateVector}(undef, 6, nmax, nmax)
        B3 =  CuCellArray{StateVector}(undef, 6, nmax, nmax)
    else # Using the CPU
        # Holds main memory in the bulk of the domain
        U1 =  CPUCellArray{StateVector}(undef, nx, ny, nz)
        U2 =  CPUCellArray{StateVector}(undef, nx, ny, nz)
        U3 =  CPUCellArray{StateVector}(undef, nx, ny, nz)

        # Holds the points on the boundary
        # Six faces of the rectangle (using the maximum length)
        # Can only accomidate one internal boundary, you need
        # more for more boundaries, or boundaries that overlap
        # in any coordinate direction. How can you do this
        # with only the memory you need?
        nmax = maximum(ns)
        B1 =  CPUCellArray{StateVector}(undef, 6, nmax, nmax)
        B2 =  CPUCellArray{StateVector}(undef, 6, nmax, nmax)
        B3 =  CPUCellArray{StateVector}(undef, 6, nmax, nmax)

        # P1 =  CPUCellArray{Data.Number}(undef, 6, nmax, nmax)
        # P2 =  CPUCellArray{Data.Number}(undef, 6, nmax, nmax)
        # P3 =  CPUCellArray{Data.Number}(undef, 6, nmax, nmax)
    end

    for xi in 1:ns[1], yi in 1:ns[2], zi in 1:ns[3]

        x = X[xi]
        y = Y[yi]
        z = Z[zi]

        ψ  = ψ_init(x,y,z)
        ψx =    ∂xψ(x,y,z)
        ψy =    ∂yψ(x,y,z)
        ψz =    ∂zψ(x,y,z)
        Ψ  =    ∂tψ(x,y,z)

        if USE_GPU
            U0[xi,yi,zi] = StateVector(ψ,ψx,ψy,ψz,Ψ)
        else
            U1[xi,yi,zi] = StateVector(ψ,ψx,ψy,ψz,Ψ)
        end

    end

    B2U = Dict{NTuple{3,Int64},NTuple{3,Int64}}()
    R = lx/4.
    p = 2
    f(x,y,z) = x^p + y^p + z^p - R^p
    s = sign(f(ls...))

    for i in 1:3

        if i==1; is = (2,3) elseif i==2; is = (1,3) else is = (1,2) end
       
        for i1 in 1:ns[is[1]], i2 in 1:ns[is[2]]

            g(i,r) = if i==1; f(r,Y[i1],Z[i2]) elseif i==2; f(X[i1],r,Z[i2]) else f(X[i1],Y[i2],r) end

            roots = find_zeros(x->g(i,x), -ls[i]/2, ls[i]/2 )

            if length(roots) == 0
                continue
            elseif length(roots) == 1
                throw(0.)
            elseif length(roots) == 2

                # first root 
                r1 = roots[1]

                temp,α1 = divrem(r1+ls[i]/2,ds[i],RoundDown)
                α1 /= ds[i]; 
                n1 = Int(temp)+1;

                # second root 
                r2 = roots[2]

                temp,α2 = divrem(r2+ls[i]/2,ds[i],RoundUp)
                α2 /= -ds[i]; 
                n2 = Int(temp)+1;

            else
                throw(0.)
            end

            if i == 1
                x1 = X[n1] + α1*ds[i]
                y1 = Y[i1]
                z1 = Z[i2]
                B2U[1,i1,i2] = (n1,i1,i2)

                x2 = X[n2] - α2*ds[i]
                y2 = Y[i1]
                z2 = Z[i2]
                B2U[4,i1,i2] = (n2,i1,i2)
            elseif i==2
                x1 = X[i1]
                y1 = Y[n1] + α1*ds[i]
                z1 = Z[i2]
                B2U[2,i1,i2] = (i1,n1,i2)

                x2 = X[i1]
                y2 = Y[n2] - α2*ds[i]
                z2 = Z[i2]
                B2U[5,i1,i2] = (i1,n2,i2)
            else
                x1 = X[i1]
                y1 = Y[i2]
                z1 = Z[n1] + α1*ds[i]
                B2U[3,i1,i2] = (i1,i2,n1)

                x2 = X[i1]
                y2 = Y[i2]
                z2 = Z[n2] - α2*ds[i]
                B2U[6,i1,i2] = (i1,i2,n2)
            end

            ψ1  = ψ_init(x1,y1,z1);  ψ2  = ψ_init(x2,y2,z2)
            ψx1 =    ∂xψ(x1,y1,z1);  ψx2 =    ∂xψ(x2,y2,z2)
            ψy1 =    ∂yψ(x1,y1,z1);  ψy2 =    ∂yψ(x2,y2,z2)
            ψz1 =    ∂zψ(x1,y1,z1);  ψz2 =    ∂zψ(x2,y2,z2)
            Ψ1  =    ∂tψ(x1,y1,z1);  Ψ2  =    ∂tψ(x2,y2,z2)

            B1[i,i1,i2]   = StateVector(ψ1,ψx1,ψy1,ψz1,Ψ1)

            B1[i+3,i1,i2] = StateVector(ψ2,ψx2,ψy2,ψz2,Ψ2)

        end

    end

    if USE_GPU; copy!(U1.data, U0.data) end

    # for i in 1:5
    #     CellArrays.field(U1,i) .= Data.Array(temp[i,:,:,:])
    # end

    copy!(U2.data, U1.data)
    copy!(U3.data, U1.data)
    
    # result = 0.
    # @parallel (1:nx,1:ny,1:nz) energy_cell(U1,ds,result)

    # return result

    # ns = ((10,26),(1,ny),(1,nz))

    # αs = ((0.25,0.25),(0.25,0.25),(0.25,0.25))

    # return [-Dx(ψu,U1,ns,αs,i,1,1)/dx - 2*(-lx/2 + (i-1)*dx) for i in ns[1][1]:ns[1][2] ]

    # ri=(1,1,1)
    # r=(1.,1.,1.)
    # return @code_warntype find_boundary(ns,ls,ds,r,ri)

    #return @benchmark @parallel rhs!($U1,$U2,$U3,$ns,$ls,$ds,1)

    #return 0.

    # Preparation to save
    if SAVE
        path = string("3D_data")
        old_files = readdir(path; join=true)
        for i in 1:length(old_files) 
            rm(old_files[i]) 
        end
        datafile = h5open(path*"/data.h5","cw")
        nsave = Int64(nt/noutsave) + 1
        φdata = create_dataset(datafile, "phi", datatype(Data.Number), 
                dataspace(nsave,5,save_size,save_size,save_size), 
                chunk=(1,5,save_size,save_size,save_size))
    end
    #gdata = create_dataset(datafile, "g", datatype(Data.Number), dataspace(nsave,6,nr,nθ), chunk=(1,6,nr,nθ))

    # coordsfile = h5open(path*"/coords.h5","cw")
    # coordsfile["r"] = Array(rM)
    # coordsfile["theta"]  = Array(θM)
    # close(coordsfile)

    # Preparation of visualisation
    ENV["GKSwstype"]="nul"; if isdir("viz3D_out")==false mkdir("viz3D_out") end; loadpath = "./viz3D_out/"; anim = Animation(loadpath,String[])
    old_files = readdir(loadpath; join=true)
    for i in 1:length(old_files) rm(old_files[i]) end
    println("Animation directory: $(anim.dir)")

    slice       = (:,:,Int(ceil(nz/2)))

    # bulk = (1:nx,1:ny,1:nz) #@parallel $bulk 
    # return @benchmark @parallel $bulk rhs!($U1,$U2,$U3,$ns,$ds,3)

    #return size(U1[1:step:nx,1:step:ny,1:step:nz])

    # .= U1[1:step:nx,1:step:ny,1:step:nz][:]

    evec = zeros(0)

    iter = 1

    try

    # Time loop
    for it = 1:nt

        if (it==11)  global wtime0 = Base.time()  end

        # Perform RK3 algorithm
        for i in 1:3
            @parallel (1:nx,1:ny,1:nz)    bulk_evolution!(U1,U2,U3,B1,B2,B3,B2U,ns,ls,ds,i) 
            @parallel (1:6,1:nmax,1:nmax) boundary_evolution!(U1,U2,U3,B1,B2,B3,B2U,ns,ls,ds,i) 
        end

        t = t + dt

        # Visualisation
        if VISUAL && (mod(it,nout)==0)

            zi = Int(ceil(nz/2))
            #A = [(Dx(fψ,U1,ns,find_boundary(ns,ls,ds,(X[xi],Y[yi],Z[zi]),(xi,yi,zi))[2:5]...,xi,yi,zi)/dx - U1[xi,yi,zi].ψx) for xi in 1:nx, yi in 1:ny]

            A = Array(CellArrays.field(U1,1))[slice...]
            heatmap(X, Y, A, aspect_ratio=1, xlims=(-lx/2,lx/2)
            ,ylims=(-ly/2,ly/2),clim=(-0.1,0.1), c=:viridis,frame=:box)


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

        # saving
        if SAVE && mod(it,noutsave)==0

            if USE_GPU
                copy!(U0.data, U1.data)
                φdata[iter,:,:,:,:] = [U0[xi,yi,zi][i] for i in 1:5, zi in 1:step:nz, yi in 1:step:ny, xi in 1:step:nx]
            else
                φdata[iter,:,:,:,:] = [U1[xi,yi,zi][i] for i in 1:5, zi in 1:step:nz, yi in 1:step:ny, xi in 1:step:nx]
            end

            iter += 1

        end

    end

    catch error

        GC.gc(true)
        if SAVE; close(datafile) end

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

    if SAVE; close(datafile) end
    GC.gc(true)

    # Performance metrics
    wtime    = Base.time() - wtime0
    bytes_accessed = sizeof(StateVector)*(2*3)    # Bytes accessed per iteration per gridpoint (1 read and write iter=1 and 2 reads 1 write for iter=2,3)
    A_eff    = bytes_accessed*nx*ny*nz/1e9       # Effective main memory access per iteration [GB] 
    wtime_it = wtime/(nt-10)                     # Execution time per iteration [s]
    T_eff    = A_eff/wtime_it                    # Effective memory throughput [GB/s]

    @printf("Total steps=%d, time=%1.3f min (@ T_eff = %1.2f GB/s) \n", nt, wtime/60, round(T_eff, sigdigits=3))
   
    if VISUAL 
        gif(anim, "acoustic3D.gif", fps = 15) 
        return #plot((evec)./evec[1], ylim = (0, 2))
    else
        return
    end

end

end