module GR3D

const USE_GPU = true  # Use GPU? If this is set false, then no GPU needs to be available
using ParallelStencil
using CellArrays

using HDF5
using FileIO

ParallelStencil.@reset_parallel_stencil()

@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
end

using Plots, Printf, Statistics, BenchmarkTools, ForwardDiff

using Tensorial, InteractiveUtils

using RootSolvers # (Roots.jl does not work on GPU)

# Alias for non-symmetric 3 tensor
const StateTensor = SymmetricSecondOrderTensor{4,Data.Number,10}

@CellType WaveCell fieldnames=(ψ,ψx,ψy,ψz,Ψ)

# Some convenience values for these types
const NotANumber = WaveCell(NaN,NaN,NaN,NaN,NaN)
const Zero = WaveCell(0.,0.,0.,0.,0.)

# Defintions of coefficents for embedded boundary finite differencing operators
# All of this can be found here: https://doi.org/10.1016/j.jcp.2016.12.034

# const q11 = -24/17.; const q21 = 59/34. ;
# const q31 = -4/17. ; const q41 = -3/34. ;
# const q51 = 0.     ; const q61 =  0.    ;

# const q12 = -1/2.  ; const q22 = 0.     ;
# const q32 = 1/2.   ; const q42 = 0.     ;
# const q52 = 0.     ; const q62 = 0.     ;

# const q13 =  4/43. ; const q23 = -59/86.;
# const q33 =  0.    ; const q43 = 59/86. ;
# const q53 =  -4/43.; const q63 = 0.     ;

# const q14 = 3/98.  ; const q24 = 0.     ;
# const q34 = -59/98.; const q44 = 0.     ;
# const q54 = 32/49. ; const q64 = -4/49. ;

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
@inline h2_22(a) = 1 - a^2/2

@inline Q2_12(a) = (a+1)/2

@inline q2_21(a) = -Q2_12(a) - el2_1(a)*el2_2(a)/2
@inline q2_22(a) = -el2_2(a)^2/2

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


@inline fψ(U::WaveCell) = U.ψ

@inline ψu(U::WaveCell) = -U.Ψ # Scalar gradient-flux

@inline vx(U::WaveCell) = U.ψx # vector gradient-flux

@inline vy(U::WaveCell) = U.ψy # vector gradient-flux

@inline vz(U::WaveCell) = U.ψz # vector gradient-flux

Base.@propagate_inbounds @inline function Dx(f,Um,ns,nls,nrs,αls,αrs,i,j,k)  # x-derivative
    nl,nr = (nls[1], nrs[1])
    nlr = (nl, nr)
    αs = (αls[1], αrs[1])
    @inbounds @inline U(x::Int) = f(getindex(Um,x,j,k))
    if nr-nl>=3 # not enough points for 4th order stencil, go to 2nd order
        D_2_1(U,nlr,αs,i)
    elseif nr-nl==2 # not enough points for 2nd order stencil, go to 1st order 3 point
        D_3point(U,nlr,αs,i)
    elseif nr-nl==1 # not enough points for 2nd order stencil, go to 1st order 2 point
        -U(nl) + U(nr) # note the two point operator happens to be the same for both points
    else # only one grid point, extrapolate derivative
        throw(0.) # not implemented
    end
end

Base.@propagate_inbounds @inline function Dy(f,Um,ns,nls,nrs,αls,αrs,i,j,k)  # y-derivative
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
        throw(0.) # not implemented
    end
end

Base.@propagate_inbounds @inline function Dz(f,Um,ns,nls,nrs,αls,αrs,i,j,k)  # z-derivative
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
        throw(0.) # not implemented
    end
end

# Base.@propagate_inbounds @inline function D_4_2(U,ns,αs,k)
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
        throw(0.)
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
        throw(0.)
    end
end

Base.@propagate_inbounds @inline function Dissipation(Um,ns,nls,nrs,αls,αrs,i,j,k)  # Dissipation in all directions
    @inbounds @inline Ux(x) = getindex(Um,x,j,k)
    @inbounds @inline Uy(x) = getindex(Um,i,x,k)
    @inbounds @inline Uz(x) = getindex(Um,i,j,x)

    D4(Ux,(nls[1],nrs[1]),(αls[1],αrs[1]),i) + D4(Uy,(nls[2],nrs[2]),(αls[2],αrs[2]),j) + D4(Uz,(nls[3],nrs[3]),(αls[3],αrs[3]),k)
end

Base.@propagate_inbounds @inline function Div(vx,vy,vz,U,ns,nls,nrs,αls,αrs,ds,i,j,k) # Calculate the divergence of the flux
    dx,dy,dz,_ = ds
    Dx(vx,U,ns,nls,nrs,αls,αrs,i,j,k)/dx + Dy(vy,U,ns,nls,nrs,αls,αrs,i,j,k)/dy + Dz(vz,U,ns,nls,nrs,αls,αrs,i,j,k)/dz
end

Base.@propagate_inbounds @inline function vectors(outer,rb,ri,ns,i,l)
    # Returns the null basis to the boundary
    # The embedded boundary method effectively treats the boundary as "Lego" objects
    # with boundary normals only in one of the 3 cardinal directions.
    # Retruns upper index vectors and a mixed index boundary 2-metric

    sx,sy,sz = (0.,0.,0)

    if     i==1
        (sx = l)
    elseif i==2 
        (sy = l)
    else#if i==3
         (sz = l)
    end

    norm = sqrt(sx^2 + sy^2 + sz^2)

    #if norm == 0.; println("what") end

    s = @Vec [0.,sx/norm,sy/norm,sz/norm]

    n = @Vec [1.,0.,0.,0.]

    ℓ = (n + s)/sqrt(2)
    k = (n - s)/sqrt(2)

    δ = one(StateTensor)

    σ = StateTensor((μ,ν) -> δ[μ,ν] + k[μ]*ℓ[ν] + ℓ[μ]*k[ν])

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

    outside = !(s==-1)

    return (outside,nl,nr,αl,αr)

end

Base.@propagate_inbounds @inline function SAT(U,ns,ls,ds,nls,nrs,αls,αrs,r,ri)
    # Performs the boundary conditions in each direction
    # Since the StateVector on the boundary is not in the domain, its value must be extrapolated.
    # Since the boundary position is not in the domain, the application of boundary conditions
    # must also be extrapolated.

    lx,ly,lz = ls
    dx,dy,dz,_=ds

    xi,yi,zi=ri
    x,y,z = r

    SATψ = 0.; SATψx = 0.; SATψy = 0.; SATψz = 0.; SATΨ  = 0.; 

    for i in 1:3

        nl = nls[i]; nr = nrs[i];

        if nr-nl<3 
            #(println(nls,nrs); @assert false) 
            throw(0.)
        end

        if (ri[i]==nl==1) || (nl ≠ 1 && ri[i] in nl:nl+1) # in the boundary region on the left side of the line

            αl = αls[i]

            # Interpolate the solution vector on the boundary 
            # and determine the boundary position on the coordinate line
            if i == 1 # On an x-line
                rb = (-lx/2+(nl-1)*dx-αl*dx,y,z)
                Ub = el2_1(αl)*U[nl,yi,zi] + el2_2(αl)*U[nl+1,yi,zi]
            elseif i == 2 # On a y-line
                rb = (x,-ly/2+(nl-1)*dy-αl*dy,z)
                Ub = el2_1(αl)*U[xi,nl,zi] + el2_2(αl)*U[xi,nl+1,zi]
            elseif i == 3 # On a z-line
                rb = (x,y,-lz/2+(nl-1)*dz-αl*dz)
                Ub = el2_1(αl)*U[xi,yi,nl] + el2_2(αl)*U[xi,yi,nl+1]
            end

            outer = (nl == 1)

            k,ℓ,σ = vectors(outer,rb,ri,ns,i,-1.) # Form boundary basis

            ψxb = Ub.ψx
            ψyb = Ub.ψy
            ψzb = Ub.ψz
            Ψb  = Ub.Ψ

            ∂ψb = @Vec [-Ψb,ψxb,ψyb,ψzb]

            ∂ψbup = @Vec [Ψb,ψxb,ψyb,ψzb]

            Upb = (@einsum k[α]*∂ψb[α]) # + γ2*ψ
            U0b = @einsum σ[α,β]*∂ψb[β]

            if outer
                UmBC = -Upb
            else
                UmBC = Upb
            end

            # if outer
            #     UmBC = -Upb
            # elseif Upb == 0.
            #     UmBC = 0.
            # else    
            #     UmBC = (@einsum ∂ψbup[α]*U0b[α])/Upb/2
            # end

            ΨBC  = -(Upb + UmBC)/sqrt(2)
            ∂ψBC =  U0b - k*UmBC - ℓ*Upb

            if ri[i] == nl 
                ε = el2_1(αl)/h2_11(αl)/ds[i]
            elseif ri[i] == nl+1
                ε = el2_2(αl)/h2_22(αl)/ds[i]
            end

            # s = (k-ℓ)/sqrt(2)

            # sbx = -x; sby = -y; sbz = -z; 

            # norm = sqrt(sbx^2 + sby^2 + sbz^2)

            # sb = @Vec [0.,sbx/norm,sby/norm,sbz/norm]

            # ε *= abs(@einsum s[μ]*sb[μ])

            SATΨ  += ε*(ΨBC - Ψb)
            SATψx += ε*(∂ψBC[2] - ψxb)
            SATψy += ε*(∂ψBC[3] - ψyb)
            SATψz += ε*(∂ψBC[4] - ψzb)

        end

        if (ri[i]==nr==ns[i]) || (nr ≠ ns[i] && ri[i] in (nr-1):nr) # in the boundary region on the right side of the line

            αr = αrs[i]

            # Interpolate the solution vector on the boundary 
            # and determine the boundary position on the coordinate line
            if i == 1 # On an x-line
                rb = (-lx/2+(nr-1)*dx+αr*dx,y,z)
                Ub = el2_1(αr)*U[nr,yi,zi] + el2_2(αr)*U[nr-1,yi,zi]
            elseif i == 2 # On a y-line
                rb = (x,-ly/2+(nr-1)*dy+αr*dy,z)
                Ub = el2_1(αr)*U[xi,nr,zi] + el2_2(αr)*U[xi,nr-1,zi]
            elseif i == 3 # On a z-line
                rb = (x,y,-lz/2+(nr-1)*dz+αr*dz)
                Ub = el2_1(αr)*U[xi,yi,nr] + el2_2(αr)*U[xi,yi,nr-1]
            end

            outer =  (nr == ns[i])
  
            k,ℓ,σ = vectors(outer,rb,ri,ns,i,1.) # Form boundary basis

            ψxb = Ub.ψx
            ψyb = Ub.ψy
            ψzb = Ub.ψz
            Ψb  = Ub.Ψ

            ∂ψb = @Vec [-Ψb,ψxb,ψyb,ψzb]

            ∂ψbup = @Vec [Ψb,ψxb,ψyb,ψzb]

            Upb = (@einsum k[α]*∂ψb[α]) #+ γ2*ψ
            U0b = @einsum σ[α,β]*∂ψb[β]

            if outer
                UmBC = -Upb
            else
                UmBC = Upb
            end

            # if outer
            #     UmBC = -Upb
            # elseif Upb == 0.
            #     UmBC = 0.
            # else    
            #     UmBC = (@einsum ∂ψb[α]*U0b[α])/Upb/2
            # end

            ΨBC  = -(Upb + UmBC)/sqrt(2)
            ∂ψBC =  U0b - k*UmBC - ℓ*Upb

            if ri[i] == nr
                ε = el2_1(αr)/h2_11(αr)/ds[i]
            elseif ri[i] == nr-1
                ε = el2_2(αr)/h2_22(αr)/ds[i]
            end

            # s = (k-ℓ)/sqrt(2)

            # sbx = -x; sby = -y; sbz = -z; 

            # norm = sqrt(sbx^2 + sby^2 + sbz^2)

            # sb = @Vec [0.,sbx/norm,sby/norm,sbz/norm]

            # ε *= abs(@einsum s[μ]*sb[μ])

            SATΨ  += ε*(ΨBC - Ψb)
            SATψx += ε*(∂ψBC[2] - ψxb)
            SATψy += ε*(∂ψBC[3] - ψyb)
            SATψz += ε*(∂ψBC[4] - ψzb)

        end
        
    end

    return WaveCell(SATψ,SATψx,SATψy,SATψz,SATΨ)

end

function energy_cell(U,ns,ls,ds,ri) # calculates wave energy in a cell

    ψx = U.ψx
    ψy = U.ψy
    ψz = U.ψz
    Ψ  = U.Ψ

    xi,yi,zi = ri

    dx,dy,dz,_ = ds
    lx,ly,lz=ls
    x = -lx/2 + (xi-1)*dx; y = -ly/2 + (yi-1)*dy; z = -lz/2 + (zi-1)*dz;
    r = (x,y,z)

    outside,nls,nrs,αls,αrs = find_boundary(ns,ls,ds,r,ri)

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

    return (Ψ^2 + ψx^2 + ψy^2 + ψz^2)*h*dx*dy*dz

end


@parallel_indices (xi,yi,zi) function rhs!(U1::Data.CellArray,U2::Data.CellArray,U3::Data.CellArray,ns::Tuple,ls::Tuple,ds::Tuple,iter::Int)
    # Performs the right hand side of the system of equations.
    # The if blocks at the beginning and end perform the 3rd order Runge-Kutta algorithm
    # which is done by calling rhs! three times, each with a different value in iter ranging from 1:3

    if iter == 1
        U = U1
        Uw = U2
    elseif iter == 2
        U = U2
        Uw = U3
    else
        U = U3
        Uw = U1
    end

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

        ψ = Uxyz.ψ; ψx = Uxyz.ψx; ψy = Uxyz.ψy; ψz = Uxyz.ψz; Ψ  = Uxyz.Ψ

        γ2 = 1
        
        Cx = Dx(fψ,U,ns,nls,nrs,αls,αrs,xi,yi,zi)/dx - ψx
        Cy = Dy(fψ,U,ns,nls,nrs,αls,αrs,xi,yi,zi)/dy - ψy
        Cz = Dz(fψ,U,ns,nls,nrs,αls,αrs,xi,yi,zi)/dz - ψz

        ∂tψ = -Ψ

        ∂tψx = Dx(ψu,U,ns,nls,nrs,αls,αrs,xi,yi,zi)/dx + γ2*Cx

        ∂tψy = Dy(ψu,U,ns,nls,nrs,αls,αrs,xi,yi,zi)/dy + γ2*Cy

        ∂tψz = Dz(ψu,U,ns,nls,nrs,αls,αrs,xi,yi,zi)/dz + γ2*Cz

        ∂tΨ  = -Div(vx,vy,vz,U,ns,nls,nrs,αls,αrs,ds,xi,yi,zi)

        ∂tU  = WaveCell(∂tψ,∂tψx,∂tψy,∂tψz,∂tΨ)

        ∂tU += SAT(U,ns,ls,ds,nls,nrs,αls,αrs,r,ri)

        ∂tU += -0.005*Dissipation(U,ns,nls,nrs,αls,αrs,xi,yi,zi)

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


        Uw[xi,yi,zi] = Zero
        

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
    nt         = Int(1000*scale)                 # number of timesteps

    SAVE = false
    VISUAL = false
    nout       = Int(10*scale)                       # plotting frequency
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

    # test = WaveCell(1,2,3,4,5)
    # return test[2]

    # Initial Conditions
    #σ = 2.; x0 = lx/2; y0 = ly/2; z0 = lz/2;
    σ = 3.; x0 = 35.; y0 = 0.; z0 = 0.;
    @inline ψ_init(x,y,z) = exp(-((x-x0)^2+(y-y0)^2+(z-z0)^2)/σ^2)

    @inline ∂xψ(x,y,z) = ForwardDiff.derivative(x -> ψ_init(x,y,z), x)
    @inline ∂yψ(x,y,z) = ForwardDiff.derivative(y -> ψ_init(x,y,z), y)
    @inline ∂zψ(x,y,z) = ForwardDiff.derivative(z -> ψ_init(x,y,z), z)

    @inline ∂tψ(x,y,z) = 0.

    # Array allocations

    U1 = @zeros(ns..., celltype=WaveCell)
    U2 = @zeros(ns..., celltype=WaveCell)
    U3 = @zeros(ns..., celltype=WaveCell)

    if USE_GPU
        # If we use the GPU, we need an intermediate array on the CPU to save
        U0 = CPUCellArray{WaveCell}(undef, nx, ny, nz)
    end

    temp  = zeros(5,ns...)

    for xi in 1:ns[1], yi in 1:ns[2], zi in 1:ns[3]

        x = X[xi]
        y = Y[yi]
        z = Z[zi]

        ψ  = ψ_init(x,y,z)
        ψx =    ∂xψ(x,y,z)
        ψy =    ∂yψ(x,y,z)
        ψz =    ∂zψ(x,y,z)
        Ψ  =    ∂tψ(x,y,z)

        temp[:,xi,yi,zi] .= [ψ,ψx,ψy,ψz,Ψ]

    end

    for i in 1:5
        CellArrays.field(U1,i) .= Data.Array(temp[i,:,:,:])
    end

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
            @parallel (1:nx,1:ny,1:nz) rhs!(U1,U2,U3,ns,ls,ds,i) 
        end

        t = t + dt

        # Visualisation
        if VISUAL && (mod(it,nout)==0)
            A = Array(CellArrays.field(U1,1))[slice...]
            heatmap(X, Y, A, aspect_ratio=1, xlims=(-lx/2,lx/2)
            ,ylims=(-ly/2,ly/2),clim=(-0.1,0.1), c=:viridis,frame=:box)


            ang = range(0, 2π, length = 60)
            circle(x,y,r) = Shape(r*sin.(ang).+x, r*cos.(ang).+y)  

            plot!(circle(0,0,25),fc=:transparent, legend = false, 
            colorbar = true)

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
        close(datafile)
        throw(error)
    end

    close(datafile)

    # Performance metrics
    wtime    = Base.time() - wtime0
    bytes_accessed = sizeof(WaveCell)*(2*3)    # Bytes accessed per iteration per gridpoint (1 read and write iter=1 and 2 reads 1 write for iter=2,3)
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