
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

    D4(Ux,(nls[1],nrs[1]),(αls[1],αrs[1]),xi)/dx + D4(Uy,(nls[2],nrs[2]),(αls[2],αrs[2]),yi)/dy + D4(Uz,(nls[3],nrs[3]),(αls[3],αrs[3]),zi)/dz
end

Base.@propagate_inbounds @inline function Div(vx,vy,vz,U,ns,nls,nrs,αls,αrs,ds,i,j,k) # Calculate the divergence of the flux
    dx,dy,dz,_ = ds
    (Dx(vx,U,ns,nls,nrs,αls,αrs,i,j,k)/dx + Dy(vy,U,ns,nls,nrs,αls,αrs,i,j,k)/dy + Dz(vz,U,ns,nls,nrs,αls,αrs,i,j,k)/dz)/rootγ(U[i,j,k])
end