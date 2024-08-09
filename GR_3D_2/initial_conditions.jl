# Initial Conditions for Kerr-Schild spinning black hole

#@@inline function (x,y,z,μ,ν)

@inline function g_init(x,y,z) # Spinning black hole in Kerr-Schild form
    # M = 0.5 # Mass of the hole
    # a = 0.5*M # Spin of the hole (0<a<M)
    #s = 1.  # Makes the hole black (+1) or white (-1)

    # M = mass
    # a = spin

    M = 4.0
    a = 0*M

    r = sqrt(x^2+y^2+z^2)

    A2 = (r^4*(r^2+a^2))/(r^4+a^2*z^2)
    B2 = (r^6*(x^2+y^2))/(r^4+a^2*z^2)
    C2 = (r^2*(x^2+y^2))/(r^2+a^2)

    R2 = (r^4+a^2*z^2)/(r^2+a^2)

    H = M*r^3/(r^4+a^2*z^2)

    t  = Vec{4}((1.,0.,0.,0.))
    ri = Vec{4}((0.,x,y,z))
    θi = Vec{4}((0.,-x*z,-y*z,x^2+y^2))
    ϕi = Vec{4}((0.,-y,x,0.))

    η = SymmetricSecondOrderTensor{4}((μ,ν) -> -t[μ]*t[ν] + ri[μ]*ri[ν]/A2 + θi[μ]*θi[ν]/B2 + ϕi[μ]*ϕi[ν]/C2  )

    l = Vec{4}((1.,x*R2/r^3 + a*y/r^2,y*R2/r^3 - a*x/r^2,z*R2/r^3))

    g = SymmetricSecondOrderTensor{4}((μ,ν) -> η[μ,ν] + 2*H*l[μ]*l[ν])

    return g

end

@inline function ∂tg_init(x,y,z) 
    ZeroST
end

# Derivatives of the initial conditions (don't use ForwardDiff, it allocates)   
# Use so-called complex-step differentiation so that it is accurate to machine precision                
@inline function ∂xgalt(x,y,z) 
    ϵ = 10*eps(Data.Number)
    StateTensor(imag(g_init(x+ϵ*im,y,z))/ϵ)
end

@inline function ∂ygalt(x,y,z) 
    ϵ = 10*eps(Data.Number)
    StateTensor(imag(g_init(x,y+ϵ*im,z))/ϵ)
end

@inline function ∂zgalt(x,y,z) 
    ϵ = 10*eps(Data.Number)
    StateTensor(imag(g_init(x,y,z+ϵ*im))/ϵ)
end

@inline ∂xg(x,y,z) = ForwardDiff.derivative(x -> g_init(x,y,z), x)
@inline ∂yg(x,y,z) = ForwardDiff.derivative(y -> g_init(x,y,z), y)
@inline ∂zg(x,y,z) = ForwardDiff.derivative(z -> g_init(x,y,z), z)

# Gauge functions and derivatives (freely specifyable)
#@inline fH_(x,y,z,μ) = (0.,0.,0.,0.)[μ] # lower index

@inline function fH_(x,y,z) 

    g = g_init(x,y,z)

    ∂tg = ∂tg_init(x,y,z) 
    dx  = ∂xg(x,y,z) 
    dy  = ∂yg(x,y,z) 
    dz  = ∂zg(x,y,z) 

    gi = inv(g)

    ∂g = Tensor{Tuple{4,@Symmetry{4,4}}}((σ,μ,ν) -> (∂tg[μ,ν], dx[μ,ν], dy[μ,ν], dz[μ,ν])[σ])

    Γ  = Tensor{Tuple{4,@Symmetry{4,4}}}((σ,μ,ν) -> 0.5*(∂g[ν,μ,σ] + ∂g[μ,ν,σ] - ∂g[σ,μ,ν]))    

    H_ = @einsum gi[ϵ,σ]*Γ[μ,ϵ,σ]

    return SVector{4}(H_.data...)

    #return FourVector((0,0,0,0))

    # M = 10.0

    # r = sqrt(x^2+y^2+z^2)

    # return (2*M/r^2,2*M*x/r^3,2*M*y/r^3,2*M*z/r^3)[μ]

end

# @inline ∂tH(x,y,z) = 0. # FourVector((0,0,0,0))

# @inline function ∂xH(x,y,z) 
#     ϵ = 10*eps(Data.Number)
#     Data.Number(imag(fH_(x+ϵ*im,y,z))/ϵ)
# end

# @inline function ∂yH(x,y,z) 
#     ϵ = 10*eps(Data.Number)
#     Data.Number(imag(fH_(x,y+ϵ*im,z))/ϵ)
# end

# @inline function ∂zH(x,y,z) 
#     ϵ = 10*eps(Data.Number)
#     Data.Number(imag(fH_(x,y,z+ϵ*im))/ϵ)
# end

@inbounds @inline function fUmBC(ℓ::FourVector,x::Data.Number,y::Data.Number,z::Data.Number) 
    StateTensor(ℓ[1]*∂tg_init(x,y,z) + ℓ[2]*∂xgalt(x,y,z) + ℓ[3]*∂ygalt(x,y,z) + ℓ[4]*∂zgalt(x,y,z))
end

# @inline ∂xH(x,y,z) = ForwardDiff.derivative(x -> fH_(x,y,z), x)
# @inline ∂yH(x,y,z) = ForwardDiff.derivative(y -> fH_(x,y,z), y)
# @inline ∂zH(x,y,z) = ForwardDiff.derivative(z -> fH_(x,y,z), z)

@inline function f∂H_(x,y,z) 

    ∂tH = (0,0,0,0)
    ∂xH = ForwardDiff.derivative(x -> fH_(x,y,z), x).data
    ∂yH = ForwardDiff.derivative(y -> fH_(x,y,z), y).data
    ∂zH = ForwardDiff.derivative(z -> fH_(x,y,z), z).data

    return SVector{10}(symmetric(FourTensor((∂tH...,∂xH...,∂yH...,∂zH...))).data...)

end