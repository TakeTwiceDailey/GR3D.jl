# Initial Conditions for Kerr-Schild spinning black hole

#@@inline function (x,y,z,μ,ν)

@inline function g_init(ρ,z) # Spinning black hole in Kerr-Schild form
    # M = 0.5 # Mass of the hole
    # a = 0.5*M # Spin of the hole (0<a<M)
    #s = 1.  # Makes the hole black (+1) or white (-1)

    # M = mass
    # a = spin

    M = 0.
    a = 0*M

    r = sqrt(ρ^2+z^2)

    R2 = (r^4+a^2*z^2)/(r^2+a^2)

    H = M*r^3/(r^4+a^2*z^2)

    #return SymmetricSecondOrderTensor{4}((-1,0,0,0,1,0,0,1,0,1))

    if ρ==0

        throw(ZeroST)

        # η1 = SymmetricSecondOrderTensor{4}((-1,0,0,0,1+a^2/z^2,0,0,1+a^2/z^2,0,1))

        # l1 = Vec{4}((1,0,0,z*R2/r^3))

        # g1 = SymmetricSecondOrderTensor{4}((μ,ν) -> η1[μ,ν] + 2*H*l1[μ]*l1[ν])

        return ZeroST

    else

        # A2 = (r^4*(r^2+a^2))/(r^4+a^2*z^2)
        # B2 = (r^6*ρ^2)/(r^4+a^2*z^2)
        # C2 = (r^2*ρ^2)/(r^2+a^2)

        # t  = Vec{4}((1,0,0,0))
        # ri = Vec{4}((0,ρ,z,0))
        # θi = Vec{4}((0,-ρ*z,ρ^2,0))
        # ϕi = Vec{4}((0,0,0,1))

        # η = SymmetricSecondOrderTensor{4}((μ,ν) -> -t[μ]*t[ν] + ri[μ]*ri[ν]/A2 + θi[μ]*θi[ν]/B2 + ϕi[μ]*ϕi[ν]/C2  )

        # l = Vec{4}((1,ρ*R2/r^3,z*R2/r^3,-a*ρ^2/r^2))

        # g = SymmetricSecondOrderTensor{4}((μ,ν) -> η[μ,ν] + 2*H*l[μ]*l[ν])

        η = SymmetricSecondOrderTensor{4}((-1,0,0,0,1,0,0,1,0,ρ^2))

        return η #g

    end

end

@inline function ∂tg_init(ρ,z) 
    ZeroST
end

# Derivatives of the initial conditions (don't use ForwardDiff, it allocates)   
# Use so-called complex-step differentiation so that it is accurate to machine precision                
@inline function ∂ρgalt(ρ,z) 
    ϵ = 10*eps(Data.Number)
    imag(g_init(ρ+ϵ*im,z))/ϵ
end

@inline function ∂zgalt(ρ,z) 
    ϵ = 10*eps(Data.Number)
    imag(g_init(ρ,z+ϵ*im))/ϵ
end

# @inline ∂xg(x,y,z) = ForwardDiff.derivative(x -> g_init(x,y,z), x)
# @inline ∂yg(x,y,z) = ForwardDiff.derivative(y -> g_init(x,y,z), y)
# @inline ∂zg(x,y,z) = ForwardDiff.derivative(z -> g_init(x,y,z), z)

@inline ∂ρg(ρ,z) = ∂ρgalt(ρ,z)
@inline ∂zg(ρ,z) = ∂zgalt(ρ,z)

# Gauge functions and derivatives (freely specifyable)
#@inline fH_(x,y,z,μ) = (0.,0.,0.,0.)[μ] # lower index

@inline function fH_(ρ,z) 

    g = g_init(ρ,z)

    ∂tg = ∂tg_init(ρ,z) 
    # dx = ForwardDiff.derivative(x -> g_init(x,y,z), x)
    # dy = ForwardDiff.derivative(y -> g_init(x,y,z), y)
    # dz = ForwardDiff.derivative(z -> g_init(x,y,z), z)
    dρ  = ∂ρg(ρ,z) 
    dz  = ∂zg(ρ,z) 

    gi = inv(g)

    ∂g = Tensor{Tuple{4,@Symmetry{4,4}}}((σ,μ,ν) -> (∂tg[μ,ν], dρ[μ,ν], dz[μ,ν], 0)[σ])

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
#     imag(fH_(x+ϵ*im,y,z))/ϵ
# end

# @inline function ∂yH(x,y,z) 
#     ϵ = 10*eps(Data.Number)
#     imag(fH_(x,y+ϵ*im,z))/ϵ
# end

# @inline function ∂zH(x,y,z) 
#     ϵ = 10*eps(Data.Number)
#     imag(fH_(x,y,z+ϵ*im))/ϵ
# end

@inbounds @inline function fUmBC(ℓ::FourVector,ρ::Data.Number,z::Data.Number) 
    StateTensor(ℓ[1]*∂tg_init(ρ,z) + ℓ[2]*∂ρgalt(ρ,z) + ℓ[4]*∂zgalt(ρ,z))
end

# @inbounds @inline function fUmBC(ℓ::FourVector,x::Data.Number,y::Data.Number,z::Data.Number) 
#     StateTensor(ℓ[1]*∂tg_init(x,y,z) + ℓ[2]*∂xg(x,y,z) + ℓ[3]*∂yg(x,y,z) + ℓ[4]*∂zg(x,y,z))
# end

# @inline ∂xH(x,y,z) = ForwardDiff.derivative(x -> fH_(x,y,z), x)
# @inline ∂yH(x,y,z) = ForwardDiff.derivative(y -> fH_(x,y,z), y)
# @inline ∂zH(x,y,z) = ForwardDiff.derivative(z -> fH_(x,y,z), z)

@inline function f∂H_(ρ,z) 

    ∂tH = (0,0,0,0)
    # ∂xHs = ForwardDiff.derivative(x -> fH_(x,y,z), x).data
    # ∂yHs = ForwardDiff.derivative(y -> fH_(x,y,z), y).data
    # ∂zHs = ForwardDiff.derivative(z -> fH_(x,y,z), z).data

    # ∂xHs = ∂xH(x,y,z) 
    # ∂yHs = ∂yH(x,y,z) 
    # ∂zHs = ∂zH(x,y,z) 

    ϵ = sqrt(eps(Data.Number))

    ∂ρHs = (fH_(ρ+ϵ,z) - fH_(ρ-ϵ,z) )/(2*ϵ)
    ∂zHs = (fH_(ρ,z+ϵ) - fH_(ρ,z-ϵ) )/(2*ϵ)

    return SVector{10}(symmetric(FourTensor((∂tH...,∂ρHs...,∂zHs...,0,0,0,0))).data...)

end