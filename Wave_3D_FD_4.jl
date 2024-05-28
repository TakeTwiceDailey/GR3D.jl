module GR3D

const USE_GPU = false  # Use GPU? If this is set false, then no GPU needs to be available
using ParallelStencil
using CellArrays
using InteractiveUtils

ParallelStencil.@reset_parallel_stencil()

@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
end

using Plots, Printf, Statistics, BenchmarkTools, ForwardDiff

using Tensorial, Roots

# Alias for SymmetricSecondOrderTensor 2x2
const TwoTensor{T} = SymmetricSecondOrderTensor{2,T,3}

# Alias for non-symmetric 3 tensor
const ThreeTensor{T} = SecondOrderTensor{3,T,9}

# Alias for non-symmetric 3 tensor
const StateTensor = SymmetricSecondOrderTensor{4,Data.Number,10}

# Alias for StateVector of a scalar field
const StateScalar{T} = Vec{4,T}

# Alias for tensor to hold metric derivatives and Christoffel Symbols
# Defined to be symmetric in the last two indices
const Symmetric3rdOrderTensor{T} = Tensor{Tuple{3,@Symmetry{3,3}},T,3,18}


@CellType WaveCell fieldnames=(ψ,ψx,ψy,ψz,Ψ)

const NotANumber = WaveCell(NaN,NaN,NaN,NaN,NaN)
const Zero = WaveCell(0.,0.,0.,0.,0.)

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
@inline el1(a) = (a+2)*(a+1)/2
@inline el2(a) = -a*(a+2)
@inline el3(a) = a*(a+1)/2

# Norm coefficents
@inline h11(a) = 17/48 + a + 11/12*a^2 + 1/3*a^3 + 1/24*a^4
@inline h22(a) = 59/48 - 3/2*a^2 - 5/6*a^3 - 1/8*a^4
@inline h33(a) = 43/48 + 3/4*a^2 + 2/3*a^3 + 1/8*a^4
@inline h44(a) = 49/48 - 1/6*a^2 - 1/6*a^3 - 1/24*a^4

# Q + Q^T = 0 coefficents
@inline Q12(a) = 7/12*a^2 + a + 1/48*a^4 + 1/6*a^3 + 59/96 
@inline Q13(a) = -1/12*a^4 - 5/12*a^3 - 7/12*a^2 - 1/4*a - 1/12 
@inline Q14(a) = 1/16*a^4 + 1/4*a^3 + 1/4*a^2 - 1/32

@inline Q23(a) = 3/16*a^4 + 5/6*a^3 + 3/4*a^2 + 59/96
@inline Q24(a) = -1/6*a^2*(a + 2)^2

@inline Q34(a) = 5/48*a^4 + 5/12*a^3 + 5/12*a^2 + 59/96

# Finite difference coefficents 
# (I have kept the norm) part out for speed
@inline q11(a) = -el1(a)^2/2
@inline q12(a) = Q12(a) - el1(a)*el2(a)/2
@inline q13(a) = Q13(a) - el1(a)*el3(a)/2
@inline q14(a) = Q14(a)

@inline q21(a) = -Q12(a) - el1(a)*el2(a)/2
@inline q22(a) = -el2(a)^2/2
@inline q23(a) = Q23(a) - el2(a)*el3(a)/2
@inline q24(a) = Q24(a)

@inline q31(a) = -Q13(a) - el1(a)*el3(a)/2
@inline q32(a) = -Q23(a) - el2(a)*el3(a)/2
@inline q33(a) = -el3(a)^2/2
@inline q34(a) = Q34(a)

@inline q41(a) = -Q14(a)
@inline q42(a) = -Q24(a)
@inline q43(a) = -Q34(a)

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


# @parallel_indices (i,y) function rhs!(Type,U1,U2,U3,U_init,C1,C2,C3,H,∂H,rm,θm,t,ns,dt,_ds,iter)

#     #Explicit slices from main memory
#     # At each iteration in an Runge-Kutta algorithm,
#     # a U-read (U) and U-write (Uw) are defined
#     if iter == 1
#         # U3 has past iteration.
#         U = U1
#         Uw = U2
#         Uxy = U[x,y]
#     elseif iter == 2
#         # U1 has past iteration.
#         U = U2
#         Uw = U3
#         Uxy = U[x,y]
#     else
#         # U2 has past iteration.
#         U = U3
#         Uw = U1
#         Uxy = U[x,y]
#     end

#     Hxy = H[x,y]; ∂Hxy = ∂H[x,y];

#     r = rm[x,y]; θ = θm[x,y];

#     ρ = Uxy.ρ

#     ψ,ψr,ψθ,Ψ,g,dr,dθ,P = unpack(Uxy,false)

#     # Calculate inverse metric components
#     gi = inverse(g)

#     # Calculate lapse and shift
#     α  = 1/sqrt(-gi[1,1])
#     βr = -gi[1,2]/gi[1,1]
#     βθ = -gi[1,3]/gi[1,1]

#     # Time derivatives of the metric
#     ∂tg = βr*dr + βθ*dθ - α*P

#     nt = 1.0/α; nr = -βr/α; nθ = -βθ/α; 

#     n = @Vec [nt,nr,nθ]

#     n_ = @Vec [-α,0.0,0.0]

#     γi = gi + symmetric(@einsum n[μ]*n[ν])

#     δ = one(ThreeTensor)

#     γm = δ + (@einsum n_[μ]*n[ν])

#     γ = g + symmetric(@einsum n_[μ]*n_[ν])

#     #Derivatives of the lapse and the shift 

#     # ∂tα = -0.5*α*(@einsum n[μ]*n[ν]*∂tg[μ,ν])
#     # ∂tβ = α*(@einsum γi[α,μ]*n[ν]*∂tg[μ,ν]) # result is a 3-vector

#     # Metric derivatives
#     ∂g = Symmetric3rdOrderTensor{Type}(
#         (σ,μ,ν) -> (σ==1 ? ∂tg[μ,ν] : σ==2 ? dr[μ,ν] : σ==3 ? dθ[μ,ν] : @assert false)
#         )

#     # Chistoffel Symbols (of the first kind, i.e. all covariant indices)
#     Γ  = Symmetric3rdOrderTensor{Type}((σ,μ,ν) -> 0.5*(∂g[ν,μ,σ] + ∂g[μ,ν,σ] - ∂g[σ,μ,ν]))

#     C_ = @einsum gi[ϵ,σ]*Γ[μ,ϵ,σ] - Hxy[μ] #- ∂ψ[α]

#     Cr = (Dρ2T(fg,U,r,θ,ns,_ds,x,y) - dr)
#     Cθ = (Dz2T(fg,U,r,θ,ns,_ds,x,y) - dθ)

#     Cψr = (Dρ2(fψ,U,r,θ,ns,_ds,x,y) - ψr)
#     Cψθ = (Dz2(fψ,U,r,θ,ns,_ds,x,y) - ψθ)

#     # Scalar Evolution
#     ######################################################################

#     γ1 = -1.;
#     γ0 =  1.# + 9/(ρ+1)
#     γ2 = 1.;

#     ∂tψ  = βr*ψr + βθ*ψθ - α*Ψ

#     ∂ψ   = @Vec [∂tψ,ψr,ψθ]
    
#     ∂trootγ = @einsum 0.5*γi[i,j]*∂tg[i,j]

#     #######################################################################
#     # Define Stress energy tensor and trace 
#     # T = zero(StateTensor{Type})
#     # Tt = 0.

#     ∂tP = -2*α*S  # + 8*pi*Tt*g - 16*pi*T 

#     #∂tP += -2*α*symmetric(@einsum (μ,ν) -> 2*∂ψ[μ]*∂ψ[ν])  # + 8*pi*Tt*g - 16*pi*T 

#     ∂tP += 2*α*symmetric(∂Hxy)

#     #∂tP -= 2*α*symmetric(@einsum (μ,ν) -> gi[ϵ,σ]*Hxy[ϵ]*∂g[μ,ν,σ])

#     ∂tP -= 2*α*symmetric(@einsum (μ,ν) -> gi[ϵ,σ]*Hxy[ϵ]*Γ[σ,μ,ν])

#     ∂tP -=  α*symmetric(@einsum (μ,ν) -> gi[λ,γ]*gi[ϵ,σ]*Γ[λ,ϵ,σ]*∂g[γ,μ,ν])

#     ∂tP += 2*α*symmetric(@einsum (μ,ν) -> gi[ϵ,σ]*gi[λ,ρ]*∂g[λ,ϵ,μ]*∂g[ρ,σ,ν])

#     ∂tP -= 2*α*symmetric(@einsum (μ,ν) -> gi[ϵ,σ]*gi[λ,ρ]*Γ[μ,ϵ,λ]*Γ[ν,σ,ρ])

#     # Constraint damping term for C_

#     ∂tP += γ0*α*symmetric(@einsum (μ,ν) -> 2C_[μ]*n_[ν] - g[μ,ν]*n[ϵ]*C_[ϵ])

#     # ρv = @Vec [0.,1.,0.]
#     # ρv_ = @einsum g[μ,ν]*ρv[μ]
    
#     # ∂tP += -2*γ0*α*exp(-ψ)*symmetric(@einsum (μ,ν) -> ρv_[μ]*n_[ν]*ρv[i]*C_[i])

#     # if (y==1 || y==ns[2])
    
#     # end

#     #∂tP += -2*α*symmetric(@einsum (μ,ν) -> 2*C_[μ]*∂ψ[ν] - g[μ,ν]*gi[α,β]*∂ψ[α]*C_[β])

#     #∂tP += γ0*α*exp(-ψ)*symmetric(@einsum (μ,ν) -> 2*n_[μ]*C_[ν] - g[μ,ν]*n[ϵ]*C_[ϵ])

#     #vt = @Vec [0.,ρ,0.]

#     #∂tP -= 2*α*g*(@einsum gi[μ,ν]*vt[μ]*C_[ν])

#     #∂tP -= 2*α*g*(@einsum gi[μ,ν]*∂ψ[μ]*C_[ν])

#     #∂tP -= γ0*α*g*exp(-ψ)*(@einsum n[ϵ]*C_[ϵ])

#     ∂tP -= ∂trootγ*P

#     ∂tP += γ1*γ2*(βr*Cr + βθ*Cθ)

#     ###########################################
#     # All finite differencing occurs here

#     # mask1 = StateTensor{Type}((1.,ρ,1.,1.,ρ,1.))
#     # mask2 = StateTensor{Type}((0.,1.,0.,0.,1.,0.))

#     ∂tP += Div4T(vr,vθ,U,r,θ,ns,_ds,x,y) 

#     ∂tdr = symmetric(Dρ4T(u,U,r,θ,ns,_ds,x,y)) + α*γ2*Cr #+ mask2.*u_odd(U) 

#     ∂tdθ = symmetric(Dz4T(u,U,r,θ,ns,_ds,x,y)) + α*γ2*Cθ

#     ##################################################

#     ∂tP = symmetric(∂tP)

#     ##################################################

#     ∂tψr = Dρ4(ψu,U,r,θ,ns,_ds,x,y) + α*γ2*Cψr

#     ∂tψθ = Dz4(ψu,U,r,θ,ns,_ds,x,y) + α*γ2*Cψθ

#     ∂tΨ  = Div4(ψvr,ψvθ,U,r,θ,ns,_ds,x,y) #- (α/4)*St 

#     #∂tΨ  -= α*(@einsum gi[i,j]*∂ψ[i]*C_[j]) 

#     #∂tΨ  += α*γ0*(@einsum n[i]*C_[i]) 

#     ∂tΨ  -= ∂trootγ*Ψ

#     # ∂tψ  = 0.
#     # ∂tψr = 0.
#     # ∂tψθ = 0.
#     # ∂tΨ  = 0.

#     ######################################################

#     ∂ρg = Dρ2T(fg,U,r,θ,ns,_ds,x,y)
#     ∂zg = Dz2T(fg,U,r,θ,ns,_ds,x,y)
#     ∂ρψ = Dρ2(fψ,U,r,θ,ns,_ds,x,y)
#     ∂zψ = Dz2(fψ,U,r,θ,ns,_ds,x,y)

#     # ∂tg = βr*∂ρg + βθ*∂zg - α*P

#     ###################################################
#     #Boundary conditions

#     c1 = (x==1); c2 = (x==ns[1]);

#     if (c1 || c2) #&& false

#         if c1 
#             b=1; p=-1
#         else 
#             b=2; p=1
#         end

#         Cy = C1[b,y]

#         h = g
#         hi = gi
#         ∂h = ∂g
    
#         α  = 1/sqrt(-hi[1,1])
#         βr = -hi[1,2]/hi[1,1]
#         βθ = -hi[1,3]/hi[1,1]
    
#         nt = 1.0/α; nr = -βr/α; nθ = -βθ/α; 
    
#         n = @Vec [nt,nr,nθ]
    
#         n_ = @Vec [-α,0.0,0.0]

#         # Form the unit normal vector to the boundary

#         s = @Vec [0.0,p*sin(θ),p*cos(θ)]

#         snorm = @einsum h[μ,ν]*s[μ]*s[ν]
    
#         s = s/sqrt(snorm) 

#         s_ = @einsum h[μ,ν]*s[ν]
    
#         # Form the unit tangent to the boundary

#         # if y == 1 || y== ns[2]
#         #     Θ_ = @Vec [0.,0.,0.]
#         # else

#         # end
#         #Θ = @Vec [0.0,r*cos(θ),-r*sin(θ)]

#         Θ_ = @Vec [βr*cos(θ)-βθ*sin(θ),cos(θ),-sin(θ)]
#         Θnorm = @einsum hi[μ,ν]*Θ_[μ]*Θ_[ν]
#         Θ_ = Θ_/sqrt(Θnorm)

#         Θ = @einsum hi[μ,ν]*Θ_[ν]

#         # Form ingoing and outgoing null vectors

#         ℓ = @einsum (n[α] + s[α])/sqrt(2)
#         k = @einsum (n[α] - s[α])/sqrt(2)

#         ℓ_ = @einsum h[μ,α]*ℓ[α]
#         k_ = @einsum h[μ,α]*k[α]

#         # σ = StateTensor{Type}((μ,ν) -> gi[μ,ν] + k[μ]*ℓ[ν] + ℓ[μ]*k[ν])
        
#         # σm = @einsum g[μ,α]*σ[α,ν] # mixed indices (raised second index)

#         # σ_ = @einsum g[μ,α]*σm[ν,α]

#         σ = @einsum Θ[μ]*Θ[ν]
        
#         σm = @einsum Θ_[μ]*Θ[ν] # mixed indices (raised second index)

#         σ_ = @einsum Θ_[μ]*Θ_[ν]

#         cp =  α - βr*s_[2] - βθ*s_[3]
#         cm = -α - βr*s_[2] - βθ*s_[3]
#         c0 =    - βr*s_[2] - βθ*s_[3]

#         # if y==1 && x==ns[1]
#         #     println(cp," ",cm," ",c0)
#         # end

#         βdotθ = βr*Θ_[2] + βθ*Θ_[3]
    
#         Up = @einsum StateTensor{Type} k[α]*∂h[α,μ,ν]
#         #Um = @einsum StateTensor{Type} ℓ[α]*∂h[α,μ,ν]
#         U0 = @einsum StateTensor{Type} Θ[α]*∂h[α,μ,ν]

#         # Up = -(P + s[2]*dr + s[3]*dθ)/sqrt(2)
#         # U0 = Θ[2]*dr + Θ[3]*dθ

#         Uψp = @einsum k[α]*∂ψ[α]
#         Uψ0 = @einsum Θ[α]*∂ψ[α]


#         # Q4 = SymmetricFourthOrderTensor{3,Type}(
#         #     (μ,ν,α,β) -> σ_[μ,ν]*σ[α,β] - 2*ℓ_[μ]*σm[ν,α]*k[β] + ℓ_[μ]*ℓ_[ν]*k[α]*k[β]
#         # ) # Four index constraint projector (indices down down up up)
    
#         Q3 = Symmetric3rdOrderTensor{Type}(
#             (α,μ,ν) -> ℓ_[μ]*σm[ν,α]/2 + ℓ_[ν]*σm[μ,α]/2 - σ_[μ,ν]*ℓ[α] - ℓ_[μ]*ℓ_[ν]*k[α]/2
#         ) # Three index constraint projector (indices up down down)
#         # Note order of indices here

#         # # O = SymmetricFourthOrderTensor{4}(
#         # #     (μ,ν,α,β) -> σm[μ,α]*σm[ν,β] - σ_[μ,ν]*σ[α,β]/2
#         # # ) # Gravitational wave projector

#         G = FourthOrderTensor{3,Type}(
#             (μ,ν,α,β) -> (2k_[μ]*ℓ_[ν]*k[α]*ℓ[β] - 2k_[μ]*σm[ν,α]*ℓ[β] + k_[μ]*k_[ν]*ℓ[α]*ℓ[β])
#         ) # Four index gauge projector (indices down down up up)

#         # G = minorsymmetric(G)

#         Amp = 0.00001
#         #Amp = 0.0
#         σ0 = 0.5
#         μ0 = 10.5

#         #f(t,z) = (μ0-t-σ0)<z<(μ0-t+σ0) ? (Amp/σ0^8)*(z-((μ0-t)-σ0))^4*(z-((μ0-t)+σ0))^4 : 0.

#         #f(t,z) = (μ0-t-σ0)<z<(μ0-t+σ0) ? Amp : 0.

#         f(t,ρ,z) = (μ0-t-σ0)<ρ<(μ0-t+σ0) ? Amp : 0.

#         if c2
#             Cf = @Vec [f(t,r*sin(θ),r*cos(θ)),r*sin(θ)*f(t,r*sin(θ),r*cos(θ)),f(t,r*sin(θ),r*cos(θ))]
#             #Cf = @Vec [0.,0.,0.] 
#             #Cf = C_
#         else
#             #Cf = Cy

#             # CBC = constraints(U[2,y]) 
#             Cf = @Vec [0.,0.,0.] 
#             #Cf = C_

#             #Cf = @Vec [f(-t,r*cos(θ)),0.,f(-t,r*cos(θ))]
#         end
    
#         A_ = @einsum (2*ℓ[μ]*Up[μ,α] - hi[μ,ν]*Up[μ,ν]*ℓ_[α] + hi[μ,ν]*U0[μ,ν]*Θ_[α] - 2*Θ[μ]*U0[μ,α] + 2*Hxy[α] + 2*Cf[α])
#         # # index down

#         # # Condition ∂tgμν = 0 on the boundary
#         Umb2 = (cp/cm)*Up + sqrt(2)*(βdotθ/cm)*U0
#         #Umb2 = ℓ[1]*∂tg + ℓ[2]*∂ρg + ℓ[3]*∂zg
#         #Umb2 = zero(StateTensor)


#         Umbh = @einsum StateTensor{Type} (Q3[α,μ,ν]*A_[α] + G[μ,ν,α,β]*Umb2[α,β])


#         Umb = symmetric(Umbh)

#         Pb  = -(Up + Umb)/sqrt(2)
#         # dxb = ∂ρg
#         # dyb = ∂zg
#         dxb = Θ_[2]*U0 - k_[2]*Umb - ℓ_[2]*Up
#         dyb = Θ_[3]*U0 - k_[3]*Umb - ℓ_[3]*Up 
        

#         ∂χ = @Vec [∂tχ,∂ρχ,∂zχ]

#         Uχp = @einsum k[α]*∂χ[α]
#         Uχ0 = @einsum Θ[α]*∂χ[α]

#         #Uχmb = (cp/cm)*Uχp + 2*(βdotθ/cm)*Uχ0
#         Uχmb = (cp/cm)*Uχp + 2*(βdotθ/cm)*Uχ0
#         #Uχmb = 0. #(Θ_[2]*Uχ0 - ℓ_[2]*Uχp)/k_[2]

#         Uψmb = Umb[2,2]/g[2,2]/4 + Uχmb/2

#         #Uψmb = (cp/cm)*Uψp + 2*(βdotθ/cm)*Uψ0

#         Ψb  = -(Uψp + Uψmb)/sqrt(2)
#         ψxb = Θ_[2]*Uψ0 - k_[2]*Uψmb - ℓ_[2]*Uψp
#         ψyb = Θ_[3]*Uψ0 - k_[3]*Uψmb - ℓ_[3]*Uψp 

#         #∂tψ = (Uψmb - ℓ[2]*ψxb - ℓ[3]*ψyb)/ℓ[1]
#         #∂tψ = (Uψmb - ℓ[2]*∂ρψ - ℓ[3]*∂zψ)/ℓ[1]

#         ##########################################################################

#         # ∂tα = -0.5*α*(@einsum n[μ]*n[ν]*∂tg[μ,ν])

#         # ∂tβ = α*(@einsum γi[α,μ]*n[ν]*∂tg[μ,ν]) # result is a 3-vector

#         # ∂t∂tg = (βr*∂tdr + βθ*∂tdθ - α*∂tP) + (∂tβ[2]*dr + ∂tβ[3]*dθ - ∂tα*P)

#         # ∂t∂g = Symmetric3rdOrderTensor{Type}(
#         #     (σ,μ,ν) -> (σ==1 ? ∂t∂tg[μ,ν] : σ==2 ? ∂tdr[μ,ν] : σ==3 ? ∂tdθ[μ,ν] : @assert false))

#         # ∂tΓ  = Symmetric3rdOrderTensor{Type}(
#         #     (σ,μ,ν) -> 0.5*(∂t∂g[ν,μ,σ] + ∂t∂g[μ,ν,σ] - ∂t∂g[σ,μ,ν])
#         #     )   

#         # ∂tH = Vec{3}((∂Hxy[1,:]...))
#         # ∂xH = Vec{3}((∂Hxy[2,:]...))
#         # ∂zH = Vec{3}((∂Hxy[3,:]...))

#         # ∂tC = (@einsum gi[ϵ,σ]*∂tΓ[λ,ϵ,σ] - gi[μ,ϵ]*gi[ν,σ]*Γ[λ,μ,ν]*∂tg[ϵ,σ]) - ∂tH

#         # # set up finite differencing for the constraints, by defining a function
#         # # that calculates the constraints for any x and y index. This
#         # # might not be the best idea, but should work.

#         # dxC = DρC(constraints,U,r,θ,ns,_ds,x,y) - ∂xH 
#         # dyC = DzC(constraints,U,r,θ,ns,_ds,x,y) - ∂zH 

#         # ∂C = ThreeTensor{Type}(
#         #     (σ,ν) ->  (σ==1 ? ∂tC[ν] : σ==2 ? dxC[ν] : σ==3 ? dyC[ν] : @assert false)
#         #     )

#         # UpC = @einsum k[α]*∂C[α,μ]
#         # U0C = @einsum Θ[α]*∂C[α,μ]

#         # UmbC = @Vec [0.,0.,0.] #(U0C).^(2)./UpC

#         # ∂tCb = zeroST(ρ)# Θ_[1]*U0C - k_[1]*UmbC - ℓ_[1]*UpC

#         #∂tCb = -γ0*C_

#         ε = 2*abs(cm)*_ds[1]

#         ∂tP  += ε*(Pb - P)
#         ∂tdr += ε*(dxb - dr)
#         ∂tdθ += ε*(dyb - dθ)
    
#         ∂tΨ  += ε*(Ψb - Ψ)
#         ∂tψr += ε*(ψxb - ψr)
#         ∂tψθ += ε*(ψyb - ψθ)

#     end

#     ∂tψv = @Vec [∂tψ,∂tψr,∂tψθ,∂tΨ]

#     ∂tU = StateVector{Type}(ρ,∂tψv,∂tg,∂tdr,∂tdθ,∂tP)

#     Dis = Dissipation(U,r,θ,x,y,ns)
#     Dis = StateVector{Type}(ρ,Dis.ψ,Dis.g,Dis.dr,Dis.dθ,Dis.P)

#     ∂tU += Dis

#     ##########################################################

#     # if ρ == 0.
#     #     mask1 = StateTensor{Type}((1.,0.,1.,1.,0.,1.))
#     #     mask2 = StateTensor{Type}((0.,1.,0.,0.,1.,0.))
#     #     mask3 = @Vec [1.,0.,1.,1.]
#     # else
#     #     mask1 = StateTensor{Type}((1.,1/ρ,1.,1.,1/ρ,1.))
#     #     mask2 = StateTensor{Type}((1/ρ,1.,1/ρ,1/ρ,1.,1/ρ))
#     #     mask3 = @Vec [1.,1/ρ,1.,1.]
#     # end

#     # ∂tU = StateVector{Type}(ρ,∂tψv,mask1.*∂tg,mask2.*∂tdr,mask1.*∂tdθ,mask1.*∂tP) #mask2.*mask3.*

#     #∂tU = StateVector{Type}(ρ,∂tψv,∂tg,∂tdr,∂tdθ,∂tP) #mask2.*mask3.*

#     #########################################################

#     # if iter == 1
#     #     U1t = unpack(Uxy)
#     #     Uwxy = U1t + dt*∂tU
#     # elseif iter == 2
#     #     U1t = unpack(U1[x,y])
#     #     U2t = unpack(Uxy)
#     #     Uwxy = (3/4)*U1t + (1/4)*U2t + (1/4)*dt*∂tU
#     # elseif iter == 3
#     #     U1t = unpack(U1[x,y])
#     #     U2t = unpack(Uxy)
#     #     Uwxy = (1/3)*U1t + (2/3)*U2t + (2/3)*dt*∂tU
#     # end

#     if iter == 1
#         U1t = Uxy
#         Uwxy = U1t + dt*∂tU
#     elseif iter == 2
#         U1t = U1[x,y]
#         U2t = Uxy
#         Uwxy = (3/4)*U1t + (1/4)*U2t + (1/4)*dt*∂tU
#     elseif iter == 3
#         U1t = U1[x,y]
#         U2t = Uxy
#         Uwxy = (1/3)*U1t + (2/3)*U2t + (2/3)*dt*∂tU
#     end

#     #Uw[x,y] = pack(Uwxy)

#     Uw[x,y] = Uwxy

#     return
    
# end

@inline fψ(U::WaveCell) = U.ψ

@inline function ψu(U::WaveCell) # Scalar gradient-flux
    
    # Give names to stored arrays from the state vector
    Ψ = U.Ψ

    # gi = inverse(g)

    # α = 1/sqrt(-gi[1,1])

    # βr = -gi[1,2]/gi[1,1]
    # βθ = -gi[1,3]/gi[1,1]

    return -Ψ

end

@inline function vx(U::WaveCell) # Scalar gradient-flux
    
    # Give names to stored arrays from the state vector
    ψx = U.ψx

    # gi = inverse(g)

    # α = 1/sqrt(-gi[1,1])

    # βr = -gi[1,2]/gi[1,1]
    # βθ = -gi[1,3]/gi[1,1]

    return ψx

end

@inline function vy(U::WaveCell) # Scalar gradient-flux
    
    # Give names to stored arrays from the state vector
    ψy = U.ψy

    # gi = inverse(g)

    # α = 1/sqrt(-gi[1,1])

    # βr = -gi[1,2]/gi[1,1]
    # βθ = -gi[1,3]/gi[1,1]

    return ψy

end

@inline function vz(U::WaveCell) # Scalar gradient-flux
    
    # Give names to stored arrays from the state vector
    ψz = U.ψz

    # gi = inverse(g)

    # α = 1/sqrt(-gi[1,1])

    # βr = -gi[1,2]/gi[1,1]
    # βθ = -gi[1,3]/gi[1,1]

    return ψz

end

Base.@propagate_inbounds @inline function Dx(f,Um,ns,ds,ls,nls,nrs,αls,αrs,r,i,j,k)
    nl,nr = (nls[1], nrs[1])
    nlr = (nl, nr)
    αs = (αls[1], αrs[1])
    dx,_,_,_ = ds
    @inline U(x) = f(getindex(Um,x,j,k))
    if nr-nl>=7 # enough points to fit a regular stencil
        D_4_2(U,nlr,αs,i)/dx
    elseif nr-nl>=3 # not enough points for 4th order stencil, go to 2nd order
        D_2_1(U,nlr,αs,i)/dx
    elseif nr-nl==2 # not enough points for 2nd order stencil, go to 1st order 3 point
        D_3point(U,nlr,αs,i)/dx
    elseif nr-nl==1 # not enough points for 2nd order stencil, go to 1st order 2 point
        (-U(nl) + U(nr))/dx # note the two point operator happens to be the same for both points
    elseif isnan(αs[1]) && isnan(αs[2]) # This is a trapped point outside of the domain
        trapped_point(f,Um,ns,ds,ls,nls,nrs,αls,αrs,r,i,j,k,1) #(f,Um,ns,ds,ls,nls,nrs,αls,αrs,r,i,j,k,m)
    else # only one grid point
        println(nl," ",nr," ",i," ",j," ",k)
        throw(0.) # not implemented
    end
    
end

Base.@propagate_inbounds @inline function Dy(f,Um,ns,ds,ls,nls,nrs,αls,αrs,r,i,j,k)
    nl,nr = (nls[2], nrs[2])
    nlr = (nl, nr)
    αs = (αls[2], αrs[2])
    _,dy,_,_ = ds
    @inline U(x) = f(getindex(Um,i,x,k))
    if nr-nl>=7
        D_4_2(U,nlr,αs,j)/dy
    elseif nr-nl>=3 # not enough points for 4th order stencil, go to 2nd order
        D_2_1(U,nlr,αs,j)/dy
    elseif nr-nl==2 # not enough points for 2nd order stencil, go to 1st order 3 point
        D_3point(U,nlr,αs,j)/dy
    elseif nr-nl==1 # not enough points for 2nd order stencil, go to 1st order 2 point
        (-U(nl) + U(nr))/dy # note the two point operator happens to be the same for both points
    elseif isnan(αs[1]) && isnan(αs[2]) # This is a trapped point outside of the domain
        trapped_point(f,Um,ns,ds,ls,nls,nrs,αls,αrs,r,i,j,k,2)
    else # only one grid point, extrapolate derivative
        throw(0.) # Not implemented
    end
    
end

Base.@propagate_inbounds @inline function Dz(f,Um,ns,ds,ls,nls,nrs,αls,αrs,r,i,j,k)
    nl,nr = (nls[3], nrs[3])
    nlr = (nl, nr)
    αs = (αls[3], αrs[3])
    _,_,dz,_ = ds
    @inline U(x) = f(getindex(Um,i,j,x))
    if nr-nl>=7
        D_4_2(U,nlr,αs,k)/dz
    elseif nr-nl>=3 # not enough points for 4th order stencil, go to 2nd order
        D_2_1(U,nlr,αs,k)/dz
    elseif nr-nl==2 # not enough points for 2nd order stencil, go to 1st order 3 point
        D_3point(U,nlr,αs,k)/dz
    elseif nr-nl==1 # not enough points for 2nd order stencil, go to 1st order 2 point
        (-U(nl) + U(nr))/dz # note the two point operator happens to be the same for both points
    elseif isnan(αs[1]) && isnan(αs[2]) # This is a trapped point outside of the domain
        trapped_point(f,Um,ns,ds,ls,nls,nrs,αls,αrs,r,i,j,k,3)
    else # only one grid point, extrapolate derivative
        throw(0.) # Not implemented
    end
    
end

Base.@propagate_inbounds @inline function D_4_2(U,ns,αs,k)
    nl,nr = ns
    αl,αr = αs
    if k in nl+4:nr-4
        (U(k-2) - 8*U(k-1) + 8*U(k+1) - U(k+2))/12
    elseif k==nl
        (q11(αl)*U(nl) + q12(αl)*U(nl+1) + q13(αl)*U(nl+2) + q14(αl)*U(nl+3))/h11(αl)
    elseif k==nl+1
        (q21(αl)*U(nl) + q22(αl)*U(nl+1) + q23(αl)*U(nl+2) + q24(αl)*U(nl+3))/h22(αl)
    elseif k==nl+2
        (q31(αl)*U(nl) + q32(αl)*U(nl+1) + q33(αl)*U(nl+2) + q34(αl)*U(nl+3) - U(nl+4)/12)/h33(αl)
    elseif k==nl+3
        (q41(αl)*U(nl) + q42(αl)*U(nl+1) + q43(αl)*U(nl+2) + (2/3)*U(nl+4) - U(nl+5)/12)/h44(αl)
    elseif k==nr
        -(q11(αr)*U(nr) + q12(αr)*U(nr-1) + q13(αr)*U(nr-2) + q14(αr)*U(nr-3))/h11(αr)
    elseif k==nr-1
        -(q21(αr)*U(nr) + q22(αr)*U(nr-1) + q23(αr)*U(nr-2) + q24(αr)*U(nr-3))/h22(αr)
    elseif k==nr-2
        -(q31(αr)*U(nr) + q32(αr)*U(nr-1) + q33(αr)*U(nr-2) + q34(αr)*U(nr-3) - U(nr-4)/12)/h33(αr)
    elseif k==nr-3
        -(q41(αr)*U(nr) + q42(αr)*U(nr-1) + q43(αr)*U(nr-2) + (2/3)*U(nr-4) - U(nr-5)/12)/h44(αr)
    else
        throw(0.)
    end
end

Base.@propagate_inbounds @inline function D_2_1(U,ns,αs,k)
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

Base.@propagate_inbounds @inline function D_3point(U,ns,αs,k)
    nl,nr = ns
    αl,αr = αs
    if k == nl
        (q11(αl,αr)*U(nl) + q12(αl,αr)*U(nl+1) + q13(αl,αr)*U(nr))/h11(αl,αr)
    elseif k==nr
        (q31(αl,αr)*U(nl) + q32(αl,αr)*U(nl+1) + q33(αl,αr)*U(nr))/h33(αl,αr)
    else
        (-U(nl) + U(nr))/2
    end
end

Base.@propagate_inbounds @inline function Div(vx,vy,vz,U,ns,ds,ls,nls,nrs,αls,αrs,r,i,j,k)
    Dx(vx,U,ns,ds,ls,nls,nrs,αls,αrs,r,i,j,k) + Dy(vy,U,ns,ds,ls,nls,nrs,αls,αrs,r,i,j,k) + Dz(vz,U,ns,ds,ls,nls,nrs,αls,αrs,r,i,j,k)
end

@inline function trapped_point(f,Um,ns,ds,ls,nls,nrs,αls,αrs,r,i,j,k,m)

    lx,ly,lz = ls
    dx,dy,dz,_ = ds

    ri = (i,j,k)

    @inline function D_i(m,i,j,k)#(f,Um,ns,ds,ls,nls,nrs,αls,αrs,r,i,j,k)
        if m==1
            r = (-lx/2 + (i-1)*dx,-ly/2 + (j-1)*dy, -lz/2 + (k-1)*dz)
            Dx(f,Um,ns,ds,ls,find_boundary(ns,ls,ds,r,(i,j,k))...,r,i,j,k)
        elseif m==2
            r = (-lx/2 + (i-1)*dx,-ly/2 + (j-1)*dy, -lz/2 + (k-1)*dz)
            Dy(f,Um,ns,ds,ls,find_boundary(ns,ls,ds,r,(i,j,k))...,r,i,j,k)
        elseif m==3
            r = (-lx/2 + (i-1)*dx,-ly/2 + (j-1)*dy, -lz/2 + (k-1)*dz)
            Dz(f,Um,ns,ds,ls,find_boundary(ns,ls,ds,r,(i,j,k))...,r,i,j,k)
        else
            throw(0.) # Not implemented
        end
    end

    @inline index(ind,dir) = if dir==1; (ind,j,k) elseif dir==2; (i,ind,k) else (i,j,ind) end

    if m==1
        i1 = 2
        i2 = 3
    elseif m==2
        i1 = 1
        i2 = 3
    else
        i1 = 1
        i2 = 2
    end

    c1 = !(nls[i1]==0 && nrs[i1]==0)
    c2 = !(nls[i2]==0 && nrs[i2]==0)

    if  c1 && c2  # both other directions are valid to extrapolate
        if nls[i1]==ri[i1]
            a = (3*D_i(m,index(ri[i1]+1,i1)...)-3*D_i(m,index(ri[i1]+2,i1)...)+D_i(m,index(ri[i1]+3,i1)...))
        elseif nrs[i1]==ri[i1]
            a = (3*D_i(m,index(ri[i1]-1,i1)...)-3*D_i(m,index(ri[i1]-2,i1)...)+D_i(m,index(ri[i1]-3,i1)...))
        else
            throw(0.)
        end
        if nls[i2]==ri[i2]
            b = (3*D_i(m,index(ri[i2]+1,i2)...)-3*D_i(m,index(ri[i2]+2,i2)...)+D_i(m,index(ri[i2]+3,i2)...))
        elseif nrs[i2]==ri[i2]
            b = (3*D_i(m,index(ri[i2]-1,i2)...)-3*D_i(m,index(ri[i2]-2,i2)...)+D_i(m,index(ri[i2]-3,i2)...))
        else
            throw(0.)
        end
        return (a+b)/2
    elseif c1 # this direction is not trapped and the other one is, extrapolate only this direction
        if nls[i1]==ri[i1]
            (3*D_i(m,index(ri[i1]+1,i1)...)-3*D_i(m,index(ri[i1]+2,i1)...)+D_i(m,index(ri[i1]+3,i1)...))
        elseif nrs[i1]==ri[i1]
            (3*D_i(m,index(ri[i1]-1,i1)...)-3*D_i(m,index(ri[i1]-2,i1)...)+D_i(m,index(ri[i1]-3,i1)...))
        else
            throw(0.)
        end
    elseif c2 # this direction is not trapped and the other one is, extrapolate only this direction
        if nls[i2]==ri[i2]
            (3*D_i(m,index(ri[i2]+1,i2)...)-3*D_i(m,index(ri[i2]+2,i2)...)+D_i(m,index(ri[i2]+3,i2)...))
        elseif nrs[i2]==ri[i2]
            (3*D_i(m,index(ri[i2]-1,i2)...)-3*D_i(m,index(ri[i2]-2,i2)...)+D_i(m,index(ri[i2]-3,i2)...))
        else
            throw(0.)
        end
    else
        println(nls," ",nrs," ",i," ",j," ",k)
        throw(0.)
    end
end

Base.@propagate_inbounds @inline function energy_cell(U,ds)

    ψx = U.ψx
    ψy = U.ψy
    ψz = U.ψz
    Ψ  = U.Ψ

    return (Ψ^2 + ψx^2 + ψy^2 + ψz^2)*ds[1]*ds[2]*ds[3]

end

Base.@propagate_inbounds @inline function vectors(outer,rb,ri,ns,i,l)
    # Returns the normal vector to the boundary.
    # Forms the normal vector to the boundary depending
    # If you are on a face, edge, or corner

    # lx,ly,lz=ls

    # X,Y,Z = -lx/2:dx:lx/2, -ly/2:dy:ly/2, -lz/2:dz:lz/2

    # x,y,z = X[xi],Y[yi],Z[zi]
    nx,ny,nz = ns

    sx,sy,sz = (0.,0.,0)

    if outer

        xi,yi,zi = ri

        if xi==1; (sx = -1.) elseif xi==nx; (sx = 1.) end
        if yi==1; (sy = -1.) elseif yi==ny; (sy = 1.) end
        if zi==1; (sz = -1.) elseif zi==nz; (sz = 1.) end

        #if nlr in (1,nx,ny,nz)

    else

        if i==1; (sx = l) end
        if i==2; (sy = l) end
        if i==3; (sz = l) end

        # x,y,z = rb

        # sx = -x; sy = -y; sz = -z; 
    end

    norm = sqrt(sx^2 + sy^2 + sz^2)

    if norm == 0.; println("what") end

    s = @Vec [0.,sx/norm,sy/norm,sz/norm]

    n = @Vec [1.,0.,0.,0.]

    ℓ = (n + s)/sqrt(2)
    k = (n - s)/sqrt(2)

    δ = one(StateTensor)

    σ = StateTensor((μ,ν) -> δ[μ,ν] + k[μ]*ℓ[ν] + ℓ[μ]*k[ν])

    return (k,ℓ,σ)
end

@inline function find_boundary(ns,ls,ds,r,ri)

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
        rb = find_zero(x->f(x,y,z), (x,x0), atol=a*dx, rtol=a*dx, Bisection())+lx/2
        temp,αrx = divrem(rb,dx,RoundNearest)
        αrx /= dx; xibr=Int(temp)+1;
    end

    i = xi-5
    x0 = x-5*dx
    if (i >= 1 && sign(f(x0,y,z))≠s)
        rb = find_zero(x->f(x,y,z), (x0,x), atol=a*dx, rtol=a*dx, Bisection())+lx/2
        temp,αlx = divrem(rb,dx,RoundNearest)
        αlx /= -dx; xibl=Int(temp)+1;
    end

    if s==-1 # Points outside the physical domain need special treatment
        if     xibr == xi; αlx = -αrx; xibl = xibr; αrx=0.; xibr = nx;
        elseif xibl == xi; αrx = -αlx; xibr = xibl; αlx=0.; xibl = 1;
        else # This direction is trapped
            αrx = NaN; xibr = 0; αlx=NaN; xibl = 0;
        end
    end

    # y-line
    i = yi+5
    y0 = y+5*dy
    if (i <= ny && sign(f(x,y0,z))≠s)
        rb = find_zero(y->f(x,y,z), (y,y0), atol=a*dy, rtol=a*dy, Bisection())+ly/2
        temp,αry = divrem(rb,dy,RoundNearest)
        αry /= dy; yibr=Int(temp)+1;
    end

    i = yi-5
    y0 = y-5*dy
    if (i >= 1 && sign(f(x,y0,z))≠s)
        rb = find_zero(y->f(x,y,z), (y0,y), atol=a*dy, rtol=a*dy, Bisection())+ly/2
        temp,αly = divrem(rb,dy,RoundNearest)
        αly /= -dy; yibl=Int(temp)+1;
    end

    if s==-1 # Points outside the physical domain need special treatment
        if     yibr == yi; αly = -αry; yibl = yibr; αry=0.; yibr = ny 
        elseif yibl == yi; αry = -αly; yibr = yibl; αly=0.; yibl = 1 
        else # This direction is trapped
            αry = NaN; yibr = 0; αly=NaN; yibl = 0;
        end
    end

    # z-line
    i = zi+5
    z0 = z+5*dz
    if (i <= nz && sign(f(x,y,z0))≠s)
        rb = find_zero(z->f(x,y,z), (z,z0), atol=a*dz, rtol=a*dz, Bisection())+lz/2
        temp,αrz = divrem(rb,dz,RoundNearest)
        αrz /= dz; zibr=Int(temp)+1;
    end

    i = zi-5
    z0 = z-5*dz
    if (i >= 1 && sign(f(x,y,z0))≠s)
        rb = find_zero(z->f(x,y,z), (z0,z), atol=a*dz, rtol=a*dz, Bisection())+lz/2
        temp,αlz = divrem(rb,dz,RoundNearest)
        αlz /= -dz; zibl=Int(temp)+1
    end

    if s==-1 # Points outside the physical domain need special treatment
        if     zibr == zi; αlz = -αrz; zibl = zibr; αrz=0.; zibr = nz;
        elseif zibl == zi; αrz = -αlz; zibr = zibl; αlz=0.; zibl = 1;
        else # This direction is trapped
            αrz = NaN; zibr = 0; αlz=NaN; zibl = 0;
        end
    end

    nl = (xibl,yibl,zibl)
    nr = (xibr,yibr,zibr)
    αl = (αlx,αly,αlz)      # this needs to be negative of the right ones.
    αr = (αrx,αry,αrz)

    #outside = !(s==-1)

    #return (outside,nl,nr,αl,αr)

    return (nl,nr,αl,αr)

end

Base.@propagate_inbounds @inline function SAT(U,ns,ls,ds,nls,nrs,αls,αrs,r,ri)

    lx,ly,lz = ls
    dx,dy,dz,_=ds

    xi,yi,zi=ri
    x,y,z = r

    SATψ = 0.; SATψx = 0.; SATψy = 0.; SATψz = 0.; SATΨ  = 0.; 

    for i in 1:3

        nl = nls[i]; nr = nrs[i];

        if nl==nr==0; break end

        if nr-nl<3 (println(nls,nrs); @assert false) end

        if (ri[i]==nl==1) || (nl ≠ 1 && ri[i] in nl:nl+1) # in the boundary region on the left side of the line

            αl = αls[i]

            # Interpolate the solution vector on the boundary 
            # and determine the boundary position on the coordinate line
            if i == 1 # On an x-line
                rb = (-lx/2+(nl-1)*dx-αl*dx,y,z)
                Ub = el1(αl)*U[nl,yi,zi] + el2(αl)*U[nl+1,yi,zi] + el3(αl)*U[nl+2,yi,zi]
            elseif i == 2 # On a y-line
                rb = (x,-ly/2+(nl-1)*dy-αl*dy,z)
                Ub = el1(αl)*U[xi,nl,zi] + el2(αl)*U[xi,nl+1,zi] + el3(αl)*U[xi,nl+2,zi]
            elseif i == 3 # On a z-line
                rb = (x,y,-lz/2+(nl-1)*dz-αl*dz)
                Ub = el1(αl)*U[xi,yi,nl] + el2(αl)*U[xi,yi,nl+1] + el3(αl)*U[xi,yi,nl+2]
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
                ε = el1(αl)/h11(αl)/ds[i]
            elseif ri[i] == nl+1
                ε = el2(αl)/h22(αl)/ds[i]
            elseif ri[i] == nl+2
                ε = el3(αl)/h33(αl)/ds[i]
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
                Ub = el1(αr)*U[nr,yi,zi] + el2(αr)*U[nr-1,yi,zi] + el3(αr)*U[nr-2,yi,zi]
            elseif i == 2 # On a y-line
                rb = (x,-ly/2+(nr-1)*dy+αr*dy,z)
                Ub = el1(αr)*U[xi,nr,zi] + el2(αr)*U[xi,nr-1,zi] + el3(αr)*U[xi,nr-2,zi]
            elseif i == 3 # On a z-line
                rb = (x,y,-lz/2+(nr-1)*dz+αr*dz)
                Ub = el1(αr)*U[xi,yi,nr] + el2(αr)*U[xi,yi,nr-1] + el3(αr)*U[xi,yi,nr-2]
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
                ε = el1(αr)/h11(αr)/ds[i]
            elseif ri[i] == nr-1
                ε = el2(αr)/h22(αr)/ds[i]
            elseif ri[i] == nr-2
                ε = el3(αr)/h33(αr)/ds[i]
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

# @inline function find_boundary(ns,ls,ds,r,ri)

#     xi,yi,zi = ri

#     lx,ly,lz=ls
#     dx,dy,dz,_=ds

#     #X,Y,Z = -lx/2:dx:lx/2, -ly/2:dy:ly/2, -lz/2:dz:lz/2

#     nx,ny,nz = ns

#     x,y,z = r

#     r = lx/4.

#     p = 2

#     f(x,y,z) = x^p + y^p + z^p - r^p
#     #f(x,y,z) = x + lx

#     xibl =  1; yibl =  1; zibl =  1;
#     xibr = nx; yibr = ny; zibr = nz;

#     αlx = 0.; αly = 0.; αlz = 0.;
#     αrx = 0.; αry = 0.; αrz = 0.;

#     s = sign(f(x,y,z))

#     #if s == 0 println("boundary"," ", x," ",y," ",z) end

#     a = 1/100

#     # Search for boundaries along the x-line
#     for i in 1:5
#         if xi+i > nx; break end
#         if sign(f(x+i*dx,y,z)) ≠ s
#             rb = find_zero(x->f(x,y,z), (x+(i-1)*dx, x+i*dx), atol=a*dx, rtol=a*dx, Bisection())+lx/2
#             temp,αrx = divrem(rb,dx,RoundNearest)
#             αrx /= dx; xibr=Int(temp)+1;
#             break
#         end
#     end

#     for i in -1:-1:-5
#         if xi+i < 1; break end
#         if sign(f(x+i*dx,y,z)) ≠ s
#             rb = find_zero(x->f(x,y,z), (x+i*dx,x+(i+1)*dx), atol=a*dx, rtol=a*dx, Bisection())+lx/2
#             temp,αlx = divrem(rb,dx,RoundNearest)
#             αlx /= -dx; xibl=Int(temp)+1;
#             break
#         end
#     end

#     if s==-1 # Points outside the physical domain need special treatment
#         if     xibr == xi; αlx = -αrx; xibl = xibr; αrx=0.; xibr = nx;
#         elseif xibl == xi; αrx = -αlx; xibr = xibl; αlx=0.; xibl = 1;
#         else # This direction is trapped
#             αrx = NaN; xibr = 0; αlx=NaN; xibl = 0;
#         end
#     end

#     # Search for boundaries along the y-line
#     for i in 1:5
#         if yi+i > ny; break end
#         if sign(f(x,y+i*dy,z)) ≠ s
#             rb = find_zero(y->f(x,y,z), (y+(i-1)*dy, y+i*dy), atol=a*dy, rtol=a*dy, Bisection())+ly/2
#             temp,αry = divrem(rb,dy,RoundNearest)
#             αry /= dy; yibr=Int(temp)+1;
#             break
#         end
#     end

#     for i in -1:-1:-5
#         if yi+i < 1; break end
#         if sign(f(x,y+i*dy,z)) ≠ s
#             rb = find_zero(y->f(x,y,z), (y+i*dy,y+(i+1)*dy), atol=a*dy, rtol=a*dy, Bisection())+ly/2
#             temp,αly = divrem(rb,dy,RoundNearest)
#             αly /= -dy; yibl=Int(temp)+1;
#             break
#         end
#     end

#     if s==-1 # Points outside the physical domain need special treatment
#         if     yibr == yi; αly = -αry; yibl = yibr; αry=0.; yibr = ny 
#         elseif yibl == yi; αry = -αly; yibr = yibl; αly=0.; yibl = 1 
#         else # This direction is trapped
#             αry = NaN; yibr = 0; αly=NaN; yibl = 0;
#         end
#     end

#     # Search for boundaries along the z-line
#     for i in 1:5
#         if zi+i > nz; break end
#         if sign(f(x,y,z+i*dz)) ≠ s
#             rb = find_zero(z->f(x,y,z), (z+(i-1)*dz, z+i*dz), atol=a*dz, rtol=a*dz, Bisection())+lz/2
#             temp,αrz = divrem(rb,dz,RoundNearest)
#             αrz /= dz; zibr=Int(temp)+1;
#             break
#         end
#     end

#     for i in (zi-1):-1:(zi-5)
#         if i < 1; break end
#         if sign(f(x,y,Z[i])) ≠ s
#             rb = find_zero(z->f(x,y,z), (Z[i],Z[i+1]), atol=a*dz, rtol=a*dz, Bisection())+lz/2
#             temp,αlz = divrem(rb,dz,RoundNearest)
#             αlz /= -dz; zibl=Int(temp)+1;
#             break
#         end
#     end

#     if s==-1 # Points outside the physical domain need special treatment
#         if     zibr == zi; αlz = -αrz; zibl = zibr; αrz=0.; zibr = nz;
#         elseif zibl == zi; αrz = -αlz; zibr = zibl; αlz=0.; zibl = 1;
#         else # This direction is trapped
#             αrz = NaN; zibr = 0; αlz=NaN; zibl = 0;
#         end
#     end

#     nl = (xibl,yibl,zibl)
#     nr = (xibr,yibr,zibr)
#     αl = (αlx,αly,αlz) # this needs to be negative of the right ones.
#     αr = (αrx,αry,αrz)

#     return (nl,nr,αl,αr)

# end

@parallel_indices (xi,yi,zi) function rhs!(U1::Data.CellArray,U2::Data.CellArray,U3::Data.CellArray,ns::Tuple,ls::Tuple,ds::Tuple,iter::Int)

    # T = Data.Number

    # xi = 2; yi = 2; zi = 2;

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
    #X,Y,Z = -lx/2:dx:lx/2, -ly/2:dy:ly/2, -lz/2:dz:lz/2
    x = -lx/2 + (xi-1)*dx; y = -ly/2 + (yi-1)*dy; z = -lz/2 + (zi-1)*dz;
    r = (x,y,z)
    nx,ny,nz = ns
    #ns = (nx,ny,nz)
    #coords = (X,Y,Z)

    ri = (xi,yi,zi)

    nls,nrs,αls,αrs = find_boundary(ns,ls,ds,r,ri)

    #inside = ((nls[1]<=xi<=nrs[1]) && (nls[2]<=yi<=nrs[2]) && (nls[3]<=zi<=nrs[3]))

    outside = (((nls[1]==1 || nls[2]==1 || nls[3]==1) || (nrs[1]==nx || nrs[2]==ny || nls[3]==nz)) 
            || ((xi==nls[1] || xi==nrs[1]) || (yi==nls[2] || yi==nrs[2]) || (zi==nls[3] || zi==nrs[3])) )

    if outside

    Uxyz = U[xi,yi,zi]

    ψ = Uxyz.ψ
    ψx = Uxyz.ψx
    ψy = Uxyz.ψy
    ψz = Uxyz.ψz
    Ψ  = Uxyz.Ψ

    γ2 = 1
    #(f,Um,ns,ds,ls,nls,nrs,αls,αrs,i,j,k)
    Cx = Dx(fψ,U,ns,ds,ls,nls,nrs,αls,αrs,r,xi,yi,zi) - ψx
    Cy = Dy(fψ,U,ns,ds,ls,nls,nrs,αls,αrs,r,xi,yi,zi) - ψy
    Cz = Dz(fψ,U,ns,ds,ls,nls,nrs,αls,αrs,r,xi,yi,zi) - ψz

    ∂tψ = -Ψ

    ∂tψx = Dx(ψu,U,ns,ds,ls,nls,nrs,αls,αrs,r,xi,yi,zi) + γ2*Cx

    ∂tψy = Dy(ψu,U,ns,ds,ls,nls,nrs,αls,αrs,r,xi,yi,zi) + γ2*Cy

    ∂tψz = Dz(ψu,U,ns,ds,ls,nls,nrs,αls,αrs,r,xi,yi,zi) + γ2*Cz

    ∂tΨ  = -Div(vx,vy,vz,U,ns,ds,ls,nls,nrs,αls,αrs,r,xi,yi,zi)

    ∂tU = WaveCell(∂tψ,∂tψx,∂tψy,∂tψz,∂tΨ)

    ∂tU += SAT(U,ns,ls,ds,nls,nrs,αls,αrs,r,ri)

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

        #Uw[xi,yi,zi] = NotANumber
        Uw[xi,yi,zi] = Zero

    end


    return

end

##################################################
@views function main()
    # Physics
    lx, ly, lz = 100.0, 100.0, 100.0  # domain extends
    ls = (lx,ly,lz)
    t  = 0.0               # physical start time

    # Numerics
    size = 100
    ns = (size,size,size)     # numerical grid resolution; should be a mulitple of 32-1 for optimal GPU perf
    nx, ny, nz = ns
    nt         = 200             # number of timesteps
    nout       = 10               # plotting frequency

    # Derived numerics
    dx, dy, dz = lx/(nx-1), ly/(ny-1), lz/(nz-1) # cell sizes
    CFL = 1/5.1
    dt = min(dx,dy,dz)*CFL
    ds = (dx,dy,dz,dt)

    coords = (-lx/2:dx:lx/2, -ly/2:dy:ly/2, -lz/2:dz:lz/2)
    #X, Y, Z = 0:dx:lx, 0:dy:ly, 0:dz:lz
    X,Y,Z = coords

    # ri = (50,10,54)
    # xi,yi,zi = ri
    # x = -lx/2 + (xi-1)*dx; y = -ly/2 + (yi-1)*dy; z = -lz/2 + (zi-1)*dz;
    # r = (x,y,z)

    # println(x," ",y," ",z)
    # return find_boundary(ns,ls,ds,r,(50,25,54))

    # println((el1(0.),el2(0.),el3(0.)))
    # return

    # Initial Conditions
    #σ = 2.; x0 = lx/2; y0 = ly/2; z0 = lz/2;
    A = 1.; σ = 2.; x0 = 35.; y0 = 0.; z0 = 0.;
    @inline ψ_init(x,y,z) = A*exp(-((x-x0)^2+(y-y0)^2+(z-z0)^2)/σ^2)

    @inline ∂xψ(x,y,z) = ForwardDiff.derivative(x -> ψ_init(x,y,z), x)
    @inline ∂yψ(x,y,z) = ForwardDiff.derivative(y -> ψ_init(x,y,z), y)
    @inline ∂zψ(x,y,z) = ForwardDiff.derivative(z -> ψ_init(x,y,z), z)

    @inline ∂tψ(x,y,z) = 0.

    # Array allocations

    U1 = @zeros(ns..., celltype=WaveCell)
    U2 = @zeros(ns..., celltype=WaveCell)
    U3 = @zeros(ns..., celltype=WaveCell)

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

    # ns = ((10,26),(1,ny),(1,nz))

    # αs = ((0.25,0.25),(0.25,0.25),(0.25,0.25))

    # return [-Dx(ψu,U1,ns,αs,i,1,1)/dx - 2*(-lx/2 + (i-1)*dx) for i in ns[1][1]:ns[1][2] ]

    #return @benchmark @parallel rhs!($U1,$U2,$U3,$ls,$ds,1)

    #return 0.

    # Preparation of visualisation
    ENV["GKSwstype"]="nul"; if isdir("viz3D_out")==false mkdir("viz3D_out") end; loadpath = "./viz3D_out/"; anim = Animation(loadpath,String[])
    old_files = readdir(loadpath; join=true)
    for i in 1:length(old_files) rm(old_files[i]) end
    println("Animation directory: $(anim.dir)")

    slice       = (:,:,Int(ceil(nz/2)))

    # bulk = (1:nx,1:ny,1:nz) #@parallel $bulk 
    # return @benchmark @parallel $bulk rhs!($U1,$U2,$U3,$ns,$ds,3)

    evec = zeros(0)

    # Time loop
    for it = 1:nt

        if (it==11)  global wtime0 = Base.time()  end

        bulk = (1:nx,1:ny,1:nz)

        # First stage (iter=1)

        @parallel bulk rhs!(U1,U2,U3,ns,ls,ds,1) 
 
        # Second stage (iter=2)
    
        @parallel bulk rhs!(U1,U2,U3,ns,ls,ds,2) 
    
        # Third stage (iter=3)
    
        @parallel bulk rhs!(U1,U2,U3,ns,ls,ds,3) 

        t = t + dt

        # Visualisation

        if (mod(it,nout)==0)
            A = Array(CellArrays.field(U1,1))[slice...]
            heatmap(X, Y, A, aspect_ratio=1, xlims=(X[1],X[end])
            ,ylims=(Y[1],Y[end]),clim=(-0.1,0.1), c=:viridis,frame=:box)


            ang = range(0, 2π, length = 60)
            circle(x,y,r) = Shape(r*sin.(ang).+x, r*cos.(ang).+y)  

            plot!(circle(0,0,25),fc=:transparent, legend = false, 
            colorbar = true)

            frame(anim)

            append!(evec,sum(U -> energy_cell(U,ds),U1))
        end

    end

    # Performance
    wtime    = Base.time() - wtime0
    bytes_accessed = sizeof(WaveCell)*(2*3)    # Bytes accessed per iteration per gridpoint (1 read and write iter=1 and 2 reads 1 write for iter=2,3)
    A_eff    = bytes_accessed*nx*ny*nz/1e9       # Effective main memory access per iteration [GB] 
    wtime_it = wtime/(nt-10)                     # Execution time per iteration [s]
    T_eff    = A_eff/wtime_it                    # Effective memory throughput [GB/s]

    @printf("Total steps=%d, time=%1.3e sec (@ T_eff = %1.2f GB/s) \n", nt, wtime, round(T_eff, sigdigits=3))
    gif(anim, "acoustic3D.gif", fps = 15)

    # if USE_GPU GC.gc(true) end

    GC.gc(true)

    # return 0.
    return plot((evec)./evec[1], ylim = (0, 2))

end

end