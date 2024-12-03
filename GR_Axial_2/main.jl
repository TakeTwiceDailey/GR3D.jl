module GR_Axial

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

using LaTeXStrings

ParallelStencil.@reset_parallel_stencil()

const USE_GPU = false  # Use GPU? If this is set false, then no GPU needs to be available

@static if USE_GPU
    @define_CuCellArray() # without this CuCellArrays don't work
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
    #@init_parallel_stencil(Polyester, Float64, 2)
end

include("types.jl")
include("initial_conditions.jl")
include("boundaries.jl")
include("differencing.jl")
include("measures.jl")

@parallel_indices (ρi,zi) function rhs!(U1,U2,U3,H,∂H,Rb,ns::Tuple,ls::Tuple,ds::Tuple,t::Data.Number,iter::Int)
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

    hρ,hz,dt = ds
    lρ,lz=ls

    ρ = hρ/2 + (ρi-1)*hρ
    z = -lz/2 + (zi-1)*hz


    r = (ρ,z)
    ri = (ρi,zi)

    in_domain,nls,nrs,αls,αrs = find_boundary(Rb,ns,ls,ds,r,ri)

    if in_domain

        Uxyz = U[ρi,zi]

        H_ = FourVector(H[ρi,zi].data) # FourVector(fH_(x,y,z))

        ∂H_ = StateTensor(∂H[ρi,zi].data) #StateTensor((μ,ν) -> 0.5*(f∂H_(x,y,z,μ,ν)+f∂H_(x,y,z,ν,μ)))

        g  = Uxyz.g
        dρ = Uxyz.dρ
        dz = Uxyz.dz
        P  = Uxyz.P

        gi = inv(g)

        α = 1/sqrt(-gi[1,1]) # Calculate lapse
    
        βρ = -gi[1,2]/gi[1,1] # Calculate shift vector
        βz = -gi[1,3]/gi[1,1]

        n  = FourVector((1,-βρ,-βz,0.))/α
        n_ = FourVector((-α,0.,0.,0.))

        γi = gi + symmetric(@einsum n[μ]*n[ν])

        ∂tg = βρ*dρ + βz*dz - α*P

        # collect metric derivatives
        ∂g = Symmetric3rdOrderTensor((σ,μ,ν) -> (∂tg[μ,ν],dρ[μ,ν],dz[μ,ν],0)[σ])

        # Calculate Christoffel symbols
        Γ  = Symmetric3rdOrderTensor((σ,μ,ν) -> 0.5*(∂g[ν,μ,σ] + ∂g[μ,ν,σ] - ∂g[σ,μ,ν]))

        ∂tlnrootγ = @einsum 0.5*γi[μ,ν]*∂tg[μ,ν]

        Γ_ = @einsum gi[ϵ,σ]*Γ[μ,ϵ,σ]

        #########################################################
        # Principle (linear) part of the evolution equations

        #try

        ∂tdρ = Dρ(u,U,ns,nls,nrs,αls,αrs,ρi,zi)/hρ

        ∂tdz = Dz(u,U,ns,nls,nrs,αls,αrs,ρi,zi)/hz

        ∂tP  = Div(vρ,vz,U,ns,nls,nrs,αls,αrs,ds,ρi,zi)

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

        γ0 = 1. # Harmonic constraint damping (>0)
        #γ1 = -1. # Linear Degeneracy parameter (=-1)
        γ2 = 1. # Derivative constraint damping (>0)

        Cρ = Dρ(fg,U,ns,nls,nrs,αls,αrs,ρi,zi)/hρ - dρ
        Cz = Dz(fg,U,ns,nls,nrs,αls,αrs,ρi,zi)/hz - dz

        ∂tdρ += γ2*α*Cρ
        ∂tdz += γ2*α*Cz

        # Generalized Harmonic constraints and damping

        C_ = Γ_ - H_

        ∂tP += γ0*symmetric(@einsum (μ,ν) -> 2C_[μ]*n_[ν] - g[μ,ν]*n[ϵ]*C_[ϵ]) #*α + C_[ν]*n_[μ]

        ∂tP += -γ2*(βρ*Cρ + βz*Cz) # γ1 = -1 just assumed here

        ########################################################

        ∂tU  = StateVector(∂tg,∂tdρ,∂tdz,∂tP)

        # Perform boundary conditions (12%)
        ∂tU += SAT(U,H,ns,ls,ds,nls,nrs,αls,αrs,r,t,ri,γ2)

        # if !in_bulk 
        # end

        # Add numerical dissipation (20%)
        ∂tU += -0.1*Dissipation(U,ns,ds,nls,nrs,αls,αrs,ρi,zi) #-0.1 α

        if iter == 1
            U1t = Uxyz
            Uwxyz = U1t + dt*∂tU
        elseif iter == 2
            U1t = U1[ρi,zi]
            U2t = Uxyz
            Uwxyz = (3/4)*U1t + (1/4)*U2t + (1/4)*dt*∂tU
        elseif iter == 3
            U1t = U1[ρi,zi]
            U2t = Uxyz
            Uwxyz = (1/3)*U1t + (2/3)*U2t + (2/3)*dt*∂tU
        end

        Uw[ρi,zi] = Uwxyz

    else # don't do anything if not in the computational domain


        Uw[ρi,zi] = NaNSV
        #Uw[ρi,zi] = MinSV
        

    end

    # 250 ms benchmark total
    return

end

@parallel_indices (ρi,zi) function test!(U1,U2,U3,Rb,ns,ds,ls,iter::Int)
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

    hρ,hz,dt = ds
    lρ,lz=ls

    ρ = hρ/2 + (ρi-1)*hρ
    z = -lz/2 + (zi-1)*hz

    r = (ρ,z)
    ri = (ρi,zi)

    in_domain,nls,nrs,αls,αrs = find_boundary(Rb,ns,ls,ds,r,ri)

    if in_domain

        g = U[ρi,zi].g

        if any(isnan.(g)); println(ri) end

        _,_,_,_,γs... = g.data

        gi = inv(g)

        if det(ThreeTensor(γs))<0
            println("γ ",(ρi,zi))
            throw(0.)
        elseif -gi[1,1]<0
            println("α ",(ρi,zi))
            throw(0.)
        end
    end

    return

end

@parallel_indices (ρi,zi) function initialize!(Uw,H,∂H,Rb,ns,ds,ls,constraint_init)
    # Initializes the initial conditions to arrays
    # the metric must be written to the array first if one desires constraint initialization
    
    hρ,hz,dt = ds
    lρ,lz=ls

    ρ = hρ/2 + (ρi-1)*hρ
    z = -lz/2 + (zi-1)*hz

    r = (ρ,z)
    ri = (ρi,zi)

    in_domain,nls,nrs,αls,αrs = find_boundary(Rb,ns,ls,ds,r,ri)

    if in_domain

        g = Uw[ρi,zi].g

        ∂tg = ∂tg_init(ρ,z)

        if constraint_init
            dρ = Dρ(fg,Uw,ns,nls,nrs,αls,αrs,ρi,zi)/hρ
            dz = Dz(fg,Uw,ns,nls,nrs,αls,αrs,ρi,zi)/hz
        else
            dρ  = ∂ρg(ρ,z)           
            dz  = ∂zg(ρ,z)
        end

        gi = inv(g)

        #if -gi[1,1] < 0; println((xi,yi,zi)) end

        α = 1/sqrt(-gi[1,1])
    
        βρ = -gi[1,2]/gi[1,1]
        βz = -gi[1,3]/gi[1,1]

        n_ = FourVector((-α,0,0,0))

        n = @einsum gi[μ,ν]*n_[ν]

        δ = one(FourTensor)

        γi = gi + symmetric(@einsum n[μ]*n[ν])

        γm = δ + (@einsum n_[μ]*n[ν])

        γ = g + symmetric(@einsum n_[μ]*n_[ν])

        P2 = -(∂tg - βρ*dρ - βz*dz)/α

        # ∂g = Symmetric3rdOrderTensor((σ,μ,ν) -> (∂tg[μ,ν], dx[μ,ν], dy[μ,ν], dz[μ,ν])[σ])

        # Γ  = Symmetric3rdOrderTensor((σ,μ,ν) -> 0.5*(∂g[ν,μ,σ] + ∂g[μ,ν,σ] - ∂g[σ,μ,ν])) 

        H[ρi,zi]  = fH_(ρ,z)

        ∂H[ρi,zi] = f∂H_(ρ,z)

        if constraint_init

            ∂g = Symmetric3rdOrderTensor((σ,μ,ν) -> (∂tg[μ,ν], dρ[μ,ν], dz[μ,ν], 0)[σ])
            
            G = minorsymmetric(NSProjection((μ,ν,α,β)-> γm[μ,α]*γm[ν,β] - n_[μ]*n_[ν]*γi[α,β]))

            C = Symmetric3rdOrderTensor((α,μ,ν) -> n_[μ]*γm[ν,α]/2 + n_[ν]*γm[μ,α]/2 - n_[μ]*n_[ν]*n[α])

            #Cf_ = FourVector((0,0,0,0))

            μ0 = 12
            σ0 = 2
            Amp = 0.1

            r=sqrt(ρ^2+z^2)

            model = (μ0-σ0)<r<(μ0+σ0) ? (Amp/σ0^8)*(r-(μ0-σ0))^4*(r-(μ0+σ0))^4 : 0.

            Cf_ = 0*model*n_

            H_ = FourVector(fH_(ρ,z).data) 

            A_ = @einsum ( 2*γi[i,μ]*∂g[i,μ,α] - gi[μ,ν]*γm[α,i]*∂g[i,μ,ν] - 2*H_[α] - 2*Cf_[α])
            # index down

            P = symmetric(@einsum C[α,μ,ν]*A_[α] + G[μ,ν,α,β]*P2[α,β])
        else
            P = P2
        end

        #P = P2

        Uw[ρi,zi] = StateVector(g,dρ,dz,P)

    else

        Uw[ρi,zi] = NaNSV
        H[ρi,zi]  = SVector{4}(NaN,NaN,NaN,NaN)
        ∂H[ρi,zi] = SVector{10}(NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN)

    end

    
    return

end

##################################################
@views function main()

    # Physics
    lρ, lz = 20.0, 20.0  # domain extends


    ls = (lρ,lz)
    t  = 0.0                          # physical start time
    tmax = 10.0

    # global mass = 0.
    # global spin = 0.5*mass

    # println("Horizon at: r_+ = ", round((1+sqrt(1-(spin/mass)^2))*mass,sigdigits=3))

    # Numerics
    num = 256 # 81,108,144,192,256
    ns = (num,num)             # numerical grid resolution; should be a mulitple of 32-1 for optimal GPU perf
    nρ, nz = ns

    Rb = 3.0 # boundary radius

    SAVE = false
    VISUAL = true
    constraint_init = true
    nout       = 1 #Int(10*scale)                       # plotting frequency
    noutsave   = 500 #Int(50*scale) 
    pdfout     = 4*nout #Int(10*scale) 

    save_size = 1
    step = Int(num/save_size)

    # Derived numerics
    hρ, hz = lρ/(nρ-1), lz/(nz-1) # cell sizes
    CFL = 1/5
    dt = min(hρ,hz)*CFL
    ds = (hρ,hz,dt)

    #return 0.05/hx

    nt = Int(round(tmax/dt, RoundUp))           # number of timesteps

    lρ += hρ/2

    coords = (hρ/2:hρ:lρ, -lz/2:hz:lz/2)

    R,Z = coords

    # return (length(R)," ", length(Z), " ",nρ," ", nz)


    # Array allocations
    # If we use the GPU, we need an intermediate array on the CPU to save
    if USE_GPU
        U0 = CPUCellArray{StateVector}(undef, nρ, nz)
        U1 =  CuCellArray{StateVector}(undef, nρ, nz)
        U2 =  CuCellArray{StateVector}(undef, nρ, nz)
        U3 =  CuCellArray{StateVector}(undef, nρ, nz)

        # H  =  CuCellArray{FourVector}(undef, nx, ny, nz)
        # ∂H =  CuCellArray{StateTensor}(undef, nx, ny, nz)
    else
        U0 =  CPUCellArray{StateVector}(undef, nρ, nz)
        U1 =  CPUCellArray{StateVector}(undef, nρ, nz)
        U2 =  CPUCellArray{StateVector}(undef, nρ, nz)
        U3 =  CPUCellArray{StateVector}(undef, nρ, nz)

        H  =  CPUCellArray{SVector{4,Data.Number}}(undef, nρ, nz)
        ∂H =  CPUCellArray{SVector{10,Data.Number}}(undef, nρ, nz)
    end

    if USE_GPU; Uw = U0 else Uw = U1 end

    # Sample the Kerr-Schild form of the metric only
    for ρi in 1:ns[1], zi in 1:ns[2]

        hρ,hz,dt = ds
        lρ,lz=ls

        ρ = hρ/2 + (ρi-1)*hρ
        z = -lz/2 + (zi-1)*hz
    
        r = (ρ,z)
        ri = (ρi,zi)
    
        in_domain,_,_,_,_ = find_boundary(Rb,ns,ls,ds,r,ri)

        if in_domain 
            g = StateTensor(g_init(ρ,z))
        else
            g = ZeroST
        end

        Uw[ρi,zi] = StateVector(g,ZeroST,ZeroST,ZeroST)

    end

    # write the rest of the state vector
    @parallel (1:nρ,1:nz) initialize!(Uw,H,∂H,Rb,ns,ds,ls,constraint_init)

    plotcolors = cgrad([RGB(0.148,0.33,0.54),RGB(0.509,0.4825,0.52),RGB(0.9,0.62,0.29),RGB(0.9625,0.77625,0.484),RGB(1,0.95,0.75)], [0.2,0.4,0.6,0.8])

    # zi = Int(ceil(nz/2))

    # B = [(U=U1[xi,yi,zi]; if any(isnan.(U.g)); NaN else U.g[2,2] end) for xi in 1:nx, yi in 1:ny]

    # #B = [(Hp=H[xi,yi,zi]; if any(isnan.(Hp)); NaN else Hp[2] end) for xi in 1:nx, yi in 1:ny]

    # p = heatmap(Y, X, B, aspect_ratio=1, xlims=(-ly/2,ly/2)
    # ,ylims=(-lx/2,lx/2),clim=(0,5), title = "Time = "*string(round(t; digits=2)), c=plotcolors,frame=:box)

    # return p

    # copy to other arrays
    if USE_GPU; copy!(U1.data, U0.data) end

    copy!(U2.data, U1.data)
    copy!(U3.data, U1.data)

    copy!(U0.data, U1.data)

    # ri=(75,53,53)
    # xi,yi,zi = ri
    # x = -lx/2 + (xi-1)*hx; y = -ly/2 + (yi-1)*hy; z = -lz/2 + (zi-1)*hz;
    # rb=(x,y,z)
    # return @code_warntype vectors(U1[1,1,1],true,ns,ri,rb,1,1)

    #return @code_warntype fUmBC(FourVector((1.,1.,1.,1.)),1.,1.,1.,1,1)

    # ri=(75,53,53)
    # xi,yi,zi = ri
    # x = -lx/2 + (xi-1)*hx; y = -ly/2 + (yi-1)*hy; z = -lz/2 + (zi-1)*hz;
    # r=(x,y,z)
    # return @code_warntype find_boundary(ns,ls,ds,r,ri)

    # k,ℓ,σ,kn,ℓn,σn = vectors(MinST,false,(1.,1.,1.),1,-1) # Form boundary basis

    # return     (ℓ - k)/sqrt(2)

    #return @code_warntype vectors(U1[1,1,1],1,1.)

    # return @code_warntype fH_(1.,1.,1.,1)

    # ri=(1,1,1)
    # r=(1.,1.,1.)
    # return @code_warntype SAT(U1,ns,ls,ds,(1,1,1),ns,(0.,0.,0.),(0.,0.,0.),r,t,ri,1.)

    #return @code_warntype BoundaryConditions(U1[1,1,1],false,(1.,1.,1.),(@Vec [1.,1.,1.,1.]),1.,(@Vec [1.,1.,1.,1.]),(@Vec [1.,1.,1.,1.]),ZeroST)

    #return @code_warntype fUmBC((@Vec [1.,1.,1.,1.]),1.,1.,1.,1,1)

    #return @benchmark rhs!($U1,$U2,$U3,$ns,$ls,$ds,1)
    #return @benchmark @parallel (1:$nx,1:$ny,1:$nz) rhs!($U1,$U2,$U3,$ns,$ls,$ds,0.,1)

    # Preparation to save

    path = string("3D_data")
    rm(path*"/data"*string(num)*".h5",force=true)
    datafile = h5open(path*"/data"*string(num)*".h5","cw")

    # nsave = 1#Int64(nt/noutsave) + 1
    # gdata = create_dataset(datafile, "phi", datatype(Data.Number), 
    #         dataspace(nsave,6,save_size,save_size,save_size), 
    #         chunk=(1,6,save_size,save_size,save_size))

    # Preparation of visualisation
    ENV["GKSwstype"]="nul"; if isdir("viz3D_out")==false mkdir("viz3D_out") end
    loadpath = "./viz3D_out/"
    anim = Animation(loadpath,String[])
    old_files1 = readdir(loadpath; join=true)
    for i in 1:length(old_files1) rm(old_files1[i]) end
    println("Animation directory: $(anim.dir)")

    if isdir("pdfs")==false mkdir("pdfs") end
    old_files2 = readdir("./pdfs/"; join=true)
    for i in 1:length(old_files2) rm(old_files2[i]) end

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

    vis_steps = Int(round(nt/nout,RoundDown)+1)

    mass1data = create_dataset(datafile, "mass1", Data.Number, (vis_steps,))
    mass2data = create_dataset(datafile, "mass2", Data.Number, (vis_steps,))
    areadata  = create_dataset(datafile, "area", Data.Number, (vis_steps,))
    condata   = create_dataset(datafile, "constraints", Data.Number, (vis_steps,))
    energydata = create_dataset(datafile, "energy", Data.Number, (vis_steps,))

    H_vec1 = fill(-10.,vis_steps)
    H_vec2 = fill(-10.,vis_steps)
    A_vec = fill(-10.,vis_steps)
    C_vec = fill(10^(-10),vis_steps)
    E_vec = fill(zero(Data.Number),vis_steps)

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
                    volume = 0.
                    Cnorm = 0.
                    for ρi in 1:nρ, zi in 1:nz
                        Cnorm += energy_cell(U0[ρi,zi],ns,ls,ds,(ρi,zi))
                    end
                else
                    # volume = 0.
                    # Cint = 0.
                    # Energy = 0.
                    # for ρi in 1:nρ, zi in 1:nz
                    #     H_ = FourVector(H[ρi,zi].data)
                    #     Cint += constraint_energy_cell(U1[ρi,zi],H_,Rb,ns,ls,ds,(ρi,zi))
                    #     volume += volume_cell(U1[ρi,zi],Rb,ns,ls,ds,(ρi,zi))
                    #     Cnorm = sqrt(Cint/volume)
                    #     Energy += state_vector_energy_cell(U1[ρi,zi],Rb,ns,ls,ds,(ρi,zi))
                    # end

                    # A1 = SurfaceIntegral(Area,U1,ns,ds,ls,Rb)
                    # int1 = SurfaceIntegral(Hawking_Mass_cell,U1,ns,ds,ls,Rb)

                    # Hmass1 = sqrt(A1/(16*pi))*(1+int1/(16*pi))

                    # A2 = SurfaceIntegral(Area,U1,ns,ds,ls,0.9*lρ/2)
                    # int2 = SurfaceIntegral(Hawking_Mass_cell,U1,ns,ds,ls,0.9*lρ/2)

                    # Hmass2 = sqrt(A2/(16*pi))*(1+int2/(16*pi))

                    # if it == 1; 
                    #     global A_init = A1; 
                    #     global Hmass1_init = Hmass1; 
                    #     global Hmass2_init = Hmass2;
                    #     println(round(Hmass1, sigdigits=3)," ",round(Hmass2, sigdigits=3)," ",
                    #     round(A1/(4π*(Rb)^2)-1, sigdigits=3)," ",round(A2/(4π*(0.9*lρ/2)^2)-1, sigdigits=3)) 
                    # end

                end

                #append!(C_vec,result)
                # ti = Int(round(it/nout+1))

                # C_vec[ti] = Cnorm
                # H_vec1[ti] = (Hmass1 - Hmass1_init)/Hmass1_init
                # H_vec2[ti] = (Hmass2 - Hmass2_init)/Hmass2_init
                # A_vec[ti] = (A1 - A_init)/A_init
                # E_vec[ti] = Energy

                # mass1data[ti]  = (Hmass1 - Hmass1_init)/Hmass1_init
                # mass2data[ti]  = (Hmass2 - Hmass2_init)/Hmass2_init
                # areadata[ti]   = (A1 - A_init)/A_init
                # condata[ti]    = Cnorm
                # energydata[ti] = Energy

                #A = Array(getindex.(CellArrays.field(U1,1),1,1))[slice...] .+ 1

                # zi = Int(ceil(nz/2))

                #B = [(Dx(fg,U1,ns,find_boundary(ns,ls,ds,(X[xi],Y[yi],Z[zi]),(xi,yi,zi))[2:5]...,xi,yi,zi)/hx - U1[xi,yi,zi].dx)[1,1] for xi in 1:nx, yi in 1:ny]
                #B = [constraints(U1[xi,yi,zi],FourVector(H[xi,yi,zi].data),Rb,ns,ds,ls,(xi,yi,zi))[1] for xi in 1:nx, yi in 1:ny]
                #B = [(rootγ(U1[xi,yi,zi])[1])^2-1 for xi in 1:nx, yi in 1:ny]
                B = [(U=U1[ρi,zi]; if any(isnan.(U.g)); NaN else u(U)[2,2] end) for  ρi in 1:nρ, zi in 1:nz]

                time = round(t; digits=1)

                heatmap(Z, R, B, aspect_ratio=1,
                ylims=(0,lρ),
                xlims=(-lz/2,lz/2),
                clim=(-0.01,0.01), 
                xlabel=L"z/M_0",
                ylabel=L"\rho/M_0",
                title = L"\partial_t g_{\rho\rho}(t = %$time M_0)", 
                c=plotcolors,frame=:box,
                fontfamily = "Computer Modern",
                tickfontsize  = 14,
                guidefontsize = 14
                )

                ang = range(0, 2π, length = 60)
                circle(x,y,r) = Shape(r*sin.(ang).+x, r*cos.(ang).+y)  

                p1 = plot!(circle(0,0,Rb),fc=:transparent, legend = false, colorbar = true)

                # if mod(it,pdfout)==0
                #     savefig("pdfs/"*lpad(it,6,"0")*".pdf")
                # end

                p2 = plot(C_vec, yaxis=:log, ylim = (10^(-5), 10^(1)), minorgrid=true) #[C_vec H_vec]

                p3 = plot([H_vec1 A_vec H_vec2], ylim = (-0.05, 0.25),label=["Mass1" "Area" "Mass2"])

                plot(p1,p2,p3, layout=grid(3,1,heights=(3/5,1/5,1/5)),size=(800,800),legend=false)

                #plot(p1)

                frame(anim)

            end

            # Perform RK3 algorithm
            for i in 1:3
                @parallel (1:nρ,1:nz) test!(U1,U2,U3,Rb,ns,ds,ls,i) 
                @parallel (1:nρ,1:nz) rhs!(U1,U2,U3,H,∂H,Rb,ns,ls,ds,t,i) 
            end

            t = t + dt

            if SAVE && mod(it,noutsave)==0

                if USE_GPU
                    copy!(U0.data, U1.data)
                    gdata[iter,:,:,:,:] = [U0[ρi,zi].g.data[i] for i in 1:6, zi in 1:step:nz, ρi in 1:step:nρ]
                else
                    gdata[iter,:,:,:,:] = [U1[ρi,zi].g.data[i] for i in 1:6, zi in 1:step:nz, ρi in 1:step:nρ]
                end

                iter += 1

            end

            it += 1

        end

    catch error
        close(datafile)
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

    #save stuff

    # mass1data = H_vec1
    # mass2data = H_vec2
    # areadata  = A_vec
    # condata   = C_vec
    # energydata = E_vec

    close(datafile)

    # Performance metrics
    wtime    = Base.time() - wtime0
    bytes_accessed = sizeof(StateVector)*(2*3)   # Bytes accessed per iteration per gridpoint (1 read and write iter=1 and 2 reads 1 write for iter=2,3)
    A_eff    = bytes_accessed*nρ*nz/1e9       # Effective main memory access per iteration [GB] 
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