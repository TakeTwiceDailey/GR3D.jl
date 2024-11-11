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

using LaTeXStrings

ParallelStencil.@reset_parallel_stencil()

const USE_GPU = false  # Use GPU? If this is set false, then no GPU needs to be available

@static if USE_GPU
    # GPU isn't faster than CPU at the moment, so I gave up on it.
    # If you want to use it, you need to include RootSolvers.jl, as Roots.jl doesn't work on GPU
    # You also have to change the root solver functions in the boundary finder
    @define_CuCellArray() # without this CuCellArrays don't work
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
    #@init_parallel_stencil(Polyester, Float64, 3) # Polyester is supposed to be faster than Threads, needs testing
end

include("types.jl")               # Define StateVector types and various other types used during the evolution
include("initial_conditions.jl")  # Sets Initial conditions. Go here if you want to change mass/spin of black hole.
include("boundaries.jl")          # Deals with finding the boundaries and applying the boundary conditions 
include("differencing.jl")        # Defines finite differencing operators in the SBP embedded boundary framework
include("measures.jl")            # Defines quasi-local measures like Hawking mass and closed surface integration

@parallel_indices (xi,yi,zi) function rhs!(U1,U2,U3,H,∂H,Rb,ns::Tuple,ls::Tuple,ds::Tuple,t::Data.Number,iter::Int)
    # Performs the right hand side of the system of generalized harmonic evolution equations.
    # The if-else blocks at the beginning and end perform the 3rd order Runge-Kutta algorithm
    # which is done by calling rhs! three times, each with a different value in iter ranging from 1:3

    if iter == 1 # U is the current iteration read memory, and Uw is the current iteration write memory
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
    x = -lx/2 + (xi-1)*hx; y = -ly/2 + (yi-1)*hy; z = -lz/2 + (zi-1)*hz; # Cartesian position of current kernel calculation

    r = (x,y,z)
    ri = (xi,yi,zi) 

    in_domain,nls,nrs,αls,αrs = find_boundary(Rb,ns,ls,ds,r,ri) 
    # finds if any boundaries are nearby, and checks if we are in a memory point inside or outside domain

    if in_domain     # Don't do evolution if outside of domain

        Uxyz = U[xi,yi,zi]  

        H_ = FourVector(H[xi,yi,zi].data) # FourVector(fH_(x,y,z))

        ∂H_ = StateTensor(∂H[xi,yi,zi].data) #StateTensor((μ,ν) -> 0.5*(f∂H_(x,y,z,μ,ν)+f∂H_(x,y,z,ν,μ)))

        g  = Uxyz.g
        dx = Uxyz.dx
        dy = Uxyz.dy
        dz = Uxyz.dz
        P  = Uxyz.P

        gi = inv(g)  # Invert metric

        α = 1/sqrt(-gi[1,1]) # Calculate lapse
    
        βx = -gi[1,2]/gi[1,1] # Calculate shift vector
        βy = -gi[1,3]/gi[1,1]
        βz = -gi[1,4]/gi[1,1]

        n  = FourVector((1,-βx,-βy,-βz))/α  # normal to spatial slices
        n_ = FourVector((-α,0.,0.,0.))

        γi = gi + symmetric(@einsum n[μ]*n[ν])  # inverse 3-metric

        ∂tg = βx*dx + βy*dy + βz*dz - α*P  # time derivative of metric

        # collect metric derivatives
        ∂g = Symmetric3rdOrderTensor((σ,μ,ν) -> (∂tg[μ,ν],dx[μ,ν],dy[μ,ν],dz[μ,ν])[σ])

        # Calculate Christoffel symbols
        Γ  = Symmetric3rdOrderTensor((σ,μ,ν) -> 0.5*(∂g[ν,μ,σ] + ∂g[μ,ν,σ] - ∂g[σ,μ,ν]))

        #  misc
        ∂tlnrootγ = @einsum 0.5*γi[μ,ν]*∂tg[μ,ν]

        Γ_ = @einsum gi[ϵ,σ]*Γ[μ,ϵ,σ]

        #########################################################
        # Principle (linear) part of the evolution equations

        ∂tdx = Dx(u,U,ns,nls,nrs,αls,αrs,xi,yi,zi)/hx

        ∂tdy = Dy(u,U,ns,nls,nrs,αls,αrs,xi,yi,zi)/hy

        ∂tdz = Dz(u,U,ns,nls,nrs,αls,αrs,xi,yi,zi)/hz

        ∂tP  = Div(vx,vy,vz,U,ns,nls,nrs,αls,αrs,ds,xi,yi,zi)

        #########################################################
        # Non-linear terms in the evolution equations (34% of runtime)

        ∂tP -= ∂tlnrootγ*P

        ∂tP += 2*α*∂H_
    
        ∂tP -= 2*α*symmetric(@einsum (μ,ν) -> gi[ϵ,σ]*H_[ϵ]*Γ[σ,μ,ν])

        ∂tP -=   α*symmetric(@einsum (μ,ν) -> gi[λ,γ]*Γ_[λ]*∂g[γ,μ,ν])

        ∂tP += 2*α*symmetric(@einsum (μ,ν) -> gi[λ,ρ]*gi[ϵ,σ]*∂g[λ,ϵ,μ]*∂g[ρ,σ,ν])
    
        ∂tP -= 2*α*symmetric(@einsum (μ,ν) -> gi[ϵ,σ]*gi[λ,ρ]*Γ[μ,ϵ,λ]*Γ[ν,σ,ρ])

        #########################################################
        # Constraints and constraint damping terms (16% of runtime)

        γ0 = 1. # Harmonic constraint damping (>0)
        #γ1 = -1. # Linear Degeneracy parameter (=-1)
        γ2 = 1. # Derivative constraint damping (>0)

        Cx = Dx(fg,U,ns,nls,nrs,αls,αrs,xi,yi,zi)/hx - dx
        Cy = Dy(fg,U,ns,nls,nrs,αls,αrs,xi,yi,zi)/hy - dy
        Cz = Dz(fg,U,ns,nls,nrs,αls,αrs,xi,yi,zi)/hz - dz

        ∂tdx += γ2*α*Cx
        ∂tdy += γ2*α*Cy
        ∂tdz += γ2*α*Cz

        # Generalized Harmonic constraints and damping

        C_ = Γ_ - H_

        ∂tP += γ0*symmetric(@einsum (μ,ν) -> 2C_[μ]*n_[ν] - g[μ,ν]*n[ϵ]*C_[ϵ]) #*α + C_[ν]*n_[μ]

        ∂tP += -γ2*(βx*Cx + βy*Cy + βz*Cz) # γ1 = -1 just assumed here

        ########################################################

        ∂tU  = StateVector(∂tg,∂tdx,∂tdy,∂tdz,∂tP)

        # Perform boundary conditions (12% of runtime)
        ∂tU += SAT(U,H,ns,ls,ds,nls,nrs,αls,αrs,r,t,ri,γ2)

        # Add numerical dissipation (20% of runtime)
        ∂tU += -0.1*Dissipation(U,ns,ds,nls,nrs,αls,αrs,xi,yi,zi)

        # Perform Runge-Kutta 3rd order current iteration step
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


        Uw[xi,yi,zi] = NaNSV  # Write NaN to these points so that we are sure no information leaves closed boundaries
        #Uw[xi,yi,zi] = MinSV
        

    end

    return

end

@parallel_indices (xi,yi,zi) function test!(U1,U2,U3,Rb,ns,ds,ls,iter::Int)
    # Allows for testing at each step. Helpful if you can't figure out where an instability is starting

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

    in_domain,_,_,_,_ = find_boundary(Rb,ns,ls,ds,r,ri)

    if in_domain

        g = U[xi,yi,zi].g

        if any(isnan.(g)); println(ri) end

        _,_,_,_,γs... = g.data

        gi = inv(g)

        if det(ThreeTensor(γs))<0
            println("γ ",(xi,yi,zi)) # The 3-metric determinant is not positive at this point
            throw(0.)
        elseif -gi[1,1]<0
            println("α ",(xi,yi,zi)) # The lapse is undefined at this point
            throw(0.)
        end
    end

    return

end

@parallel_indices (xi,yi,zi) function initialize!(Uw,H,∂H,Rb,ns,ds,ls,constraint_init)
    # Initializes the initial conditions to arrays
    # the metric must be written to the array first if one desires constraint satisfying initialization
    
    hx,hy,hz,dt = ds
    lx,ly,lz=ls
    x = -lx/2 + (xi-1)*hx; y = -ly/2 + (yi-1)*hy; z = -lz/2 + (zi-1)*hz;

    r = (x,y,z)
    ri = (xi,yi,zi)

    in_domain,nls,nrs,αls,αrs = find_boundary(Rb,ns,ls,ds,r,ri)

    if in_domain

        g = Uw[xi,yi,zi].g

        ∂tg = ∂tg_init(x,y,z)

        if constraint_init
            dx = Dx(fg,Uw,ns,nls,nrs,αls,αrs,xi,yi,zi)/hx
            dy = Dy(fg,Uw,ns,nls,nrs,αls,αrs,xi,yi,zi)/hy
            dz = Dz(fg,Uw,ns,nls,nrs,αls,αrs,xi,yi,zi)/hz
        else
            dx  = ∂xg(x,y,z)      
            dy  = ∂yg(x,y,z)      
            dz  = ∂zg(x,y,z)
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

        H[xi,yi,zi]  = fH_(x,y,z)

        ∂H[xi,yi,zi] = f∂H_(x,y,z)

        if constraint_init

            ∂g = Symmetric3rdOrderTensor((σ,μ,ν) -> (∂tg[μ,ν], dx[μ,ν], dy[μ,ν], dz[μ,ν])[σ])
            
            G = minorsymmetric(NSProjection((μ,ν,α,β)-> γm[μ,α]*γm[ν,β] - n_[μ]*n_[ν]*γi[α,β]))

            C = Symmetric3rdOrderTensor((α,μ,ν) -> n_[μ]*γm[ν,α]/2 + n_[ν]*γm[μ,α]/2 - n_[μ]*n_[ν]*n[α])

            #Cf_ = FourVector((0,0,0,0))

            μ0 = 12
            σ0 = 2
            Amp = 0.1

            r=sqrt(x^2+y^2+z^2)

            model = (μ0-σ0)<r<(μ0+σ0) ? (Amp/σ0^8)*(r-(μ0-σ0))^4*(r-(μ0+σ0))^4 : 0.

            Cf_ = 0*model*n_   # set desired form of initial gauge constraint violation

            H_ = FourVector(fH_(x,y,z).data) 

            A_ = @einsum ( 2*γi[i,μ]*∂g[i,μ,α] - gi[μ,ν]*γm[α,i]*∂g[i,μ,ν] - 2*H_[α] - 2*Cf_[α])
            # index down

            P = symmetric(@einsum C[α,μ,ν]*A_[α] + G[μ,ν,α,β]*P2[α,β])
        else
            P = P2
        end

        Uw[xi,yi,zi] = StateVector(g,dx,dy,dz,P)

    else

        # Write NaN here to be sure these points never reach into the computational domain

        Uw[xi,yi,zi] = NaNSV
        H[xi,yi,zi]  = SVector{4}(NaN,NaN,NaN,NaN)
        ∂H[xi,yi,zi] = SVector{10}(NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN)

    end

    
    return

end

##################################################
@views function main()

    # Spacetime Domain size
    lx, ly, lz = 20.0, 20.0, 20.0  # spatial domain extends
    ls = (lx,ly,lz)
    t  = 0.0                       # physical start time
    tmax = 100.0                   # physical end time

    # Numerics
    num = 81 # 81,108,144,192,256
    ns = (num,num,num)   # numerical grid resolution; should be a mulitple of 32-1 for optimal GPU perf
    nx, ny, nz = ns

    Rb = 2.0 # inner boundary radius (check this is outside or inside the horizon as desired)

    SAVE = false
    VISUAL = true
    constraint_init = true
    nout       = 100 #Int(10*scale)         # plotting frequency
    noutsave   = 500 #Int(50*scale)         # not implemented
    pdfout     = 4*nout #Int(10*scale)      # pdf frequency

    save_size = 1
    step = Int(num/save_size)

    # Derived numerics
    hx, hy, hz = lx/(nx-1), ly/(ny-1), lz/(nz-1) # cell sizes
    CFL = 1/5                                    # conservative CFL factor
    dt = min(hx,hy,hz)*CFL      # time step size
    ds = (hx,hy,hz,dt)          # spacetime size of discrete cells

    #return 0.05/hx

    nt = Int(round(tmax/dt, RoundUp))           # total number of timesteps that will be taken

    coords = (-lx/2:hx:lx/2, -ly/2:hy:ly/2, -lz/2:hz:lz/2)
    X,Y,Z = coords

    # Array allocations
    # If we use the GPU, we need an intermediate array on the CPU to save
    if USE_GPU
        U0 = CPUCellArray{StateVector}(undef, nx, ny, nz)
        U1 =  CuCellArray{StateVector}(undef, nx, ny, nz)
        U2 =  CuCellArray{StateVector}(undef, nx, ny, nz)
        U3 =  CuCellArray{StateVector}(undef, nx, ny, nz)

        # H  =  CuCellArray{FourVector}(undef, nx, ny, nz)
        # ∂H =  CuCellArray{StateTensor}(undef, nx, ny, nz)
    else
        U0 =  CPUCellArray{StateVector}(undef, nx, ny, nz)
        U1 =  CPUCellArray{StateVector}(undef, nx, ny, nz)
        U2 =  CPUCellArray{StateVector}(undef, nx, ny, nz)
        U3 =  CPUCellArray{StateVector}(undef, nx, ny, nz)

        H  =  CPUCellArray{SVector{4,Data.Number}}(undef, nx, ny, nz)
        ∂H =  CPUCellArray{SVector{10,Data.Number}}(undef, nx, ny, nz)
    end

    if USE_GPU; Uw = U0 else Uw = U1 end # which Array do we initially write to?

    # Sample the Kerr-Schild form of the metric only
    for xi in 1:ns[1], yi in 1:ns[2], zi in 1:ns[3]

        hx,hy,hz,dt = ds
        lx,ly,lz=ls
        x = -lx/2 + (xi-1)*hx; y = -ly/2 + (yi-1)*hy; z = -lz/2 + (zi-1)*hz;

        r = (x,y,z)
        ri = (xi,yi,zi)
    
        in_domain,_,_,_,_ = find_boundary(Rb,ns,ls,ds,r,ri)

        if in_domain 
            g = StateTensor(g_init(x,y,z))
        else
            g = ZeroST   # Sample a zero metric (undefined determinant) so we are sure these potitions never reach into domain
        end

        Uw[xi,yi,zi] = StateVector(g,ZeroST,ZeroST,ZeroST,ZeroST)

    end

    # write the rest of the state vector
    @parallel (1:nx,1:ny,1:nz) initialize!(Uw,H,∂H,Rb,ns,ds,ls,constraint_init)

    plotcolors = cgrad([RGB(0.148,0.33,0.54),RGB(0.509,0.4825,0.52),RGB(0.9,0.62,0.29),RGB(0.9625,0.77625,0.484),RGB(1,0.95,0.75)], [0.2,0.4,0.6,0.8])

    # copy to other arrays
    if USE_GPU; copy!(U1.data, U0.data) end

    copy!(U2.data, U1.data)
    copy!(U3.data, U1.data)

    copy!(U0.data, U1.data)

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

    vis_steps = Int(round(nt/nout,RoundDown)+1)

    # Allocate storage for saving quasi-local measures 
    mass1data = create_dataset(datafile, "mass1", Data.Number, (vis_steps,))
    mass2data = create_dataset(datafile, "mass2", Data.Number, (vis_steps,))
    areadata  = create_dataset(datafile, "area", Data.Number, (vis_steps,))
    condata   = create_dataset(datafile, "constraints", Data.Number, (vis_steps,))
    energydata = create_dataset(datafile, "energy", Data.Number, (vis_steps,))
    
    # Allocate system memory for saving quasi-local measures 
    H_vec1 = fill(-10.,vis_steps)
    H_vec2 = fill(-10.,vis_steps)
    A_vec = fill(-10.,vis_steps)
    C_vec = fill(10^(-10),vis_steps)
    E_vec = fill(zero(Data.Number),vis_steps)

    iter = 1
    it = 1

    try # try block to catch errors, close datafile if so

        while it <= nt         # main time loop

            if (it==11)  global wtime0 = Base.time()  end # performance timing 

            # Visualisation
            if VISUAL && (it==1 || mod(it,nout)==0)

                # Calculate quasi0local measures
                if USE_GPU
                    copy!(U0.data, U1.data)
                    volume = 0.
                    Cnorm = 0.
                    for xi in 1:nx, yi in 1:ny, zi in 1:nz
                        Cnorm += energy_cell(U0[xi,yi,zi],ns,ls,ds,(xi,yi,zi))
                    end
                else
                    volume = 0.
                    Cint = 0.
                    Energy = 0.
                    # Calculate 3-volume based statevector numerical energy
                    for xi in 1:nx, yi in 1:ny, zi in 1:nz
                        H_ = FourVector(H[xi,yi,zi].data)
                        Cint += constraint_energy_cell(U1[xi,yi,zi],H_,Rb,ns,ls,ds,(xi,yi,zi))
                        volume += volume_cell(U1[xi,yi,zi],Rb,ns,ls,ds,(xi,yi,zi))
                        Cnorm = sqrt(Cint/volume)
                        Energy += state_vector_energy_cell(U1[xi,yi,zi],Rb,ns,ls,ds,(xi,yi,zi))
                    end

                    A1 = SurfaceIntegral(Area,U1,ns,ds,ls,Rb) # Calculate surface area of inner boundary
                    int1 = SurfaceIntegral(Hawking_Mass_cell,U1,ns,ds,ls,Rb) # Calculate Hawking mass of inner boundary

                    Hmass1 = sqrt(A1/(16*pi))*(1+int1/(16*pi))

                    rmax = 0.9*ly/2

                    A2 = SurfaceIntegral(Area,U1,ns,ds,ls,rmax) # Calculate surface area of largest sphere
                    int2 = SurfaceIntegral(Hawking_Mass_cell,U1,ns,ds,ls,rmax) # Calculate Hawking mass of largest sphere

                    Hmass2 = sqrt(A2/(16*pi))*(1+int2/(16*pi))

                    if it == 1; 
                        global A_init = A1; 
                        global Hmass1_init = Hmass1; 
                        global Hmass2_init = Hmass2;
                        println(round(Hmass1, sigdigits=3)," ",round(Hmass2, sigdigits=3)," ",
                        round(A1/(4π*(Rb)^2)-1, sigdigits=3)," ",round(A2/(4π*(rmax)^2)-1, sigdigits=3)) 
                    end

                end

                ti = Int(round(it/nout+1))

                # save various quasi-local measures or their fractional change
                C_vec[ti] = Cnorm
                H_vec1[ti] = (Hmass1 - Hmass1_init)/Hmass1_init
                H_vec2[ti] = (Hmass2 - Hmass2_init)/Hmass2_init
                A_vec[ti] = (A1 - A_init)/A_init
                E_vec[ti] = Energy

                mass1data[ti]  = (Hmass1 - Hmass1_init)/Hmass1_init
                mass2data[ti]  = (Hmass2 - Hmass2_init)/Hmass2_init
                areadata[ti]   = (A1 - A_init)/A_init
                condata[ti]    = Cnorm
                energydata[ti] = Energy

                #A = Array(getindex.(CellArrays.field(U1,1),1,1))[slice...] .+ 1

                zi = Int(ceil(nz/2))

                #B = [(Dx(fg,U1,ns,find_boundary(ns,ls,ds,(X[xi],Y[yi],Z[zi]),(xi,yi,zi))[2:5]...,xi,yi,zi)/hx - U1[xi,yi,zi].dx)[1,1] for xi in 1:nx, yi in 1:ny]
                #B = [constraints(U1[xi,yi,zi],FourVector(H[xi,yi,zi].data),Rb,ns,ds,ls,(xi,yi,zi))[1] for xi in 1:nx, yi in 1:ny]
                #B = [(rootγ(U1[xi,yi,zi])[1])^2-1 for xi in 1:nx, yi in 1:ny]
                B = [(U=U1[xi,yi,zi]; if any(isnan.(U.g)); NaN else u(U)[2,2] end) for yi in 1:ny, xi in 1:nx]

                time = round(t; digits=1)

                # Create equitorial slice heatmap 
                heatmap(X, Y, B, aspect_ratio=1,
                xlims=(-lx/2,lx/2),
                ylims=(-ly/2,ly/2),
                clim=(-0.01,0.01), 
                xlabel=L"x/M_0",
                ylabel=L"y/M_0",
                title = L"\partial_t g_{xx}(t = %$time M_0\,, z = 0)", 
                c=plotcolors,frame=:box,
                fontfamily = "Computer Modern",
                tickfontsize  = 14,
                guidefontsize = 14
                )

                ang = range(0, 2π, length = 60)
                circle(x,y,r) = Shape(r*sin.(ang).+x, r*cos.(ang).+y)  

                p1 = plot!(circle(0,0,Rb),fc=:transparent, legend = false, colorbar = true)

                if mod(it,pdfout)==0
                    savefig("pdfs/"*lpad(it,6,"0")*".pdf")
                end

                p2 = plot(C_vec, yaxis=:log, ylim = (10^(-5), 10^(1)), minorgrid=true) #[C_vec H_vec]

                p3 = plot([H_vec1 A_vec H_vec2], ylim = (-0.05, 0.25),label=["Mass1" "Area" "Mass2"])

                plot(p1,p2,p3, layout=grid(3,1,heights=(3/5,1/5,1/5)),size=(700,800),legend=false)

                #plot(p1)

                frame(anim)

            end

            # Perform RK3 algorithm
            for i in 1:3
                #@parallel (1:nx,1:ny,1:nz) test!(U1,U2,U3,Rb,ns,ds,ls,i)  # uncomment for testing
                @parallel (1:nx,1:ny,1:nz) rhs!(U1,U2,U3,H,∂H,Rb,ns,ls,ds,t,i) 
            end

            t = t + dt # advance time

            if SAVE && mod(it,noutsave)==0  # saving of result if desired

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

    catch error    # If an error occured, show what line it happened on, close datafiles, clear memory, etc. 
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

    close(datafile)

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