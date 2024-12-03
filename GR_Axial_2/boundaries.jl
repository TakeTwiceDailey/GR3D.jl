


@inline function vectors(U,outer,ns,ri,rb)
    # Returns the null basis to the boundary
    # The embedded boundary method effectively treats the boundary as "Lego" objects
    # with boundary normals only in one of the 3 cardinal directions.
    # Retruns upper index vectors and a mixed index boundary 2-metric

    ρi,zi= ri

    g = U.g

    sρ,sz = (0.,0.)

    if outer

        if ρi==1; (sρ = -1.) elseif ρi==ns[1]; (sρ = 1.) end
        if zi==1; (sz = -1.) elseif zi==ns[2]; (sz = 1.) end

    else
        ρ,z = rb
        sρ = -ρ
        sz = -z
    end

    gi = inv(g)

    α = 1/sqrt(-gi[1,1])

    βρ = -gi[1,2]/gi[1,1]
    βz = -gi[1,3]/gi[1,1]

    n = FourVector((1,βρ,βz,0))/α

    st_ = FourVector((βρ*sρ+βz*sz,sρ,sz,0))

    snorm = @einsum gi[μ,ν]*st_[μ]*st_[ν]
    
    s_ = st_/sqrt(snorm) 

    s = @einsum gi[μ,ν]*s_[ν]

    ℓ = FourVector((n + s)/sqrt(2))
    k = FourVector((n - s)/sqrt(2))

    σ = StateTensor((μ,ν) -> gi[μ,ν] + k[μ]*ℓ[ν] + ℓ[μ]*k[ν]) # all indices up

    return (k,ℓ,σ) # spherical

end

@inline function find_boundary(Rb,ns,ls,ds,r,ri)
    # Returns a tuple of various objects needed to know where the boundary is.
    # The nl and nr are the left and right farthest grid cell indices in all 3 directions
    # The αl and αr are the left and right distances from that last grid cell to the boundary 
    # in units of the grid spacing in all 3 directions
    #
    # To prevent unnecessary computations, we only perform the root solver if the boundary is within
    # five grid cells of the current point, as if this is not the case, then the boundary position 
    # does not matter to the stencils or boundary conditions.

    ρi,zi = ri

    lρ,lz=ls
    dρ,dz,_=ds

    nρ,nz = ns

    ρ,z = r

    p = 2

    f(ρ,z) = (ρ^p + z^p) - Rb^p # Definitions of the boundary position

    #f(x,y,z) = x + 200

    nlρ =  1; nlz =  1;
    nrρ = nρ; nrz = nz;

    # αlx = 0.5; αly = 0.5; αlz = 0.5;
    # αrx = 0.5; αry = 0.5; αrz = 0.5;

    αlρ = 0.; αlz = 0.;
    αrρ = 0.; αrz = 0.;

    s = sign(f(ρ,z))

    #a = 10^(-10)

    # # x-line
    # i = xi+3
    # x0 = x+3*dx
    # if (i <= nx && sign(f(x0,y,z))≠s)
    #     rb = find_zero(x->f(x,y,z), (x,x0), atol=a*dx, rtol=a*dx, Bisection())+lx/2
    #     #rb = find_zero(x->f(x,y,z), SecantMethod{Float64}(x,x0), CompactSolution()).root+lx/2
    #     temp,αrx = divrem(rb,dx,RoundDown)
    #     αrx /= dx; xibr=Int(temp)+1;
    # end

    # i = xi-3
    # x0 = x-3*dx
    # if (i >= 1 && sign(f(x0,y,z))≠s)
    #     rb = find_zero(x->f(x,y,z), (x0,x), atol=a*dx, rtol=a*dx, Bisection())+lx/2
    #     #rb = find_zero(x->f(x,y,z), SecantMethod{Float64}(x0,x), CompactSolution()).root+lx/2
    #     temp,αlx = divrem(rb,dx,RoundUp)
    #     αlx /= -dx; xibl=Int(temp)+1;
    # end

    # # y-line
    # i = yi+3
    # y0 = y+3*dy
    # if (i <= ny && sign(f(x,y0,z))≠s)
    #     rb = find_zero(y->f(x,y,z), (y,y0), atol=a*dy, rtol=a*dy, Bisection())+ly/2
    #     #rb = find_zero(y->f(x,y,z), SecantMethod{Float64}(y,y0), CompactSolution()).root+ly/2
    #     temp,αry = divrem(rb,dy,RoundDown)
    #     αry /= dy; yibr=Int(temp)+1;
    # end

    # i = yi-3
    # y0 = y-3*dy
    # if (i >= 1 && sign(f(x,y0,z))≠s)
    #     rb = find_zero(y->f(x,y,z), (y0,y), atol=a*dy, rtol=a*dy, Bisection())+ly/2
    #     #rb = find_zero(y->f(x,y,z), SecantMethod{Float64}(y0,y), CompactSolution()).root+ly/2
    #     temp,αly = divrem(rb,dy,RoundUp)
    #     αly /= -dy; yibl=Int(temp)+1;
    # end

    # # z-line
    # i = zi+3
    # z0 = z+3*dz
    # if (i <= nz && sign(f(x,y,z0))≠s)
    #     rb = find_zero(z->f(x,y,z), (z,z0), atol=a*dz, rtol=a*dz, Bisection())+lz/2
    #     #rb = find_zero(z->f(x,y,z), SecantMethod{Float64}(z,z0), CompactSolution()).root+lz/2
    #     temp,αrz = divrem(rb,dz,RoundDown)
    #     αrz /= dz; zibr=Int(temp)+1;
    # end

    # i = zi-3
    # z0 = z-3*dz
    # if (i >= 1 && sign(f(x,y,z0))≠s)
    #     rb = find_zero(z->f(x,y,z), (z0,z), atol=a*dz, rtol=a*dz, Bisection())+lz/2
    #     #rb = find_zero(z->f(x,y,z), SecantMethod{Float64}(z0,z), CompactSolution()).root+lz/2
    #     temp,αlz = divrem(rb,dz,RoundUp)
    #     αlz /= -dz
    #     zibl=Int(temp)+1
    # end

    # # x-line
    # i = xi+3
    # x0 = x+3*dx
    # if (i <= nx && sign(f(x0,y,z))≠s)
    #     rb = -sqrt(Rb^2-y^2-z^2)+lx/2
    #     temp,αrx = divrem(rb,dx,RoundDown)
    #     αrx /= dx; xibr=Int(temp)+1;
    # end

    # i = xi-3
    # x0 = x-3*dx
    # if (i >= 1 && sign(f(x0,y,z))≠s)
    #     rb = sqrt(Rb^2-y^2-z^2)+lx/2
    #     temp,αlx = divrem(rb,dx,RoundUp)
    #     αlx /= -dx; xibl=Int(temp)+1;
    # end

    # # y-line
    # i = yi+3
    # y0 = y+3*dy
    # if (i <= ny && sign(f(x,y0,z))≠s)
    #     rb = -sqrt(Rb^2-x^2-z^2)+lx/2
    #     temp,αry = divrem(rb,dy,RoundDown)
    #     αry /= dy; yibr=Int(temp)+1;
    # end

    # i = yi-3
    # y0 = y-3*dy
    # if (i >= 1 && sign(f(x,y0,z))≠s)
    #     rb = sqrt(Rb^2-x^2-z^2)+lx/2
    #     temp,αly = divrem(rb,dy,RoundUp)
    #     αly /= -dy; yibl=Int(temp)+1;
    # end

    # # z-line
    # i = zi+3
    # z0 = z+3*dz
    # if (i <= nz && sign(f(x,y,z0))≠s)
    #     rb = -sqrt(Rb^2-x^2-y^2)+lx/2
    #     temp,αrz = divrem(rb,dz,RoundDown)
    #     αrz /= dz; zibr=Int(temp)+1;
    # end

    # i = zi-3
    # z0 = z-3*dz
    # if (i >= 1 && sign(f(x,y,z0))≠s)
    #     rb = sqrt(Rb^2-x^2-y^2)+lx/2
    #     temp,αlz = divrem(rb,dz,RoundUp)
    #     αlz /= -dz
    #     zibl=Int(temp)+1
    # end

    if Rb^2 > z^2

        ρl = sqrt(Rb^2-z^2)
        nltρ = (ρl - dρ/2)/dρ + 1
        nlρ = Int(round(nltρ, RoundUp))
        αlρ = abs(nlρ-nltρ)

    end

    if  Rb^2 > ρ^2
        if  z<=0
            zr = -sqrt(Rb^2-ρ^2)
            nrtz = (zr + lz/2)/dz + 1
            nrz = Int(round(nrtz, RoundDown))
            αrz = abs(nrz-nrtz)
        end

        if z>=0
            zl = sqrt(Rb^2-ρ^2)
            nltz = (zl + lz/2)/dz + 1
            nlz = Int(round(nltz, RoundUp))
            αlz = abs(nlz-nltz)
        end
    end

    nl = (nlρ,nlz)
    nr = (nrρ,nrz)
    αl = (αlρ,αlz) 
    αr = (αrρ,αrz)

    in_domain = !(s==-1)
    #in_domain = all(nl.<=ri.<=nr)&&!(s==-1)

    return (in_domain,nl,nr,αl,αr)

end

@inline function BoundaryConditions(Ub,H_,outer,rb,C_BC,fGW1,fGW2,k,ℓ,σ)

    ρb,zb = rb

    g  = Ub.g
    dρ = Ub.dρ
    dz = Ub.dz
    P  = Ub.P

    #H_ = FourVector((0.,0.,0.,0.))
    #H_ = FourVector(μ -> fH_(xb,yb,zb,μ)) #FourVector(fH_(xb,yb,zb))

    ℓ_ = @einsum g[μ,α]*ℓ[α]
    k_ = @einsum g[μ,α]*k[α]

    # ℓn_ = @einsum g[μ,α]*ℓn[α]
    # kn_ = @einsum g[μ,α]*kn[α]

    σm = @einsum g[μ,α]*σ[α,ν]  # mixed indices down up
    σ_ = @einsum g[μ,α]*σm[ν,α] # all indices down

    gi = inv(g)

    α = 1/sqrt(-gi[1,1])

    βρ = -gi[1,2]/gi[1,1]
    βz = -gi[1,3]/gi[1,1]

    ∂tg = βρ*dρ + βz*dz - α*P

    ∂gb = Symmetric3rdOrderTensor((σ,μ,ν) -> (σ==1 ? ∂tg[μ,ν] : σ==2 ? dρ[μ,ν] : σ==3 ? dz[μ,ν] : σ==4 ? 0 : throw(ZeroST)))

    Upb = (@einsum  k[α]*∂gb[α,μ,ν])# + γ2*g/sqrt(2)
    U0b = @einsum  σm[γ,β]*∂gb[β,μ,ν]
    Um = (@einsum ℓ[α]*∂gb[α,μ,ν])# + γ2*g/sqrt(2)

    s_ = (ℓ_ - k_)/sqrt(2)

    β = FourVector((0,βρ,βz,0))

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
        UmBC1  = fUmBC(ℓ,ρb,zb)
        UmBCGW = (1+fGW1)*fUmBC(ℓ,ρb,zb) #*

        # UmBC1  = Neumann(0.)
        # UmBCGW = Neumann(0.)
    else
        #UmBC1  = Dirichlet(0.)
        #UmBCGW = Dirichlet(0.)

        # UmBC1  = Neumann(0.)
        # UmBCGW = Neumann(0.)

        UmBC1  = fUmBC(ℓ,ρb,zb)
        UmBCGW = (1+fGW2)*fUmBC(ℓ,ρb,zb)
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

    return (StateVector(ZeroST,∂gBC[2,:,:],∂gBC[3,:,:],PBC),cm)

end


Base.@propagate_inbounds @inline function SAT(U,H,ns,ls,ds,nls,nrs,αls,αrs,r,t,ri,γ2)
    # Performs the boundary conditions in each direction
    # Since the StateVector on the boundary is not in the domain, its value must be extrapolated.
    # Since the boundary position is not in the domain, the application of boundary conditions
    # must also be extrapolated.

    lρ,lz = ls
    hρ,hz,_=ds

    ρi,zi=ri
    ρ,z = r

    # x = -lx/2 + (xi-1)*hx; y = -ly/2 + (yi-1)*hy; z = -lz/2 + (zi-1)*hz;

    SATg = ZeroST; SATdρ = ZeroST; SATdz = ZeroST; SATP = ZeroST; 

    for i in 1:2

        nl = nls[i]; nr = nrs[i];

        # if nr-nl<3 
        #     #(println(nls,nrs); @assert false) 
        #     throw(ZeroSV)
        # end

        if (ri[i]==nl==1 && i==2) || (nl ≠ 1 && ri[i] in nl:nl+1) #ri[i] in nl:nl+1 # in the boundary region on the left side of the line

            let 
            # I want to use the same variable names here in the left branch 
            # as in the right branch, so there is a let block to allow this
            # and prevent type instability

                αl = αls[i]

                #xb,yb,zb = x,y,z

                # Interpolate the solution vector on the boundary 
                # and determine the boundary position on the coordinate line
                if i == 1 # On an x-line
                    ρb = hρ/2+(nl-1)*hρ-αl*hρ
                    zb = z
                    Ub = el2_1(αl)*U[nl,zi] + el2_2(αl)*U[nl+1,zi]
                    Hb = FourVector((el2_1(αl)*H[nl,zi] + el2_2(αl)*H[nl+1,zi]).data)
                elseif i == 2 # On a z-line
                    zb = -lz/2+(nl-1)*hz-αl*hz
                    ρb = ρ
                    Ub = el2_1(αl)*U[ρi,nl] + el2_2(αl)*U[ρi,nl+1]
                    Hb = FourVector((el2_1(αl)*H[ρi,nl] + el2_2(αl)*H[ρi,nl+1]).data)
                end

                dρ = Ub.dρ
                dz = Ub.dz
                P  = Ub.P

                rb = (ρb,zb)

                outer =  (nl == 1)

                Amp = 1.

                σz = 5.
                σρ = 0.9*lρ
                r0 = -lz/2-σz

                # σx = 1.
                # σy = 0.9*2.5
                # σz = 0.9*2.5
                # r0 = 0.

                rp = sqrt(((zb-(r0+t))/σz)^2+(ρb/σρ)^2)

                model =  (outer&&(rp<1)) ? Amp*(rp-1)^4*(rp+1)^4 : 0. #outer&&

                #C_BC = model*FourVector((0.,0.,0.,0.))
                #C_BC = model*FourVector((1.,0.,0.,0.))

                #UmBCGW = zero(StateTensor)
                fGW1 = 130*model
                #fGW = 1.
                
                # μ0 = 0.
                # σ0 = 2.5
                # Amp2 = 2000.

                # rc=sqrt(xb^2+yb^2+zb^2)

                # model2 = (μ0-σ0)<rc<(μ0+σ0) ? (Amp2/σ0^8)*(rc-(μ0+t-σ0))^4*(rc-(μ0+t+σ0))^4 : 0.

                fGW2 = 0 #model2*(3*zb^2/rc^2-1)

                k,ℓ,σ = vectors(Ub,outer,ns,ri,rb) # Form boundary basis

                C_BC = 0*model*FourVector((1.,0.,0.,0.))

                UBC,cm = BoundaryConditions(Ub,Hb,outer,rb,C_BC,fGW1,fGW2,k,ℓ,σ)

                if cm < 0

                    dρBC = UBC.dρ
                    dzBC = UBC.dz
                    PBC  = UBC.P

                    if ri[i] == nl 
                        ε = abs(cm)*el2_1(αl)/h2_11(αl)/ds[i]
                    elseif ri[i] == nl+1
                        ε = abs(cm)*el2_2(αl)/h2_22(αl)/ds[i]
                    end

                    SATP  += ε*(PBC - P)
                    SATdρ += ε*(dρBC - dρ)
                    SATdz += ε*(dzBC - dz)

                end

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
                    rb = (hρ/2+(nr-1)*hρ+αr*hρ,z)
                    # rb = (x,y,z)
                    ρb,zb = rb
                    Ub = el2_1(αr)*U[nr,zi] + el2_2(αr)*U[nr-1,zi]
                    Hb = FourVector((el2_1(αr)*H[nr,zi] + el2_2(αr)*H[nr-1,zi]).data)
                elseif i == 2 # On a z-line
                    rb = (ρ,-lz/2+(nr-1)*hz+αr*hz)
                    ρb,zb = rb
                    # rb = (x,y,z)
                    Ub = el2_1(αr)*U[ρi,nr] + el2_2(αr)*U[ρi,nr-1]
                    Hb = FourVector((el2_1(αr)*H[ρi,nr] + el2_2(αr)*H[ρi,nr-1]).data)
                end

                dρ = Ub.dρ
                dz = Ub.dz
                P  = Ub.P
        
                outer =  (nr == ns[i])

                Amp = 1.

                σz = 5.
                σρ = 0.9*lρ
                r0 = lz/2+σz

                # σx = 1.
                # σy = 0.9*2.5
                # σz = 0.9*2.5
                # r0 = 0.

                rp = sqrt(((zb-(r0-t))/σz)^2+(zb/σz)^2)

                model =  (outer&&(rp<1)) ? Amp*(rp-1)^4*(rp+1)^4 : 0. #outer&&

                fGW1 =  130*model
                #fGW = 1.

                
                # μ0 = 0.
                # σ0 = 2.5
                # Amp2 = 2000.

                # rc=sqrt(xb^2+yb^2+zb^2)

                # model2 = (μ0-σ0)<rc<(μ0+σ0) ? (Amp2/σ0^8)*(rc-(μ0+t-σ0))^4*(rc-(μ0+t+σ0))^4 : 0.

                fGW2 = 0. #model2*(3*zb^2/rc^2-1)

                k,ℓ,σ = vectors(Ub,outer,ns,ri,rb) # Form boundary basis

                C_BC = 0*model*FourVector((1.,0.,0.,0.))

                UBC,cm = BoundaryConditions(Ub,Hb,outer,rb,C_BC,fGW1,fGW2,k,ℓ,σ)

                if cm < 0
                    dρBC = UBC.dρ
                    dzBC = UBC.dz
                    PBC  = UBC.P

                    if ri[i] == nr
                        ε = abs(cm)*el2_1(αr)/h2_11(αr)/ds[i]
                    elseif ri[i] == nr-1
                        ε = abs(cm)*el2_2(αr)/h2_22(αr)/ds[i]
                    end
            
                    SATP  += ε*(PBC - P)
                    SATdρ += ε*(dρBC - dρ)
                    SATdz += ε*(dzBC - dz)
                end

            end

        end
        
    end

    return StateVector(SATg,SATdρ,SATdz,SATP)

end