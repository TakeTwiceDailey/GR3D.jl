
function constraints(U::StateVector,H_,Rb,ns,ds,ls,ri)

    g  = U.g
    dx = U.dx
    dy = U.dy
    dz = U.dz
    P  = U.P

    xi,yi,zi = ri

    hx,hy,hz,_ = ds
    lx,ly,lz=ls
    x = -lx/2 + (xi-1)*hx; y = -ly/2 + (yi-1)*hy; z = -lz/2 + (zi-1)*hz;

    #H_ = FourVector(μ -> fH_(x,y,z,μ)) # FourVector(fH_(x,y,z))

    r = (x,y,z)

    in_domain,_,_,_,_ = find_boundary(Rb,ns,ls,ds,r,ri)

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

function constraint_energy_cell(U,H_,Rb,ns,ls,ds,ri)
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

    in_domain,nls,nrs,αls,αrs = find_boundary(Rb,ns,ls,ds,r,ri)

    if in_domain

        gi = inv(g)

        α = 1/sqrt(-gi[1,1])

        βx = -gi[1,2]/gi[1,1]
        βy = -gi[1,3]/gi[1,1]
        βz = -gi[1,4]/gi[1,1]

        n  = FourVector((1,-βx,-βy,-βz))/α

        m = gi + 2*symmetric(@einsum n[μ]*n[ν])

        C_ = constraints(U,H_,Rb,ns,ds,ls,ri)

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

function volume_cell(U,Rb,ns,ls,ds,ri)
    # Calculates the volume element of the 3-space

    xi,yi,zi = ri

    dx,dy,dz,_ = ds
    lx,ly,lz=ls
    x = -lx/2 + (xi-1)*dx; y = -ly/2 + (yi-1)*dy; z = -lz/2 + (zi-1)*dz;
    r = (x,y,z)

    in_domain,nls,nrs,αls,αrs = find_boundary(Rb,ns,ls,ds,r,ri)

    if in_domain

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

        return rootγ(U)*h*dx*dy*dz

    else

        return 0.

    end

end

function invariant(U,∂tU,Rb,ns,ls,ds,ri)
    # Calculates the Kretschmann Scalar

    hx,hy,hz,_ = ds
    lx,ly,lz=ls
    x = -lx/2 + (xi-1)*dx; y = -ly/2 + (yi-1)*dy; z = -lz/2 + (zi-1)*dz;
    r = (x,y,z)

    in_domain,nls,nrs,αls,αrs = find_boundary(Rb,ns,ls,ds,r,ri)

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

# @inline function VolumeIntegral(f,U,ns,ls,ds,ri)
#     # Calculates the volume integral

#     xi,yi,zi = ri

#     dx,dy,dz,_ = ds
#     lx,ly,lz=ls
#     x = -lx/2 + (xi-1)*dx; y = -ly/2 + (yi-1)*dy; z = -lz/2 + (zi-1)*dz;
#     r = (x,y,z)

#     in_domain,nls,nrs,αls,αrs = find_boundary(ns,ls,ds,r,ri)

#     if in_domain

#         h = 1.

#         for i in 1:3
#             if ri[i] == nrs[i]
#                 h *= h2_11(αrs[i])
#             elseif ri[i] == nrs[i]-1
#                 h *= h2_22(αrs[i])
#             elseif ri[i] == nls[i]
#                 h *= h2_11(αls[i])
#             elseif ri[i] == nls[i]+1
#                 h *= h2_22(αls[i])
#             end
#         end

#         return f(U)*h*dx*dy*dz

#     else

#         return 0.

#     end


#     return result

# end