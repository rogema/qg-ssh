import numpy


def psi2uv(psi,grd):
    ny,nx,=numpy.shape(grd.mask)
    i_c = slice(1, -1)
    i_n = slice(2, None)
    i_p = slice(0, -2)
    u=numpy.zeros((ny,nx))
    v=numpy.zeros((ny,nx))
    
    u[i_c, 1:] = -(psi[i_n, :-1] + psi[i_n, 1:] - psi[i_p, 1:] - psi[i_p,:-1]) / grd.dy[i_c, 1:] * .25
    v[1:, i_c] = (psi[1:, i_n] + psi[:-1, i_n] - psi[:-1, i_p] - psi[1:, i_p]) / grd.dx[1:, i_c] *.25
    u[numpy.isnan(u)]=0
    v[numpy.isnan(v)]=0
    
    return u,v


def qrhs(u,um,v,vm,q,qm,betax,betay,grd,way):
    
    rq=numpy.zeros((grd.ny,grd.nx))
    
    uplus=way*0.5*(u[2:-2,2:-2]+u[2:-2,3:-1])
    uplus[uplus<0]=0
    uminus=way*0.5*(u[2:-2,2:-2]+u[2:-2,3:-1])
    uminus[uminus>0]=0
    vplus=way*0.5*(v[2:-2,2:-2]+v[3:-1,2:-2])
    vplus[vplus<0]=0
    vminus=way*0.5*(v[2:-2,2:-2]+v[3:-1,2:-2])
    vminus[vminus>=0]=0

    uplusm=way*0.5*(um[2:-2,2:-2]+um[2:-2,3:-1])
    uplusm[uplusm<0]=0
    uminusm=way*0.5*(um[2:-2,2:-2]+um[2:-2,3:-1])
    uminusm[uminusm>0]=0
    vplusm=way*0.5*(vm[2:-2,2:-2]+vm[3:-1,2:-2])
    vplusm[vplusm<0]=0
    vminusm=way*0.5*(vm[2:-2,2:-2]+vm[3:-1,2:-2])
    vminusm[vminusm>=0]=0

    rq[2:-2,2:-2] =rq[2:-2,2:-2] - uplus*1/(6*grd.dx[2:-2,2:-2])*(2*q[2:-2,3:-1]+3*q[2:-2,2:-2]- 6*q[2:-2,1:-3]+q[2:-2,:-4]) \
                                 + uminus*1/(6*grd.dx[2:-2,2:-2])*(q[2:-2,4:]-6*q[2:-2,3:-1]+ 3*q[2:-2,2:-2]+2*q[2:-2,1:-3]) \
                                 - vplus*1/(6*grd.dy[2:-2,2:-2])*(2*q[3:-1,2:-2]+3*q[2:-2,2:-2]- 6*q[1:-3,2:-2]+q[:-4,2:-2]) \
                                 + vminus*1/(6*grd.dy[2:-2,2:-2])*(q[4:,2:-2]-6*q[3:-1,2:-2]+ 3*q[2:-2,2:-2]+2*q[1:-3,2:-2])

    rq[2:-2,2:-2] =rq[2:-2,2:-2] - uplusm*1/(6*grd.dx[2:-2,2:-2])*(2*q[2:-2,3:-1]+3*q[2:-2,2:-2]- 6*q[2:-2,1:-3]+q[2:-2,:-4]) \
                                 + uminusm*1/(6*grd.dx[2:-2,2:-2])*(q[2:-2,4:]-6*q[2:-2,3:-1]+ 3*q[2:-2,2:-2]+2*q[2:-2,1:-3]) \
                                 - vplusm*1/(6*grd.dy[2:-2,2:-2])*(2*q[3:-1,2:-2]+3*q[2:-2,2:-2]- 6*q[1:-3,2:-2]+q[:-4,2:-2]) \
                                 + vminusm*1/(6*grd.dy[2:-2,2:-2])*(q[4:,2:-2]-6*q[3:-1,2:-2]+ 3*q[2:-2,2:-2]+2*q[1:-3,2:-2])

    rq[2:-2,2:-2] =rq[2:-2,2:-2] - uplus*1/(6*grd.dx[2:-2,2:-2])*(2*qm[2:-2,3:-1]+3*qm[2:-2,2:-2]- 6*qm[2:-2,1:-3]+qm[2:-2,:-4]) \
                                 + uminus*1/(6*grd.dx[2:-2,2:-2])*(qm[2:-2,4:]-6*qm[2:-2,3:-1]+ 3*qm[2:-2,2:-2]+2*qm[2:-2,1:-3]) \
                                 - vplus*1/(6*grd.dy[2:-2,2:-2])*(2*qm[3:-1,2:-2]+3*qm[2:-2,2:-2]- 6*qm[1:-3,2:-2]+qm[:-4,2:-2]) \
                                 + vminus*1/(6*grd.dy[2:-2,2:-2])*(qm[4:,2:-2]-6*qm[3:-1,2:-2]+ 3*qm[2:-2,2:-2]+2*qm[1:-3,2:-2])

    rq[2:-2,2:-2]=rq[2:-2,2:-2] -(betay[2:-2,2:-2]+ (grd.f1[3:-1,2:-2]-grd.f1[1:-3,2:-2])/(2*grd.dy[2:-2,2:-2]))*way*0.5*(v[2:-2,2:-2]+v[3:-1,2:-2]) -betax[2:-2,2:-2]*way*0.5*(u[2:-2,2:-2]+u[2:-2,3:-1]) # +beta y


                               
    rq[grd.mask<=1]=0
    rq[grd.mask<=1]=0

    return rq,


def qrhsdiff(psi,snu,grd):
    i_c = slice(2, -2)
    i_n = slice(3, -1)
    i_p = slice(1, -3)

    rq=numpy.zeros((grd.ny,grd.nx))

    rq[i_c, i_c] += -snu[i_c, i_c] * (psi[i_c, i_n] + psi[i_c, i_p] - 2 * psi[i_c, i_c]) / (grd.dx[i_c, i_c] ** 2) \
                    -snu[i_c, i_c] * (psi[i_n, i_c] + psi[i_p, i_c] - 2 * psi[i_c, i_c]) / (grd.dy[i_c, i_c] ** 2)

    rq[grd.mask<=1]=0 

    return rq


