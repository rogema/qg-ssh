import numpy

def psi2pv(psi1,grd):
    q1 = 1. / (grd.Rd ** 2) * (-psi1)

    q1[1:-1, 1:-1] = \
        ((psi1[2:, 1:-1] + psi1[:-2, 1:-1] - 2 * psi1[1:-1, 1:-1]) / grd.dy[1:-1, 1:-1] ** 2 \
        + (psi1[1:-1, 2:] + psi1[1:-1, :-2] - 2 * psi1[1:-1, 1:-1]) / grd.dx[1:-1, 1:-1] ** 2) \
        + 1. / (grd.Rd[1:-1, 1:-1] ** 2) * (-psi1[1:-1, 1:-1])

    m = grd.mask == 1
    q1[m] = 1. / (grd.Rd[m] ** 2) * (-psi1[m])

    q1[grd.mask == 0] = 0
    return q1,


def pv2psi(q1, psi1g, grd, nitr=1):
    ny, nx = psi1g.shape
    x = psi1g[grd.indi, grd.indj]
    q1d = q1[grd.indi, grd.indj]
    aaa = numpy.ones(grd.np)
    aaa[grd.vp1] = 0.

    fff1 = 1. / (grd.Rd1d ** 2)
    fff1[grd.vp1] = 1.

    ccc = q1[grd.indi,grd.indj]
    ccc[grd.vp1] = x[grd.vp1]
    vec = x
    avec = compute_avec(vec, aaa, fff1, grd)
    gg = avec - ccc
    p = - gg  

    for itr in range(nitr-1):
        vec = p
        avec = compute_avec(vec, aaa, fff1, grd)
        tmp = numpy.dot(p, avec)
        
        if tmp != 0.:
            s = - numpy.dot(p, gg) / tmp
        else:
            s = 1.
            
        a1 = numpy.dot(gg, gg)
        x = x + s * p
        vec = x
        avec = compute_avec(vec, aaa, fff1, grd)
        gg = avec - ccc

        a2 = numpy.dot(gg, gg)
        if a1 != 0:
            beta = a2 / a1
        else:
            beta = 1.
        p = - gg + beta * p
        
    vec = p
    avec = compute_avec(vec, aaa, fff1, grd)
    val1 = -numpy.dot(p, gg)
    val2 = numpy.dot(p, avec)
    
    if (val2 == 0.): 
        s = 1.
    else: 
        s = val1 / val2
        
    a1 = numpy.dot(gg, gg)
    x = x + s * p
    
    # back to 2D
    psi1 = numpy.empty((ny, nx))
    psi1[:, :] = numpy.NAN
    psi1[grd.indi, grd.indj] = x[:grd.np]
   
    return psi1


def compute_avec(vec, aaa, fff1, grd):
    """Pourquoi ne pas utilise de slice? au lieu des index  pour vp2, ...
    """
    
    avec=numpy.empty(grd.np)
    
    avec[grd.vp2] = \
        aaa[grd.vp2] * ((vec[grd.vp2e] + vec[grd.vp2w] - 2 * vec[grd.vp2]) / (grd.dx1d[grd.vp2] ** 2) + (vec[grd.vp2n] + vec[grd.vp2s] - 2 * vec[grd.vp2]) / (grd.dy1d[grd.vp2] ** 2)) \
        + fff1[grd.vp2] * (-vec[grd.vp2])

    avec[grd.vp1]=vec[grd.vp1]
    return avec


