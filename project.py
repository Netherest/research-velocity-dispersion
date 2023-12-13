import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import lbparticles

G = 0.00449987


def particle_ivp2(t, y):
    M = 5.5e5
    '''The derivative function for scipy's solve_ivp - used to compare integrated particle trajectories to our semi-analytic versions'''
    vcirc = 220.0
    nu0 = np.sqrt(4.0*np.pi*G*0.2)
    alpha = 2.2

    xx1 = y[0,:]
    yy1 = y[1,:]
    zz1 = y[2,:]
    vx1 = y[3,:]
    vy1 = y[4,:]
    vz1 = y[5,:]

    xx2 = y[6,:]
    yy2 = y[7,:]
    zz2 = y[8,:]
    vx2 = y[9,:]
    vy2 = y[10,:]
    vz2 = y[11,:]

    r1 = np.sqrt(xx1*xx1 + yy1*yy1)
    g1 = -vcirc*vcirc / r1
    r2 = np.sqrt(xx2*xx2 + yy2*yy2)
    g2 = -vcirc*vcirc / r2
    nu1 = nu0 * (r1/8100.0)**(-alpha/2.0)
    nu2 = nu0 * (r2/8100.0)**(-alpha/2.0)
    d = np.sqrt((xx1-xx2)**2+(yy1-yy2)**2+(zz1-zz2)**2)


    res = np.zeros( y.shape )
    res[0,:] = vx1
    res[1,:] = vy1
    res[2,:] = vz1
    res[3,:] = g1*xx1/r1 + G*M*(xx1-xx2)/d**3
    res[4,:] = g1*yy1/r1 + G*M*(yy1-yy2)/d**3
    res[5,:] = - zz1*nu1*nu1 + G*M*(zz1-zz2)/d**3
    res[6,:] = vx2
    res[7,:] = vy2
    res[8,:] = vz2
    res[9,:] = g2*xx2/r2 + G*M*1e-18*(xx2-xx1)/d**3
    res[10,:] = g2*yy2/r2 + G*M*1e-18*(yy2-yy1)/d**3
    res[11,:] = - zz2*nu2*nu2 + G*M*1e-18*(zz2-zz1)/d**3

    return res

def integrateVariable(t,p1,p2):
    i = p1.xvabs(t)
    j = p2.xvabs(t)
    k = np.zeros((12,1))
    k[0:3,0] = i[0]
    k[3:6,0] = i[1]
    k[6:9,0] = j[0]
    k[9:,0] = j[1]
    return k

def distance(t,orb1,orb2):
    trefOrb1, solnOrb1 = orb1.getpart(t)
    trefOrb2, solnOrb2 = orb2.getpart(t)

    xcart1 = np.array(solnOrb1.xabs(t - trefOrb1))
    xcart2 = np.array(solnOrb2.xabs(t - trefOrb2))
    return np.sqrt(np.sum((xcart1 - xcart2) ** 2))

def findOptimums(data):
    delta = np.append(data[1:] - data[:-1],0)
    optimum = np.append(delta[1:] * delta[:-1],0)
    return optimum, delta

def relativeDistanceData(orb1, orb2, tmin):
    ts = np.linspace(tmin + 0.001, tmin + orb1.stitchedSolutions[0].Tr + orb2.stitchedSolutions[0].Tr,
                     1000)
    dis = np.array([])
    for t in ts:
        dis = np.append(dis, (distance(t,orb1,orb2)))
    return ts,dis

def findClosestApproach(orb1, orb2, tmin):
    ts,dis = relativeDistanceData(orb1,orb2,tmin)
    optimum, delta = findOptimums(dis)
    time_opt = ts[np.logical_and(optimum < 0, delta < 0)]
    return time_opt[0]

def applyPerturbation(orbISO, orbStar, tPerturb, mstar):
    # Assume tPerturb is the relevant time of closest approach. The ISO, whose 'perturbedParticle' orbit

    trefISO, solnISO = orbISO.getpart(tPerturb)
    trefStar, solnStar = orbStar.getpart(tPerturb)

    xcartStar = solnStar.xabs(tPerturb - trefStar)
    vcartStar = solnStar.vabs(tPerturb - trefStar)

    xcartISO = solnISO.xabs(tPerturb - trefISO)
    vcartISO = np.array(solnISO.vabs(tPerturb - trefISO))

    # initial
    vrel = np.array(vcartISO) - np.array(vcartStar)
    xrel = np.array(xcartISO) - np.array(xcartStar)

    b = np.sqrt(np.sum(xrel ** 2))
    V0 = np.sqrt(np.sum(vrel ** 2))

    dvperp = 2 * b * V0 ** 3 / (G * mstar) * 1.0 / (1.0 + b * b * V0 ** 4 / (G * G * mstar * mstar))
    dvpar = 2.0 * V0 / (1.0 + b * b * V0 ** 4 / (G * G * mstar * mstar))

    vnew = vcartISO - dvpar * vrel / V0 - dvperp * xrel / b

    return vnew

def main():
    lbpre = lbparticles.lbprecomputer.load("big_10_1000_alpha2p2_lbpre.pickle")

    ordershape = int(10 ** (1 + np.random.random() * 2))
    ordertime = int(np.random.random() * 7 + 1)
    psir=lbparticles.logpotential(220.0)
    nu = np.sqrt(4.0 * np.pi * lbparticles.G * 0.2)

    xCart = np.array([8100.0, 0.0, 21.0])
    vCart = np.array([10.0, 230.0, 10.0])
    partarray = []
    for j in range(2):

        dvcart = np.random.randn(3) * 10.0

        vCartThis = vCart + dvcart

        part = lbparticles.particleLB(xCart, vCartThis, psir, nu, lbpre, ordershape=ordershape, ordertime=ordertime)
        partarray.append(part)

    p1 = lbparticles.perturbedParticle()
    p1.add(0.0,partarray[0])
    p2 = lbparticles.perturbedParticle()
    p2.add(0.0,partarray[1])
    tPerb = findClosestApproach(p1,p2,0)
    print(tPerb)
    print(applyPerturbation(p1,p2,tPerb,5e5))
    a = integrateVariable(tPerb,p1,p2)
    relVelocity = np.sqrt((a[3]-a[9])**2+(a[4]-a[10])**2+(a[5]-a[11])**2)
    print(distance(tPerb,p1,p2))
    print(relVelocity)
    tInit = max(0,tPerb-5*distance(tPerb,p1,p2)/relVelocity[0])
    print(tInit)
    ts = np.linspace(tInit,tPerb + 5*distance(tPerb,p1,p2)/relVelocity[0], 10000)
    print(integrateVariable(tInit,p1,p2))
    data = scipy.integrate.solve_ivp( particle_ivp2, [np.min(ts),np.max(ts)],(integrateVariable(tInit,p1,p2)).flatten(),vectorized=True, t_eval=ts, atol=1.0e-7, rtol=1.0e-7, method='DOP853')
    np.savetxt("data.csv",data)
    return 0



if __name__ == '__main__':
    main()
