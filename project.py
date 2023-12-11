import numpy as np
import matplotlib.pyplot as plt
import LBParticles

G = 0.00449987

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
    lbpre = LBParticles.lbprecomputer.load("big_10_1000_alpha2p2_lbpre.pickle")

    ordershape = int(10 ** (1 + np.random.random() * 2))
    ordertime = int(np.random.random() * 7 + 1)
    psir=LBParticles.logpotential(220.0)
    nu = np.sqrt(4.0 * np.pi * LBParticles.G * 0.2)

    xCart = np.array([8100.0, 0.0, 21.0])
    vCart = np.array([10.0, 230.0, 10.0])
    partarray = []
    for j in range(2):

        dvcart = np.random.randn(3) * 10.0

        vCartThis = vCart + dvcart

        part = LBParticles.particleLB(xCart, vCartThis, psir, nu, lbpre, ordershape=ordershape, ordertime=ordertime)
        partarray.append(part)

    p1 = LBParticles.perturbedParticle()
    p1.add(0.0,partarray[0])
    p2 = LBParticles.perturbedParticle()
    p2.add(0.0,partarray[1])
    tPerb = findClosestApproach(p1,p2,0)
    print(tPerb)
    print(applyPerturbation(p1,p2,tPerb,5e5))
    return 0


if __name__ == '__main__':
    main()

