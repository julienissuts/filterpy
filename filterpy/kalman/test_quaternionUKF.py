#!/usr/bin/env python3
from filterpy.kalman.quaternionUKF import QuaternionSigmaPoint, QuaternionUKF
from filterpy.kalman.sigma_points import JulierSigmaPoints
import numpy as np
from scipy.linalg import cholesky


def test_quaternion_sigma_points():

    x = np.array([1.0, 2.0, 3.0])

    P = np.array([
        [1.0, 0.2, 0.1],
        [0.2, 1.0, 0.3],
        [0.1, 0.3, 1.0]
    ])

    Q = np.array([
        [0.01, 0.0, 0.0],
        [0.0, 0.01, 0.0],
        [0.0, 0.0, 0.01]
    ])

    n = len(x)
    sigmapoints = QuaternionSigmaPoint(n, kappa=0.1, sqrt_method=cholesky)
    normal_sigmapoints = JulierSigmaPoints(n, kappa=0.1, sqrt_method=cholesky)

    sigmas = sigmapoints.sigma_points(x, P, Q)
    sigmas_normal = normal_sigmapoints.sigma_points(x, Q + P)
    print(f"sigmas Quaternion: {sigmas}")
    print(f"sigmas: {sigmas_normal}")

def fx(x, dt):
    return x

def test_quaternion_ukf():
    x = np.array([1.0, 2.0, 3.0])

    P = np.array([
        [1.0, 0.2, 0.1],
        [0.2, 1.0, 0.3],
        [0.1, 0.3, 1.0]
    ])

    Q = np.array([
        [0.01, 0.0, 0.0],
        [0.0, 0.01, 0.0],
        [0.0, 0.0, 0.01]
    ])

    n = len(x)

    sigmapoints = JulierSigmaPoints(n, kappa=0.1, sqrt_method=cholesky)

    quatukf = QuaternionUKF(dim_x=n, dim_z=n, points=sigmapoints, dt=1, fx=fx, hx=None)

    quatukf.x = x
    quatukf.P = P
    quatukf.Q = Q

    quatukf.predict(dt=1)

    print(f"sigmas new: {quatukf.sigmas_f}")
    print(f"xnew: {quatukf.x}")
    print(f"Pnew: {quatukf.P}")


test_quaternion_ukf()

test_quaternion_sigma_points()