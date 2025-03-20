#!/usr/bin/env python3
from filterpy.kalman.quaternionUKF import QuaternionSigmaPoint, QuaternionUKF
from filterpy.kalman.sigma_points import JulierSigmaPoints
import numpy as np
from scipy.linalg import cholesky
from scipy.spatial.transform import Rotation as R


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


def mean_fn(sigmas, Wm):

    sigmas = R.from_quat(sigmas)

    avg_quat = sigmas.mean(Wm)

    return avg_quat

def residual_fn(a,b): # residual for quaternions
    a = R.from_quat(a)
    b = R.from_quat(b)
    diff = a * b.inv() #a and b need to be quaternions
    return diff


def test_mean_and_residual_fn():
    print("=== Testing Quaternion Averaging Function ===")

    euler_angles = np.array([[0,0,0],
                            [10,0,0],
                            [-10,0,0]])
    
    quats = R.from_euler('xyz', euler_angles, degrees=True).as_quat()

    Wm = np.array([0.333, 0.333, 0.333])

    avg_quat = mean_fn(quats, Wm) # calc avg of sigmas 

    print(f"avg_quat: {avg_quat.as_quat()}")

    error_quats = residual_fn(quats, avg_quat.as_quat()) # calc error quaternions ei

    error_rotvecs = error_quats.as_rotvec() # transform ei to rot vecs 

    avg_euler = avg_quat.as_euler('xyz', degrees=True)
    
    print("\nInput Euler Angles (degrees):")
    print(euler_angles)

    print("\nConverted Quaternions:")
    print(quats)

    print("\nWeights:")
    print(Wm)

    print("\nComputed Mean Quaternion:")
    print(avg_quat.as_quat())

    print("\nMean Rotation in Euler Angles (degrees):")
    print(avg_euler)

    print("\nError rotation vectors:")
    print(error_rotvecs)


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

test_mean_and_residual_fn()