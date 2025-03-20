from filterpy.kalman import UKF
from filterpy.kalman import unscented_transform
import numpy as np
import sys

class QuaternionUKF():
    def __init__(self, dim_x, dim_z, dt, hx, fx, points,
                 sqrt_fn=None, x_mean_fn=None, z_mean_fn=None,
                 residual_x=None,
                 residual_z=None,
                 state_add=None):

        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x)
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)
        self.Q = np.eye(dim_x)
        self.R = np.eye(dim_z)
        self._dim_x = dim_x
        self._dim_z = dim_z
        self.points_fn = points
        self._dt = dt
        self._num_sigmas = points.num_sigmas()
        self.hx = hx
        self.fx = fx
        self.x_mean = x_mean_fn
        self.z_mean = z_mean_fn

        # Only computed only if requested via property
        self._log_likelihood = np.log(sys.float_info.min)
        self._likelihood = sys.float_info.min
        self._mahalanobis = None

        if sqrt_fn is None:
            self.msqrt = cholesky
        else:
            self.msqrt = sqrt_fn

        # weights for the means and covariances.
        self.Wm, self.Wc = points.Wm, points.Wc

        if residual_x is None:
            self.residual_x = np.subtract
        else:
            self.residual_x = residual_x

        if residual_z is None:
            self.residual_z = np.subtract
        else:
            self.residual_z = residual_z

        if state_add is None:
            self.state_add = np.add
        else:
            self.state_add = state_add

        # sigma points transformed through f(x) and h(x)
        # variables for efficiency so we don't recreate every update

        self.sigmas_f = np.zeros((self._num_sigmas, self._dim_x))
        self.sigmas_h = np.zeros((self._num_sigmas, self._dim_z))

        self.K = np.zeros((dim_x, dim_z))    # Kalman gain
        self.y = np.zeros((dim_z))           # residual
        self.z = np.array([[None]*dim_z]).T  # measurement
        self.S = np.zeros((dim_z, dim_z))    # system uncertainty
        self.SI = np.zeros((dim_z, dim_z))   # inverse system uncertainty

        self.inv = np.linalg.inv

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
        
    def predict(self, dt=None, UT=None, fx=None, **fx_args):

        if dt is None:
            dt = self._dt

        if UT is None:
            UT = unscented_transform

        # calculate sigma points for given mean and covariance
        self.compute_process_sigmas(dt, fx, **fx_args)

        #and pass sigmas through the unscented transform to compute prior
        # self.x, self.P = UT(self.sigmas_f, self.Wm, self.Wc, self.Q,
        #                     self.x_mean, self.residual_x)

        # update sigma points to reflect the new variance of the points
        # self.sigmas_f = self.points_fn.sigma_points(self.x, self.P)

        # save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)

    def compute_process_sigmas(self, dt, fx=None, **fx_args):
        """
        computes the values of sigmas_f. Normally a user would not call
        this, but it is useful if you need to call update more than once
        between calls to predict (to update for multiple simultaneous
        measurements), so the sigmas correctly reflect the updated state
        x, P.
        """

        if fx is None:
            fx = self.fx

        # calculate sigma points for given mean and covariance
        sigmas = self.points_fn.sigma_points(self.x, self.P + self.Q)

        for i, s in enumerate(sigmas):
            self.sigmas_f[i] = fx(s, dt, **fx_args)



from filterpy.kalman.sigma_points import JulierSigmaPoints
from scipy.linalg import cholesky
from filterpy.common import pretty_str


class QuaternionSigmaPoint():
    def __init__(self, n, kappa=0., sqrt_method=None, subtract=None):

        self.n = n
        self.kappa = kappa
        if sqrt_method is None:
            self.sqrt = cholesky
        else:
            self.sqrt = sqrt_method

        if subtract is None:
            self.subtract = np.subtract
        else:
            self.subtract = subtract

        self._compute_weights() # Juliers sigma weights

    def sigma_points(self, x, P, Q):
        # create sigma points
        if self.n != np.size(x):
            raise ValueError("expected size(x) {}, but size is {}".format(
                self.n, np.size(x)))

        n = self.n

        if np.isscalar(x):
            x = np.asarray([x])

        n = np.size(x)  # dimension of problem

        if np.isscalar(P):
            P = np.eye(n) * P
        else:
            P = np.atleast_2d(P)

        sigmas = np.zeros((2*n+1, n))

        # Calculate sqrt of P
        W = self.sqrt((n + self.kappa) * (P + Q)) # W = 3x3, P and Q 3x3
        print(x)
        sigmas[0] = x # unperturbed sigma point
        for k in range(n):
            # pylint: disable=bad-whitespace
            sigmas[k+1]   = self.subtract(x, -W[k]) #sigma pts from 1 to n
            sigmas[n+k+1] = self.subtract(x, W[k]) # sigma points from n+1 to 2n+1
        return sigmas

    def _compute_weights(self):
        """ Computes the weights for the unscented Kalman filter. In this
        formulation the weights for the mean and covariance are the same.
        """

        n = self.n
        k = self.kappa

        self.Wm = np.full(2*n+1, .5 / (n + k))
        self.Wm[0] = k / (n+k)
        self.Wc = self.Wm


    def __repr__(self):

        return '\n'.join([
            'JulierSigmaPoints object',
            pretty_str('n', self.n),
            pretty_str('kappa', self.kappa),
            pretty_str('Wm', self.Wm),
            pretty_str('Wc', self.Wc),
            pretty_str('subtract', self.subtract),
            pretty_str('sqrt', self.sqrt)
            ])