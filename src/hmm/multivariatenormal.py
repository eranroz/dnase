__author__ = 'eran'
import numpy as np


class MultivariateNormal():
    """
    Normal distribution for multidimensional data
    @param mean: mean array
    @param cov: covariance matrix
    """

    def __init__(self, mean, cov):
        self.mean = mean
        cov = np.array(cov)  # in case cov is list not np array
        d = cov.shape[0]

        s, u = np.linalg.eigh(cov)
        s_pinv = np.array([0 if abs(x) < 1e-5 else 1 / x for x in s], dtype=float)
        prec_U = np.multiply(u, np.sqrt(s_pinv))
        pdet = np.prod(s[s > 1e-5])

        self.norm_p = d * np.log(2 * np.pi) + np.log(pdet)
        self.prec_U = prec_U

    def pdf(self, x):
        """
        probability density function
        @param x: d-point or n X d-points array
        @return:
        """
        maha = np.sum(np.square(np.dot(x - self.mean, self.prec_U)), axis=-1)
        return np.exp(-0.5 * (self.norm_p + maha))


def test_mtuidim_hmm():
    from matplotlib.pylab import plt
    x, y = np.mgrid[-1:1:.1, -1:1:.1]
    pos = np.array([x.ravel(), y.ravel()])
    rv = MultivariateNormal([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]])
    z = rv.pdf(pos.T)

    plt.contourf(x, y, z.reshape((20, 20)))
    plt.show()

if __name__ == "__main__":
    test_mtuidim_hmm()