import math
from matplotlib import pyplot
import numpy
from scipy.stats import norminvgauss

class NigDistribution:
    """A particular normal-inverse Gaussian distribution."""
    def __init__(self, alpha, beta, mu, delta):
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.delta = delta
        
        self.gamma = gamma = math.sqrt(alpha**2 - beta**2)
        
        scipy_alpha = delta * alpha
        scipy_beta = delta * beta
        
        self.scipy_dist = norminvgauss(scipy_alpha, scipy_beta, mu, delta)

        # Moment formulas from Wikipedia.
        self.mean = mu + delta * beta / gamma
        self.variance = delta * alpha**2 / gamma**3
        self.skewness = 3 * beta / alpha / math.sqrt(delta * gamma)
        self.kurtosis = 3 + 3 * (1 + 4 * beta**2 / alpha**2) / delta / gamma
        
        self.stddev = math.sqrt(self.variance)
        self.kappa_3 = self.skewness * self.stddev**3
        self.kappa_4 = (self.kurtosis - 3) * self.stddev**4
        
        # Extra variables...
        self.rho = self.alpha**2 / self.beta**2
        
    @staticmethod
    def from_moments(mean, variance, skewness, excess_kurt):
        # Source: I solved the equations from Wikipedia myself.
        
        rho = 3 * excess_kurt / skewness**2 - 4
        beta = 3 / (skewness * (rho - 1) * math.sqrt(variance))
        alpha = abs(beta) * math.sqrt(rho)
        gamma = abs(beta) * math.sqrt(rho - 1)
        delta = 9 / (skewness**2 * rho * gamma)
        mu = mean - (beta * delta) / gamma
        
        return NigDistribution(alpha, beta, mu, delta)
    
    def logpdf(self, x):
        scipy_alpha = delta * alpha
        scipy_beta = delta * beta
        return norminvgauss.logpdf(x, scipy_alpha, scipy_beta, mu, delta)
    
    def plot_cdf(self, start, stop):
        pyplot.clf()
        #pyplot.figure(figsize=(17,11))
        space = numpy.linspace(start, stop, 1000)
        pyplot.plot(space, self.scipy_dist.cdf(space))
        pyplot.show()