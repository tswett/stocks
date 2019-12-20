import math
from scipy.stats import norminvgauss

class NigDistribution:
    def __init__(self, alpha, beta, mu, delta):
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.delta = delta
        
        self.gamma = gamma = math.sqrt(alpha**2 - beta**2)

        # Moment formulas from Wikipedia.
        self.mean = mu + delta * beta / gamma
        self.variance = delta * alpha**2 / gamma**3
        self.stddev = math.sqrt(self.variance)
        self.skewness = 3 * beta / alpha / math.sqrt(delta * gamma)
        self.kurtosis = 3 + 3 * (1 + 4 * beta**2 / alpha**2) / delta / gamma
        
        # Extra variables...
        self.rho = self.alpha**2 / self.beta**2
        
    @staticmethod
    def from_moments(mean, variance, skewness, excess_kurt):
        # Formula from "The Normal Inverse Gaussian Distribution and the Pricing of Derivatives",
        # Eriksson et al., 2009
        
        rho = 3 * excess_kurt / skewness**2 - 4
        print(f'rho is {rho}')
        
        alpha = 3 * (4 / rho + 1) / math.sqrt(1 - 1 / rho) / excess_kurt
        beta = 3 * math.copysign(1, skewness) * (4 / rho + 1) / math.sqrt(rho - 1) / excess_kurt
        mu = mean - math.copysign(1, skewness) * math.sqrt(3 / rho * (4 / rho + 1) / excess_kurt * variance)
        delta = math.sqrt(3 * (4 / rho + 1) * (1 - 1 / rho) / excess_kurt * variance)
        
        return NigDistribution(alpha, beta, mu, delta)
    
    def print_scipy_moment_comparison(self):
        print(f'us: mean {self.mean}, variance {self.variance}, skewness {self.skewness}, excess kurtosis {self.kurtosis - 3}')
        
        alpha_for_scipy = self.delta * self.alpha
        beta_for_scipy = self.delta * self.beta
        loc = self.mu
        scale = self.delta
        
        sp_mean_a, sp_var_a, sp_skew_a, sp_kurt_a = norminvgauss.stats(alpha_for_scipy, beta_for_scipy, loc, scale, moments='mvsk')
        
        print(f'scipy: mean {sp_mean_a}, variance {sp_var_a}, skewness {sp_skew_a}, excess kurtosis {sp_kurt_a}')
    
    #@staticmethod
    #def from_moments_solver(mean, variance, skewness, excess_kurt):
    #    def evaluate(params):
    #        alpha, beta_over_alpha, mu, delta = params
    #        return -norminvgauss.logpdf(self.sorted_gains, alpha, beta, loc, scale).sum()
    #    
    #    starting_point = [self.alpha, self.beta, self.mu, self.delta]
    #    bounds = [(0,None),(-2,2),(None,None),(None,None)]
    #    optimization = minimize(evaluate, starting_point, bounds=bounds)
    #    
    #    if do_print:
    #        print(optimization)
    #        
    #    alpha, beta, loc, scale = optimization.x
    #        
    #    return NigModel(alpha, beta, loc / scale, scale**2, self.stock)