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
        # Source: I solved the equations from Wikipedia myself.
        
        rho = 3 * excess_kurt / skewness**2 - 4
        beta = 3 / (skewness * (rho - 1) * math.sqrt(variance))
        alpha = abs(beta) * math.sqrt(rho)
        gamma = abs(beta) * math.sqrt(rho - 1)
        delta = 9 / (skewness**2 * rho * gamma)
        mu = mean - (beta * delta) / gamma
        
        return NigDistribution(alpha, beta, mu, delta)
    
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