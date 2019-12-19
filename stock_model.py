import math
from matplotlib import pyplot
from scipy.stats import norm, norminvgauss
from scipy.optimize import minimize

class NigModel:
    def __init__(self, alpha, beta, mu, delta, stock):
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.delta = delta
        
        gamma = math.sqrt(alpha**2 - beta**2)

        self.mean = mu + delta * beta / gamma
        self.variance = delta * alpha**2 / gamma**3
        self.stddev = math.sqrt(self.variance)
        self.skewness = 3 * beta / alpha / math.sqrt(delta * gamma)
        self.kurtosis = 3 + 3 * (1 + 4 * beta**2 / alpha**2) / delta / gamma
        
        self.stock = stock
        self.sorted_gains = stock.log_pct_gain.sort_values()
        
    @staticmethod
    def from_stock_moments(stock):
        # Formula from "The Normal Inverse Gaussian Distribution and the Pricing of Derivatives",
        # Eriksson et al., 2009
        
        mean = stock.summary.mean
        variance = stock.summary.variance
        skewness = stock.summary.skewness
        excess_kurt = stock.summary.kurtosis - 3
        
        rho = 3 * excess_kurt / skewness**2 - 4
        print(f'rho is {rho}')
        
        alpha = 3 * (4 / rho + 1) / math.sqrt(1 - 1 / rho) / excess_kurt
        beta = 3 * math.copysign(1, skewness) * (4 / rho + 1) / math.sqrt(rho - 1) / excess_kurt
        mu = mean - math.copysign(1, skewness) * math.sqrt(3 / rho * (4 / rho + 1) / excess_kurt * variance)
        delta = math.sqrt(3 * (4 / rho + 1) * (1 - 1 / rho) / excess_kurt * variance)
        
        return NigModel(alpha, beta, mu, delta, stock)
        
    def reoptimize(self, do_print=False):
        def evaluate(params):
            alpha, beta, loc, scale = params
            return -norminvgauss.logpdf(self.sorted_gains, alpha, beta, loc, scale).sum()
        
        starting_point = [self.alpha, self.beta, self.mu, self.delta]
        bounds = [(0,None),(-2,2),(None,None),(None,None)]
        optimization = minimize(evaluate, starting_point, bounds=bounds)
        
        if do_print:
            print(optimization)
            
        alpha, beta, mu, delta = optimization.x
            
        return NigModel(alpha, beta, mu, delta, self.stock)
    
    def plot_comparison(self):
        def cdf(x):
            return norminvgauss.cdf(x, self.alpha, self.beta, self.mu, self.delta)

        pyplot.clf()
        pyplot.figure(figsize=(17,11))
        self.stock.plot_empirical_cdf()
        pyplot.plot(self.sorted_gains, cdf(self.sorted_gains))
        pyplot.show()
        
    def scipy_stats(self):
        return norminvgauss.stats(self.alpha, self.beta, self.mu, self.delta, moments='mvsk')