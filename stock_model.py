from distributions import NigDistribution
import math
from matplotlib import pyplot
from scipy.stats import norm, norminvgauss
from scipy.optimize import minimize

class NigModel:
    def __init__(self, dist, stock):
        self.dist = dist
        self.stock = stock
        
        self.alpha = dist.alpha
        self.beta = dist.beta
        self.mu = dist.mu
        self.delta = dist.delta
        
        self.scipy_alpha = dist.delta * dist.alpha
        self.scipy_beta = dist.delta * dist.beta
        
        self.mean = dist.mean
        self.variance = dist.variance
        self.stddev = dist.stddev
        self.skewness = dist.skewness
        self.kurtosis = dist.kurtosis
        
        self.sorted_gains = stock.log_pct_gain.sort_values()
        
    @staticmethod
    def from_parameters(alpha, beta, mu, delta, stock):
        dist = NigDistribution(alpha, beta, mu, delta)
        return NigModel(dist, stock)
        
    @staticmethod
    def from_stock_moments(stock):
        mean = stock.summary.mean
        variance = stock.summary.variance
        skewness = stock.summary.skewness
        excess_kurt = stock.summary.kurtosis - 3
        
        dist = NigDistribution.from_moments(mean, variance, skewness, excess_kurt)
        
        return NigModel(dist, stock)
        
    def reoptimize(self, do_print=False):
        dist = self.dist
        
        def evaluate(params):
            alpha, beta_over_alpha, mu, delta = params
            beta = beta_over_alpha * alpha
            scipy_alpha = delta * alpha
            scipy_beta = delta * beta
            return -norminvgauss.logpdf(self.sorted_gains, scipy_alpha, scipy_beta, mu, delta).sum()
        
        starting_point = [self.alpha, self.beta / self.alpha, self.mu, self.delta]
        bounds = [(0,None),(-1,1),(None,None),(None,None)]
        optimization = minimize(evaluate, starting_point, bounds=bounds)
        
        if do_print:
            print(optimization)
            
        alpha, beta_over_alpha, mu, delta = optimization.x
            
        return NigModel.from_parameters(alpha, beta_over_alpha * alpha, mu, delta, self.stock)
    
    def plot_comparison(self):
        dist = self.dist
        
        def cdf(x):
            return norminvgauss.cdf(x, self.scipy_alpha, self.scipy_beta, self.mu, self.delta)

        pyplot.clf()
        pyplot.figure(figsize=(17,11))
        self.stock.plot_empirical_cdf()
        pyplot.plot(self.sorted_gains, cdf(self.sorted_gains))
        pyplot.show()
        
    def scipy_stats(self):
        dist = self.dist
        
        return norminvgauss.stats(self.scipy_alpha, self.scipy_beta, self.mu, self.delta, moments='mvsk')