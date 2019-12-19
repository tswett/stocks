import math, pandas

class Summary():
    def __init__(self, x_series, days_series):
        self.dataframe = df = pandas.DataFrame({'days': days_series, 'x': x_series})
        
        total_days = df.days.sum()
        
        self.mean = df.x.sum() / total_days
        self.variance = ((df.x - self.mean)**2).sum() / total_days
        self.stddev = math.sqrt(self.variance)
        
        df['z_score'] = (df.x - self.mean) / self.stddev
        
        self.skewness = (df.z_score**3).sum() / total_days
        self.kurtosis = (df.z_score**4).sum() / total_days
        
    def _ipython_display_(self):
        from IPython.display import display
        print(f'Mean: {self.mean:.4g}. Standard deviation: {self.stddev:.4g}. Skewness: {self.skewness:.4g}. Kurtosis: {self.kurtosis:.4g}.')
        display(self.dataframe)
        
    def plot_empirical_cdf(self):
        # from matplotlib import pyplot
        # pyplot.clf()
        self.dataframe.x.hist(cumulative=True, density=True, histtype='step', bins=1000)
        # pyplot.show()