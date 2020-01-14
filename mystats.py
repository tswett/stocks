import math, pandas

class Summary():
    """Statistical summary information about a time series."""
    def __init__(self, x_series, days_series, adjust_for_days=False):
        """Create a Summary a value series and a length-of-time series.
        
        The x_series parameter contains the values of the series. The
        days_series parameter indicates the number of days that each value
        corresponds to.
        
        If adjust_for_days is True, then the summary statistics will be
        calculated using the total number of days, in order to give per-day
        statistics. If adjust_for_days is False, then days_series will be
        ignored when calculating the summary statistics.
        """
        self.dataframe = df = pandas.DataFrame({'days': days_series, 'x': x_series})
        
        if adjust_for_days:
            divisor = df.days.sum()
        else:
            divisor = df.days.count()
        
        self.mean = df.x.sum() / divisor
        self.variance = ((df.x - self.mean)**2).sum() / divisor
        self.stddev = math.sqrt(self.variance)
        
        df['z_score'] = (df.x - self.mean) / self.stddev
        
        self.skewness = (df.z_score**3).sum() / divisor
        self.kurtosis = (df.z_score**4).sum() / divisor
        
    def __str__(self):
        return f'Mean: {self.mean:.4g}. Standard deviation: {self.stddev:.4g}. Skewness: {self.skewness:.4g}. Kurtosis: {self.kurtosis:.4g}.'
        
    def _ipython_display_(self):
        from IPython.display import display
        print(self.__str__())
        display(self.dataframe)
        
    def plot_empirical_cdf(self):
        # from matplotlib import pyplot
        # pyplot.clf()
        self.dataframe.x.hist(cumulative=True, density=True, histtype='step', bins=1000)
        # pyplot.show()