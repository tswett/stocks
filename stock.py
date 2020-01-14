from mystats import Summary
import numpy
import pandas

class Stock:
    """Gain data for a particular stock, security or index."""
    def __init__(self, symbol, input_frame):
        self.symbol = symbol
        self.log_pct_gain = input_frame.log_pct_gain
        self.cal_days = input_frame.cal_days
        self.summary = Summary(self.log_pct_gain, self.cal_days)
        
    def __str__(self):
        return f"<Stock '{self}'>"
    
    def plot_empirical_cdf(self):
        self.summary.plot_empirical_cdf()

def import_spy_weekly():
    spy_weekly = pandas.read_csv('SPY.csv', parse_dates=['Date'])

    spy_weekly['log_pct_gain'] = 100 * numpy.log(spy_weekly['Adj Close']).diff()

    spy_weekly = (spy_weekly.groupby(pandas.Grouper(freq='W', key='Date'))
        .agg({'Date': numpy.max, 'log_pct_gain': numpy.sum}))

    spy_weekly['cal_days'] = spy_weekly['Date'].diff()

    spy_weekly = spy_weekly.iloc[1:]

    spy_weekly['cal_days'] = spy_weekly.cal_days.dt.days
    
    return Stock('SPY', spy_weekly)