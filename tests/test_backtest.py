import unittest
import backtest
import data

DATA_FILE_NAME = 'SPY_TEST.csv'

class BacktestTester(unittest.TestCase):
    def setUp(self):

        self.data = data.Data(1, live=False)
        print("Loading Data:", DATA_FILE_NAME)
        data.loadData(DATA_FILE_NAME)
        self.backtester = backtest.Backtest(cash=10000, data=self.data, minBrokerFee=1.00, perShareFee=0.0075,
                                            simple=True, stateObj=True)
