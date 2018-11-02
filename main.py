import data
import manager
import trader
from agents import ddpg
import featureBuild
import backtest

QUEUE_SIZE = 15

if __name__ == "__main__":
    DATA_FILE_NAME = "resource\\OIH_adjusted.csv"

    trader = trader.Trader()
    data = data.Data(1, live=False)
    print("Loading Data:", DATA_FILE_NAME)
    data.loadData(DATA_FILE_NAME)
    print("Data Loaded")
    featureB = featureBuild.FeatureBuilder(data=data, queueSize=QUEUE_SIZE)
    print("Feature Builder Created")
    backTester = backtest.Backtest(cash=10000, data=data, counter=QUEUE_SIZE, minBrokerFee=1.00, perShareFee=0.0075, simple=True, stateObj=True)
    print("Backtester Created")
    agent = ddpg.DeepDPG(state_shape=QUEUE_SIZE, normal_state_shape=featureB.numFeatures, criticParams=ddpg.CriticParams(), policyParams=ddpg.PolicyParams(), OUParams=ddpg.OUParams())
    print("Agent Created")
    manager = manager.Manager(data=data, agent=agent, featureB=featureB, backtester=backTester, trader=trader, train=True)
    manager.run()