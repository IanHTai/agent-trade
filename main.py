import data
import manager
import trader
from agents import ddpg

if __name__ == "__main__":
    DATA_FILE_NAME = "resource\\OIH_adjusted.csv"

    trader = trader.Trader()
    data = data.Data(1, live=False)
    print("Loading Data:", DATA_FILE_NAME)
    data.loadData(DATA_FILE_NAME)
    print("Data Loaded")
    agent = ddpg.DeepDPG(state_shape=10, criticParams=ddpg.CriticParams(), policyParams=ddpg.PolicyParams(), OUParams=ddpg.OUParams())
    print("Agent Created")
    manager = manager.Manager(data, agent, trader, train=True, queueSize=10)
    manager.run()