import data
import manager
import trader
from agents import ddpg

if __name__ == "__main__":
    trader = trader.Trader()
    data = data.Data(1, live=False)
    print("Data Loading")
    data.loadData("resource\\OIH_adjusted.csv")
    print("Data Loaded")
    agent = ddpg.DeepDPG(state_shape=10, criticParams=ddpg.CriticParams(), policyParams=ddpg.PolicyParams(), OUParams=ddpg.OUParams())
    print("Agent Created")
    manager = manager.Manager(data, agent, trader, train=True, queueSize=10)
    manager.run()