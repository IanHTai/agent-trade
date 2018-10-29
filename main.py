import data
import manager
import trader
from agents import ddpg

if __name__ == "__main__":
    trader = trader.Trader()
    data = data.Data(1, live=False)
    data.loadData("resource\\OIH_adjusted.csv")
    agent = ddpg.DeepDPG(state_shape=10)
    manager = manager.Manager(data, agent, trader)
    manager.run()