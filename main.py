import data
import manager
import trader
from agents import baseAgent

if __name__ == "__main__":
    trader = trader.Trader()
    data = data.Data()
    agent = baseAgent.Agent()
    manager = manager.Manager(data, agent, trader)
    manager.run()