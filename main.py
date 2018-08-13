import manager, trader, agent, data

if __name__ == "__main__":
    trader = trader.Trader()
    data = data.Data()
    agent = agent.Agent()
    manager = manager.Manager(data, agent, trader)
    manager.run()