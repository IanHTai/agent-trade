import backtest
class Manager:
    def __init__(self, data, agent, trader, train=True):
        self.__data = data
        self.__agent = agent
        self.__trader = trader
        self.__train = train

    def run(self):
        """
        Liaison between info, agent, and trader modules
        :return: Null
        """

        if self.__train:
            self.__backtest = backtest.Backtest(cash=10000, data=self.__data, minBrokerFee=1.00, perShareFee=0.0075)

            self.__agent.train()

        else:
            stop_flag = False

            while not stop_flag:
                info = self.__data.new()
                decision = self.__agent.update(info)
                if not decision == None:
                    success = self.__trader.execute(decision)
                else:
                    success = True
                stop_flag = self.checkStop()

    def checkStop(self):
        """
        Checks for interrupt
        :return: boolean indicating interrupt
        """
        return False