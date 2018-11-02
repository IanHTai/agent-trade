import backtest
import featureBuild

class Manager:
    def __init__(self, data, agent, featureB, backtester, trader, train=True):
        self.__data = data
        self.__agent = agent
        self.__trader = trader
        self.__train = train
        self.__backtest = backtester
        self.__featureBuild = featureB

    def run(self):
        """
        Liaison between info, agent, and trader modules
        :return: Null
        """

        if self.__train:
            self.__agent.train(featureBuilder=self.__featureBuild, backtester=self.__backtest)

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