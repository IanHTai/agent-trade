from executeInterface import ExecuteInterface

class LiveExecute(ExecuteInterface):
    def __init__(self, cash, data):
        self.__cash = cash
        self.__stockAmount = 0

    def buy(self, amount):
        """
        Send buy order to exchange
        :param amount: amount of stock to buy
        :return: successfully put order or not
        """
        return True

    def sell(self, amount):
        """
        Send sell order to exchange
        :param amount: amount of stock to sell
        :return: successfully put order or not
        """
        return True

    def doNothing(self):
        """
        Do nothing for this time interval
        Useful function as a reinforcement learning action
        :return: True
        """
        return True


    def value(self):
        """
        Get current value of stock + cash
        :return: amount of stock * avg stock price + cash
        """

        # Get values from broker

        return 0