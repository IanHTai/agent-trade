from executeInterface import ExecuteInterface
from threading import Thread
import random
import time

class Backtest(ExecuteInterface):
    FAILURE_CHANCE = 0.001
    SLIPPAGE_CHANCE = 0.42

    def __init__(self, cash, data, minBrokerFee, perShareFee):
        self.__cash = cash
        self.__stockAmount = 0
        self.__data = data
        self.__counter = 0

        # Set default amount of money spent on fees on every trade
        self.__minBrokerFee = minBrokerFee
        self.__perShareFee = perShareFee

    def buy(self, amount):
        """
        Send buy order to simulated broker
        :param amount: amount of stock to buy
        :return: order successfully received by simulated broker
        """

        if self.__cash < max(self.__minBrokerFee, self.__perShareFee * amount):
            self.__counter += 1
            return False
        else:

            # Random chance of failure
            if random.random() < self.FAILURE_CHANCE:
                return False
            else:
                self.__cash -= max(self.__minBrokerFee, self.__perShareFee * amount)
                # TODO: figure out data structure
                price = self.__data.get('avgPrice', self.__counter)
                self.__counter += 1
                Thread(target=self.buyOrder, args=(price, amount)).start()
                return True

    def orderSim(self, buy, price, amount):
        """
        Simulate order placing/fulfilment on market
        :param buy: Boolean for buy or sell
        :param price: Price of stock (either buy or sell)
        :param amount: Amount to buy/sell
        :return: nothing
        """

        """
        According to https://blog.quantopian.com/accurate-slippage-model-comparing-real-simulated-transaction-costs/
        Real-life slippage is roughly normal but has a huge spike at 0, with mean at 0.02% higher than desired price
        chance of non-zero slippage looks roughly 42%
        """
        if random.random < self.SLIPPAGE_CHANCE:
            slippagePercent = random.gauss(0.02, 0.02)
            price = price * slippagePercent

        # Wait some amount of time (ms) before order fulfilled, proportional to volume?
        timeToWaitBetween = amount

        time.sleep(float(timeToWaitBetween)/100)

    def sell(self, amount):
        if self.__cash < max(self.__minBrokerFee, self.__perShareFee * amount):
            self.__counter += 1
            return False
        else:
            self.__cash -= max(self.__minBrokerFee, self.__perShareFee * amount)

            self.__counter += 1


    def doNothing(self):


        self.__counter += 1

    def value(self):
