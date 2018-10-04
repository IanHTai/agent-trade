from executeInterface import ExecuteInterface
from threading import Thread
from queue import Queue
import random
import time

class Backtest(ExecuteInterface):
    FAILURE_CHANCE = 0.001
    SLIPPAGE_CHANCE = 0.42

    """
    TODO: Do logic/control for test/train split  in another class (maybe manager?)
    """
    def __init__(self, cash, data, minBrokerFee, perShareFee, counter=0):
        """
        :param cash: starting cash
        :param data: stock resources
        :param minBrokerFee:
        :param perShareFee:
        :param train: train/test
        :param split: index of start of test set in resources
        """
        self.__cash = cash
        self.__stockAmount = 0
        self.__data = data
        self.__counter = counter
        self.__concQueue = Queue()

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
                # TODO: figure out resources structure
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
        if buy:
            self.__concQueue.put([amount, -price*amount])
        else:
            self.__concQueue.put([-amount, price*amount])

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
        cashDelta = 0
        amountDelta = 0
        while not self.__concQueue.empty():
            [amount, cash] = self.__concQueue.get()
            amountDelta += amount
            cashDelta += cash
        self.__cash += cashDelta
        self.__stockAmount += amountDelta
        return self.__cash + self.__stockAmount
