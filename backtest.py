from executeInterface import ExecuteInterface
from threading import Thread
from queue import Queue
import random
import time
import math

class Backtest(ExecuteInterface):
    FAILURE_CHANCE = 0.001
    SLIPPAGE_CHANCE = 0.42

    """
    TODO: Do logic/control for test/train split  in another class (maybe manager?)
    """
    def __init__(self, cash, data, minBrokerFee, perShareFee, counter=0, trainSplit=0.8, simple=True, stateObj=True):
        """
        :param cash: starting cash
        :param data: stock resources
        :param minBrokerFee:
        :param perShareFee:
        :param counter: current index pointer in dataset
        :param trainSplit: train/test split ratio
        :param simple: False: use slipping/delay simulator
        """

        self.__cash = cash
        self.__stockAmount = 0
        self.__data = data
        self.__counter = counter
        self.__initCounter = counter
        self.__concQueue = Queue()
        self.__simple = simple

        self.__train_len = trainSplit*len(data)
        self.__test = False

        self.__counter_limit = self.__train_len

        # Set default amount of money spent on fees on every trade
        self.__minBrokerFee = minBrokerFee
        self.__perShareFee = perShareFee

        self.__stateObj = stateObj

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
                price = self.__data[self.__counter]['Close']
                if amount * price + max(self.__minBrokerFee, self.__perShareFee * amount) > self.__cash:
                    amount = math.floor((self.__cash - max(self.__minBrokerFee, self.__perShareFee * amount))/price)
                self.__cash -= max(self.__minBrokerFee, self.__perShareFee * amount)
                self.__counter += 1
                if not amount == 0:
                    if not self.__simple:
                        Thread(target=self.orderSim, args=(True, price, amount)).start()
                    else:
                        self.__cash -= amount * price
                        self.__stockAmount += amount
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
        if amount > self.__stockAmount:
            amount == self.__stockAmount
        if self.__cash < max(self.__minBrokerFee, self.__perShareFee * amount):
            self.__counter += 1
            return False
        else:
            self.__cash -= max(self.__minBrokerFee, self.__perShareFee * amount)
            price = self.__data[self.__counter]['Close']
            self.__counter += 1
            if not amount == 0:
                if not self.__simple:
                    Thread(target=self.orderSim, args=(False, price, amount)).start()
                else:
                    self.__cash += price * amount
                    self.__stockAmount -= amount
            return True

    def doNothing(self):
        self.__counter += 1

    def value(self):
        cashDelta = 0
        amountDelta = 0
        qCounter = 0
        while not self.__concQueue.empty():
            [amount, cash] = self.__concQueue.queue.index(qCounter)
            amountDelta += amount
            cashDelta += cash
        self.__cash += cashDelta
        self.__stockAmount += amountDelta
        return self.__cash + self.__stockAmount*self.__data[self.__counter]['Close']

    def step(self, actionValue):
        old_value = self.value()
        if actionValue > 0:
            success = self.buy(math.ceil(actionValue*math.floor(self.__cash/self.__data[self.__counter]['Close'])))
        elif actionValue < 0:
            actionValue = -actionValue
            success = self.sell(math.ceil(actionValue*self.__stockAmount))
        else:
            success = self.doNothing()

        reward = self.value() / float(old_value) - 1

        if self.__counter >= self.__counter_limit - 1:
            done = True
        else:
            done = False

        state = self.returnState(self.__data[self.__counter], self.__cash, self.__stockAmount, self.value())

        return state, reward, done, {}


    def returnState(self, inputDict, cash, amount, value):
        if self.__stateObj:
            return State(inputDict, cash, amount, value)
        else:
            return inputDict

    def reset(self):
        self.__counter = self.__initCounter

    def test(self, test_bool=False):
        self.__test = test_bool

        if test_bool:
            self.__counter = self.__train_len
            self.__counter_limit = len(self.__data)
        else:
            self.__counter = self.__initCounter
            self.__counter_limit = self.__train_len

    def cash(self):
        return self.__cash

    def stockAmount(self):
        return self.__stockAmount

class State:
    def __init__(self, inputDict, cash, amount, value):
        self.inputDict = inputDict
        self.cash = cash
        self.amount = amount
        self.value = value