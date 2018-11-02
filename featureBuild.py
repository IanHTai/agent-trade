import numpy as np
from backtest import State

class FeatureBuilder:

    def __init__(self, data, queueSize=15):
        self.__data = data
        self.__queues = StockQueues(data, queueSize)
        self.__queueSize = queueSize

        # Starting values used for RSI
        diffs = [t - s for s, t in zip(self.__queues.getQueue('Close').peekAll(), self.__queues.getQueue('Close').peekAll()[1:])]

        self.upChangeAvg = len([x for x in diffs if x > 0]) / float(queueSize)
        self.downChangeAvg = len([x for x in diffs if x < 0]) / float(queueSize)

        # Starting values used for TEMA
        self.EMA = self.__queues.getQueue('Close').peek(0)
        self.EMA_2 = self.EMA
        self.EMA_3 = self.EMA
        alpha = 0.25
        for i in range(1, queueSize):
            old_ema = self.EMA
            old_ema_2 = self.EMA_2
            self.EMA = alpha * self.__queues.getQueue('Close').peek(i) + (1 - alpha) * self.EMA
            self.EMA_2 = alpha * self.EMA + (1 - alpha) * old_ema
            self.EMA_3 = alpha * self.EMA_2 + (1 - alpha) * old_ema_2

        # Change this whenever feature is added/removed
        self.numFeatures = 6

    def getFeatures(self):

        stockFeat = self.__queues.getQueue('Close')

        return np.array(stockFeat.peekAll()).reshape(1, self.__queueSize, 1)

    def add(self, inputDict):
        self.__queues.putAll(inputDict)
        return self.getFeatures()

    def addStateObj(self, stateObj):
        self.__queues.putAll(stateObj.inputDict)
        return self.getCombinedFeatures(stateObj)

    def getCombinedFeatures(self, stateObj=None):
        convFeats = self.getFeatures()
        normalFeats = np.array([self.cashAmountRatio(stateObj), self.volume(), self.williamsR(), self.roc(window=2),
                                self.rsi(window=14), self.tema(alpha=0.25)]).reshape(1, -1)
        return [convFeats, normalFeats]
    def reset(self):
        self.__queues = StockQueues(self.__data, self.__queueSize)


    """
    The following functions are for calculating technical indicators
    """

    def cashAmountRatio(self, stateObj):
        if stateObj is None:
            return 1
        stock_value = self.__queues.getQueue('Close').peek()
        if stateObj.amount == 0:
            return 1
        else:
            stock_amount_value = stock_value * stateObj.amount
            return stateObj.cash/float(stock_amount_value)

    def volume(self):
        return self.__queues.getQueue('Volume').peek()

    def williamsR(self, window=0):
        if window == 0:
            window = self.__queueSize
        highest = max(self.__queues.getQueue('High').peekAll()[-window:])
        lowest = min(self.__queues.getQueue('Low').peekAll()[-window:])
        close = self.__queues.getQueue('Close').peek()
        return -1.0*(highest - close)/(highest - lowest)

    def roc(self, window=0):
        # NOTE: window=2 means compare second-most recent close and most recent close
        if window == 0:
            window = self.__queueSize
        return (self.__queues.getQueue('Close').peek(-window) - self.__queues.getQueue('Close').peek()) / \
               self.__queues.getQueue('Close').peek(-window)

    def rsi(self, window=0):
        if window == 0:
            window = self.__queueSize
        queueSize = self.__queues.getQueue('Close').size
        last = self.__queues.getQueue('Close').peek(-2)
        current = self.__queues.getQueue('Close').peek()
        currentGain = max(0, current - last)
        currentLoss = min(0, current - last)

        self.upChangeAvg = (self.upChangeAvg * (window - 1) + currentGain) / window
        self.downChangeAvg = (self.downChangeAvg * (window - 1) + currentLoss) / window

        RS = self.upChangeAvg / self.downChangeAvg

        return 1. - 1. / (1 + RS)

    def tema(self, alpha):
        old_ema = self.EMA
        old_ema_2 = self.EMA_2
        self.EMA = alpha * self.__queues.getQueue('Close').peek() + (1 - alpha) * self.EMA
        self.EMA_2 = alpha * self.EMA + (1 - alpha) * old_ema
        self.EMA_3 = alpha * self.EMA_2 + (1 - alpha) * old_ema_2
        return 3. * self.EMA - 3. * self.EMA_2 + self.EMA_3



class StockQueues:

    def __init__(self, data, queueSize):

        # Definitions for the queues of different data points that are tracked
        openQueue = self.fillQueue('Open', data=data, queueSize=queueSize)
        highQueue = self.fillQueue('High', data=data, queueSize=queueSize)
        lowQueue = self.fillQueue('Low', data=data, queueSize=queueSize)
        closeQueue = self.fillQueue('Close', data=data, queueSize=queueSize)
        volumeQueue = self.fillQueue('Volume', data=data, queueSize=queueSize)

        self.__queueDict = {
            'Open': openQueue,
            'High': highQueue,
            'Low': lowQueue,
            'Close': closeQueue,
            'Volume': volumeQueue
        }

    def fillQueue(self, dtype, data, queueSize):
        outQueue = PeekQueue(maxsize=queueSize)
        for i in range(0, queueSize):
            outQueue.put(data[i][dtype])
        return outQueue

    def getQueue(self, queueName):
        return self.__queueDict[queueName]

    def getAll(self):
        return self.__queueDict

    def putAll(self, inputDict):
        for key in self.__queueDict.keys():
            self.__queueDict[key].get()
            self.__queueDict[key].put(inputDict[key])

class PeekQueue:
    def __init__(self, maxsize):
        self.__maxSize = maxsize
        self.__list = []
        self.__size = 0

    def put(self, object):
        if self.__size < self.__maxSize:
            self.__list.append(object)
            self.__size += 1
        else:
            raise QueueFullException('Inserting Into Full PeekQueue')

    def get(self):
        object = self.__list.pop(0)
        self.__size -= 1
        return object

    def peekAll(self):
        return self.__list

    def peek(self, index=-1):
        if index == -1:
            return self.__list[-1]
        else:
            return self.__list[index]

    @property
    def size(self):
        return self.__size


class FormatException(Exception):
    def __init__(self, message):
        self.message = message

class QueueFullException(Exception):
    def __init__(self, message):
        self.message = message
