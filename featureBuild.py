import numpy as np
import queue

class FeatureBuilder:

    def __init__(self, data, listOfStocks, queueSize=10):
        self.__data = data
        self.__queues = {}
        self.__stockNames = listOfStocks
        for stock in listOfStocks:
            self.__queues[stock] = StockQueues(data, stock, queueSize)


    def getFeatures(self):
        stockFeatures = {}
        for stock in self.__stockNames:
            stockFeat = StockFeatures(stock)
            stockName, stockQueue = self.__queues[stock].getAll()
            assert stock == stockName
            stockFeat.avgPrice = stockQueue['avgPrice'].peekAll()
            stockFeat.volume = stockQueue['volumeQueue'].peekAll()
            stockFeat.buyPrice = stockQueue['buyPrice'].peekAll()
            stockFeat.sellPrice = stockQueue['sellPrice'].peekAll()

            stockFeatures[stock] = stockFeat
        return stockFeatures


class StockQueues:

    def __init__(self, data, stockName, queueSize):
        self.__name = stockName

        # Definitions for the queues of different data points that are tracked
        avgPriceQueue = self.fillQueue('AvgPrice', data, stockName, queueSize)
        volumeQueue = self.fillQueue('Volume', data, stockName, queueSize)
        buyPriceQueue = self.fillQueue('BuyPrice', data, stockName, queueSize)
        sellPriceQueue = self.fillQueue('SellPrice', data, stockName, queueSize)
        self.__queueDict = {
            'avgPrice': avgPriceQueue,
            'volumeQueue': volumeQueue,
            'buyPrice': buyPriceQueue,
            'sellPrice': sellPriceQueue
        }

    def fillQueue(self, type, data, stockName, queueSize):
        outQueue = PeekQueue(maxsize=queueSize)
        for i in range(0, queueSize):
            outQueue.put(data[stockName][type][i])
        return outQueue

    def getAll(self):
        return self.__name, self.__queueDict

    def putAll(self, inputDict):
        # Check format
        if not sorted(self.__queueDict.keys()) == sorted(inputDict.keys()):
            raise FormatException("Feature Build Stockqueue Update Format Clash -- Missing or Extra Keys")
        for key in inputDict.keys():
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

    def peek(self, index):
        return self.__list[index]

class FormatException(Exception):
    def __init__(self, message):
        self.message = message

class QueueFullException(Exception):
    def __init__(self, message):
        self.message = message

class StockFeatures:
    # Data object, doesn't have getters and setters
    def __init__(self, stockName):
        self.name = stockName