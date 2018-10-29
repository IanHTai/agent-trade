import numpy as np

class FeatureBuilder:

    def __init__(self, data, queueSize=10):
        self.__data = data
        self.__queues = StockQueues(data, queueSize)


    def getFeatures(self):
        stockQueue = self.__queues.getAll()

        stockFeat = stockQueue['Close']

        return np.array(stockFeat)

    def add(self, inputDict):
        self.__queues.putAll(inputDict)
        return self.getFeatures()


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

    def getAll(self):
        return self.__queueDict

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
