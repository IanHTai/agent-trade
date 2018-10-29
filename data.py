import csv

class Data:

    """
    The data class is essentially structured as a list of dicts, complete with support for square bracket notation

    """

    def __init__(self, interval, live=False):
        self.__interval = interval
        self.__live = live

    def checkMarket(self):
        """
        Check if market is open
        :return: boolean
        """
        return True
    def new(self):
        """
        Get update from data source at relevant time intervals
        :return: Array of data info
        """

        # wait until relevant time

        return None

    def liveData(self):
        return self.__live

    def loadData(self, filename):
        with open(filename, 'r') as dataFile:
            reader = csv.DictReader(dataFile, delimiter=',')
            self.__data = [r for r in reader]

    def __getitem__(self, item):
        return self.__data[item]

    def __len__(self):
        return len(self.__data)