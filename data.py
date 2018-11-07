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
        Get update from resources source at relevant time intervals
        :return: Array of resources info
        """

        # wait until relevant time

        return None

    def liveData(self):
        return self.__live

    def loadData(self, filename):
        with open(filename, 'r') as dataFile:
            reader = TypedDictReader(dataFile, delimiter=',', fieldtypes=[str, str, float, float, float, float, int])
            self.__data = [r for r in reader]

    def __getitem__(self, item):
        return self.__data[item]

    def __len__(self):
        return len(self.__data)


class TypedDictReader(csv.DictReader):
    def __init__(self, f, fieldnames=None, restkey=None, restval=None, dialect="excel", fieldtypes=None, *args, **kwds):

        csv.DictReader.__init__(self, f, fieldnames, restkey, restval, dialect, *args, **kwds)
        self._fieldtypes = fieldtypes

    def __next__(self):
        d = csv.DictReader.__next__(self)
        if len(self._fieldtypes) >= len(d) :
            # extract the values in the same order as the csv header
            ivalues = map(d.get, self._fieldnames)
            # apply type conversions
            iconverted = (x(y) for (x,y) in zip(self._fieldtypes, ivalues))
            # pass the field names and the converted values to the dict constructor
            out = dict(zip(self._fieldnames, iconverted))

        return out