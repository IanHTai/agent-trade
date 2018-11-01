import time

class Timer():

    def __init__(self, name='', profile=False, rounding=4):
        self.name = name
        self.timerDict = {}
        self.lastTime = time.time()
        self.DO_PROFILE = profile
        self.rounding = rounding

    @property
    def elapsed(self):
        return time.time() - self.lastTime

    def checkpoint(self, name=''):
        if self.DO_PROFILE:
            if not name in self.timerDict:
                self.timerDict[name] = 0.0
            self.timerDict[name] += round(self.elapsed, self.rounding)
            self.lastTime = time.time()

    def printDict(self):
        if self.DO_PROFILE:
            print("Times for each operation:")
            print(self.timerDict)
            percentageDict = {}
            timerSum = sum(self.timerDict.values())

            for key in self.timerDict.keys():
                percentageDict[key] = round(self.timerDict[key] / timerSum * 100, 2)

            print("Percentage of time taken for each operation:")
            print(percentageDict)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if self.DO_PROFILE:
            self.checkpoint('finished')
        pass
