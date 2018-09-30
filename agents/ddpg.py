from .baseAgent import BaseAgent
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Dropout
import random

class DeepDPG(BaseAgent):
    def __init__(self, input_size, params=Params()):
        self.input_size = input_size
        self.params = params
        self.model = Sequential()
        self.model.add(Conv1D(self.params.filters[0], self.params.kernelSize[0], activation='relu', input_shape=(self.input_size)))
        self.model.add(Conv1D(self.params.filters[1], self.params.kernelSize[1], activation='relu'))
        self.model.add(Conv1D(self.params.filters[2], self.params.kernelSize[2], activation='relu'))
        self.model.add(Dropout(self.params.dropout))
        self.model.add(Dense(self.params.denseUnits[0], activation='relu'))
        self.model.add(Dense(self.params.denseUnits[1], activation='tanh'))

    def runEpisode(self, backtester):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()

    def runOnce(self, features):
        raise NotImplementedError()

class Params:
    def __init__(self, filters=[32,32,32], kernel_size=[3,3,3], dense_units=[200,200], dropout=0.25):
        self.filters = filters
        self.kernelSize = kernel_size
        self.denseUnits = dense_units
        self.dropout = 0.25

class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.replayList = []

    def add(self, state, action, reward, next_state):
        while len(self.replayList) >= self.max_size:
            del self.replayList[0]
        self.replayList.append([state, action, reward, next_state])

    def get(self):
        return self.replayList[random.randint(0, len(self.replayList) - 1)]