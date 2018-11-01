from keras.layers.convolutional import Conv1D


class BaseAgent:
    def __init__(self):
        raise NotImplementedError()

    def runEpisode(self, backtester):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()

    def runOnce(self, features):
        raise NotImplementedError()