from .baseAgent import BaseAgent
from keras.models import Model, clone_model
from keras.layers import Input, merge, Conv1D, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
import random

class DeepDPG(BaseAgent):
    def __init__(self, state_shape, params=Params()):
        self.state_shape = state_shape
        self.params = params

        self.critic_model, self.A, self.S = self.buildCritic(state_shape, params)


        self.critic_copy = clone_model(self.critic_model)
        self.critic_copy.set_weights(self.critic_model.get_weights())

        self.behaviour_policy_model = clone_model(self.target_policy_model)
        self.behaviour_policy_model.set_weights(self.target_policy_model.get_weights())

    def buildCritic(self, state_shape, params):
        _state = Input(shape=(state_shape,))
        _action = Input(shape=(1,))
        norm = BatchNormalization()(_state)
        h1_conv = Conv1D(params.filters[0], params.kernelSize[0], activation='relu')(norm)
        h2_conv = Conv1D(params.filters[1], params.kernelSize[1], activation='relu')(h1_conv)
        h3_conv = Conv1D(params.filters[2], params.kernelSize[2], activation='relu')(h2_conv)
        h4_drop = Dropout(params.dropout)(h3_conv)
        h4_action_dense = Dense(1, activation='relu')(_action)
        h5 = merge([h4_drop, h4_action_dense], mode='concat')
        h6 = Dense(params.denseUnits[0], activation='relu')(h5)
        _out = Dense(1, activation='tanh')(h6)

        model = Model(input=[_state, _action], output=_out)

        adam = Adam(lr=params.learning_rate)
        model.compile(loss='mse', optimizer=adam)

        return model, _action, _state

    def runEpisode(self, backtester):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()

    def runOnce(self, features):
        raise NotImplementedError()

    def updateClones(self):
        self.critic_copy = clone_model(self.critic_model)
        self.critic_copy.set_weights(self.critic_model.get_weights())

class Params:
    def __init__(self, filters=[32,32,32], kernel_size=[3,3,3], dense_units=[200,200], dropout=0.25):
        self.filters = filters
        self.kernelSize = kernel_size
        self.denseUnits = dense_units
        self.dropout = 0.25
        self.learning_rate = 0.99 # PLACEHOLDER

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