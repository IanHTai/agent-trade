from .baseAgent import BaseAgent
from keras.models import Model, clone_model
from keras.layers import Input, merge, Conv1D, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.initializers import RandomUniform
import random
import numpy as np

class DeepDPG(BaseAgent):
    def __init__(self, state_shape, criticParams=CriticParams(), policyParams=PolicyParams(), OUParams=OUParams()):
        self.state_shape = state_shape
        self.criticParams = criticParams
        self.current_noise = 0
        self.OUParams = OUParams

        """
        HYPERPARAM FOR # EPISODES, minibatch size
        """
        self.episodes = 1000
        self.batch_size = 64

        self.replayBuffer = ReplayBuffer(max_size=1e6)

        self.critic_model = self.buildCriticNet(state_shape, criticParams)
        self.policy_model = self.buildPolicyNet(state_shape, policyParams)

        self.critic_copy = clone_model(self.critic_model)
        self.critic_copy.set_weights(self.critic_model.get_weights())

        self.target_policy_model = clone_model(self.policy_model)
        self.target_policy_model.set_weights(self.policy_model.get_weights())

    def buildCriticNet(self, state_shape, params):
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

        uniform_init = RandomUniform(minval=params.init_minval, maxval=params.init_maxval)

        _out = Dense(1, activation='tanh', kernel_initializer=uniform_init, bias_initializer=uniform_init)(h6)

        model = Model(input=[_state, _action], output=_out)

        adam = Adam(lr=params.learning_rate, decay=params.weight_decay)
        model.compile(loss='mse', optimizer=adam)

        return model

    def buildPolicyNet(self, state_shape, params):
        _state = Input(shape=(state_shape,))

        norm = BatchNormalization()(_state)
        h1_conv = Conv1D(params.filters[0], params.kernelSize[0], activation='relu')(norm)
        h2_conv = Conv1D(params.filters[1], params.kernelSize[1], activation='relu')(h1_conv)
        h3_conv = Conv1D(params.filters[2], params.kernelSize[2], activation='relu')(h2_conv)
        h4_drop = Dropout(params.dropout)(h3_conv)
        h5 = Dense(params.denseUnits[0], activation='relu')(h4_drop)

        uniform_init = RandomUniform(minval=params.init_minval, maxval=params.init_maxval)

        _out = Dense(1, activation='tanh', kernel_initializer=uniform_init, bias_initializer=uniform_init)(h5)

        model = Model(input=[_state], output=_out)
        adam = Adam(lr=params.learning_rate)
        model.compile(loss='mse', optimizer=adam)
        return model

    def noise(self, params):
        self.current_noise += params.theta * (params.mu + self.current_noise) + params.sigma * random.gauss(0, 1)
        return self.current_noise

    def runEpisode(self, backtester):
        raise NotImplementedError()

    def train(self, featureBuilder, backtester):
        #TODO : FEATUREBUILDER and add stock amount to state info


        # Populate replaybuffer with random actions first
        for episode in range(0, 2):
            self.current_noise = 0
            state = featureBuilder.getFeatures()
            while not featureBuilder.isEnd():
                prev_value = backtester.value()
                action = self.zeroFriendly(self.noise(self.OUParams))
                new_state, reward, done, info = backtester.step(action)
                self.replayBuffer.add(state=state, action=action, reward=reward, next_state=new_state)
                state = new_state



        for episode in range(0, self.episodes):
            self.current_noise = 0
            state = featureBuilder.getFeatures()
            done = False
            while not done:
                prev_value = backtester.value()
                action = self.zeroFriendly(self.policy_model.predict(state) + self.noise(self.OUParams))

                new_state, reward, done, info = backtester.step(action)
                self.replayBuffer.add(state=state, action=action, reward=reward, next_state=new_state)
                state = new_state

                samples = []
                for i in range(0, self.batch_size):
                    samples.append(self.replayBuffer.get())

                states = np.asarray(e[0] for e in samples)
                actions = np.asarray(e[1] for e in samples)
                rewards = np.asarray(e[2] for e in samples)
                next_states = np.asarray(e[3] for e in samples)
                y_i = np.add(rewards, self.criticParams.discount * self.critic_copy.predict(next_states, self.target_policy_model.predict(next_states)))
                # TODO finish up training sequence






    def zeroFriendly(self, action):
        """
        Makes action space between [-1e-1, 1e-1] = 0, since continuous NN will likely not output perfect 0.0
        :param action: action output from NN
        :return: zero-friendly action
        """

        if action <= 1e-1 or action >= -1e-1:
            action = 0.0
        return action

    def runOnce(self, features):
        raise NotImplementedError()

    def updateClones(self):
        self.critic_copy = clone_model(self.critic_model)
        self.critic_copy.set_weights(self.critic_model.get_weights())

class CriticParams:
    def __init__(self, filters=[32,32,32], kernel_size=[3,3,3], dense_units=[200,200], dropout=0.25):
        self.filters = filter
        self.kernelSize = kernel_size
        self.denseUnits = dense_units
        self.dropout = 0.25
        self.learning_rate = 1e-3 # PLACEHOLDER
        self.discount = 0.99
        self.weight_decay = 1e-2
        self.target_update = 0.001
        self.init_minval = -3e-4
        self.init_maxval = 3e-4

class PolicyParams:
    def __init__(self, filters=[32,32,32], kernel_size=[3,3,3], dense_units=[200,200], dropout=0.25):
        self.filters = filter
        self.kernelSize = kernel_size
        self.denseUnits = dense_units
        self.dropout = 0.25
        self.learning_rate = 1e-4 # PLACEHOLDER
        self.init_minval = -3e-4
        self.init_maxval = 3e-4

class OUParams:
    def __init__(self, theta=0.15, mu=0.0, sigma=0.2):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma

class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.replayList = []

    def add(self, state, action, reward, next_state, done):
        while len(self.replayList) > self.max_size:
            del self.replayList[0]
        self.replayList.append([state, action, reward, next_state])

    def get(self):
        return self.replayList[random.randint(0, len(self.replayList) - 1)]