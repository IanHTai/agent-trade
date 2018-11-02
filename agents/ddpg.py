from .baseAgent import BaseAgent
from keras.models import Model, clone_model
from keras.layers import Input, concatenate, Conv1D, Dense, Dropout, BatchNormalization, Flatten
from keras.optimizers import Adam
from keras.initializers import RandomUniform
import random
import numpy as np
import keras.backend as K
import tensorflow as tf
import datetime
from random import choices
from benchmark import Timer
from backtest import State

DO_PROFILE = True

"""
Modified DDPG to run N steps per batch of training
This is to speed up training (better GPU memory usage with larger batches)
while still maintaining adequate exploration and exposure to environment
"""


class DeepDPG(BaseAgent):
    def __init__(self, state_shape, normal_state_shape, criticParams, policyParams, OUParams, episodes=6, batch_size=8192, tau=0.001, steps_per_batch=128):
        self.state_shape = state_shape
        self.normal_state_shape = normal_state_shape
        self.criticParams = criticParams
        self.policyParams = policyParams
        self.current_noise = 0
        self.OUParams = OUParams
        self.steps_per_batch = steps_per_batch

        """
        HYPERPARAM FOR # EPISODES, minibatch size
        """
        self.episodes = episodes
        self.batch_size = batch_size

        # Adjust tau to match steps per batch
        self.tau = tau * steps_per_batch

        print("Batch Size:", self.batch_size)
        print("Steps per batch:", self.steps_per_batch)


        # """
        # Disallowing GPU memory pre-allocation to see memory usage (for tuning batch size etc.)
        # """
        #
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        #
        # K.tensorflow_backend.set_session(tf.Session(config=config))


        self.replayBuffer = ReplayBuffer(max_size=1e6)

        self.critic_model, self.criticS, self.criticN_S, self.criticA, self.criticOut = self.buildCriticNet(state_shape, normal_state_shape, criticParams)


        # Improvised from https://github.com/germain-hug/Deep-RL-Keras/blob/master/DDPG/critic.py
        self.action_grads = K.function([self.criticS, self.criticA],
                                       K.gradients(self.criticOut, [self.criticA]))

        self.policy_model, self.policyS, self.policyN_S, self.policyOut = self.buildPolicyNet(state_shape, normal_state_shape, policyParams)

        self.critic_target = clone_model(self.critic_model)
        self.critic_target.set_weights(self.critic_model.get_weights())

        self.target_policy_model = clone_model(self.policy_model)
        self.target_policy_model.set_weights(self.policy_model.get_weights())


    def buildCriticNet(self, state_shape, normal_state_shape, params):
        _conv_state = Input(batch_shape=(None, state_shape, 1))
        _normal_state = Input(shape=(normal_state_shape,))
        _action = Input(shape=(1,))

        norm = BatchNormalization()(_conv_state)
        h1_conv = Conv1D(params.filters[0], params.kernelSize[0], activation='relu')(norm)
        h2_conv = Conv1D(params.filters[1], params.kernelSize[1], activation='relu')(h1_conv)
        h3 = Flatten()(h2_conv)
        norm_norm_state = BatchNormalization()(_normal_state)
        h3_5 = concatenate([h3, norm_norm_state])
        h4_drop = Dropout(params.dropout)(h3_5)
        h5 = concatenate([h4_drop, _action])
        h6 = Dense(params.denseUnits[0], activation='relu')(h5)
        h7 = Dense(params.denseUnits[1], activation='relu')(h6)

        uniform_init = RandomUniform(minval=params.init_minval, maxval=params.init_maxval)

        _out = Dense(1, activation='tanh', kernel_initializer=uniform_init, bias_initializer=uniform_init)(h7)

        model = Model(inputs=[_conv_state, _normal_state, _action], outputs=_out)

        adam = Adam(lr=params.learning_rate, decay=params.weight_decay)
        model.compile(loss='mse', optimizer=adam)

        return model, _conv_state, _normal_state, _action, _out

    def buildPolicyNet(self, state_shape, normal_state_shape, params):
        _state = Input(batch_shape=(None, state_shape, 1))
        _normal_state = Input(shape=(normal_state_shape,))

        norm = BatchNormalization()(_state)
        h1_conv = Conv1D(params.filters[0], params.kernelSize[0], activation='relu')(norm)
        h2_conv = Conv1D(params.filters[1], params.kernelSize[1], activation='relu')(h1_conv)
        h3 = Flatten()(h2_conv)
        norm_norm_state = BatchNormalization()(_normal_state)
        h3_5 = concatenate([h3, norm_norm_state])
        h4_drop = Dropout(params.dropout)(h3_5)
        h5 = Dense(params.denseUnits[0], activation='relu')(h4_drop)
        h6 = Dense(params.denseUnits[1], activation='relu')(h5)

        uniform_init = RandomUniform(minval=params.init_minval, maxval=params.init_maxval)

        _out = Dense(1, activation='tanh', kernel_initializer=uniform_init, bias_initializer=uniform_init)(h6)

        model = Model(inputs=[_state, _normal_state], outputs=_out)

        return model, _state, _normal_state, _out

    def criticGrads(self, states, actions):
        """
        Returns the gradients of the critic network w.r.t. action taken
        Used to find the policy gradient of the actor network
        """
        return self.action_grads([states, actions])

    def trainActor(self, action_grads, params):
        params_grad = tf.gradients(self.policyOut, self.policy_model.trainable_weights, np.negative(action_grads))
        zipped_grads = zip(params_grad, self.policy_model.trainable_weights)
        return tf.train.AdamOptimizer(learning_rate=params.learning_rate).apply_gradients(zipped_grads)


    def noise(self, params):
        self.current_noise += params.theta * (params.mu - self.current_noise) + params.sigma * random.gauss(0, 1)
        return self.current_noise

    def runEpisode(self, backtester):
        raise NotImplementedError()


    def train(self, featureBuilder, backtester):
        print("Time at start of training:", datetime.datetime.now())
        self.timer = Timer('Training', profile=DO_PROFILE)


        #TODO : FEATUREBUILDER and add stock amount to state info

        train_cumulative_rewards = []
        test_cumulative_rewards = []

        # Populate replaybuffer with random actions first
        # for episode in range(0, 1):
        #     self.current_noise = 0
        #     state = featureBuilder.getFeatures()
        #     done = False
        #     while not done:
        #         prev_value = backtester.value()
        #         action = self.zeroFriendly(self.noise(self.OUParams))
        #         raw_new_state, reward, done, info = backtester.step(action)
        #         new_state = featureBuilder.add(raw_new_state)
        #         self.replayBuffer.add(state=state, action=action, reward=reward, next_state=new_state, done=done)
        #         state = new_state
        #     featureBuilder.reset()
        #     backtester.reset()
        # self.timer.checkpoint("Buffer Populated")

        print("Replaybuffer populated with random actions")

        for episode in range(0, self.episodes):
            self.current_noise = 0
            state = featureBuilder.getFeatures()
            cumulative_reward = 1
            done = False
            counter_in_episode = 0
            div_counter = 0

            while not done:

                self.timer.checkpoint("Start of steps per batch")

                prev_value = backtester.value()
                for step_per_batch in range(0, self.steps_per_batch):
                    counter_in_episode += 1
                    action = self.zeroFriendly(self.policy_model.predict(state)[0][0] + self.noise(self.OUParams))
                    self.timer.checkpoint("Policy model predict")
                    raw_new_state, reward, done, info = backtester.step(action)
                    self.timer.checkpoint("Backtester step")
                    new_state = featureBuilder.add(raw_new_state)
                    self.timer.checkpoint("Featurebuilder add")
                    self.replayBuffer.add(state=state, action=action, reward=reward, next_state=new_state, done=done)
                    self.timer.checkpoint("Replaybuffer added to")
                    state = new_state
                    cumulative_reward *= (reward + 1)

                samples = self.replayBuffer.getBatch(self.batch_size)

                self.timer.checkpoint("Batch retrieve from replaybuffer")

                states = np.array(list(e[0] for e in samples))
                actions = np.array(list(e[1] for e in samples)).reshape(self.batch_size, 1)
                rewards = np.array(list(e[2] for e in samples))
                next_states = np.array(list(e[3] for e in samples))
                dones = np.array(list(e[4] for e in samples))
                y_i = np.array(list(0.0 for _ in samples))

                self.timer.checkpoint("Batch separated into states, actions, etc.")

                states = states.reshape(self.batch_size, self.state_shape, 1)
                next_states = next_states.reshape(self.batch_size, self.state_shape, 1)

                self.timer.checkpoint("States reshaped")

                done_mask = dones == True
                not_done_mask = dones == False

                np.putmask(y_i, mask=done_mask, values=rewards[done_mask])

                target_critic_predicted = self.critic_target.predict([next_states[not_done_mask], self.target_policy_model.predict(next_states[not_done_mask])])

                not_done_values = rewards[not_done_mask] + self.criticParams.discount * target_critic_predicted[0]
                np.putmask(y_i, mask=not_done_mask, values=not_done_values)


                self.timer.checkpoint("y_i's updated with bellman equation")


                self.critic_model.train_on_batch([states, actions], y_i)

                self.timer.checkpoint("Critic model batch trained")

                criticGrads = self.criticGrads(states, actions)

                self.timer.checkpoint("Critic gradients computed")

                self.trainActor(criticGrads, params=self.policyParams)

                self.timer.checkpoint("Actor model batch trained")

                self.updateTargets()

                self.timer.checkpoint("Targets updated and episode step complete")

                if not (counter_in_episode // 10000 == div_counter):
                    div_counter = counter_in_episode // 10000
                    print(datetime.datetime.now(), "Data Point", counter_in_episode)
                    print("Cumulative reward in episode so far", cumulative_reward)
                    print("Value so far", backtester.value())
                    self.timer.printDict()
                

            backtester.reset()
            print("Training Episode", episode, "Complete. Cumulative Reward:", cumulative_reward)

            train_cumulative_rewards.append(cumulative_reward)

            if episode % 3 == 0:
                """
                Testing, without action space noise
                """
                backtester.test(True)
                state = featureBuilder.getFeatures()

                #Feature builder shouldn't need to be set in testing mode since its queues should already be filled with the
                #last of the training sequence's data

                test_cumulative_reward = 1
                test_done = False

                while not test_done:
                    prev_value = backtester.value()
                    action = self.zeroFriendly(self.policy_model.predict(state))

                    raw_new_state, reward, test_done, info = backtester.step(action)
                    new_state = featureBuilder.add(raw_new_state)

                    state = new_state

                    test_cumulative_reward *= (reward + 1)

                backtester.test(False)
                backtester.reset()

                print("Testing Episode Complete. Cumulative Test Reward:", test_cumulative_reward)

                test_cumulative_rewards.append(test_cumulative_reward)

            featureBuilder.reset()


        # SAVE WEIGHTS

        # Use most recent test_cumulative_reward for inclusion in filename

        filename_root = "model_backups\\" + datetime.datetime.now().strftime("%y_%m_%d_%H_%M") + "_" + str(test_cumulative_reward)

        actor_model_name = filename_root + "_actor.h5"

        critic_model_name = filename_root + "_critic.h5"

        self.policy_model.save_weights(actor_model_name)
        self.critic_model.save_weights(critic_model_name)

        return train_cumulative_rewards, test_cumulative_rewards, actor_model_name, critic_model_name

    def loadModels(self, actor_model_name, critic_model_name):
        self.critic_model.load_weights(critic_model_name)
        self.policy_model.load_weights(actor_model_name)
        self.target_policy_model.load_weights(actor_model_name)
        self.critic_target.load_weights(critic_model_name)

    def updateTargets(self):
        critic_target_weights = self.critic_target.get_weights()
        critic_weights = self.critic_model.get_weights()

        self.timer.checkpoint("Critic target weights gotten")
        # Have to do the updates in loops because the format of the weights is a list of np arrays
        # The np arrays can be directly multiplied by scalars but the list cannot

        new_critic_target_weights = [self.tau * critic_weights[i] + (1 - self.tau) * critic_target_weights[i] for i in
                                     range(0, len(critic_weights))]
        self.timer.checkpoint("New critic target weights calculated")

        self.critic_target.set_weights(new_critic_target_weights)

        self.timer.checkpoint("Critic target weights set")

        actor_target_weights = self.target_policy_model.get_weights()
        actor_weights = self.policy_model.get_weights()

        self.timer.checkpoint("Actor target weights gotten")
        new_actor_target_weights = [self.tau * actor_weights[i] + (1 - self.tau) * actor_target_weights[i] for i in
                                    range(0, len(actor_weights))]

        self.timer.checkpoint("New actor target weights calculated")

        self.target_policy_model.set_weights(new_actor_target_weights)

        self.timer.checkpoint("Actor target weights set")

    def zeroFriendly(self, action):
        """
        Makes action space between [-5e-2, 5e-2] = 0, since continuous NN will likely not output perfect 0.0
        :param action: action output from NN
        :return: zero-friendly action
        """

        if action <= 5e-2 and action >= -5e-2:
            action = 0.0
        return action

    def runOnce(self, features):
        raise NotImplementedError()


class CriticParams:
    def __init__(self, filters=[16,16,16], kernel_size=[3,3,3], dense_units=[100,100], dropout=0.25):
        self.filters = filters
        self.kernelSize = kernel_size
        self.denseUnits = dense_units
        self.dropout = dropout
        self.learning_rate = 1e-3 # PLACEHOLDER
        self.discount = 0.99
        self.weight_decay = 1e-2
        self.target_update = 0.001
        self.init_minval = -3e-4
        self.init_maxval = 3e-4

class PolicyParams:
    def __init__(self, filters=[16,16,16], kernel_size=[3,3,3], dense_units=[100,100], dropout=0.25):
        self.filters = filters
        self.kernelSize = kernel_size
        self.denseUnits = dense_units
        self.dropout = dropout
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
        self.replayList.append([state, action, reward, next_state, done])

    def get(self):
        return self.replayList[random.randint(0, len(self.replayList) - 1)]

    def getBatch(self, batchSize):
        return choices(self.replayList, k=batchSize)


