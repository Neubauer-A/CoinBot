import numpy as np
from tf_agents.specs import array_spec
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import time_step as ts
from CoinBot.data.sampler import DataSampler

def TFKrakenTestEnv(sleep_interval=6.5):
    return tf_py_environment.TFPyEnvironment(KrakenTestEnv(sleep_interval))

class KrakenTestEnv(py_environment.PyEnvironment):  

    def __init__(self, sleep_interval=6.5):
        '''
        - The traded attribute indicates whether or not the agent has purchased btc 
          in the current episode.
        - The sampler parses and reshapes the data received from the Kraken client.
          The sample interval determines the number of seconds to wait before
          getting a new sample. 
        - The data attribute is the most recent sample.
        - The curr/prev ask/bid attributes correspond to the current lowest ask 
          price, the lowest ask price from the previous sample, the current highest
          bid price and the highest bid price from the previous sample. Each episode 
          begins with matching curr and prev attributes (no reward on step 1).
        - The action spec corresponds to doing nothing (0) or making a trade (1).
        - The observation spec corresponds to the five arrays provided by the 
          sampler, the sentiment score, and the value of the traded attribute.
        - The state holds the aforementioned arrays.
        '''

        self._traded = 0
        self._sampler = DataSampler(sleep_interval)
        self._data = self._sampler.get_sample()
        self._curr_ask = self._data[1][-1][0]
        self._curr_bid = self._data[2][0][0]
        self._prev_ask = self._data[1][-1][0]
        self._prev_bid = self._data[2][0][0]
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')
        self._observation_spec = {
            'submodel_1_obs': array_spec.ArraySpec(shape=(100, 8, 1), dtype=np.float32),
            'submodel_2_obs': array_spec.ArraySpec(shape=(100, 3, 1), dtype=np.float32),
            'submodel_3_obs': array_spec.ArraySpec(shape=(100, 3, 1), dtype=np.float32),
            'submodel_4_obs': array_spec.ArraySpec(shape=(100, 5, 1), dtype=np.float32),
            'submodel_5_obs': array_spec.ArraySpec(shape=(100, 3, 1), dtype=np.float32),
            'submodel_6_obs': array_spec.ArraySpec(shape=(1,), dtype=np.float32),                                             
            'submodel_7_obs': array_spec.ArraySpec(shape=(1,), dtype=np.float32),                                             
            }
        self._state = [self._data[0], self._data[1], self._data[2], self._data[3], \
                       self._data[4], self._data[5], np.array([0]).astype('float32')]
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec  

    def observation_spec(self):
        return self._observation_spec

    def _compute_reward(self):
        '''
        If the agent hasn't yet traded for btc, get % change in most recent asks and
        return the negative of the change as a reward (positive if avoiding a loss,
        negative if missing a gain). If the agent already has btc, get % change in 
        most recent bids and return the change as a reward.
        '''
        if not self._traded:
            return -((self._curr_ask / self._prev_ask) - 1)[0]
        return (self._curr_bid / self._prev_bid)[0] - 1

    def _reset(self):
        self._traded = 0
        self._data = self._sampler.get_sample()
        self._curr_ask = self._data[1][-1][0]
        self._curr_bid = self._data[2][0][0]
        self._prev_ask = self._data[1][-1][0]
        self._prev_bid = self._data[2][0][0]
        self._state = [self._data[0], self._data[1], self._data[2], self._data[3], \
                       self._data[4], self._data[5], np.array([0]).astype('float32')]
        self._episode_ended = False
        return ts.restart(self._state)

    def _step(self, action):    
        if self._episode_ended:
            return self.reset() 
      
        # do nothing
        if action == 0:
            # update prev values
            self._prev_ask = self._curr_ask
            self._prev_bid = self._curr_bid
            # update data, curr values and state
            self._data = self._sampler.get_sample()
            self._curr_ask = self._data[1][-1][0]
            self._curr_bid = self._data[2][0][0]
            self._state = [self._data[0], self._data[1], self._data[2], self._data[3], \
                           self._data[4], self._data[5], self._state[-1]]
            return ts.transition(self._state, reward=self._compute_reward(), discount=1.)
      
        # trade
        if action == 1:
            if self._traded:
                # sell btc and end the episode
                self._episode_ended = True
                return ts.termination(self._state, reward=0.)
            # buy btc
            self._traded = 1
            # update prev values
            self._prev_ask = self._curr_ask
            self._prev_bid = self._curr_bid
            # update data, curr values and state
            self._data = self._sampler.get_sample()
            self._curr_ask = self._data[1][-1][0]
            self._curr_bid = self._data[2][0][0]
            self._state = [self._data[0], self._data[1], self._data[2], self._data[3], \
                           self._data[4], self._data[5], np.array([1]).astype('float32')]

            return ts.transition(self._state, reward=0., discount=1.)