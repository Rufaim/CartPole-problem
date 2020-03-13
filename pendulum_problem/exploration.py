import numpy as np

class RandomActionExploration(object):
    def __init__(self,p_rand_action, action_size, action_bounds, seed=None):
        self.p_rand_action = p_rand_action
        self.action_size = action_size
        self.action_bounds = np.array(action_bounds)
        self._random_generator = np.random.RandomState(seed)

    def __call__(self,action):
        if self._random_generator.rand() < self.p_rand_action:
            a = self.action_bounds * (2*self._random_generator.random(self.action_size)-1)
            return a
        return action

# Based on https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
class OrnsteinUhlenbeckActionNoise(object):
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None,seed=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.seed = seed
        self.reset()

    def __call__(self,action):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * self._random_generator.normal(size=self.mu.shape)
        self.x_prev = x
        return action + x

    def reset(self):
        self._random_generator = np.random.RandomState(seed=self.seed)
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __str__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)