import tensorflow as tf
from utils import clone_net_structure

class ActorNet(tf.keras.Model):
    """Actor network for Actor-Critic model.
    It receives state as input and produces action vector
    """
    def __init__(self, net_structure, action_bounds, tau, learning_rate):
        super(ActorNet, self).__init__()
        self.net_structure = net_structure
        self.action_bounds = tf.constant(action_bounds, shape=[1, len(action_bounds)], dtype=tf.float32)
        self.tau = tf.constant(tau, dtype=tf.float32)
        self.learning_rate = learning_rate

    @tf.function
    def call(self, input, training=None, mask=None):
        out = input
        for layer in self.net_structure:
            kwargs = {}
            if training is not None:
                kwargs['training'] = training
            if mask is not None:
                kwargs['mask'] = mask
            out = layer(out, **kwargs)
        scaled_out = out * self.action_bounds
        return scaled_out

    def clone(self):
        structure = clone_net_structure(self.net_structure)
        return ActorNet(structure, self.action_bounds, self.tau, self.learning_rate)


class CriticNet(tf.keras.Model):
    """Critic network for Actor-Critic model.
    It receives state and action as input and produces Q-value
    """
    def __init__(self, net_structure, tau, learning_rate, grad_norm):
        super(CriticNet, self).__init__()
        self.net_structure = net_structure
        self.tau = tf.constant(tau, dtype=tf.float32)
        self.learning_rate = learning_rate
        self.grad_norm = grad_norm

    @tf.function
    def call(self, input, action, training=None, mask=None):
        out = input
        for i, layer in enumerate(self.net_structure):
            if i == 1:
                out = tf.concat([out, action], axis=-1)
            kwargs = {}
            if training is not None:
                kwargs['training'] = training
            if mask is not None:
                kwargs['mask'] = mask
            out = layer(out, **kwargs)
        return out

    def clone(self):
        structure = clone_net_structure(self.net_structure)
        return CriticNet(structure, self.tau, self.learning_rate, self.grad_norm)