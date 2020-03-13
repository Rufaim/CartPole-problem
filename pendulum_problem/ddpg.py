import tensorflow as tf
import numpy as np

class DeepDeterministicPolicyGradients(object):
    def __init__(self, actor_net, critic_net, action_exploration_func, replay_buffer, \
                 action_size, state_size, discount_factor,batch_size):
        self.actor_net = actor_net
        self.critic_net = critic_net
        self.action_exploration_func = action_exploration_func
        self.replay_buffer = replay_buffer
        self.action_size = action_size
        self.state_size = state_size
        self.discount_factor = discount_factor
        self.batch_size = batch_size

        self._init_target_nets()

        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_net.learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_net.learning_rate)

    def _init_target_nets(self):
        # train nets warmup
        inp = tf.zeros((0, self.state_size))
        a = self.actor_net(inp)
        self.critic_net(inp, a)

        self.target_actor_net = self.actor_net.clone()
        self.target_critic_net = self.critic_net.clone()

        # target nets warmup
        a = self.target_actor_net(inp)
        self.target_critic_net(inp, a)

        for v2, v1 in zip(self.target_actor_net.trainable_variables, self.actor_net.trainable_variables):
            v2.assign(v1)

        for v2, v1 in zip(self.target_critic_net.trainable_variables, self.critic_net.trainable_variables):
            v2.assign(v1)

    def actor_predict(self, state):
        s = np.atleast_2d(state)
        return self.actor_net(s).numpy()

    def critic_predict(self, state):
        s = np.atleast_2d(state)
        a = self.actor_net(s)
        return self.critic_net(s, a).numpy()

    def get_action(self, state):
        return self.action_exploration_func(self.actor_predict(state))

    def add_to_buffer(self, S, A, R, T, S1):
        self.replay_buffer.add(S, A, R, T, S1)

    def update(self):
        if self.replay_buffer.size() >= self.batch_size:
            states, actions, rewards, terminates, next_states = self.replay_buffer.sample_batch(self.batch_size)

            target_actions = self.target_actor_net(next_states)
            target_q_vals = self.target_critic_net(next_states, target_actions).numpy()

            y_is = rewards.reshape((self.batch_size, 1))
            terminates = terminates.reshape((self.batch_size, 1))
            y_is[~terminates] += self.discount_factor * target_q_vals[~terminates]

            self._update_critic(states, actions, y_is)
            self._update_actor(states)

        self._update_target_networks()

    @tf.function
    def _update_critic(self, states, actions, target_qs):
        with tf.GradientTape() as tape:
            predicted_q_value = self.critic_net(states, actions)
            critic_loss = tf.reduce_sum((target_qs - predicted_q_value) ** 2)

        grads = tape.gradient(critic_loss, self.critic_net.trainable_variables)
        #grads = tf.clip_by_global_norm(grads, self.critic_net.grad_norm)
        self.critic_optimizer.apply_gradients(zip(grads, self.critic_net.trainable_variables))

    @tf.function
    def _update_actor(self, states):
        with tf.GradientTape() as tape:
            actions = self.actor_net(states)
            q_vals = self.critic_net(states, actions)
            # minus since we are ascending
            q_vals = -q_vals

        grads = tape.gradient(q_vals, self.actor_net.trainable_variables)
        # expectation from sum
        grads = [g / self.batch_size for g in grads]
        self.actor_optimizer.apply_gradients(zip(grads, self.actor_net.trainable_variables))

    @tf.function
    def _update_target_networks(self):
        for v2, v1 in zip(self.target_actor_net.trainable_variables, self.actor_net.trainable_variables):
            v2.assign(self.actor_net.tau * v1 + (1 - self.actor_net.tau) * v2)

        for v2, v1 in zip(self.target_critic_net.trainable_variables, self.critic_net.trainable_variables):
            v2.assign(self.critic_net.tau * v1 + (1 - self.critic_net.tau) * v2)