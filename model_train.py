import tensorflow as tf
import numpy as np
import gym
from nn_layers import DenseLayer, DenseLayerConcat
from actor_critic_deep_q_learning import ActorCriticModel


# Train constants
MINIBATCH_SIZE = 200
MAX_EPISODES = 5000
MAX_EP_STEPS = 1000

# Model constants
TAU = 0.001
ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.001
BUFFER_SIZE = 1000000 
DISCOUNT_FACTOR = 0.99
ACTION_BOUNDS = 1
CRITIC_NET_STRUCTURE = [DenseLayer(150,tf.nn.relu),DenseLayerConcat(300,tf.nn.relu),DenseLayer(10,tf.nn.relu)]
ACTOR_NET_STRUCTURE = [DenseLayer(150,tf.nn.relu),DenseLayer(100,tf.nn.relu)]


def build_summaries():
	cum_episode_reward = tf.Variable(0.)
	tf.summary.scalar("Cumulative reward", cum_episode_reward)

	episode_length = tf.Variable(0.)
	tf.summary.scalar("Episode length", episode_length)
	
	summary_vars = [cum_episode_reward,episode_length]
	summary_ops = tf.summary.merge_all()

	return summary_ops, summary_vars


with tf.Session() as sess:
	
	env = gym.make("CartPole-v1")

	summary_ops, summary_vars = build_summaries()
	writer = tf.summary.FileWriter("logdir", sess.graph)

	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	
	model = ActorCriticModel(sess,ACTION_BOUNDS,ACTOR_LEARNING_RATE,CRITIC_LEARNING_RATE,
							DISCOUNT_FACTOR,state_dim,action_dim,
							CRITIC_NET_STRUCTURE,ACTOR_NET_STRUCTURE,
							TAU,MINIBATCH_SIZE,BUFFER_SIZE,p_rand_action=0.05)

	model.initialazer()

	for i in range(MAX_EPISODES):
		state = env.reset()
		ep_reward = 0

		for j in range(MAX_EP_STEPS):
			a = model.forward_pass(np.reshape(state, (1, state_dim)))
			next_state, r, terminal, info = env.step(a[0])
			model.add_to_buffer(state,a[0],r,terminal,next_state)

			model.update()

			state = next_state
			ep_reward += r

			if terminal:
				summary_str = sess.run(summary_ops, feed_dict={
					summary_vars[0]: ep_reward,
					summary_vars[1]: j
				})

				writer.add_summary(summary_str, i)
				writer.flush()

				print('| Reward: {:.2f} | Episode: {:d} | Qmax: {:.4f}'.format(ep_reward, i))
				break

	env.close()