import tensorflow as tf
import numpy as np
import gym
from nn_layers import DenseLayer, DenseLayerConcat
from actor_critic_deep_q_learning import ActorCriticModel


# Train constants
MINIBATCH_SIZE = 60
MAX_EPISODES = 5000
MAX_EP_STEPS = 1000

# Model constants
TAU = 0.001
ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.001
BUFFER_SIZE = 1000000 
DISCOUNT_FACTOR = 0.99
P_RAND_ACTION = 0.05
CRITIC_NET_STRUCTURE = [DenseLayer(300,tf.nn.relu),DenseLayerConcat(400,tf.nn.relu)]
ACTOR_NET_STRUCTURE = [DenseLayer(300,tf.nn.relu),DenseLayer(400,tf.nn.relu)]


def build_summaries():
	cum_episode_reward = tf.Variable(0.)
	tf.summary.scalar("Cumulative_reward", cum_episode_reward)

	episode_length = tf.Variable(0.)
	tf.summary.scalar("Episode_length", episode_length)

	episode_ave_max_q = tf.Variable(0.)
	tf.summary.scalar("Qmax_Value", episode_ave_max_q)
	
	summary_vars = [cum_episode_reward,episode_length,episode_ave_max_q]
	summary_ops = tf.summary.merge_all()

	return summary_ops, summary_vars


with tf.Session() as sess:
	
	env = gym.make('Pendulum-v0')

	summary_ops, summary_vars = build_summaries()
	writer = tf.summary.FileWriter("logdir", sess.graph)

	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	action_bounds = env.action_space.high[0]
	
	model = ActorCriticModel(sess,action_bounds,ACTOR_LEARNING_RATE,CRITIC_LEARNING_RATE,
							DISCOUNT_FACTOR,state_dim,action_dim,
							CRITIC_NET_STRUCTURE,ACTOR_NET_STRUCTURE,
							TAU,MINIBATCH_SIZE,BUFFER_SIZE,p_rand_action=P_RAND_ACTION)

	model.initialazer()

	for i in range(MAX_EPISODES):
		state = env.reset()
		ep_reward = 0
		ep_ave_max_q = 0

		for j in range(MAX_EP_STEPS):
			state = state.reshape( (state_dim,) )
			a = model.forward_pass(state)
			next_state, r, terminal, info = env.step(a)
			next_state = next_state.reshape( (state_dim,) )
			model.add_to_buffer(state,a,r,terminal,next_state)

			max_q = model.update()

			state = next_state.copy()
			ep_reward += r
			ep_ave_max_q += max_q

			if terminal:
				break
		summary_str = sess.run(summary_ops, feed_dict={
			summary_vars[0]: ep_reward,
			summary_vars[1]: j,
			summary_vars[2]: ep_ave_max_q
		})

		writer.add_summary(summary_str, i)
		writer.flush()

		print('| Reward: {:.2f} | Episode: {:d} | Qmax: {:.4f}'.format(ep_reward, i, ep_ave_max_q / float(j) ))
	env.close()