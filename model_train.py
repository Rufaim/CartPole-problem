import tensorflow as tf
import gym
from nn_layers import DenseLayer, DenseLayerConcat
from actor_critic_deep_q_learning import ActorCriticModel


# Train constants
MINIBATCH_SIZE = 64
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
	episode_reward = tf.Variable(0.)
	tf.summary.scalar("Reward", episode_reward)
	episode_ave_max_q = tf.Variable(0.)
	tf.summary.scalar("Qmax Value", episode_ave_max_q)

	summary_vars = [episode_reward, episode_ave_max_q]
	summary_ops = tf.summary.merge_all()

	return summary_ops, summary_vars

with tf.Session() as sess:
	
	env = gym.make("CartPole-v1")

	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	
	model = ActorCriticModel(sess,ACTION_BOUNDS,ACTOR_LEARNING_RATE,CRITIC_LEARNING_RATE,
							DISCOUNT_FACTOR,state_dim,action_dim,
							CRITIC_NET_STRUCTURE,ACTOR_NET_STRUCTURE,
							TAU,MINIBATCH_SIZE,BUFFER_SIZE,p_rand_action=0.05)

	model.initialazer()

	
	env.close()