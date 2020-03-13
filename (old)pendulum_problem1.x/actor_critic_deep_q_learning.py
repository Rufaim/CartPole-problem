import numpy as np
import tensorflow as tf

from replay_buffer import ReplayBuffer
from nn_layers import DenseLayer

class ActorNetwork(object):

	def __init__(self, sess, state_size, action_size, net_structure,
						action_bound, learning_rate, tau, batch_size):
		"""
		net_structure is list of tuples (num_neirons, activation)
		tau is exponential decay coeffecient
		"""
		self.sess = sess
		self.s_dim = state_size
		self.a_dim = action_size
		self.net_structure = net_structure
		self.action_bound = action_bound
		self.learning_rate = learning_rate
		self.tau = tau
		self.batch_size = batch_size
		self.scope = "actor"

		# Actor Network
		self.inputs, self.scaled_out, self.network_params = \
							self._create_actor_network("training")
		# Target Network
		self.target_inputs, self.target_scaled_out, self.target_network_params = \
							self._create_actor_network("target")
		# Op for periodically updating target network with online network
		# weights
		self.update_target_network_params = \
			[self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
										tf.multiply(self.target_network_params[i], 1. - self.tau))
												for i in range(len(self.target_network_params))]
		# This gradient will be provided by the critic network
		self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])
		# Combine the gradients here
		actor_gradient = tf.gradients(
			self.scaled_out, self.network_params, -self.action_gradient)
		actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), actor_gradient))
		# Optimization Op
		self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
				apply_gradients(zip(actor_gradients, self.network_params))

	def _create_actor_network(self, sub_scope):
		inputs = self._get_inputs()
		params = []
		net = inputs

		with tf.variable_scope(self.scope):
			for layer in self.net_structure:
				net, p_list = layer(sub_scope,net)
				params.extend(p_list)
			out, p_list = DenseLayer(self.a_dim,activation=tf.nn.tanh,w_init=tf.random_uniform_initializer(minval=-0.003, maxval=0.003))(sub_scope,net)
			params.extend(p_list)

			# Scale output to -action_bound to action_bound
			scaled_out = tf.multiply(out, self.action_bound)
		return inputs, scaled_out, params

	def _get_inputs(self):
		return tf.placeholder(tf.float32,shape=[None, self.s_dim])

	def train(self, inputs, a_gradient):
		self.sess.run(self.optimize, feed_dict={
			self.inputs: np.reshape(inputs,(-1,self.s_dim)),
			self.action_gradient: a_gradient})

	def predict(self, inputs):
		return self.sess.run(self.scaled_out, feed_dict={
			self.inputs: np.reshape(inputs,(-1,self.s_dim))})

	def predict_target(self, inputs):
		return self.sess.run(self.target_scaled_out, feed_dict={
			self.target_inputs: np.reshape(inputs,(-1,self.s_dim))})

	def update_target_network(self):
		self.sess.run(self.update_target_network_params)



class CriticNetwork(object):
	"""
	Input to the network is the state and action, output is Q(s,a).
	The action must be obtained from the output of the Actor network.
	
	"""

	def __init__(self, sess, state_size, action_size, net_structure,
			 					learning_rate, tau):
		self.sess = sess
		self.s_dim = state_size
		self.a_dim = action_size
		self.net_structure = net_structure
		self.learning_rate = learning_rate
		self.tau = tau
		self.scope = "critic"

		# Create the critic network
		self.inputs, self.action, self.out, self.network_params = \
							self._create_critic_network("training")
		# Target Network
		self.target_inputs, self.target_action, self.target_out, self.target_network_params = \
								self._create_critic_network("target")
		# Op for periodically updating target network with online network
		# weights with regularization
		self.update_target_network_params = \
			[self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) 
							+ tf.multiply(self.target_network_params[i], 1. - self.tau))
									for i in range(len(self.target_network_params))]
		# Network target (y_i)
		self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])
		# Define loss and optimization Op
		loss = tf.reduce_mean(tf.square(self.predicted_q_value - self.out))
		self.optimize = tf.train.AdamOptimizer(self.learning_rate) \
										.minimize(loss)
		# Get the gradient of the net w.r.t. the action.
		# For each action in the minibatch (i.e., for each x in xs),
		# this will sum up the gradients of each critic output in the minibatch
		# w.r.t. that action. Each output is independent of all
		# actions except for one.
		self.action_grads = tf.gradients(self.out, self.action)

	def _create_critic_network(self, sub_scope):
		inputs, action = self._get_inputs()

		with tf.variable_scope(self.scope):
			net, params = self.net_structure[0](sub_scope,inputs)
			net, p_list = self.net_structure[1](sub_scope,net,action)
			params.extend(p_list)

			for layer in self.net_structure[2:]:
				net, p_list = layer(sub_scope,net)
				params.extend(p_list)

			out, p_list = DenseLayer(1,w_init=tf.random_uniform_initializer(minval=-0.0003, maxval=0.0003))(sub_scope,net)
			params.extend(p_list)
		return inputs, action, out, params

	def _get_inputs(self):
		return tf.placeholder(tf.float32,shape=[None, self.s_dim]), \
				tf.placeholder(tf.float32,shape=[None, self.a_dim])

	def train(self, inputs, action, predicted_q_value):
		return self.sess.run([self.out, self.optimize], feed_dict={
			self.inputs: np.reshape(inputs,(-1,self.s_dim)),
			self.action: np.reshape(action,(-1,self.a_dim)),
			self.predicted_q_value: predicted_q_value})

	def predict(self, inputs, action):
		return self.sess.run(self.out, feed_dict={
			self.inputs: np.reshape(inputs,(-1,self.s_dim)),
			self.action: np.reshape(action,(-1,self.a_dim))})

	def predict_target(self, inputs, action):
		return self.sess.run(self.target_out, feed_dict={
			self.target_inputs: np.reshape(inputs,(-1,self.s_dim)),
			self.target_action: np.reshape(action,(-1,self.a_dim))})

	def action_gradients(self, inputs, actions):
		return self.sess.run(self.action_grads, feed_dict={
			self.inputs: np.reshape(inputs,(-1,self.s_dim)),
			self.action: np.reshape(actions,(-1,self.a_dim))})

	def update_target_network(self):
		self.sess.run(self.update_target_network_params)



class ActorCriticModel(object):
	def __init__(self,sess,action_bound,actor_learning_rate,
				critic_learning_rate,DF,
				state_vec_size,action_vec_size,
				critic_net_structure,actor_net_structure,
				tau=0.01,batch_size=180,
				base_len=-1,p_rand_action=0.05):
		self.sess = sess
		self.DF = DF
		self.batch_size = batch_size
		self.p_rand_action = p_rand_action
		self.RB = ReplayBuffer(base_len)

		self.critic = CriticNetwork(sess, state_vec_size, action_vec_size, critic_net_structure,
			 					critic_learning_rate, tau)
		self.actor = ActorNetwork(sess, state_vec_size, action_vec_size, actor_net_structure,
						action_bound, actor_learning_rate, tau, batch_size)


	def forward_pass(self,S):		
		if np.random.rand()<self.p_rand_action:
			action = self.actor.action_bound*(2*np.random.rand(self.actor.a_dim)-1)
		else:
			action = self.actor.predict(S)[0]
		return action

	def add_to_buffer(self,S,A,R,T,S1):
		self.RB.add(S,A,R,T,S1)
	
	def update(self):
		if self.RB.size >= self.batch_size:
			s_batch, a_batch, r_batch, t_batch, s1_batch = \
					self.RB.sample_batch(self.batch_size)

			# Calculate targets
			target_q = self.critic.predict_target(s1_batch,
								self.actor.predict_target(s1_batch))

			y_i = r_batch
			target_q = target_q.reshape((self.batch_size, ))
			y_i[~t_batch] += self.DF * target_q[~t_batch]

			# Update the critic given the targets
			predicted_q_value, _ = self.critic.train(s_batch, 
								a_batch, np.reshape(y_i, (self.batch_size, 1)))

			# Update the actor policy using the sampled gradient
			a_outs = self.actor.predict(s_batch)
			grads = self.critic.action_gradients(s_batch, a_outs)
			self.actor.train(s_batch, grads[0])

			# Update target networks
			self.actor.update_target_network()
			self.critic.update_target_network()
			return np.amax(predicted_q_value)
		else:
			return 0

	def initialazer(self):
		self.sess.run(tf.global_variables_initializer())

		self.actor.update_target_network()
		self.critic.update_target_network()