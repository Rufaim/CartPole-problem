import tensorflow as tf

class BatchNormalization(object):
	def __init__(self,gamma=1.0,beta=0.0,gamma_stddev=0.02,
				epsilon=1e-5,axis=[0],name=None):

		self.beta_init = tf.constant_initializer(beta)
		self.gamma_init = tf.random_normal_initializer(mean=gamma, stddev=gamma_stddev)
		self.epsilon = epsilon
		self.axis = axis
		self.name = name
	
	def  __call__(self, input):
		TShape = input.shape

		beta = tf.Variable(self.beta_init(TShape[-1:]))
		gamma = tf.Variable(self.gamma_init(TShape[-1:]))

		mean, variance = tf.nn.moments(input, self.axis)
		out = tf.nn.batch_normalization(input, mean, variance, beta, gamma, self.epsilon)
		return out, [beta, gamma]
