import tensorflow as tf
from .interfaces import Layer, LayerType


class DenseLayer(Layer):
	def __init__(self,num_neiros,activation=tf.identity,w_init=tf.truncated_normal_initializer(stddev=0.02),
					b_init=tf.constant_initializer(0.),name=None):
		Layer.__init__(self,LayerType.DENSE)
		self.num_neiros = num_neiros
		self.activation = activation
		self.w_init = w_init
		self.b_init = b_init
		self.name = name
	def __call__(self, scope, input):
		TShape = input.shape

		assert TShape.ndims == 2

		with tf.variable_scope(scope):
			W = tf.Variable(self.w_init([TShape.as_list()[1],self.num_neiros]))
			b = tf.Variable(self.b_init([1,self.num_neiros]))

			out = self.activation(tf.matmul(input,W)+b,name=self.name)
		return out, [W, b]

class DenseLayerConcat(DenseLayer):
	def __init__(self,num_neiros,activation=tf.identity,w_init=tf.truncated_normal_initializer(stddev=0.02),
					b_init=tf.constant_initializer(0.),name=None):
		DenseLayer.__init__(self,num_neiros,activation=activation,w_init=w_init,b_init=b_init,name=name)
		self._setLtype(LayerType.DENSE_CONCAT)
	def __call__(self, scope, input1, input2):
		TShape1 = input1.shape
		TShape2 = input2.shape

		assert TShape1.ndims == 2
		assert TShape2.ndims == 2

		with tf.variable_scope(scope):
			W_1 = tf.Variable(self.w_init([TShape1.as_list()[1],self.num_neiros]))
			b = tf.Variable(self.b_init([1,self.num_neiros]))
			W_2 = tf.Variable(self.w_init([TShape2.as_list()[1],self.num_neiros]))

			out = self.activation(tf.matmul(input1, W_1) + tf.matmul(input2, W_2) + b, name=self.name)
		return out, [W_1, W_2, b]
