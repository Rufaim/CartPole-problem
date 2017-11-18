from enum import Enum


class LayerType(Enum):
	DENSE = 10
	DENSE_CONCAT = 11
	BASICLSTM = 20
	BIDIRLSTM = 21


class Layer(object):
	def __init__(self,ltype):
		self._setLtype(ltype)
	def getLtype(self):
		return self.ltype
	def _setLtype(self,ltype):
		assert isinstance(ltype, LayerType)
		self.ltype = ltype
	def __call__(self, *args):
		raise NotImplementedError
	