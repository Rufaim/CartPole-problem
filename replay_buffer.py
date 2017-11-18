import numpy as np

class ReplayBuffer(object):

	def __init__(self, buffer_size=-1):
		"""
		The right side of the buffer contains recent experiments
		If buffer_size equals to -1 than size is unlimited
		"""
		self.buffer_size = buffer_size
		self.size = 0
		self._s  = []
		self._a  = []
		self._r  = []
		self._s1 = []

	def __str__(self):
		if self.buffer_size > 0:
			total =  self.buffer_size
		else:
			total = "inf"
		return "ReplayBuffer size {} of {}".format(self.size,total)

	def add(self, s, a, r, s1):
		self._s.append(s)
		self._a.append(a)
		self._r.append(r)
		self._s1.append(s1)

		if self.size < self.buffer_size or self.buffer_size == -1: 
			self.size += 1
		else:
			self._s.pop(0)
			self._a.pop(0)
			self._r.pop(0)
			self._s1.pop(0)
		return self

	def extend(self, S, A, R, S1):
		L = len(S)
		assert L == len(A) and L == len(R) and L==len(S1)

		self._s.extend(S)
		self._a.extend(A)
		self._r.extend(R)
		self._s1.extend(S1)

		if self.size+L < self.buffer_size or self.buffer_size == -1: 
			self.size += L
		else:
			L =  L - self.buffer_size
			self._s = self._s[L:]
			self._a = self._a[L:]
			self._r = self._r[L:]
			self._s1 = self._s1[L:]
			self.size = self.buffer_size
		return self

	def sample_batch(self, batch_size):
		batch = min(self.size,batch_size)
		idx = np.random.choice(self.size,batch)

		s_batch = np.array(self._s)[idx]
		a_batch = np.array(self._a)[idx]
		r_batch = np.array(self._r)[idx]
		s1_batch = np.array(self._s1)[idx]

		return s_batch, a_batch, r_batch, s1_batch

	def clear(self):
		self.size = 0
		self._s.clear()
		self._a.clear()
		self._r.clear()
		self._s1.clear()
		return self
